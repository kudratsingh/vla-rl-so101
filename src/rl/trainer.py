"""GRPO training loop (ADR-0002), single process / single device; rollout and update alternate.

    eval@0 -> for step in 1..steps:
        rollout (K samples per prompt) -> binary reward -> dynamic-sampling drop (+ refill)
        -> recompute old log-probs -> group advantage -> PPO epochs x mini-batches on LoRA params
        -> [fixed-batch diagnostic] -> [spend gates] -> [eval] -> [checkpoint] -> metrics.jsonl

What decides what (review 3.1 / 5, 2026-11-08):
  * The fixed held-out protocol eval (scripts/wo02_eval.py, src/eval/protocol.py) is the ONLY
    measurement of whether the policy improved. Nothing in this loop is.
  * Spend gates (src/rl/gates.py) on windowed `rollout/success_rate` and rollout entropy decide
    whether to keep paying for this run (flat / decline / entropy collapse). They replaced the
    WO-04 64-episode greedy kill switch and the 32-episode peak stop, which could not tell +3
    points from 0.
  * The periodic greedy evals (`eval_every`, `eval_episodes` on the trained tasks) are sanity logs
    only (is greedy success roughly where the protocol eval put it, did something break).
  * The fixed-batch diagnostic (src/rl/diagnostics.py) measures how far each whole update moved
    the policy on a held set of the update's own trajectories: per-token |dlogp| split by
    advantage sign, k3 KL and clip fraction BEFORE and AFTER all optimizer steps, plus grad_norm.
    Upstream's in-update `ppo_kl` / `clip_frac` (averaged over mini-batches, each measured before
    its own step, the first exactly on-policy) are no longer logged as such: the per-optimizer-
    step k3 KL / clip fraction are logged as lists, and their mean over the off-policy steps
    (first mini-batch of the first epoch excluded) as train/offpolicy_*.

Resume: checkpoints hold the adapter, the exact trainable tensors, optimizer + scheduler state,
the step counter, prompt/task counters, the eval history, the gate windows and every RNG state;
`load_checkpoint` restores them so a resumed run continues bit-for-bit on CPU (tests).

Loop order and update adapted from PRIME-RL/SimpleVLA-RL @ 7c51662df27b586f9e8a1ab35fcf849f2b8852f9
(MIT; file headers Apache-2.0, Copyright 2024 Bytedance Ltd.): verl/trainer/ppo/ray_trainer.py
`RayTrainer.fit` (generate until `train_batch_size` groups survive the accuracy filter, truncate
the excess, then advantage, then `update_actor`) and verl/workers/actor/dp_rob.py
`RobDataParallelPPOActor.update_policy` + verl/workers/fsdp_workers.py `generate_sequences`
(old log-probs recomputed by a no-grad eval-mode forward at the rollout temperature; one pass
over contiguous mini-batches of `ppo_mini_batch` trajectories, one optimizer step each,
grad-norm clip 1.0, AdamW betas (0.9, 0.999) weight decay 0.01, constant lr).

Intentional differences (none changes the gradient of a step except the refill cap):
  * refill is per env batch (stop as soon as `rollout_batch` groups survived) and capped at
    `max_rollout_rounds` x the batches of one full round; upstream refills a whole round at a time
    without a cap. If the cap is hit the step trains on the groups it has (logged), and skips the
    update if it has none.
  * decisions whose every action is padding are not forwarded (upstream forwards then masks them).
  * micro-batching is over decisions (chunks), not whole trajectories; the loss weights make the
    gradient identical to upstream's per-trajectory accumulation (src/rl/grpo.py).
  * entropy is logged from the update forwards (pre-step params) instead of a separate pass, and
    the rollout's entropy is estimated from the sampled tokens' log-probs (free).
  * no KL / reference model: `kl_coef` must be 0 (ADR-0002); anything else is rejected.
  * the constant lr runs through a LambdaLR(1.0) scheduler so its state is checkpointed.
  * mini-batches are EQUAL WHOLE-GROUP splits of the kept groups (`ppo_minibatches`, owner
    2026-11-08): ceil(groups / (ppo_mini_batch / K)) mini-batches per epoch whose sizes differ by
    at most one group, no group split. Upstream's batch is always full (uncapped refill), so its
    contiguous `ppo_mini_batch` split never leaves a tail; ours can (refill cap hit, e.g. 11 kept
    groups at 8 groups per mini-batch was 64 + a 24-trajectory tail with its own full Adam step;
    now 6 + 5 groups). A full batch (groups a multiple of ppo_mini_batch / K) splits exactly as
    before, so its gradients are unchanged.
  * `rollout_action_clip_deg` (owner 2026-11-08: 4.0 in configs/rl.yaml, null in tiny and the
    code default): protocol 0.2's per-joint step clip on the action SENT TO THE ENV in rollouts
    and in-training evals (src/rl/rollout.py). It is part of the environment, not of the policy:
    the batch keeps the sampled tokens and their sampler log-probs, so old log-probs (sampler or
    recomputed), ratios and advantages are all on the sampled tokens. Upstream has no clip.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from src.config import Config
from src.rl.diagnostics import argmax_flip_frac, entropy_from_sampled_log_probs, movement_stats
from src.rl.gates import GateResult, RolloutGates, TaskTally
from src.rl.grpo import (
    ADV_EPS,
    clip_higher_policy_loss,
    dynamic_sampling_keep,
    group_advantages,
    seq_mean_token_mean_weights,
)
from src.rl.paired import paired_group_rewards
from src.rl.policy_api import RLPolicy
from src.rl.rollout import (
    STORAGE,
    TIMING_KEYS,
    ClipStats,
    ObsFn,
    RolloutBatch,
    RolloutTimer,
    VecEnv,
    collect_rollouts,
)

log = logging.getLogger(__name__)

EnvFactory = Callable[[str, int], VecEnv]  # (task, num_envs) -> batched env
TRAIN_SEED_STRIDE = 10_000_000  # train prompt seeds: seed * stride + counter
EVAL_SEED_BASE = 4_000_000_000  # eval seeds: fixed, shared by all run seeds, disjoint from train
# probe_clip: torch RNG seed of env batch b = this + seed * TRAIN_SEED_STRIDE + b, the same in both
# arms (a sampling seed, not an env reset seed)
CLIP_PROBE_SAMPLING_SEED = 9_000_000_000

# rl keys removed 2026-11-08 (review 3.1): the greedy-eval kill switch and peak stop -> gate_*
REMOVED_KEYS = {
    "early_stop_check_at_frac": "gate_check_frac",
    "early_stop_on_flat": "gate_stop",
    "early_stop_eval_episodes": "eval_episodes (periodic greedy evals are sanity logs only)",
    "early_stop_min_improvement": "gate_min_gain",
    "early_stop_rule": "gate_rule",
    "peak_stop_step": "gate_max_drop",
    "peak_ref_step": "gate_max_drop",
}
StopReason = Literal["flat", "decline", "entropy"]
GATES: tuple[StopReason, ...] = ("flat", "decline", "entropy")


@dataclass(frozen=True)
class GRPOConfig:
    """`rl:` config section. Defaults are SimpleVLA-RL's LIBERO values unless noted."""

    samples_per_prompt: int = 8  # K
    rollout_envs: int = 32  # N parallel envs per env batch (multiple of K); ours
    rollout_batch: int = 64  # prompts (groups) per update -> rollout_batch * K trajectories
    max_rollout_rounds: int = 3  # dynamic-sampling refill cap, in full rounds; ours
    steps: int = 300
    lr: float = 5e-5  # upstream 5e-6 full fine-tune; LoRA value: owner, WO-04
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    ppo_epochs: int = 1
    ppo_mini_batch: int = 128  # max trajectories per optimizer step (equal whole-group split)
    micro_batch: int = 8  # decisions (chunks) per forward/backward; memory knob only
    clip_low: float = 0.2
    clip_high: float = 0.28
    adv_eps: float = ADV_EPS
    temperature: float = 1.6
    dynamic_sampling: bool = True
    recompute_old_log_probs: bool = True
    eval_every: int = 25
    eval_episodes: int = 32  # per trained task, greedy, fixed seeds: sanity logs only
    checkpoint_every: int = 50
    # Spend gates (src/rl/gates.py): decide whether to keep spending, never whether RL improved
    gate_window: int = 10  # steps per rollout-success / entropy window
    gate_check_frac: float = 0.25  # flat gate at ceil(frac x steps); 0 = off
    gate_min_gain: float = 0.03  # flat: rollout success gain from step 1 (trend) at the check step
    gate_flat_z: float = 0.0  # flat stops only if gain + z x its SE < min_gain (src/rl/gates.py)
    gate_rule: Literal["any_task", "mean"] = "any_task"  # best trained task, or the task mean
    gate_max_drop: float = 0.0  # decline: score < best non-overlapping window - this; 0 = off
    gate_decline_z: float = 3.0  # decline also needs drop > z x SE of (best - window), twice
    gate_entropy_min: float = 0.0  # entropy collapse: window rollout entropy < this; 0 = off
    gate_stop: bool = True  # False: gates only log their events (gate/would_stop every step)
    # Fixed-batch pre/post-update diagnostic (src/rl/diagnostics.py)
    diag_every: int = 1  # steps; 0 = off
    diag_trajectories: int = 32  # the update batch's first trajectories (whole groups); 0 = all
    diag_argmax: bool = False  # also the greedy argmax flip rate (2 extra no-grad forwards)
    require_nonzero_baseline: bool = True  # CLAUDE.md rule 3: never train from a 0%-success base
    group_reset_check: Literal["error", "warn", "off"] = "error"
    train_tasks: tuple[str, ...] = ()  # RL rollouts and periodic evals on these; () = every task
    max_episode_steps: int = 0  # RL rollout + in-training eval horizon; 0 = env.max_episode_steps
    # Per-joint step clip (deg) on every executed RL-rollout and in-training-eval action, as
    # protocol 0.2's sim eval applies it (4 deg/step from configs/real.yaml). None = off: the code
    # default, the tiny preset and every RL run so far (WO-04). configs/rl.yaml sets 4.0 (owner
    # 2026-11-08). Env action only: PPO trains on the sampled tokens (module docstring).
    rollout_action_clip_deg: float | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> GRPOConfig:
        rl = dict(cfg.get("rl", {}))
        if rl.pop("algorithm", "grpo") != "grpo":
            raise ValueError("only rl.algorithm=grpo is implemented (ADR-0002)")
        if float(rl.pop("kl_coef", 0.0)) != 0.0:
            raise ValueError("kl_coef must be 0: ADR-0002 has no KL penalty and no reference model")
        rl.pop("env", None)  # consumed by the entry script
        removed = sorted(set(rl) & set(REMOVED_KEYS))
        if removed:
            hint = "; ".join(f"{k} -> {REMOVED_KEYS[k]}" for k in removed)
            raise ValueError(
                f"rl keys {removed} were removed (review 3.1: the greedy-eval kill switch and "
                f"peak stop are replaced by rollout-window spend gates): {hint}"
            )
        known = {f.name: f for f in fields(cls)}
        unknown = set(rl) - set(known)
        if unknown:
            raise ValueError(f"unknown rl config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in rl.items():
            default = getattr(cls, key)
            if key == "rollout_action_clip_deg":  # null = off
                kwargs[key] = None if value is None else float(value)
            elif isinstance(default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"rl.{key} must be true/false, got {value!r}")
                kwargs[key] = value
            elif isinstance(default, str):
                kwargs[key] = str(value)
            elif isinstance(default, tuple):
                kwargs[key] = tuple(str(v) for v in value)
            else:  # int/float; also parses PyYAML's string "1e-5"
                kwargs[key] = type(default)(value)
        out = cls(**kwargs)
        out.validate()
        return out

    def validate(self) -> None:
        k = self.samples_per_prompt
        if k < 2:
            raise ValueError("samples_per_prompt must be >= 2 (GRPO needs a group baseline)")
        if self.rollout_envs % k:
            raise ValueError(f"rollout_envs={self.rollout_envs} must be a multiple of K={k}")
        counts = (self.rollout_batch, self.steps, self.ppo_epochs, self.micro_batch)
        if min(*counts, self.eval_every, self.eval_episodes, self.checkpoint_every) < 1:
            raise ValueError("batch/step/epoch/eval/checkpoint counts must be >= 1")
        if self.ppo_mini_batch < k or self.ppo_mini_batch % k:
            raise ValueError(
                f"ppo_mini_batch={self.ppo_mini_batch} (trajectories) must be a positive multiple "
                f"of K={k}: mini-batches hold whole groups"
            )
        if not (0 < self.clip_low < 1 and self.clip_high > 0):
            raise ValueError("need 0 < clip_low < 1 and clip_high > 0")
        if self.temperature <= 0:
            raise ValueError("rollout temperature must be > 0")
        if self.gate_window < 1:
            raise ValueError("gate_window must be >= 1 step")
        if not 0 <= self.gate_check_frac <= 1:
            raise ValueError("gate_check_frac must be in [0, 1] (0 = flat gate off)")
        check = gate_check_step(self)
        if check and check < 2 * self.gate_window:
            raise ValueError(
                f"flat-gate check step {check} (gate_check_frac x steps) must be >= 2 x "
                f"gate_window ({self.gate_window}): the first and the check window would overlap"
            )
        if self.gate_rule not in ("any_task", "mean"):
            raise ValueError("gate_rule must be any_task|mean")
        if self.gate_max_drop < 0 or self.gate_entropy_min < 0:
            raise ValueError("gate_max_drop and gate_entropy_min must be >= 0 (0 = off)")
        if self.gate_decline_z < 0 or self.gate_flat_z < 0:
            raise ValueError("gate_decline_z and gate_flat_z must be >= 0")
        if self.diag_every < 0 or self.diag_trajectories < 0 or self.diag_trajectories % k:
            raise ValueError("diag_every >= 0, diag_trajectories >= 0 and a multiple of K")
        if self.group_reset_check not in ("error", "warn", "off"):
            raise ValueError("group_reset_check must be error|warn|off")
        if self.max_episode_steps < 0:
            raise ValueError("max_episode_steps must be >= 0 (0 = env.max_episode_steps)")
        clip = self.rollout_action_clip_deg
        if clip is not None and not 0 < clip <= 180:
            raise ValueError("rollout_action_clip_deg must be null (off) or in (0, 180] deg/step")

    @property
    def action_clip_rad(self) -> float | None:
        clip = self.rollout_action_clip_deg
        return None if clip is None else math.radians(clip)


def ppo_minibatches(n_groups: int, groups_per_minibatch: int) -> list[range]:
    """One PPO epoch's mini-batches as ranges of GROUP indices: ceil(n_groups / groups_per_mb)
    mini-batches of whole groups whose sizes differ by at most one (larger ones first), together
    covering every group exactly once, in order. 16 groups at 8 per mini-batch -> 8 + 8 (the
    contiguous split upstream uses); 11 -> 6 + 5, not 8 + a 3-group tail with its own Adam step."""
    if groups_per_minibatch < 1:
        raise ValueError("groups_per_minibatch must be >= 1")
    if n_groups < 1:
        return []
    n_mb = math.ceil(n_groups / groups_per_minibatch)
    base, extra = divmod(n_groups, n_mb)
    out: list[range] = []
    start = 0
    for i in range(n_mb):
        size = base + (1 if i < extra else 0)
        out.append(range(start, start + size))
        start += size
    return out


def group_trajectories(groups: range, group_size: int) -> Tensor:
    """Trajectory indices of `groups` (RolloutBatch layout: trajectory e = g * K + k)."""
    g = torch.tensor(list(groups), dtype=torch.long)
    return (g.unsqueeze(1) * group_size + torch.arange(group_size)).reshape(-1)


def gate_check_step(rl: GRPOConfig) -> int:
    """Step of the flat gate (0 = off). The epsilon keeps ceil(0.2667 x 75) at 21, not 21.0025."""
    if rl.gate_check_frac <= 0:
        return 0
    return max(1, math.ceil(rl.gate_check_frac * rl.steps - 1e-9))


def eval_schedule(rl: GRPOConfig) -> dict[int, int]:
    """{step: greedy eval episodes per trained task}: step 0 (rule-3 baseline), every
    `eval_every` steps and the last step, all `eval_episodes` on the same seeds (`evaluate` always
    uses EVAL_SEED_BASE + [0, n)). Sanity logs only; they gate nothing but rule 3 at step 0."""
    steps = {0, rl.steps, *range(rl.eval_every, rl.steps + 1, rl.eval_every)}
    return dict.fromkeys(sorted(steps), rl.eval_episodes)


def eval_steps(rl: GRPOConfig) -> set[int]:
    return set(eval_schedule(rl))


def cuda_memory_gb() -> dict[str, float]:
    """Peak torch-allocated memory and current device-wide use (incl. PhysX / Vulkan), GB."""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "mem/torch_peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "mem/torch_peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "mem/device_used_gb": (total - free) / 1e9,
        "mem/device_total_gb": total / 1e9,
    }


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class DiagSpec:
    """Which trajectories the fixed-batch diagnostic scores, and whether greedy tokens too."""

    trajectories: int  # the batch's first N trajectories; 0 = all
    argmax: bool
    pre_greedy: Tensor | None = None  # cached pre-update greedy tokens (lr sweep)
    # cached pre-update log-probs of ALL trajectories (lr sweep: identical params for every lr);
    # also the old log-probs when rl.recompute_old_log_probs
    pre_scores: Tensor | None = None


@dataclass
class _Snapshot:
    params: list[Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


CHECKPOINT_KEYS = (
    "step",
    "completed_step",
    "next_prompt_seed",
    "next_task",
    "seed",
    "eval_history",
    "gates",
    "rl",
)
CHECKPOINT_FILES = ("trainable.pt", "optimizer.pt", "scheduler.pt", "rng.pt")


def read_checkpoint_state(path: Path) -> dict[str, Any]:
    """trainer_state.json of a resumable `save_checkpoint` directory. FileNotFoundError without
    one; ValueError naming what is missing for a checkpoint written before resume support (PR #28:
    main's checkpoints hold only adapter/, optimizer.pt and {step, next_prompt_seed,
    eval_history}) or an incomplete one. Cheap: the entry script calls it before any model load."""
    state_file = path / "trainer_state.json"
    if not state_file.is_file():
        raise FileNotFoundError(f"{path}: no trainer_state.json (a checkpoint dir?)")
    state = cast(dict[str, Any], json.loads(state_file.read_text()))
    missing = [k for k in CHECKPOINT_KEYS if k not in state]
    missing += [name for name in CHECKPOINT_FILES if not (path / name).is_file()]
    if not (path / "adapter").is_dir():
        missing.append("adapter/")
    if missing:
        raise ValueError(
            f"{path}: not a resumable checkpoint (written before resume support, PR #28, or "
            f"incomplete): missing {missing}"
        )
    return state


class GRPOTrainer:
    def __init__(
        self,
        *,
        rl: GRPOConfig,
        policy: RLPolicy,
        env_factory: EnvFactory,
        obs_fn: ObsFn,
        tasks: Sequence[str],
        max_episode_steps: int,
        run_dir: Path,
        seed: int,
    ) -> None:
        if not tasks:
            raise ValueError("need at least one task")
        if not 0 <= seed < EVAL_SEED_BASE // TRAIN_SEED_STRIDE:
            raise ValueError(f"seed must be in [0, {EVAL_SEED_BASE // TRAIN_SEED_STRIDE})")
        self.rl = rl
        self.seed = seed
        self.policy = policy
        self.env_factory = env_factory
        self.obs_fn = obs_fn
        self.tasks = list(tasks)
        self.train_tasks = list(rl.train_tasks) or list(self.tasks)
        if not set(self.train_tasks) <= set(self.tasks):
            raise ValueError(f"rl.train_tasks {self.train_tasks} must be a subset of {self.tasks}")
        self.max_episode_steps = rl.max_episode_steps or max_episode_steps
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.params = policy.trainable_parameters()
        self.num_trainable = sum(p.numel() for p in self.params)
        self.optimizer = torch.optim.AdamW(
            self.params, lr=rl.lr, betas=(0.9, 0.999), weight_decay=rl.weight_decay
        )
        # constant lr (upstream); a scheduler object so its state round-trips through checkpoints
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _constant_lr)
        self.gates = RolloutGates(
            tasks=self.train_tasks,
            window=rl.gate_window,
            check_step=gate_check_step(rl),
            min_gain=rl.gate_min_gain,
            rule=rl.gate_rule,
            max_drop=rl.gate_max_drop,
            entropy_min=rl.gate_entropy_min,
            decline_z=rl.gate_decline_z,
            flat_z=rl.gate_flat_z,
        )
        self._envs: dict[str, VecEnv] = {}
        self._next_prompt = seed * TRAIN_SEED_STRIDE
        self._next_task = 0
        self._step = 0  # step in progress (0 = before training)
        self._completed_step = 0
        self._start_step = 1
        self._history: list[tuple[int, float]] = []  # (step, greedy eval success)
        self._last_tallies: dict[str, TaskTally] = {}
        self._last_eval_batch_s: list[float] = []
        self._last_entropy = (0.0, 0)
        self.resumed_from: Path | None = None
        self.resume_overrides: dict[str, list[Any]] = {}  # rl key -> [checkpoint, this run]
        # gate decisions, logged whether or not rl.gate_stop acts on them (owner 2026-11-08):
        # (step, gates that fired) for every step a gate would have stopped the run, and the flat
        # gate's numbers at its check step
        self._gate_log: list[tuple[int, list[str]]] = []
        self._flat_check: dict[str, Any] | None = None
        # first step the gate log covers: > 1 after resuming a checkpoint that has no gate log
        # (written before 2026-11-08 round 2), whose earlier gate decisions are unknown
        self._gate_log_from = 1

    # ------------------------------------------------------------------ envs / logging
    def _env(self, task: str) -> VecEnv:
        if task not in self._envs:
            self._envs[task] = self.env_factory(task, self.rl.rollout_envs)
        return self._envs[task]

    def close(self) -> None:
        for env in self._envs.values():
            close = getattr(env, "close", None)
            if callable(close):
                close()
        self._envs.clear()

    def _log(self, row: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    # ------------------------------------------------------------------ eval (sanity only)
    def evaluate(self, episodes: int | None = None) -> dict[str, float]:
        """Greedy (temperature 0) success over `episodes` per TRAINED task (default
        `eval_episodes`). A sanity log; the protocol eval measures improvement.

        Seeds are EVAL_SEED_BASE + [0, episodes) for every task, every call and every run seed, so
        two evals with the same count see the same scenes. Envs run in batches of `rollout_envs`;
        surplus envs of the last batch are not counted.
        """
        episodes = self.rl.eval_episodes if episodes is None else int(episodes)
        t0 = time.perf_counter()
        n = self.rl.rollout_envs
        seeds = [EVAL_SEED_BASE + i for i in range(math.ceil(episodes / n) * n)]
        out: dict[str, float] = {}
        rates = []
        batch_s = []
        for task in self.train_tasks:
            successes = []
            for b in range(0, len(seeds), n):
                tb = time.perf_counter()
                batch, _ = collect_rollouts(
                    self.policy,
                    self._env(task),
                    self.obs_fn,
                    prompt_seeds=seeds[b : b + n],
                    group_size=1,
                    temperature=0.0,
                    max_episode_steps=self.max_episode_steps,
                    task=task,
                    group_reset_check="off",
                    action_clip=self.rl.action_clip_rad,
                )
                batch_s.append(time.perf_counter() - tb)
                successes.append(batch.rewards)
            rate = float(torch.cat(successes)[:episodes].mean())
            out[f"eval/success/{task}"] = rate
            rates.append(rate)
        out["eval/success"] = sum(rates) / len(rates)
        out["eval/success_train"] = out["eval/success"]  # = eval/success (trained tasks only)
        out["eval/episodes_per_task"] = float(episodes)
        out["eval/episodes"] = float(episodes * len(self.train_tasks))
        out["eval/batch_s_median"] = statistics.median(batch_s)
        out["eval/time_s"] = time.perf_counter() - t0
        self._last_eval_batch_s = batch_s  # per env batch, in order (probe: warm-up excluded)
        return out

    def _check_baseline(self, baseline: dict[str, float]) -> None:
        """CLAUDE.md rule 3: RL cannot rescue a 0%-success base (every group is all-zero)."""
        if self.rl.require_nonzero_baseline and baseline["eval/success"] <= 0.0:
            raise RuntimeError(
                f"step-0 eval success is 0 on every trained task "
                f"({int(baseline['eval/episodes'])} greedy episodes): RL cannot start from a "
                "0%-success base (CLAUDE.md rule 3). Fix SFT; do not launch RL."
            )
        for task in self.train_tasks:
            if baseline.get(f"eval/success/{task}", 1.0) <= 0.0:
                log.warning(
                    "step-0 eval success is 0 on %s: its groups will be all-zero and dropped "
                    "(rollout time spent for no gradient)",
                    task,
                )

    # ------------------------------------------------------------------ rollout
    def collect(self) -> tuple[RolloutBatch | None, dict[str, float]]:
        """Roll out env batches until `rollout_batch` groups survive dynamic sampling (or cap).

        Also leaves the step's per-task group success and sampled-token entropy in
        `_last_tallies` / `_last_entropy` for the spend gates (every generated group counts)."""
        rl = self.rl
        k = rl.samples_per_prompt
        groups_per_batch = rl.rollout_envs // k
        round_batches = math.ceil(rl.rollout_batch / groups_per_batch)
        max_batches = round_batches * (rl.max_rollout_rounds if rl.dynamic_sampling else 1)
        t0 = time.perf_counter()
        timer = RolloutTimer()
        kept: list[RolloutBatch] = []
        n_kept = 0
        per_task: dict[str, list[int]] = {
            t: [0, 0, 0] for t in self.train_tasks
        }  # gen/kept/trained
        group_rates: dict[str, list[float]] = {t: [] for t in self.train_tasks}
        gen_rewards: list[Tensor] = []
        gen_len: list[Tensor] = []
        ent_sum, ent_tokens = 0.0, 0
        reset_diff = 0.0
        n_batches = 0
        clip_stats = ClipStats()
        for _ in range(max_batches):
            task = self.train_tasks[self._next_task % len(self.train_tasks)]
            self._next_task += 1
            seeds = list(range(self._next_prompt, self._next_prompt + groups_per_batch))
            self._next_prompt += groups_per_batch
            batch, diff = collect_rollouts(
                self.policy,
                self._env(task),
                self.obs_fn,
                prompt_seeds=seeds,
                group_size=k,
                temperature=rl.temperature,
                max_episode_steps=self.max_episode_steps,
                task=task,
                group_reset_check=rl.group_reset_check,
                timer=timer,
                action_clip=rl.action_clip_rad,
                clip_stats=clip_stats,
            )
            n_batches += 1
            reset_diff = max(reset_diff, diff)
            gen_rewards.append(batch.group_rewards)
            gen_len.append(batch.episode_len)
            group_rates[task] += batch.group_rewards.mean(dim=1).tolist()
            es, et = entropy_from_sampled_log_probs(batch.log_probs, _token_mask(batch))
            ent_sum, ent_tokens = ent_sum + es, ent_tokens + et
            if rl.dynamic_sampling:
                batch = batch.select_groups(dynamic_sampling_keep(batch.group_rewards))
            per_task[task][0] += groups_per_batch
            per_task[task][1] += batch.num_groups
            per_task[task][2] += min(batch.num_groups, max(0, rl.rollout_batch - n_kept))
            if batch.num_groups:
                kept.append(batch)
                n_kept += batch.num_groups
            if n_kept >= rl.rollout_batch:
                break
        elapsed = time.perf_counter() - t0
        rewards = torch.cat(gen_rewards)  # (G_generated, K)
        episodes = rewards.numel()
        trained = min(n_kept, rl.rollout_batch)
        self._last_tallies = {t: TaskTally.from_group_rates(r) for t, r in group_rates.items() if r}
        self._last_entropy = (ent_sum, ent_tokens)
        stats = {
            "rollout/success_rate": float(rewards.mean()),
            "rollout/entropy": ent_sum / max(ent_tokens, 1),  # MC estimate, nats/token at T
            "rollout/group_reward_std": float(rewards.std(dim=1).mean()),
            "rollout/frac_groups_dropped": 1.0 - n_kept / rewards.shape[0],
            "rollout/groups_generated": float(rewards.shape[0]),
            "rollout/groups_trained": float(trained),
            "rollout/env_batches": float(n_batches),
            "rollout/refill_cap_hit": float(n_batches == max_batches and n_kept < rl.rollout_batch),
            "rollout/episodes": float(episodes),
            "rollout/episodes_per_hour": episodes / elapsed * 3600.0,
            "rollout/mean_episode_len": float(torch.cat(gen_len).float().mean()),
            "rollout/group_reset_diff_frac": reset_diff,
            "rollout/time_s": elapsed,
        }
        for key, value in timer.totals(elapsed).items():  # sums to rollout/time_s
            stats[f"rollout/time/{key}_s"] = value
        if rl.rollout_action_clip_deg is not None:  # clip intervention over this step's rollouts
            stats.update(clip_stats.summary("rollout/action_clip"))
        for t, (gen_t, kept_t, trained_t) in per_task.items():
            if gen_t:
                stats[f"rollout/frac_groups_dropped/{t}"] = 1.0 - kept_t / gen_t
                stats[f"rollout/success_rate/{t}"] = sum(group_rates[t]) / len(group_rates[t])
            stats[f"rollout/groups_trained/{t}"] = float(trained_t)
        if not kept:
            return None, stats
        merged = RolloutBatch.cat(kept)
        if merged.num_groups > rl.rollout_batch:  # upstream truncates the excess too
            keep = torch.zeros(merged.num_groups, dtype=torch.bool)
            keep[: rl.rollout_batch] = True
            merged = merged.select_groups(keep)
        return merged, stats

    # ------------------------------------------------------------------ scoring helpers
    def _gather(
        self, batch: RolloutBatch, e: Tensor, s: Tensor
    ) -> tuple[dict[str, Tensor], list[str]]:
        obs = {key: value[e, s] for key, value in batch.obs.items()}
        return obs, [batch.instructions[i] for i in e.tolist()]

    @torch.no_grad()
    def _score(self, batch: RolloutBatch, traj: Tensor, fill: Tensor) -> Tensor:
        """No-grad eval-mode log-probs of the stored tokens for trajectories `traj`, at the
        rollout temperature; decisions with no executed action keep `fill`'s values."""
        self.policy.eval()
        out = fill.clone()
        dv = batch.action_mask[traj].any(dim=-1)
        l_idx, s_idx = dv.nonzero(as_tuple=True)
        for i in range(0, len(l_idx), self.rl.micro_batch):
            loc, s = l_idx[i : i + self.rl.micro_batch], s_idx[i : i + self.rl.micro_batch]
            e = traj[loc]
            obs, instr = self._gather(batch, e, s)
            lp = self.policy.log_prob(obs, instr, batch.tokens[e, s], self.rl.temperature)
            out[loc, s] = lp.log_probs.to(STORAGE, torch.float32)
        return out

    @torch.no_grad()
    def _greedy(self, batch: RolloutBatch, traj: Tensor) -> Tensor:
        """Greedy (argmax) tokens of the current policy at every stored decision of `traj`."""
        self.policy.eval()
        out = torch.zeros_like(batch.tokens[traj])
        dv = batch.action_mask[traj].any(dim=-1)
        l_idx, s_idx = dv.nonzero(as_tuple=True)
        for i in range(0, len(l_idx), self.rl.micro_batch):
            loc, s = l_idx[i : i + self.rl.micro_batch], s_idx[i : i + self.rl.micro_batch]
            obs, instr = self._gather(batch, traj[loc], s)
            out[loc, s] = self.policy.sample(obs, instr, 0.0).tokens.to(STORAGE, torch.long)
        return out

    def _diag_traj(self, batch: RolloutBatch, spec: DiagSpec) -> Tensor:
        n = batch.num_trajectories
        return torch.arange(min(n, spec.trajectories) if spec.trajectories else n)

    def _scheduled_diag(self) -> DiagSpec | None:
        every = self.rl.diag_every
        if not every or self._step % every:
            return None
        return DiagSpec(trajectories=self.rl.diag_trajectories, argmax=self.rl.diag_argmax)

    # ------------------------------------------------------------------ update
    def update(
        self,
        batch: RolloutBatch,
        *,
        epochs: int | None = None,
        opt_steps: int | None = None,
        diag: DiagSpec | None = None,
    ) -> dict[str, Any]:
        """PPO epochs over the batch. `epochs` / `opt_steps` (a cap on optimizer steps) override
        the config for the lr sweep; `diag` overrides the diagnostic schedule."""
        rl = self.rl
        t0 = time.perf_counter()
        n_traj = batch.num_trajectories
        advantages = group_advantages(batch.group_rewards, rl.adv_eps).reshape(n_traj)
        decision_valid = batch.action_mask.any(dim=-1)  # (E, S)
        token_mask = _token_mask(batch)  # (E, S, C, A)
        old_log_probs = batch.log_probs
        mismatch = mismatch_mean = 0.0
        cached = diag.pre_scores if diag is not None else None
        if rl.recompute_old_log_probs:
            old_log_probs = (
                cached
                if cached is not None
                else self._score(batch, torch.arange(n_traj), batch.log_probs)
            )
            dev_tok = (old_log_probs - batch.log_probs).abs()[token_mask]
            mismatch, mismatch_mean = float(dev_tok.max()), float(dev_tok.mean())

        spec = diag if diag is not None else self._scheduled_diag()
        diag_stats: dict[str, float] = {}
        diag_state: tuple[Tensor, Tensor, Tensor, Tensor | None] | None = None
        if spec is not None:
            td = time.perf_counter()
            traj = self._diag_traj(batch, spec)
            ref = old_log_probs[traj]
            # recompute on: the pre-update scores ARE the reference (same params, same forward)
            if rl.recompute_old_log_probs:
                pre = ref
            elif spec.pre_scores is not None:
                pre = spec.pre_scores[traj]
            else:
                pre = self._score(batch, traj, ref)
            greedy = spec.pre_greedy
            if spec.argmax and greedy is None:
                greedy = self._greedy(batch, traj)
            adv_tok = advantages[traj].view(-1, 1, 1, 1).expand_as(ref)
            for key, v in movement_stats(
                pre, ref, adv_tok, token_mask[traj], clip_low=rl.clip_low, clip_high=rl.clip_high
            ).items():
                diag_stats[f"diag/pre/{key}"] = v
            diag_state = (traj, ref, adv_tok, greedy)
            diag_stats["diag/trajectories"] = float(len(traj))
            diag_stats["diag/time_s"] = time.perf_counter() - td

        k = batch.group_size
        if rl.ppo_mini_batch % k:
            raise ValueError(f"ppo_mini_batch {rl.ppo_mini_batch} is not a multiple of K={k}")
        minibatches = ppo_minibatches(batch.num_groups, rl.ppo_mini_batch // k)
        self.policy.train()
        sums = {"pg_loss": 0.0, "entropy": 0.0}
        opt_clip: list[float] = []
        opt_k3: list[float] = []
        opt_grad: list[float] = []
        first_dev = 0.0
        first_dev_sum = first_dev_n = 0.0
        n_opt = 0
        n_epochs = rl.ppo_epochs if epochs is None else epochs
        for epoch in range(n_epochs):
            for j, groups in enumerate(minibatches):
                if opt_steps is not None and n_opt >= opt_steps:
                    break
                traj_mb = group_trajectories(groups, k)  # whole groups, gathered by group index
                weights = seq_mean_token_mean_weights(token_mask[traj_mb])  # sums to 1
                e_loc, s_idx = decision_valid[traj_mb].nonzero(as_tuple=True)
                self.optimizer.zero_grad(set_to_none=True)
                mb_clip = mb_k3 = 0.0
                for i in range(0, len(e_loc), rl.micro_batch):
                    el, s = e_loc[i : i + rl.micro_batch], s_idx[i : i + rl.micro_batch]
                    e = traj_mb[el]
                    obs, instr = self._gather(batch, e, s)
                    out = self.policy.log_prob(obs, instr, batch.tokens[e, s], rl.temperature)
                    dev = out.log_probs.device
                    w = weights[el, s].to(dev)
                    old = old_log_probs[e, s].to(dev)
                    adv = advantages[e].to(dev).view(-1, 1, 1).expand_as(out.log_probs)
                    pl = clip_higher_policy_loss(
                        out.log_probs, old, adv, w, rl.clip_low, rl.clip_high
                    )
                    torch.autograd.backward(pl.loss)
                    sums["pg_loss"] += float(pl.loss.detach())
                    sums["entropy"] += float((out.entropy.detach() * w).sum())
                    mb_clip += float(pl.clip_frac)
                    mb_k3 += float(pl.k3_kl)
                    if epoch == 0 and j == 0:  # params == rollout params: ratio must be 1
                        dev_tok = (out.log_probs.detach() - old).abs()[w > 0]
                        first_dev = max(first_dev, float(dev_tok.max()))
                        first_dev_sum += float(dev_tok.sum())
                        first_dev_n += dev_tok.numel()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, rl.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                opt_clip.append(mb_clip)
                opt_k3.append(mb_k3)
                opt_grad.append(float(grad_norm))
                n_opt += 1
        self.optimizer.zero_grad(set_to_none=True)
        n = max(n_opt, 1)
        stats: dict[str, Any] = {f"train/{key}": value / n for key, value in sums.items()}
        stats.update(
            {
                "train/grad_norm": sum(opt_grad) / n,  # pre-clip L2 norm, mean over steps
                "train/grad_norm_max": max(opt_grad, default=0.0),
                # per optimizer step (epoch-major), each measured before its own step: index 0 is
                # on-policy by construction; these are NOT the update's total movement (diag/post)
                "train/opt_step_clip_frac": opt_clip,
                "train/opt_step_k3_kl": opt_k3,
                "train/opt_step_grad_norm": opt_grad,
                "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                "train/reward_mean": float(batch.rewards.mean()),
                "train/adv_abs_mean": float(advantages.abs().mean()),
                "train/trajectories": float(n_traj),
                "train/decisions": float(decision_valid.sum()),
                "train/optimizer_steps": float(n_opt),
                # groups per mini-batch of one epoch (equal whole-group split, ppo_minibatches)
                "train/minibatch_groups": [len(g) for g in minibatches],
                "train/logprob_mismatch": mismatch,
                "train/logprob_mismatch_mean": mismatch_mean,
                "train/first_minibatch_max_abs_log_ratio": first_dev,
                "train/first_minibatch_mean_abs_log_ratio": first_dev_sum / max(first_dev_n, 1),
            }
        )
        if n_opt > 1:  # first mini-batch of the first epoch excluded (exactly on-policy)
            stats["train/offpolicy_clip_frac"] = sum(opt_clip[1:]) / (n_opt - 1)
            stats["train/offpolicy_k3_kl"] = sum(opt_k3[1:]) / (n_opt - 1)
        if diag_state is not None:
            td = time.perf_counter()
            traj, ref, adv_tok, greedy = diag_state
            post = self._score(batch, traj, ref)
            for key, v in movement_stats(
                post, ref, adv_tok, token_mask[traj], clip_low=rl.clip_low, clip_high=rl.clip_high
            ).items():
                diag_stats[f"diag/post/{key}"] = v
            if greedy is not None:
                after = self._greedy(batch, traj)
                flip = argmax_flip_frac(greedy, after, token_mask[traj])
                diag_stats["diag/post/argmax_flip_frac"] = flip
            diag_stats["diag/time_s"] += time.perf_counter() - td
            stats.update(diag_stats)
        stats["train/update_time_s"] = time.perf_counter() - t0  # includes diag/time_s
        return stats

    # ------------------------------------------------------------------ checkpoint / resume
    def save_checkpoint(self, step: int, suffix: str = "") -> Path:
        """Adapter (for eval) + everything `load_checkpoint` needs to continue exactly."""
        path = self.run_dir / "checkpoints" / f"step_{step:06d}{suffix}"
        self.policy.save(path / "adapter")
        torch.save([p.detach().to(STORAGE).clone() for p in self.params], path / "trainable.pt")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        torch.save(self._scheduler_state(), path / "scheduler.pt")
        torch.save(rng_state(), path / "rng.pt")
        state = {
            "step": step,
            # last fully finished step; < step for an _interrupted checkpoint, whose weights may
            # hold part of step `step` (resuming from one is approximate)
            "completed_step": self._completed_step,
            "next_prompt_seed": self._next_prompt,
            "next_task": self._next_task,
            "seed": self.seed,
            "eval_history": self._history,
            "gates": self.gates.state_dict(),
            "rl": asdict(self.rl),
            # optional on load (checkpoints written before 2026-11-08 round 2 lack it)
            "gate_decisions": {
                "log": self._gate_log,
                "flat_check": self._flat_check,
                "log_from": self._gate_log_from,
            },
        }
        (path / "trainer_state.json").write_text(json.dumps(state, indent=2))
        return path

    def load_checkpoint(self, path: Path, *, allow_partial: bool = False) -> None:
        """Restore a `save_checkpoint` directory: trainable tensors, optimizer, scheduler, RNG,
        counters, eval history and gate windows. Training continues at completed_step + 1.

        An `_interrupted` checkpoint (step != completed_step) holds weights, Adam state and
        counters from inside step `step`, so resuming re-runs part or all of that step on top of
        it. Refused unless `allow_partial` (`--resume-approximate`); then the gate windows and the
        eval history are cut back to completed_step so the re-run step can be recorded again."""
        state = read_checkpoint_state(path)
        step, completed = int(state["step"]), int(state["completed_step"])
        if step != completed:
            plain = sorted(
                p.name
                for p in path.parent.glob("step_*")
                if p.name[5:].isdigit() and int(p.name[5:]) <= completed
            )
            exact = f"{path.parent / plain[-1]}" if plain else "none in this run"
            if not allow_partial:
                raise ValueError(
                    f"{path} is an _interrupted checkpoint: its weights may hold part of step "
                    f"{step} (completed_step {completed}). Resume exactly from the latest plain "
                    f"checkpoint ({exact}), or pass --resume-approximate to re-run step {step} "
                    "on top of these weights"
                )
            log.warning(
                "APPROXIMATE RESUME from %s: step %d is re-run on weights that may already "
                "contain part of it (exact resume point: %s)",
                path,
                step,
                exact,
            )
        tensors = torch.load(path / "trainable.pt", map_location=STORAGE, weights_only=True)
        if len(tensors) != len(self.params) or any(
            t.shape != p.shape for t, p in zip(tensors, self.params, strict=True)
        ):
            raise ValueError(f"{path}: trainable tensors do not match this policy's parameters")
        with torch.no_grad():
            for p, t in zip(self.params, tensors, strict=True):
                p.copy_(t.to(p.device, p.dtype))
        self.optimizer.load_state_dict(
            torch.load(path / "optimizer.pt", map_location=STORAGE, weights_only=True)
        )
        self._load_scheduler(torch.load(path / "scheduler.pt", weights_only=True))
        # load_state_dict restores the SAVED lr / weight_decay (param groups, scheduler base_lrs)
        # over this run's config. The config wins, so record.json describes what trains; Adam
        # moments and step counts stay as loaded.
        self._set_lr(self.rl.lr)
        for group in self.optimizer.param_groups:
            group["weight_decay"] = self.rl.weight_decay
        # our own file: python/numpy RNG states are tuples / ndarrays, not tensors
        set_rng_state(torch.load(path / "rng.pt", weights_only=False))
        saved_rl = state["rl"]
        diff = {k: v for k, v in asdict(self.rl).items() if saved_rl.get(k) != _jsonable(v)}
        self.resume_overrides = {k: [saved_rl.get(k), _jsonable(v)] for k, v in diff.items()}
        if diff:
            log.warning(
                "resuming %s with rl settings that differ from it (checkpoint -> this run; the "
                "run's values apply, lr and weight_decay included; Adam moments kept): %s",
                path,
                self.resume_overrides,
            )
        if int(state["seed"]) != self.seed:
            log.warning("resuming seed-%s checkpoint under seed %d", state["seed"], self.seed)
        self._completed_step = completed
        self._start_step = self._completed_step + 1
        self._next_prompt = int(state["next_prompt_seed"])
        self._next_task = int(state["next_task"])
        self._history = [
            (int(s), float(v)) for s, v in state["eval_history"] if int(s) <= completed
        ]
        self.gates.load_state_dict(state["gates"])
        self.gates.truncate(completed)
        decisions = state.get("gate_decisions")
        if decisions is None:  # older checkpoint: gate decisions for steps <= completed unknown
            log.warning(
                "checkpoint %s has no gate log: gate decisions for steps 1-%d are unknown, the "
                "gate summary will say so",
                path,
                completed,
            )
            decisions = {}
            self._gate_log_from = completed + 1
        else:
            self._gate_log_from = min(int(decisions.get("log_from", 1)), completed + 1)
        self._gate_log = [
            (int(st), [str(r) for r in rs])
            for st, rs in decisions.get("log", [])
            if int(st) <= completed
        ]
        fc = decisions.get("flat_check")
        self._flat_check = fc if fc is not None and int(fc["step"]) <= completed else None
        self.resumed_from = path
        log.info("resumed from %s: continuing at step %d", path, self._start_step)

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            params=[p.detach().clone() for p in self.params],
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            scheduler=copy.deepcopy(self._scheduler_state()),
        )

    def _restore(self, snap: _Snapshot) -> None:
        with torch.no_grad():
            for p, t in zip(self.params, snap.params, strict=True):
                p.copy_(t)
        # load_state_dict keeps (does not copy) same-device tensors, and Adam updates them in
        # place: hand it a copy so the snapshot stays pristine for the next lr
        self.optimizer.load_state_dict(copy.deepcopy(snap.optimizer))
        self._load_scheduler(copy.deepcopy(snap.scheduler))

    def _scheduler_state(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.scheduler.state_dict())  # type: ignore[no-untyped-call]

    def _load_scheduler(self, state: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state)  # type: ignore[no-untyped-call]

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"] = lr
        self.scheduler.base_lrs = [lr] * len(self.optimizer.param_groups)

    # ------------------------------------------------------------------ loop
    def _train_step(self) -> dict[str, Any]:
        batch, row = self.collect()
        if batch is None:
            log.warning("every group was all-equal; update skipped")
            row["train/update_skipped"] = 1.0
        else:
            row.update(self.update(batch))
        row.update(cuda_memory_gb())
        return row

    def run(self) -> dict[str, Any]:
        rl = self.rl
        t_start = time.perf_counter()
        log.info("GRPO: %d trainable params, rl=%s", self.num_trainable, asdict(rl))
        schedule = eval_schedule(rl)
        reason: StopReason | None = None
        step = self._completed_step
        try:
            if self._start_step == 1:
                baseline = self.evaluate(schedule[0])
                self._log({"kind": "eval", "step": 0, **baseline})
                log.info("step 0 eval success %.3f", baseline["eval/success"])
                self._check_baseline(baseline)
                self._history = [(0, baseline["eval/success"])]
            for step in range(self._start_step, rl.steps + 1):
                t0 = time.perf_counter()
                self._step = step
                row = self._train_step()
                ent_sum, ent_tokens = self._last_entropy
                self.gates.record(step, self._last_tallies, ent_sum, ent_tokens)
                gate = self.gates.evaluate(step)
                row.update(gate.metrics)
                row.update(self._gate_decision(step, gate))
                if step in schedule:
                    row.update(self.evaluate(schedule[step]))
                    self._history.append((step, row["eval/success"]))
                if gate.reason is not None:
                    self._gate_warning(step, gate.reason, gate.metrics)
                    if rl.gate_stop:
                        reason = gate.reason
                    else:
                        row["event/gate_ignored"] = 1.0
                self._completed_step = step
                if step % rl.checkpoint_every == 0 or step == rl.steps or reason:
                    self.save_checkpoint(step)
                row["time/step_s"] = time.perf_counter() - t0
                row["time/elapsed_s"] = time.perf_counter() - t_start
                self._log({"kind": "train", "step": step, **row})
                self._console(step, row)
                if reason:
                    break
        except (KeyboardInterrupt, SystemExit):
            # SIGTERM from the pod job's soft deadline (scripts/wo03_rl.py maps it to SystemExit):
            # keep the weights trained since the last checkpoint (possibly mid-step)
            if step >= 1:
                path = self.save_checkpoint(step, suffix="_interrupted")
                log.warning("interrupted during step %d; saved %s", step, path)
            raise
        finally:
            self.close()
        gates = self.gate_summary()
        self._log({"kind": "gate_summary", "step": self._completed_step, **gates})
        self._console_gate_summary(gates)
        hist = self._history
        return {
            "steps_completed": self._completed_step,
            "stop_reason": reason,
            "gate_decisions": gates,
            "stopped_flat": reason == "flat",
            "resumed_from": None if self.resumed_from is None else str(self.resumed_from),
            "initial_eval_success": hist[0][1] if hist else None,
            "final_eval_success": hist[-1][1] if hist else None,
            "best_eval_success": max(s for _, s in hist) if hist else None,
            "eval_history": hist,
            "trainable_params": self.num_trainable,
            "wall_time_s": time.perf_counter() - t_start,
        }

    def _gate_decision(self, step: int, gate: GateResult) -> dict[str, Any]:
        """Per-step gate decision for metrics.jsonl, logged whether or not rl.gate_stop acts on
        it: gate/would_stop (0/1), gate/would_stop_reason (comma-separated gates that fired, ""
        if none), gate/first_would_stop_step (first step any gate fired so far; 0 = none yet)."""
        if gate.reasons:
            self._gate_log.append((step, list(gate.reasons)))
        check = gate_check_step(self.rl)
        if check and step == check:
            m = gate.metrics
            self._flat_check = {
                "step": step,
                "would_stop": "flat" in gate.reasons,
                "gain": _finite(m.get("gate/flat_gain")),
                "gain_se": _finite(m.get("gate/flat_gain_se")),
                "gain_upper": _finite(m.get("gate/flat_gain_upper")),
                "window_gain_old_rule": _finite(m.get("gate/flat_window_gain")),
                "min_gain": self.rl.gate_min_gain,
                "flat_z": self.rl.gate_flat_z,
                "rule": self.rl.gate_rule,
                "per_task": {
                    k.removeprefix("gate/flat_gain/"): _finite(v)
                    for k, v in m.items()
                    if k.startswith("gate/flat_gain/")
                },
            }
        first = self._gate_log[0][0] if self._gate_log else 0
        return {
            "gate/would_stop": float(bool(gate.reasons)),
            "gate/would_stop_reason": ",".join(gate.reasons),
            "gate/first_would_stop_step": float(first),
        }

    def gate_summary(self) -> dict[str, Any]:
        """When each spend gate would have stopped this run (whether or not rl.gate_stop let it),
        and the flat gate's decision at its check step (step 38 in configs/rl.yaml, where WO-04
        seeds 1-2 stopped under the older rule). For the run record and the console."""
        check = gate_check_step(self.rl)
        first_by_gate: dict[str, int | None] = dict.fromkeys(GATES)
        fired: dict[str, int] = dict.fromkeys(GATES, 0)
        for st, reasons in self._gate_log:
            for r in reasons:
                fired[r] += 1
                if first_by_gate[r] is None:
                    first_by_gate[r] = st
        first = self._gate_log[0] if self._gate_log else None
        reached = bool(check) and self._completed_step >= check
        fc = self._flat_check
        log_from = self._gate_log_from
        complete = log_from <= 1  # False: resumed from a checkpoint without a gate log
        # the flat check step fell in the unknown stretch before the log starts
        flat_unknown = fc is None and bool(check) and check < log_from
        if not reached:
            any_by_check: bool | None = None
        elif first is not None and first[0] <= check:
            any_by_check = True  # fired at a logged step, so true whatever came before
        else:
            any_by_check = False if complete else None
        return {
            "gate_stop": self.rl.gate_stop,
            "steps_completed": self._completed_step,
            "gate_log_complete": complete,
            # steps before this have no gate log (older checkpoint resumed); the first_* and
            # per-gate fields below then cover only steps >= gate_log_from_step
            "gate_log_from_step": log_from,
            "first_would_stop_step": (first[0] if first else None) if complete else None,
            "first_would_stop_reasons": (first[1] if first else []) if complete else [],
            "first_logged_would_stop_step": first[0] if first else None,
            "first_step_by_gate": first_by_gate,
            "steps_fired_by_gate": fired,
            "flat_check_step": check or None,
            "flat_check": fc,
            "flat_check_unknown": flat_unknown,
            # None: the check step was not reached, the flat gate is off, or (flat_check_unknown)
            # it fell before the gate log starts
            "would_have_stopped_at_flat_check": None if fc is None else bool(fc["would_stop"]),
            "any_gate_would_have_stopped_by_flat_check_step": any_by_check,
        }

    def _console_gate_summary(self, g: dict[str, Any]) -> None:
        check = g["flat_check_step"]
        fc = g["flat_check"]
        if fc is not None:
            verdict = "WOULD HAVE STOPPED" if fc["would_stop"] else "would NOT have stopped"
            flat = (
                f"flat gate at step {check} {verdict} the run (gain {_fmt(fc['gain'])}, SE "
                f"{_fmt(fc['gain_se'])}, gain + {fc['flat_z']:g} x SE = {_fmt(fc['gain_upper'])} vs "
                f"min_gain {fc['min_gain']:.3f}, rule {fc['rule']})"
            )
        elif g["flat_check_unknown"]:
            flat = (
                f"flat gate check step {check}: UNKNOWN (resumed from a checkpoint without a "
                f"gate log; steps 1-{g['gate_log_from_step'] - 1} not logged)"
            )
        elif check:
            flat = f"flat gate check step {check} not reached ({g['steps_completed']} steps)"
        else:
            flat = "flat gate off"
        first = g["first_would_stop_step"]
        firsts = ", ".join(
            f"{k} {v if v is not None else '-'}" for k, v in g["first_step_by_gate"].items()
        )
        log.info(
            "GATE SUMMARY (rl.gate_stop=%s): %s; first step any gate would have stopped: %s %s; "
            "first step per gate: %s",
            str(self.rl.gate_stop).lower(),
            flat,
            (
                (first if first is not None else "none")
                if g["gate_log_complete"]
                else f"unknown (log starts at step {g['gate_log_from_step']}; first logged "
                f"{g['first_logged_would_stop_step'] or 'none'})"
            ),
            g["first_would_stop_reasons"] or "",
            firsts,
        )

    def _gate_warning(self, step: int, reason: StopReason, m: dict[str, float]) -> None:
        bar = "!" * 78
        what = {
            "flat": (
                f"FLAT: rollout success trend gained {m.get('gate/flat_gain', 0.0):+.3f} "
                f"(SE {m.get('gate/flat_gain_se', 0.0):.3f}) since step 1 (rule {self.rl.gate_rule}); "
                f"gain + {self.rl.gate_flat_z:g} x SE = {m.get('gate/flat_gain_upper', 0.0):+.3f}"
                f" is under {self.rl.gate_min_gain:.3f}. Check "
                "rollout/frac_groups_dropped, temperature, diag/post/* (did updates move the "
                "policy?)"
            ),
            "decline": (
                f"DECLINE: windowed rollout success {m.get('gate/success_window', 0.0):.3f} is "
                f"{m.get('gate/decline_drop', 0.0):.3f} below its best "
                f"{m.get('gate/best_success_window', 0.0):.3f} (window ending at step "
                f"{int(m.get('gate/best_success_step', 0))}), more than the margin "
                f"{m.get('gate/decline_margin', 0.0):.3f} = max(gate_max_drop "
                f"{self.rl.gate_max_drop:.3f}, {self.rl.gate_decline_z:g} x SE), on this and the "
                "previous non-overlapping window"
            ),
            "entropy": (
                f"ENTROPY: windowed rollout entropy {m.get('gate/entropy_window', 0.0):.3f} "
                f"nats/token < {self.rl.gate_entropy_min:.3f}: sampling has collapsed"
            ),
        }[reason]
        action = (
            "STOPPING (rl.gate_stop=true)"
            if self.rl.gate_stop
            else "continuing: rl.gate_stop=false, logged only (gate/would_stop)"
        )
        log.warning(
            "\n%s\nSPEND GATE at step %d: %s. %s. This gate decides spending only; the protocol "
            "eval decides improvement.\n%s",
            bar,
            step,
            what,
            action,
            bar,
        )

    # ------------------------------------------------------------------ probes ($: GPU only)
    def probe(self, steps: int) -> dict[str, Any]:
        """Measure the real loop for the pod job's probe -> approval gate: build the envs, run the
        step-0 baseline eval (as `run` does), then `steps` RL steps. Saves nothing; the policy is
        discarded afterwards (the full run restarts from the SFT adapter). Step 1 carries CUDA
        warm-up, so timings use steps 2.. when there are any; likewise the per-batch greedy eval
        time skips the baseline's first env batch."""
        out: dict[str, Any] = {
            "steps": steps,
            "trainable_params": self.num_trainable,
            "max_episode_steps": self.max_episode_steps,
            "train_tasks": list(self.train_tasks),
        }
        rows: list[dict[str, Any]] = []
        try:
            t0 = time.perf_counter()
            for task in self.train_tasks:
                self._env(task)
            out["env_setup_s"] = time.perf_counter() - t0
            baseline = self.evaluate(self.rl.eval_episodes)
            eval_batch_s = list(self._last_eval_batch_s)
            self._log({"kind": "probe_eval", "step": 0, **baseline})
            out["baseline"] = baseline
            self._check_baseline(baseline)
            for step in range(1, steps + 1):
                t0 = time.perf_counter()
                self._step = step
                row = self._train_step()
                row["time/step_s"] = time.perf_counter() - t0
                self._log({"kind": "probe", "step": step, **row})
                self._console(step, row)
                rows.append(row)
        finally:
            self.close()
        timed = rows[1:] or rows

        def total(key: str) -> float:
            return float(sum(r.get(key, 0.0) for r in timed))

        n_batches = total("rollout/env_batches")
        split = {k: total(f"rollout/time/{k}_s") / n_batches for k in TIMING_KEYS}
        diag_rows = [r for r in rows if "diag/post/k3_kl" in r]
        out.update(
            {
                "s_per_step_median": statistics.median(r["time/step_s"] for r in timed),
                "s_per_step_max": max(r["time/step_s"] for r in timed),
                "rollout_s_per_env_batch": total("rollout/time_s") / n_batches,
                "rollout_split_s_per_env_batch": split,
                "update_s_median": statistics.median(
                    r.get("train/update_time_s", 0.0) for r in timed
                ),
                "rollout_episodes_per_hour": total("rollout/episodes")
                / total("rollout/time_s")
                * 3600.0,
                # the baseline's first env batch is the process's first policy forward / render
                # (CUDA warm-up): drop it as the RL timings drop step 1. With one batch (one
                # trained task) fall back to the warm rollout batch (same envs and horizon)
                "eval_s_per_env_batch": (
                    statistics.median(eval_batch_s[1:])
                    if len(eval_batch_s) > 1
                    else min(eval_batch_s[0], total("rollout/time_s") / n_batches)
                ),
                "eval_s_first_env_batch": eval_batch_s[0],
                "frac_groups_dropped": total("rollout/frac_groups_dropped") / len(timed),
                "refill_cap_hit_frac": total("rollout/refill_cap_hit") / len(timed),
                # mean over steps of the env-step fraction the clip changed; None = clip off
                "action_clip_frac": (
                    total("rollout/action_clip_frac") / len(timed)
                    if self.rl.rollout_action_clip_deg is not None
                    else None
                ),
                "first_minibatch_max_abs_log_ratio": max(
                    r.get("train/first_minibatch_max_abs_log_ratio", 0.0) for r in rows
                ),
                "first_minibatch_mean_abs_log_ratio": max(
                    r.get("train/first_minibatch_mean_abs_log_ratio", 0.0) for r in rows
                ),
                "group_reset_diff_frac": max(r["rollout/group_reset_diff_frac"] for r in rows),
                "diag_post": {
                    k.removeprefix("diag/post/"): statistics.median(r[k] for r in diag_rows)
                    for k in (diag_rows[0] if diag_rows else {})
                    if k.startswith("diag/post/")
                },
                "rows": rows,
            }
        )
        out.update(cuda_memory_gb())
        return out

    def lr_sweep(
        self,
        lrs: Sequence[float],
        adam_steps: int,
        batch: RolloutBatch | None = None,
        batch_clip_deg: float | str | None = "unknown",
    ) -> dict[str, Any]:
        """Offline replay lr sweep (review 8, Track B): ONE rollout batch (collected here, or a
        stored one), then for each lr: restore the adapter + optimizer state, run `adam_steps`
        optimizer steps over its mini-batches (epoch-major, as the loop does), and score the
        WHOLE batch with the fixed-batch diagnostic (k3 KL, clip fraction, dlogp by advantage
        sign, greedy argmax flip rate). Saves the batch as run_dir/sweep_batch.pt; leaves the
        policy and optimizer as they were. Costs one rollout collection + 2 forwards of the batch
        before the first lr (greedy tokens; log-probs, reused as every lr's old log-probs and
        diag reference) + per lr: adam_steps x a mini-batch forward/backward + 2 forwards of the
        batch (post scores, post greedy).

        The rollout action clip recorded is the batch's own: this preset's when collected here
        (also logged to metrics.jsonl as a `lr_sweep_batch` row with its clip intervention stats
        and written next to the batch as sweep_batch.json), else `batch_clip_deg` (the stored
        batch's sweep_batch.json; "unknown" without one), never the current preset's."""
        if not lrs or adam_steps < 1:
            raise ValueError("need >= 1 lr and adam_steps >= 1")
        rows: list[dict[str, Any]] = []
        out: dict[str, Any] = {
            "adam_steps": adam_steps,
            "lrs": list(lrs),
            # of the batch's rollouts: a stored batch may come from another clip setting
            "rollout_action_clip_deg": (
                batch_clip_deg if batch is not None else (self.rl.rollout_action_clip_deg)
            ),
            "batch_source": "stored" if batch is not None else "collected",
        }
        try:
            stats: dict[str, float] = {}
            if batch is None:
                batch, stats = self.collect()
                if batch is None:
                    raise RuntimeError("every collected group was all-equal: nothing to replay")
                self._log({"kind": "lr_sweep_batch", "step": 0, **stats})
            batch.save(self.run_dir / "sweep_batch.pt")
            meta = {"rollout_action_clip_deg": out["rollout_action_clip_deg"]}
            (self.run_dir / "sweep_batch.json").write_text(json.dumps(meta) + "\n")
            out["batch"] = {
                "trajectories": batch.num_trajectories,
                "groups": batch.num_groups,
                "decisions": int(batch.action_mask.any(-1).sum()),
                "tasks": sorted(set(batch.tasks)),
                **stats,
            }
            t0 = time.perf_counter()
            snap = self._snapshot()
            # optimizer steps per epoch: the trainer's equal whole-group split of this batch
            n_mb = len(
                ppo_minibatches(batch.num_groups, self.rl.ppo_mini_batch // batch.group_size)
            )
            every = torch.arange(batch.num_trajectories)
            pre_greedy = self._greedy(batch, every)
            out["pre_greedy_s"] = time.perf_counter() - t0
            # the same params before every lr: score once (old log-probs + diag reference)
            t1 = time.perf_counter()
            pre_scores = self._score(batch, every, batch.log_probs)
            out["pre_score_s"] = time.perf_counter() - t1
            spec = DiagSpec(
                trajectories=0, argmax=True, pre_greedy=pre_greedy, pre_scores=pre_scores
            )
            for lr in lrs:
                self._restore(snap)
                self._set_lr(float(lr))
                t_lr = time.perf_counter()
                res = self.update(
                    batch, epochs=math.ceil(adam_steps / n_mb), opt_steps=adam_steps, diag=spec
                )
                row: dict[str, Any] = {
                    "kind": "lr_sweep",
                    "step": 0,
                    "lr": float(lr),
                    "adam_steps": adam_steps,
                    **res,
                    "time/lr_s": time.perf_counter() - t_lr,
                }
                self._log(row)
                rows.append(row)
                log.info(
                    "lr %.1e: k3 %.2e clip %.4f |dlogp| %.4f (A+ %+.4f, A- %+.4f) flip %.4f",
                    lr,
                    res["diag/post/k3_kl"],
                    res["diag/post/clip_frac"],
                    res["diag/post/mean_abs_dlogp"],
                    res["diag/post/mean_dlogp_pos_adv"],
                    res["diag/post/mean_dlogp_neg_adv"],
                    res["diag/post/argmax_flip_frac"],
                )
            self._restore(snap)
        finally:
            self.close()
        out["rows"] = rows
        out.update(cuda_memory_gb())
        return out

    def probe_temperatures(self, temperatures: Sequence[float], groups: int) -> dict[str, Any]:
        """Rollouts only (review 8, Track B): for each temperature, roll out the SAME `groups`
        prompts (rounded up to whole env batches, tasks round-robin as in training; no dynamic
        sampling, no update) and report sampled success, group-drop rate (all-equal groups),
        rollout entropy and the env batches one update would need at that drop rate."""
        if not temperatures or groups < 1:
            raise ValueError("need >= 1 temperature and groups >= 1")
        rl = self.rl
        k = rl.samples_per_prompt
        gpb = rl.rollout_envs // k
        n_batches = math.ceil(groups / gpb)
        cap = math.ceil(rl.rollout_batch / gpb) * rl.max_rollout_rounds
        start_prompt, start_task = self._next_prompt, self._next_task
        rows: list[dict[str, Any]] = []
        try:
            for temp in temperatures:
                if temp <= 0:
                    raise ValueError("probe temperatures must be > 0")
                self._next_prompt, self._next_task = start_prompt, start_task  # paired prompts
                timer = RolloutTimer()
                clip_stats = ClipStats()
                t0 = time.perf_counter()
                by_task: dict[str, list[Tensor]] = {t: [] for t in self.train_tasks}
                ent_sum, ent_tokens = 0.0, 0
                for _ in range(n_batches):
                    task = self.train_tasks[self._next_task % len(self.train_tasks)]
                    self._next_task += 1
                    seeds = list(range(self._next_prompt, self._next_prompt + gpb))
                    self._next_prompt += gpb
                    batch, _ = collect_rollouts(
                        self.policy,
                        self._env(task),
                        self.obs_fn,
                        prompt_seeds=seeds,
                        group_size=k,
                        temperature=float(temp),
                        max_episode_steps=self.max_episode_steps,
                        task=task,
                        group_reset_check=rl.group_reset_check,
                        timer=timer,
                        action_clip=rl.action_clip_rad,
                        clip_stats=clip_stats,
                    )
                    by_task[task].append(batch.group_rewards)
                    es, et = entropy_from_sampled_log_probs(batch.log_probs, _token_mask(batch))
                    ent_sum, ent_tokens = ent_sum + es, ent_tokens + et
                elapsed = time.perf_counter() - t0
                row: dict[str, Any] = {
                    "kind": "temp_probe",
                    "step": 0,
                    "temperature": temp,
                    "rollout_action_clip_deg": rl.rollout_action_clip_deg,  # the preset's
                }
                if rl.rollout_action_clip_deg is not None:
                    row.update(clip_stats.summary("probe/action_clip"))
                everything = torch.cat([r for rs in by_task.values() for r in rs])
                row.update(_group_summary(everything, "probe"))
                for t, rs in by_task.items():
                    if rs:
                        row.update(_group_summary(torch.cat(rs), f"probe/{t}"))
                drop = row["probe/frac_groups_dropped"]
                need = math.inf if drop >= 1 else rl.rollout_batch / (gpb * (1.0 - drop))
                row["probe/env_batches_per_update"] = min(float(cap), need)
                row["probe/refill_cap_binds"] = float(need > cap)
                row["probe/entropy"] = ent_sum / max(ent_tokens, 1)
                row["probe/time_s"] = elapsed
                row["probe/s_per_env_batch"] = elapsed / n_batches
                for key, value in timer.totals(elapsed).items():
                    row[f"probe/time/{key}_s"] = value
                self._log(row)
                rows.append(row)
                log.info(
                    "T %.2f: sampled success %.3f (SE %.3f), groups dropped %.3f, entropy %.3f",
                    temp,
                    row["probe/success_rate"],
                    row["probe/success_rate_se"],
                    drop,
                    row["probe/entropy"],
                )
        finally:
            self.close()
        return {
            "groups": n_batches * gpb,
            "env_batches_per_temperature": n_batches,
            "rollout_action_clip_deg": rl.rollout_action_clip_deg,
            "rows": rows,
        }

    def probe_clip(self, groups: int) -> dict[str, Any]:
        """Paired clip on/off probe (owner 2026-11-08), rollouts only (no update, no dynamic
        sampling, no refill). Both arms play the SAME `groups` prompts (reset seeds and tasks
        round-robin from the trainer's counters, as in training; rounded up to whole env batches)
        with the SAME sampling seeds: the global torch RNG is reseeded to the same value before
        each env batch of each arm, so a slot's first decision draws identical tokens and later
        decisions use common random numbers. K = rl.samples_per_prompt, T = rl.temperature (the
        preset's). Arm "on" applies rl.rollout_action_clip_deg to the env action; arm "off" sends
        the sampled action unchanged and only measures how often the clip WOULD have changed it
        (against its own previous executed action). Per arm: sampled success, group-drop rate
        (all-equal groups, which dynamic sampling drops), entropy, episode length and clip
        intervention; paired: per-group success difference and the episode / group count tables
        (src/rl/paired.py)."""
        rl = self.rl
        bound = rl.action_clip_rad
        if bound is None:
            raise ValueError(
                "probe_clip compares the preset's rl.rollout_action_clip_deg against no clip; "
                "it is null"
            )
        if groups < 1:
            raise ValueError("groups must be >= 1")
        k = rl.samples_per_prompt
        gpb = rl.rollout_envs // k
        n_batches = math.ceil(groups / gpb)
        arms = {"off": False, "on": True}  # arm -> clip applied to the env action
        rewards: dict[str, list[Tensor]] = {a: [] for a in arms}
        by_task: dict[str, dict[str, list[Tensor]]] = {
            a: {t: [] for t in self.train_tasks} for a in arms
        }
        first_tokens: dict[str, list[Tensor]] = {a: [] for a in arms}
        ep_len: dict[str, list[Tensor]] = {a: [] for a in arms}
        ent = {a: [0.0, 0] for a in arms}
        clip_stats = {a: ClipStats() for a in arms}
        timers = {a: RolloutTimer() for a in arms}
        elapsed = {a: 0.0 for a in arms}
        batch_s: dict[str, list[float]] = {a: [] for a in arms}
        sampling_seeds: list[int] = []
        try:
            # build every env this probe visits BEFORE any seeding or timing: the lazy build would
            # otherwise land inside arm "off"'s time (it runs first) and run after the reseed
            first_task = self._next_task
            for i in range(min(n_batches, len(self.train_tasks))):
                self._env(self.train_tasks[(first_task + i) % len(self.train_tasks)])
            for b in range(n_batches):
                task = self.train_tasks[self._next_task % len(self.train_tasks)]
                self._next_task += 1
                seeds = list(range(self._next_prompt, self._next_prompt + gpb))
                self._next_prompt += gpb
                sampling_seed = CLIP_PROBE_SAMPLING_SEED + self.seed * TRAIN_SEED_STRIDE + b
                sampling_seeds.append(sampling_seed)
                env = self._env(task)  # built above: no RNG use, no build time from here on
                for arm, apply in arms.items():
                    torch.manual_seed(sampling_seed)  # CPU and every CUDA device: same draws
                    t0 = time.perf_counter()
                    batch, _ = collect_rollouts(
                        self.policy,
                        env,
                        self.obs_fn,
                        prompt_seeds=seeds,
                        group_size=k,
                        temperature=rl.temperature,
                        max_episode_steps=self.max_episode_steps,
                        task=task,
                        group_reset_check=rl.group_reset_check,
                        timer=timers[arm],
                        action_clip=bound,
                        clip_stats=clip_stats[arm],
                        clip_apply=apply,
                    )
                    batch_s[arm].append(time.perf_counter() - t0)
                    elapsed[arm] += batch_s[arm][-1]
                    rewards[arm].append(batch.group_rewards)
                    by_task[arm][task].append(batch.group_rewards)
                    first_tokens[arm].append(batch.tokens[:, 0])
                    ep_len[arm].append(batch.episode_len)
                    es, et = entropy_from_sampled_log_probs(batch.log_probs, _token_mask(batch))
                    ent[arm][0] += es
                    ent[arm][1] += int(et)
        finally:
            self.close()
        rows: list[dict[str, Any]] = []
        for arm, apply in arms.items():
            row: dict[str, Any] = {
                "kind": "clip_probe",
                "step": 0,
                "arm": arm,
                "clip_applied": apply,
                "rollout_action_clip_deg": rl.rollout_action_clip_deg,
                "temperature": rl.temperature,
                "samples_per_prompt": k,
            }
            row.update(_group_summary(torch.cat(rewards[arm]), "probe"))
            for t, rs in by_task[arm].items():
                if rs:
                    row.update(_group_summary(torch.cat(rs), f"probe/{t}"))
            row["probe/entropy"] = ent[arm][0] / max(int(ent[arm][1]), 1)
            row["probe/mean_episode_len"] = float(torch.cat(ep_len[arm]).float().mean())
            # arm on: how often the clip changed the env action; arm off: how often it WOULD have
            row.update(clip_stats[arm].summary("probe/action_clip"))
            row["probe/time_s"] = elapsed[arm]
            row["probe/s_per_env_batch"] = elapsed[arm] / n_batches
            # batch 0 of arm "off" carries the process's first forwards (CUDA warm-up): compare
            # the arms' speed on the later batches
            later = batch_s[arm][1:]
            row["probe/s_per_env_batch_after_first"] = (
                sum(later) / len(later) if later else elapsed[arm]
            )
            for key, value in timers[arm].totals(elapsed[arm]).items():
                row[f"probe/time/{key}_s"] = value
            self._log(row)
            rows.append(row)
        paired = paired_group_rewards(torch.cat(rewards["off"]), torch.cat(rewards["on"]))
        tok_off, tok_on = torch.cat(first_tokens["off"]), torch.cat(first_tokens["on"])
        # pairing check: same scene + same RNG -> the first decision's tokens agree (1.0 on CPU;
        # GPU kernel nondeterminism can flip a few)
        paired["first_decision_token_match"] = float(
            (tok_off == tok_on).flatten(1).all(-1).float().mean()
        )
        self._log({"kind": "clip_probe_paired", "step": 0, **paired})
        log.info(
            "clip probe: success off %.3f / on %.3f (paired diff %+.3f, SE %.3f), dropped "
            "off %.3f / on %.3f, clip changed %.3f of steps (on), would change %.3f (off)",
            rows[0]["probe/success_rate"],
            rows[1]["probe/success_rate"],
            paired["success_diff_on_minus_off"],
            paired["success_diff_se"],
            rows[0]["probe/frac_groups_dropped"],
            rows[1]["probe/frac_groups_dropped"],
            rows[1]["probe/action_clip_frac"],
            rows[0]["probe/action_clip_frac"],
        )
        return {
            "groups": n_batches * gpb,
            "env_batches_per_arm": n_batches,
            "samples_per_prompt": k,
            "temperature": rl.temperature,
            "rollout_action_clip_deg": rl.rollout_action_clip_deg,
            "sampling_seeds": sampling_seeds,
            "rows": rows,
            "paired": paired,
        }

    @staticmethod
    def _console(step: int, row: dict[str, Any]) -> None:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "-" if value is None else f"{cast(float, value):.3g}"

        log.info(
            "step %d | rollout success %s dropped %s ent %s | update |dlogp| %s k3 %s clip %s "
            "grad %s | gate window %s | eval %s | %ss",
            step,
            fmt("rollout/success_rate"),
            fmt("rollout/frac_groups_dropped"),
            fmt("rollout/entropy"),
            fmt("diag/post/mean_abs_dlogp"),
            fmt("diag/post/k3_kl"),
            fmt("diag/post/clip_frac"),
            fmt("train/grad_norm"),
            fmt("gate/success_window"),
            fmt("eval/success"),
            fmt("time/step_s"),
        )


def _constant_lr(_: int) -> float:
    return 1.0


def _token_mask(batch: RolloutBatch) -> Tensor:
    """(E, S, C, A) bool: tokens of executed actions."""
    return batch.action_mask.unsqueeze(-1).expand_as(batch.tokens)


def _finite(value: float | None) -> float | None:
    """JSON-safe gate number: None for a missing or non-finite value (no trend -> -inf)."""
    return None if value is None or not math.isfinite(value) else float(value)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}"


def _jsonable(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value


def _group_summary(rewards: Tensor, prefix: str) -> dict[str, float]:
    """Sampled success (+ SE over groups) and drop rates of (G, K) binary rewards."""
    g = rewards.shape[0]
    means = rewards.float().mean(dim=1)
    dropped = ~dynamic_sampling_keep(rewards)
    drop = float(dropped.float().mean())
    se = float(means.std() / math.sqrt(g)) if g > 1 else 0.0
    return {
        f"{prefix}/groups": float(g),
        f"{prefix}/episodes": float(rewards.numel()),
        f"{prefix}/success_rate": float(means.mean()),
        f"{prefix}/success_rate_se": se,
        f"{prefix}/frac_groups_dropped": drop,
        f"{prefix}/frac_groups_dropped_se": math.sqrt(drop * (1 - drop) / g),
        f"{prefix}/frac_groups_all_zero": float((rewards.amax(dim=1) == 0).float().mean()),
        f"{prefix}/frac_groups_all_one": float((rewards.amin(dim=1) == 1).float().mean()),
    }
