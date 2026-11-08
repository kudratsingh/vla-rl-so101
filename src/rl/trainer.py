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
from src.rl.gates import RolloutGates, TaskTally
from src.rl.grpo import (
    ADV_EPS,
    clip_higher_policy_loss,
    dynamic_sampling_keep,
    group_advantages,
    seq_mean_token_mean_weights,
)
from src.rl.policy_api import RLPolicy
from src.rl.rollout import (
    STORAGE,
    TIMING_KEYS,
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
    ppo_mini_batch: int = 128  # trajectories per optimizer step
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
    gate_min_gain: float = 0.03  # flat: windowed rollout success gain over the first window
    gate_rule: Literal["any_task", "mean"] = "any_task"  # best trained task, or the task mean
    gate_max_drop: float = 0.0  # decline: score < best non-overlapping window - this; 0 = off
    gate_decline_z: float = 3.0  # decline also needs drop > z x SE of (best - window), twice
    gate_entropy_min: float = 0.0  # entropy collapse: window rollout entropy < this; 0 = off
    gate_stop: bool = True  # False: gates only log their events
    # Fixed-batch pre/post-update diagnostic (src/rl/diagnostics.py)
    diag_every: int = 1  # steps; 0 = off
    diag_trajectories: int = 32  # the update batch's first trajectories (whole groups); 0 = all
    diag_argmax: bool = False  # also the greedy argmax flip rate (2 extra no-grad forwards)
    require_nonzero_baseline: bool = True  # CLAUDE.md rule 3: never train from a 0%-success base
    group_reset_check: Literal["error", "warn", "off"] = "error"
    train_tasks: tuple[str, ...] = ()  # RL rollouts and periodic evals on these; () = every task
    max_episode_steps: int = 0  # RL rollout + in-training eval horizon; 0 = env.max_episode_steps

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
            if isinstance(default, bool):
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
            raise ValueError("ppo_mini_batch (trajectories) must be a positive multiple of K")
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
        if self.gate_decline_z < 0:
            raise ValueError("gate_decline_z must be >= 0")
        if self.diag_every < 0 or self.diag_trajectories < 0 or self.diag_trajectories % k:
            raise ValueError("diag_every >= 0, diag_trajectories >= 0 and a multiple of K")
        if self.group_reset_check not in ("error", "warn", "off"):
            raise ValueError("group_reset_check must be error|warn|off")
        if self.max_episode_steps < 0:
            raise ValueError("max_episode_steps must be >= 0 (0 = env.max_episode_steps)")


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


@dataclass
class _Snapshot:
    params: list[Tensor]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]


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
        )
        self._envs: dict[str, VecEnv] = {}
        self._next_prompt = seed * TRAIN_SEED_STRIDE
        self._next_task = 0
        self._step = 0  # step in progress (0 = before training)
        self._completed_step = 0
        self._start_step = 1
        self._history: list[tuple[int, float]] = []  # (step, greedy eval success)
        self._last_tallies: dict[str, TaskTally] = {}
        self._last_entropy = (0.0, 0)
        self.resumed_from: Path | None = None

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
        if rl.recompute_old_log_probs:
            old_log_probs = self._score(batch, torch.arange(n_traj), batch.log_probs)
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
            pre = ref if rl.recompute_old_log_probs else self._score(batch, traj, ref)
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
            for start in range(0, n_traj, rl.ppo_mini_batch):
                if opt_steps is not None and n_opt >= opt_steps:
                    break
                traj_mb = torch.arange(start, min(n_traj, start + rl.ppo_mini_batch))
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
                    if epoch == 0 and start == 0:  # params == rollout params: ratio must be 1
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
        state = json.loads((path / "trainer_state.json").read_text())
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
        # our own file: python/numpy RNG states are tuples / ndarrays, not tensors
        set_rng_state(torch.load(path / "rng.pt", weights_only=False))
        diff = {k: v for k, v in asdict(self.rl).items() if state["rl"].get(k) != _jsonable(v)}
        if diff:
            log.warning("resuming %s with rl settings that differ from it: %s", path, diff)
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
        hist = self._history
        return {
            "steps_completed": self._completed_step,
            "stop_reason": reason,
            "stopped_flat": reason == "flat",
            "resumed_from": None if self.resumed_from is None else str(self.resumed_from),
            "initial_eval_success": hist[0][1] if hist else None,
            "final_eval_success": hist[-1][1] if hist else None,
            "best_eval_success": max(s for _, s in hist) if hist else None,
            "eval_history": hist,
            "trainable_params": self.num_trainable,
            "wall_time_s": time.perf_counter() - t_start,
        }

    def _gate_warning(self, step: int, reason: StopReason, m: dict[str, float]) -> None:
        bar = "!" * 78
        what = {
            "flat": (
                f"FLAT: windowed rollout success gained {m.get('gate/flat_gain', 0.0):+.3f} "
                f"(SE {m.get('gate/flat_gain_se', 0.0):.3f}) over the first {self.rl.gate_window}-"
                f"step window, under {self.rl.gate_min_gain:.3f}. Check "
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
        action = "STOPPING (rl.gate_stop=true)" if self.rl.gate_stop else "continuing (gate_stop)"
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
        warm-up, so timings use steps 2.. when there are any."""
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
                "eval_s_per_env_batch": baseline["eval/batch_s_median"],
                "frac_groups_dropped": total("rollout/frac_groups_dropped") / len(timed),
                "refill_cap_hit_frac": total("rollout/refill_cap_hit") / len(timed),
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
        self, lrs: Sequence[float], adam_steps: int, batch: RolloutBatch | None = None
    ) -> dict[str, Any]:
        """Offline replay lr sweep (review 8, Track B): ONE rollout batch (collected here, or a
        stored one), then for each lr: restore the adapter + optimizer state, run `adam_steps`
        optimizer steps over its mini-batches (epoch-major, as the loop does), and score the
        WHOLE batch with the fixed-batch diagnostic (k3 KL, clip fraction, dlogp by advantage
        sign, greedy argmax flip rate). Saves the batch as run_dir/sweep_batch.pt; leaves the
        policy and optimizer as they were. Costs one rollout collection + per lr: adam_steps x a
        mini-batch forward/backward + 2 forwards of the batch (scores, greedy)."""
        if not lrs or adam_steps < 1:
            raise ValueError("need >= 1 lr and adam_steps >= 1")
        rows: list[dict[str, Any]] = []
        out: dict[str, Any] = {"adam_steps": adam_steps, "lrs": list(lrs)}
        try:
            stats: dict[str, float] = {}
            if batch is None:
                batch, stats = self.collect()
                if batch is None:
                    raise RuntimeError("every collected group was all-equal: nothing to replay")
            batch.save(self.run_dir / "sweep_batch.pt")
            out["batch"] = {
                "trajectories": batch.num_trajectories,
                "groups": batch.num_groups,
                "decisions": int(batch.action_mask.any(-1).sum()),
                "tasks": sorted(set(batch.tasks)),
                **stats,
            }
            t0 = time.perf_counter()
            snap = self._snapshot()
            n_mb = math.ceil(batch.num_trajectories / self.rl.ppo_mini_batch)
            pre_greedy = self._greedy(batch, torch.arange(batch.num_trajectories))
            out["pre_greedy_s"] = time.perf_counter() - t0
            spec = DiagSpec(trajectories=0, argmax=True, pre_greedy=pre_greedy)
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
                    )
                    by_task[task].append(batch.group_rewards)
                    es, et = entropy_from_sampled_log_probs(batch.log_probs, _token_mask(batch))
                    ent_sum, ent_tokens = ent_sum + es, ent_tokens + et
                elapsed = time.perf_counter() - t0
                row: dict[str, Any] = {"kind": "temp_probe", "step": 0, "temperature": temp}
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
        return {"groups": n_batches * gpb, "env_batches_per_temperature": n_batches, "rows": rows}

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
