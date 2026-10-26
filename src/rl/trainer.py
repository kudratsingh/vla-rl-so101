"""GRPO training loop (ADR-0002), single process / single device; rollout and update alternate.

    eval@0 -> for step in 1..steps:
        rollout (K samples per prompt) -> binary reward -> dynamic-sampling drop (+ refill)
        -> recompute old log-probs -> group advantage -> PPO epochs x mini-batches on LoRA params
        -> [eval] -> [FLAT-RUN kill switch] -> [checkpoint] -> metrics.jsonl row

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
  * entropy is logged from the update forwards (pre-step params) instead of a separate pass.
  * no KL / reference model: `kl_coef` must be 0 (ADR-0002); anything else is rejected.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from src.config import Config
from src.rl.grpo import (
    ADV_EPS,
    clip_higher_policy_loss,
    dynamic_sampling_keep,
    group_advantages,
    seq_mean_token_mean_weights,
)
from src.rl.policy_api import RLPolicy
from src.rl.rollout import STORAGE, ObsFn, RolloutBatch, VecEnv, collect_rollouts

log = logging.getLogger(__name__)

EnvFactory = Callable[[str, int], VecEnv]  # (task, num_envs) -> batched env
TRAIN_SEED_STRIDE = 10_000_000  # train prompt seeds: seed * stride + counter
EVAL_SEED_BASE = 4_000_000_000  # eval seeds: fixed, shared by all run seeds, disjoint from train


@dataclass(frozen=True)
class GRPOConfig:
    """`rl:` config section. Defaults are SimpleVLA-RL's LIBERO values unless noted."""

    samples_per_prompt: int = 8  # K
    rollout_envs: int = 32  # N parallel envs per env batch (multiple of K); ours
    rollout_batch: int = 64  # prompts (groups) per update -> rollout_batch * K trajectories
    max_rollout_rounds: int = 3  # dynamic-sampling refill cap, in full rounds; ours
    steps: int = 300
    lr: float = 1e-5  # upstream 5e-6 full fine-tune; LoRA value is ours
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
    eval_episodes: int = 32  # per task, greedy, fixed seeds
    checkpoint_every: int = 50
    early_stop_check_at_frac: float = 0.25  # WO-04: flat by here -> stop and diagnose
    early_stop_on_flat: bool = True
    early_stop_min_improvement: float = 0.0
    group_reset_check: Literal["error", "warn", "off"] = "error"

    @classmethod
    def from_config(cls, cfg: Config) -> GRPOConfig:
        rl = dict(cfg.get("rl", {}))
        if rl.pop("algorithm", "grpo") != "grpo":
            raise ValueError("only rl.algorithm=grpo is implemented (ADR-0002)")
        if float(rl.pop("kl_coef", 0.0)) != 0.0:
            raise ValueError("kl_coef must be 0: ADR-0002 has no KL penalty and no reference model")
        rl.pop("env", None)  # consumed by the entry script
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
        if not 0 < self.early_stop_check_at_frac <= 1:
            raise ValueError("early_stop_check_at_frac must be in (0, 1]")
        if self.group_reset_check not in ("error", "warn", "off"):
            raise ValueError("group_reset_check must be error|warn|off")


def kill_switch_step(rl: GRPOConfig) -> int:
    return max(1, math.ceil(rl.early_stop_check_at_frac * rl.steps))


def eval_steps(rl: GRPOConfig) -> set[int]:
    """Steps after which the loop evaluates: 0, every `eval_every`, the kill-switch step, last."""
    every = set(range(rl.eval_every, rl.steps + 1, rl.eval_every))
    return {0, kill_switch_step(rl), rl.steps} | every


def is_flat_run(baseline: float, current: float, min_improvement: float = 0.0) -> bool:
    """WO-04 kill switch: flat unless eval success rose by more than `min_improvement`."""
    return not current > baseline + min_improvement


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
        self.policy = policy
        self.env_factory = env_factory
        self.obs_fn = obs_fn
        self.tasks = list(tasks)
        self.max_episode_steps = max_episode_steps
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.jsonl"
        self.params = policy.trainable_parameters()
        self.num_trainable = sum(p.numel() for p in self.params)
        self.optimizer = torch.optim.AdamW(
            self.params, lr=rl.lr, betas=(0.9, 0.999), weight_decay=rl.weight_decay
        )
        self._envs: dict[str, VecEnv] = {}
        self._next_prompt = seed * TRAIN_SEED_STRIDE
        self._next_task = 0

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

    # ------------------------------------------------------------------ eval
    def evaluate(self) -> dict[str, float]:
        """Greedy (temperature 0) success on fixed seeds, `eval_episodes` (rounded up) per task."""
        t0 = time.perf_counter()
        n = self.rl.rollout_envs
        batches = math.ceil(self.rl.eval_episodes / n)
        out: dict[str, float] = {}
        rates = []
        for task in self.tasks:
            successes = []
            for b in range(batches):
                seeds = [EVAL_SEED_BASE + b * n + i for i in range(n)]
                batch, _ = collect_rollouts(
                    self.policy,
                    self._env(task),
                    self.obs_fn,
                    prompt_seeds=seeds,
                    group_size=1,
                    temperature=0.0,
                    max_episode_steps=self.max_episode_steps,
                    task=task,
                    group_reset_check="off",
                )
                successes.append(batch.rewards)
            rate = float(torch.cat(successes).mean())
            out[f"eval/success/{task}"] = rate
            rates.append(rate)
        out["eval/success"] = sum(rates) / len(rates)
        out["eval/episodes"] = float(batches * n * len(self.tasks))
        out["eval/time_s"] = time.perf_counter() - t0
        return out

    # ------------------------------------------------------------------ rollout
    def collect(self) -> tuple[RolloutBatch | None, dict[str, float]]:
        """Roll out env batches until `rollout_batch` groups survive dynamic sampling (or cap)."""
        rl = self.rl
        k = rl.samples_per_prompt
        groups_per_batch = rl.rollout_envs // k
        round_batches = math.ceil(rl.rollout_batch / groups_per_batch)
        max_batches = round_batches * (rl.max_rollout_rounds if rl.dynamic_sampling else 1)
        t0 = time.perf_counter()
        kept: list[RolloutBatch] = []
        n_kept = 0
        gen_rewards: list[Tensor] = []
        gen_len: list[Tensor] = []
        reset_diff = 0.0
        n_batches = 0
        for _ in range(max_batches):
            task = self.tasks[self._next_task % len(self.tasks)]
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
            )
            n_batches += 1
            reset_diff = max(reset_diff, diff)
            gen_rewards.append(batch.group_rewards)
            gen_len.append(batch.episode_len)
            if rl.dynamic_sampling:
                batch = batch.select_groups(dynamic_sampling_keep(batch.group_rewards))
            if batch.num_groups:
                kept.append(batch)
                n_kept += batch.num_groups
            if n_kept >= rl.rollout_batch:
                break
        elapsed = time.perf_counter() - t0
        rewards = torch.cat(gen_rewards)  # (G_generated, K)
        episodes = rewards.numel()
        trained = min(n_kept, rl.rollout_batch)
        stats = {
            "rollout/success_rate": float(rewards.mean()),
            "rollout/group_reward_std": float(rewards.std(dim=1).mean()),
            "rollout/frac_groups_dropped": 1.0 - n_kept / rewards.shape[0],
            "rollout/groups_generated": float(rewards.shape[0]),
            "rollout/groups_trained": float(trained),
            "rollout/env_batches": float(n_batches),
            "rollout/episodes": float(episodes),
            "rollout/episodes_per_hour": episodes / elapsed * 3600.0,
            "rollout/mean_episode_len": float(torch.cat(gen_len).float().mean()),
            "rollout/group_reset_max_diff": reset_diff,
            "rollout/time_s": elapsed,
        }
        if not kept:
            return None, stats
        merged = RolloutBatch.cat(kept)
        if merged.num_groups > rl.rollout_batch:  # upstream truncates the excess too
            keep = torch.zeros(merged.num_groups, dtype=torch.bool)
            keep[: rl.rollout_batch] = True
            merged = merged.select_groups(keep)
        return merged, stats

    # ------------------------------------------------------------------ update
    def _gather(
        self, batch: RolloutBatch, e: Tensor, s: Tensor
    ) -> tuple[dict[str, Tensor], list[str]]:
        obs = {key: value[e, s] for key, value in batch.obs.items()}
        return obs, [batch.instructions[i] for i in e.tolist()]

    @torch.no_grad()
    def _recompute_log_probs(self, batch: RolloutBatch, decision_valid: Tensor) -> Tensor:
        """Old log-probs from a no-grad eval-mode forward (upstream generate_sequences)."""
        self.policy.eval()
        out = batch.log_probs.clone()
        e_idx, s_idx = decision_valid.nonzero(as_tuple=True)
        for i in range(0, len(e_idx), self.rl.micro_batch):
            e, s = e_idx[i : i + self.rl.micro_batch], s_idx[i : i + self.rl.micro_batch]
            obs, instr = self._gather(batch, e, s)
            lp = self.policy.log_prob(obs, instr, batch.tokens[e, s], self.rl.temperature)
            out[e, s] = lp.log_probs.to(STORAGE, torch.float32)
        return out

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        rl = self.rl
        t0 = time.perf_counter()
        n_traj = batch.num_trajectories
        advantages = group_advantages(batch.group_rewards, rl.adv_eps).reshape(n_traj)
        decision_valid = batch.action_mask.any(dim=-1)  # (E, S)
        token_mask = batch.action_mask.unsqueeze(-1).expand_as(batch.tokens)  # (E, S, C, A)
        old_log_probs = batch.log_probs
        mismatch = 0.0
        if rl.recompute_old_log_probs:
            old_log_probs = self._recompute_log_probs(batch, decision_valid)
            mismatch = float((old_log_probs - batch.log_probs).abs()[token_mask].max())

        self.policy.train()
        sums = {
            "pg_loss": 0.0,
            "clip_frac": 0.0,
            "approx_kl": 0.0,
            "entropy": 0.0,
            "grad_norm": 0.0,
        }
        first_dev = 0.0
        n_opt = 0
        for epoch in range(rl.ppo_epochs):
            for start in range(0, n_traj, rl.ppo_mini_batch):
                traj = torch.arange(start, min(n_traj, start + rl.ppo_mini_batch))
                weights = seq_mean_token_mean_weights(token_mask[traj])  # sums to 1
                e_loc, s_idx = decision_valid[traj].nonzero(as_tuple=True)
                self.optimizer.zero_grad(set_to_none=True)
                for i in range(0, len(e_loc), rl.micro_batch):
                    el, s = e_loc[i : i + rl.micro_batch], s_idx[i : i + rl.micro_batch]
                    e = traj[el]
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
                    sums["clip_frac"] += float(pl.clip_frac)
                    sums["approx_kl"] += float(pl.approx_kl)
                    sums["entropy"] += float((out.entropy.detach() * w).sum())
                    if epoch == 0 and start == 0:  # params == rollout params: ratio must be 1
                        dev_tok = (out.log_probs.detach() - old).abs()[w > 0]
                        first_dev = max(first_dev, float(dev_tok.max()))
                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, rl.grad_clip)
                self.optimizer.step()
                sums["grad_norm"] += float(grad_norm)
                n_opt += 1
        self.optimizer.zero_grad(set_to_none=True)
        stats = {f"train/{key}": value / n_opt for key, value in sums.items()}
        stats.update(
            {
                "train/reward_mean": float(batch.rewards.mean()),
                "train/adv_abs_mean": float(advantages.abs().mean()),
                "train/trajectories": float(n_traj),
                "train/decisions": float(decision_valid.sum()),
                "train/optimizer_steps": float(n_opt),
                "train/logprob_mismatch": mismatch,
                "train/first_minibatch_max_abs_log_ratio": first_dev,
                "train/update_time_s": time.perf_counter() - t0,
            }
        )
        return stats

    # ------------------------------------------------------------------ checkpoint / loop
    def save_checkpoint(self, step: int, eval_history: list[tuple[int, float]]) -> Path:
        path = self.run_dir / "checkpoints" / f"step_{step:06d}"
        self.policy.save(path / "adapter")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        state = {"step": step, "next_prompt_seed": self._next_prompt, "eval_history": eval_history}
        (path / "trainer_state.json").write_text(json.dumps(state, indent=2))
        return path

    def run(self) -> dict[str, Any]:
        rl = self.rl
        t_start = time.perf_counter()
        log.info("GRPO: %d trainable params, rl=%s", self.num_trainable, asdict(rl))
        baseline = self.evaluate()
        self._log({"kind": "eval", "step": 0, **baseline})
        log.info("step 0 eval success %.3f", baseline["eval/success"])
        history: list[tuple[int, float]] = [(0, baseline["eval/success"])]
        check_step = kill_switch_step(rl)
        evals = eval_steps(rl)
        stopped_flat = False
        step = 0
        try:
            for step in range(1, rl.steps + 1):
                t0 = time.perf_counter()
                batch, row = self.collect()
                if batch is None:
                    log.warning("step %d: every group was all-equal; update skipped", step)
                    row["train/update_skipped"] = 1.0
                else:
                    row.update(self.update(batch))
                if step in evals:
                    row.update(self.evaluate())
                    history.append((step, row["eval/success"]))
                if step == check_step and is_flat_run(
                    baseline["eval/success"], row["eval/success"], rl.early_stop_min_improvement
                ):
                    stopped_flat = rl.early_stop_on_flat
                    self._flat_run_warning(step, baseline["eval/success"], row["eval/success"])
                    row["event/flat_run"] = 1.0
                if step % rl.checkpoint_every == 0 or step == rl.steps or stopped_flat:
                    self.save_checkpoint(step, history)
                row["time/step_s"] = time.perf_counter() - t0
                row["time/elapsed_s"] = time.perf_counter() - t_start
                self._log({"kind": "train", "step": step, **row})
                self._console(step, row)
                if stopped_flat:
                    break
        finally:
            self.close()
        return {
            "steps_completed": step,
            "stopped_flat": stopped_flat,
            "initial_eval_success": history[0][1],
            "final_eval_success": history[-1][1],
            "best_eval_success": max(s for _, s in history),
            "eval_history": history,
            "trainable_params": self.num_trainable,
            "wall_time_s": time.perf_counter() - t_start,
        }

    def _flat_run_warning(self, step: int, baseline: float, current: float) -> None:
        bar = "!" * 78
        action = (
            "STOPPING (rl.early_stop_on_flat=true)"
            if self.rl.early_stop_on_flat
            else ("continuing because rl.early_stop_on_flat=false")
        )
        log.warning(
            "\n%s\nFLAT-RUN: eval success %.3f at step %d (%.0f%% of %d steps) has not improved "
            "over the step-0 eval %.3f (min improvement %.3f).\nCheck rollout/frac_groups_dropped "
            "(all-0 / all-1 groups), temperature, LoRA lr. %s.\n%s",
            bar,
            current,
            step,
            100 * self.rl.early_stop_check_at_frac,
            self.rl.steps,
            baseline,
            self.rl.early_stop_min_improvement,
            action,
            bar,
        )

    @staticmethod
    def _console(step: int, row: dict[str, Any]) -> None:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "-" if value is None else f"{cast(float, value):.3f}"

        log.info(
            "step %d | rollout success %s dropped %s | clip %s kl %s ent %s | eval %s | %ss",
            step,
            fmt("rollout/success_rate"),
            fmt("rollout/frac_groups_dropped"),
            fmt("train/clip_frac"),
            fmt("train/approx_kl"),
            fmt("train/entropy"),
            fmt("eval/success"),
            fmt("time/step_s"),
        )
