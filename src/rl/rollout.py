"""Rollout worker: N parallel envs, K samples per prompt, binary terminal reward (ADR-0002).

Loop logic adapted from PRIME-RL/SimpleVLA-RL @ 7c51662df27b586f9e8a1ab35fcf849f2b8852f9 (MIT;
file header Apache-2.0, Copyright 2024 Bytedance Ltd.), verl/workers/rollout/rob_rollout.py
`RobHFRollout._generate_minibatch_libero` + `env_worker` and verl/trainer/main_ppo.py
`RobRewardManager.verify`:
  * a group = one prompt (task + reset seed = initial state) repeated K times in consecutive envs
    (upstream: `trial_id.repeat_interleave(n_samples)`); trajectory index e = g * K + k;
  * every decision samples one action chunk for all envs, which is executed step by step; an env
    whose episode ended (success, env termination/truncation, or the step limit) is no longer
    counted; `finish_step` = env steps executed, and only the tokens of executed actions are
    trained on (upstream mask `steps < finish_step` at action granularity);
  * reward = 1.0 if `info["success"]` fired before the episode ended, else 0.0, one scalar per
    trajectory (upstream `complete`).
Differences: envs are one batched (GPU) env instead of one process per env, so finished envs keep
receiving (ignored) actions; and the loop stops as soon as every env is done.

The env must reset deterministically per seed: identical seeds must give identical initial states,
otherwise a "group" does not share a prompt and the group baseline is wrong. `check_group_reset`
verifies this on every batch at no cost (tolerances in its docstring).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from src.rl.policy_api import RLPolicy

log = logging.getLogger(__name__)

ObsFn = Callable[[Any], dict[str, Tensor]]
STORAGE = torch.device("cpu")  # rollout buffers live on CPU; the policy moves inputs as needed
# check_group_reset tolerances (see its docstring)
RESET_FLOAT_ATOL = 1e-4  # joint positions, rad
RESET_IMAGE_DIFF_FRAC = 1e-3  # of an image's uint8 values


class VecEnv(Protocol):
    """The slice of the gymnasium / ManiSkill batched env API the rollout uses."""

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Tensor) -> tuple[Any, Any, Any, Any, dict[str, Any]]: ...

    @property
    def unwrapped(self) -> Any: ...


@dataclass
class RolloutBatch:
    """E = G * K trajectories, S = ceil(max_episode_steps / C) decision slots. All on CPU."""

    obs: dict[str, Tensor]  # each (E, S, ...): observation at each decision
    instructions: list[str]  # (E,)
    tasks: list[str]  # (E,)
    tokens: Tensor  # (E, S, C, A) int64
    log_probs: Tensor  # (E, S, C, A) float32, from the sampler
    action_mask: Tensor  # (E, S, C) bool: action executed while the episode was running
    rewards: Tensor  # (E,) float32 in {0, 1}
    episode_len: Tensor  # (E,) int64 env steps executed (upstream finish_step)
    group_size: int  # K

    @property
    def num_trajectories(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def num_groups(self) -> int:
        return self.num_trajectories // self.group_size

    @property
    def group_rewards(self) -> Tensor:
        return self.rewards.view(self.num_groups, self.group_size)

    def select_groups(self, keep: Tensor) -> RolloutBatch:
        """Keep whole groups (keep: (G,) bool), preserving the contiguous group layout."""
        traj = keep.repeat_interleave(self.group_size).nonzero(as_tuple=True)[0]
        return self._index(traj)

    def _index(self, idx: Tensor) -> RolloutBatch:
        ids = idx.tolist()
        return RolloutBatch(
            obs={k: v[idx] for k, v in self.obs.items()},
            instructions=[self.instructions[i] for i in ids],
            tasks=[self.tasks[i] for i in ids],
            tokens=self.tokens[idx],
            log_probs=self.log_probs[idx],
            action_mask=self.action_mask[idx],
            rewards=self.rewards[idx],
            episode_len=self.episode_len[idx],
            group_size=self.group_size,
        )

    @staticmethod
    def cat(batches: Sequence[RolloutBatch]) -> RolloutBatch:
        if not batches:
            raise ValueError("nothing to concatenate")
        k = batches[0].group_size
        if any(b.group_size != k for b in batches):
            raise ValueError("cannot concatenate batches with different group sizes")
        return RolloutBatch(
            obs={key: torch.cat([b.obs[key] for b in batches]) for key in batches[0].obs},
            instructions=[s for b in batches for s in b.instructions],
            tasks=[s for b in batches for s in b.tasks],
            tokens=torch.cat([b.tokens for b in batches]),
            log_probs=torch.cat([b.log_probs for b in batches]),
            action_mask=torch.cat([b.action_mask for b in batches]),
            rewards=torch.cat([b.rewards for b in batches]),
            episode_len=torch.cat([b.episode_len for b in batches]),
            group_size=k,
        )


def _flag(x: Any, n: int) -> Tensor:
    """Env flag (bool tensor on any device, or a scalar bool) -> (n,) bool on CPU."""
    t = torch.as_tensor(x).to(STORAGE, torch.bool)
    return t.expand(n).clone() if t.numel() == 1 else t.reshape(n)


def check_group_reset(
    obs: dict[str, Tensor], group_size: int, mode: Literal["error", "warn", "off"]
) -> float:
    """Largest fraction of one env's reset-observation values that differ from its group's first
    env (max over groups, envs and keys; 0.0 = every group shares its initial observation).

    Fails ("error" raises, "warn" logs) if any float value (joint positions) differs by more than
    RESET_FLOAT_ATOL, or more than RESET_IMAGE_DIFF_FRAC of an integer (uint8 image) observation
    differs at all. Measured on the CPU sim (WO-04, 3 tasks, 64 and 224 px): identical seeds are
    bit-identical; different seeds differ in >= 4.5% of the pixel values and >= 0.03 rad of qpos
    (the init-qpos noise alone). The image allowance is for the GPU sim, which keeps every
    sub-scene at its own world offset, so float32 pose rounding (~1e-6 m) can flip a few
    silhouette pixels between identical sub-scenes; it is ~45x below a real scene difference.
    """
    if mode == "off" or group_size == 1:
        return 0.0
    worst = 0.0
    bad = []
    for key, value in obs.items():
        v = value.to(STORAGE)
        grouped = v.reshape(-1, group_size, *v.shape[1:])
        first = grouped[:, :1]
        if v.is_floating_point():
            differs = (grouped.float() - first.float()).abs() > RESET_FLOAT_ATOL
            tol = 0.0
        else:
            differs = grouped != first
            tol = RESET_IMAGE_DIFF_FRAC
        frac = float(differs.reshape(grouped.shape[0], group_size, -1).float().mean(-1).max())
        worst = max(worst, frac)
        if frac > tol:
            bad.append(f"{key}: {frac:.2%} of values differ (allowed {tol:.2%})")
    if bad:
        msg = (
            f"envs reset with the same seed differ ({'; '.join(bad)}): a GRPO group would not "
            "share its prompt. The env must draw its initial state from the per-env seed."
        )
        if mode == "error":
            raise RuntimeError(msg + " Set rl.group_reset_check=warn to override.")
        log.warning(msg)
    return worst


@torch.no_grad()
def collect_rollouts(
    policy: RLPolicy,
    env: VecEnv,
    obs_fn: ObsFn,
    *,
    prompt_seeds: Sequence[int],
    group_size: int,
    temperature: float,
    max_episode_steps: int,
    task: str,
    group_reset_check: Literal["error", "warn", "off"] = "error",
) -> tuple[RolloutBatch, float]:
    """Run one batch of episodes: len(prompt_seeds) prompts x group_size samples each.

    The env must have exactly len(prompt_seeds) * group_size parallel envs. Returns the batch and
    the group-reset max diff (0.0 when groups share their initial observation).
    """
    seeds = [int(s) for s in prompt_seeds for _ in range(group_size)]
    n = len(seeds)
    policy.eval()
    raw_obs, _ = env.reset(seed=seeds)
    pobs = obs_fn(raw_obs)
    reset_diff = check_group_reset(pobs, group_size, group_reset_check)
    instructions = list(env.unwrapped.get_instructions())
    if len(instructions) != n:
        raise ValueError(f"env returned {len(instructions)} instructions for {n} envs")

    active = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    episode_len = torch.zeros(n, dtype=torch.long)
    obs_steps: list[dict[str, Tensor]] = []
    token_steps: list[Tensor] = []
    logp_steps: list[Tensor] = []
    mask_steps: list[Tensor] = []
    chunk_len = 0
    while bool(active.any()):
        step = policy.sample(pobs, instructions, temperature)
        chunk_len = int(step.tokens.shape[1])
        obs_steps.append({k: v.to(STORAGE).clone() for k, v in pobs.items()})
        token_steps.append(step.tokens.to(STORAGE, torch.long))
        logp_steps.append(step.log_probs.to(STORAGE, torch.float32))
        executed = torch.zeros(n, chunk_len, dtype=torch.bool)
        for c in range(chunk_len):
            if not bool(active.any()):
                break
            raw_obs, _, terminated, truncated, info = env.step(step.actions[:, c])
            executed[:, c] = active
            episode_len += active.long()
            hit = _flag(info["success"], n) & active
            success |= hit
            ended = hit | _flag(terminated, n) | _flag(truncated, n)
            ended |= episode_len >= max_episode_steps
            active &= ~ended
        mask_steps.append(executed)
        pobs = obs_fn(raw_obs)

    num_slots = math.ceil(max_episode_steps / chunk_len)
    if len(token_steps) > num_slots:
        raise RuntimeError(f"{len(token_steps)} decisions exceed {num_slots} slots")
    pad = num_slots - len(token_steps)

    def stack(steps: list[Tensor]) -> Tensor:
        out = torch.stack(steps, dim=1)
        if pad:
            out = torch.cat([out, out.new_zeros((n, pad, *out.shape[2:]))], dim=1)
        return out

    batch = RolloutBatch(
        obs={k: stack([o[k] for o in obs_steps]) for k in obs_steps[0]},
        instructions=instructions,
        tasks=[task] * n,
        tokens=stack(token_steps),
        log_probs=stack(logp_steps),
        action_mask=stack(mask_steps),
        rewards=success.float(),
        episode_len=episode_len,
        group_size=group_size,
    )
    return batch, reset_diff
