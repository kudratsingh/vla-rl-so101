"""Synthetic verification task for the GRPO loop (WO-03): a batched grid "reach" with a known optimum.

Same contract shape as the ManiSkill env (src.env): batched torch tensors, gymnasium-style
`reset(seed=list[int]) -> (obs, info)` and `step(action) -> (obs, reward, terminated, truncated,
info)`, `info["success"]` bool (N,), `env.unwrapped.get_instructions() -> list[str]`, and
`policy_obs(obs) -> {"image": uint8 (N,H,W,3), "state": float32 (N,6)}`. Actions are float32 (N, 2)
in [-1, 1].

Task: the agent (gripper) and a goal cell sit on a G x G grid, both drawn from the per-env reset
seed (goal != start). Each env step moves the agent by m = round-to-{-1,0,+1} of each action dim
(|a| <= 1/3 -> 0), clamped to the grid; diagonal moves are allowed. Success (reward 1, terminated)
the moment agent == goal; truncated after `max_episode_steps`. Reward is sparse and binary.

Known optimal policy: move each coordinate one cell toward the goal per step (`optimal_chunk`).
The Chebyshev distance is <= G - 1 = 4 < max_episode_steps = 6, so the optimum succeeds on every
episode: optimal return = 1.0. A uniformly random policy succeeds ~15% of the time (G=5, 6 steps;
measured over 2e5 episodes), and the policy acts in open-loop chunks of C=2 steps from one
observation, so it must learn when the second step of a chunk should stop (|d| = 1) and when it
should continue (|d| >= 2): random init is clearly sub-optimal and RL has to do real work.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.config import Config

STATE_DIM = 6
ACTION_DIM = 2
MOVE_THRESHOLD = 1.0 / 3.0


class ToyGridEnv:
    def __init__(
        self,
        num_envs: int,
        *,
        grid_size: int = 5,
        max_episode_steps: int = 6,
        task: str = "reach",
        device: str | torch.device = "cpu",
    ) -> None:
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if max_episode_steps < grid_size - 1:
            raise ValueError(
                "max_episode_steps < grid_size - 1: the optimum could not always succeed"
            )
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.max_episode_steps = max_episode_steps
        self.task = task
        self.device = torch.device(device)
        self._pos = torch.zeros(num_envs, 2, dtype=torch.long, device=self.device)
        self._goal = torch.zeros(num_envs, 2, dtype=torch.long, device=self.device)
        self._done = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._t = 0

    @property
    def unwrapped(self) -> ToyGridEnv:
        return self

    def get_instructions(self) -> list[str]:
        return [f"{self.task}: move the gripper to the green cell"] * self.num_envs

    def reset(
        self, *, seed: int | Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        if seed is None:
            seeds: list[int | None] = [None] * self.num_envs
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(self.num_envs)]
        else:
            seeds = list(seed)
            if len(seeds) != self.num_envs:
                raise ValueError(f"got {len(seeds)} seeds for {self.num_envs} envs")
        g = self.grid_size
        pos, goal = [], []
        for s in seeds:
            rng = np.random.default_rng(s)
            start = rng.integers(0, g, size=2)
            target = rng.integers(0, g, size=2)
            while (target == start).all():
                target = rng.integers(0, g, size=2)
            pos.append(start)
            goal.append(target)
        self._pos = torch.as_tensor(np.stack(pos), dtype=torch.long, device=self.device)
        self._goal = torch.as_tensor(np.stack(goal), dtype=torch.long, device=self.device)
        self._done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._t = 0
        return self._obs(), {}

    def step(
        self, action: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(f"action must be ({self.num_envs}, {ACTION_DIM}), got {action.shape}")
        move = (action > MOVE_THRESHOLD).long() - (action < -MOVE_THRESHOLD).long()
        move = move * (~self._done).long().unsqueeze(1)  # finished envs stay put
        self._pos = (self._pos + move).clamp(0, self.grid_size - 1)
        self._t += 1
        success = (self._pos == self._goal).all(dim=1)
        terminated = success.clone()
        truncated = ~success & (self._t >= self.max_episode_steps)
        self._done = self._done | terminated | truncated
        return self._obs(), success.float(), terminated, truncated, {"success": success}

    def close(self) -> None:
        pass

    def _obs(self) -> dict[str, Tensor]:
        scale = float(self.grid_size - 1)
        p, g = self._pos.float(), self._goal.float()
        state = torch.cat([p, g, g - p], dim=1) / scale  # (N, 6): agent, goal, goal - agent
        image = torch.zeros(
            self.num_envs, self.grid_size, self.grid_size, 3, dtype=torch.uint8, device=self.device
        )
        idx = torch.arange(self.num_envs, device=self.device)
        image[idx, self._pos[:, 1], self._pos[:, 0], 0] = 255  # agent: red
        image[idx, self._goal[:, 1], self._goal[:, 0], 1] = 255  # goal: green
        return {"state": state.to(torch.float32), "image": image}


def policy_obs(obs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Toy counterpart of src.env.policy_obs: the env already emits {"image", "state"}."""
    return {"image": obs["image"], "state": obs["state"]}


def optimal_chunk(state: Tensor, grid_size: int, chunk_len: int) -> Tensor:
    """The known optimal open-loop chunk from `state` (N, 6): (N, chunk_len, 2) in env units.

    Per coordinate, step i moves sign(d_i) with d_0 = goal - agent and d_{i+1} = d_i - sign(d_i):
    full speed toward the goal, stopping on it.
    """
    d = torch.round(state[:, 4:6] * (grid_size - 1)).long()
    steps = []
    for _ in range(chunk_len):
        m = torch.sign(d)
        steps.append(m)
        d = d - m
    return torch.stack(steps, dim=1).float()


EnvFactory = Callable[[str, int], ToyGridEnv]


def make_toy_env_factory(cfg: Config) -> EnvFactory:
    toy = cfg["toy"]

    def factory(task: str, num_envs: int) -> ToyGridEnv:
        return ToyGridEnv(
            num_envs,
            grid_size=int(toy["grid_size"]),
            max_episode_steps=int(toy["max_episode_steps"]),
            task=task,
            device="cpu",
        )

    return factory
