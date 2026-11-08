"""Serial batch of single CPU ManiSkill envs behind the same batched-tensor API as a GPU env.

ManiSkill's CPU backend (physx_cpu) supports one sub-env per instance, so on the Mac / `tiny`
preset `make_env` runs N instances serially in-process and stacks their outputs along dim 0.

Seeding matches ManiSkill's GPU convention: `reset(seed=s)` gives sub-env i the episode seed
[s, *RandomState(s).randint(2**31, size=N-1)][i], and the tasks sample per-env state from
ManiSkill's per-env `_batched_episode_rng`, so a seed yields the same episodes on CPU and GPU.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


def _stack(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {k: _stack([it[k] for it in items]) for k in first}
    if isinstance(first, torch.Tensor):
        if first.ndim == 0:
            return torch.stack(items)
        return torch.cat(items, dim=0)
    if isinstance(first, np.ndarray):
        return np.concatenate(items, axis=0)
    if isinstance(first, (bool, int, float)):
        return torch.tensor(items)
    return items


def _replace(old: Any, new: Any, idx: list[int]) -> Any:
    """Write rows `idx` of batched tree `new` (len(idx) rows) into batched tree `old`."""
    if isinstance(old, dict):
        return {k: _replace(old[k], new[k], idx) for k in old}
    if isinstance(old, torch.Tensor) and old.ndim > 0:
        out = old.clone()
        out[idx] = new.to(out.dtype)
        return out
    return new


def episode_seeds(seed: int | list[int] | None, n: int) -> list[int | None]:
    if seed is None:
        return [None] * n
    if isinstance(seed, (list, tuple, np.ndarray)):
        if len(seed) != n:
            raise ValueError(f"got {len(seed)} seeds for {n} envs")
        return [int(s) for s in seed]
    extra = np.random.RandomState(seed).randint(2**31, size=n - 1).tolist() if n > 1 else []
    return [int(seed), *extra]


class SerialVecEnv(gym.Env):
    """N single-env ManiSkill instances stepped serially; batched torch I/O like a GPU env."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"]}  # noqa: RUF012 (gym API)

    def __init__(self, envs: list[gym.Env]) -> None:
        if not envs:
            raise ValueError("need at least one env")
        self.envs = envs
        self.num_envs = len(envs)
        base: Any = envs[0].unwrapped
        self.device = base.device
        self.single_action_space = base.single_action_space
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.single_observation_space = base.single_observation_space
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, self.num_envs
        )
        self.render_mode = envs[0].render_mode
        self.control_mode = base.control_mode
        self.task: str = getattr(base, "TASK", "")
        self.cell: str = getattr(base, "cell", "")
        self.max_episode_steps = getattr(envs[0], "_max_episode_steps", None)
        self._obs: Any = None
        self._info: Any = None

    def reset(
        self, *, seed: int | list[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        options = dict(options or {})
        env_idx = options.pop("env_idx", None)
        idx = list(range(self.num_envs)) if env_idx is None else [int(i) for i in env_idx]
        seeds = episode_seeds(seed, self.num_envs)
        results = [self.envs[i].reset(seed=seeds[i], options=options) for i in idx]
        obs = _stack([r[0] for r in results])
        info = _stack([r[1] for r in results])
        if self._obs is None or len(idx) == self.num_envs:
            self._obs, self._info = obs, info
        else:
            self._obs = _replace(self._obs, obs, idx)
            self._info = _replace(self._info, info, idx)
        return self._obs, self._info

    def step(  # type: ignore[override]  # batched tensors, like ManiSkill's own envs
        self, action: Any
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        a = torch.as_tensor(action, dtype=torch.float32)
        if a.ndim == 1:
            a = a[None].expand(self.num_envs, -1)
        out = [env.step(a[i : i + 1]) for i, env in enumerate(self.envs)]
        obs = _stack([o[0] for o in out])
        rew = _stack([o[1] for o in out])
        term = _stack([o[2] for o in out])
        trunc = _stack([o[3] for o in out])
        info = _stack([o[4] for o in out])
        self._obs, self._info = obs, info
        return obs, rew, term, trunc, info

    def render(self) -> Any:
        return _stack([env.render() for env in self.envs])

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_instructions(self) -> list[str]:
        out: list[str] = []
        for env in self.envs:
            base: Any = env.unwrapped
            out += base.get_instructions()
        return out

    @property
    def elapsed_steps(self) -> torch.Tensor:
        bases: list[Any] = [env.unwrapped for env in self.envs]
        return torch.cat([b.elapsed_steps for b in bases])

    # ------------------------------------------------------------------ state (src/env/state.py)
    def get_state_dict(self) -> dict[str, Any]:
        """ManiSkill state dict of every sub-env, rows stacked: same layout as a GPU env's."""
        bases: list[Any] = [env.unwrapped for env in self.envs]
        out: dict[str, Any] = _stack([b.get_state_dict() for b in bases])
        return out

    def set_state_dict(self, state: dict[str, Any], env_idx: list[int] | None = None) -> None:
        """Write row i of `state` into sub-env i (all rows, or the sub-envs in `env_idx` with one
        row each, in that order). Call `get_obs()` afterwards: cached observations are stale."""
        idx = list(range(self.num_envs)) if env_idx is None else [int(i) for i in env_idx]
        for row, i in enumerate(idx):
            base: Any = self.envs[i].unwrapped
            base.set_state_dict(_rows(state, row))

    def get_obs(self) -> Any:
        """Fresh batched observation of the current sim state (re-rendered), cached for resets."""
        bases: list[Any] = [env.unwrapped for env in self.envs]
        self._obs = _stack([b.get_obs() for b in bases])
        return self._obs


def _rows(tree: Any, row: int) -> Any:
    """Row `row` of every tensor in a batched tree, keeping a batch dim of 1."""
    if isinstance(tree, dict):
        return {k: _rows(v, row) for k, v in tree.items()}
    if isinstance(tree, torch.Tensor) and tree.ndim > 0:
        return tree[row : row + 1]
    return tree
