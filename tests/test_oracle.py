"""Smoke test: the scripted oracle solves each task on a couple of seeded episodes (CPU)."""

from __future__ import annotations

import pytest
import torch

from src.config import load_config
from src.data.oracle import MAX_JOINT_STEP, make_oracle
from src.env import TASKS, make_env


@pytest.mark.parametrize("task", TASKS)
def test_oracle_solves_task(task: str) -> None:
    cfg = load_config("tiny")
    env = make_env(cfg, task, num_envs=2, render=False)
    oracle = make_oracle(task, env)
    try:
        obs, _ = env.reset(seed=1)
        oracle.reset(obs)
        solved = torch.zeros(2, dtype=torch.bool)
        prev = obs["agent"]["qpos"].clone()
        for _ in range(cfg["env"]["max_episode_steps"]):
            action = oracle.act(obs)
            assert action.shape == (2, 6) and action.dtype == torch.float32
            # demos must respect the real arm's per-step joint cap
            assert float((action - prev).abs().max()) <= MAX_JOINT_STEP + 1e-5
            prev = action
            obs, _, term, trunc, info = env.step(action)
            solved |= term & info["success"]
            if bool((solved | trunc).all()):
                break
        assert bool(solved.all()), f"{task}: oracle solved {solved.tolist()}"
    finally:
        env.close()
