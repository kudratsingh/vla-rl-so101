"""Snapshot / restore of sim episodes (src/env/state.py) on the CPU SerialVecEnv path."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.config import load_config
from src.data.oracle import make_oracle
from src.env import make_env, policy_obs
from src.env.state import EnvSnapshot, restore, snapshot, state_error


def _run_oracle(env: Any, task: str, seeds: list[int], steps: int) -> Any:
    obs, _ = env.reset(seed=seeds)
    oracle = make_oracle(task, env)
    oracle.reset(obs)
    for _ in range(steps):
        obs, *_ = env.step(oracle.act(obs))
    return obs


@pytest.mark.parametrize("task", ["PickPlace", "Push"])
def test_restore_rows_into_other_envs(task: str) -> None:
    env = make_env(load_config("tiny"), task, num_envs=2, render=False)
    try:
        seeds = [3_000_000_000, 3_000_000_001]
        obs = _run_oracle(env, task, seeds, 30)
        snap = snapshot(env)
        assert snap.backend == "physx_cpu" and snap.num_rows == 2
        if task == "Push":
            assert set(snap.task) == {"_ever_lifted"}
            snap.task["_ever_lifted"][1] = True  # a latch set mid-episode must survive
        qpos = obs["agent"]["qpos"].clone()
        for _ in range(10):  # move on, then put the saved rows back in swapped order
            obs, *_ = env.step(qpos)
        swapped = EnvSnapshot.stack([snap.row(1), snap.row(0)])
        restored = restore(env, [seeds[1], seeds[0]], swapped)
        assert state_error(snapshot(env), swapped) < 1e-6
        assert torch.allclose(restored["agent"]["qpos"], qpos.flip(0), atol=1e-6)
        if task == "Push":
            assert env.envs[0].unwrapped._ever_lifted.tolist() == [True]
            assert env.envs[1].unwrapped._ever_lifted.tolist() == [False]
        with pytest.raises(ValueError, match="restore needs 2"):
            restore(env, seeds[:1], swapped)
        cuda_snap = EnvSnapshot(swapped.sim, "physx_cuda", swapped.task)
        with pytest.raises(ValueError, match="cannot restore"):
            restore(env, seeds, cuda_snap)
    finally:
        env.close()


@pytest.mark.needs_render
def test_restore_rerenders_the_saved_frame() -> None:
    env = make_env(load_config("tiny"), "PickPlace", num_envs=1)
    try:
        obs = _run_oracle(env, "PickPlace", [3_000_000_002], 40)
        image = policy_obs(obs)["image"].clone()
        snap = snapshot(env)
        reset_obs, _ = env.reset(seed=[3_000_000_002])
        assert (policy_obs(reset_obs)["image"].float() - image.float()).abs().mean() > 1.0
        restored = restore(env, [3_000_000_002], snap)
        assert (policy_obs(restored)["image"].float() - image.float()).abs().mean() < 1.0
    finally:
        env.close()


def test_traced_rollout_snapshots_restore_on_the_real_env() -> None:
    """run_batch_traced on the CPU env: same outcomes as run_batch, and a chunk-boundary snapshot
    restores the qpos the trace recorded at that step."""
    from src.eval.sim_eval import run_batch, run_batch_traced

    def obs_fn(raw: Any) -> dict[str, torch.Tensor]:
        return {"state": raw["agent"]["qpos"]}

    def act(obs: Any, instructions: Any) -> torch.Tensor:  # a fixed wobble, never succeeds
        g = torch.Generator().manual_seed(len(instructions))
        noise = 0.05 * torch.randn(len(instructions), 8, 6, generator=g)
        return obs["state"][:, None, :] + noise

    env = make_env(load_config("tiny"), "PickPlace", num_envs=2, render=False)
    try:
        seeds = [3_000_000_010, 3_000_000_011]
        s, ln = run_batch(act, env, obs_fn, seeds, 24)
        st, lt, trace = run_batch_traced(act, env, obs_fn, seeds, 24, snapshot_fn=snapshot)
        assert torch.equal(s, st) and torch.equal(ln, lt) and ln.tolist() == [24, 24]
        assert [b.step for b in trace.boundaries] == [0, 8, 16]
        assert {"qpos", "obj_pose", "goal_pose", "tcp_pose", "is_grasped"} <= set(trace.steps[8])
        b = trace.boundaries[1]
        obs = restore(env, seeds, b.state)
        assert torch.allclose(obs["agent"]["qpos"], trace.steps[8]["qpos"], atol=1e-6)
        assert torch.equal(b.last_action, trace.steps[8]["action"])
    finally:
        env.close()
