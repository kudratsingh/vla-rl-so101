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


def _approach_speeds(profile: str) -> tuple[list[float], bool]:
    """TCP step lengths (m) within 10 cm of the cube before the grasp closes, and whether the
    script ran to its end (released, retreated) after the success check fired."""
    cfg = load_config("tiny")
    env = make_env(cfg, "PickPlace", num_envs=1, render=False)
    oracle = make_oracle("PickPlace", env, profile)
    try:
        obs, _ = env.reset(seed=0)
        oracle.reset(obs)
        speeds: list[float] = []
        succeeded_at = None
        for t in range(cfg["env"]["max_episode_steps"] + 45):
            tcp0 = obs["extra"]["tcp_pose"][0, :3].clone()
            near = float((obs["extra"]["obj_pose"][0, :3] - tcp0).norm()) < 0.10
            before_grasp = int(oracle.phase[0]) < oracle.contact_phase()
            obs, _, _, _, info = env.step(oracle.act(obs))
            if near and before_grasp:
                speeds.append(float((obs["extra"]["tcp_pose"][0, :3] - tcp0).norm()))
            if succeeded_at is None and bool(info["success"][0]):
                succeeded_at = t
            if succeeded_at is not None and bool(oracle.finished()[0]):
                return speeds, t - succeeded_at <= 45
        return speeds, False
    finally:
        env.close()


def test_v2_profile_slows_the_final_approach() -> None:
    fast, _ = _approach_speeds("v1")
    slow, finished = _approach_speeds("v2")
    assert max(slow) <= 0.0055 < max(fast)  # ~0.5 cm/step vs the v1 joint-cap speed
    assert len(slow) >= len(fast) + 6  # more control steps (decisions) near the cube
    assert finished  # release + retreat complete within the recorder's tail cap


def test_oracle_eval_reports_decisions_per_episode() -> None:
    import importlib.util
    import sys

    from src.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "wo01_oracle_eval", REPO_ROOT / "scripts" / "wo01_oracle_eval.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    r = mod.run_cell(load_config("tiny"), "Push", "seen", 1, 1, 0, profile="v2", chunk=8)
    assert r["rate"] == 1.0
    assert r["decisions_success"] == -(-r["mean_steps_success"] // 8)
    assert r["near_decisions"] == pytest.approx(r["near_steps"] / 8, abs=0.01)
    assert r["near_tcp_cm_per_step"] < 0.6
