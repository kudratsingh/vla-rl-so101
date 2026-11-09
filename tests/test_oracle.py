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


def test_oracle_takes_over_an_in_hand_state() -> None:
    """Recovery corrections: restore a mid-carry state, start in "lift" with the last gripper
    command, finish the task (src/env/state.py restore + reset(start_phase="infer"))."""
    from src.env.state import restore, snapshot

    cfg = load_config("tiny")
    env = make_env(cfg, "PickPlace", num_envs=1, render=False)
    try:
        seeds = [3_000_000_007]
        obs, _ = env.reset(seed=seeds)
        oracle = make_oracle("PickPlace", env)
        oracle.reset(obs)
        lift = oracle.names.index("lift")
        grasped, cmd = False, None
        for _ in range(cfg["env"]["max_episode_steps"]):
            cmd = oracle.act(obs)
            obs, _, _, _, info = env.step(cmd)
            grasped = bool(info["is_grasped"][0])
            if grasped and int(oracle.phase[0]) > lift:  # lifted and carrying
                break
        assert grasped and cmd is not None
        snap = snapshot(env)
        # reset runs no physics step, so it reports the held grasp of the episode before; the
        # trace's reset step (recovery failure mining) must not
        from src.eval.sim_eval import run_batch_traced

        _, reset_info = env.reset(seed=seeds)
        assert bool(reset_info["is_grasped"][0])
        hold = cmd[:, None].expand(1, 8, 6)
        trace = run_batch_traced(lambda o, i: hold, env, lambda o: o, seeds, 1)[2]
        assert not bool(trace.steps[0]["is_grasped"].any())
        obs = restore(env, seeds, snap)
        fresh = make_oracle("PickPlace", env)
        with pytest.raises(ValueError, match="start_phase"):
            fresh.reset(obs, start_phase="fly")
        fresh.reset(obs, start_phase="infer", grasped=torch.tensor([True]), grip_cmd=cmd[:, 5])
        assert fresh.names[int(fresh.phase[0])] == "lift"
        first = fresh.act(obs)
        assert float(first[0, 5]) <= float(cmd[0, 5]) + 1e-6  # no squeeze lost on the first step
        solved = False
        for _ in range(cfg["env"]["max_episode_steps"]):
            obs, _, _, trunc, info = env.step(first)
            solved = bool(info["success"][0])
            if solved or bool(trunc[0]):
                break
            first = fresh.act(obs)
        assert solved, "oracle did not finish from the in-hand state"
        assert int(fresh.retries[0]) == 0  # carried on, did not drop and re-grasp
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
