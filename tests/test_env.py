"""Smoke tests for the SO-101 ManiSkill env (CPU, tiny preset)."""

from __future__ import annotations

import math

import pytest
import torch
import yaml

from src.config import CONFIG_DIR, load_config
from src.env import CONTROL_HZ, JOINT_NAMES, TASKS, make_env, policy_obs
from src.env import success as S
from src.eval import protocol as P


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config("tiny")


@pytest.mark.parametrize("task", TASKS)
def test_task_builds_steps_and_reports(cfg: dict, task: str) -> None:
    env = make_env(cfg, task, num_envs=2)
    try:
        obs, info = env.reset(seed=0)
        po = policy_obs(obs)
        size = cfg["env"]["image_size"]
        assert po["image"].shape == (2, size, size, 3) and po["image"].dtype == torch.uint8
        assert po["state"].shape == (2, len(JOINT_NAMES)) and po["state"].dtype == torch.float32
        instr = env.unwrapped.get_instructions()
        assert len(instr) == 2 and all("red cube" in s for s in instr)
        for key in ("tcp_pose", "obj_pose", "goal_pose"):
            assert obs["extra"][key].shape == (2, 7)
        obs, rew, term, trunc, info = env.step(po["state"])  # hold still
        assert info["success"].shape == (2,) and info["success"].dtype == torch.bool
        assert not bool(info["success"].any())
        assert rew.shape == term.shape == trunc.shape == (2,)
        # partial reset keeps the other sub-env untouched
        before = obs["extra"]["obj_pose"][1].clone()
        obs, _ = env.reset(seed=5, options={"env_idx": [0]})
        assert torch.allclose(obs["extra"]["obj_pose"][1], before)
    finally:
        env.close()


def test_unseen_cell_uses_held_out_objects(cfg: dict) -> None:
    env = make_env(cfg, "Stack", num_envs=1, cell="unseen_objects", render=False)
    try:
        labels = set()
        for seed in range(8):
            env.reset(seed=seed)
            labels.add(env.unwrapped.get_instructions()[0])
        held_out = {o.label for o in P.UNSEEN_OBJECTS}
        assert all(any(h in s for h in held_out) for s in labels)
        assert len(labels) == 2  # both held-out objects appear
    finally:
        env.close()


def test_shifted_cell_spawns_outside_seen_region(cfg: dict) -> None:
    env = make_env(cfg, "PickPlace", num_envs=1, cell="shifted", render=False)
    try:
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            x, y = obs["extra"]["obj_pose"][0, :2].tolist()
            assert P.SHIFTED_REGION.contains(x, y) and not P.SEEN_REGION.contains(x, y)
    finally:
        env.close()


def test_policy_obs_requires_images(cfg: dict) -> None:
    env = make_env(cfg, "Push", num_envs=1, render=False)
    try:
        obs, _ = env.reset(seed=0)
        with pytest.raises(ValueError):
            policy_obs(obs)
    finally:
        env.close()


def test_control_rate_matches_real_arm() -> None:
    real = yaml.safe_load((CONFIG_DIR / "real.yaml").read_text())
    assert CONTROL_HZ == real["robot"]["control_hz"]


def test_success_checks_resist_gaming() -> None:
    z = torch.zeros(1, 3)
    bowl = torch.zeros(1, 3)
    up = torch.ones(1)
    in_bowl = torch.tensor([[0.0, 0.0, 0.021]])
    no = torch.tensor([False])
    yes = torch.tensor([True])
    assert S.pick_place(in_bowl, z, z, bowl, up, no, 0.015)["success"].item()
    assert not S.pick_place(in_bowl, z, z, bowl, up, yes, 0.015)["success"].item()  # still held
    held_above = torch.tensor([[0.0, 0.0, 0.08]])
    assert not S.pick_place(held_above, z, z, bowl, up, no, 0.015)["success"].item()
    moving = torch.tensor([[0.2, 0.0, 0.0]])
    assert not S.pick_place(in_bowl, moving, z, bowl, up, no, 0.015)["success"].item()

    on_target = torch.tensor([[0.0, 0.0, 0.015]])
    assert S.push(on_target, z, z, bowl, no, no, 0.015)["success"].item()
    assert not S.push(on_target, z, z, bowl, no, yes, 0.015)["success"].item()  # was carried
    assert not S.push(on_target, z, z, bowl, yes, no, 0.015)["success"].item()  # grasped

    base = torch.tensor([[0.0, 0.0, 0.015]])
    on_top = torch.tensor([[0.004, 0.0, 0.045]])
    assert S.stack(on_top, z, z, base, no, 0.015, 0.015)["success"].item()
    assert not S.stack(on_top, z, z, base, yes, 0.015, 0.015)["success"].item()  # touching
    beside = torch.tensor([[0.03, 0.0, 0.015]])
    assert not S.stack(beside, z, z, base, no, 0.015, 0.015)["success"].item()


def test_protocol_helpers() -> None:
    assert len(P.protocol_hash()) == 64
    lo, hi = P.wilson_interval(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99
    assert P.wilson_interval(0, 0) == (0.0, 1.0)
    assert P.SHIFTED_REGION.half_extent == tuple(1.5 * h for h in P.SEEN_REGION.half_extent)


def test_delta_controller_available(cfg: dict) -> None:
    env = make_env(cfg, "Push", num_envs=1, render=False, control_mode="pd_joint_delta_pos")
    try:
        obs, _ = env.reset(seed=0)
        q0 = obs["agent"]["qpos"].clone()
        step = torch.zeros(1, 6)
        step[0, 0] = 0.05  # shoulder_pan +0.05 rad per step
        for _ in range(4):
            obs, *_ = env.step(step)
        assert float(obs["agent"]["qpos"][0, 0] - q0[0, 0]) > 0.1
    finally:
        env.close()


def test_front_camera_frames_whole_workspace() -> None:
    """Project the workspace bounds (all cells' object region, bowl rim, gripper up to 13 cm)
    with the env's own camera matrices; everything must land inside the image with a margin."""
    import itertools

    cfg = load_config("sim")
    cfg["device"] = "cpu"
    env = make_env(cfg, "PickPlace", num_envs=1)
    try:
        obs, _ = env.reset(seed=0)
        param = obs["sensor_param"]["front"]
        K, E = param["intrinsic_cv"][0].double(), param["extrinsic_cv"][0].double()
        size = cfg["env"]["image_size"]
        assert abs(float(K[1, 1]) - (size / 2) / math.tan(math.radians(20.0))) < 1e-3  # fovy 40°
        (hx, hy), (cx, cy) = P.SHIFTED_REGION.half_extent, P.SHIFTED_REGION.center
        r = S.BOWL_FLOOR_RADIUS + 0.017  # bowl rim radius
        corners = [
            *itertools.product([cx - hx - r, cx + hx + r], [cy - hy - r, cy + hy + r], [0.0, 0.04]),
            *itertools.product([cx - hx, cx + hx], [cy - hy, cy + hy], [0.0, 0.13]),
        ]
        pts = torch.tensor([[*c, 1.0] for c in corners], dtype=torch.float64)
        cam = (E @ pts.T)[:3]
        uv = (K @ cam)[:2] / cam[2]
        margin = 0.04 * size
        assert bool((cam[2] > 0).all())
        assert float(uv.min()) >= margin and float(uv.max()) <= size - margin, uv
    finally:
        env.close()
