"""Camera jitter is drawn from the episode seed (review 2026-11-07, 3.2), not the global torch RNG;
with randomization off (sim.yaml, tiny.yaml) there is no jitter at all."""

from __future__ import annotations

import torch

from src.config import load_config
from src.env import make_env

CAMERA_ONLY = {
    "object_color_jitter": 0.0,
    "object_physics": False,
    "lighting": False,
    "camera_pos_jitter_m": 0.02,
    "camera_target_jitter_m": 0.01,
    "camera_roll_jitter_rad": 0.02,
    "camera_fov_jitter_deg": 0.0,
    "robot_color": False,
    "action_latency_max_steps": 0,
    "reconfigure_every": 0,
}


def test_camera_jitter_follows_episode_seed() -> None:
    assert load_config("sim")["env"]["randomize"] is False  # reported evals: no jitter
    cfg = load_config("tiny")
    cfg["env"] = {**cfg["env"], "randomize": True, "randomization": CAMERA_ONLY}
    env = make_env(cfg, "PickPlace", num_envs=2, render=False)
    try:

        def mount_pose(seeds: list[int], torch_seed: int) -> torch.Tensor:
            torch.manual_seed(torch_seed)
            env.reset(seed=seeds)
            return torch.cat([e.unwrapped.camera_mount.pose.raw_pose for e in env.envs])

        a = mount_pose([3, 4], torch_seed=1)
        b = mount_pose([3, 4], torch_seed=2)  # global torch RNG differs: same camera
        c = mount_pose([5, 4], torch_seed=1)
        assert torch.equal(a, b)
        assert torch.equal(a[1], c[1]) and not torch.equal(a[0], c[0])
        eye = a[:, :3]
        nominal = torch.tensor(env.envs[0].unwrapped.camera_spec.eye)
        assert float((eye - nominal).abs().max()) <= 0.02 + 1e-6
    finally:
        env.close()

    plain = make_env(load_config("tiny"), "PickPlace", num_envs=1, render=False)
    try:
        assert not hasattr(plain.envs[0].unwrapped, "camera_mount")
    finally:
        plain.close()
