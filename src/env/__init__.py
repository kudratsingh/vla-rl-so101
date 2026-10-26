"""SO-101 tabletop simulation (ManiSkill 3): public API.

Contract with the data / policy / RL code — keep these names and semantics stable:

- ``make_env(cfg, task, num_envs=None, cell="seen", render=True)`` -> gymnasium env with batched
  torch I/O (N sub-envs). ``env.unwrapped.get_instructions()`` -> list[str], one per sub-env.
- ``policy_obs(obs)`` -> {"image": uint8 (N, H, W, 3) front camera, "state": float32 (N, 6) joint
  positions in radians, order JOINT_NAMES}.
- Actions: float32 (N, 6) absolute joint position targets in radians (``pd_joint_pos``).
- Privileged oracle state lives in ``obs["extra"]`` (never feed it to a policy).

Heavy imports (ManiSkill/SAPIEN) happen inside ``make_env`` so importing this package is cheap.
"""

from __future__ import annotations

from typing import Any

from src.eval.protocol import CELLS, TASKS

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
CONTROL_HZ: int = 15  # = configs/real.yaml robot.control_hz
CAMERA_NAME = "front"


MOLTENVK_ICD = "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"


def _mac_vulkan_defaults() -> None:
    """On macOS SAPIEN needs MoltenVK's Vulkan ICD (scripts/mac_env.sh); default it if unset.
    Only effective before SAPIEN's renderer is first created in this process."""
    import os
    import sys

    if sys.platform == "darwin" and os.path.exists(MOLTENVK_ICD):
        os.environ.setdefault("VK_ICD_FILENAMES", MOLTENVK_ICD)
        os.environ.setdefault("MVK_CONFIG_LOG_LEVEL", "0")


def make_env(
    cfg: dict[str, Any],
    task: str,
    *,
    num_envs: int | None = None,
    cell: str = "seen",
    render: bool = True,
    **kwargs: Any,
) -> Any:
    """Build the ManiSkill env for `task`.

    Backend from cfg["device"]: "cpu" -> physx_cpu (N single envs run serially, see vector.py),
    "cuda" -> physx_cuda (one GPU-parallel env). `render=False` drops images (obs_mode
    "state_dict") for fast oracle/eval runs. Extra kwargs go to gym.make (e.g. control_mode,
    render_mode, robot_init_qpos_noise).
    """
    _mac_vulkan_defaults()
    import gymnasium as gym

    from src.env.randomization import randomization_from_cfg
    from src.env.tasks import TASK_ENV_IDS
    from src.env.vector import SerialVecEnv

    if task not in TASK_ENV_IDS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")
    env_cfg = cfg.get("env", {})
    n = int(num_envs if num_envs is not None else env_cfg.get("num_envs", 1))
    device = str(cfg.get("device", "cpu"))
    make_kwargs: dict[str, Any] = {
        "obs_mode": "rgb" if render else "state_dict",
        "control_mode": "pd_joint_pos",
        "render_mode": "rgb_array",
        "max_episode_steps": int(env_cfg.get("max_episode_steps", 150)),
        "cell": cell,
        "image_size": int(env_cfg.get("image_size", 128)),
        "camera": env_cfg.get("camera"),
        "randomization": randomization_from_cfg(
            bool(env_cfg.get("randomize", False)), env_cfg.get("randomization")
        ),
    }
    make_kwargs.update(kwargs)
    env_id = TASK_ENV_IDS[task]
    if device.startswith("cuda"):
        return gym.make(env_id, num_envs=n, sim_backend="physx_cuda", **make_kwargs)
    if device != "cpu":
        raise ValueError(f"unsupported device {device!r}; use 'cpu' or 'cuda'")
    envs = [gym.make(env_id, num_envs=1, sim_backend="physx_cpu", **make_kwargs) for _ in range(n)]
    return SerialVecEnv(envs)


def policy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    """{"image": uint8 (N, H, W, 3), "state": float32 (N, 6)} from a raw env observation."""
    import torch

    if "sensor_data" not in obs:
        raise ValueError("observation has no images; build the env with make_env(render=True)")
    image = obs["sensor_data"][CAMERA_NAME]["rgb"]
    state = obs["agent"]["qpos"]
    return {
        "image": torch.as_tensor(image).to(torch.uint8),
        "state": torch.as_tensor(state).float(),
    }


__all__ = [
    "CAMERA_NAME",
    "CELLS",
    "CONTROL_HZ",
    "JOINT_NAMES",
    "TASKS",
    "make_env",
    "policy_obs",
]
