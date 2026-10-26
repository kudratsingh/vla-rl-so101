"""The single policy camera, ``front``: named pose -> extrinsics (robot base frame, metres).

``front_left_45``: in front of the workspace (+x, beyond it as seen from the robot) and to the
robot's left (+y), 45° azimuth between the two, looking down at 45° at the centre of the seen
spawn region, 0.70 m from it. On a real table: lens 0.35 m further from the robot than the
workspace centre, 0.35 m to the robot's left, 0.495 m above the table, aimed at the centre
(robot base frame: eye (0.555, 0.350, 0.495), target (0.205, 0, 0)).

``fov_deg`` is the VERTICAL field of view (ManiSkill passes CameraConfig.fov to SAPIEN as
``fovy``); images are square, so horizontal = vertical. 40° vertical fits common webcams
(e.g. C920 ~43° vertical at 640x480) after cropping. At 40° the 0.70 m distance keeps every
cell's spawn region, the bowl rim and the gripper (TCP up to 13 cm high) inside the frame with a
>= 5% margin (checked by projecting the workspace bounds; see tests/test_env.py).

The resolved values are written to ``configs/sim.yaml`` (``env.camera.eye`` / ``env.camera.target``);
an explicit ``eye``/``target`` in the config overrides the named pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.eval.protocol import SEEN_REGION

CAMERA_NAME = "front"
DEFAULT_POSE = "front_left_45"
DEFAULT_FOV_DEG = 40.0  # vertical

# name -> (azimuth from +x toward +y, elevation above the table, distance to target)
NAMED_POSES: dict[str, tuple[float, float, float]] = {
    "front_left_45": (45.0, 45.0, 0.70),
}


@dataclass(frozen=True)
class CameraSpec:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float
    pose_name: str


def workspace_center() -> tuple[float, float, float]:
    return (SEEN_REGION.center[0], SEEN_REGION.center[1], 0.0)


def named_pose(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """(eye, target) for a named camera pose."""
    if name not in NAMED_POSES:
        raise ValueError(f"unknown camera pose {name!r}; known: {sorted(NAMED_POSES)}")
    az, el, dist = NAMED_POSES[name]
    target = workspace_center()
    a, e = math.radians(az), math.radians(el)
    eye = (
        target[0] + dist * math.cos(e) * math.cos(a),
        target[1] + dist * math.cos(e) * math.sin(a),
        target[2] + dist * math.sin(e),
    )
    return tuple(round(v, 4) for v in eye), target  # type: ignore[return-value]


def resolve_front_camera(cam_cfg: dict[str, Any] | None) -> CameraSpec:
    """Camera spec from cfg ``env.camera`` (keys: pose, fov_deg, optional eye/target)."""
    cam_cfg = cam_cfg or {}
    name = str(cam_cfg.get("pose", DEFAULT_POSE))
    eye, target = named_pose(name)
    if "eye" in cam_cfg:
        eye = tuple(float(v) for v in cam_cfg["eye"])  # type: ignore[assignment]
    if "target" in cam_cfg:
        target = tuple(float(v) for v in cam_cfg["target"])  # type: ignore[assignment]
    return CameraSpec(eye, target, float(cam_cfg.get("fov_deg", DEFAULT_FOV_DEG)), name)
