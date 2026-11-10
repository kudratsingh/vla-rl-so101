"""configs/real.yaml as shipped: the values WO-06 bring-up writes into it reset to the placeholders.

``wo06_calibrate.py find-port/check --write-limits``, ``wo06_camera_intrinsics.py`` and the
owner's hand edits (camera index, robot.units, teleop.leader) change configs/real.yaml in place.
The robot tests must not depend on one arm's values, so they see this copy (conftest.py patches
``robot_fakes.real_cfg``'s loader); every other key still comes from the real file.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from src.config import load_config
from src.robot.units import UnitMap
from src.robot.yamlio import set_yaml_value

_U = UnitMap()
BRINGUP_PLACEHOLDERS: dict[tuple[str, ...], Any] = {
    ("robot", "port"): "/dev/tty.usbmodemXXXX",
    ("robot", "units"): {
        "arm_sign": [int(s) for s in _U.arm_sign],
        "arm_offset_deg": list(_U.arm_offset_deg),
        "gripper_closed_rad": _U.gripper_closed_rad,
        "gripper_open_rad": _U.gripper_open_rad,
    },
    ("robot", "safety", "joint_limits_deg"): {},
    ("cameras", "front", "index"): 0,
    ("cameras", "front", "intrinsics"): None,
    ("cameras", "wrist", "index"): 1,
    ("cameras", "wrist", "intrinsics"): None,
    ("teleop", "leader"): None,
}


def placeholder_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    for keys, value in BRINGUP_PLACEHOLDERS.items():
        node = out
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = copy.deepcopy(value)
    return out


def placeholder_text(text: str) -> str:
    """configs/real.yaml text with the bring-up values reset (via yamlio; unchanged if shipped)."""
    for keys, value in BRINGUP_PLACEHOLDERS.items():
        node: Any = yaml.safe_load(text)
        for k in keys:
            node = node[k]
        if node != value:
            text = set_yaml_value(text, keys, value)
    return text


def load_config_placeholders(preset: str, config: str | Path | None = None) -> dict[str, Any]:
    """``src.config.load_config`` with the real preset's bring-up values reset."""
    cfg = load_config(preset, config)
    return placeholder_config(cfg) if preset == "real" and config is None else cfg
