"""scripts/wo06_camera_match.py: the sim side poses the arm at given joints on the CPU sim (Mac)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.config import REPO_ROOT, load_config
from src.robot.units import JOINT_NAMES

_spec = importlib.util.spec_from_file_location(
    "wo06_camera_match", REPO_ROOT / "scripts" / "wo06_camera_match.py"
)
assert _spec is not None and _spec.loader is not None
match = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match)


@pytest.mark.needs_render
def test_with_arm_poses_the_cpu_sim_arm() -> None:
    sim_cfg = load_config("sim")
    sim_cfg["device"] = "cpu"  # what the script forces: make_env returns a SerialVecEnv
    rest = match.sim_frame(sim_cfg, "PickPlace", 0, "seen", None)
    posed = match.sim_frame(
        sim_cfg, "PickPlace", 0, "seen", np.array([0.5, 0.3, -0.3, 0.2, 0.4, 1.0])
    )
    assert rest is not None and posed is not None and rest.shape == posed.shape
    assert np.abs(rest.astype(int) - posed.astype(int)).mean() > 1.0  # the arm moved in the image


class ReadBus:
    """Read-only bus for --with-arm: raw degrees per motor, Torque_Enable per motor."""

    is_calibrated = True

    def __init__(self, torque: int) -> None:
        self.torque = torque
        self.calls: list[str] = []

    def connect(self) -> None:
        self.calls.append("connect")

    def sync_read(self, data_name: str) -> dict[str, float]:
        self.calls.append(data_name)
        if data_name == "Torque_Enable":
            return {m: float(self.torque) for m in JOINT_NAMES}
        return {m: 50.0 if m == "gripper" else 0.0 for m in JOINT_NAMES}

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(f"disconnect(disable_torque={disable_torque})")


def test_with_arm_refuses_a_torqued_follower() -> None:
    from robot_fakes import real_cfg

    cfg = real_cfg()
    bus = ReadBus(torque=1)
    with pytest.raises(SystemExit, match="torque is ON"):
        match.read_arm_joints(cfg, bus=bus)
    assert bus.calls == ["connect", "Torque_Enable", "disconnect(disable_torque=False)"]
    q = match.read_arm_joints(cfg, bus=ReadBus(torque=0))
    assert q.shape == (6,) and np.allclose(q[:5], 0.0)
