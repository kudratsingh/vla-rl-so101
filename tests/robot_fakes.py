"""Fakes for the robot-side tests. No serial port, no camera, no lerobot hardware access."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.config import load_config
from src.robot.units import JOINT_NAMES


def toy_fk(q: np.ndarray) -> np.ndarray:
    """TEST ONLY - NOT the SO-101 kinematics (that is src/env/kinematics.py from WO-01).

    Pan + a planar 2-link arm: at q = 0 the TCP is at (0.26, 0, 0.10) m; positive shoulder_lift
    or elbow_flex lowers it, so a large lift drives the TCP below the table (z < 0).
    """
    pan, lift, elbow = float(q[0]), float(q[1]), float(q[2])
    l1, l2, base_h = 0.12, 0.14, 0.10
    r = l1 * math.cos(lift) + l2 * math.cos(lift + elbow)
    z = base_h - l1 * math.sin(lift) - l2 * math.sin(lift + elbow)
    return np.array([r * math.cos(pan), r * math.sin(pan), z])


def real_cfg() -> dict[str, Any]:
    return load_config("real")


class FakeBus:
    """Records torque/goal traffic like lerobot's FeetechMotorsBus (raw ticks = 2048 + deg*4095/360)."""

    def __init__(self, follower: FakeFollower):
        self.follower = follower
        self.torque_enabled = False
        self.calls: list[tuple[str, Any]] = []
        self.fail_disable = False

    def sync_read(self, data_name: str, motors: Any = None, *, normalize: bool = True) -> Any:
        self.calls.append(("sync_read", data_name))
        return {m: 2048 + int(v * 4095 / 360) for m, v in self.follower.positions_lerobot().items()}

    def sync_write(self, data_name: str, values: Any, *, normalize: bool = True) -> None:
        self.calls.append(("sync_write", (data_name, dict(values))))

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.calls.append(("enable_torque", None))
        self.torque_enabled = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))
        self.follower.connected = False

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self.calls.append(("disable_torque", None))
        if self.fail_disable:
            raise ConnectionError("bus gone")
        self.torque_enabled = False


class FakeFollower:
    """Stands in for lerobot's SO101Follower. Positions in LeRobot units (deg, gripper 0-100).

    ``track``: the arm reaches every goal instantly (perfect servo).
    """

    def __init__(self, start: dict[str, float] | None = None, track: bool = True):
        self.pos = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
        self.pos["gripper.pos"] = 50.0
        if start:
            self.pos.update(start)
        self.track = track
        self.bus = FakeBus(self)
        self.calibration = {"shoulder_pan": object()}
        self._calibrated = True
        self.connected = False
        self.sent: list[dict[str, float]] = []
        self.fail_read = False
        self.fail_write = False
        self.fail_connect = False

    def positions_lerobot(self) -> dict[str, float]:
        return {k.removesuffix(".pos"): v for k, v in self.pos.items()}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        assert calibrate is False, "the driver must never auto-calibrate"
        self.connected = True
        self.bus.disable_torque()  # like SOFollower.configure(): torque off ... torque on
        if self.fail_connect:
            raise ConnectionError("Failed to write 'P_Coefficient'")
        self.bus.enable_torque()

    def disconnect(self) -> None:
        self.bus.disable_torque(num_retry=5)  # disable_torque_on_disconnect=True
        self.connected = False

    def get_observation(self) -> dict[str, Any]:
        if self.fail_read:
            raise ConnectionError("Failed to sync read 'Present_Position'")
        return dict(self.pos)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.fail_write:
            raise ConnectionError("Failed to sync write 'Goal_Position'")
        self.sent.append(dict(action))
        if self.track:
            self.pos.update(action)
        return dict(action)
