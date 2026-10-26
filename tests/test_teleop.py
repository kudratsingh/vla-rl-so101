"""Teleop session logic (scripts/wo06_teleop_record.py) with a fake leader, follower and camera."""

from __future__ import annotations

import importlib.util
import time
from typing import Any

import numpy as np
import pytest
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.robot.camera import CameraFault, Frame
from src.robot.driver import DeadmanTripped, So101Driver
from src.robot.safety import make_safety
from src.robot.units import JOINT_NAMES, UnitMap

_spec = importlib.util.spec_from_file_location(
    "wo06_teleop_record", REPO_ROOT / "scripts" / "wo06_teleop_record.py"
)
assert _spec is not None and _spec.loader is not None
teleop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teleop)


class FakeLeader:
    def __init__(self, wrist_flex: float) -> None:
        self.pose = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
        self.pose.update({"wrist_flex.pos": wrist_flex, "gripper.pos": 50.0})
        self.fail = False

    def get_action(self) -> dict[str, float]:
        if self.fail:
            raise ConnectionError("leader unplugged")
        return dict(self.pose)


class FakeCam:
    def __init__(self) -> None:
        self.fail = False

    def read(self) -> Frame:
        if self.fail:
            raise CameraFault("timed out")
        return Frame(np.zeros((480, 640, 3), np.uint8), time.perf_counter())


def _session(leader: Any, cam: Any) -> tuple[Any, FakeFollower]:
    fol = FakeFollower()
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
    arm.connect()
    return teleop.Session(arm, leader, cam, UnitMap()), fol


def test_follower_eases_onto_leader_through_safety() -> None:
    sess, fol = _session(FakeLeader(wrist_flex=12.0), FakeCam())
    assert sess.sync_to_leader()
    steps = np.diff([0.0, *(a["wrist_flex.pos"] for a in fol.sent)])
    assert len(fol.sent) >= 3 and steps.max() <= 4.0 + 1e-9  # rate-limited, never a jump
    assert fol.pos["wrist_flex.pos"] == pytest.approx(12.0)
    assert teleop.features(224)[teleop.IMAGE_KEY]["shape"] == (224, 224, 3)


def test_lost_camera_or_leader_holds_the_follower() -> None:
    cam = FakeCam()
    sess, fol = _session(FakeLeader(wrist_flex=2.0), cam)
    sess.tick()
    cam.fail = True
    with pytest.raises(DeadmanTripped, match="no camera frame"):
        sess.tick()
    assert fol.bus.torque_enabled  # hold, not limp
    leader = FakeLeader(wrist_flex=2.0)
    sess, fol = _session(leader, FakeCam())
    sess.tick()
    leader.fail = True
    with pytest.raises(DeadmanTripped, match="leader read failed"):
        sess.tick()
    assert fol.sent[-1] == pytest.approx(fol.pos)  # last write = hold at the measured pose
