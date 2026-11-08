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


def test_sync_targets_the_safety_clipped_leader_pose(capsys: pytest.CaptureFixture[str]) -> None:
    leader = FakeLeader(wrist_flex=94.0)  # past the follower's 90 deg safety limit
    leader.pose["gripper.pos"] = 0.0  # trigger fully closed: -10 deg, safety floor -5 deg
    fol = FakeFollower(start={"wrist_flex.pos": 85.0})
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
    arm.connect()
    sess = teleop.Session(arm, leader, FakeCam(), UnitMap())
    assert sess.sync_to_leader(timeout_s=3.0)
    assert fol.pos["wrist_flex.pos"] == pytest.approx(90.0, abs=0.01)
    assert UnitMap().gripper_pct_to_rad(fol.pos["gripper.pos"]) == pytest.approx(
        np.radians(-5.0), abs=1e-4
    )

    out_of_box = FakeLeader(wrist_flex=0.0)
    out_of_box.pose["shoulder_lift.pos"] = 60.0  # toy_fk: TCP below the table
    sess, fol = _session(out_of_box, FakeCam())
    assert not sess.sync_to_leader(timeout_s=0.3)
    printed = capsys.readouterr().out
    assert "workspace_box_m" in printed and "shoulder_lift" in printed
    assert all(a["shoulder_lift.pos"] == pytest.approx(0.0) for a in fol.sent)  # held


class FakeDataset:
    def __init__(self) -> None:
        self.episode_buffer = {"size": 0}
        self.saved = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        assert frame["task"] == "push the red cube to the target"  # protocol string
        self.episode_buffer["size"] += 1

    def save_episode(self) -> None:
        self.saved += 1
        self.episode_buffer["size"] = 0

    def clear_episode_buffer(self) -> None:
        self.episode_buffer["size"] = 0


def test_redo_then_ctrl_c_at_a_prompt_ends_through_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    # ENTER (record), "n" (redo), Ctrl-C at the next scene prompt, ENTER once the arm is supported
    answers: list[Any] = ["", "n", KeyboardInterrupt, ""]
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        a = answers.pop(0)
        if a is KeyboardInterrupt:
            raise KeyboardInterrupt
        return str(a)

    monkeypatch.setattr("builtins.input", fake_input)
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    data = FakeDataset()
    instr = teleop.protocol_instructions(["Push"])
    saved = teleop.record_session(sess, data, ["Push"], instr, 1, 0.2, lambda im: im)
    assert saved == 0 and data.saved == 0 and data.episode_buffer["size"] == 0 and not answers
    assert "[n]o, redo / [q]uit" in prompts[1] and "torque goes OFF" in prompts[3]
    assert not sess.arm.deadman.tripped and fol.bus.torque_enabled  # holding, not dropped
    assert fol.sent[-1] == pytest.approx(fol.pos)  # last write = hold at the measured pose
    sess.arm.__exit__(None, None, None)  # the normal end of the driver's `with` block
    assert not fol.bus.torque_enabled
