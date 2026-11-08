"""Teleop session logic (scripts/wo06_teleop_record.py) with a fake leader, follower and camera."""

from __future__ import annotations

import importlib.util
import math
import signal
import time
from typing import Any

import numpy as np
import pytest
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.robot.camera import CameraFault, Frame
from src.robot.driver import DeadmanTripped, ParkSettings, ServoFault, So101Driver
from src.robot.safety import make_safety
from src.robot.servo_health import monitor_from_config
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

    closed = FakeLeader(wrist_flex=0.0)  # gripper resting closed, below the floor: held there
    closed.pose["gripper.pos"] = 0.0
    fol = FakeFollower(start={"gripper.pos": 0.0})
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
    arm.connect()
    assert teleop.Session(arm, closed, FakeCam(), UnitMap()).sync_to_leader(timeout_s=1.0)
    assert fol.pos["gripper.pos"] == pytest.approx(0.0, abs=1e-6)

    out_of_box = FakeLeader(wrist_flex=0.0)
    out_of_box.pose["shoulder_lift.pos"] = 60.0  # toy_fk: TCP below the table
    sess, fol = _session(out_of_box, FakeCam())
    assert not sess.sync_to_leader(timeout_s=0.3)
    printed = capsys.readouterr().out
    assert "workspace_box_m" in printed and "shoulder_lift" in printed
    assert all(a["shoulder_lift.pos"] == pytest.approx(0.0) for a in fol.sent)  # held


class FakeDataset:
    def __init__(self, save_fails: bool = False, ctrl_c_at_frame: int = 0) -> None:
        self.episode_buffer = {"size": 0}
        self.saved = 0
        self.save_fails = save_fails
        self.ctrl_c_at_frame = ctrl_c_at_frame

    def add_frame(self, frame: dict[str, Any]) -> None:
        assert frame["task"] == "push the red cube to the target"  # protocol string
        if self.episode_buffer["size"] + 1 == self.ctrl_c_at_frame:
            signal.raise_signal(signal.SIGINT)  # Ctrl-C halfway through writing a frame
        self.episode_buffer["size"] += 1

    def save_episode(self) -> None:
        signal.raise_signal(signal.SIGINT)  # a "did it hang?" Ctrl-C while the video encodes
        if self.save_fails:
            raise ValueError("Column 2 named timestamp expected length 5 but got length 6")
        self.saved += 1
        self.episode_buffer["size"] = 0

    def clear_episode_buffer(self) -> None:
        self.episode_buffer["size"] = 0


def test_redo_then_ctrl_c_at_a_prompt_ends_through_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    # ENTER (record), "n" (redo), Ctrl-C at the next scene prompt, a panicked Ctrl-C at the
    # support prompt (ignored), then ENTER once the arm is supported
    answers: list[Any] = ["", "n", KeyboardInterrupt, KeyboardInterrupt, ""]
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
    assert "[n]o, redo / [q]uit" in prompts[1]
    assert "torque goes OFF" in prompts[3] and prompts[4] == prompts[3]  # asked again
    assert not sess.arm.deadman.tripped and fol.bus.torque_enabled  # holding, not dropped
    assert fol.sent[-1] == pytest.approx(fol.pos)  # last write = hold at the measured pose
    sess.arm.__exit__(None, None, None)  # the normal end of the driver's `with` block
    assert not fol.bus.torque_enabled


class CtrlCLeader(FakeLeader):
    """Sends SIGINT from inside the leader read on the given calls: Ctrl-C mid-tick."""

    def __init__(self, at: set[int]) -> None:
        super().__init__(wrist_flex=20.0)  # several rate-limited sync ticks away
        self.at, self.calls = at, 0

    def get_action(self) -> dict[str, float]:
        self.calls += 1
        if self.calls in self.at:
            signal.raise_signal(signal.SIGINT)
        return super().get_action()


def _scripted_input(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> list[str]:
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return answers.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_ctrl_c_mid_tick_ends_sync_or_episode_never_a_bus_call(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Ctrl-C inside the first sync tick's leader read -> sync aborted, retry; inside the 3rd
    # add_frame -> that frame completes and the episode ends; a Ctrl-C while it saves is ignored;
    # then q and ENTER at the support prompt
    answers = ["", "", "y", "q", ""]
    prompts = _scripted_input(monkeypatch, answers)
    sess, fol = _session(CtrlCLeader(at={1}), FakeCam())
    data = FakeDataset(ctrl_c_at_frame=3)
    instr = teleop.protocol_instructions(["Push"])
    saved = teleop.record_session(sess, data, ["Push"], instr, 2, 5.0, lambda im: im)
    out = capsys.readouterr().out
    assert "sync aborted" in out and "episode ended" in out and not answers
    assert "(3 frames)" in prompts[2] and saved == 1 and data.saved == 1
    assert "torque goes OFF" in prompts[-1]
    assert not sess.arm.deadman.tripped and fol.bus.torque_enabled  # holding, not dropped
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler  # restored


def test_failed_save_discards_the_episode_and_keeps_holding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    answers = ["", "y", "q", ""]  # record, keep -> save raises, end at the next scene prompt
    _scripted_input(monkeypatch, answers)
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    data = FakeDataset(save_fails=True)
    instr = teleop.protocol_instructions(["Push"])
    saved = teleop.record_session(sess, data, ["Push"], instr, 1, 0.2, lambda im: im)
    assert "saving failed (ValueError" in capsys.readouterr().out and not answers
    assert saved == 0 and data.episode_buffer["size"] == 0
    assert fol.bus.torque_enabled and fol.sent[-1] == pytest.approx(fol.pos)


def test_healthy_end_parks_the_follower_then_torque_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = _scripted_input(monkeypatch, ["q"])  # q at the first scene prompt: healthy end
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    park = (0.0, 0.0, 0.0, math.radians(10.0), 0.0, math.radians(5.0))
    sess.park = ParkSettings(qpos=park, tol_rad=math.radians(1.0), timeout_s=5.0)
    instr = teleop.protocol_instructions(["Push"])
    saved = teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2, lambda im: im)
    assert saved == 0 and len(prompts) == 1  # no "support the arm" prompt: it parked
    assert "follower parked; torque OFF" in capsys.readouterr().out
    assert fol.sent[-1]["wrist_flex.pos"] == pytest.approx(10.0)
    steps = np.diff([0.0, *(a["wrist_flex.pos"] for a in fol.sent)])
    assert max(steps) <= 4.0 + 1e-9  # through the safety filter's velocity cap
    assert not fol.bus.torque_enabled and not sess.arm.deadman.tripped


def test_hot_servo_after_a_prompt_cuts_torque_before_any_motion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _scripted_input(monkeypatch, [""])
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    sess.arm.monitor = monitor_from_config(real_cfg())
    fol.temperature_c["gripper"] = 61.0  # heated up while holding at the scene prompt
    instr = teleop.protocol_instructions(["Push"])
    with pytest.raises(ServoFault):
        teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2, lambda im: im)
    assert "SUPPORT THE ARM NOW" in capsys.readouterr().out
    assert not fol.sent and not fol.bus.torque_enabled  # nothing moved, torque off at once


def test_defaults_no_auto_park_and_a_bounded_torque_on_wait() -> None:
    from src.robot.driver import park_from_config

    cfg = real_cfg()
    assert park_from_config(cfg) is None  # robot.park.enabled: false until the owner verifies
    tw = teleop.TorqueWait.from_config(cfg, "teleop", allow_none=True)
    assert tw == teleop.TorqueWait(60.0, 10.0)
    cfg["teleop"]["prompt_torque_timeout_s"] = None
    unbounded = teleop.TorqueWait.from_config(cfg, "teleop", allow_none=True)  # teleop only
    assert unbounded == teleop.TorqueWait(None, 10.0)  # no bound; the grace period stays


def _health_reads(fol: FakeFollower) -> int:
    return sum(
        c == ("sync_read", r)
        for c in fol.bus.calls
        for r in ("Present_Temperature", "Present_Current")
    )


def test_prompt_timeout_under_torque_supports_then_torque_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    sess.arm.monitor = monitor_from_config(real_cfg())
    sess.guard.wait = teleop.TorqueWait(timeout_s=60.0, grace_s=10.0)
    waits: list[tuple[str, float, int]] = []

    def no_answer(prompt: str, timeout_s: float) -> str | None:
        waits.append((prompt, timeout_s, _health_reads(fol)))
        return None  # nobody at the keyboard

    sess.guard.read_line = no_answer
    instr = teleop.protocol_instructions(["Push"])
    saved = teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2, lambda im: im)
    assert saved == 0
    # the scene prompt, then the support grace
    assert [w[1] for w in waits] == [60.0, pytest.approx(10.0, abs=0.5)]
    assert waits[0][2] == 2  # temperature + current read BEFORE the wait, by this thread
    out = capsys.readouterr().out
    assert "SUPPORT THE ARM NOW" in out and "Torque goes OFF in 10 s" in out
    assert not fol.sent and not fol.bus.torque_enabled  # nothing moved; torque released
    assert sess.torque_released and sess.arm.deadman.response is not None
    assert sess.session_end == "released_unconfirmed"  # never "supported"


def test_answered_prompt_reads_servos_before_and_after_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    sess.arm.monitor = monitor_from_config(real_cfg())
    sess.guard.wait = teleop.TorqueWait(timeout_s=60.0, grace_s=0.0)
    before: list[int] = []

    def answer(prompt: str, timeout_s: float) -> str | None:
        before.append(_health_reads(fol))
        fol.temperature_c["elbow_flex"] = 58.0  # heated up while the operator thought
        return ""

    sess.guard.read_line = answer
    with pytest.raises(ServoFault):
        teleop.torque_on_ask(sess, "ENTER: ")
    assert before == [2] and _health_reads(fol) == 3  # one read after: the trip
    assert not fol.bus.torque_enabled


def test_end_session_after_a_failed_torque_off_names_the_power_switch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    fol.bus.fail_disable = True
    assert sess.arm.torque_off() is False
    n = len(fol.sent)
    teleop.end_session(sess.arm, sess)
    assert "inline power switch" in capsys.readouterr().out and len(fol.sent) == n


class BrokenDataset(FakeDataset):
    def add_frame(self, frame: dict[str, Any]) -> None:
        raise ValueError("feature 'observation.state' has dtype float64, expected float32")

    def clear_episode_buffer(self) -> None:
        raise KeyError("size")  # discard_episode falls back to a fresh buffer

    def create_episode_buffer(self) -> dict[str, int]:
        return {"size": 0}


def test_unexpected_error_holds_and_asks_for_support_before_torque_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression (PR 29 review): a non-deadman exception used to leave the driver's `with`
    block at once: torque off, the raised follower dropped with no support prompt."""
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    at_prompt: list[tuple[str, bool, int]] = []

    def fake_input(prompt: str = "") -> str:
        at_prompt.append((prompt, fol.bus.torque_enabled, len(fol.sent)))
        return ""  # ENTER at the scene prompt, then at the support prompt

    monkeypatch.setattr("builtins.input", fake_input)
    instr = teleop.protocol_instructions(["Push"])
    with pytest.raises(ValueError, match="dtype float64"):
        try:
            teleop.record_session(sess, BrokenDataset(), ["Push"], instr, 1, 0.2, lambda im: im)
        except ValueError as e:
            sess.arm.__exit__(type(e), e, None)  # what the driver's `with` block then does
            raise
    out = capsys.readouterr().out
    assert "STOPPED: unexpected ValueError" in out and "episode discarded" in out
    assert len(at_prompt) == 2 and at_prompt[1][0] == teleop.SUPPORT_PROMPT
    assert at_prompt[1][1] is True  # torque still on at the support prompt: the arm is held
    assert fol.sent[at_prompt[1][2] - 1] == pytest.approx(fol.pos)  # a hold write came first
    assert (
        sess.arm.deadman.reason is not None
        and "teleop error: ValueError" in sess.arm.deadman.reason
    )
    assert not fol.bus.torque_enabled  # torque off only after the prompt


@pytest.mark.parametrize(
    ("script", "end", "torque_on_after"),
    [
        ([""], "supported", True),  # torque goes off when the driver's `with` block closes
        ([KeyboardInterrupt, ""], "supported", True),  # Ctrl-C ignored, then ENTER
        ([KeyboardInterrupt, None], "released_unconfirmed", False),  # Ctrl-C never confirms
        ([EOFError], "released_unconfirmed", False),  # stdin closed: grace, torque off
        ([None], "released_unconfirmed", False),
    ],
)
def test_end_session_claims_support_only_after_enter(
    capsys: pytest.CaptureFixture[str], script: list[Any], end: str, torque_on_after: bool
) -> None:
    from robot_fakes import FakeClock, FakeReader

    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    clock = FakeClock()
    reader = FakeReader(script, clock, fol.bus)
    sess.guard = teleop.TorqueGuard(
        sess.arm,
        teleop.TorqueWait(timeout_s=60.0, grace_s=10.0),
        read=reader,
        clock=clock,
        sleep=clock.sleep,
    )
    t0 = clock()
    assert teleop.end_session(sess.arm, sess) == end and sess.session_end == end
    assert fol.bus.torque_enabled is torque_on_after
    assert all(c[2] for c in reader.calls[: len(script)])  # the arm held during every wait
    out = capsys.readouterr().out
    if EOFError in script:
        assert "STDIN CLOSED" in out and clock() - t0 == pytest.approx(10.0)
    if end == "released_unconfirmed":
        assert sess.torque_released and "support was NOT confirmed" in out
