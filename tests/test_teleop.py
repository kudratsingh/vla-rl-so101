"""Teleop session logic (scripts/wo06_teleop_record.py) with a fake leader, follower and camera."""

from __future__ import annotations

import importlib.util
import math
import os
import signal
import time
from collections.abc import Callable
from typing import Any, Self

import numpy as np
import pytest
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.robot.camera import CameraFault, RigFrame
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
    """Both convention cameras (front + wrist), read together like ``CameraRig``."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.fail = False
        self.clock = clock  # the frame stamps: the driver's clock

    def read(self) -> RigFrame:
        if self.fail:
            raise CameraFault("wrist camera: timed out")
        t, img = self.clock(), np.zeros((480, 640, 3), np.uint8)
        return RigFrame({"front": img, "wrist": img}, {"front": t, "wrist": t})


def _input_reader(prompt: str, _timeout_s: float | None) -> str | None:
    """The operator, faked per test through ``builtins.input``: the real reader
    (``timed_input``) reads fd 0 directly, also with no timeout (review F1)."""
    return input(prompt)


def _session(leader: Any, cam: Any) -> tuple[Any, FakeFollower]:
    fol = FakeFollower()
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
    arm.connect()
    return teleop.Session(arm, leader, cam, UnitMap(), read=_input_reader), fol


def test_follower_eases_onto_leader_through_safety() -> None:
    sess, fol = _session(FakeLeader(wrist_flex=12.0), FakeCam())
    assert sess.sync_to_leader()
    steps = np.diff([0.0, *(a["wrist_flex.pos"] for a in fol.sent)])
    assert len(fol.sent) >= 3 and steps.max() <= 4.0 + 1e-9  # rate-limited, never a jump
    assert fol.pos["wrist_flex.pos"] == pytest.approx(12.0)
    feats = teleop.features()
    assert feats["observation.images.front"]["shape"] == (480, 640, 3)
    assert feats["observation.images.wrist"]["shape"] == (480, 640, 3)


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
    saved = teleop.record_session(sess, data, ["Push"], instr, 1, 0.2)
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
    saved = teleop.record_session(sess, data, ["Push"], instr, 2, 5.0)
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
    saved = teleop.record_session(sess, data, ["Push"], instr, 1, 0.2)
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
    saved = teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2)
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
        teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2)
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
    saved = teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 0.2)
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
            teleop.record_session(sess, BrokenDataset(), ["Push"], instr, 1, 0.2)
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


def test_teleop_refuses_a_non_tty_stdin_before_the_leader_or_the_follower(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import sys

    opened: list[str] = []

    def no(what: str) -> Any:
        def f(*_a: Any, **_k: Any) -> Any:
            opened.append(what)
            raise AssertionError(f"{what} was reached")

        return f

    monkeypatch.setattr(teleop, "load_config", no("config"))
    monkeypatch.setattr(teleop, "make_leader", no("leader"))
    monkeypatch.setattr(teleop.So101Driver, "from_config", no("follower"))
    monkeypatch.setattr(sys, "argv", ["wo06_teleop_record.py", "--preset", "real"])
    from robot_fakes import tty_stdout

    r, w = os.pipe()
    piped = open(r, "r")  # noqa: SIM115 - closed below
    monkeypatch.setattr(sys, "stdin", piped)
    try:
        with tty_stdout() as tty_out:
            # a terminal on stdout (review T29-R1: pytest's capture never is): only the piped
            # stdin can refuse here
            monkeypatch.setattr(sys, "stdout", tty_out)
            with pytest.raises(SystemExit, match="stdin is not a tty(.|\n)*nothing was opened"):
                teleop.main()
            monkeypatch.undo()
    finally:
        piped.close()
        os.close(w)
    assert opened == []


def test_teleop_ctrl_z_mid_episode_continues_and_a_lost_foreground_ends_like_a_hangup(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Job control in teleop (owner 2026-11-08). A Ctrl-Z (SIGTSTP) inside an episode leaves
    the recording running; the terminal's foreground lost mid-episode (``Session.tick`` polls
    it; faked here) ends the episode like a hang-up: every later prompt is ``lost_foreground``,
    the support is never confirmed, torque goes off after the grace (``released_unconfirmed``)."""
    from robot_fakes import FakeClock, FakeReader

    from src.robot import prompts as P

    lost = [False]
    monkeypatch.setattr(P, "not_foreground", lambda _fd: "pgrp 1 != 2" if lost[0] else None)

    class JobControlDataset(FakeDataset):
        frames = 0

        def add_frame(self, frame: dict[str, Any]) -> None:
            super().add_frame(frame)
            n = self.frames = self.episode_buffer["size"]
            if n == 2:
                assert callable(signal.getsignal(signal.SIGTSTP))  # else this would stop pytest
                signal.raise_signal(signal.SIGTSTP)  # Ctrl-Z mid-episode: ignored
            if n == 5:
                lost[0] = True

    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    clock = FakeClock()
    reader = FakeReader([""], clock, fol.bus)  # ENTER at the scene prompt, nothing after

    def read(prompt: str, timeout_s: float | None) -> str | None:
        with P._reading():  # the real check timed_input makes before every read
            return reader(prompt, timeout_s)

    sess.guard = teleop.TorqueGuard(
        sess.arm, teleop.TorqueWait(60.0, 10.0), read=read, clock=clock, sleep=clock.sleep
    )
    data = JobControlDataset()
    instr = teleop.protocol_instructions(["Push"])
    with P.job_control():
        P.INTERRUPTS.tty_fd = 0  # as with stdin on the terminal (teleop.main's job_control)
        saved = teleop.record_session(sess, data, ["Push"], instr, 1, 30.0)
    out = capsys.readouterr().out
    assert saved == 0 and data.saved == 0  # the unsaved episode was discarded
    assert data.frames == 5  # the Ctrl-Z did not stop it; the loss stopped the next tick
    assert "episode ended" in out and "LOST THE TERMINAL'S FOREGROUND" in out
    assert sess.session_end == "released_unconfirmed" and sess.guard.released is not None
    assert sess.guard.released["cause"] == "lost_foreground"
    assert not fol.bus.torque_enabled and len(reader.calls) == 1  # only the scene prompt read


def test_teleop_park_stops_at_once_after_a_lost_foreground(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review F4: with ``robot.park.enabled``, a session that lost the terminal's foreground
    never parks (moves) at its end: the park's stop check sees the loss before the first step,
    as in the pilot and the real eval; the support prompt then ends as ``lost_foreground``."""
    from robot_fakes import FakeClock, FakeReader

    from src.robot import prompts as P

    monkeypatch.setattr(P, "not_foreground", lambda _fd: "pgrp 1 != 2")
    sess, fol = _session(FakeLeader(wrist_flex=2.0), FakeCam())
    park = (0.0, 0.0, 0.0, math.radians(10.0), 0.0, math.radians(5.0))
    sess.park = ParkSettings(qpos=park, tol_rad=math.radians(1.0), timeout_s=5.0)
    clock = FakeClock()
    reader = FakeReader([], clock, fol.bus)

    def read(prompt: str, timeout_s: float | None) -> str | None:
        with P._reading():  # the real check timed_input makes before every read
            return reader(prompt, timeout_s)

    sess.guard = teleop.TorqueGuard(
        sess.arm, teleop.TorqueWait(60.0, 10.0), read=read, clock=clock, sleep=clock.sleep
    )
    with P.job_control():
        P.INTERRUPTS.tty_fd = 0  # as with stdin on the terminal (teleop.main's job_control)
        end = teleop.end_session(sess.arm, sess)
    out = capsys.readouterr().out
    assert end == "released_unconfirmed" and "follower parked" not in out
    assert sess.guard.released is not None and sess.guard.released["cause"] == "lost_foreground"
    assert all(a["wrist_flex.pos"] == pytest.approx(0.0, abs=1e-6) for a in fol.sent)  # no park
    assert not fol.bus.torque_enabled


def test_teleop_main_ignores_ctrl_z_from_torque_on_until_its_record_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Review F4: teleop's own ``job_control`` in ``main`` covers the follower's torque enable,
    the session, the TORQUE OFF UNCONFIRMED check and the provenance write after it; every
    session handler is restored when ``main`` returns."""
    import sys

    from src.robot import prompts as P

    seen: dict[str, bool] = {}

    def in_session() -> bool:
        return P.INTERRUPTS.job_control and callable(signal.getsignal(signal.SIGTSTP))

    class MainFollower(FakeFollower):
        def connect(self, calibrate: bool = True) -> None:
            seen["torque_on"] = in_session()
            super().connect(calibrate)

    class Leader(FakeLeader):
        is_calibrated, is_connected = True, False

        def connect(self, calibrate: bool = False) -> None:
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    class Dataset(FakeDataset):
        num_episodes = 0

        def finalize(self) -> None:
            pass

    class Cam(FakeCam):
        @classmethod
        def from_config(cls, _cfg: Any, _names: Any) -> Cam:
            return cls()

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def record_session(sess: Any, *_a: Any) -> int:
        seen["session"] = in_session()
        sess.session_end = "supported"
        return 0

    real_unconfirmed = teleop._torque_off_unconfirmed

    def torque_off_unconfirmed(arm: Any) -> bool:
        seen.setdefault("record", in_session())  # the finally that writes the provenance
        return bool(real_unconfirmed(arm))

    cfg = real_cfg()
    cfg["teleop"] = {**cfg["teleop"], "leader": {"type": "so101_leader"}}
    monkeypatch.setattr(teleop, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(teleop, "require_terminal", lambda _what: None)
    monkeypatch.setattr(teleop, "load_config", lambda *_a: cfg)
    monkeypatch.setattr(teleop, "sim_mismatches", lambda *_a: [])
    monkeypatch.setattr(teleop, "load_sim_fk", lambda: toy_fk)
    monkeypatch.setattr(teleop, "make_leader", lambda _lc: Leader(wrist_flex=2.0))
    monkeypatch.setattr(teleop, "CameraRig", Cam)
    monkeypatch.setattr(teleop, "confirm_camera_roles", lambda *_a, **_k: {"method": "test"})
    monkeypatch.setattr(teleop, "record_session", record_session)
    monkeypatch.setattr(teleop, "_torque_off_unconfirmed", torque_off_unconfirmed)
    monkeypatch.setattr(
        teleop.So101Driver,
        "from_config",
        classmethod(lambda cls, cfg, fk: cls(MainFollower(), UnitMap(), make_safety(cfg, toy_fk))),
    )
    import lerobot.datasets.lerobot_dataset as lds

    monkeypatch.setattr(lds.LeRobotDataset, "create", staticmethod(lambda **_k: Dataset()))
    monkeypatch.setattr(
        sys, "argv", ["wo06_teleop_record.py", "--preset", "real", "--task", "Push"]
    )
    names = ("SIGINT", "SIGHUP", "SIGTERM", "SIGTSTP", "SIGTTIN", "SIGTTOU")
    before = {n: signal.getsignal(getattr(signal, n)) for n in names}
    teleop.main()
    assert seen == {"torque_on": True, "session": True, "record": True}
    assert {n: signal.getsignal(getattr(signal, n)) for n in names} == before
    assert not P.INTERRUPTS.job_control


# ------------------------------------------------------------------------------------------------
# Owner decision 2026-11-08: no teleop park after a hang-up (SIGHUP / SIGTERM) or a lost
# foreground; one during the park stops it before its next goal write. Parking enabled, a fake
# arm that logs every goal write and every park stop check in order, real signals (os.kill on
# this process) through the real operator_signals handlers, the real end_session, _park,
# So101Driver.park / move_to, TorqueGuard and timed_input. Only the grace period's clock is fake.
PARK_WRIST_DEG = 30.0  # 8 rate-limited steps of <= 4 deg from 0


class _HardTimeout(Exception):
    pass


@pytest.fixture
def hard_timeout() -> Any:
    """Fail (never hang) if the test runs past 30 s: SIGALRM, which no session code touches."""

    def fire(_signum: int, _frame: Any) -> None:
        raise _HardTimeout("test exceeded its 30 s hard timeout")

    prev = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, 30.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


class ParkFollower(FakeFollower):
    """Logs every goal write in order as ("motion", n) when it moves a joint or ("hold", n) when
    it re-writes the present pose (Goal := measured), then runs ``hook(self)``."""

    def __init__(self) -> None:
        super().__init__()
        self.log: list[tuple[str, Any]] = []
        self.hook: Callable[[ParkFollower], None] | None = None
        self.read_hook: Callable[[ParkFollower], None] | None = None  # in get_observation

    def get_observation(self) -> dict[str, Any]:
        if self.read_hook is not None:
            self.read_hook(self)
        return super().get_observation()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        moved = any(abs(float(v) - self.pos[k]) > 1e-9 for k, v in action.items())
        out = super().send_action(action)
        self.log.append(("motion" if moved else "hold", len(self.sent)))
        if self.hook is not None:
            self.hook(self)
        return out

    def motion_writes(self) -> int:
        return sum(kind == "motion" for kind, _ in self.log)


def _park_session(
    monkeypatch: pytest.MonkeyPatch, foreground: bool, fake_arm_clock: bool = True
) -> Any:
    """A teleop Session with parking enabled on ParkFollower; its stop checks are logged into
    the follower's log as ("stop_check", result). ``foreground``: stdin is watched as the
    session's terminal and ``tcgetpgrp`` keeps naming this process group (``fg["pgrp"]``: set
    it to another group to take the foreground away); else no terminal is watched.
    ``fake_arm_clock``: the driver and the camera stamps share a FakeClock (the park's ticks are
    exactly 1 / control_hz apart, so each rate-limited step is exactly 4 deg however loaded the
    machine is); False for the teleop loops, which pace themselves on ``time.perf_counter``."""
    from robot_fakes import FakeClock

    from src.robot import prompts as P

    monkeypatch.setattr(P, "_detach_stdio", lambda: None)  # keep pytest's capture
    fol = ParkFollower()
    arm_clock = FakeClock() if fake_arm_clock else None
    if arm_clock is not None:
        arm = So101Driver(
            fol, UnitMap(), make_safety(real_cfg(), toy_fk), arm_clock, sleep=arm_clock.sleep
        )
        cam = FakeCam(arm_clock)
    else:
        arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
        cam = FakeCam()
    arm.connect()
    sess = teleop.Session(arm, FakeLeader(wrist_flex=0.0), cam, UnitMap())
    q = [0.0, 0.0, 0.0, math.radians(PARK_WRIST_DEG), 0.0, UnitMap().gripper_pct_to_rad(50.0)]
    sess.park = ParkSettings(qpos=tuple(q), tol_rad=math.radians(1.0), timeout_s=5.0)
    sess.clock = FakeClock()
    sess.reads = []  # (prompt, how the real timed_input ended, torque on at the read)
    sess.fg = {"pgrp": os.getpgrp(), "polls": 0}
    sess.foreground = foreground

    def tcgetpgrp(_fd: int) -> int:
        sess.fg["polls"] += 1
        return int(sess.fg["pgrp"])

    monkeypatch.setattr(P.os, "tcgetpgrp", tcgetpgrp)

    def read(prompt: str, timeout_s: float | None) -> str | None:
        try:
            line = P.timed_input(prompt, timeout_s)  # the real reader and its checks
        except EOFError as e:
            sess.reads.append((prompt, type(e).__name__, fol.bus.torque_enabled))
            raise
        sess.reads.append((prompt, "line", fol.bus.torque_enabled))
        return line

    sess.guard = teleop.TorqueGuard(
        arm, teleop.TorqueWait(60.0, 10.0), read=read, clock=sess.clock, sleep=sess.clock.sleep
    )
    real_stop = sess.stop_requested

    def stop_requested() -> bool:
        stop = bool(real_stop())
        fol.log.append(("stop_check", stop))
        return stop

    sess.stop_requested = stop_requested
    return sess, fol


def _run_end(sess: Any, before: Callable[[], None] | None = None) -> str:
    """end_session inside the session's real signal handling (``record_session`` /
    ``main``'s shape): ``before`` runs first, under those handlers."""
    from src.robot import prompts as P

    with P.job_control(), teleop.ctrl_c_as_flag(sess):
        if sess.foreground:
            P.INTERRUPTS.tty_fd = 0  # as with stdin on the terminal (teleop.main's job_control)
        if before is not None:
            before()
        return str(teleop.end_session(sess.arm, sess))


def _assert_held_then_released(sess: Any, fol: ParkFollower, cause: str, t0: float) -> None:
    """The hold -> support prompt (ends at once: nobody can answer) -> grace without stdin ->
    torque off path, recorded as unconfirmed support."""
    assert sess.session_end == "released_unconfirmed"
    assert fol.log[-1][0] == "hold"  # the last goal written is the hold at the measured pose
    assert fol.sent[-1] == pytest.approx(fol.pos)
    (prompt, how, torque_on), *rest = sess.reads
    assert teleop.SUPPORT_PROMPT.strip() in prompt and torque_on  # held under torque, asked
    assert how == ("LostForeground" if cause == "lost_foreground" else "Hangup") and not rest
    assert sess.clock() - t0 == pytest.approx(10.0)  # the whole grace period, no stdin
    assert sess.guard.released is not None and sess.guard.released["cause"] == cause
    assert sess.arm.deadman.response is teleop.DeadmanResponse.TORQUE_OFF
    assert not fol.bus.torque_enabled  # torque off
    rec = teleop.session_record(sess.arm, sess)
    assert rec["session_end"] == "released_unconfirmed" and rec["support_confirmed"] is False
    assert rec["support_release"]["cause"] == cause


def _assert_no_motion_after_stop(fol: ParkFollower) -> int:
    """Every park motion write directly follows a stop check that said go on; nothing moves
    after the first check that said stop. Returns the motion writes before the stop."""
    for i, (kind, _) in enumerate(fol.log):
        if kind == "motion":  # i > 0: the first write too (log[-1] would be the log's tail)
            assert i > 0 and fol.log[i - 1] == ("stop_check", False), fol.log[: i + 1]
    first_stop = fol.log.index(("stop_check", True))
    assert all(kind != "motion" for kind, _ in fol.log[first_stop:]), fol.log
    return sum(kind == "motion" for kind, _ in fol.log[:first_stop])


@pytest.mark.parametrize("foreground", [False, True], ids=["no_tty", "tty_still_foreground"])
@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM], ids=["sighup", "sigterm"])
def test_teleop_hangup_recorded_before_the_park_skips_it(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hard_timeout: None,
    sig: signal.Signals,
    foreground: bool,
) -> None:
    """(a) SIGHUP / (b) SIGTERM, a real signal, delivered before the park would start: zero park
    goal writes and zero park stop checks (the park never starts), then hold, support prompt
    (``hangup`` at once), the 10 s grace, torque off, support unconfirmed. (d) With the terminal
    still reporting this process as its foreground (``tcgetpgrp == getpgrp``, polled and true
    afterwards) the recorded signal alone skips the park. Fails against a041e46 (it parked)."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground)

    def hang_up() -> None:
        assert not P.INTERRUPTS.check_foreground()  # foreground (or no terminal watched)
        os.kill(os.getpid(), sig)
        assert P.INTERRUPTS.hangup == sig.name  # recorded by the session's handler

    t0 = sess.clock()
    end = _run_end(sess, hang_up)
    out = capsys.readouterr().out
    assert end == "released_unconfirmed"
    assert fol.motion_writes() == 0 and not any(k == "stop_check" for k, _ in fol.log)
    assert sess.park_outcome == "skipped_after_hangup"
    assert "NOT parking" in out and "TERMINAL HUNG UP" in out and "follower parked" not in out
    _assert_held_then_released(sess, fol, "hangup", t0)
    rec = teleop.session_record(sess.arm, sess)
    assert rec["park"] == "skipped_after_hangup" and rec["hangup_signal"] == sig.name
    assert rec["lost_foreground"] is None
    if foreground:  # the terminal still says this process is in its foreground
        assert P.not_foreground(0) is None and sess.fg["polls"] >= 1


@pytest.mark.parametrize("foreground", [False, True], ids=["no_tty", "tty_still_foreground"])
@pytest.mark.parametrize("sig", [signal.SIGHUP, signal.SIGTERM], ids=["sighup", "sigterm"])
def test_teleop_hangup_during_the_park_stops_it_before_the_next_goal_write(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hard_timeout: None,
    sig: signal.Signals,
    foreground: bool,
) -> None:
    """(c) A real SIGHUP / SIGTERM sent from the fake's goal-write hook right after the 3rd park
    goal: that write (already in flight) completes, the next stop check sees the hang-up and no
    goal is written after it but the hold; then support prompt, grace, torque off, unconfirmed.
    (d) The same with the terminal still reporting this process as its foreground: the stop
    checks before the signal polled ``tcgetpgrp`` and saw the foreground."""
    sess, fol = _park_session(monkeypatch, foreground)

    def hook(f: ParkFollower) -> None:
        if f.motion_writes() == 3 and f.log[-1][0] == "motion":
            f.log.append(("signal", sig.name))
            os.kill(os.getpid(), sig)

    fol.hook = hook
    t0 = sess.clock()
    end = _run_end(sess)
    out = capsys.readouterr().out
    assert end == "released_unconfirmed" and sess.park_outcome == "interrupted_by_hangup"
    assert _assert_no_motion_after_stop(fol) == 3 and fol.motion_writes() == 3
    sent_at = fol.log.index(("signal", sig.name))
    assert fol.log[sent_at + 1] == ("stop_check", True)  # the next check, before any write
    assert fol.pos["wrist_flex.pos"] == pytest.approx(12.0)  # 3 x 4 deg; 30 deg never reached
    assert "park interrupted" in out and "follower parked" not in out
    _assert_held_then_released(sess, fol, "hangup", t0)
    rec = teleop.session_record(sess.arm, sess)
    assert rec["park"] == "interrupted_by_hangup" and rec["hangup_signal"] == sig.name
    if foreground:
        assert sess.fg["polls"] >= 3  # the checks before the signal asked the terminal


def test_teleop_hangup_during_the_last_park_write_ends_in_the_hold_not_parked(
    monkeypatch: pytest.MonkeyPatch, hard_timeout: None
) -> None:
    """A SIGHUP during the park's last (converging) goal write: ``So101Driver.park`` checks the
    stop once more before its torque-off, so the session still holds, waits the grace and
    records unconfirmed support instead of ``parked``."""
    sess, fol = _park_session(monkeypatch, foreground=False)

    def hook(f: ParkFollower) -> None:
        done = f.pos["wrist_flex.pos"] == pytest.approx(PARK_WRIST_DEG)
        if done and f.log[-1][0] == "hold" and not any(k == "signal" for k, _ in f.log):
            f.log.append(("signal", "SIGHUP"))  # the converged tick's write (no motion left)
            os.kill(os.getpid(), signal.SIGHUP)

    fol.hook = hook
    t0 = sess.clock()
    assert _run_end(sess) == "released_unconfirmed"
    assert sess.park_outcome == "interrupted_by_hangup"
    sent_at = fol.log.index(("signal", "SIGHUP"))
    assert fol.log[sent_at + 1] == ("stop_check", True)  # the driver's check before torque off
    _assert_no_motion_after_stop(fol)
    _assert_held_then_released(sess, fol, "hangup", t0)


def test_teleop_lost_foreground_before_the_park_skips_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """(e) The terminal's foreground taken by another process group before the park (polled
    ``tcgetpgrp``): no park goal, then hold, ``lost_foreground`` at the support prompt, grace,
    torque off, unconfirmed."""
    sess, fol = _park_session(monkeypatch, foreground=True)
    t0 = sess.clock()
    end = _run_end(sess, lambda: sess.fg.update(pgrp=os.getpgrp() + 1))
    assert end == "released_unconfirmed" and fol.motion_writes() == 0
    assert sess.park_outcome == "skipped_after_lost_foreground"
    assert "NOT parking" in capsys.readouterr().out
    _assert_held_then_released(sess, fol, "lost_foreground", t0)
    rec = teleop.session_record(sess.arm, sess)
    assert rec["park"] == "skipped_after_lost_foreground" and rec["hangup_signal"] is None
    assert rec["lost_foreground"] is not None


@pytest.mark.parametrize("how", ["sigttou", "tcgetpgrp"])
def test_teleop_lost_foreground_during_the_park_stops_it_before_the_next_goal_write(
    monkeypatch: pytest.MonkeyPatch, hard_timeout: None, how: str
) -> None:
    """(e) The foreground lost after the 2nd park goal, by a real SIGTTOU (``job_control``'s
    handler) or by the shell taking the terminal (polled ``tcgetpgrp``): no goal after the stop
    is observed but the hold; grace, torque off, unconfirmed."""
    sess, fol = _park_session(monkeypatch, foreground=True)

    def hook(f: ParkFollower) -> None:
        if f.motion_writes() == 2 and f.log[-1][0] == "motion":
            f.log.append(("signal", how))
            if how == "sigttou":
                os.kill(os.getpid(), signal.SIGTTOU)
            else:
                sess.fg["pgrp"] = os.getpgrp() + 1

    fol.hook = hook
    t0 = sess.clock()
    assert _run_end(sess) == "released_unconfirmed"
    assert sess.park_outcome == "interrupted_by_lost_foreground"
    assert _assert_no_motion_after_stop(fol) == 2 and fol.motion_writes() == 2
    assert fol.log[fol.log.index(("signal", how)) + 1] == ("stop_check", True)
    _assert_held_then_released(sess, fol, "lost_foreground", t0)


def test_teleop_healthy_end_with_parking_still_parks_to_the_end(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """(f) No hang-up, no lost foreground (terminal watched and foreground), deadman clear:
    every park goal is written (each after a stop check that said go on) up to the park pose,
    torque off, ``parked``; no support prompt; the record says parked."""
    sess, fol = _park_session(monkeypatch, foreground=True)
    assert _run_end(sess) == "parked" and sess.park_outcome == "parked"
    assert "follower parked; torque OFF" in capsys.readouterr().out
    assert not any(k == "stop_check" and v for k, v in fol.log)
    assert fol.log[0] == ("stop_check", False)  # checked before the very first park write
    for i, (kind, _) in enumerate(fol.log):
        if kind == "motion":
            assert i > 0 and fol.log[i - 1] == ("stop_check", False), fol.log[: i + 1]
    wrist = [a["wrist_flex.pos"] for a in fol.sent]
    assert fol.motion_writes() == math.ceil(PARK_WRIST_DEG / 4.0)  # 4 deg/step velocity cap
    assert wrist[-1] == pytest.approx(PARK_WRIST_DEG) and np.all(np.diff([0.0, *wrist]) >= 0)
    assert not fol.bus.torque_enabled and not sess.arm.deadman.tripped and sess.reads == []
    assert sess.fg["polls"] >= fol.motion_writes()  # the foreground was asked before each write
    rec = teleop.session_record(sess.arm, sess)
    assert rec["session_end"] == "parked" and rec["park"] == "parked"
    assert rec["support_confirmed"] is None and rec["support_release"] is None
    assert rec["hangup_signal"] is None and rec["lost_foreground"] is None


def test_teleop_sighup_at_the_scene_prompt_ends_without_a_park(
    monkeypatch: pytest.MonkeyPatch, hard_timeout: None
) -> None:
    """The whole loop: a real SIGHUP just before the scene prompt reads (the Terminal closed,
    the terminal still naming this process as foreground): the prompt ends as ``hangup``, the
    session ends, the park is skipped, hold, grace, torque off, unconfirmed."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground=True)
    real_read = sess.guard.read_line

    def read(prompt: str, timeout_s: float | None) -> str | None:
        if not sess.reads:
            os.kill(os.getpid(), signal.SIGHUP)
        return real_read(prompt, timeout_s)

    sess.guard.read_line = read
    t0 = sess.clock()
    instr = teleop.protocol_instructions(["Push"])
    with P.job_control():
        P.INTERRUPTS.tty_fd = 0
        saved = teleop.record_session(sess, FakeDataset(), ["Push"], instr, 1, 5.0)
    assert saved == 0 and fol.motion_writes() == 0
    scene, *rest = sess.reads
    assert "arrange the scene" in scene[0] and scene[1] == "Hangup"
    sess.reads = rest
    assert sess.park_outcome == "skipped_after_hangup"
    _assert_held_then_released(sess, fol, "hangup", t0)


# Round 8 (review F1 / F2): the stop is evaluated immediately before every goal write, after the
# blocking camera read and the servo read, so a hang-up recorded while a tick waits on a frame
# or reads the servos never produces one more motion goal (park, sync, episode).
def _assert_nothing_moves_after(fol: ParkFollower, marker: tuple[str, Any]) -> int:
    """No motion write after ``marker`` (the moment the signal was sent and recorded); the
    next logged event is a stop check that says stop. Returns the motion writes before it."""
    at = fol.log.index(marker)
    assert fol.log[at + 1] == ("stop_check", True), fol.log
    assert all(kind != "motion" for kind, _ in fol.log[at:]), fol.log
    return sum(kind == "motion" for kind, _ in fol.log[:at])


@pytest.mark.parametrize("where", ["camera_frame", "servo_read"])
@pytest.mark.parametrize("foreground", [False, True], ids=["no_tty", "tty_still_foreground"])
def test_teleop_hangup_while_a_park_tick_reads_writes_no_further_goal(
    monkeypatch: pytest.MonkeyPatch, hard_timeout: None, where: str, foreground: bool
) -> None:
    """Review F1 (round 8): a real SIGHUP sent while the park's 4th tick waits on its camera
    frame (``Session.frame_time``, after that tick's first stop check) or reads the servos
    (``get_observation`` inside ``step``): the check inside ``step`` right before the goal
    write sees it, so no 4th goal is written; hold, grace, torque off, unconfirmed. Before
    round 8 the 4th goal (up to 4 deg) was still written."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground)
    marker = ("signal", "SIGHUP")

    def send(f: ParkFollower) -> None:
        if f.motion_writes() == 3 and marker not in f.log:
            f.log.append(marker)
            os.kill(os.getpid(), signal.SIGHUP)
            assert P.INTERRUPTS.hangup == "SIGHUP"  # recorded before the write would happen

    if where == "camera_frame":
        real_frame_time = sess.frame_time

        def frame_time() -> float | None:
            send(fol)
            return real_frame_time()

        sess.frame_time = frame_time
    else:
        fol.read_hook = send
    t0 = sess.clock()
    assert _run_end(sess) == "released_unconfirmed"
    assert sess.park_outcome == "interrupted_by_hangup"
    assert _assert_nothing_moves_after(fol, marker) == 3 and fol.motion_writes() == 3
    assert fol.log[fol.log.index(marker) - 1] == ("stop_check", False)  # this tick's 1st check
    assert fol.pos["wrist_flex.pos"] == pytest.approx(12.0)  # 3 x 4 deg, no 4th step
    _assert_no_motion_after_stop(fol)
    _assert_held_then_released(sess, fol, "hangup", t0)
    if foreground:
        assert P.not_foreground(0) is None  # still foreground: the recorded signal decided


def test_teleop_hangup_between_the_skip_check_and_the_park_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """Review F1 (test gap, round 8): a real SIGHUP after ``_park``'s skip check and before it
    clears ``sess.stop`` (sent from the "follower PARKS" print, which can block on a tty), the
    terminal still foreground. ``stop_requested`` reads the RECORDED hang-up, not only
    ``stop``, so the first park check stops it: zero goal writes, ``interrupted_by_hangup``.
    With round 6's ``check_foreground() or self.stop`` the arm parked (8 goals)."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground=True)

    def print_then_hang_up(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)
        if args and "follower PARKS" in str(args[0]):
            fol.log.append(("signal", "SIGHUP"))
            os.kill(os.getpid(), signal.SIGHUP)
            assert P.INTERRUPTS.hangup == "SIGHUP" and sess.stop  # handler ran

    monkeypatch.setattr(teleop, "print", print_then_hang_up, raising=False)
    t0 = sess.clock()
    assert _run_end(sess) == "released_unconfirmed"
    assert "follower PARKS" in capsys.readouterr().out
    assert sess.park_outcome == "interrupted_by_hangup" and fol.motion_writes() == 0
    assert _assert_nothing_moves_after(fol, ("signal", "SIGHUP")) == 0
    assert fol.log[0] == ("signal", "SIGHUP")  # nothing was written before it either
    _assert_held_then_released(sess, fol, "hangup", t0)
    assert P.not_foreground(0) is None


def test_teleop_stop_requested_reads_the_recorded_hangup_not_only_stop(
    monkeypatch: pytest.MonkeyPatch, hard_timeout: None
) -> None:
    """The stop check is True for a recorded SIGHUP with ``sess.stop`` cleared (as ``_park``
    clears it) and the terminal still foreground."""
    from src.robot import prompts as P

    sess, _ = _park_session(monkeypatch, foreground=True)
    real_stop = teleop.Session.stop_requested
    with P.job_control(), teleop.ctrl_c_as_flag(sess):
        P.INTERRUPTS.tty_fd = 0
        assert not real_stop(sess)
        os.kill(os.getpid(), signal.SIGHUP)
        sess.stop = False
        assert P.not_foreground(0) is None and real_stop(sess)


def test_teleop_sigttou_recorded_before_the_park_skips_it_while_still_foreground(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """Review F3 (round 8): a lost foreground RECORDED before the park (a real SIGTTOU through
    ``job_control``'s handler) while ``tcgetpgrp`` still names this process group (e.g. the
    operator ran ``fg``): the recorded cause alone skips the park. Zero stop checks, zero
    motion writes, ``skipped_after_lost_foreground``."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground=True)

    def lose() -> None:
        os.kill(os.getpid(), signal.SIGTTOU)
        assert P.INTERRUPTS.lost_foreground is not None  # recorded by the handler

    t0 = sess.clock()
    assert _run_end(sess, lose) == "released_unconfirmed"
    assert "NOT parking" in capsys.readouterr().out
    assert fol.motion_writes() == 0 and not any(k == "stop_check" for k, _ in fol.log)
    assert sess.park_outcome == "skipped_after_lost_foreground"
    _assert_held_then_released(sess, fol, "lost_foreground", t0)
    rec = teleop.session_record(sess.arm, sess)
    assert rec["park"] == "skipped_after_lost_foreground" and rec["lost_foreground"] is not None
    assert P.not_foreground(0) is None  # the terminal still reports this process as foreground


class SignalCam(FakeCam):
    """Sends a real SIGHUP from inside ``read`` (the tick's blocking frame wait) when ``when()``
    first returns True, and logs it into the follower's log (real-clock frame stamps)."""

    def __init__(self, fol: ParkFollower, when: Callable[[], bool]) -> None:
        super().__init__()
        self.fol, self.when, self.sent = fol, when, False

    def read(self) -> RigFrame:
        if not self.sent and self.when():
            self.sent = True
            self.fol.log.append(("signal", "SIGHUP"))
            os.kill(os.getpid(), signal.SIGHUP)
        return super().read()


def test_teleop_hangup_while_a_sync_tick_reads_its_frame_writes_no_goal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """Review F2 (round 8): a real SIGHUP while ``sync_to_leader``'s 3rd tick waits on its
    frame (after that tick's first stop check), the terminal still foreground: the check in
    ``step`` before the write stops it, no 3rd leader-following goal, sync aborted."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground=True, fake_arm_clock=False)
    sess.leader = FakeLeader(wrist_flex=20.0)  # several rate-limited ticks away
    sess.cam = SignalCam(fol, lambda: fol.motion_writes() == 2)
    with P.job_control(), teleop.ctrl_c_as_flag(sess):
        P.INTERRUPTS.tty_fd = 0
        assert sess.sync_to_leader(timeout_s=5.0) is False
    assert "sync aborted" in capsys.readouterr().out
    assert _assert_nothing_moves_after(fol, ("signal", "SIGHUP")) == 2
    assert fol.motion_writes() == 2 and fol.pos["wrist_flex.pos"] < 20.0  # not synced
    assert fol.sent[-1] == pytest.approx(fol.pos)  # the last goal is the 2nd tick's, then hold
    _assert_no_motion_after_stop(fol)
    assert P.INTERRUPTS.hangup == "SIGHUP" and P.not_foreground(0) is None


def test_teleop_hangup_while_an_episode_tick_reads_its_frame_writes_no_goal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], hard_timeout: None
) -> None:
    """Review F2 (round 8), the whole ``record_session``: ENTER at the scene prompt, sync, then
    a real SIGHUP while the 3rd recording tick waits on its frame (terminal still foreground):
    no goal and no frame for that tick, the episode ends, the keep prompt ends as ``hangup``,
    the episode is discarded, the park is skipped, then hold, grace, torque off, unconfirmed."""
    from src.robot import prompts as P

    sess, fol = _park_session(monkeypatch, foreground=True, fake_arm_clock=False)
    leader = FakeLeader(wrist_flex=2.0)  # sync converges at 2 deg
    sess.leader = leader
    phase = {"recording": False, "ticks": 0}

    def when() -> bool:
        if not phase["recording"]:
            return False
        phase["ticks"] += 1
        return phase["ticks"] == 3

    sess.cam = SignalCam(fol, when)

    def spy_print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)
        if args and str(args[0]).startswith("recording"):
            phase["recording"] = True
            leader.pose["wrist_flex.pos"] = 30.0  # the operator moves the leader
            fol.log.append(("recording", None))

    monkeypatch.setattr(teleop, "print", spy_print, raising=False)
    real_read = sess.guard.read_line
    answers = [""]  # ENTER at the scene prompt; later prompts go through the real reader

    def read(prompt: str, timeout_s: float | None) -> str | None:
        if answers:
            return answers.pop(0)
        return real_read(prompt, timeout_s)

    sess.guard.read_line = read
    data = FakeDataset()
    t0 = sess.clock()
    instr = teleop.protocol_instructions(["Push"])
    with P.job_control():
        P.INTERRUPTS.tty_fd = 0
        saved = teleop.record_session(sess, data, ["Push"], instr, 1, 5.0)
    assert saved == 0 and data.saved == 0 and data.episode_buffer["size"] == 0
    rec_at = fol.log.index(("recording", None))
    episode_motion = _assert_nothing_moves_after(fol, ("signal", "SIGHUP")) - sum(
        kind == "motion" for kind, _ in fol.log[:rec_at]
    )
    assert episode_motion == 2  # ticks 1 and 2 moved; tick 3 wrote nothing
    _assert_no_motion_after_stop(fol)
    keep, *rest = sess.reads
    assert "keep this episode (2 frames)" in keep[0] and keep[1] == "Hangup"
    sess.reads = rest
    assert sess.park_outcome == "skipped_after_hangup"
    _assert_held_then_released(sess, fol, "hangup", t0)
    assert P.not_foreground(0) is None
