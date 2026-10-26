"""Safety layer (CLAUDE.md rule 8). FK is the TEST-ONLY toy_fk; no hardware is touched."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any

import numpy as np
import pytest
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.robot.driver import SerialFault, So101Driver
from src.robot.safety import (
    Deadman,
    DeadmanResponse,
    SafetyConfig,
    SafetyFilter,
    resolve_joint_limits_deg,
)
from src.robot.units import UnitMap

DT = 1.0 / 15.0
D = math.radians


def make_filter(**safety_overrides: Any) -> SafetyFilter:
    cfg = copy.deepcopy(real_cfg())
    cfg["robot"]["safety"].update(safety_overrides)
    sc = SafetyConfig.from_config(cfg)
    return SafetyFilter(sc, toy_fk, Deadman(sc.frame_timeout_s, clock=lambda: 0.0))


def q(**deg: float) -> np.ndarray:
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    return np.array([D(deg.get(n, 0.0)) for n in names])


def test_real_yaml_safety_config() -> None:
    sc = SafetyConfig.from_config(real_cfg())
    assert math.degrees(sc.max_joint_step_rad) == pytest.approx(4.0)
    assert math.degrees(sc.max_velocity_rad_s) == pytest.approx(60.0)
    assert sc.control_hz == 15 and sc.frame_timeout_s == pytest.approx(0.3)
    # no joint_limits_deg configured -> URDF limits shrunk by 5 deg
    assert math.degrees(sc.upper_rad[0]) == pytest.approx(110.0 - 5.0, abs=1e-3)
    assert math.degrees(sc.lower_rad[5]) == pytest.approx(-10.0 + 5.0, abs=1e-3)
    assert sc.workspace_lo_m == (0.10, -0.20, 0.00) and sc.workspace_hi_m == (0.40, 0.20, 0.25)


def test_safe_command_passes_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    f = make_filter()
    measured = q(wrist_flex=10, gripper=30)
    target = q(shoulder_pan=2, shoulder_lift=-3, elbow_flex=3.9, wrist_flex=7, gripper=33)
    with caplog.at_level(logging.INFO, logger="src.robot.safety"):
        r = f.filter(target, measured, now=0.0)
    assert r.status == "pass" and r.interventions == ()
    assert r.command is not None
    np.testing.assert_array_equal(r.command, target)  # bit-for-bit
    assert not caplog.records


def test_out_of_bounds_command_is_blocked(caplog: pytest.LogCaptureFixture) -> None:
    """WO-06 benchmark: a command whose TCP leaves the workspace box never reaches the bus."""
    f = make_filter()
    measured = q()  # toy TCP (0.26, 0, 0.10): inside the box
    into_table = q(shoulder_lift=70)  # toy TCP z < 0: below the table
    assert toy_fk(into_table)[2] < 0
    with caplog.at_level(logging.WARNING, logger="src.robot.safety"):
        r = f.filter(into_table, measured, now=0.0)
    assert r.status == "rejected"
    assert r.command is not None
    np.testing.assert_array_equal(r.command, measured)  # hold, not projected
    assert [iv.kind for iv in r.interventions][-1] == "workspace_reject"
    assert any("workspace_reject" in rec.getMessage() for rec in caplog.records)
    # held pose is the new reference: the next safe command is measured from it
    r2 = f.filter(q(wrist_flex=1), measured, now=DT)
    assert r2.status == "pass"

    # one small step that would cross the table plane is rejected too
    f = make_filter()
    near_table = np.array([0.0, 0.38, 0.0, 0.0, 0.0, 0.0])  # toy TCP z ~ +4 mm
    assert 0 < toy_fk(near_table)[2] < 0.01
    step_down = near_table + np.array([0.0, D(2.5), 0.0, 0.0, 0.0, 0.0])
    r = f.filter(step_down, near_table, now=0.0)
    assert r.status == "rejected"
    assert r.command is not None
    np.testing.assert_array_equal(r.command, near_table)


def test_joint_limit_clip(caplog: pytest.LogCaptureFixture) -> None:
    f = make_filter()
    with caplog.at_level(logging.WARNING, logger="src.robot.safety"):
        r = f.filter(q(wrist_flex=120, gripper=-30), q(wrist_flex=88, gripper=-3), now=0.0)
    assert r.status == "clipped" and r.command is not None
    assert math.degrees(r.command[3]) == pytest.approx(95 - 5, abs=1e-3)  # URDF 95 - 5 margin
    assert math.degrees(r.command[5]) == pytest.approx(-10 + 5, abs=1e-3)
    kinds = {(iv.kind, iv.joint) for iv in r.interventions}
    assert {("joint_limit", "wrist_flex"), ("joint_limit", "gripper")} <= kinds
    assert sum("joint_limit" in rec.getMessage() for rec in caplog.records) == 2


def test_configured_limits_only_narrow() -> None:
    lim = resolve_joint_limits_deg({"joint_limits_deg": {"wrist_flex": [-200, 30]}})
    assert lim["wrist_flex"] == pytest.approx((-95 + 5, 30), abs=1e-3)
    f = make_filter(joint_limits_deg={"wrist_flex": [-20, 20]})
    r = f.filter(q(wrist_flex=25), q(wrist_flex=19), now=0.0)
    assert r.command is not None and math.degrees(r.command[3]) == pytest.approx(20)
    with pytest.raises(ValueError):
        resolve_joint_limits_deg({"joint_limits_deg": {"elbow": [0, 1]}})


def test_never_further_out_than_start_pose() -> None:
    """Folded start pose beyond the shrunk limit (and above the box): only inward motion."""
    f = make_filter()
    rest = q(shoulder_lift=-97.5)  # beyond -95; toy TCP above the box
    r = f.filter(q(shoulder_lift=-110), rest, now=0.0)
    assert r.command is not None
    np.testing.assert_array_equal(r.command, rest)  # no outward motion
    r = f.filter(q(shoulder_lift=-80), rest, now=DT)
    assert r.status == "clipped" and r.command is not None
    assert math.degrees(r.command[1]) == pytest.approx(-97.5 + 4.0)  # inward, one step


def test_step_clip() -> None:
    f = make_filter(max_velocity_deg_s=180.0)
    r = f.filter(q(wrist_flex=30, wrist_roll=-30), q(), now=0.0)
    assert r.status == "clipped" and r.command is not None
    np.testing.assert_allclose(np.degrees(r.command), [0, 0, 0, 4.0, -4.0, 0])
    assert {iv.kind for iv in r.interventions} == {"step_clip"}


def test_velocity_cap_over_several_steps() -> None:
    f = make_filter(max_velocity_deg_s=30.0)  # 2 deg per step at 15 Hz (< 4 deg step clip)
    target = q(wrist_flex=60)
    measured, t, prev = q(), 0.0, 0.0
    for _ in range(10):
        r = f.filter(target, measured, now=t)
        assert r.command is not None and r.status == "clipped"
        assert {iv.kind for iv in r.interventions} == {"velocity_cap"}
        pos = math.degrees(r.command[3])
        assert pos - prev == pytest.approx(2.0)
        measured, prev = r.command, pos
        t += DT
    assert prev == pytest.approx(20.0)  # 10 steps x 30 deg/s x 1/15 s
    t -= DT  # time of the last command

    # loop running twice as fast -> half the step: the cap is a velocity, not a per-call step
    for _ in range(5):
        t += DT / 2
        r = f.filter(target, measured, now=t)
        assert r.command is not None
        pos = math.degrees(r.command[3])
        assert (pos - prev) / (DT / 2) == pytest.approx(30.0)
        measured, prev = r.command, pos
    # a stalled loop does not earn a bigger step; a clock going backwards earns none
    r = f.filter(target, measured, now=t + 5.0)
    assert r.command is not None
    assert math.degrees(r.command[3]) - prev == pytest.approx(2.0)
    r2 = f.filter(target, r.command, now=t)
    assert r2.command is not None
    np.testing.assert_array_equal(r2.command, r.command)


def test_invalid_inputs_fail_safe() -> None:
    f = make_filter()
    r = f.filter([0, 0, float("nan"), 0, 0, 0], q(), now=0.0)
    assert r.status == "rejected" and r.command is not None
    np.testing.assert_array_equal(r.command, q())
    r = f.filter(q()[:5], q(), now=DT)
    assert r.status == "rejected"
    # FK failure fails closed
    f = SafetyFilter(f.config, lambda _q: np.array([np.nan, 0, 0]), Deadman(0.3))
    assert f.filter(q(wrist_flex=1), q(), now=0.0).status == "rejected"
    # a broken clock cannot earn a full step
    f = make_filter()
    assert f.filter(q(wrist_flex=1), q(), now=float("nan")).status == "deadman"
    # unreadable joints: deadman, torque off, no command
    f = make_filter()
    r = f.filter(q(), [0, 0, 0, float("inf"), 0, 0], now=0.0)
    assert r.status == "deadman" and r.command is None
    assert f.deadman.response is DeadmanResponse.TORQUE_OFF


def test_stall_trips_deadman() -> None:
    f = make_filter()
    r = f.filter(q(wrist_flex=3.9), q(), now=0.0)
    assert r.status == "pass"
    blocked = q(wrist_flex=-17)  # arm pushed/blocked 21 deg away from the command
    r = f.filter(q(wrist_flex=8), blocked, now=DT)
    assert r.status == "deadman" and r.command is None
    assert f.deadman.response is DeadmanResponse.HOLD and "tracking error" in f.deadman.reason
    # gripper squeezing an object is not a stall
    f = make_filter()
    f.filter(q(gripper=-4), q(gripper=-4), now=0.0)
    assert f.filter(q(gripper=-5), q(gripper=25), now=DT).status != "deadman"


def test_deadman_trips_on_stale_frame_and_latches(caplog: pytest.LogCaptureFixture) -> None:
    dm = Deadman(0.3, clock=lambda: 100.0)
    assert dm.check_frame(99.8)  # 200 ms old: fine
    with caplog.at_level(logging.CRITICAL, logger="src.robot.safety"):
        assert not dm.check_frame(99.6)  # 400 ms old
    assert dm.tripped and dm.response is DeadmanResponse.HOLD and "stale" in dm.reason
    assert any("DEADMAN" in rec.getMessage() for rec in caplog.records)
    assert not dm.check_frame(100.0)  # latched: a fresh frame does not clear it
    f = SafetyFilter(make_filter().config, toy_fk, dm)
    assert f.filter(q(), q(), now=100.0).command is None
    with pytest.raises(ValueError):
        dm.reset("")
    dm.reset("operator checked the camera")
    assert not dm.tripped and dm.check_frame(100.0)
    assert not dm.check_frame(None) and "no camera frame" in dm.reason
    dm.reset("ok")
    assert not dm.check_frame(100.5) and "future" in dm.reason  # mismatched clocks


def _driver(follower: FakeFollower, t: list[float]) -> So101Driver:
    sc = SafetyConfig.from_config(real_cfg())
    dm = Deadman(sc.frame_timeout_s, clock=lambda: t[0])
    return So101Driver(follower, UnitMap(), SafetyFilter(sc, toy_fk, dm), clock=lambda: t[0])


def test_deadman_trips_on_serial_exception() -> None:
    t = [10.0]
    fol = FakeFollower()
    drv = _driver(fol, t)
    drv.connect()
    assert fol.bus.torque_enabled
    drv.step(q(wrist_flex=1), frame_time=t[0])
    fol.fail_read = True
    with pytest.raises(SerialFault):
        drv.step(q(wrist_flex=2), frame_time=t[0])
    assert drv.deadman.response is DeadmanResponse.TORQUE_OFF
    assert "serial error" in drv.deadman.reason
    assert not fol.bus.torque_enabled  # torque cut
    n_sent = len(fol.sent)
    fol.fail_read = False
    with pytest.raises(RuntimeError, match="deadman"):
        drv.step(q(wrist_flex=2), frame_time=t[0])  # latched: nothing more is sent
    assert len(fol.sent) == n_sent

    # a write failure trips it too, and escalates an earlier HOLD trip to TORQUE_OFF
    fol2 = FakeFollower()
    drv2 = _driver(fol2, t)
    drv2.connect()
    drv2.deadman.trip("test hold", DeadmanResponse.HOLD)
    drv2.deadman.serial_error(ConnectionError("Failed to sync write"))
    assert drv2.deadman.response is DeadmanResponse.TORQUE_OFF


def test_hard_ceilings_reject_aggressive_configs() -> None:
    with pytest.raises(ValueError):
        make_filter(max_joint_step_deg=30.0)
    with pytest.raises(ValueError):
        make_filter(max_velocity_deg_s=500.0)
    with pytest.raises(ValueError):
        make_filter(workspace_box_m={"x": [0.4, 0.1], "y": [-0.2, 0.2], "z": [0.0, 0.25]})
