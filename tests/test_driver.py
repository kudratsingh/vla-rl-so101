"""SO-101 driver glue against a fake follower: no serial port, no camera."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pytest
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.robot.driver import (
    DeadmanTripped,
    NotCalibrated,
    So101Driver,
    follower_from_config,
)
from src.robot.safety import Deadman, SafetyConfig, SafetyFilter
from src.robot.units import UnitMap


def make(follower: FakeFollower, clock: object = None) -> So101Driver:
    sc = SafetyConfig.from_config(real_cfg())
    clk = clock if callable(clock) else (lambda: 50.0)
    return So101Driver(
        follower, UnitMap(), SafetyFilter(sc, toy_fk, Deadman(sc.frame_timeout_s, clk)), clk
    )


def test_connect_holds_before_torque_enable() -> None:
    fol = FakeFollower(start={"wrist_flex.pos": 12.0})
    drv = make(fol)
    drv.connect()
    names = [c[0] for c in fol.bus.calls]
    i = names.index("enable_torque")
    assert names[i - 2 : i] == ["sync_read", "sync_write"]  # Goal := Present, then torque on
    data_name, goal = fol.bus.calls[i - 1][1]
    assert data_name == "Goal_Position" and goal["wrist_flex"] == 2048 + int(12 * 4095 / 360)
    drv.disconnect()
    assert not fol.bus.torque_enabled and not fol.connected


def test_step_sends_safe_lerobot_units() -> None:
    fol = FakeFollower()
    drv = make(fol)
    drv.connect()
    target = np.array([0.0, 0.0, 0.0, math.radians(3.0), 0.0, math.radians(43.0)])
    out = drv.step(target, frame_time=50.0)
    assert out.safety.status == "pass"
    sent = fol.sent[-1]
    assert sent["wrist_flex.pos"] == pytest.approx(3.0)
    assert sent["gripper.pos"] == pytest.approx(UnitMap().gripper_rad_to_pct(math.radians(43.0)))
    np.testing.assert_allclose(out.measured, UnitMap().to_canonical(FakeFollower().pos))
    big = np.array([0.0, 0.0, 0.0, math.radians(40.0), 0.0, math.radians(43.0)])
    out = drv.step(big, frame_time=50.0, now=50.0 + 1 / 15)
    assert out.safety.status == "clipped" and fol.sent[-1]["wrist_flex.pos"] == pytest.approx(7.0)
    out = drv.step(big, frame_time=50.0, now=50.0 + 1 / 15)  # same instant: dt 0 -> no motion
    assert fol.sent[-1]["wrist_flex.pos"] == pytest.approx(7.0)


def test_stale_frame_holds_and_stops() -> None:
    fol = FakeFollower(start={"elbow_flex.pos": 20.0})
    drv = make(fol)
    drv.connect()
    with pytest.raises(DeadmanTripped, match="stale"):
        drv.step(np.zeros(6), frame_time=49.0)  # 1 s old
    assert fol.sent[-1]["elbow_flex.pos"] == pytest.approx(20.0)  # hold = measured pose
    assert fol.bus.torque_enabled  # hold keeps torque on
    n = len(fol.sent)
    with pytest.raises(DeadmanTripped):
        drv.step(np.zeros(6), frame_time=50.0)
    assert len(fol.sent) == n


def test_context_manager_torque_off_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    fol = FakeFollower()
    with (
        caplog.at_level(logging.CRITICAL, logger="src.robot.driver"),
        pytest.raises(KeyError),
        make(fol) as drv,
    ):
        drv.step(np.zeros(6), frame_time=50.0)
        raise KeyError("policy crashed")
    assert not fol.bus.torque_enabled and not fol.connected
    assert any("torque off" in r.getMessage() for r in caplog.records)
    fol2 = FakeFollower()
    with make(fol2):
        pass
    assert not fol2.bus.torque_enabled and not fol2.connected


def test_refuses_uncalibrated_arm() -> None:
    fol = FakeFollower()
    fol.calibration = {}
    with pytest.raises(NotCalibrated):
        make(fol).connect()
    assert not fol.connected
    fol = FakeFollower()
    fol._calibrated = False
    with pytest.raises(NotCalibrated):
        make(fol).connect()
    assert not fol.bus.torque_enabled and not fol.connected


def test_connect_failure_leaves_arm_limp_and_port_closed() -> None:
    fol = FakeFollower()
    fol.fail_connect = True
    with pytest.raises(ConnectionError):
        make(fol).connect()
    assert not fol.bus.torque_enabled and not fol.connected
    assert ("disconnect", False) in fol.bus.calls


def test_torque_off_failure_is_loud(caplog: pytest.LogCaptureFixture) -> None:
    fol = FakeFollower()
    drv = make(fol)
    drv.connect()
    fol.bus.fail_disable = True
    with caplog.at_level(logging.CRITICAL, logger="src.robot.driver"):
        assert drv.torque_off() is False
    assert any("CUT THE ARM'S POWER" in r.getMessage() for r in caplog.records)


def test_move_to_goes_through_safety() -> None:
    fol = FakeFollower()
    drv = make(fol, clock=time.perf_counter)
    drv.connect()
    goal = np.array([0.0, 0.0, 0.0, math.radians(10.0), 0.0, math.radians(5.0)])
    assert drv.move_to(goal, frame_time_fn=time.perf_counter, timeout_s=5.0)
    steps = [s["wrist_flex.pos"] for s in fol.sent]
    assert max(np.diff([0.0, *steps])) <= 4.0 + 1e-9  # never more than one step per tick


def test_follower_from_config_rejects_placeholder_port() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        follower_from_config(real_cfg())
