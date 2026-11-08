"""SO-101 driver glue against a fake follower: no serial port, no camera."""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import pytest
from robot_fakes import PORT_BUSY, FakeFollower, full_sweep_calibration, real_cfg, toy_fk

from src.robot.driver import (
    DeadmanTripped,
    NotCalibrated,
    ParkSettings,
    RestTolerance,
    So101Driver,
    TorqueEnableRefused,
    follower_from_config,
    release_port,
)
from src.robot.safety import Deadman, DeadmanResponse, SafetyConfig, SafetyFilter
from src.robot.units import UnitMap


def make(
    follower: FakeFollower, clock: object = None, rest_tol: RestTolerance | None = None
) -> So101Driver:
    sc = SafetyConfig.from_config(real_cfg())
    clk = clock if callable(clock) else (lambda: 50.0)
    return So101Driver(
        follower,
        UnitMap(),
        SafetyFilter(sc, toy_fk, Deadman(sc.frame_timeout_s, clk)),
        clk,
        rest_tol=rest_tol,
        sleep=lambda _s: None,
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


# -- torque-enable refusal ------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("start", "match"),
    [
        ({"gripper.pos": -10.0}, "gripper rests at raw"),  # past the closed end of its range
        ({"wrist_flex.pos": 97.0}, "wrist_flex rests at raw"),  # past the URDF/calibrated limit
    ],
)
def test_torque_enable_refused_outside_calibrated_range(
    start: dict[str, float], match: str
) -> None:
    fol = FakeFollower(start=start)
    with pytest.raises(TorqueEnableRefused, match=match) as e:
        make(fol).connect()
    assert "torque stays OFF" in str(e.value) and "move the joint(s) by hand" in str(e.value)
    assert not fol.bus.torque_enabled and "enable_torque" not in [c[0] for c in fol.bus.calls]
    assert ("disconnect", False) in fol.bus.calls  # port closed, nothing driven


def test_torque_enable_refused_on_eeprom_mismatch_or_safety_limits() -> None:
    fol = FakeFollower()
    eeprom = full_sweep_calibration()
    c = eeprom["elbow_flex"]
    eeprom["elbow_flex"] = type(c)(
        c.id, c.drive_mode, c.homing_offset, c.range_min + 50, c.range_max
    )
    fol.bus.eeprom = eeprom
    with pytest.raises(TorqueEnableRefused, match="EEPROM limits .* differ"):
        make(fol).connect()
    assert not fol.bus.torque_enabled
    # 94 deg: inside the calibrated range (95), 4 deg outside the safety limit (90)
    ok = FakeFollower(start={"wrist_flex.pos": 94.0})
    make(ok).connect()  # default tolerance 6 deg: a joint resting on its stop passes
    assert ok.bus.torque_enabled
    tight = FakeFollower(start={"wrist_flex.pos": 94.0})
    with pytest.raises(TorqueEnableRefused, match="outside the safety limits"):
        make(tight, rest_tol=RestTolerance(safety_limits_deg=2.0)).connect()
    assert not tight.bus.torque_enabled


# -- Ctrl-C port-busy backstop --------------------------------------------------------------------
def test_ctrl_c_mid_transaction_still_reaches_the_bus() -> None:
    fol = FakeFollower(start={"elbow_flex.pos": 20.0})
    drv = make(fol)
    drv.connect()
    fol.interrupt_next = "write"  # Ctrl-C inside the Goal_Position sync_write
    with pytest.raises(KeyboardInterrupt):
        drv.step(np.zeros(6), frame_time=50.0)
    assert fol.bus.port_handler.is_using  # what scservo_sdk leaves behind
    with pytest.raises(ConnectionError, match=PORT_BUSY.replace("[", r"\[").replace("]", r"\]")):
        fol.bus.sync_read("Present_Position")  # every later transaction would fail
    n = len(fol.sent)
    drv.trip("operator Ctrl-C")  # the client's response: hold, after releasing the port
    assert not fol.bus.port_handler.is_using and fol.bus.port_handler.clears == 1
    assert fol.bus.port_handler.ser.input_resets == 1
    assert len(fol.sent) == n + 1 and fol.sent[-1] == pytest.approx(fol.pos)  # hold reached it
    assert drv.deadman.response is DeadmanResponse.HOLD and fol.bus.torque_enabled
    fol.interrupt_next = "read"  # and a Ctrl-C inside a read, then torque off
    with pytest.raises(KeyboardInterrupt):
        drv.read_state()
    assert drv.torque_off() and not fol.bus.torque_enabled
    assert release_port(fol.bus) is False  # nothing to clear between transactions


# -- healthy-shutdown park ------------------------------------------------------------------------
PARK = (0.0, 0.0, 0.0, math.radians(10.0), 0.0, math.radians(5.0))


def test_park_drives_through_safety_then_torque_off() -> None:
    fol = FakeFollower()
    drv = make(fol, clock=time.perf_counter)
    drv.connect()
    settings = ParkSettings(qpos=PARK, tol_rad=math.radians(1.0), timeout_s=5.0)
    assert drv.park(settings, frame_time_fn=time.perf_counter)
    steps = [s["wrist_flex.pos"] for s in fol.sent]
    assert steps[-1] == pytest.approx(10.0) and max(np.diff([0.0, *steps])) <= 4.0 + 1e-9
    assert not fol.bus.torque_enabled and fol.bus.calls[-1] == ("disable_torque", None)
    cfg = real_cfg()
    assert ParkSettings.from_config(cfg).qpos == tuple(cfg["deploy"]["home_qpos"])  # null -> home


def test_park_never_runs_on_a_fault_path() -> None:
    fol = FakeFollower()
    drv = make(fol, clock=time.perf_counter)
    drv.connect()
    settings = ParkSettings(qpos=PARK, tol_rad=math.radians(1.0), timeout_s=5.0)
    drv.trip("stale frame")  # any deadman trip: hold, then no park
    n = len(fol.sent)
    assert drv.park(settings, frame_time_fn=time.perf_counter) is False
    assert len(fol.sent) == n and fol.bus.torque_enabled  # nothing moved; still holding
    fol2 = FakeFollower()
    drv2 = make(fol2, clock=time.perf_counter)
    drv2.connect()
    stop = iter([False, False, True])  # the operator's Ctrl-C flag, checked between ticks
    assert drv2.park(settings, time.perf_counter, stop_fn=lambda: next(stop)) is False
    assert len(fol2.sent) == 3 and fol2.sent[-1] == pytest.approx(fol2.pos)  # 2 steps + hold
    assert fol2.bus.torque_enabled  # the caller asks the operator to support the arm first
