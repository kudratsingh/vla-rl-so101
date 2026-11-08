"""Servo temperature / current monitor (src/robot/servo_health.py) and the driver's fault response,
against the fake follower: no serial port."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import numpy as np
import pytest
from robot_fakes import LAST_TORQUE_OFF_CALL, FakeClock, FakeFollower, real_cfg, toy_fk

from src.robot.driver import DeadmanTripped, ParkSettings, ServoFault, So101Driver
from src.robot.safety import DeadmanResponse, make_safety
from src.robot.servo_health import (
    CURRENT,
    TEMPERATURE,
    ServoHealthConfig,
    ServoMonitor,
    monitor_from_config,
)
from src.robot.units import JOINT_NAMES, UnitMap


def config(**over: float) -> ServoHealthConfig:
    c = ServoHealthConfig.from_config(real_cfg())
    assert c is not None
    return dataclasses.replace(c, **over)


def driver(fol: FakeFollower, clock: FakeClock) -> So101Driver:
    cfg = real_cfg()
    arm = So101Driver(
        fol,
        UnitMap(),
        make_safety(cfg, toy_fk, clock=clock),
        clock=clock,
        monitor=monitor_from_config(cfg),
        sleep=clock.sleep,
    )
    arm.connect()
    return arm


def test_config_rate_and_bus_cost_are_explicit() -> None:
    c = config()
    assert (c.read_every_ticks, c.control_hz) == (4, 15.0)
    assert c.register_period_s == pytest.approx(8 / 15)  # each register every 533 ms
    assert c.max_temperature_c < 70  # below the STS3215 firmware cut-out
    assert c.bus_budget_ms < 1e3 / c.control_hz
    with pytest.raises(ValueError, match="70 C"):
        config(max_temperature_c=75.0)
    with pytest.raises(ValueError):
        config(read_every_ticks=0)


def test_reads_one_register_every_k_ticks_alternating() -> None:
    m = ServoMonitor(config())
    due = [m.due() for _ in range(16)]
    assert [i + 1 for i, r in enumerate(due) if r] == [4, 8, 12, 16]
    assert [r for r in due if r] == [TEMPERATURE, CURRENT, TEMPERATURE, CURRENT]


def test_current_trips_only_when_sustained(caplog: pytest.LogCaptureFixture) -> None:
    m = ServoMonitor(config(max_current_ma=1500.0, current_trip_s=1.0))
    ok = {n: 100 for n in JOINT_NAMES}
    hi = {**ok, "gripper": round(2000 / 6.5)}
    assert m.update(CURRENT, hi, 0.0) is None  # a spike
    assert m.update(CURRENT, hi, 0.5) is None
    assert m.update(CURRENT, ok, 0.6) is None  # back under: the timer resets
    assert m.update(CURRENT, hi, 1.0) is None
    assert m.update(CURRENT, hi, 1.9) is None
    reason = m.update(CURRENT, hi, 2.0)
    assert reason is not None and "over-current: gripper" in reason
    assert m.current_ma(0x8000 | 100) == pytest.approx(650.0)  # bit 15 (direction) masked
    with caplog.at_level(logging.WARNING, logger="src.robot.servo_health"):
        m.note_read_ms(3.0)
        m.note_read_ms(25.0)  # slower than the bus budget: warned, counted
    s = m.summary()
    assert s["reads"] == 2 and s["read_ms"]["over_budget"] == 1 and s["read_ms"]["max"] == 25.0
    assert any("bus budget" in r.getMessage() for r in caplog.records)


def test_over_temperature_cuts_torque_at_once_not_hold_not_park() -> None:
    clock = FakeClock()
    fol = FakeFollower()
    arm = driver(fol, clock)
    target = np.zeros(6)
    for _ in range(3):  # ticks 1-3: no health read
        clock.advance(1 / 15)
        arm.step(target, frame_time=clock())
    fol.temperature_c["elbow_flex"] = 60.0
    n_sent = len(fol.sent)
    clock.advance(1 / 15)
    with pytest.raises(ServoFault, match="over-temperature: elbow_flex") as e:
        arm.step(target, frame_time=clock())  # tick 4: writes, then reads the temperature
    assert "Support it" in str(e.value)
    assert len(fol.sent) == n_sent + 1  # this tick's command only: no hold write afterwards
    assert not fol.bus.torque_enabled and fol.bus.calls[-1] == LAST_TORQUE_OFF_CALL
    assert arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF
    assert arm.servo_fault is not None and "elbow_flex" in arm.servo_fault
    # nothing reaches the bus any more: no hold, no park, no step
    arm.hold()
    settings = ParkSettings(qpos=(0.0,) * 6, tol_rad=math.radians(3.0), timeout_s=5.0)
    assert arm.park(settings, clock) is False
    with pytest.raises(DeadmanTripped, match="deadman latched: servo fault"):
        arm.step(target, frame_time=clock())
    assert len(fol.sent) == n_sent + 1 and not fol.bus.torque_enabled


def test_check_servos_on_demand_and_sustained_current_trip() -> None:
    clock = FakeClock()
    fol = FakeFollower()
    arm = driver(fol, clock)
    arm.check_servos()  # all healthy
    assert ("sync_read", TEMPERATURE) in fol.bus.calls and ("sync_read", CURRENT) in fol.bus.calls
    fol.current_ma["gripper"] = 2500.0  # squeezing
    target = np.zeros(6)
    with pytest.raises(ServoFault, match="over-current: gripper"):
        for _ in range(15 * 3):  # the current register is read every 533 ms; trips after >= 1 s
            clock.advance(1 / 15)
            arm.step(target, frame_time=clock())
    assert not fol.bus.torque_enabled and arm.deadman.response is DeadmanResponse.TORQUE_OFF
    s = arm.monitor.summary() if arm.monitor else {}
    assert s["max_current_ma"]["gripper"] >= 2500.0 - 6.5


def test_over_current_across_an_unmonitored_gap_is_not_sustained() -> None:
    """Regression (PR 29 review): one over-threshold reading at a settle read and one spike at
    the next homing read minutes later used to trip "for 90 s"."""
    m = ServoMonitor(config(max_current_ma=1500.0, current_trip_s=1.0))
    hi = {**{n: 100 for n in JOINT_NAMES}, "shoulder_lift": round(1599 / 6.5)}
    assert m.update(CURRENT, hi, 0.0) is None
    assert m.update(CURRENT, hi, 90.0) is None  # 90 s with no current read: the window restarts
    assert m.update(CURRENT, hi, 90.533) is None
    reason = m.update(CURRENT, hi, 91.066)  # continuously monitored for >= 1 s: trips
    assert reason is not None and "3 consecutive readings" in reason
    m2 = ServoMonitor(config(max_current_ma=1500.0, current_trip_s=1.0))
    assert [m2.update(CURRENT, hi, t) for t in (0.0, 0.533)] == [None, None]
    assert m2.update(CURRENT, hi, 1.066) is not None


@pytest.mark.parametrize("stays_high", [True, False])
def test_check_servos_rereads_an_over_current_before_a_prompt(stays_high: bool) -> None:
    """Regression (PR 29 review): an over-current found by the pre-prompt read used to wait for
    the next read after the whole prompt (up to 60 s) before it could trip."""
    clock = FakeClock()
    fol = FakeFollower()
    arm = driver(fol, clock)
    fol.current_ma["elbow_flex"] = 1800.0  # holding against an obstacle
    real_read = fol.bus.sync_read

    def sync_read(data_name: str, motors: Any = None, *, normalize: bool = True) -> Any:
        out = real_read(data_name, motors, normalize=normalize)
        if data_name == CURRENT and not stays_high:
            fol.current_ma["elbow_flex"] = 200.0  # the next read is back under the limit
        return out

    fol.bus.sync_read = sync_read  # type: ignore[method-assign]
    t0 = clock()
    if stays_high:
        with pytest.raises(ServoFault, match="over-current: elbow_flex"):
            arm.check_servos()
        assert not fol.bus.torque_enabled and arm.deadman.response is DeadmanResponse.TORQUE_OFF
    else:
        arm.check_servos()
        assert fol.bus.torque_enabled and not arm.deadman.tripped
    assert clock() - t0 == pytest.approx(1.0)  # one current_trip_s re-read, inside the call
    assert sum(c == ("sync_read", CURRENT) for c in fol.bus.calls) == 2  # the read + the re-read
