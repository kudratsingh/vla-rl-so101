"""src/robot/prompts.py: bounded torque-on prompts and the support prompt, on the fake arm with a
scripted stdin (tests/robot_fakes.FakeReader). No serial port, no terminal."""

from __future__ import annotations

import math
from typing import Any

import pytest
from robot_fakes import FakeClock, FakeFollower, FakeReader, real_cfg, toy_fk

from src.robot import prompts as P
from src.robot.driver import So101Driver
from src.robot.safety import DeadmanResponse, make_safety
from src.robot.servo_health import monitor_from_config
from src.robot.units import UnitMap

WAIT = P.TorqueWait(60.0, 10.0)


def _guard(
    script: list[Any], wait: P.TorqueWait | None = WAIT
) -> tuple[P.TorqueGuard, FakeReader, FakeFollower, list[str], FakeClock]:
    fol = FakeFollower()
    arm = So101Driver(
        fol, UnitMap(), make_safety(real_cfg(), toy_fk), monitor=monitor_from_config(real_cfg())
    )
    arm.connect()
    clock = FakeClock()
    reader = FakeReader(script, clock, fol.bus)
    said: list[str] = []
    guard = P.TorqueGuard(arm, wait, read=reader, clock=clock, sleep=clock.sleep, say=said.append)
    return guard, reader, fol, said, clock


def _health_reads(fol: FakeFollower) -> int:
    regs = ("Present_Temperature", "Present_Current")
    return sum(c[0] == "sync_read" and c[1] in regs for c in fol.bus.calls)


def test_support_prompt_enter_confirms_and_keeps_torque_for_the_caller() -> None:
    g, reader, fol, _said, _ = _guard([""])
    assert g.support("support it, ENTER: ") == P.ENTER
    assert reader.calls == [("support it, ENTER: ", 60.0, True)]
    assert _health_reads(fol) == 2  # temperature + current, before the wait
    assert g.released is None and fol.bus.torque_enabled  # the caller turns torque off


def test_ctrl_c_at_the_support_prompt_is_ignored_and_the_wait_keeps_running() -> None:
    g, reader, fol, said, _ = _guard([KeyboardInterrupt, KeyboardInterrupt, ""])
    assert g.support("support it, ENTER: ") == P.ENTER
    # each Ctrl-C came 5 s into the wait: the deadline is not reset, the arm kept holding
    assert [c[1] for c in reader.calls] == [60.0, 55.0, 50.0]
    assert all(c[2] for c in reader.calls) and fol.bus.torque_enabled
    assert said.count(P.CTRL_C_IGNORED) == 2 and g.released is None


def test_ctrl_c_then_no_answer_releases_torque_unconfirmed() -> None:
    g, reader, fol, said, _ = _guard([KeyboardInterrupt, None, KeyboardInterrupt, None])
    assert g.support("support it, ENTER: ") == P.TIMEOUT  # Ctrl-C never confirms
    # 60 s wait (Ctrl-C at 5 s, then no answer for the remaining 55 s), then the 10 s grace with
    # a Ctrl-C in it (ignored, the grace keeps running) while torque is still on
    assert [(c[1], c[2]) for c in reader.calls] == [(60.0, True), (55.0, True), (10.0, True)] + [
        (5.0, True)
    ]
    assert any("SUPPORT THE ARM NOW" in m for m in said) and P.CTRL_C_IGNORED_GRACE in said
    assert g.released is not None and g.released["cause"] == P.TIMEOUT
    assert g.released["torque_off_confirmed"] and not fol.bus.torque_enabled
    assert g.arm.deadman.response is DeadmanResponse.TORQUE_OFF  # nothing holds or parks now


def test_closed_stdin_at_the_support_prompt_waits_the_grace_without_stdin() -> None:
    g, reader, fol, said, clock = _guard([EOFError])
    t0 = clock()
    assert g.support("support it, ENTER: ") == P.EOF
    assert len(reader.calls) == 1  # stdin is not read again
    assert clock() - t0 == pytest.approx(10.0)  # the grace, slept
    assert any("STDIN CLOSED" in m for m in said)
    assert g.released is not None and g.released["cause"] == P.EOF
    assert not fol.bus.torque_enabled and g.arm.deadman.response is DeadmanResponse.TORQUE_OFF


def test_enter_during_the_grace_releases_now() -> None:
    g, _reader, fol, _, clock = _guard([None, ""])
    t0 = clock()
    assert g.support("support it, ENTER: ") == P.TIMEOUT
    assert clock() - t0 == pytest.approx(60.0)  # the wait, then ENTER at once in the grace
    assert not fol.bus.torque_enabled and g.released and g.released["cause"] == P.TIMEOUT


def test_ask_reads_servos_around_the_wait_and_times_out() -> None:
    g, _reader, fol, _said, _ = _guard(["y"])
    assert g.ask("score? ").answer == "y"
    assert _health_reads(fol) == 4 and fol.bus.torque_enabled  # 2 before, 2 after
    g2, reader2, fol2, _said2, _ = _guard([None, None])
    assert g2.ask("score? ").how == P.TIMEOUT
    assert [c[1] for c in reader2.calls] == [60.0, 10.0]  # the prompt, then the grace
    assert _health_reads(fol2) == 2 and not fol2.bus.torque_enabled
    g3, _, fol3, _, _ = _guard([KeyboardInterrupt])
    assert g3.ask("score? ").how == P.CTRL_C and fol3.bus.torque_enabled  # the caller decides
    g4, _, fol4, _, _ = _guard([EOFError])
    assert g4.ask("score? ").how == P.EOF and fol4.bus.torque_enabled


def test_acknowledge_with_torque_off_is_bounded_and_records_how() -> None:
    for script, how in (([""], P.ENTER), ([KeyboardInterrupt], P.CTRL_C), ([EOFError], P.EOF)):
        g, _, _, _, _ = _guard(script)
        assert g.acknowledge("rests safely? ") == how
    g, reader, fol, _said, _ = _guard([None])
    assert fol.bus.torque_enabled
    assert g.acknowledge("rests safely? ") == P.TIMEOUT and g.released is None  # no release
    assert reader.calls[0][1] == 60.0 and _health_reads(fol) == 0
    assert P.ack_record("rests safely? ", P.TIMEOUT) == {
        "prompt": "rests safely?",
        "given": False,
        "how": "timeout",
    }


def test_from_config_refuses_an_unbounded_wait_on_the_real_arm() -> None:
    cfg = real_cfg()
    assert P.TorqueWait.from_config(cfg, "deploy", allow_none=False) == P.TorqueWait(60.0, 10.0)
    for bad in (None, 0, -5, math.nan, math.inf):
        cfg["deploy"]["prompt_torque_timeout_s"] = bad
        with pytest.raises(ValueError, match="prompt_torque_timeout_s"):
            P.TorqueWait.from_config(cfg, "deploy", allow_none=False)
    cfg["deploy"]["prompt_torque_timeout_s"] = None
    assert P.TorqueWait.from_config(cfg, "deploy", allow_none=True) is None
