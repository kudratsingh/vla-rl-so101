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
    cfg["deploy"]["prompt_support_grace_s"] = 7.0
    # teleop's null = no bound, but the grace period is still read and kept (F5)
    assert P.TorqueWait.from_config(cfg, "deploy", allow_none=True) == P.TorqueWait(None, 7.0)


# -- review round 3: Ctrl-C outside a read, typed-ahead input, hang-up, the null-timeout grace ----
def test_ctrl_c_during_the_closed_stdin_grace_does_not_drop_the_arm() -> None:
    g, _reader, fol, said, clock = _guard([EOFError])
    presses = {"n": 0}

    def sleep(dt: float) -> None:  # the operator panics 3 s into the grace and presses Ctrl-C
        if presses["n"] == 0:
            presses["n"] += 1
            clock.advance(3.0)
            assert fol.bus.torque_enabled  # still holding when the press lands
            raise KeyboardInterrupt
        clock.advance(dt)

    g.sleep = sleep
    t0 = clock()
    assert g.support("support it, ENTER: ") == P.EOF  # returns; nothing escapes
    assert clock() - t0 == pytest.approx(10.0)  # the whole grace, not cut short by the press
    assert P.CTRL_C_IGNORED_GRACE in said
    assert g.released is not None and g.released["cause"] == P.EOF
    assert not fol.bus.torque_enabled  # torque off only after the grace


def test_ctrl_c_in_the_servo_read_before_the_support_wait_is_ignored() -> None:
    g, reader, fol, said, _ = _guard([""])
    real = g.arm.check_servos
    calls = {"n": 0}

    def check_servos() -> None:
        calls["n"] += 1
        if calls["n"] == 1:  # lands mid-transfer: the port is left busy
            fol.bus.port_handler.is_using = True
            raise KeyboardInterrupt
        real()

    g.arm.check_servos = check_servos  # type: ignore[method-assign]
    assert g.support("support it, ENTER: ") == P.ENTER  # still asked, still confirmed by ENTER
    assert calls["n"] == 2 and not fol.bus.port_handler.is_using  # port released, read retried
    assert P.CTRL_C_IGNORED in said and fol.bus.torque_enabled  # the caller turns torque off
    assert reader.calls[0][1] == 60.0  # the bounded wait (its deadline set before the read)


def test_null_timeout_keeps_the_grace_on_a_closed_stdin() -> None:
    g, reader, fol, said, clock = _guard([EOFError], wait=P.TorqueWait(None, 10.0))
    t0 = clock()
    assert g.support("support it, ENTER: ") == P.EOF
    assert reader.calls[0][1] is None  # no bound (teleop null) ...
    assert clock() - t0 == pytest.approx(10.0)  # ... but the grace still runs before torque off
    assert any("Torque goes OFF in 10 s" in m for m in said) and not fol.bus.torque_enabled
    g2, _, _fol2, _, clock2 = _guard([EOFError], wait=None)  # no TorqueWait at all: the default
    t0 = clock2()
    assert g2.support("x") == P.EOF and clock2() - t0 == pytest.approx(P.DEFAULT_GRACE_S)


def _stdin_from(monkeypatch: pytest.MonkeyPatch, fd: int) -> Any:
    f = open(fd, "r", closefd=True)  # noqa: SIM115 - closed by the test
    monkeypatch.setattr(P.sys, "stdin", f)
    return f


def test_timed_input_reads_times_out_and_sees_eof_on_a_real_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    try:
        os.write(w, b"abc\n")
        assert P.timed_input("p? ", 1.0) == "abc"
        assert P.timed_input("p? ", 0.01) is None  # nothing typed: the bound expires
        os.close(w)
        with pytest.raises(EOFError):
            P.timed_input("p? ", 1.0)
    finally:
        f.close()


def test_a_line_typed_before_the_prompt_never_answers_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import pty
    import select
    import threading

    master, slave = pty.openpty()
    f = _stdin_from(monkeypatch, slave)
    try:
        os.write(master, b"\n")  # ENTER pressed during the motion, before the prompt appeared
        assert select.select([f], [], [], 2.0)[0]  # it is queued on the tty
        assert P.timed_input("support? ", 0.2) is None  # discarded: no answer
        threading.Timer(0.1, os.write, (master, b"y\n")).start()  # typed after the prompt
        assert P.timed_input("score? ", 5.0) == "y"
        # the support prompt: a queued ENTER does not confirm support
        os.write(master, b"\n")
        assert select.select([f], [], [], 2.0)[0]
        fol = FakeFollower()
        arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
        arm.connect()
        g = P.TorqueGuard(arm, P.TorqueWait(0.2, 0.0), say=lambda _m: None)
        assert g.support("support it, ENTER: ") == P.TIMEOUT  # not ENTER: released unconfirmed
        assert g.released is not None and g.released["cause"] == P.TIMEOUT
        assert not fol.bus.torque_enabled
    finally:
        f.close()
        os.close(master)


def test_operator_signals_make_ctrl_c_a_flag_and_a_hangup_the_closed_stdin_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import signal
    import threading

    monkeypatch.setattr(P, "_detach_stdio", lambda: None)  # keep pytest's capture
    before = signal.getsignal(signal.SIGHUP)
    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    try:
        with P.operator_signals() as flags:
            os.kill(os.getpid(), signal.SIGINT)  # outside a read: only a flag, nothing raised
            assert flags.ctrl_c and flags.hangup is None
            with pytest.raises(KeyboardInterrupt):  # inside a read Ctrl-C still answers
                threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGINT)).start()
                P.timed_input("p? ", 5.0)
            threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGHUP)).start()
            with pytest.raises(P.Hangup):  # the terminal closes during a wait
                P.timed_input("p? ", 5.0)
            assert flags.hangup == "SIGHUP"
            with pytest.raises(P.Hangup):  # every later prompt: nobody can answer
                P.timed_input("p? ", 5.0)
            g, _, fol, said, clock = _guard([], wait=WAIT)
            g.read_line = P.timed_input
            t0 = clock()
            assert g.support("support it, ENTER: ") == P.HANGUP
            assert clock() - t0 == pytest.approx(10.0)  # the grace, without stdin
            assert any("TERMINAL HUNG UP" in m for m in said) and not fol.bus.torque_enabled
            assert g.released is not None and g.released["cause"] == P.HANGUP
        assert signal.getsignal(signal.SIGHUP) == before  # restored
    finally:
        f.close()
        os.close(w)
