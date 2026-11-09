"""src/robot/prompts.py: bounded torque-on prompts and the support prompt, on the fake arm with a
scripted stdin (tests/robot_fakes.FakeReader). No serial port, no terminal."""

from __future__ import annotations

import contextlib
import math
import sys
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


# -- review round 4 (owner repro 2026-11-08): the deadline covers the whole line ----------------
SLACK_S = 0.5  # scheduler slack on a loaded CI runner


def _timed(prompt: str, timeout_s: float) -> tuple[str | None, float]:
    import time

    t0 = time.monotonic()
    line = P.timed_input(prompt, timeout_s)
    return line, time.monotonic() - t0


def test_a_partial_line_in_an_open_pipe_times_out_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner repro: select() reported the pipe readable, readline() then waited for a newline
    that never came, far past the deadline."""
    import os

    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    try:
        os.write(w, b"ab")  # no newline; the writer stays open
        line, elapsed = _timed("support? ", 0.3)
        assert line is None and 0.3 <= elapsed < 0.3 + SLACK_S
        # partial, then the rest of the line within the deadline: the whole line
        import threading

        os.write(w, b"ab")
        threading.Timer(0.1, os.write, (w, b"c\n")).start()
        line, elapsed = _timed("score? ", 5.0)
        assert line == "abc" and elapsed < 5.0  # the timed-out "ab" was dropped, not carried
        # nothing past the newline is consumed: the next line is still in the pipe, intact
        os.write(w, b"y\nn\n")
        assert P.timed_input("p? ", 1.0) == "y" and P.timed_input("p? ", 1.0) == "n"
        # a partial line, then EOF: the EOF path (no confirmation), not a line
        os.write(w, b"partial")
        os.close(w)
        w = -1
        with pytest.raises(EOFError, match="mid-line"):
            P.timed_input("support? ", 5.0)
    finally:
        f.close()
        if w >= 0:
            os.close(w)


def test_ctrl_c_in_mid_line_still_raises_and_a_hangup_still_ends_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import signal
    import threading

    monkeypatch.setattr(P, "_detach_stdio", lambda: None)  # keep pytest's capture
    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    try:
        with P.operator_signals() as flags:
            os.write(w, b"ab")  # the read is mid-line when the signal lands
            threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGINT)).start()
            with pytest.raises(KeyboardInterrupt):
                P.timed_input("p? ", 5.0)
            assert not flags.ctrl_c  # inside a read: KeyboardInterrupt, not the flag
            os.write(w, b"cd")
            threading.Timer(0.1, os.kill, (os.getpid(), signal.SIGHUP)).start()
            with pytest.raises(P.Hangup):
                P.timed_input("p? ", 5.0)
    finally:
        f.close()
        os.close(w)


def test_a_partial_line_on_a_tty_times_out_and_never_answers_the_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On a tty a Ctrl-D in mid-line hands the partial line to read() with no newline: the same
    blocking readline() as the owner's pipe repro. Now a timeout; the next prompt starts clean."""
    import os
    import pty
    import threading

    master, slave = pty.openpty()
    f = _stdin_from(monkeypatch, slave)
    try:
        threading.Timer(0.1, os.write, (master, b"ab\x04")).start()  # typed after the prompt
        line, elapsed = _timed("support? ", 0.5)
        assert line is None and 0.5 <= elapsed < 0.5 + SLACK_S
        threading.Timer(0.1, os.write, (master, b"y\n")).start()
        assert P.timed_input("score? ", 5.0) == "y"  # not "aby": the partial bytes were dropped
        threading.Timer(0.1, os.write, (master, b"ab\x04c\n")).start()  # Ctrl-D, then the rest
        assert P.timed_input("score? ", 5.0) == "abc"
        threading.Timer(0.1, os.write, (master, b"\x04")).start()  # Ctrl-D on an empty line
        with pytest.raises(EOFError):
            P.timed_input("support? ", 5.0)
    finally:
        f.close()
        os.close(master)


def test_no_timeout_reads_the_same_way_a_partial_line_then_eof_never_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F1: with teleop's null timeout the support prompt went through input(), which
    returned 'ab' + Ctrl-D, Ctrl-D as a truncated line: recorded as an ENTER ("supported")."""
    import os
    import pty
    import threading

    master, slave = pty.openpty()
    f = _stdin_from(monkeypatch, slave)
    try:
        threading.Timer(0.1, os.write, (master, b"y\n")).start()
        assert P.timed_input("score? ", None) == "y"  # unbounded: a whole line as before
        threading.Timer(0.1, os.write, (master, b"ab\x04\x04")).start()
        with pytest.raises(EOFError, match="mid-line"):
            P.timed_input("support? ", None)
        fol = FakeFollower()
        arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
        arm.connect()
        g = P.TorqueGuard(arm, P.TorqueWait(None, 0.0), say=lambda _m: None)  # real reader
        threading.Timer(0.1, os.write, (master, b"ab\x04\x04")).start()
        assert g.support("support it, ENTER: ") == P.EOF  # not ENTER: released unconfirmed
        assert g.released is not None and g.released["cause"] == P.EOF
        assert not fol.bus.torque_enabled
    finally:
        f.close()
        os.close(master)
    r, w = os.pipe()  # the same on a pipe: a partial line then EOF is EOF, not an answer
    f = _stdin_from(monkeypatch, r)
    try:
        os.write(w, b"partial")
        os.close(w)
        with pytest.raises(EOFError, match="mid-line"):
            P.timed_input("support? ", None)
    finally:
        f.close()


def test_require_terminal_wants_a_tty_on_stdin_and_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import pty

    master, slave = pty.openpty()
    r, w = os.pipe()
    tty_in = open(slave, "r", closefd=False)  # noqa: SIM115 - closed below
    tty_out = open(slave, "w", closefd=False)  # noqa: SIM115
    pipe_in, pipe_out = open(r, "r"), open(w, "w")  # noqa: SIM115
    try:
        monkeypatch.setattr(P.sys, "stdin", tty_in)
        monkeypatch.setattr(P.sys, "stdout", tty_out)
        assert P.not_a_terminal() is None
        # a tty, but not this process's controlling terminal: not in its foreground (job control)
        with pytest.raises(SystemExit, match="not in its terminal's foreground"):
            P.require_terminal("x")
        monkeypatch.setattr(P.os, "tcgetpgrp", lambda _fd: os.getpgrp())  # the foreground job
        P.require_terminal("x")  # passes
        monkeypatch.setattr(P.os, "tcgetpgrp", lambda _fd: os.getpgrp() + 1)  # `cmd &` / `bg`
        with pytest.raises(SystemExit, match="foreground process group is(.|\n)*nothing was op"):
            P.require_terminal("wo07_pilot --hardware")
        monkeypatch.setattr(P.sys, "stdout", pipe_out)  # e.g. `| tee log`: prompts may be unseen
        assert P.not_a_terminal() == "stdout"
        monkeypatch.setattr(P.sys, "stdin", pipe_in)
        with pytest.raises(SystemExit, match="stdin is not a tty(.|\n)*nothing was opened"):
            P.require_terminal("wo07_pilot --hardware")
        monkeypatch.setattr(P.sys, "stdin", None)
        assert P.not_a_terminal() == "stdin"
    finally:
        for fh in (tty_in, tty_out, pipe_in, pipe_out):
            fh.close()
        os.close(slave)
        os.close(master)


# -- job control (owner decision 2026-11-08): Ctrl-Z ignored, foreground required --------------
# The pty end-to-end tests (a real Ctrl-Z typed on a terminal) are in tests/test_job_control.py.
SESSION_SIGNALS = ("SIGINT", "SIGHUP", "SIGTERM", "SIGTSTP", "SIGTTIN", "SIGTTOU")


def _sentinels() -> dict[Any, Any]:
    """A distinct Python handler on every signal the session touches."""
    import signal

    out: dict[Any, Any] = {}
    for name in SESSION_SIGNALS:
        sig = getattr(signal, name)

        def h(signum: int, frame: Any, name: str = name) -> None:
            raise AssertionError(f"sentinel {name} called")

        out[sig] = h
    return out


@pytest.mark.parametrize("shape", ["operator_signals", "job_control", "nested"])
def test_every_session_signal_handler_is_restored_also_on_an_exception(shape: str) -> None:
    import signal

    sentinels = _sentinels()
    before = {s: signal.signal(s, h) for s, h in sentinels.items()}
    try:
        for fail in (False, True):
            with contextlib.suppress(RuntimeError):
                if shape == "operator_signals":
                    cm: Any = P.operator_signals()
                elif shape == "job_control":
                    cm = P.job_control()
                else:  # teleop: job_control around the arm, operator_signals inside it
                    cm = contextlib.ExitStack()
                    cm.enter_context(P.job_control())
                    cm.enter_context(P.operator_signals())
                with cm:
                    assert P.INTERRUPTS.job_control
                    for name in ("SIGTSTP", "SIGTTIN", "SIGTTOU"):
                        h = signal.getsignal(getattr(signal, name))
                        assert callable(h) and h is not sentinels[getattr(signal, name)]
                    if shape != "job_control":
                        assert signal.getsignal(signal.SIGINT) is not sentinels[signal.SIGINT]
                    if fail:
                        raise RuntimeError("bug inside the session")
            for sig, h in sentinels.items():
                assert signal.getsignal(sig) is h, signal.Signals(sig).name
            assert not P.INTERRUPTS.job_control and P.INTERRUPTS.tty_fd is None
            assert P.INTERRUPTS.on_lost is None
    finally:
        for s, h in before.items():
            signal.signal(s, h)


def test_ctrl_z_is_a_notice_and_a_background_read_or_write_is_a_lost_foreground(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-process: SIGTSTP leaves the session running (the handler is checked to be installed
    before the signal is sent); SIGTTOU (a background write / tcflush) marks the foreground
    lost, calls on_signal like a hang-up and is ignored from then on; every later prompt takes
    the closed-stdin path with cause lost_foreground; stdio is not detached."""
    import os
    import signal

    detached: list[int] = []
    monkeypatch.setattr(P, "_detach_stdio", lambda: detached.append(1))
    seen: list[int] = []
    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    try:
        with P.operator_signals(lambda: seen.append(1)) as flags:
            assert callable(signal.getsignal(signal.SIGTSTP))  # else the kill would stop pytest
            os.kill(os.getpid(), signal.SIGTSTP)
            assert flags.lost_foreground is None and not flags.ctrl_c and not seen
            assert P.timed_input("p? ", 0.05) is None  # prompts still work after a Ctrl-Z
            assert callable(signal.getsignal(signal.SIGTTOU))
            os.kill(os.getpid(), signal.SIGTTOU)
            assert flags.lost_foreground is not None and flags.lost_foreground.startswith("SIGTTOU")
            assert seen == [1] and flags.hangup is None and not detached
            assert signal.getsignal(signal.SIGTTIN) is signal.SIG_IGN  # never signalled again
            assert signal.getsignal(signal.SIGTTOU) is signal.SIG_IGN
            os.write(w, b"y\n")  # nothing typed now can answer: the terminal is not ours
            with pytest.raises(P.LostForeground):
                P.timed_input("p? ", 5.0)
            g, _, fol, said, clock = _guard([], wait=WAIT)
            g.read_line = P.timed_input
            t0 = clock()
            assert g.support("support it, ENTER: ") == P.LOST_FOREGROUND
            assert clock() - t0 == pytest.approx(10.0)  # the grace, without stdin
            assert any("LOST THE TERMINAL'S FOREGROUND" in m for m in said)
            assert g.released is not None and g.released["cause"] == P.LOST_FOREGROUND
            assert not fol.bus.torque_enabled and "foreground" in g.arm.deadman.reason
        assert signal.getsignal(signal.SIGTTOU) is signal.SIG_DFL  # restored, not left ignored
    finally:
        f.close()
        os.close(w)


def test_a_lost_foreground_ends_a_prompt_wait_at_the_next_poll_not_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import time

    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    lost_at = [math.inf]
    monkeypatch.setattr(
        P, "not_foreground", lambda _fd: "bg" if time.monotonic() >= lost_at[0] else None
    )
    try:
        with P.job_control() as flags:
            flags.tty_fd = r  # as with stdin on the terminal
            line, elapsed = _timed("p? ", 0.6)  # still in the foreground: the deadline as before
            assert line is None and 0.6 <= elapsed < 0.6 + SLACK_S
            lost_at[0] = time.monotonic() + 0.3
            t0 = time.monotonic()
            with pytest.raises(P.LostForeground, match="foreground"):
                P.timed_input("support? ", 30.0)
            assert time.monotonic() - t0 < 0.3 + P.FOREGROUND_POLL_S + SLACK_S
            assert flags.lost_foreground == "bg" and flags.check_foreground()
            with pytest.raises(P.LostForeground):  # every later prompt, at once
                P.prompt_input("type 'go': ")
    finally:
        f.close()
        os.close(w)


def test_job_control_refuses_a_session_out_of_the_foreground_before_its_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-check on entry (Ctrl-Z before the session started, then ``bg``): nothing inside
    the ``with`` (the camera, the arm's torque on) runs, and no handler is left installed."""
    import os
    import signal

    from robot_fakes import tty_stdout

    master, slave = os.openpty()
    tty_in = open(slave, "r", closefd=False)  # noqa: SIM115 - closed below
    before = {s: signal.getsignal(s) for s in (signal.SIGTSTP, signal.SIGINT)}
    monkeypatch.setattr(P.sys, "stdin", tty_in)
    monkeypatch.setattr(P.os, "tcgetpgrp", lambda _fd: os.getpgrp() + 1)
    ran: list[int] = []
    try:
        with tty_stdout() as out:
            monkeypatch.setattr(P.sys, "stdout", out)
            refused = pytest.raises(SystemExit, match="not in its terminal's foreground")
            with refused, P.operator_signals():
                ran.append(1)
            monkeypatch.setattr(P.sys, "stdout", sys.__stdout__)
        assert ran == [] and not P.INTERRUPTS.job_control
        assert {s: signal.getsignal(s) for s in before} == before
    finally:
        tty_in.close()
        os.close(slave)
        os.close(master)


def test_a_ctrl_z_inside_a_bounded_wait_keeps_its_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review F2: SIGTSTP from another thread 0.9 s into a 1.5 s wait. The wait ends at the
    original deadline (a reset would end it at about 2.4 s), with the line still unanswered."""
    import os
    import signal
    import threading

    r, w = os.pipe()
    f = _stdin_from(monkeypatch, r)
    timer = threading.Timer(0.9, os.kill, (os.getpid(), signal.SIGTSTP))
    try:
        with P.job_control():
            assert callable(signal.getsignal(signal.SIGTSTP))  # else the kill would stop pytest
            timer.start()
            line, elapsed = _timed("p? ", 1.5)
        assert line is None and 1.5 <= elapsed < 1.5 + SLACK_S, elapsed
    finally:
        timer.cancel()
        f.close()
        os.close(w)


def test_job_control_keeps_a_half_typed_line_across_ctrl_z_and_restores_noflsh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F2: without NOFLSH the terminal throws the half-typed line away on Ctrl-Z, and the
    next ENTER answers with an empty line. ``job_control`` sets NOFLSH on stdin's terminal for
    the session and puts the previous bit back on exit (not while the foreground is lost)."""
    import os
    import termios

    master, slave = os.openpty()
    tty_in = open(slave, "r", closefd=False)  # noqa: SIM115 - closed below
    monkeypatch.setattr(P.sys, "stdin", tty_in)
    fg = [True]
    monkeypatch.setattr(P.os, "tcgetpgrp", lambda _fd: os.getpgrp() + (0 if fg[0] else 1))

    def noflsh() -> bool:
        return bool(termios.tcgetattr(slave)[3] & termios.NOFLSH)

    try:
        assert not noflsh()
        with P.job_control() as flags:
            assert flags.tty_fd == slave and noflsh()
        assert not noflsh()  # restored
        with P.job_control():
            fg[0] = False  # the shell took the terminal back: its modes are not ours to set
        assert noflsh()
        fg[0] = True
        with P.job_control():  # already set before the session: left set
            assert noflsh()
        assert noflsh()
    finally:
        tty_in.close()
        os.close(slave)
        os.close(master)
