"""Operator prompts while the arm holds under torque: bounded waits, honest support answers.

Shared by the teleop recorder (scripts/wo06_teleop_record.py), the WO-07 pilot
(scripts/wo07_pilot.py) and the real eval (scripts/wo07_real_eval.py) through
``src/robot/client.py``. Owner decisions of 2026-11-08:

* Bounded torque-on wait (``TorqueWait``: ``<section>.prompt_torque_timeout_s``, default 60 s,
  and ``<section>.prompt_support_grace_s``, default 10 s). Every prompt answered while the arm
  holds under torque reads servo health (temperature, current) before the wait, waits at most
  the timeout, and reads it again after the answer (``TorqueGuard.ask``). With no answer in
  time: ``TORQUE_WAIT_EXPIRED`` ("support the arm now"), the grace period (ENTER = now), a
  deadman trip with TORQUE_OFF, torque off (``TorqueGuard.release``). Nothing after that holds,
  parks or moves. The pilot and the real eval refuse a null or non-positive timeout on the
  real arm (``allow_none=False``); teleop keeps ``null`` = wait forever (the grace period still
  applies to a closed stdin).
* Only an explicit ENTER confirms support (``TorqueGuard.support``). Ctrl-C at the support
  prompt does not confirm and does not drop the arm: it is ignored with a message, the arm keeps
  holding, the bounded wait keeps running (its deadline is not reset). A closed stdin (EOF) or a
  hung-up terminal (SIGHUP / SIGTERM: ``hangup``): no one can confirm; a loud message, the grace
  period without stdin, then torque off. A timeout: as above. Callers record those ends as
  unconfirmed support (``released_unconfirmed`` with the cause), never as "supported".
* Every reply says how the prompt ended: ``enter`` (a line was typed), ``ctrl_c``, ``eof``,
  ``hangup`` or ``timeout`` (``Reply.how``), so acknowledgements are recorded with how they were
  given.
* Typed-ahead input is discarded: a line typed before a prompt appeared (an ENTER pressed during
  the motion, a double ENTER at the previous prompt) never answers it (``discard_typeahead``,
  ``termios.tcflush`` on a tty, before every prompt).
* Signals (``operator_signals``, installed by the hardware sessions): Ctrl-C (SIGINT) outside a
  prompt only sets ``INTERRUPTS.ctrl_c``, so it never lands inside a servo-bus transaction, a
  grace sleep or the shutdown; the motion loops check it between ticks. The prompt reads switch
  SIGINT back to KeyboardInterrupt for the read itself. SIGHUP / SIGTERM (the Terminal window
  closed, the ssh session dropped) set ``INTERRUPTS.hangup``, detach stdout / stderr (the tty
  is gone), stop motion at the next tick, and make every later prompt end as ``hangup`` (the
  closed-stdin path above), so torque goes off after the grace period instead of the process
  dying with the arm held under torque.

The wait is ``select()`` on stdin in the calling thread (``timed_input``): no reader thread, and
the servo bus keeps a single owner (the health reads happen in this thread, before and after the
wait, never during it).
"""

from __future__ import annotations

import contextlib
import math
import os
import select
import signal
import sys
import termios
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import FrameType
from typing import Any, TypeVar

from src.robot.driver import So101Driver, release_port
from src.robot.safety import DeadmanResponse

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_GRACE_S = 10.0

LineReader = Callable[[str, float | None], str | None]
"""(prompt, timeout_s or None = no bound) -> the line typed (no newline), or None on a timeout.
Raises KeyboardInterrupt on Ctrl-C and EOFError on a closed stdin (``Hangup`` on a hang-up)."""

ENTER, CTRL_C, EOF, HANGUP, TIMEOUT = "enter", "ctrl_c", "eof", "hangup", "timeout"
STDIN_GONE = (EOF, HANGUP)
"""Prompt ends where nobody can answer any more: the closed-stdin path."""

TORQUE_WAIT_EXPIRED = (
    "\nNo answer for {timeout:.0f} s with torque ON. SUPPORT THE ARM NOW: put a hand under the "
    "forearm. Torque goes OFF in {grace:.0f} s (ENTER = now) and the session ends."
)
STDIN_CLOSED = (
    "\n!!! {what} at a prompt with torque ON: nobody can confirm that the arm is "
    "supported. SUPPORT THE ARM NOW: put a hand under the forearm. Torque goes OFF in "
    "{grace:.0f} s and the session ends. !!!"
)
CTRL_C_IGNORED = (
    "\nCtrl-C ignored: the arm still holds under torque. Support it, then press ENTER (or "
    "switch off the servo supply)."
)
CTRL_C_IGNORED_GRACE = "\nCtrl-C ignored: torque goes OFF when the grace period ends (ENTER = now)."

T = TypeVar("T")


class Hangup(EOFError):
    """The operator's terminal went away (SIGHUP / SIGTERM): handled as a closed stdin."""


class Interrupts:
    """Process-wide signal flags of a hardware session (``operator_signals``)."""

    def __init__(self) -> None:
        self.ctrl_c = False  # SIGINT outside a prompt read
        self.hangup: str | None = None  # "SIGHUP" | "SIGTERM" once received; never cleared

    def reset(self) -> None:
        self.ctrl_c, self.hangup = False, None


INTERRUPTS = Interrupts()


def _detach_stdio() -> None:
    """After a hang-up the tty is gone: writes to it would raise (EIO) inside the shutdown."""
    try:
        null = open(os.devnull, "w")  # noqa: SIM115 - lives for the rest of the process
    except OSError:
        return
    sys.stdout = sys.stderr = null


def _on_hangup(signum: int, frame: FrameType | None) -> None:
    INTERRUPTS.hangup = signal.Signals(signum).name
    _detach_stdio()


def _raise_hangup(signum: int, frame: FrameType | None) -> None:
    _on_hangup(signum, frame)
    raise Hangup(f"{signal.Signals(signum).name}: the operator's terminal closed")


HANGUP_SIGNALS = (signal.SIGHUP, signal.SIGTERM)


@contextlib.contextmanager
def operator_signals(on_signal: Callable[[], None] | None = None) -> Iterator[Interrupts]:
    """For a real-arm session (module doc): SIGINT only sets ``INTERRUPTS.ctrl_c``; SIGHUP and
    SIGTERM set ``INTERRUPTS.hangup`` and detach stdio. ``on_signal()`` runs after either (teleop
    sets its loop-stop flag). The previous handlers are restored on exit."""
    INTERRUPTS.reset()

    def on_sigint(signum: int, frame: FrameType | None) -> None:
        INTERRUPTS.ctrl_c = True
        if on_signal is not None:
            on_signal()

    def on_hup(signum: int, frame: FrameType | None) -> None:
        _on_hangup(signum, frame)
        if on_signal is not None:
            on_signal()

    prev: dict[signal.Signals, Any] = {signal.SIGINT: signal.signal(signal.SIGINT, on_sigint)}
    for s in HANGUP_SIGNALS:
        prev[s] = signal.signal(s, on_hup)
    try:
        yield INTERRUPTS
    finally:
        for s, h in prev.items():
            signal.signal(s, h)


@contextlib.contextmanager
def _reading() -> Iterator[None]:
    """During a prompt read only: Ctrl-C raises KeyboardInterrupt, a hang-up raises ``Hangup``."""
    if INTERRUPTS.hangup is not None:
        raise Hangup(f"{INTERRUPTS.hangup}: the operator's terminal closed")
    prev: dict[signal.Signals, Any] = {
        signal.SIGINT: signal.signal(signal.SIGINT, signal.default_int_handler)
    }
    for s in HANGUP_SIGNALS:
        prev[s] = signal.signal(s, _raise_hangup)
    try:
        yield
    finally:
        for s, h in prev.items():
            signal.signal(s, h)


def discard_typeahead() -> bool:
    """Drop input typed before the prompt appeared (a tty's queued lines): only an answer typed
    after the prompt is shown can answer it. True if stdin is a tty and was flushed."""
    try:
        if not sys.stdin.isatty():
            return False
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return True
    except (OSError, ValueError, AttributeError, termios.error):
        return False


def _write_prompt(prompt: str) -> None:
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    except OSError as e:  # the terminal is gone: nobody can answer
        raise EOFError(f"stdout: {e}") from e


def prompt_input(prompt: str) -> str:
    """``input()`` with Ctrl-C raising KeyboardInterrupt, also under ``operator_signals``; typed-
    ahead input discarded first."""
    with _reading():
        discard_typeahead()
        try:
            return input(prompt)
        except OSError as e:  # EIO from a hung-up tty
            raise EOFError(f"stdin: {e}") from e


def timed_input(prompt: str, timeout_s: float | None) -> str | None:
    """``prompt_input`` that gives up after ``timeout_s`` (None = never): None. ``select()`` on
    stdin in this thread (no reader thread; the servo bus is never touched here). Typed-ahead
    input is discarded first. EOF raises EOFError (``Hangup`` on a hang-up)."""
    if timeout_s is None:
        return prompt_input(prompt)
    with _reading():
        discard_typeahead()
        _write_prompt(prompt)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout_s))
            if not ready:
                _write_prompt("\n")
                return None
            line = sys.stdin.readline()
        except OSError as e:  # EIO from a hung-up tty
            raise EOFError(f"stdin: {e}") from e
        if not line:
            raise EOFError
        return line.rstrip("\r\n")


@dataclass(frozen=True)
class TorqueWait:
    """Bound on a prompt answered with the arm holding under torque (module doc).
    ``timeout_s`` None = no bound (teleop only); the grace period applies either way."""

    timeout_s: float | None  # <section>.prompt_torque_timeout_s
    grace_s: float = DEFAULT_GRACE_S  # <section>.prompt_support_grace_s

    def __post_init__(self) -> None:
        t = self.timeout_s
        ok = t is None or (math.isfinite(t) and t > 0)
        if not (ok and math.isfinite(self.grace_s) and self.grace_s >= 0):
            raise ValueError(
                f"prompt_torque_timeout_s must be > 0 and prompt_support_grace_s >= 0, got "
                f"{self.timeout_s} / {self.grace_s}"
            )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], section: str, *, allow_none: bool) -> TorqueWait:
        """``cfg[section].prompt_torque_timeout_s`` / ``prompt_support_grace_s``. A null timeout
        (wait forever) is allowed only with ``allow_none``; otherwise, and for any non-positive
        or non-finite value, ValueError. The grace period is always read."""
        s = cfg.get(section) or {}
        timeout = s.get("prompt_torque_timeout_s", DEFAULT_TIMEOUT_S)
        if timeout is None and not allow_none:
            raise ValueError(
                f"{section}.prompt_torque_timeout_s is null: a real-arm session needs a bounded "
                "wait at every prompt answered with torque on (default 60 s)"
            )
        grace = s.get("prompt_support_grace_s", DEFAULT_GRACE_S)
        return cls(
            None if timeout is None else float(timeout),
            float(DEFAULT_GRACE_S if grace is None else grace),
        )

    def record(self) -> dict[str, float | None]:
        return {"timeout_s": self.timeout_s, "grace_s": self.grace_s}


@dataclass(frozen=True)
class Reply:
    """How a prompt ended: ``enter`` (``text`` = the line typed), ``ctrl_c``, ``eof``, ``hangup``,
    ``timeout``."""

    how: str
    text: str = ""

    @property
    def answer(self) -> str:
        return self.text.strip().lower()


def ack_record(prompt: str, how: str, **extra: Any) -> dict[str, Any]:
    """An acknowledgement wait, for a record: ``given`` only on an explicit ENTER."""
    return {"prompt": prompt.strip(), "given": how == ENTER, "how": how, **extra}


def _say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except OSError:
        pass  # the terminal is gone; the record still says what happened


def ignoring_ctrl_c(fn: Callable[[], T], arm: So101Driver, say: Callable[[str], None]) -> T:
    """Run a bus call to completion: a Ctrl-C that lands inside it (no ``operator_signals``
    handler installed) is ignored with a message; the port is released and the call retried. A
    panicked press must not drop the arm or skip the shutdown."""
    while True:
        try:
            return fn()
        except KeyboardInterrupt:
            say(CTRL_C_IGNORED)
            release_port(arm.follower.bus)


class TorqueGuard:
    """The prompts of one session on one arm (module doc). ``wait`` None = no bound and the
    default grace period. ``released`` is set once a prompt's timeout or a closed stdin released
    torque: {cause, prompt, timeout_s, grace_s, torque_off_confirmed}."""

    def __init__(
        self,
        arm: So101Driver,
        wait: TorqueWait | None,
        *,
        read: LineReader = timed_input,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        say: Callable[[str], None] = _say,
    ):
        self.arm = arm
        self.wait = TorqueWait(None) if wait is None else wait
        self.read_line, self.clock, self.sleep, self.say = read, clock, sleep, say
        self.released: dict[str, Any] | None = None

    def torque_held(self) -> bool:
        """The arm holds under torque as far as this process knows (no torque-off latch)."""
        arm = self.arm
        if arm.torque_off_unconfirmed is not None:
            return False
        return not (arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF)

    def read(self, prompt: str, timeout_s: float | None) -> Reply:
        """One read; never raises for Ctrl-C, EOF, a hang-up or a timeout."""
        try:
            line = self.read_line(prompt, timeout_s)
        except KeyboardInterrupt:
            return Reply(CTRL_C)
        except EOFError as e:
            gone = isinstance(e, Hangup) or INTERRUPTS.hangup is not None
            return Reply(HANGUP if gone else EOF)
        return Reply(TIMEOUT) if line is None else Reply(ENTER, line)

    def _check_servos(self) -> None:
        ignoring_ctrl_c(self.arm.check_servos, self.arm, self.say)

    def ask(self, prompt: str) -> Reply:
        """A prompt answered with torque on: servo health before the wait, the wait bounded by
        ``wait.timeout_s``, servo health after the answer. A timeout releases torque (``release``)
        and returns ``timeout``. Ctrl-C and EOF / hang-up are returned for the caller to act on.
        With torque already off it is a plain unbounded read. Raises ServoFault / SerialFault from
        the health reads (torque is then off)."""
        if not self.torque_held():
            return self.read(prompt, None)
        self._check_servos()  # heat from the hold so far
        reply = self.read(prompt, self.wait.timeout_s)
        if reply.how == TIMEOUT:
            self.release(TIMEOUT, prompt)
            return reply
        self._check_servos()  # heat built up while the arm held at the prompt
        return reply

    def acknowledge(self, prompt: str) -> str:
        """An acknowledgement with torque already off (or not confirmed off): no servo read, no
        release. Bounded by ``wait.timeout_s`` when set, so the session record is always written.
        Returns how it ended; only ``enter`` is an acknowledgement."""
        reply = self.read(prompt, self.wait.timeout_s)
        if reply.how == TIMEOUT:
            self.say("no acknowledgement in time: recorded as NOT acknowledged")
        elif reply.how != ENTER:
            self.say(f"\n{reply.how}: recorded as NOT acknowledged")
        return reply.how

    def support(self, prompt: str) -> str:
        """The final "support the arm, then ENTER" prompt with torque on. Returns ``enter`` (the
        operator confirmed; the caller turns torque off), or ``eof`` / ``hangup`` / ``timeout``
        (torque already released here, after the grace period: the support was NOT confirmed).
        Ctrl-C is ignored with a message and the bounded wait keeps running, also when it lands
        in the servo read. Servo health is read once before the wait (raises ServoFault /
        SerialFault; torque is then off)."""
        if not self.torque_held():
            return self.acknowledge(prompt)
        timeout = self.wait.timeout_s
        deadline = None if timeout is None else self.clock() + timeout
        self._check_servos()  # heat from the hold so far, before the support wait
        while True:
            remaining = None if deadline is None else deadline - self.clock()
            if remaining is not None and remaining <= 0:
                reply = Reply(TIMEOUT)
            else:
                reply = self.read(prompt, remaining)
            if reply.how == CTRL_C:
                self.say(CTRL_C_IGNORED)  # a panicked second press must not drop the arm
                continue
            if reply.how == ENTER:
                return ENTER
            self.release(reply.how, prompt)
            return reply.how

    def release(self, cause: str, prompt: str) -> None:
        """Nobody answered (``timeout``) or nobody can (``eof`` / ``hangup``): tell the operator to
        support the arm, wait the grace period (ENTER = now; Ctrl-C ignored; without stdin after
        an EOF or a hang-up), trip the deadman TORQUE_OFF and turn torque off."""
        grace = self.wait.grace_s
        if cause in STDIN_GONE:
            what = "TERMINAL HUNG UP" if cause == HANGUP else "STDIN CLOSED"
            self.say(STDIN_CLOSED.format(what=what, grace=grace))
            self._sleep_until(self.clock() + grace)
        else:
            timeout = self.wait.timeout_s or 0.0
            self.say(TORQUE_WAIT_EXPIRED.format(timeout=timeout, grace=grace))
            if grace > 0:
                self._grace(grace)
        what = {EOF: "stdin closed", HANGUP: "terminal hung up"}.get(cause, "no answer in time")
        self.arm.deadman.trip(
            f"{what} at a prompt with torque on ({prompt.strip()[:80]!r})",
            DeadmanResponse.TORQUE_OFF,
        )
        confirmed = ignoring_ctrl_c(self.arm.torque_off, self.arm, self.say)
        self.released = {
            "cause": cause,
            "prompt": prompt.strip(),
            "timeout_s": self.wait.timeout_s,
            "grace_s": grace,
            "torque_off_confirmed": confirmed,
        }
        if confirmed:
            self.say("torque OFF: the arm is limp (support was NOT confirmed)")

    def _sleep_until(self, deadline: float) -> None:
        """The grace period without stdin: a Ctrl-C does not shorten it (the arm keeps holding)."""
        while (remaining := deadline - self.clock()) > 0:
            try:
                self.sleep(remaining)
            except KeyboardInterrupt:
                self.say(CTRL_C_IGNORED_GRACE)

    def _grace(self, grace_s: float) -> None:
        deadline = self.clock() + grace_s
        while (remaining := deadline - self.clock()) > 0:
            reply = self.read("", remaining)
            if reply.how == CTRL_C:
                self.say(CTRL_C_IGNORED_GRACE)
                continue
            if reply.how in STDIN_GONE:
                self._sleep_until(deadline)
            return  # ENTER (= now), the grace ran out, or stdin closed and the rest was waited
