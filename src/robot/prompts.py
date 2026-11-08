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
  real arm (``allow_none=False``); teleop keeps ``null`` = wait forever.
* Only an explicit ENTER confirms support (``TorqueGuard.support``). Ctrl-C at the support
  prompt does not confirm and does not drop the arm: it is ignored with a message, the arm keeps
  holding, the bounded wait keeps running (its deadline is not reset). A closed stdin (EOF): no
  one can confirm; a loud message, the grace period without stdin, then torque off. A timeout:
  as above. Callers record those ends as unconfirmed support (``released_unconfirmed`` with the
  cause), never as "supported".
* Every reply says how the prompt ended: ``enter`` (a line was typed), ``ctrl_c``, ``eof`` or
  ``timeout`` (``Reply.how``), so acknowledgements are recorded with how they were given.

The wait is ``select()`` on stdin in the calling thread (``timed_input``): no reader thread, and
the servo bus keeps a single owner (the health reads happen in this thread, before and after the
wait, never during it).
"""

from __future__ import annotations

import math
import select
import signal
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.robot.driver import So101Driver
from src.robot.safety import DeadmanResponse

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_GRACE_S = 10.0

LineReader = Callable[[str, float | None], str | None]
"""(prompt, timeout_s or None = no bound) -> the line typed (no newline), or None on a timeout.
Raises KeyboardInterrupt on Ctrl-C and EOFError on a closed stdin."""

ENTER, CTRL_C, EOF, TIMEOUT = "enter", "ctrl_c", "eof", "timeout"

TORQUE_WAIT_EXPIRED = (
    "\nNo answer for {timeout:.0f} s with torque ON. SUPPORT THE ARM NOW: put a hand under the "
    "forearm. Torque goes OFF in {grace:.0f} s (ENTER = now) and the session ends."
)
STDIN_CLOSED = (
    "\n!!! STDIN CLOSED at a prompt with torque ON: nobody can confirm that the arm is "
    "supported. SUPPORT THE ARM NOW: put a hand under the forearm. Torque goes OFF in "
    "{grace:.0f} s and the session ends. !!!"
)
CTRL_C_IGNORED = (
    "\nCtrl-C ignored: the arm still holds under torque. Support it, then press ENTER (or "
    "switch off the servo supply)."
)
CTRL_C_IGNORED_GRACE = "\nCtrl-C ignored: torque goes OFF when the grace period ends (ENTER = now)."


def prompt_input(prompt: str) -> str:
    """``input()`` with Ctrl-C raising KeyboardInterrupt, also under a flag-only SIGINT handler."""
    prev = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return input(prompt)
    finally:
        signal.signal(signal.SIGINT, prev)


def timed_input(prompt: str, timeout_s: float | None) -> str | None:
    """``prompt_input`` that gives up after ``timeout_s`` (None = never): None. ``select()`` on
    stdin in this thread (no reader thread; the servo bus is never touched here). EOF raises
    EOFError."""
    if timeout_s is None:
        return prompt_input(prompt)
    prev = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout_s))
        if not ready:
            sys.stdout.write("\n")
            return None
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\r\n")
    finally:
        signal.signal(signal.SIGINT, prev)


@dataclass(frozen=True)
class TorqueWait:
    """Bound on a prompt answered with the arm holding under torque (module doc)."""

    timeout_s: float  # <section>.prompt_torque_timeout_s
    grace_s: float  # <section>.prompt_support_grace_s

    def __post_init__(self) -> None:
        ok = math.isfinite(self.timeout_s) and self.timeout_s > 0
        if not (ok and math.isfinite(self.grace_s) and self.grace_s >= 0):
            raise ValueError(
                f"prompt_torque_timeout_s must be > 0 and prompt_support_grace_s >= 0, got "
                f"{self.timeout_s} / {self.grace_s}"
            )

    @classmethod
    def from_config(
        cls, cfg: Mapping[str, Any], section: str, *, allow_none: bool
    ) -> TorqueWait | None:
        """``cfg[section].prompt_torque_timeout_s`` / ``prompt_support_grace_s``. A null timeout is
        None (wait forever) only with ``allow_none``; otherwise, and for any non-positive or
        non-finite value, ValueError."""
        s = cfg.get(section) or {}
        timeout = s.get("prompt_torque_timeout_s", DEFAULT_TIMEOUT_S)
        if timeout is None:
            if allow_none:
                return None
            raise ValueError(
                f"{section}.prompt_torque_timeout_s is null: a real-arm session needs a bounded "
                "wait at every prompt answered with torque on (default 60 s)"
            )
        grace = s.get("prompt_support_grace_s", DEFAULT_GRACE_S)
        return cls(float(timeout), float(DEFAULT_GRACE_S if grace is None else grace))

    def record(self) -> dict[str, float]:
        return {"timeout_s": self.timeout_s, "grace_s": self.grace_s}


@dataclass(frozen=True)
class Reply:
    """How a prompt ended: ``enter`` (``text`` = the line typed), ``ctrl_c``, ``eof``, ``timeout``."""

    how: str
    text: str = ""

    @property
    def answer(self) -> str:
        return self.text.strip().lower()


def ack_record(prompt: str, how: str, **extra: Any) -> dict[str, Any]:
    """An acknowledgement wait, for a record: ``given`` only on an explicit ENTER."""
    return {"prompt": prompt.strip(), "given": how == ENTER, "how": how, **extra}


def _say(msg: str) -> None:
    print(msg, flush=True)


class TorqueGuard:
    """The prompts of one session on one arm (module doc). ``wait`` None = no bound (teleop with
    ``prompt_torque_timeout_s: null``). ``released`` is set once a prompt's timeout or a closed
    stdin released torque: {cause, prompt, timeout_s, grace_s, torque_off_confirmed}."""

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
        self.arm, self.wait = arm, wait
        self.read_line, self.clock, self.sleep, self.say = read, clock, sleep, say
        self.released: dict[str, Any] | None = None

    def torque_held(self) -> bool:
        """The arm holds under torque as far as this process knows (no torque-off latch)."""
        arm = self.arm
        if arm.torque_off_unconfirmed is not None:
            return False
        return not (arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF)

    def read(self, prompt: str, timeout_s: float | None) -> Reply:
        """One read; never raises for Ctrl-C, EOF or a timeout."""
        try:
            line = self.read_line(prompt, timeout_s)
        except KeyboardInterrupt:
            return Reply(CTRL_C)
        except EOFError:
            return Reply(EOF)
        return Reply(TIMEOUT) if line is None else Reply(ENTER, line)

    def ask(self, prompt: str) -> Reply:
        """A prompt answered with torque on: servo health before the wait, the wait bounded by
        ``wait.timeout_s``, servo health after the answer. A timeout releases torque (``release``)
        and returns ``timeout``. Ctrl-C and EOF are returned for the caller to act on. With
        torque already off it is a plain unbounded read. Raises ServoFault / SerialFault from the
        health reads (torque is then off)."""
        if not self.torque_held():
            return self.read(prompt, None)
        self.arm.check_servos()  # heat from the hold so far
        reply = self.read(prompt, None if self.wait is None else self.wait.timeout_s)
        if reply.how == TIMEOUT:
            self.release(TIMEOUT, prompt)
            return reply
        self.arm.check_servos()  # heat built up while the arm held at the prompt
        return reply

    def acknowledge(self, prompt: str) -> str:
        """An acknowledgement with torque already off (or not confirmed off): no servo read, no
        release. Bounded by ``wait.timeout_s`` when set, so the session record is always written.
        Returns how it ended; only ``enter`` is an acknowledgement."""
        reply = self.read(prompt, None if self.wait is None else self.wait.timeout_s)
        if reply.how == TIMEOUT:
            self.say("no acknowledgement in time: recorded as NOT acknowledged")
        elif reply.how != ENTER:
            self.say(f"\n{reply.how}: recorded as NOT acknowledged")
        return reply.how

    def support(self, prompt: str) -> str:
        """The final "support the arm, then ENTER" prompt with torque on. Returns ``enter`` (the
        operator confirmed; the caller turns torque off), or ``eof`` / ``timeout`` (torque already
        released here, after the grace period: the support was NOT confirmed). Ctrl-C is ignored
        with a message and the bounded wait keeps running. Servo health is read once before the
        wait (raises ServoFault / SerialFault; torque is then off)."""
        if not self.torque_held():
            return self.acknowledge(prompt)
        self.arm.check_servos()  # heat from the hold so far, before the support wait
        deadline = None if self.wait is None else self.clock() + self.wait.timeout_s
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
        """Nobody answered (``timeout``) or nobody can (``eof``): tell the operator to support
        the arm, wait the grace period (ENTER = now; Ctrl-C ignored; after an EOF without
        stdin), trip the deadman TORQUE_OFF and turn torque off."""
        grace = 0.0 if self.wait is None else self.wait.grace_s
        if cause == EOF:
            self.say(STDIN_CLOSED.format(grace=grace))
            if grace > 0:
                self.sleep(grace)
        else:
            timeout = 0.0 if self.wait is None else self.wait.timeout_s
            self.say(TORQUE_WAIT_EXPIRED.format(timeout=timeout, grace=grace))
            if grace > 0:
                self._grace(grace)
        what = "stdin closed" if cause == EOF else "no answer in time"
        self.arm.deadman.trip(
            f"{what} at a prompt with torque on ({prompt.strip()[:80]!r})",
            DeadmanResponse.TORQUE_OFF,
        )
        confirmed = self.arm.torque_off()
        self.released = {
            "cause": cause,
            "prompt": prompt.strip(),
            "timeout_s": None if self.wait is None else self.wait.timeout_s,
            "grace_s": grace,
            "torque_off_confirmed": confirmed,
        }
        if confirmed:
            self.say("torque OFF: the arm is limp (support was NOT confirmed)")

    def _grace(self, grace_s: float) -> None:
        deadline = self.clock() + grace_s
        while (remaining := deadline - self.clock()) > 0:
            reply = self.read("", remaining)
            if reply.how == CTRL_C:
                self.say(CTRL_C_IGNORED_GRACE)
                continue
            if reply.how == EOF:
                self.sleep(max(0.0, deadline - self.clock()))
            return  # ENTER (= now), the grace ran out, or stdin closed and the rest was waited
