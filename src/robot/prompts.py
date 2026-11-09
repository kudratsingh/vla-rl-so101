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
* Job control (owner decision 2026-11-08, ``job_control``, entered by ``operator_signals`` and,
  in teleop, around the whole torque-on block): Ctrl-Z (SIGTSTP) never suspends a hardware
  session. It prints ``CTRL_Z_IGNORED`` and does nothing else: motion and prompts continue, a
  bounded wait keeps its deadline. The scope is the whole hardware session, from before torque
  is enabled to after the final torque-off, torque-off windows inside it included (a session
  stopped there still holds the port, has not written its record, and ``bg`` would resume it
  in the background). A session needs the terminal's foreground: ``require_terminal`` refuses
  a process that is not in its terminal's foreground process group (started with ``&``,
  resumed with ``bg``) before any port is opened, and ``job_control`` checks again on entry.
  Losing the foreground during the session (SIGTTIN / SIGTTOU, sent only to a background
  process group that reads or writes the terminal; an ``os.tcgetpgrp`` that no longer names
  this process group, polled at every prompt read, every 0.25 s inside a wait and every motion
  tick; an EIO read from the background) sets ``INTERRUPTS.lost_foreground`` and goes down the
  hang-up path above: motion stops at the next tick (fault kind ``lost_foreground``), every
  later prompt ends as ``lost_foreground`` (grace period without stdin, torque off,
  ``released_unconfirmed`` with cause ``lost_foreground``), stdio is NOT detached (the terminal
  still exists). SIGTTIN / SIGTTOU are ignored after the first one, so a retried background
  read fails with EIO and a background write goes through instead of signalling again: the
  kernel never stops the process. Every handler touched (SIGINT, SIGHUP, SIGTERM, SIGTSTP,
  SIGTTIN, SIGTTOU) is restored when the session ends, also on an exception. SIGSTOP (``kill
  -STOP``) cannot be caught: the inline power switch remains the hard stop.

The wait is ``select()`` on stdin's file descriptor in the calling thread (``timed_input``): no
reader thread, and the servo bus keeps a single owner (the health reads happen in this thread,
before and after the wait, never during it). The deadline covers the whole line (owner repro of
2026-11-08: ``select()`` then ``sys.stdin.readline()`` blocked past the deadline on a partial line
in an open pipe; on a tty a Ctrl-D in mid-line hands over a partial line the same way): fd 0 is
read with ``os.read``, each read only after ``select()`` with the time left, until the newline,
the deadline or EOF. A pipe or file is read one byte at a time, so nothing past the newline is
ever consumed; a tty in canonical mode is read up to a whole line per read (the kernel returns at
most one line, ending at the newline or a Ctrl-D, so a mid-line Ctrl-D is a partial line, not
EOF). Either way no input is held in a private buffer between prompts. A partial line at the
deadline is a timeout and its bytes are dropped (on a tty the next prompt's ``tcflush`` drops
anything typed before it appears); a partial line then EOF is EOF. ``sys.stdin``'s own buffered
reader is
not used by ``timed_input``, also with no deadline (teleop's null timeout: the same reader, no
bound, so a partial line then EOF is EOF, not a truncated "confirmation"). The remaining line
reads (``prompt_input``, torque-off prompts only; ``input()`` for "type 'go'" and the resume
check) run with torque off and read a tty in canonical mode, where one read returns at most one
line, so they never leave bytes buffered either.

The hardware entry points (pilot, real eval, teleop) refuse to start unless stdin and stdout are
a terminal and the process is in that terminal's foreground (``require_terminal``), before any
port is opened: the prompts need an operator who can see them and answer them. Mock / scripted
runs need no terminal.
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
LOST_FOREGROUND = "lost_foreground"
TERMINAL_GONE = (HANGUP, LOST_FOREGROUND)
"""The operator's terminal is gone (hang-up) or no longer this session's (lost foreground)."""
STDIN_GONE = (EOF, *TERMINAL_GONE)
"""Prompt ends where nobody can answer any more: the closed-stdin path."""
FOREGROUND_POLL_S = 0.25
"""How often a prompt wait on the terminal re-checks that the session still has its foreground."""

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
CTRL_Z_IGNORED = (
    "\nCtrl-Z ignored: the arm session is running (torque may be on); use Ctrl-C to stop motion. "
    "An open prompt is still waiting for its answer.\n"
)

T = TypeVar("T")


class Hangup(EOFError):
    """The operator's terminal went away (SIGHUP / SIGTERM): handled as a closed stdin."""

    how = HANGUP


class LostForeground(Hangup):
    """The session's process group lost the terminal's foreground: handled as a hang-up."""

    how = LOST_FOREGROUND


def gone_error(how: str, reason: str) -> Hangup:
    """The exception for a terminal that is gone (``TERMINAL_GONE``): ``Hangup`` or
    ``LostForeground``."""
    return LostForeground(reason) if how == LOST_FOREGROUND else Hangup(reason)


def not_foreground(fd: int) -> str | None:
    """Why this process is not in the foreground process group of the terminal on ``fd``, or
    None when it is. A terminal that is not this process's controlling terminal, or is gone,
    fails ``tcgetpgrp`` and counts as not in the foreground."""
    try:
        fg = os.tcgetpgrp(fd)
    except OSError as e:
        return f"tcgetpgrp: {e.strerror or e}"
    pgrp = os.getpgrp()
    if fg == pgrp:
        return None
    return f"the terminal's foreground process group is {fg}, this process's is {pgrp}"


class Interrupts:
    """Process-wide signal flags of a hardware session (``operator_signals``)."""

    def __init__(self) -> None:
        self.ctrl_c = False  # SIGINT outside a prompt read
        self.hangup: str | None = None  # "SIGHUP" | "SIGTERM" once received; never cleared
        # why the session lost its terminal's foreground, once it did; never cleared
        self.lost_foreground: str | None = None
        self.tty_fd: int | None = None  # the terminal ``job_control`` watches (None: no tty)
        self.on_lost: Callable[[], None] | None = None  # operator_signals' on_signal
        self.job_control = False  # inside ``job_control``

    def reset(self) -> None:
        self.ctrl_c, self.hangup, self.lost_foreground = False, None, None

    def lose_foreground(self, why: str) -> None:
        """Record the loss of the terminal's foreground (once) and tell the session
        (``on_lost``, like a hang-up)."""
        if self.lost_foreground is None:
            self.lost_foreground = why
            if self.on_lost is not None:
                self.on_lost()

    def check_foreground(self) -> bool:
        """True once the session lost its terminal's foreground; polls ``tcgetpgrp`` on the
        watched terminal (``job_control`` with a tty only; cheap: one ioctl)."""
        if self.lost_foreground is None and self.tty_fd is not None:
            why = not_foreground(self.tty_fd)
            if why is not None:
                self.lose_foreground(why)
        return self.lost_foreground is not None

    def terminal_gone(self) -> tuple[str, str] | None:
        """(how, why) once the operator's terminal is gone: a hang-up first (the tty is gone,
        so its foreground is too), then a lost foreground. None while it is there."""
        if self.hangup is not None:
            return HANGUP, f"{self.hangup}: the operator's terminal closed"
        if self.check_foreground():
            return LOST_FOREGROUND, f"lost the terminal's foreground ({self.lost_foreground})"
        return None


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


def _tty_fd(stream: Any) -> int | None:
    try:
        return int(stream.fileno()) if stream is not None and stream.isatty() else None
    except (OSError, ValueError, AttributeError):
        return None


@contextlib.contextmanager
def job_control() -> Iterator[Interrupts]:
    """For the whole of a real-arm session (module doc), from before torque is enabled to after
    the final torque-off: SIGTSTP (Ctrl-Z) prints ``CTRL_Z_IGNORED`` and nothing else; SIGTTIN /
    SIGTTOU mark the loss of the foreground (``Interrupts.lose_foreground``) and are ignored from
    then on, so neither can stop the process. With stdin on a terminal it is watched for the
    foreground (``Interrupts.check_foreground``), and a process already out of the foreground
    exits here (``NOT_FOREGROUND``) before anything inside the ``with`` is opened. Nested entries
    (teleop's ``operator_signals`` inside its own ``job_control``) change nothing. The previous
    handlers are restored on exit, also on an exception."""
    if INTERRUPTS.job_control:
        yield INTERRUPTS
        return
    INTERRUPTS.reset()
    tty = _tty_fd(sys.stdin)
    if tty is not None and (why := not_foreground(tty)) is not None:
        sys.exit(NOT_FOREGROUND.format(what="the arm session", why=why))
    notice = _tty_fd(sys.stdout)
    message = CTRL_Z_IGNORED.encode()

    def on_tstp(signum: int, frame: FrameType | None) -> None:
        # os.write, not print: a print here could re-enter a print the session was doing
        if notice is not None and INTERRUPTS.hangup is None:
            with contextlib.suppress(OSError):
                os.write(notice, message)

    def on_background_io(signum: int, frame: FrameType | None) -> None:
        # ignored from now on: the retried read fails with EIO, a write or tcflush proceeds,
        # instead of the kernel signalling (or stopping) again
        for s in (signal.SIGTTIN, signal.SIGTTOU):
            signal.signal(s, signal.SIG_IGN)
        INTERRUPTS.lose_foreground(
            f"{signal.Signals(signum).name}: terminal I/O from the background"
        )

    handlers = {
        signal.SIGTSTP: on_tstp,
        signal.SIGTTIN: on_background_io,
        signal.SIGTTOU: on_background_io,
    }
    prev: dict[signal.Signals, Any] = {}
    try:
        for s, h in handlers.items():
            prev[s] = signal.signal(s, h)
        INTERRUPTS.tty_fd, INTERRUPTS.job_control = tty, True
        yield INTERRUPTS
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
        INTERRUPTS.tty_fd, INTERRUPTS.job_control = None, False


@contextlib.contextmanager
def operator_signals(on_signal: Callable[[], None] | None = None) -> Iterator[Interrupts]:
    """For a real-arm session (module doc): SIGINT only sets ``INTERRUPTS.ctrl_c``; SIGHUP and
    SIGTERM set ``INTERRUPTS.hangup`` and detach stdio; ``job_control`` (Ctrl-Z ignored, the
    foreground watched). ``on_signal()`` runs after any of them and after a lost foreground
    (teleop sets its loop-stop flag). The previous handlers are restored on exit, also on an
    exception."""
    with job_control(), _operator_handlers(on_signal):
        yield INTERRUPTS


@contextlib.contextmanager
def _operator_handlers(on_signal: Callable[[], None] | None) -> Iterator[None]:
    INTERRUPTS.ctrl_c, INTERRUPTS.hangup = False, None

    def on_sigint(signum: int, frame: FrameType | None) -> None:
        INTERRUPTS.ctrl_c = True
        if on_signal is not None:
            on_signal()

    def on_hup(signum: int, frame: FrameType | None) -> None:
        _on_hangup(signum, frame)
        if on_signal is not None:
            on_signal()

    prev_lost = INTERRUPTS.on_lost
    prev: dict[signal.Signals, Any] = {}
    try:
        prev[signal.SIGINT] = signal.signal(signal.SIGINT, on_sigint)
        for s in HANGUP_SIGNALS:
            prev[s] = signal.signal(s, on_hup)
        INTERRUPTS.on_lost = on_signal
        yield
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
        INTERRUPTS.on_lost = prev_lost


@contextlib.contextmanager
def _reading() -> Iterator[None]:
    """During a prompt read only: Ctrl-C raises KeyboardInterrupt, a hang-up raises ``Hangup``,
    a lost foreground (before the read, or an EOF / EIO from the background) ``LostForeground``."""
    gone = INTERRUPTS.terminal_gone()
    if gone is not None:
        raise gone_error(*gone)
    prev: dict[signal.Signals, Any] = {
        signal.SIGINT: signal.signal(signal.SIGINT, signal.default_int_handler)
    }
    for s in HANGUP_SIGNALS:
        prev[s] = signal.signal(s, _raise_hangup)
    try:
        yield
    except EOFError as e:
        gone = None if isinstance(e, Hangup) else INTERRUPTS.terminal_gone()
        if gone is not None:
            raise gone_error(*gone) from e
        raise
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
    ahead input discarded first. Unbounded, and only for prompts answered with torque OFF: every
    prompt read while the arm may hold under torque goes through ``TorqueGuard`` ->
    ``timed_input``, including teleop's null (wait forever) timeout."""
    with _reading():
        discard_typeahead()
        try:
            return input(prompt)
        except OSError as e:  # EIO from a hung-up tty
            raise EOFError(f"stdin: {e}") from e


def timed_input(prompt: str, timeout_s: float | None) -> str | None:
    """A prompt read that gives up after ``timeout_s`` (None = never): None. The deadline covers
    the whole line (module doc): fd 0 is read with ``os.read``, each read after ``select()`` with
    the time left, in this thread (no reader thread; the servo bus is never touched here). A partial
    line at the deadline is a timeout (its bytes dropped); EOF, also after a partial line, raises
    EOFError (``Hangup`` on a hang-up). Typed-ahead input is discarded first. ``timeout_s`` None
    (teleop's null ``prompt_torque_timeout_s``; ``TorqueGuard`` reads with torque already off)
    goes through the same reader with no deadline, so a partial line then EOF is EOF there too,
    never a truncated answer (review F1: it was ``input()``, which returned the partial line)."""
    with _reading():
        discard_typeahead()
        _write_prompt(prompt)
        deadline = math.inf if timeout_s is None else time.monotonic() + max(0.0, timeout_s)
        line = _read_line_by(deadline)
        if line is None:
            _write_prompt("\n")
        return line


def _read_line_by(deadline: float) -> str | None:
    """One line from fd 0 (no newline), or None once ``deadline`` (``time.monotonic``; ``inf`` =
    never) passes, also in mid-line. Never reads past the newline (``_read_size``). EOF ->
    EOFError; OSError (EIO from a hung-up tty) and a closed / fd-less stdin -> EOFError. Ctrl-C
    and a hang-up raise from ``select`` / ``os.read`` under ``_reading``. On a watched terminal
    (``job_control``) the wait wakes every ``FOREGROUND_POLL_S`` to check the foreground (the
    deadline is unchanged): ``LostForeground`` once it is lost."""
    got = bytearray()
    poll = math.inf if INTERRUPTS.tty_fd is None else FOREGROUND_POLL_S
    try:
        fd = sys.stdin.fileno()
        encoding = getattr(sys.stdin, "encoding", None) or "utf-8"
        size = _read_size(fd)
        while (left := deadline - time.monotonic()) > 0:
            wait = min(left, poll)
            if not select.select([fd], [], [], None if math.isinf(wait) else wait)[0]:
                gone = INTERRUPTS.terminal_gone()
                if gone is not None:
                    raise gone_error(*gone)
                continue  # the deadline is checked above, maybe in mid-line
            chunk = os.read(fd, size)  # readable: returns at once (bytes, or b"" at EOF)
            if not chunk:
                raise EOFError("stdin closed" + (" in mid-line" if got else ""))
            end = chunk.find(b"\n")
            if end >= 0:  # canonical tty / 1 byte: the newline ends the chunk
                got += chunk[:end]
                return got.decode(encoding, errors="replace").rstrip("\r")
            got += chunk
    except (OSError, ValueError) as e:  # EIO from a hung-up tty; a closed or fd-less stdin
        raise EOFError(f"stdin: {e}") from e
    return None  # timeout: the partial line (if any) is dropped


def _read_size(fd: int) -> int:
    """Bytes per ``os.read``: a tty in canonical mode returns at most one line per read (up to the
    newline or a Ctrl-D), so read it whole: a mid-line Ctrl-D then hands over a partial line, as
    for any program, instead of a 1-byte read stopping at it and seeing EOF. Anything else (a
    pipe, a file, a raw tty): 1 byte, so nothing past the newline is consumed."""
    try:
        if os.isatty(fd) and termios.tcgetattr(fd)[3] & termios.ICANON:
            return 4096
    except termios.error:
        pass
    return 1


NOT_FOREGROUND = (
    "{what}: this process is not in its terminal's foreground ({why}): Ctrl-C and the "
    "torque-on prompts would not reach it. Run it in the foreground (no '&', not resumed with "
    "'bg'); nothing was opened"
)
NOT_A_TERMINAL = (
    "{what}: stdin and stdout must be an interactive terminal for the real arm ({bad} is not a "
    "tty): the torque-on prompts need an operator who sees them and answers them. Run it "
    "directly in Terminal (no pipe, no redirect; to keep a log use `script -q log.txt uv run "
    "...`); nothing was opened"
)


def not_a_terminal() -> str | None:
    """The first of stdin / stdout that is not a tty, or None when both are."""
    for name, stream in (("stdin", sys.stdin), ("stdout", sys.stdout)):
        try:
            if stream is None or not stream.isatty():
                return name
        except (OSError, ValueError, AttributeError):
            return name
    return None


def foreground_problem() -> str | None:
    """Why this process is not in the foreground of its terminal (stdin's, and stdout's when it
    is the same terminal), or None when it is."""
    fds = [fd for fd in (_tty_fd(sys.stdin), _tty_fd(sys.stdout)) if fd is not None]
    if len(fds) == 2:
        try:
            a, b = os.fstat(fds[0]), os.fstat(fds[1])
            if a.st_rdev != b.st_rdev:
                fds = fds[:1]  # another terminal on stdout: stdin's foreground is the one read
        except OSError:
            pass
    for fd in fds:
        why = not_foreground(fd)
        if why is not None:
            return why
    return None if fds else "stdin is not a tty"


def require_terminal(what: str) -> None:
    """Exit (``NOT_A_TERMINAL``) unless stdin and stdout are a tty, and (``NOT_FOREGROUND``)
    unless this process is in that terminal's foreground process group. The hardware entry
    points call it before any port is opened or torque enabled."""
    bad = not_a_terminal()
    if bad is not None:
        sys.exit(NOT_A_TERMINAL.format(what=what, bad=bad))
    why = foreground_problem()
    if why is not None:
        sys.exit(NOT_FOREGROUND.format(what=what, why=why))


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
    ``lost_foreground``, ``timeout``."""

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
            if INTERRUPTS.hangup is not None:
                return Reply(HANGUP)
            if isinstance(e, Hangup):
                return Reply(e.how)
            return Reply(LOST_FOREGROUND if INTERRUPTS.lost_foreground is not None else EOF)
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
        """Nobody answered (``timeout``) or nobody can (``STDIN_GONE``): tell the operator to
        support the arm, wait the grace period (ENTER = now; Ctrl-C ignored; without stdin after
        an EOF, a hang-up or a lost foreground), trip the deadman TORQUE_OFF and turn torque off."""
        grace = self.wait.grace_s
        if cause in STDIN_GONE:
            loud = {HANGUP: "TERMINAL HUNG UP", LOST_FOREGROUND: "LOST THE TERMINAL'S FOREGROUND"}
            self.say(STDIN_CLOSED.format(what=loud.get(cause, "STDIN CLOSED"), grace=grace))
            self._sleep_until(self.clock() + grace)
        else:
            timeout = self.wait.timeout_s or 0.0
            self.say(TORQUE_WAIT_EXPIRED.format(timeout=timeout, grace=grace))
            if grace > 0:
                self._grace(grace)
        what = {
            EOF: "stdin closed",
            HANGUP: "terminal hung up",
            LOST_FOREGROUND: "lost the terminal's foreground",
        }.get(cause, "no answer in time")
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
