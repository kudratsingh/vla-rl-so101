"""Ctrl-Z, the foreground requirement and a lost foreground on a real pseudo-terminal.

Owner decision 2026-11-08: Ctrl-Z never suspends the pilot, the real eval or teleop while the
arm session runs; the session needs the terminal's foreground; losing it takes the controlled
shutdown path of a hang-up. Each test runs tests/pty_shell.py (a minimal job-control shell: the
pty is its controlling terminal, the child runs in a process group of its own, which is not
orphaned, so the kernel really stops it on a Ctrl-Z it does not handle) with
tests/pty_arm_child.py (the real ``--hardware`` code path of scripts/wo07_pilot.py or
scripts/wo07_real_eval.py on fakes: no serial port, no camera, a mock policy on 127.0.0.1) and
types into the pty's master side like the operator would. ``test_harness_really_stops_a_child_
that_does_not_handle_ctrl_z`` is the negative control.

Every wait has a hard timeout; the shell, the child and their process groups are killed at the
end of every test, pass or fail.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import termios
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from src.config import REPO_ROOT

pytestmark = pytest.mark.skipif(
    sys.platform not in ("darwin", "linux") or not hasattr(termios, "TIOCSCTTY"),
    reason="needs POSIX job control on a pseudo-terminal",
)

TESTS = REPO_ROOT / "tests"
SHELL = TESTS / "pty_shell.py"
CHILD = TESTS / "pty_arm_child.py"
CTRL_Z = b"\x1a"  # VSUSP on a default termios
START_S = 90.0  # interpreter + numpy + grpc imports on a cold CI runner
WAIT_S = 30.0
TIMEOUT_S = 8.0  # the torque-on prompt bound in the child (deploy.prompt_torque_timeout_s)
GRACE_S = 1.0


class Pty:
    """tests/pty_shell.py on a fresh pty, running ``argv`` in the foreground (``fg``) or the
    background (``bg``). Output, shell events and the child's marks are collected by threads."""

    def __init__(self, mode: str, argv: list[str], mark_w: int, mark_r: int):
        self.master, slave = os.openpty()
        ctl_r, self.ctl_w = os.pipe()
        evt_r, evt_w = os.pipe()
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(TESTS)])}
        env.pop("PYTEST_CURRENT_TEST", None)
        cmd = [sys.executable, str(SHELL), str(ctl_r), str(evt_w), mode, str(mark_w), "--"]
        self.proc = subprocess.Popen(
            [*cmd, *argv],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            pass_fds=(ctl_r, evt_w, mark_w),
            cwd=REPO_ROOT,
            env=env,
        )
        for fd in (slave, ctl_r, evt_w, mark_w):
            os.close(fd)
        self.out = bytearray()
        self.events: list[str] = []
        self.marks: list[str] = []
        self.lock = threading.Lock()
        self.child_pid: int | None = None
        self.threads = [
            threading.Thread(target=self._drain, args=(self.master, None), daemon=True),
            threading.Thread(target=self._drain, args=(evt_r, self.events), daemon=True),
            threading.Thread(target=self._drain, args=(mark_r, self.marks), daemon=True),
        ]
        for t in self.threads:
            t.start()
        started = self.wait_event("STARTED", START_S)
        self.child_pid = int(started.split()[1])

    def _drain(self, fd: int, lines: list[str] | None) -> None:
        buf = b""
        with contextlib.suppress(OSError):  # EIO on the master once every slave fd closed
            while chunk := os.read(fd, 4096):
                with self.lock:
                    if lines is None:
                        self.out += chunk
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        lines.append(line.decode())
        if lines is not None:
            os.close(fd)

    def text(self) -> str:
        with self.lock:
            return self.out.decode(errors="replace")

    def until(self, cond: Callable[[], Any], timeout: float, what: str) -> Any:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            got = cond()
            if got:
                return got
            time.sleep(0.01)
        tail = self.text()[-3000:]
        raise AssertionError(f"timed out after {timeout} s waiting for {what}\n--- pty ---\n{tail}")

    def expect(self, text: str, after: int = 0, timeout: float = WAIT_S) -> int:
        """Wait for ``text`` in the output after offset ``after``; returns the offset past it."""

        def found() -> int:
            i = self.text().find(text, after)
            return 0 if i < 0 else i + len(text)

        return int(self.until(found, timeout, repr(text)))

    def wait_event(self, prefix: str, timeout: float = WAIT_S) -> str:
        def ev() -> str | None:
            with self.lock:
                return next((e for e in self.events if e.startswith(prefix)), None)

        return str(self.until(ev, timeout, f"shell event {prefix}"))

    def writes(self) -> int:
        with self.lock:
            return sum(m.startswith("WRITE ") for m in self.marks)

    def wait_writes(self, n: int, timeout: float = WAIT_S) -> int:
        return int(self.until(lambda: self.writes() >= n and self.writes(), timeout, f"{n} writes"))

    def send(self, data: bytes) -> None:
        os.write(self.master, data)

    def command(self, cmd: str) -> None:
        os.write(self.ctl_w, (cmd + "\n").encode())

    def assert_running(self) -> None:
        """Not stopped: the shell (its parent) saw no WUNTRACED stop, and ps agrees."""
        with self.lock:
            stops = [e for e in self.events if e.startswith("STOPPED")]
        assert not stops, stops
        st = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(self.child_pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        assert st and "T" not in st, f"child state {st!r}"

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.ctl_w)  # the shell kills its child on EOF
        for pg in (self.child_pid, self.proc.pid):
            if pg is not None:
                with contextlib.suppress(OSError):
                    os.killpg(pg, signal.SIGKILL)
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(10)
        for t in self.threads:
            t.join(5)
        with contextlib.suppress(OSError):
            os.close(self.master)


@contextlib.contextmanager
def session(mode: str, argv: Callable[[int], list[str]]) -> Iterator[Pty]:
    """``argv(mark_fd)``: the child's command line, given the fd its marks go to."""
    mark_r, mark_w = os.pipe()
    p = Pty(mode, argv(mark_w), mark_w, mark_r)
    try:
        yield p
    finally:
        p.close()


LAUNCHER = "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:], close_fds=False))"
"""Stands in for ``uv run``: a parent in the job's process group that does not handle SIGTSTP
(uv 0.13.5 on macOS: probed with the same pty shell, the ``uv`` process stops on Ctrl-Z while
its Python child keeps running)."""


def arm_session(
    entry: str, runs: Path, mode: str = "fg", launcher: bool = False
) -> contextlib.AbstractContextManager[Pty]:
    def argv(mark_fd: int) -> list[str]:
        cmd = [sys.executable, str(CHILD), entry, str(mark_fd), str(runs)]
        cmd += ["--timeout", str(TIMEOUT_S), "--grace", str(GRACE_S), "--steps", "60"]
        cmd += ["--wait-foreground"] if mode != "bg" else []
        return [sys.executable, "-c", LAUNCHER, *cmd] if launcher else cmd

    return session(mode, argv)


def record(runs: Path) -> dict[str, Any]:
    (rec,) = runs.glob("*/record.json")
    return dict(json.loads(rec.read_text())["results"])


def ctrl_z(p: Pty, after: int) -> int:
    """Type Ctrl-Z; the notice comes back and the child is not stopped."""
    p.send(CTRL_Z)
    end = p.expect("Ctrl-Z ignored", after)
    time.sleep(0.3)  # a stop would have reached the shell by now
    p.assert_running()
    return end


# ------------------------------------------------------------------------------------------------
def test_harness_really_stops_a_child_that_does_not_handle_ctrl_z() -> None:
    """Negative control: the same shell and pty stop a plain foreground child on Ctrl-Z, so the
    tests below would see a stop if the session did not handle it."""
    code = (
        "import os, time\n"
        "while os.tcgetpgrp(0) != os.getpgrp(): time.sleep(0.01)\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    with session("fg", lambda _fd: [sys.executable, "-c", code]) as p:
        p.expect("READY", timeout=START_S)
        p.send(CTRL_Z)
        assert p.wait_event("STOPPED") == "STOPPED SIGTSTP"


def test_pilot_ctrl_z_during_motion_and_at_torque_on_prompts_never_stops_it(
    tmp_path: Path,
) -> None:
    with arm_session("pilot", tmp_path) as p:
        i = p.expect("type 'go' to continue", timeout=START_S)
        p.send(b"go\n")
        i = p.expect("ENTER moves the arm HOME", i)  # torque on from here
        assert "CONNECT" in p.marks
        i = ctrl_z(p, i)  # at a torque-on prompt
        p.send(b"\n")  # the same prompt still answers
        i = p.expect("ENTER starts", i)
        i = ctrl_z(p, i)
        home_writes = p.writes()
        p.send(b"\n")
        n = p.wait_writes(home_writes + 10)  # the policy moves the arm
        i = ctrl_z(p, i)  # during motion
        p.wait_writes(n + 15)  # it keeps ticking after the Ctrl-Z
        i = p.expect("success? [y/n]", i)
        i = ctrl_z(p, i)
        p.send(b"y\n")
        i = p.expect("support it, then press ENTER to disable torque", i)
        t_prompt = time.monotonic()
        time.sleep(1.0)
        i = ctrl_z(p, i)  # the bounded wait keeps running: its deadline is not reset
        i = p.expect("No answer for 8 s with torque ON", i, timeout=TIMEOUT_S + 10)
        waited = time.monotonic() - t_prompt
        assert TIMEOUT_S - 0.5 <= waited <= TIMEOUT_S + 3.0, waited
        p.expect("torque OFF: the arm is limp", i)
        assert p.wait_event("EXIT") == "EXIT 0"
        assert not [e for e in p.events if e.startswith("STOPPED")]  # exited, never stopped
    res = record(tmp_path)
    (t,) = res["trials"]
    assert t["outcome"] == "success" and t["outcome_source"] == "operator"
    assert t["stop_reason"] == "max_steps" and t["fault"] is None
    assert res["session_end"] == "completed"
    assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
    assert res["shutdown_detail"]["cause"] == "timeout"
    assert not res["torque_off_unconfirmed"]


def test_real_eval_ctrl_z_during_motion_then_enter_confirms_support(tmp_path: Path) -> None:
    with arm_session("real_eval", tmp_path) as p:
        i = p.expect("type 'go' to continue", timeout=START_S)
        p.send(b"go\n")
        i = p.expect("ENTER moves the arm HOME", i)
        p.send(b"\n")
        i = p.expect("ENTER starts", i)
        i = ctrl_z(p, i)
        home_writes = p.writes()
        p.send(b"\n")
        n = p.wait_writes(home_writes + 10)
        i = ctrl_z(p, i)
        p.wait_writes(n + 15)
        i = p.expect("success? [y/n]", i)
        p.send(b"n\n")
        i = p.expect("support it, then press ENTER to disable torque", i)
        i = ctrl_z(p, i)
        p.send(b"\n")  # responsive: ENTER after the Ctrl-Z confirms support
        assert p.wait_event("EXIT") == "EXIT 0"
    res = record(tmp_path)
    (t,) = res["trials"]
    assert t["outcome"] == "failure" and t["outcome_source"] == "operator"
    assert res["shutdown"] == "supported" and res["support_confirmed"] is True


def test_pilot_refuses_a_background_process_before_anything_opens(tmp_path: Path) -> None:
    with arm_session("pilot", tmp_path, mode="bg") as p:
        p.expect("not in its terminal's foreground", timeout=START_S)
        assert p.wait_event("EXIT") == "EXIT 1"
        out = p.text()
        assert "nothing was opened" in out and "type 'go'" not in out
        assert "CONNECT" not in p.marks  # the follower's port never opened, torque never on
    assert not list(tmp_path.glob("*/record.json"))


def test_pilot_losing_the_foreground_in_motion_takes_the_hangup_shutdown(tmp_path: Path) -> None:
    with arm_session("pilot", tmp_path) as p:
        i = p.expect("type 'go' to continue", timeout=START_S)
        p.send(b"go\n")
        i = p.expect("ENTER moves the arm HOME", i)
        p.send(b"\n")
        i = p.expect("ENTER starts", i)
        home_writes = p.writes()
        p.send(b"\n")
        p.wait_writes(home_writes + 10)
        p.command("TAKE")  # the shell takes the terminal back (as after a stop and `bg`)
        at_take = p.writes()
        i = p.expect("LOST THE TERMINAL'S FOREGROUND", i)
        assert p.writes() - at_take <= 5  # stopped at the next tick (+ the hold write)
        p.expect("torque OFF: the arm is limp", i, timeout=GRACE_S + WAIT_S)
        assert p.wait_event("EXIT") == "EXIT 0"
    res = record(tmp_path)
    (t,) = res["trials"]
    assert t["policy_started"] and t["outcome"] == "failure"
    assert t["stop_reason"] == "lost_foreground" and t["fault"]["kind"] == "lost_foreground"
    assert res["session_end"].startswith("lost_foreground:")
    assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
    assert res["shutdown_detail"]["cause"] == "lost_foreground"
    assert not res["torque_off_unconfirmed"]


def test_pilot_losing_the_foreground_at_a_torque_on_prompt_ends_the_wait_at_once(
    tmp_path: Path,
) -> None:
    with arm_session("pilot", tmp_path) as p:
        i = p.expect("type 'go' to continue", timeout=START_S)
        p.send(b"go\n")
        i = p.expect("ENTER moves the arm HOME", i)
        p.send(b"\n")
        i = p.expect("ENTER starts", i)
        t0 = time.monotonic()
        p.command("TAKE")
        i = p.expect("LOST THE TERMINAL'S FOREGROUND", i)
        assert time.monotonic() - t0 < 3.0 < TIMEOUT_S  # the poll, not the 8 s timeout
        p.expect("torque OFF: the arm is limp", i, timeout=GRACE_S + WAIT_S)
        assert p.wait_event("EXIT") == "EXIT 1"  # LostForeground re-raised, like a hang-up
    res = record(tmp_path)
    (t,) = res["trials"]
    assert t["outcome"] == "not_started" and t["stop_reason"] == "lost_foreground"
    assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
    assert res["shutdown_detail"]["cause"] == "lost_foreground"
    assert not res["torque_off_unconfirmed"]


def test_under_a_launcher_like_uv_run_ctrl_z_ends_the_session_through_the_lost_foreground_path(
    tmp_path: Path,
) -> None:
    """``uv run scripts/wo07_pilot.py``: Ctrl-Z goes to the whole foreground job. The pilot
    ignores it, the launcher (uv) does not and stops; the shell then takes the terminal back
    ("suspended"). The pilot is never stopped: it loses the foreground and ends through the
    controlled shutdown (motion stops, grace, torque off, ``lost_foreground``)."""
    with arm_session("pilot", tmp_path, mode="zsh", launcher=True) as p:
        i = p.expect("type 'go' to continue", timeout=START_S)
        p.send(b"go\n")
        i = p.expect("ENTER moves the arm HOME", i)
        p.send(b"\n")
        i = p.expect("ENTER starts", i)
        home_writes = p.writes()
        p.send(b"\n")
        p.wait_writes(home_writes + 10)
        p.send(CTRL_Z)
        assert p.wait_event("STOPPED") == "STOPPED SIGTSTP"  # the launcher, not the pilot
        at_stop = p.writes()
        i = p.expect("Ctrl-Z ignored", i)  # the pilot itself ignored it
        i = p.expect("LOST THE TERMINAL'S FOREGROUND", i)
        assert p.writes() - at_stop <= 5
        p.expect("torque OFF: the arm is limp", i, timeout=GRACE_S + WAIT_S)
        p.until(lambda: any(m.startswith("DONE") for m in p.marks), WAIT_S, "the record")
    res = record(tmp_path)
    (t,) = res["trials"]
    assert t["stop_reason"] == "lost_foreground" and t["fault"]["kind"] == "lost_foreground"
    assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
    assert res["shutdown_detail"]["cause"] == "lost_foreground"
