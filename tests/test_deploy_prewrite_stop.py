"""Pre-write stop check in the WO-07 pilot / real-eval control loop (owner 2026-11-09).

``DeployLoop`` passes its stop predicate (``stop_requested``: a recorded hang-up / SIGTERM, a lost
foreground, the Ctrl-C flag) to ``So101Driver.step`` as ``stop_fn``, so a stop recorded while a
tick waits on its camera frame, the policy query or the servo read never produces one more goal.
The trial starts only once its first policy goal write was begun
(``So101Driver.goal_write_attempted``): a first goal blocked by the check is ``not_started`` with
no ``trial_started`` row, and a resume retries the slot with the same placement. The first goal
writes the durable start marker and checks again after that I/O, right before the write: a stop
recorded during the marker I/O, or a marker that cannot be saved, blocks the goal and retracts
the marker (``trial_start_retracted``), so the attempt is ``not_started`` and resume retries it.

Real signals (``os.kill`` on this process) through the real ``operator_signals`` /
``job_control`` handlers; the terminal keeps reporting this process group as foreground
(``tcgetpgrp`` faked), so a lost foreground is the recorded SIGTTOU alone. Fake arm, camera and
clock; a real gRPC mock policy on 127.0.0.1. Every test has a hard SIGALRM timeout.

Run by name before the real-eval protocol is frozen (PR #32):
    uv run python -m pytest -q tests/test_deploy_prewrite_stop.py
"""

from __future__ import annotations

import errno
import functools
import json
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from robot_fakes import real_cfg
from test_deploy import Rig, _console, _pilot_kw, _record, pilot, rig
from test_real_eval import _args, ev

from src import runs as runs_mod
from src.robot import prompts as P
from src.robot.client import DeployLoop, ScriptedOperator
from src.robot.driver import So101Driver
from src.robot.safety import DeadmanResponse
from src.runs import Run

SIGNAL_KIND = {  # the signal -> the stop reason / fault kind the loop records
    "SIGHUP": "hangup",
    "SIGTERM": "hangup",
    "SIGINT": "operator_abort",  # the Ctrl-C flag
    "SIGTTOU": "lost_foreground",
}


class _HardTimeout(BaseException):
    """A BaseException: no session ``except Exception`` handler (``run_trial``'s client_error
    path, a prompt's) can swallow it and carry on."""


@pytest.fixture(autouse=True)
def hard_timeout() -> Iterator[None]:
    """Fail (never hang) past 60 s: SIGALRM, which no session code touches, re-armed every 5 s
    after that so a swallowed alarm fires again."""

    def fire(_signum: int, _frame: Any) -> None:
        raise _HardTimeout("test exceeded its 60 s hard timeout")

    prev = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, 60.0, 5.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """stdin watched as the session's terminal, which keeps naming this process group as its
    foreground (``tcgetpgrp``); stdio stays attached after a hang-up (pytest's capture)."""
    monkeypatch.setattr(P.os, "tcgetpgrp", lambda _fd: os.getpgrp())
    monkeypatch.setattr(P, "_detach_stdio", lambda: None)


class Probe:
    """Order log of one session: ("signal", name), ("stop_check", where, stopped) for every stop
    check (``where`` = "pre_write" inside ``step``'s ``stop_fn``, else "loop"), ("write", kind)
    for every Goal_Position write (kind "goal": a ``step`` goal, policy or move home; "hold":
    the hold at the measured pose, also the deadman's). Sends ONE real signal from inside the
    chosen wait:

    * ``camera``: the frame read of policy write tick ``n`` of the first chunk (1 = the tick that
      writes the trial's first policy goal);
    * ``policy``: policy query ``n`` (the signal arrives while the query is in flight);
    * ``servo``: the joint read inside ``step`` of policy write tick ``n`` of the first chunk;
    * ``home_camera``: the frame read of move-home tick ``n``;
    * ``marker``: the write of the ``n``-th durable ``trial_started`` row (``Run.log_metrics``,
      reached through the real ``on_policy_start``), right after the row is on disk.

    ``action`` "serial" instead makes the goal write of that tick fail (a bus fault mid-write);
    with ``where="marker"``, "marker_fail" makes that row's write raise ENOSPC (nothing written)
    and "marker_fail_all" makes it and every later log write raise (the disk stays full);
    "none" never fires (an order log only); "sync_fail" lets that row reach metrics.jsonl and makes its sync to stable storage
    (``src.runs.sync_fd``, through ``Run.log_durable``) raise EIO. Every sync is logged as
    ("sync", method or "EIO")."""

    def __init__(
        self,
        loop: DeployLoop,
        sig: str | None,
        where: str,
        n: int,
        action: str = "signal",
        monkeypatch: pytest.MonkeyPatch | None = None,
    ) -> None:
        self.loop, self.sig, self.where, self.n, self.action = loop, sig, where, n, action
        self.log: list[tuple[Any, ...]] = []
        self.queries = 0
        self.cam_reads = {"home": 0, "chunk": 0}
        self.joint_reads = 0
        self.fired = False
        self._in_pre_write = False
        self._hold_depth = 0
        arm, fol = loop.arm, loop.arm.follower

        cam_read = loop.cam.read

        def read() -> Any:
            if self.queries == 0:
                self.cam_reads["home"] += 1
                self._maybe_fire("home_camera", self.cam_reads["home"])
            elif self.queries == 1:
                self.cam_reads["chunk"] += 1
                self._maybe_fire("camera", self.cam_reads["chunk"])
            return cam_read()

        get_obs = fol.get_observation

        def get_observation() -> dict[str, Any]:
            if self.queries == 1 and self._in_step:
                self.joint_reads += 1
                self._maybe_fire("servo", self.joint_reads)
            return get_obs()

        act = loop.client.act

        def act_wrapped(*a: Any, **k: Any) -> Any:
            self.queries += 1
            self._maybe_fire("policy", self.queries)
            return act(*a, **k)

        send = fol.send_action

        def send_action(action: dict[str, Any]) -> dict[str, Any]:
            out = send(action)
            self.log.append(("write", "hold" if self._hold_depth else "goal"))
            return out

        hold = arm.hold

        def hold_wrapped(*a: Any, **k: Any) -> None:
            self._hold_depth += 1
            try:
                hold(*a, **k)
            finally:
                self._hold_depth -= 1

        step = arm.step

        def step_wrapped(*a: Any, **k: Any) -> Any:
            self._in_step = True
            try:
                return step(*a, **k)
            finally:
                self._in_step = False

        stop_requested = loop.stop_requested

        def stop_requested_logged() -> Any:
            stop = stop_requested()
            where = "pre_write" if self._in_pre_write else "loop"
            self.log.append(("stop_check", where, stop is not None))
            return stop

        pre_write = loop._pre_write_stop

        def pre_write_logged() -> bool:
            self._in_pre_write = True
            try:
                return pre_write()
            finally:
                self._in_pre_write = False

        self._sync_fail_armed = False
        if monkeypatch is not None:
            real_sync = runs_mod.sync_fd

            def sync_fd(fd: int, *a: Any) -> str:
                if self._sync_fail_armed:
                    self._sync_fail_armed = False
                    self.log.append(("sync", "EIO"))
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                method = real_sync(fd, *a)
                self.log.append(("sync", method))
                return method

            monkeypatch.setattr(runs_mod, "sync_fd", sync_fd)
        if where == "marker":
            assert monkeypatch is not None
            real_log = Run.log_metrics
            markers = [0]

            def log_metrics(run: Run, step: int | None = None, **row: Any) -> None:
                if self.fired and self.action == "marker_fail_all":
                    raise OSError(errno.ENOSPC, "No space left on device")
                self.log.append(("row", row.get("kind")))
                if row.get("kind") == "trial_started":
                    markers[0] += 1
                    if self.action == "sync_fail":  # on disk, then its sync fails
                        real_log(run, step, **row)
                        if self._maybe_fire("marker", markers[0]):
                            self._sync_fail_armed = True
                        return
                    if self.action != "signal" and self._maybe_fire("marker", markers[0]):
                        raise OSError(errno.ENOSPC, "No space left on device")
                    real_log(run, step, **row)
                    if self.action == "signal":
                        self._maybe_fire("marker", markers[0])
                    return
                real_log(run, step, **row)

            monkeypatch.setattr(Run, "log_metrics", log_metrics)

        self._in_step = False
        loop.cam.read = read  # type: ignore[method-assign]
        fol.get_observation = get_observation  # type: ignore[method-assign]
        loop.client.act = act_wrapped  # type: ignore[method-assign]
        fol.send_action = send_action  # type: ignore[method-assign]
        arm.hold = hold_wrapped  # type: ignore[method-assign]
        arm.step = step_wrapped  # type: ignore[method-assign]
        loop.stop_requested = stop_requested_logged  # type: ignore[method-assign]
        loop._pre_write_stop = pre_write_logged  # type: ignore[method-assign]

    def _maybe_fire(self, where: str, count: int) -> bool:
        if self.fired or where != self.where or count != self.n or self.action == "none":
            return False
        self.fired = True
        if self.action == "serial":
            self.loop.arm.follower.fail_write = True  # type: ignore[attr-defined]
            self.log.append(("serial_fault_armed",))
            return True
        if self.action.startswith("marker_fail") or self.action == "sync_fail":
            self.log.append(("marker_failed",))
            return True
        assert self.sig is not None
        self.log.append(("signal", self.sig))
        os.kill(os.getpid(), getattr(signal, self.sig))
        recorded = {  # the handler ran: the stop is recorded before the tick goes on
            "SIGHUP": P.INTERRUPTS.hangup,
            "SIGTERM": P.INTERRUPTS.hangup,
            "SIGINT": P.INTERRUPTS.ctrl_c or None,
            "SIGTTOU": P.INTERRUPTS.lost_foreground,
        }[self.sig]
        assert recorded, f"{self.sig} not recorded by its handler"
        return True

    def _event(self) -> int:
        return next(
            i
            for i, e in enumerate(self.log)
            if e[0] in ("signal", "serial_fault_armed", "marker_failed")
        )

    def after_signal(self) -> list[tuple[Any, ...]]:
        return self.log[self._event() + 1 :]

    def goals_after_signal(self) -> int:
        return sum(e == ("write", "goal") for e in self.after_signal())

    def goals_before_signal(self) -> int:
        return sum(e == ("write", "goal") for e in self.log[: self._event()])


def _metrics(tmp_path: Path) -> list[dict[str, Any]]:
    (m,) = tmp_path.glob("*/metrics.jsonl")
    return [json.loads(x) for x in m.read_text().splitlines()]


def _pilot_session(
    tmp_path: Path,
    sig: str | None,
    where: str,
    n: int,
    action: str = "signal",
    script: tuple[str, ...] = ("", "", ""),  # home, placement, support (or the score, support)
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Probe, Rig]:
    """One pilot trial (``scripts/wo07_pilot.session``) under the real signal handlers, the real
    ConsoleOperator on a scripted stdin read through the real ``_reading`` checks."""
    with P.operator_signals(), rig() as r:
        P.INTERRUPTS.tty_fd = 0  # as with stdin on the terminal (the hardware session)
        r.loop.interrupts = P.INTERRUPTS
        op, _reader = _console(r, list(script))
        fake = op.guard.read_line

        def read_line(prompt: str, timeout_s: float | None) -> str | None:
            with P._reading():  # the real check timed_input makes before every read
                return fake(prompt, timeout_s)

        op.guard.read_line = read_line
        probe = Probe(r.loop, sig, where, n, action, monkeypatch)
        pilot.session(real_cfg(), r.loop, op, r.server_info, **_pilot_kw(tmp_path))
        return _record(tmp_path), _metrics(tmp_path), probe, r


def _assert_shutdown(res: dict[str, Any], r: Rig, kind: str) -> None:
    if kind == "operator_abort":  # held, the operator supported the arm (ENTER), torque off
        assert res["session_end"] == "operator_abort"
        assert res["shutdown"] == "supported" and res["support_confirmed"] is True
    else:  # the closed-stdin path: hold, grace period, torque off, unconfirmed
        assert res["session_end"].startswith(f"{kind}: ")
        assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
        assert res["shutdown_detail"]["cause"] == kind
        assert r.arm.deadman.response is DeadmanResponse.TORQUE_OFF
    assert not r.fol.bus.torque_enabled


FIRST_GOAL_CASES = [(sig, where) for sig in SIGNAL_KIND for where in ("camera", "policy", "servo")]


@pytest.mark.parametrize(("sig", "where"), FIRST_GOAL_CASES)
def test_stop_before_the_first_policy_goal_is_not_started(
    tmp_path: Path, terminal: None, sig: str, where: str
) -> None:
    """AC2(a): a stop recorded inside the first write tick's camera read, the first policy query
    or the first step's servo read: no policy goal is ever written, the attempt is
    ``not_started`` (stop cause kept), no ``trial_started`` row; then hold and the shutdown."""
    res, rows, probe, r = _pilot_session(tmp_path, sig, where, 1)
    kind = SIGNAL_KIND[sig]
    assert probe.fired and probe.goals_after_signal() == 0  # AC1: no goal after the stop
    after = probe.after_signal()
    first_check = next(e for e in after if e[0] == "stop_check")
    # the camera / servo waits sit after the top-of-tick check: only the pre-write check sees it
    assert first_check == (
        "stop_check",
        "loop" if where == "policy" else "pre_write",
        True,
    )
    assert ("write", "hold") in after  # the hold at the measured pose still runs
    (t,) = res["trials"]
    assert (t["outcome"], t["outcome_source"]) == ("not_started", "not_scored")
    assert not t["policy_started"] and t["policy_started_utc"] is None
    assert t["stop_reason"] == kind and t["fault"] is None and t["steps"] == 0
    assert t["stop_detail"]  # the stop cause
    # nothing resume counts: no marker, so nothing to retract either
    assert not any(row["kind"] in ("trial_started", "trial_start_retracted") for row in rows)
    assert [row["kind"] for row in rows if row["kind"] == "trial"] == ["trial"]
    assert res["summary"]["cells"]["PickPlace/seen"]["n"] == 0
    assert res["summary"]["cells"]["PickPlace/seen"]["not_started"] == 1
    _assert_shutdown(res, r, kind)


LATER_GOAL_CASES = [
    (sig, where, n)
    for sig in SIGNAL_KIND
    for where, n in (("camera", 3), ("policy", 2), ("servo", 3))
]


@pytest.mark.parametrize(("sig", "where", "n"), LATER_GOAL_CASES)
def test_stop_after_the_first_policy_goal_is_a_started_failure(
    tmp_path: Path, terminal: None, sig: str, where: str, n: int
) -> None:
    """AC2(b): the same stops after the first policy goal was written: blocked before the next
    goal, a started trial scored a failure with the fault recorded, the durable start row kept."""
    res, rows, probe, r = _pilot_session(tmp_path, sig, where, n)
    kind = SIGNAL_KIND[sig]
    assert probe.fired and probe.goals_after_signal() == 0
    (t,) = res["trials"]
    assert t["policy_started"] and t["policy_started_utc"]
    assert (t["outcome"], t["outcome_source"]) == ("failure", "fault")
    assert t["stop_reason"] == kind and t["fault"]["kind"] == kind
    written = 8 if where == "policy" else n - 1  # policy goals before the stop
    assert t["steps"] == written
    kinds = [row["kind"] for row in rows if row["kind"] in ("trial_started", "trial")]
    assert kinds == ["trial_started", "trial"]
    c = res["summary"]["cells"]["PickPlace/seen"]
    assert (c["n"], c["successes"], c["faulted"]) == (1, 0, 1)
    _assert_shutdown(res, r, kind)


def test_stop_inside_a_move_home_tick_blocks_the_home_goal(tmp_path: Path, terminal: None) -> None:
    """AC1: move-home goals go through the same pre-write check (SIGHUP inside the camera read of
    home tick 2): no goal after it, home not reached, the attempt not started, then the hold."""
    res, rows, probe, r = _pilot_session(tmp_path, "SIGHUP", "home_camera", 2)
    assert probe.fired and probe.goals_after_signal() == 0 and probe.goals_before_signal() == 1
    assert next(e for e in probe.after_signal() if e[0] == "stop_check") == (
        "stop_check",
        "pre_write",
        True,
    )
    (t,) = res["trials"]
    assert t["outcome"] == "not_started" and t["stop_reason"] == "hangup"
    assert t["home"]["reached"] is False and probe.queries == 0
    assert not any(row["kind"] == "trial_started" for row in rows)
    _assert_shutdown(res, r, "hangup")


def test_serial_fault_during_the_first_policy_write_is_a_started_failure(
    tmp_path: Path, terminal: None
) -> None:
    """AC2(b): the first policy goal's bus write raises (it may have reached the servos: doubt
    counts as sent): a started failure, fault kind serial, with its durable start row."""
    res, rows, probe, r = _pilot_session(tmp_path, None, "camera", 1, action="serial")
    assert probe.fired and probe.goals_after_signal() == 0  # the write raised part-way
    (t,) = res["trials"]
    assert t["policy_started"] and t["policy_started_utc"]
    assert (t["outcome"], t["fault"]["kind"], t["stop_reason"]) == (
        "failure",
        "serial",
        "serial_fault",
    )
    assert t["steps"] == 0  # the write never completed
    assert t["stop_detail"] and "retracted" not in t["stop_detail"]
    kinds = [row["kind"] for row in rows if row["kind"].startswith("trial")]
    assert kinds == ["trial_started", "trial"]  # the marker stands: no retraction
    assert r.arm.goal_write_attempted and not r.fol.bus.torque_enabled


def test_joint_read_fault_before_the_first_policy_write_is_not_started() -> None:
    """The first step fails at its joint read, before its goal write: no goal was sent, so the
    policy never started (``goal_write_attempted`` False)."""
    from test_deploy import _trials

    rows: list[dict[str, Any]] = []
    with rig() as r:
        real_act = r.client.act

        def act(*a: Any, **k: Any) -> Any:
            out = real_act(*a, **k)
            r.fol.fail_read = True  # the first write tick's joint read fails
            return out

        r.client.act = act  # type: ignore[method-assign]
        trials, end = _trials(r, ScriptedOperator(), log_row=lambda **row: rows.append(row))
        (t,) = trials
        assert t["outcome"] == "not_started" and t["stop_reason"] == "serial_fault"
        assert not t["policy_started"] and not r.arm.goal_write_attempted
        assert not any(row["kind"] == "trial_started" for row in rows)
        assert end.startswith("torque_off")


def test_healthy_trial_writes_every_goal_after_a_passing_pre_write_check(
    tmp_path: Path, terminal: None
) -> None:
    """No stop: every move-home and policy goal is written, each right after a pre-write check
    that passed; the trial runs to its step limit, started at its first write."""
    res, rows, probe, _r = _pilot_session(tmp_path, None, "never", 0, script=("", "", "y", ""))
    (t,) = res["trials"]
    assert t["stop_reason"] == "max_steps" and t["steps"] == 16 and t["outcome"] == "success"
    goals = [i for i, e in enumerate(probe.log) if e == ("write", "goal")]
    assert len(goals) >= 16
    assert all(probe.log[i - 1] == ("stop_check", "pre_write", False) for i in goals)
    # exactly one goal (the first policy goal) has two passing pre-write checks before it: the
    # one before the start marker and the final one after it, right before the write
    twice = [i for i in goals if probe.log[i - 2] == ("stop_check", "pre_write", False)]
    assert len(twice) == 1
    kinds = [row["kind"] for row in rows if row["kind"].startswith("trial")]
    assert kinds == ["trial_started", "trial"]
    started = next(row for row in rows if row["kind"] == "trial_started")
    first_decision = next(row for row in rows if row["kind"] == "decision")
    assert started["policy_started"] and first_decision["decision"] == 1
    assert res["shutdown"] == "supported"


@pytest.mark.parametrize("first_write", ["ok", "serial_fault"])
def test_start_marker_is_durable_and_stamped_when_the_first_policy_write_begins(
    first_write: str,
) -> None:
    """The durable ``trial_started`` row is already logged when the first policy goal's bus
    write begins (a process killed mid-write still leaves it, so resume never re-rolls a slot
    whose goal may have moved the arm), and ``policy_started_t`` is that write's begin time: after
    the step's joint read (the clock advances inside reads and writes here), not when the step
    was entered and not after the write returned or a fault mid-write was handled."""
    from test_deploy import _trials

    rows: list[dict[str, Any]] = []
    results: list[Any] = []
    t_step: list[float] = []  # clock at the first policy step's entry
    at_send: list[tuple[float, int]] = []  # (clock, trial_started rows) at that write's begin
    with rig() as r:
        policy_phase = [False]
        real_act, get_obs, send, step = (
            r.client.act,
            r.fol.get_observation,
            r.fol.send_action,
            r.arm.step,
        )

        def act(*a: Any, **k: Any) -> Any:
            policy_phase[0] = True
            return real_act(*a, **k)

        def get_observation() -> dict[str, Any]:
            r.clock.advance(0.002)  # a joint read takes time
            return get_obs()

        def send_action(action: dict[str, Any]) -> dict[str, Any]:
            if policy_phase[0] and not at_send:
                at_send.append((r.clock(), sum(row["kind"] == "trial_started" for row in rows)))
                r.fol.fail_write = first_write == "serial_fault"
            r.clock.advance(0.003)  # a write takes time
            return send(action)

        def step_wrapped(*a: Any, **k: Any) -> Any:
            if policy_phase[0] and not t_step:
                t_step.append(r.clock())
            return step(*a, **k)

        real_run = r.loop.run_trial

        def run_trial(*a: Any, **k: Any) -> Any:
            results.append(real_run(*a, **k))
            return results[-1]

        r.client.act = act  # type: ignore[method-assign]
        r.fol.get_observation = get_observation  # type: ignore[method-assign]
        r.fol.send_action = send_action  # type: ignore[method-assign]
        r.arm.step = step_wrapped  # type: ignore[method-assign]
        r.loop.run_trial = run_trial  # type: ignore[method-assign]
        trials, _end = _trials(r, ScriptedOperator(), log_row=lambda **row: rows.append(row))
    ((t_send, started_rows),) = at_send
    assert started_rows == 1  # durable before any byte of the first goal could go out
    (res,) = results
    assert res.policy_started_t == t_send > t_step[0]  # the write's begin, after the joint read
    assert res.policy_started_utc
    (t,) = trials
    if first_write == "ok":
        assert res.t_first_write > res.policy_started_t and t["outcome"] == "success"
    else:
        assert t["outcome"] == "failure" and t["fault"]["kind"] == "serial"
    kinds = [row["kind"] for row in rows if row["kind"] in ("trial_started", "trial")]
    assert kinds == ["trial_started", "trial"]


def test_mutation_without_the_pre_write_check_a_goal_follows_the_stop(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sensitivity: with ``DeployLoop``'s pre-write check disabled (the mutation), a SIGHUP
    inside the first write tick's camera read is followed by a policy goal and the trial counts
    as started: the not-started tests above depend on the check."""
    monkeypatch.setattr(DeployLoop, "_pre_write_stop", lambda self: False)
    res, rows, probe, _r = _pilot_session(tmp_path, "SIGHUP", "camera", 1)
    assert probe.goals_after_signal() == 1
    (t,) = res["trials"]
    assert t["policy_started"] and t["outcome"] == "failure"
    assert any(row["kind"] == "trial_started" for row in rows)


# -- real eval end to end: the blocked slot is retried on --resume with the same placement --------
def test_real_eval_blocked_first_goal_leaves_no_start_and_resume_retries_the_slot(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``scripts/wo07_real_eval.py --mock`` under the real signal handlers: SIGTERM inside the
    servo read of slot 0's first policy step. The attempt is ``not_started`` (stop cause
    hangup), metrics.jsonl has no ``trial_started`` row, and ``--resume`` re-runs slot 0 with the
    same placement and scores it once."""
    probes: list[Probe] = []

    class SignalLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **{**k, "interrupts": P.INTERRUPTS})
            probes.append(Probe(self, "SIGTERM", "servo", 1))

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setattr(ev, "DeployLoop", SignalLoop)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen")
    with P.operator_signals():
        P.INTERRUPTS.tty_fd = 0
        rec = ev.main(argv)
    (probe,) = probes
    assert probe.fired and probe.goals_after_signal() == 0
    assert ("write", "hold") in probe.after_signal()  # the hold still runs after the block
    r1 = json.loads(rec.read_text())["results"]
    assert r1["session_end"].startswith("hangup: SIGTERM")
    # shutdown: held, the (scripted) operator supported the arm, torque off and confirmed
    assert (r1["shutdown"], r1["support_confirmed"]) == ("supported", True)
    assert r1["shutdown_detail"]["ack"]["given"] and r1["torque_off_unconfirmed"] is None
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (t0,) = r1["trials"]
    assert (t0["trial_idx"], t0["outcome"], t0["policy_started"]) == (0, "not_started", False)
    assert t0["stop_reason"] == "hangup" and "SIGTERM" in t0["stop_detail"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert not any(row.get("kind") == "trial_started" for row in rows)
    assert ev.done_indices(r1["trials"], "seen") == set()  # resume treats slot 0 as not done
    assert not r1["summary"]["complete"]

    monkeypatch.setattr(ev, "DeployLoop", DeployLoop)  # the resumed run: no stop
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["resumed_from"] == rec.parent.name and r2["session_end"] == "completed"
    retried = [t for t in r2["trials"][1:] if t["trial_idx"] == 0]
    assert len(retried) == 1 and retried[0]["policy_started"]
    assert retried[0]["placement"] == t0["placement"]  # the same placement
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 0, 1]
    c = r2["summary"]["cells"]["PickPlace/seen"]
    assert (c["n"], c["not_started"]) == (2, 1) and r2["summary"]["complete"]
    assert np.isfinite(c["wilson_95"][0])


def test_driver_reports_goal_sent_only_once_the_write_began(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``So101Driver.goal_write_attempted``: False after ``MotionStopped`` (nothing written),
    True after a returned step and after a write that raised from inside."""
    from robot_fakes import FakeClock, FakeFollower, toy_fk

    from src.robot.driver import MotionStopped, SerialFault
    from src.robot.safety import make_safety
    from src.robot.units import UnitMap

    clock = FakeClock()
    fol = FakeFollower()
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk, clock=clock), clock=clock)
    arm.connect()
    q = arm.read_state()
    with pytest.raises(MotionStopped):
        arm.step(q, frame_time=clock(), stop_fn=lambda: True)
    assert not arm.goal_write_attempted and fol.sent == []
    arm.step(q, frame_time=clock(), stop_fn=lambda: False)
    assert arm.goal_write_attempted and len(fol.sent) == 1
    fol.fail_write = True
    with pytest.raises(SerialFault):
        arm.step(q, frame_time=clock())
    assert arm.goal_write_attempted


# -- the start marker: the final stop check after its I/O, and a marker that cannot be saved -----
def _trial_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [row["kind"] for row in rows if str(row.get("kind", "")).startswith("trial")]


@pytest.mark.parametrize("sig", list(SIGNAL_KIND))
def test_stop_during_the_start_marker_blocks_the_first_goal_and_retracts_it(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch, sig: str
) -> None:
    """(a) The stop is recorded while the first goal's durable ``trial_started`` row is being
    saved (the real signal, right after the row reached metrics.jsonl): the final check after
    the marker I/O blocks the goal. Zero policy goals, the hold still runs, the marker is
    retracted by a durable row, and the attempt is ``not_started`` with the stop cause."""
    res, rows, probe, r = _pilot_session(tmp_path, sig, "marker", 1, monkeypatch=monkeypatch)
    kind = SIGNAL_KIND[sig]
    assert probe.fired and probe.goals_after_signal() == 0
    assert probe.queries == 1  # the first policy goal's tick: no policy goal was ever written
    after = probe.after_signal()
    assert after[0] == ("stop_check", "pre_write", True)  # the final check, right after the I/O
    assert ("write", "hold") in after
    (t,) = res["trials"]
    assert (t["outcome"], t["outcome_source"]) == ("not_started", "not_scored")
    assert not t["policy_started"] and t["policy_started_utc"] is None
    assert t["stop_reason"] == kind and t["fault"] is None and t["steps"] == 0
    assert "marker was retracted" in t["stop_detail"]
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    started, retraction = (
        next(row for row in rows if row["kind"] == k)
        for k in ("trial_started", "trial_start_retracted")
    )
    assert (retraction["cell"], retraction["trial_idx"], retraction["start_utc"]) == (
        started["cell"],
        started["trial_idx"],
        started["start_utc"],
    )
    assert retraction["reason"].startswith("stop recorded while the trial_started marker")
    c = res["summary"]["cells"]["PickPlace/seen"]
    assert (c["n"], c["not_started"]) == (0, 1)
    _assert_shutdown(res, r, kind)


def test_ctrl_c_raised_inside_the_marker_without_the_handler_is_not_started() -> None:
    """(a) Mock (no flag handler): a Ctrl-C raised from inside the marker write propagates, the
    goal is never written, the marker is retracted, and the attempt is not started."""
    from test_deploy import _trials

    rows: list[dict[str, Any]] = []

    def log_row(**row: Any) -> None:
        rows.append(row)
        if row["kind"] == "trial_started":
            raise KeyboardInterrupt

    with rig() as r:
        sent_before: list[int] = []
        real_act = r.client.act

        def act(*a: Any, **k: Any) -> Any:
            sent_before.append(len(r.fol.sent))
            return real_act(*a, **k)

        r.client.act = act  # type: ignore[method-assign]
        trials, end = _trials(r, ScriptedOperator(), log_row=log_row)
        (t,) = trials
        assert len(r.fol.sent) == sent_before[0] + 1  # only the hold after the first query
        assert t["outcome"] == "not_started" and t["stop_reason"] == "operator_abort"
        assert not t["policy_started"] and not r.arm.goal_write_attempted
        assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
        assert end == "operator_abort" and r.arm.deadman.tripped


@pytest.mark.parametrize("disk", ["marker_fail", "marker_fail_all"])
def test_a_marker_that_cannot_be_saved_blocks_the_first_goal_and_ends_the_session(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch, disk: str
) -> None:
    """(b) Saving the ``trial_started`` row raises ENOSPC: the first policy goal is never written,
    the arm holds, the attempt is ``not_started`` with the reason, and the session ends
    (``start_marker_failed``: a started slot without its marker could be re-rolled). Then the
    hold-and-support shutdown and torque off. ``marker_fail_all``: the disk stays full, so the
    retraction and the attempt's own row fail too; the session ends on that error, after the
    same shutdown, and record.json (written separately) still holds the attempt."""
    if disk == "marker_fail":
        res, rows, probe, r = _pilot_session(
            tmp_path, None, "marker", 1, action=disk, monkeypatch=monkeypatch
        )
    else:
        with pytest.raises(OSError, match="No space left"):
            _pilot_session(tmp_path, None, "marker", 1, action=disk, monkeypatch=monkeypatch)
        res = _record(tmp_path)
        # nothing was logged before the first goal's marker, and nothing could be after it
        assert not list(tmp_path.glob("*/metrics.jsonl"))
        rows = []
    if disk == "marker_fail":
        assert probe.fired and probe.goals_after_signal() == 0
        assert ("write", "hold") in probe.after_signal()
        assert probe.queries == 1
        assert res["session_end"].startswith("start_marker_failed: the trial_started marker")
        assert _trial_rows(rows) == ["trial_start_retracted", "trial"]  # no marker row
        r_fol = r.fol
    else:
        assert res["session_end"] == "error"
        assert _trial_rows(rows) == []  # nothing could be logged from the marker on
        r_fol = None
    (t,) = res["trials"]
    assert (t["outcome"], t["outcome_source"]) == ("not_started", "not_scored")
    assert not t["policy_started"] and t["policy_started_utc"] is None and t["steps"] == 0
    assert t["stop_reason"] == "start_marker_failed" and t["fault"] is None
    assert "No space left on device" in t["stop_detail"]
    assert "first policy goal was not sent" in t["stop_detail"]
    if disk == "marker_fail_all":
        assert "the retraction could not be saved" in t["stop_detail"]
    assert res["shutdown"] == "supported" and res["support_confirmed"] is True
    assert res["torque_off_unconfirmed"] is None
    if r_fol is not None:
        assert not r_fol.bus.torque_enabled


def test_mutation_without_the_post_marker_check_a_goal_follows_the_stop(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sensitivity: without the final check after the marker I/O (the mutation), a SIGHUP
    recorded during the marker is followed by the first policy goal and the trial counts as
    started: the test above depends on that check."""

    def no_recheck(self: DeployLoop, res: Any, *_: Any) -> bool:
        if self._pre_write_stop():
            return True
        self._policy_started(res)
        return False

    monkeypatch.setattr(DeployLoop, "_first_goal_check", no_recheck)
    res, rows, probe, _r = _pilot_session(tmp_path, "SIGHUP", "marker", 1, monkeypatch=monkeypatch)
    assert probe.goals_after_signal() == 1
    (t,) = res["trials"]
    assert t["policy_started"] and t["outcome"] == "failure"
    assert "trial_start_retracted" not in _trial_rows(rows)


def _real_eval_with_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sig: str | None, action: str
) -> tuple[Path, Probe]:
    probes: list[Probe] = []

    class SignalLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **{**k, "interrupts": P.INTERRUPTS})
            probes.append(Probe(self, sig, "marker", 1, action, monkeypatch))

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setattr(ev, "DeployLoop", SignalLoop)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen")
    with P.operator_signals():
        P.INTERRUPTS.tty_fd = 0
        rec = ev.main(argv)
    monkeypatch.setattr(ev, "DeployLoop", DeployLoop)  # the resumed run: no stop
    (probe,) = probes
    return rec, probe


@pytest.mark.parametrize("cut", ["none", "after_retraction", "before_retraction"])
def test_real_eval_stop_during_the_marker_is_not_started_and_resume_retries_the_slot(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch, cut: str
) -> None:
    """(a) ``wo07_real_eval --mock``: SIGTERM while slot 0's ``trial_started`` row is saved. No
    policy goal, the hold runs, slot 0 is ``not_started`` and the marker is retracted. The real
    ``--resume`` then retries slot 0 with the same placement:

    * ``none``: from the run's full logs;
    * ``after_retraction``: the run was killed right after the retraction row (metrics.jsonl cut
      there, record.json without the attempt): ``load_prior`` honours the retraction;
    * ``before_retraction`` (control): killed between the marker and its retraction, only the
      bare marker survives; the logs cannot show the goal was blocked, so the resume scores slot 0
      a ``session_lost`` failure and does not re-run it (doubt = started)."""
    rec, probe = _real_eval_with_probe(tmp_path, monkeypatch, "SIGTERM", "signal")
    assert probe.fired and probe.goals_after_signal() == 0
    assert probe.after_signal()[0][0] == "sync"  # the marker's sync to stable storage, then
    assert probe.after_signal()[1] == ("stop_check", "pre_write", True)  # the final check
    assert ("write", "hold") in probe.after_signal()
    run_dir = rec.parent
    r1 = json.loads(rec.read_text())["results"]
    assert r1["session_end"].startswith("hangup: SIGTERM")
    assert (r1["shutdown"], r1["support_confirmed"]) == ("supported", True)
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (t0,) = r1["trials"]
    assert (t0["trial_idx"], t0["outcome"], t0["policy_started"]) == (0, "not_started", False)
    assert t0["stop_reason"] == "hangup" and "SIGTERM" in t0["stop_detail"]
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    assert ev.done_indices(r1["trials"], "seen") == set()
    if cut != "none":  # simulate the kill: nothing after the cut reached the disk
        i = next(i for i, x in enumerate(rows) if x.get("kind") == "trial_start_retracted")
        kept = rows[: i + 1] if cut == "after_retraction" else rows[:i]
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in kept))
        data = json.loads(rec.read_text())
        data["results"]["trials"] = []
        rec.write_text(json.dumps(data))
        _settings, prior = ev.load_prior(run_dir)
        if cut == "after_retraction":
            assert prior == []  # the retraction cancels the bare marker
        else:
            ((lost),) = prior
            assert lost["fault"]["kind"] == "session_lost" and lost["policy_started"]

    rec2 = ev.main(_args(tmp_path, "--resume", run_dir.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["resumed_from"] == run_dir.name and r2["session_end"] == "completed"
    by_idx: dict[int, list[dict[str, Any]]] = {}
    for t in r2["trials"]:
        by_idx.setdefault(t["trial_idx"], []).append(t)
    c = r2["summary"]["cells"]["PickPlace/seen"]
    if cut == "before_retraction":
        assert [t["trial_idx"] for t in r2["trials"]] == [0, 1]
        assert by_idx[0][0]["fault"]["kind"] == "session_lost"  # never re-run
        assert (c["n"], c["faults_by_kind"]) == (2, {"session_lost": 1})
        return
    expected = [0, 0, 1] if cut == "none" else [0, 1]
    assert [t["trial_idx"] for t in r2["trials"]] == expected
    retried = by_idx[0][-1]
    assert retried["policy_started"] and retried["outcome"] in ("success", "failure")
    assert retried["placement"] == t0["placement"]  # the same placement
    assert (c["n"], c["not_started"]) == (2, 1 if cut == "none" else 0)
    assert r2["summary"]["complete"]


def test_real_eval_marker_failure_ends_the_session_and_resume_retries_the_slot(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) ``wo07_real_eval --mock``: slot 0's ``trial_started`` row cannot be saved (ENOSPC).
    No policy goal, the hold runs, slot 0 is ``not_started`` with the reason, the session ends
    ``start_marker_failed`` with the support shutdown and torque off; ``--resume`` retries slot 0
    with the same placement."""
    rec, probe = _real_eval_with_probe(tmp_path, monkeypatch, None, "marker_fail")
    assert probe.fired and probe.goals_after_signal() == 0
    assert ("write", "hold") in probe.after_signal()
    r1 = json.loads(rec.read_text())["results"]
    assert r1["session_end"].startswith("start_marker_failed: ")
    assert (r1["shutdown"], r1["support_confirmed"]) == ("supported", True)
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (t0,) = r1["trials"]
    assert (t0["outcome"], t0["policy_started"], t0["stop_reason"]) == (
        "not_started",
        False,
        "start_marker_failed",
    )
    assert "No space left on device" in t0["stop_detail"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows) == ["trial_start_retracted", "trial"]
    assert not r1["summary"]["complete"]
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 0, 1]
    assert r2["trials"][1]["placement"] == t0["placement"] and r2["trials"][1]["policy_started"]


# -- review round 3: freshness after the marker, the window after the final check, the pending ---
STALE_CASES = {  # case -> (rig kwargs, marker delay s, stop reason)
    "frame_350ms": ({}, 0.35, "deadman"),  # past deadman_frame_timeout_ms (300)
    "frame_1500ms": ({}, 1.5, "deadman"),  # past both 300 ms and max_chunk_age_ms (800)
    # the chunk check alone: a slow query (350 ms) + a 200 ms marker > a 500 ms chunk limit,
    # while this tick's frame is only 200 ms old
    "chunk_550ms": ({"infer_s": 0.35, "max_chunk_age_ms": 500}, 0.2, "stale_chunk"),
}


def _slow_marker_trial(
    case: str, *, skip_recheck: bool = False, monkeypatch: pytest.MonkeyPatch | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, Rig, int]:
    """One mock ``run_trials`` trial whose ``trial_started`` row takes ``delay`` on the fake clock
    (a slow / stalled log volume). Returns (trials, rows, end, rig, policy goals written)."""
    from test_deploy import _trials

    kw, delay, _reason = STALE_CASES[case]
    if skip_recheck:  # the mutation: no freshness re-check after the marker
        assert monkeypatch is not None
        monkeypatch.setattr(DeployLoop, "_fresh_after_marker", lambda *a, **k: None)
    rows: list[dict[str, Any]] = []
    with rig(**kw) as r:

        def log_row(**row: Any) -> None:
            if row["kind"] == "trial_started":
                r.clock.advance(delay)
            rows.append(row)

        sent_at_query: list[int] = []
        real_act = r.client.act

        def act(*a: Any, **k: Any) -> Any:
            sent_at_query.append(len(r.fol.sent))
            return real_act(*a, **k)

        hold_writes = [0]
        hold = r.arm.hold

        def hold_counted(*a: Any, **k: Any) -> None:
            before = len(r.fol.sent)
            hold(*a, **k)
            hold_writes[0] += len(r.fol.sent) - before

        r.client.act = act  # type: ignore[method-assign]
        r.arm.hold = hold_counted  # type: ignore[method-assign]
        trials, end = _trials(r, ScriptedOperator(), log_row=log_row)
        goals = len(r.fol.sent) - sent_at_query[0] - hold_writes[0]
        return trials, rows, end, r, goals


@pytest.mark.parametrize("case", list(STALE_CASES))
def test_a_frame_or_chunk_gone_stale_during_the_marker_blocks_the_first_goal(case: str) -> None:
    """F33-R2-1: the marker I/O sits between the frame / chunk checks and the write. A marker that
    takes 350 ms or 1.5 s ages this tick's frame past the deadman's 300 ms; one that takes 200 ms
    after a 350 ms query ages the chunk past a 500 ms limit. The re-check on the clock after the
    I/O trips the deadman (HOLD): zero policy goals, the marker retracted, ``not_started`` (the
    slot is retried on a resume), with the marker's duration in the stop detail."""
    trials, rows, end, r, goals = _slow_marker_trial(case)
    reason = STALE_CASES[case][2]
    assert goals == 0
    (t,) = trials
    assert (t["outcome"], t["outcome_source"]) == ("not_started", "not_scored")
    assert not t["policy_started"] and t["policy_started_utc"] is None and t["steps"] == 0
    assert t["stop_reason"] == reason and t["fault"] is None
    assert "start marker took" in t["stop_detail"] and "not sent" in t["stop_detail"]
    assert t["deadman_response"] == DeadmanResponse.HOLD.value
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    retraction = next(row for row in rows if row["kind"] == "trial_start_retracted")
    assert "DeadmanTripped after the trial_started marker" in retraction["reason"]
    assert end == "completed"  # the HOLD stop was acknowledged (scripted 'resume')
    assert not r.arm.goal_write_attempted


@pytest.mark.parametrize("case", ["frame_350ms", "chunk_550ms"])
def test_mutation_without_the_freshness_recheck_a_stale_first_goal_is_sent(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Sensitivity: without the re-check after the marker (the mutation), the same slow marker is
    followed by the first policy goal on a stale frame / chunk, and the trial counts as started."""
    trials, rows, _end, _r, goals = _slow_marker_trial(
        case, skip_recheck=True, monkeypatch=monkeypatch
    )
    assert goals >= 1
    (t,) = trials
    assert t["policy_started"] and t["steps"] >= 1
    assert "trial_start_retracted" not in _trial_rows(rows)


def test_real_eval_stale_frame_after_the_marker_resume_retries_the_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F33-R2-1 end to end (``wo07_real_eval --mock --fake-clock``): slot 0's ``trial_started``
    row takes 400 ms. The deadman trips (HOLD) before the first goal, slot 0 is ``not_started``,
    the scripted operator resumes, slot 1 runs; ``--resume`` retries slot 0 with the same
    placement."""
    loops: list[DeployLoop] = []

    class SlowMarkerLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            loops.append(self)

    real_log = Run.log_metrics
    slowed = [0]

    def log_metrics(run: Run, step: int | None = None, **row: Any) -> None:
        if row.get("kind") == "trial_started" and not slowed[0]:
            slowed[0] = 1
            loops[0].clock.advance(0.4)  # type: ignore[attr-defined]
        real_log(run, step, **row)

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setattr(ev, "DeployLoop", SlowMarkerLoop)
    monkeypatch.setattr(Run, "log_metrics", log_metrics)
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    r1 = json.loads(rec.read_text())["results"]
    t0, t1 = r1["trials"]
    assert (t0["trial_idx"], t0["outcome"], t0["stop_reason"]) == (0, "not_started", "deadman")
    assert not t0["policy_started"] and "start marker took 400 ms" in t0["stop_detail"]
    assert t1["trial_idx"] == 1 and t1["policy_started"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows)[:3] == ["trial_started", "trial_start_retracted", "trial"]
    assert ev.done_indices(r1["trials"], "seen") == {1}
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 1, 0]
    assert r2["trials"][2]["placement"] == t0["placement"] and r2["trials"][2]["policy_started"]


@pytest.mark.parametrize("where", ["final_check", "after_final_check"])
def test_ctrl_c_after_the_marker_before_the_write_is_not_started(
    monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """F33-R2: mock (no flag handler), a Ctrl-C after the marker was saved but before the write
    began: raised inside the final stop check, or right after ``stop_fn`` returned False (the
    window before ``goal_write_attempted`` is set). Nothing was sent: the stamp is retracted,
    the attempt is ``not_started`` (``operator_abort``), not a started failure."""
    from test_deploy import _trials

    rows: list[dict[str, Any]] = []
    with rig() as r:
        if where == "final_check":
            calls = [0]
            real_check = r.loop._pre_write_stop

            def check() -> bool:
                if r.loop.on_policy_start is not None and any(
                    row["kind"] == "trial_started" for row in rows
                ):
                    calls[0] += 1
                    if calls[0] == 1:
                        raise KeyboardInterrupt
                return real_check()

            r.loop._pre_write_stop = check  # type: ignore[method-assign]
        else:
            real_step = r.arm.step

            def step(*a: Any, stop_fn: Any = None, **k: Any) -> Any:
                if not isinstance(stop_fn, functools.partial):  # not the first policy goal
                    return real_step(*a, stop_fn=stop_fn, **k)

                def then_ctrl_c() -> bool:
                    if stop_fn():
                        return True
                    raise KeyboardInterrupt  # after False, before goal_write_attempted

                return real_step(*a, stop_fn=then_ctrl_c, **k)

            r.arm.step = step  # type: ignore[method-assign]
        sent_at_query: list[int] = []
        real_act = r.client.act

        def act(*a: Any, **k: Any) -> Any:
            sent_at_query.append(len(r.fol.sent))
            return real_act(*a, **k)

        r.client.act = act  # type: ignore[method-assign]
        trials, end = _trials(r, ScriptedOperator(), log_row=lambda **row: rows.append(row))
        assert len(r.fol.sent) == sent_at_query[0] + 1  # only the hold after the first query
        (t,) = trials
        assert t["outcome"] == "not_started" and t["stop_reason"] == "operator_abort"
        assert not t["policy_started"] and not r.arm.goal_write_attempted
        assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
        assert end == "operator_abort"


def test_a_retracted_start_is_never_recorded_session_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F33-R1: the retraction clears ``run_trials``' pending start. A stop flag recorded during
    the marker blocks the goal (retracted); then recording the attempt raises (a bug in
    ``trial_record``). The exception propagates, and no ``session_lost`` / started record is
    appended or logged for the attempt that never sent a goal."""
    from test_deploy import _trials

    from src.robot import client
    from src.robot.prompts import Interrupts

    rows: list[dict[str, Any]] = []
    with rig() as r:
        flags = Interrupts()
        r.loop.interrupts = flags

        def log_row(**row: Any) -> None:
            rows.append(row)
            if row["kind"] == "trial_started":
                flags.ctrl_c = True  # the Ctrl-C flag, recorded during the marker I/O

        real_record = client.trial_record
        calls = [0]

        def trial_record(*a: Any, **k: Any) -> dict[str, Any]:
            calls[0] += 1
            if calls[0] == 2:  # the attempt's own record (the first call is the marker)
                raise RuntimeError("trial_record bug")
            return real_record(*a, **k)

        monkeypatch.setattr(client, "trial_record", trial_record)
        trials: list[dict[str, Any]] = []
        with pytest.raises(RuntimeError, match="trial_record bug"):
            _trials(r, ScriptedOperator(), log_row=log_row, trials=trials)
    assert trials == []  # no session_lost failure for the retracted attempt
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted"]


def test_real_eval_disk_full_including_record_json_resume_retries_the_slot(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F33-R5: the disk stays full from slot 0's marker on, for metrics.jsonl AND record.json.
    The first goal is never written, the hold-and-support shutdown and torque off still run,
    then the error propagates. The surviving record.json is the last good flush (no attempt in
    it); once the disk is freed, ``--resume`` retries slot 0 with the same placement."""
    probes: list[Probe] = []

    class SignalLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **{**k, "interrupts": P.INTERRUPTS})
            probes.append(Probe(self, None, "marker", 1, "marker_fail_all", monkeypatch))

    real_write = Run._write

    def write(run: Run) -> None:
        if probes and probes[0].fired:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_write(run)

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setattr(ev, "DeployLoop", SignalLoop)
    monkeypatch.setattr(Run, "_write", write)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen")
    with P.operator_signals(), pytest.raises(OSError, match="No space left"):
        P.INTERRUPTS.tty_fd = 0
        ev.main(argv)
    (probe,) = probes
    assert probe.fired and probe.goals_after_signal() == 0
    assert ("write", "hold") in probe.after_signal()
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (rec,) = tmp_path.glob("*/record.json")
    r1 = json.loads(rec.read_text())["results"]
    assert not r1.get("trials")  # the attempt never reached record.json
    monkeypatch.undo()  # the disk was freed (and the session's patches are gone)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 1]
    plan = ev.G.sample_plan("PickPlace", "seen", 2, int(r1["session"]["slot_seed"]))
    assert r2["trials"][0]["placement"] == plan[0] and r2["trials"][0]["policy_started"]


# -- owner 2026-11-09 (PR #32 sync): the start / retraction rows are synced to stable storage ----
def test_real_eval_start_marker_is_synced_before_the_final_checks_and_the_first_goal(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``wo07_real_eval --mock``: the ``trial_started`` row is appended, then synced (F_FULLFSYNC
    on macOS, fsync on Linux), and only then come the final stop check and the first policy goal
    (the freshness re-check sits between them and logs nothing). The run's record names the
    synced rows and the method; ordinary rows are never synced."""
    rec, probe = _real_eval_with_probe(tmp_path, monkeypatch, None, "none")
    log = probe.log
    i = log.index(("row", "trial_started"))
    j = next(k for k in range(i + 1, len(log)) if log[k][0] != "sync")
    want = "F_FULLFSYNC" if sys.platform == "darwin" else "fsync"
    assert log[i + 1 : j] == [("sync", want)]  # the marker's own sync (the dir: session row)
    assert log[j] == ("stop_check", "pre_write", False)  # the final check, after the sync
    assert log[j + 1] == ("write", "goal")  # the first policy goal
    r1 = json.loads(rec.read_text())["results"]
    assert r1["durability"]["sync"] == want
    assert r1["durability"]["synced_rows"] == ["trial_started", "trial_start_retracted", "session"]
    decisions = [e for e in log if e == ("row", "decision")]
    assert decisions and not any(  # no decision row is followed by a sync
        log[k + 1][0] == "sync" for k, e in enumerate(log[:-1]) if e == ("row", "decision")
    )
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert rows[0]["kind"] == "session" and rows[0]["session"]["session_id"]
    marks = [r["policy_start_mark_ms"] for r in rows if "policy_start_mark_ms" in r]
    assert len(marks) == 2  # one per trial: its first decision


def test_real_eval_a_failed_marker_sync_blocks_the_first_goal_and_resume_retries_the_slot(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``trial_started`` row reaches metrics.jsonl but its sync raises EIO: handled exactly
    like a failed marker write. No policy goal, the hold runs, slot 0 is ``not_started``
    (``start_marker_failed``, the error in the detail), the marker is retracted (that row is
    synced), the session ends ``start_marker_failed`` with the support shutdown and torque off;
    ``--resume`` retries slot 0 with the same placement."""
    rec, probe = _real_eval_with_probe(tmp_path, monkeypatch, None, "sync_fail")
    assert probe.fired and probe.goals_after_signal() == 0
    after = probe.after_signal()
    assert after[0] == ("sync", "EIO") and ("write", "hold") in after
    k = after.index(("row", "trial_start_retracted"))
    assert after[k + 1][0] == "sync" and after[k + 1][1] != "EIO"  # the retraction is synced
    r1 = json.loads(rec.read_text())["results"]
    assert r1["session_end"].startswith("start_marker_failed: ")
    assert (r1["shutdown"], r1["support_confirmed"]) == ("supported", True)
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (t0,) = r1["trials"]
    assert (t0["outcome"], t0["policy_started"], t0["stop_reason"]) == (
        "not_started",
        False,
        "start_marker_failed",
    )
    assert "Input/output error" in t0["stop_detail"] and "not sent" in t0["stop_detail"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    _settings, prior = ev.load_prior(rec.parent)
    assert [t["outcome"] for t in prior] == ["not_started"]  # the retraction cancels the marker
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 0, 1]
    assert r2["trials"][1]["placement"] == t0["placement"] and r2["trials"][1]["policy_started"]


@pytest.mark.parametrize("lost", ["record_missing", "record_truncated", "metrics_missing"])
def test_real_eval_resumes_after_a_power_loss_took_one_of_the_two_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: str
) -> None:
    """Power loss: record.json missing or cut short, the session is rebuilt from metrics.jsonl's
    synced ``session`` row and the trials from its rows (the torque off of that run counts as
    unconfirmed: unknown). metrics.jsonl missing: record.json's trials stand. Either way the
    started slot is not re-run, slot 1 runs, the session completes."""
    from test_real_eval import QuitAfter

    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))  # 1 trial, then quit
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    r1 = json.loads(rec.read_text())["results"]
    assert [t["trial_idx"] for t in r1["trials"]] == [0] and r1["trials"][0]["policy_started"]
    if lost == "record_missing":
        rec.unlink()
    elif lost == "record_truncated":
        rec.write_text(rec.read_text()[: len(rec.read_text()) // 2])
    else:
        (rec.parent / "metrics.jsonl").unlink()
    settings, prior = ev.load_prior(rec.parent)
    assert settings["session_id"] == r1["session"]["session_id"]
    assert [t["trial_idx"] for t in prior] == [0] and prior[0]["policy_started"]
    assert bool(settings.get("prior_torque_off_unconfirmed")) == (lost != "metrics_missing")
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["trials"]] == [0, 1]  # slot 0 never re-run
    assert r2["resumed_from"] == rec.parent.name


def _started_idx(trials: list[dict[str, Any]]) -> list[int]:
    return [int(t["trial_idx"]) for t in trials if t.get("policy_started")]


def _two_run_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int) -> tuple[Path, Path]:
    """(run 1, run 2) of an ``n``-trial seen session: run 1 scores slot 0 and quits; run 2
    resumes it and scores slot 1 and quits (``n`` > 2) or completes."""
    from test_real_eval import QuitAfter

    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))  # 1 trial, then quit
    rec1 = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", str(n), "--cells", "seen"))
    rec2 = ev.main(_args(tmp_path, "--resume", rec1.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert _started_idx(r2["trials"]) == [0, 1] and r2["resumed_from"] == rec1.parent.name
    return rec1.parent, rec2.parent


@pytest.mark.parametrize("lost", ["record_missing", "record_truncated"])
def test_real_eval_power_loss_of_a_resumed_runs_record_keeps_the_earlier_runs_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: str
) -> None:
    """S32-D1: a resumed run's metrics.jsonl never holds the earlier runs' trials (only its
    record.json does). Its record.json lost, the rebuild takes them from the run it resumed from
    (``resumed_from`` of its synced session row): slot 0 (run 1) and slot 1 (run 2) stand, and
    the next resume runs slot 2 only; run 1's score of slot 0 is the session's."""
    run1, run2 = _two_run_chain(tmp_path, monkeypatch, 3)
    rec = run2 / "record.json"
    if lost == "record_missing":
        rec.unlink()
    else:
        rec.write_text(rec.read_text()[: len(rec.read_text()) // 2])
    settings, prior = ev.load_prior(run2)
    assert _started_idx(prior) == [0, 1] and settings["prior_torque_off_unconfirmed"]
    r1 = json.loads((run1 / "record.json").read_text())["results"]
    assert prior[0]["start_utc"] == r1["trials"][0]["start_utc"]  # run 1's record of slot 0
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec3 = ev.main(_args(tmp_path, "--resume", run2.name))
    r3 = json.loads(rec3.read_text())["results"]
    assert r3["session_end"] == "completed" and r3["summary"]["complete"]
    assert _started_idx(r3["trials"]) == [0, 1, 2]  # no slot scored twice
    assert r3["trials"][0]["start_utc"] == r1["trials"][0]["start_utc"]


def test_real_eval_power_loss_rebuild_refuses_when_the_parent_run_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S32-D1: the resumed run's record lost AND the run it resumed from gone: the earlier
    trials cannot be rebuilt, so the resume refuses (it would re-run them)."""
    import shutil

    run1, run2 = _two_run_chain(tmp_path, monkeypatch, 3)
    (run2 / "record.json").unlink()
    shutil.rmtree(run1)
    with pytest.raises(SystemExit, match="which is gone"):
        ev.load_prior(run2)


@pytest.mark.parametrize("chain", [1, 2])
def test_real_eval_resume_needs_only_the_synced_rows_and_the_synced_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chain: int
) -> None:
    """S32-D4: power lost right after the last ``trial_started`` sync of the newest run: on disk
    are metrics.jsonl up to that synced offset (every later decision / trial / capture row gone)
    and record.json as of its last write before it. ``load_prior`` and ``--resume`` still never
    re-run a started slot, and the in-flight trial is a ``session_lost`` failure. Over a single
    run and a two-run chain (the newest run cut)."""
    snaps: dict[Path, list[tuple[str, int, str]]] = {}
    last_record: dict[Path, str] = {}
    real_durable, real_write = Run.log_durable, Run._write

    def write(run: Run) -> None:
        real_write(run)
        last_record[run.dir] = run.record_path.read_text()

    def log_durable(run: Run, step: int | None = None, **row: Any) -> str:
        method = real_durable(run, step, **row)
        snaps.setdefault(run.dir, []).append(
            (str(row.get("kind")), run.metrics_path.stat().st_size, last_record[run.dir])
        )
        return method

    monkeypatch.setattr(Run, "_write", write)
    monkeypatch.setattr(Run, "log_durable", log_durable)
    n, in_flight = chain + 1, 1  # slot 1 is in flight at the cut (run 2's only trial: chain 2)
    if chain == 1:
        monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
        cut_dir = rec.parent
    else:
        _run1, cut_dir = _two_run_chain(tmp_path, monkeypatch, n)
    _kind, size, record_text = [s for s in snaps[cut_dir] if s[0] == "trial_started"][-1]
    metrics = cut_dir / "metrics.jsonl"
    with metrics.open("r+b") as f:  # the unsynced tail is gone
        f.truncate(size)
    (cut_dir / "record.json").write_text(record_text)
    monkeypatch.undo()
    rows = [json.loads(x) for x in metrics.read_text().splitlines()]
    assert rows[-1]["kind"] == "trial_started" and rows[-1]["trial_idx"] == in_flight
    _settings, prior = ev.load_prior(cut_dir)
    assert _started_idx(prior) == [0, in_flight]
    lost = prior[-1]
    assert lost["outcome"] == "failure" and lost["fault"]["kind"] == "session_lost"
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec_next = ev.main(_args(tmp_path, "--resume", cut_dir.name))
    r = json.loads(rec_next.read_text())["results"]
    assert r["session_end"] == "completed" and r["summary"]["complete"]
    started = _started_idx(r["trials"])
    assert sorted(started) == list(range(n)) and len(set(started)) == len(started)
    assert next(t for t in r["trials"] if t["trial_idx"] == in_flight)["fault"]["kind"] == (
        "session_lost"
    )


# -- owner 2026-11-10: real-arm journal writes on macOS require F_FULLFSYNC -----------------------
class _MacSyncs:
    """A simulated macOS under the session: ``fcntl.F_FULLFSYNC`` (51) present and succeeding
    (logged, no real flush; Linux CI has none) and ``os.fsync`` logged, each as
    (method, "file" | "dir", basename). ``fail(how)`` from then on: ``missing`` removes the
    attribute (fcntl without F_FULLFSYNC), an errno name makes fcntl(F_FULLFSYNC) raise it;
    ``heal()`` undoes it. ``("fail", how)`` marks the moment in the log."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import fcntl
        import stat

        self.log: list[tuple[str, ...]] = []
        self.errno: int | None = None
        self._mp, self._fcntl = monkeypatch, fcntl
        real_fcntl, real_fsync = fcntl.fcntl, os.fsync

        def name(fd: int) -> tuple[str, str]:
            kind = "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"
            try:
                path = os.readlink(f"/proc/self/fd/{fd}")  # Linux
            except OSError:
                buf = real_fcntl(fd, getattr(fcntl, "F_GETPATH", 50), bytes(1024))  # macOS
                path = bytes(buf).rstrip(b"\0").decode()
            return kind, Path(path).name

        def fake_fcntl(fd: int, cmd: int, *a: Any) -> Any:
            if cmd != 51:
                return real_fcntl(fd, cmd, *a)
            self.log.append(("F_FULLFSYNC", *name(fd)))
            if self.errno is not None:
                raise OSError(self.errno, os.strerror(self.errno))
            return 0

        def fsync(fd: int) -> None:
            self.log.append(("fsync", *name(fd)))
            real_fsync(fd)

        monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
        monkeypatch.setattr(fcntl, "fcntl", fake_fcntl)
        monkeypatch.setattr(os, "fsync", fsync)

    def fail(self, how: str) -> None:
        self.log.append(("fail", how))
        if how == "missing":  # through monkeypatch: its undo stays consistent (Linux: none)
            self._mp.delattr(self._fcntl, "F_FULLFSYNC", raising=False)
        else:
            self.errno = getattr(errno, how)

    def heal(self) -> None:
        self.log.append(("heal",))
        self.errno = None
        self._mp.setattr(self._fcntl, "F_FULLFSYNC", 51, raising=False)

    def file_fsyncs_after_fail(self) -> list[tuple[str, ...]]:
        i = next(k for k, e in enumerate(self.log) if e[0] == "fail")
        return [e for e in self.log[i:] if e[:2] == ("fsync", "file")]


def _as_hardware_on_darwin(monkeypatch: pytest.MonkeyPatch, mod: Any = ev) -> None:
    """Review F2: a mock-rig session of ``mod`` (wo07_real_eval or wo07_pilot) driven as the
    HARDWARE session on a simulated macOS: ``session`` gets ``mode="hardware"`` (what
    ``run_hardware`` passes) and the platform is darwin; the script's own ``journal_policy(mode)``
    then picks the policy, so a mutation of that choice (or of the mode string) fails. A
    ``--mock --resume`` is checked as the hardware resume (``check_resume_mode``) too."""
    darwin = functools.partial(runs_mod.journal_sync_policy, platform="darwin")
    monkeypatch.setattr(mod, "journal_sync_policy", darwin)
    real_session = mod.session
    if hasattr(mod, "check_resume_mode"):
        real_check = mod.check_resume_mode

        def check_resume_mode(run_dir: Path, mode: str) -> str:
            assert mode == "mock"
            return str(real_check(run_dir, "hardware"))

        monkeypatch.setattr(mod, "check_resume_mode", check_resume_mode)

    def session(*a: Any, **k: Any) -> Path:
        assert k.get("mode") == "mock"  # the mock rig's entry, now as the hardware one
        return Path(real_session(*a, **{**k, "mode": "hardware"}))

    monkeypatch.setattr(mod, "session", session)


def _darwin_real_arm_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str, persistent: bool
) -> tuple[Path | None, Probe, _MacSyncs, BaseException | None]:
    """``wo07_real_eval`` on the mock rig as a simulated macOS REAL-ARM session
    (``_as_hardware_on_darwin``: mode "hardware", the script's own policy choice): F_FULLFSYNC
    fails (``how``) from slot 0's ``trial_started`` row on, for that row's sync only or
    (``persistent``) for good.
    Returns (record.json or None if the session raised, the probe, the syncs, the exception)."""
    probes: list[Probe] = []
    syncs = _MacSyncs(monkeypatch)
    _as_hardware_on_darwin(monkeypatch)

    class SignalLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **{**k, "interrupts": P.INTERRUPTS})
            probes.append(Probe(self, None, "marker", 1, "none", monkeypatch))

    real_durable = Run.log_durable
    fired = [False]

    def log_durable(run: Run, step: int | None = None, **row: Any) -> str:
        if row.get("kind") != "trial_started" or fired[0]:
            return real_durable(run, step, **row)
        fired[0] = True
        syncs.fail(how)
        try:
            return real_durable(run, step, **row)
        finally:
            if not persistent:
                syncs.heal()

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setattr(ev, "DeployLoop", SignalLoop)
    monkeypatch.setattr(Run, "log_durable", log_durable)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen")
    rec: Path | None = None
    err: BaseException | None = None
    with P.operator_signals():
        P.INTERRUPTS.tty_fd = 0
        try:
            rec = ev.main(argv)
        except OSError as e:
            err = e
    monkeypatch.setattr(ev, "DeployLoop", DeployLoop)
    monkeypatch.setattr(Run, "log_durable", real_durable)
    (probe,) = probes
    return rec, probe, syncs, err


def _goals_after_the_marker(probe: Probe) -> int:
    i = probe.log.index(("row", "trial_started"))
    return sum(e == ("write", "goal") for e in probe.log[i:])


@pytest.mark.parametrize("how", ["missing", "ENOTSUP", "EIO"])
def test_darwin_real_arm_unsupported_f_fullfsync_blocks_the_first_goal_no_fsync_fallback(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Owner 2026-11-10. A simulated macOS real-arm session; slot 0's ``trial_started`` row
    reaches metrics.jsonl but its F_FULLFSYNC is unsupported (fcntl has no F_FULLFSYNC, or it
    raises ENOTSUP) or fails (EIO). ``os.fsync`` is NOT run on the journal file instead; the first
    policy goal is blocked (zero policy writes), the hold runs, slot 0 is ``not_started`` with the
    reason (the method and errno), the retraction is synced with F_FULLFSYNC, and the session
    takes the existing safe shutdown (hold, support, torque off). Every journal file sync of the
    session was an F_FULLFSYNC, and the record says so. ``--resume`` retries slot 0."""
    rec, probe, syncs, err = _darwin_real_arm_session(tmp_path, monkeypatch, how, False)
    assert err is None and rec is not None
    assert syncs.file_fsyncs_after_fail() == []  # no fsync stood in for the failed F_FULLFSYNC
    assert not [e for e in syncs.log if e[:2] == ("fsync", "file")]  # nor anywhere else
    assert _goals_after_the_marker(probe) == 0  # the first policy goal was never written
    assert ("write", "hold") in probe.log[probe.log.index(("row", "trial_started")) :]
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    rec_data = json.loads(rec.read_text())
    r1 = rec_data["results"]
    assert r1["mode"] == "hardware"
    assert rec_data["sync"] == {
        "policy": "full_fsync_required",
        "methods": ["F_FULLFSYNC"],
        "dir_methods": ["F_FULLFSYNC"],
    }
    assert r1["durability"]["sync_policy"] == "full_fsync_required"
    assert r1["session_end"].startswith("start_marker_failed: ")
    assert (r1["shutdown"], r1["support_confirmed"]) == ("supported", True)
    (t0,) = r1["trials"]
    assert (t0["outcome"], t0["policy_started"], t0["stop_reason"]) == (
        "not_started",
        False,
        "start_marker_failed",
    )
    code = "ENOTSUP" if how == "missing" else how
    assert "FullFsyncRequired" in t0["stop_detail"] and "F_FULLFSYNC" in t0["stop_detail"]
    assert f"[Errno {getattr(errno, code)}]" in t0["stop_detail"]
    assert "no fsync fallback" in t0["stop_detail"] and "not sent" in t0["stop_detail"]
    if how == "missing":
        assert "fcntl has no F_FULLFSYNC" in t0["stop_detail"]
    else:
        assert code in t0["stop_detail"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    heal = syncs.log.index(("heal",))
    assert ("F_FULLFSYNC", "file", "metrics.jsonl") in syncs.log[heal:]  # the retraction
    _settings, prior = ev.load_prior(rec.parent)
    assert [t["outcome"] for t in prior] == ["not_started"]
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))  # still "macOS, real arm"
    r2 = json.loads(rec2.read_text())
    assert r2["sync"]["policy"] == "full_fsync_required"
    assert r2["results"]["session_end"] == "completed" and r2["results"]["summary"]["complete"]
    assert [t["trial_idx"] for t in r2["results"]["trials"]] == [0, 0, 1]
    assert r2["results"]["trials"][1]["placement"] == t0["placement"]


@pytest.mark.parametrize("how", ["missing", "ENOTSUP"])
def test_darwin_real_arm_f_fullfsync_gone_for_good_still_never_moves_or_falls_back(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """As above, but F_FULLFSYNC stays unavailable from the marker on (the retraction, the
    attempt's record.json flush and the final record cannot be synced either). Still no fsync on
    any journal file, zero policy goals, the hold and the support shutdown with torque off; the
    attempt is in metrics.jsonl as ``not_started`` with the reason; the error then propagates
    (like the full disk of F33-R5) and record.json is the last good (synced) one. Once the
    filesystem is back, ``--resume`` retries slot 0 with the same placement."""
    rec, probe, syncs, err = _darwin_real_arm_session(tmp_path, monkeypatch, how, True)
    assert rec is None and isinstance(err, runs_mod.FullFsyncRequired)
    assert syncs.file_fsyncs_after_fail() == []
    assert _goals_after_the_marker(probe) == 0
    assert ("write", "hold") in probe.log[probe.log.index(("row", "trial_started")) :]
    assert not probe.loop.arm.follower.bus.torque_enabled  # type: ignore[attr-defined]
    (run_dir,) = [p.parent for p in tmp_path.glob("*/record.json")]
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert _trial_rows(rows) == ["trial_started", "trial_start_retracted", "trial"]
    (attempt,) = [r for r in rows if r["kind"] == "trial"]
    assert (attempt["outcome"], attempt["stop_reason"]) == ("not_started", "start_marker_failed")
    assert "FullFsyncRequired" in attempt["stop_detail"]
    assert "the retraction could not be saved" in attempt["stop_detail"]
    r1 = json.loads((run_dir / "record.json").read_text())
    assert r1["sync"]["policy"] == "full_fsync_required" and not r1["results"].get("trials")
    syncs.heal()
    rec2 = ev.main(_args(tmp_path, "--resume", run_dir.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    started = [t for t in r2["trials"] if t["policy_started"]]
    assert [t["trial_idx"] for t in started] == [0, 1]  # slot 0 retried, never session_lost
    assert started[0]["placement"] == attempt["placement"]


def test_mutation_an_fsync_fallback_on_the_real_arm_journal_lets_the_first_goal_through(
    tmp_path: Path, terminal: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation the regression test above guards against: ``sync_fd`` back to the
    best-effort fallback whatever the policy. The same unsupported F_FULLFSYNC is then papered
    over by an ``os.fsync`` of metrics.jsonl and the first policy goal IS written, so each of the
    test's assertions (no file fsync after the failure, zero goals, not_started) would fail."""
    real_sync = runs_mod.sync_fd

    def fallback(fd: int, policy: Any = None) -> str:
        return real_sync(fd)  # best effort: F_FULLFSYNC, else os.fsync

    monkeypatch.setattr(runs_mod, "sync_fd", fallback)
    rec, probe, syncs, err = _darwin_real_arm_session(tmp_path, monkeypatch, "missing", False)
    assert err is None and rec is not None
    assert ("fsync", "file", "metrics.jsonl") in syncs.file_fsyncs_after_fail()
    assert _goals_after_the_marker(probe) > 0
    r1 = json.loads(rec.read_text())["results"]
    assert r1["trials"][0]["policy_started"] and r1["session_end"] == "completed"


def test_hardware_session_refuses_at_preflight_without_f_fullfsync_before_anything_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``wo07_real_eval --hardware`` and ``wo07_pilot --hardware`` on a simulated macOS: the
    preflight syncs a temp file in the runs directory under the real-arm policy before the
    policy, camera or arm is opened; without F_FULLFSYNC the session refuses there. With it, the
    preflight passes (and leaves no file behind) and the session goes on to its next check."""
    from test_deploy import pilot
    from test_real_eval import _with_cfg

    syncs = _MacSyncs(monkeypatch)
    darwin = functools.partial(runs_mod.journal_sync_policy, platform="darwin")
    monkeypatch.setattr(ev, "journal_sync_policy", darwin)  # the script's journal_policy runs
    monkeypatch.setattr(pilot, "journal_sync_policy", darwin)
    intr = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "dist": [0.0] * 5}
    _with_cfg(monkeypatch, lambda c: c["cameras"]["front"].update(intrinsics=intr))
    monkeypatch.setattr(ev, "require_terminal", lambda _what: None)
    monkeypatch.setattr(pilot, "require_terminal", lambda _what: None)
    monkeypatch.setattr("builtins.input", lambda _msg="": "go")

    def opened(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("opened before the journal preflight passed")

    monkeypatch.setattr(ev, "_hardware_backend", opened)
    monkeypatch.setattr(pilot, "PolicyClient", opened)
    runs = tmp_path / "runs"
    ev_argv = ["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"]
    ev_argv += ["--protocol", "0.2", "--runs-dir", str(runs)]
    pilot_argv = ["--preset", "real", "--hardware", "--i-am-present", "--runs-dir", str(runs)]
    pilot_argv.append("--allow-uncalibrated-camera")
    with pytest.raises(SystemExit, match="must use the same placements"):  # the next check
        ev.main([*ev_argv, "--slot-seed", "7"])
    assert ("F_FULLFSYNC", "file") in [e[:2] for e in syncs.log] and list(runs.iterdir()) == []
    with pytest.raises(AssertionError, match="opened before"):  # past the preflight
        pilot.main(pilot_argv)
    syncs.fail("missing")
    with pytest.raises(SystemExit, match="refusing the hardware session; nothing was opened") as e:
        ev.main(ev_argv)
    assert "fcntl has no F_FULLFSYNC" in str(e.value) and "FullFsyncRequired" in str(e.value)
    with pytest.raises(SystemExit, match="refusing the hardware session; nothing was opened"):
        pilot.main(pilot_argv)
    assert syncs.file_fsyncs_after_fail() == []
    assert list(runs.iterdir()) == []  # no probe file, no run


def _file_syncs(syncs: _MacSyncs, method: str) -> set[str]:
    return {e[2] for e in syncs.log if e[:2] == (method, "file")}


def test_darwin_real_arm_capture_journal_is_f_fullfsynced_never_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F1/F2 (round 2026-11-10): a real-0.1 session with the evaluation capture, as the
    hardware session on a simulated macOS. Every capture journal file sync (provenance, the new
    dataset's meta and calibration copy, the staged journal index, the saved episode's parquet /
    meta files, the end-state stills) is an F_FULLFSYNC and none is an ``os.fsync``. A mutation
    that drops the session policy anywhere on the capture's path (``open_capture``,
    ``EvalCapture.open``, ``sync_tree``) or picks the mock policy for ``mode="hardware"`` fails."""
    from test_protocol_real import real_args

    syncs = _MacSyncs(monkeypatch)
    _as_hardware_on_darwin(monkeypatch)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec = ev.main(real_args(tmp_path / "runs", "--task", "PickPlace", "--trials", "1"))
    data = json.loads(rec.read_text())
    r = data["results"]
    assert r["mode"] == "hardware" and r["session_end"] == "completed"
    assert data["sync"]["policy"] == "full_fsync_required"
    assert all(t["capture"]["episode_index"] is not None for t in r["trials"])
    assert all("sync_error" not in t["capture"] for t in r["trials"])
    assert [e for e in syncs.log if e[:2] == ("fsync", "file")] == []
    full = _file_syncs(syncs, "F_FULLFSYNC")
    for name in ("calibration_follower.json", "info.json", "vla_rl_so101.json"):
        assert name in full, name
    for name in ("vla_rl_so101.json.tmp", "staged.json.tmp"):  # write_json: tmp synced, renamed
        assert name in full, name
    assert any(n.endswith(".parquet") for n in full)  # the saved episode's data / meta
    first = {e: syncs.log.index(("F_FULLFSYNC", "file", e)) for e in full}
    # the new dataset (meta, calibration copy) is synced at open, before the first trial
    assert first["calibration_follower.json"] < first["staged.json.tmp"]
    assert first["info.json"] < first["staged.json.tmp"]
    assert {f"episode_000000_{c}.png" for c in ("front", "wrist")} <= full  # the stills


@pytest.mark.parametrize("how", ["missing", "ENOTSUP"])
def test_darwin_real_arm_capture_sync_failure_keeps_the_journal_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """Review F1 (round 2026-11-10): the first saved episode's sync (``sync_tree`` in
    ``EvalCapture.finish``) meets an unsupported F_FULLFSYNC: the capture block records the
    ``sync_error`` (FullFsyncRequired), the staged journal is kept for a resume, and no
    ``os.fsync`` stands in on any file."""
    from test_protocol_real import real_args

    from src.robot import recording

    syncs = _MacSyncs(monkeypatch)
    _as_hardware_on_darwin(monkeypatch)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    real_tree = recording.sync_tree
    fired = [False]

    def sync_tree(root: Path, *a: Any, **k: Any) -> int:
        if k.get("since") is None or fired[0]:  # only the first episode save's sync
            return int(real_tree(root, *a, **k))
        fired[0] = True
        syncs.fail(how)
        try:
            return int(real_tree(root, *a, **k))
        finally:
            syncs.heal()

    monkeypatch.setattr(recording, "sync_tree", sync_tree)
    rec = ev.main(real_args(tmp_path / "runs", "--task", "PickPlace", "--trials", "1"))
    r = json.loads(rec.read_text())["results"]
    assert fired[0] and r["mode"] == "hardware"
    cap0 = r["trials"][0]["capture"]
    assert cap0["episode_index"] == 0
    assert "FullFsyncRequired" in cap0["sync_error"] and "staged journal is kept" in (
        cap0["sync_error"]
    )
    sid = r["session"]["session_id"]
    assert recording.has_staged(tmp_path / "runs" / sid / "capture", r["trials"][0]["trial_id"])
    assert syncs.file_fsyncs_after_fail() == []
    assert [e for e in syncs.log if e[:2] == ("fsync", "file")] == []


def test_darwin_hardware_resume_preflights_before_the_capture_recovery_changes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F1 (round 2026-11-10): ``--resume --hardware`` of a real-0.1 session whose capture
    a kill left mid-save. On a simulated macOS without F_FULLFSYNC the journal preflight refuses
    FIRST: the capture is untouched (no quarantine, info.json and provenance as they were) and
    nothing ran ``os.fsync`` on a file. F_FULLFSYNC lost after the preflight: the recovery's own
    sync refuses the resume (the session policy reaches ``check_capture_resumable``), still no
    fsync. With F_FULLFSYNC, the recovery's writes (provenance pruned, info.json, stats.json)
    are F_FULLFSYNCed, never fsynced, before the session goes on to its next check
    (``--i-am-present`` missing: refused, nothing opened)."""
    runs = tmp_path / "runs"
    run_dir = _killed_real_session(runs, monkeypatch, "hardware")  # PR #32 hold: a hardware one
    settings, _ = ev.load_prior(run_dir)
    capture = runs / str(settings["session_id"]) / "capture"
    ds_root = capture / "dataset"
    prov_path = ds_root / "meta" / "vla_rl_so101.json"
    before = {p: p.read_bytes() for p in (prov_path, ds_root / "meta" / "info.json")}

    syncs = _MacSyncs(monkeypatch)
    monkeypatch.setattr(ev, "journal_sync_policy", functools.partial(
        runs_mod.journal_sync_policy, platform="darwin"))  # fmt: skip
    argv = ["--preset", "real", "--hardware", "--resume", run_dir.name, "--runs-dir", str(runs)]
    syncs.fail("missing")
    with pytest.raises(SystemExit, match="refusing the hardware session; nothing was opened"):
        ev.main(argv)
    assert {p: p.read_bytes() for p in before} == before
    assert not list(capture.glob("recovered-*")) and (ds_root / "images").exists()
    assert syncs.file_fsyncs_after_fail() == []

    syncs.heal()
    real_preflight = ev.preflight_journal

    def preflight_then_lost(runs_dir: Path) -> dict[str, Any]:
        out: dict[str, Any] = real_preflight(runs_dir)
        syncs.fail("ENOTSUP")  # the disk loses F_FULLFSYNC between the preflight and the recovery
        return out

    monkeypatch.setattr(ev, "preflight_journal", preflight_then_lost)
    with pytest.raises(
        SystemExit, match="--resume refused: .*F_FULLFSYNC required.*no fsync fallback"
    ):
        ev.main(argv)
    assert syncs.file_fsyncs_after_fail() == []
    monkeypatch.setattr(ev, "preflight_journal", real_preflight)

    syncs.heal()
    with pytest.raises(SystemExit) as e:
        ev.main(argv)  # preflight passes, the capture is recovered, then --i-am-present refuses
    assert e.value.code == 2
    assert list(capture.glob("recovered-*")) and not (ds_root / "images").exists()
    assert "3" not in json.loads(prov_path.read_text())["episodes"]  # pruned
    heal = len(syncs.log) - syncs.log[::-1].index(("heal",))  # after the last heal
    full = {e[2] for e in syncs.log[heal:] if e[:2] == ("F_FULLFSYNC", "file")}
    assert {"vla_rl_so101.json.tmp", "info.json", "stats.json"} <= full
    assert [e for e in syncs.log if e[:2] == ("fsync", "file")] == []


def _tree_digest(root: Path) -> dict[str, str]:
    """{relative path: sha256 of its bytes, or "dir"} of every entry under ``root``."""
    import hashlib

    return {
        str(p.relative_to(root)): (
            "dir" if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in sorted(root.rglob("*"))
    }


def _killed_real_session(runs: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> Path:
    """A finished real-0.1 session recorded as ``mode`` (hardware: the mock rig driven as the
    hardware session on a simulated macOS, ``_as_hardware_on_darwin``), then a kill mid-save of
    its capture and a provenance entry the recovery would prune: a resume's recovery would
    quarantine, prune and rewrite. Returns the run dir."""
    from test_pr2_review_fixes import _simulate_kill_mid_save
    from test_protocol_real import real_args

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        if mode == "hardware":
            _MacSyncs(mp)
            _as_hardware_on_darwin(mp)
        rec = ev.main(real_args(runs, "--task", "PickPlace", "--trials", "1"))
    assert json.loads(rec.read_text())["results"]["mode"] == mode
    settings, _ = ev.load_prior(rec.parent)
    ds_root = runs / str(settings["session_id"]) / "capture" / "dataset"
    prov_path = ds_root / "meta" / "vla_rl_so101.json"
    prov = json.loads(prov_path.read_text())
    prov["episodes"]["3"] = {"trial_id": "lost-in-the-kill"}
    prov_path.write_text(json.dumps(prov))
    _simulate_kill_mid_save(ds_root, 3)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    return rec.parent


@pytest.mark.parametrize(
    ("recorded", "flag"), [("hardware", "--mock"), ("mock", "--hardware")]
)  # fmt: skip
def test_a_resume_in_the_other_mode_is_refused_before_any_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded: str, flag: str
) -> None:
    """Owner, PR #32 hold (P1): ``--mock --resume`` of a HARDWARE session reached the capture
    recovery under the best-effort sync. A resume reads the recorded session mode read only and
    refuses a mismatch (either way) before the journal preflight, ``load_prior`` or the
    capture recovery: the whole runs tree (capture, metrics.jsonl, record.json) is unchanged."""
    runs = tmp_path / "runs"
    run_dir = _killed_real_session(runs, monkeypatch, recorded)
    before = _tree_digest(runs)
    argv = ["--preset", "real", flag, "--resume", run_dir.name, "--runs-dir", str(runs)]
    if flag == "--mock":  # the session's own step cap: nothing else would refuse it
        argv += ["--fake-clock", "--max-steps", "6"]
    refused: str | None = None
    try:
        ev.main(argv)
    except SystemExit as e:
        refused = str(e.code)
    assert _tree_digest(runs) == before, f"the resume changed the session's files ({refused})"
    assert refused is not None and f"recorded as a {recorded} session" in refused, refused
    assert refused.startswith("--resume refused: ")


def test_a_resume_without_a_recorded_mode_is_refused_before_any_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session whose record.json is lost and whose synced ``session`` row carries no mode (or
    whose record has none) is refused, not guessed; nothing changes. With the row's mode (rows
    written from this fix on) the rebuilt session's mode is the recorded one."""
    runs = tmp_path / "runs"
    run_dir = _killed_real_session(runs, monkeypatch, "hardware")
    assert ev.recorded_mode(run_dir) == "hardware"
    (run_dir / "record.json").unlink()
    assert ev.recorded_mode(run_dir) == "hardware"  # from the synced session row
    metrics = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    for row in rows:
        row.pop("mode", None)
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = _tree_digest(runs)
    argv = ["--preset", "real", "--mock", "--fake-clock", "--max-steps", "6", "--resume",
            run_dir.name, "--runs-dir", str(runs)]  # fmt: skip
    refused: str | None = None
    try:
        ev.main(argv)
    except SystemExit as e:
        refused = str(e.code)
    assert _tree_digest(runs) == before, f"the resume changed the session's files ({refused})"
    assert refused is not None and "records no session mode" in refused, refused


def test_precheck_capture_derives_the_sync_policy_from_the_recorded_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hardware session's capture recovery runs under the hardware journal policy (macOS:
    F_FULLFSYNC required), never under the best-effort one: ``precheck_capture`` takes the
    recorded mode, not a policy, and the recovery's writes are F_FULLFSYNCed, never fsynced."""
    runs = tmp_path / "runs"
    run_dir = _killed_real_session(runs, monkeypatch, "hardware")
    settings, _ = ev.load_prior(run_dir)
    syncs = _MacSyncs(monkeypatch)
    monkeypatch.setattr(ev, "journal_sync_policy", functools.partial(
        runs_mod.journal_sync_policy, platform="darwin"))  # fmt: skip
    seen: list[Any] = []
    real_check = ev.check_capture_resumable

    def check(*a: Any) -> Any:
        seen.append(a[-1])
        return real_check(*a)

    monkeypatch.setattr(ev, "check_capture_resumable", check)
    report = ev.precheck_capture(runs, settings, ev.recorded_mode(run_dir))
    assert report is not None and report["moved"]
    assert seen == [runs_mod.SyncPolicy.FULL_FSYNC_REQUIRED]
    assert [e for e in syncs.log if e[:2] == ("fsync", "file")] == []


def test_darwin_pilot_hardware_session_record_requires_f_fullfsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2 (round 2026-11-10): the pilot's hardware session on a simulated macOS: its Run
    (``pilot.journal_policy("hardware")``) requires F_FULLFSYNC; every record.json sync is one and
    none is an ``os.fsync``. The mock session on the same simulated macOS stays best effort."""
    syncs = _MacSyncs(monkeypatch)
    _as_hardware_on_darwin(monkeypatch, pilot)
    argv = ["--preset", "real", "--mock", "--fake-clock", "--trials", "1"]
    rec = pilot.main([*argv, "--runs-dir", str(tmp_path / "hw")])
    data = json.loads(rec.read_text())
    assert data["results"]["mode"] == "hardware" and data["results"]["session_end"] == "completed"
    assert data["sync"]["policy"] == "full_fsync_required"
    assert data["sync"]["methods"] == ["F_FULLFSYNC"]
    assert "record.json.tmp" in _file_syncs(syncs, "F_FULLFSYNC")
    assert [e for e in syncs.log if e[:2] == ("fsync", "file")] == []
    assert pilot.journal_policy("hardware") is runs_mod.SyncPolicy.FULL_FSYNC_REQUIRED
    assert pilot.journal_policy("mock") is runs_mod.SyncPolicy.BEST_EFFORT
