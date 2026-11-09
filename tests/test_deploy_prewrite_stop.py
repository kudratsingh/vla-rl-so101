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

            def sync_fd(fd: int) -> str:
                if self._sync_fail_armed:
                    self._sync_fail_armed = False
                    self.log.append(("sync", "EIO"))
                    raise OSError(errno.EIO, os.strerror(errno.EIO))
                method = real_sync(fd)
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
