"""Pre-write stop check in the WO-07 pilot / real-eval control loop (owner 2026-11-09).

``DeployLoop`` passes its stop predicate (``stop_requested``: a recorded hang-up / SIGTERM, a lost
foreground, the Ctrl-C flag) to ``So101Driver.step`` as ``stop_fn``, so a stop recorded while a
tick waits on its camera frame, the policy query or the servo read never produces one more goal.
The trial starts only once its first policy goal write was begun
(``So101Driver.goal_write_attempted``): a first goal blocked by the check is ``not_started`` with
no ``trial_started`` row, and a resume retries the slot with the same placement.

Real signals (``os.kill`` on this process) through the real ``operator_signals`` /
``job_control`` handlers; the terminal keeps reporting this process group as foreground
(``tcgetpgrp`` faked), so a lost foreground is the recorded SIGTTOU alone. Fake arm, camera and
clock; a real gRPC mock policy on 127.0.0.1. Every test has a hard SIGALRM timeout.

Run by name before the real-eval protocol is frozen (PR #32):
    uv run python -m pytest -q tests/test_deploy_prewrite_stop.py
"""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from robot_fakes import real_cfg
from test_deploy import Rig, _console, _pilot_kw, _record, pilot, rig
from test_real_eval import _args, ev

from src.robot import prompts as P
from src.robot.client import DeployLoop, ScriptedOperator
from src.robot.driver import So101Driver
from src.robot.safety import DeadmanResponse

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
    * ``home_camera``: the frame read of move-home tick ``n``.

    ``action`` "serial" instead makes the goal write of that tick fail (a bus fault mid-write)."""

    def __init__(
        self, loop: DeployLoop, sig: str | None, where: str, n: int, action: str = "signal"
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

        self._in_step = False
        loop.cam.read = read  # type: ignore[method-assign]
        fol.get_observation = get_observation  # type: ignore[method-assign]
        loop.client.act = act_wrapped  # type: ignore[method-assign]
        fol.send_action = send_action  # type: ignore[method-assign]
        arm.hold = hold_wrapped  # type: ignore[method-assign]
        arm.step = step_wrapped  # type: ignore[method-assign]
        loop.stop_requested = stop_requested_logged  # type: ignore[method-assign]
        loop._pre_write_stop = pre_write_logged  # type: ignore[method-assign]

    def _maybe_fire(self, where: str, count: int) -> None:
        if self.fired or where != self.where or count != self.n:
            return
        self.fired = True
        if self.action == "serial":
            self.loop.arm.follower.fail_write = True  # type: ignore[attr-defined]
            self.log.append(("serial_fault_armed",))
            return
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

    def after_signal(self) -> list[tuple[Any, ...]]:
        i = next(i for i, e in enumerate(self.log) if e[0] in ("signal", "serial_fault_armed"))
        return self.log[i + 1 :]

    def goals_after_signal(self) -> int:
        return sum(e == ("write", "goal") for e in self.after_signal())

    def goals_before_signal(self) -> int:
        i = next(i for i, e in enumerate(self.log) if e[0] in ("signal", "serial_fault_armed"))
        return sum(e == ("write", "goal") for e in self.log[:i])


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
        probe = Probe(r.loop, sig, where, n, action)
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
    assert not any(row["kind"] == "trial_started" for row in rows)  # nothing resume counts
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
    kinds = [row["kind"] for row in rows if row["kind"] in ("trial_started", "trial")]
    assert kinds == ["trial_started", "trial"]
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
    kinds = [row["kind"] for row in rows if row["kind"] in ("trial_started", "trial")]
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
