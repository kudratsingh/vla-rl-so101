"""WO-07 robot client (Mac): camera -> policy server -> safety -> bus, at ``robot.control_hz``.

    client = PolicyClient("127.0.0.1:50051", timeout_s=dcfg.request_timeout_s)
    check_server(client.info(), dcfg)                     # refuses on any mismatch
    loop = DeployLoop(arm, cam, pipe, client, dcfg)
    home = loop.go_home()                                 # MOVES THE ARM (through safety)
    result = loop.run_trial(protocol.instruction("PickPlace", RED_CUBE))

Every write goes through ``So101Driver.step`` (deadman frame check -> ``SafetyFilter`` -> bus);
the only other writes are the driver's own hold-at-measured-pose responses.

Stop semantics (CLAUDE.md rule 8). An action chunk (C = 8 at 15 Hz) runs open loop for ~0.53 s,
longer than ``deadman_frame_timeout_ms`` (300), so the deadman cannot check the age of the frame
the chunk was predicted from. Three separate bounds instead:

1. Camera liveness (the deadman). A fresh frame is read on EVERY control tick, including the
   tick that writes a chunk's first action and the decision tick before a query, and that
   frame's capture time is what the driver checks. A camera that faults, or stalls (returns the
   same stamp), trips the deadman (HOLD) within ``deadman_frame_timeout_ms`` plus one tick.
2. Chunk staleness. ``max_chunk_age_ms`` bounds the time from the chunk's SOURCE frame capture to
   each write. A later action that is too old is dropped with the rest of the chunk and the
   policy is re-queried (counted in ``stale_chunk_drops``); a chunk whose FIRST action is already
   too old trips the deadman (HOLD), since the loop can no longer act on current observations.
3. Server timeout. ``request_timeout_ms`` is the gRPC deadline, and also checked on the loop's
   own clock. A timeout, an unreachable server, a server error or a malformed reply trips the
   deadman (HOLD). Nothing is written after a trip except the driver's single hold write.

A serial error ends everything: the driver has already cut torque. A HOLD trip ends the trial;
resuming needs an operator acknowledgement (``Deadman.reset``). Ctrl-C holds (a deadman trip),
then ``safe_shutdown`` waits for the operator to support the arm before disabling torque.

Latency, all from the injected ``clock`` (``time.perf_counter`` on the arm; the camera stamps with
the same clock): capture -> preprocessed (state read + image pipeline) -> send -> recv -> first
write, plus the server's ``infer_ms`` / ``server_ms``. The WO-07 metric is
``capture_to_first_write_ms``. LeRobot stamps a frame after its driver read, so this is
host-stamped capture to bus write: sensor exposure and USB / driver buffering are NOT included.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Protocol, Self

import grpc
import numpy as np
import numpy.typing as npt

from src.config import Config
from src.eval import protocol
from src.robot.camera import CameraFault, Frame
from src.robot.driver import DeadmanTripped, SerialFault, So101Driver
from src.robot.safety import DeadmanResponse
from src.robot.serve import ACT_METHOD, INFO_METHOD, WIRE_VERSION, WireError, pack, unpack
from src.robot.units import NUM_JOINTS

log = logging.getLogger(__name__)

MAX_LOGGED_INTERVENTIONS = 20


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DeployConfig:
    control_hz: float
    chunk_len: int
    action_dim: int
    image_size: int
    exec_steps: int  # actions executed per chunk (C = the whole chunk, as the sim eval runs it)
    request_timeout_s: float
    max_chunk_age_s: float
    trial_max_steps: int
    home_qpos: tuple[float, ...]
    home_tol_rad: float
    home_timeout_s: float
    latency_target_ms: float
    stop_on_reject: bool

    def __post_init__(self) -> None:
        if not 1 <= self.exec_steps <= self.chunk_len:
            raise ValueError(f"exec_steps must be in [1, {self.chunk_len}], got {self.exec_steps}")
        if not 0 < self.request_timeout_s < self.max_chunk_age_s:
            raise ValueError("need 0 < request_timeout_ms < max_chunk_age_ms")
        if len(self.home_qpos) != NUM_JOINTS or not all(map(math.isfinite, self.home_qpos)):
            raise ValueError(f"home_qpos must be {NUM_JOINTS} finite radians")
        if self.trial_max_steps < 1 or self.control_hz <= 0:
            raise ValueError("trial_max_steps and control_hz must be positive")

    @classmethod
    def from_config(cls, cfg: Config) -> DeployConfig:
        d = cfg["deploy"]
        chunk_len = int(d["chunk_len"])
        exec_steps = d.get("exec_steps")
        return cls(
            control_hz=float(cfg["robot"]["control_hz"]),
            chunk_len=chunk_len,
            action_dim=int(d["action_dim"]),
            image_size=int(cfg["cameras"]["front"]["crop_to"]),
            exec_steps=chunk_len if exec_steps is None else int(exec_steps),
            request_timeout_s=float(d["request_timeout_ms"]) / 1e3,
            max_chunk_age_s=float(d["max_chunk_age_ms"]) / 1e3,
            trial_max_steps=int(d["trial_max_steps"]),
            home_qpos=tuple(float(v) for v in d["home_qpos"]),
            home_tol_rad=math.radians(float(d["home_tol_deg"])),
            home_timeout_s=float(d["home_timeout_s"]),
            latency_target_ms=float(d["latency_target_ms"]),
            stop_on_reject=bool(d.get("stop_on_reject", True)),
        )


# ------------------------------------------------------------------------------------------------
# Policy client
# ------------------------------------------------------------------------------------------------
class ServerFault(RuntimeError):
    """The policy server did not return a usable chunk. ``kind``: timeout | unavailable | error |
    bad_reply."""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind


class ServerMismatch(RuntimeError):
    """The server's Info does not match what this client was configured for."""


@dataclass(frozen=True)
class ActReply:
    actions: np.ndarray  # (C, A) float64 rad
    infer_ms: float
    server_ms: float
    t_send: float
    t_recv: float


_STATUS_KIND = {
    grpc.StatusCode.DEADLINE_EXCEEDED: "timeout",
    grpc.StatusCode.UNAVAILABLE: "unavailable",
}


class PolicyClient:
    def __init__(
        self,
        target: str,
        *,
        timeout_s: float,
        chunk_len: int = 8,
        action_dim: int = NUM_JOINTS,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.target = target
        self.timeout_s = float(timeout_s)
        self.chunk_len, self.action_dim = int(chunk_len), int(action_dim)
        self.clock = clock
        self._channel = grpc.insecure_channel(target)
        self._info = self._channel.unary_unary(INFO_METHOD)
        self._act = self._channel.unary_unary(ACT_METHOD)
        self._next_id = 0

    def info(self, timeout_s: float = 10.0) -> dict[str, Any]:
        try:
            header, _ = unpack(self._info(b"", timeout=timeout_s))
        except grpc.RpcError as e:
            raise self._fault(e) from e
        return header

    def act(self, image: npt.ArrayLike, state: npt.ArrayLike, instruction: str) -> ActReply:
        self._next_id += 1
        rid = self._next_id
        req = pack(
            {"request_id": rid, "instruction": instruction},
            {
                "image": np.asarray(image, dtype=np.uint8),
                "state": np.asarray(state, dtype=np.float32),
            },
        )
        t_send = self.clock()
        try:
            raw = self._act(req, timeout=self.timeout_s)
        except grpc.RpcError as e:
            raise self._fault(e) from e
        t_recv = self.clock()
        if t_recv - t_send > self.timeout_s:  # the deadline on the loop's own clock
            raise ServerFault(
                "timeout",
                f"reply after {(t_recv - t_send) * 1e3:.0f} ms > {self.timeout_s * 1e3:.0f} ms",
            )
        try:
            header, arrays = unpack(raw)
            actions = np.asarray(arrays["actions"], dtype=np.float64)
            infer_ms, server_ms = float(header["infer_ms"]), float(header["server_ms"])
        except (WireError, KeyError, TypeError, ValueError) as e:
            raise ServerFault("bad_reply", f"undecodable reply: {e}") from e
        if header.get("request_id") != rid:
            raise ServerFault(
                "bad_reply", f"reply for request {header.get('request_id')}, sent {rid}"
            )
        if actions.shape != (self.chunk_len, self.action_dim):
            raise ServerFault(
                "bad_reply",
                f"actions {actions.shape}, expected ({self.chunk_len}, {self.action_dim})",
            )
        return ActReply(actions, infer_ms, server_ms, t_send, t_recv)

    @staticmethod
    def _fault(e: grpc.RpcError) -> ServerFault:
        code = e.code() if hasattr(e, "code") else None
        details = e.details() if hasattr(e, "details") else str(e)
        return ServerFault(_STATUS_KIND.get(code, "error"), f"{code}: {details}")

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def server_mismatches(info: Mapping[str, Any], dcfg: DeployConfig) -> list[str]:
    expected = {
        "wire_version": WIRE_VERSION,
        "chunk_len": dcfg.chunk_len,
        "action_dim": dcfg.action_dim,
        "image_size": dcfg.image_size,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
    }
    out = [
        f"{k}: server {info.get(k)!r} != client {v!r}"
        for k, v in expected.items()
        if info.get(k) != v
    ]
    if not info.get("policy_id"):
        out.append("policy_id: server sent none")
    return out


def check_server(info: Mapping[str, Any], dcfg: DeployConfig) -> None:
    bad = server_mismatches(info, dcfg)
    if bad:
        raise ServerMismatch("policy server does not match this client:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------------------------------------------
# Control loop
# ------------------------------------------------------------------------------------------------
class FrameSource(Protocol):
    def read(self) -> Frame: ...


@dataclass
class TrialResult:
    stop_reason: str = ""  # max_steps | safety_reject | deadman | stale_chunk | serial_fault |
    # server_{timeout,unavailable,error,bad_reply} | operator_abort | client_error
    stop_detail: str = ""
    deadman_response: str | None = None
    steps: int = 0
    decisions: int = 0
    stale_chunk_drops: int = 0
    safety_status: Counter[str] = field(default_factory=Counter)
    interventions_by_kind: Counter[str] = field(default_factory=Counter)
    interventions_by_group: Counter[str] = field(default_factory=Counter)
    first_interventions: list[dict[str, Any]] = field(default_factory=list)
    timings: list[dict[str, float]] = field(default_factory=list)
    max_frame_age_ms: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0

    def latency_ms(self, key: str = "capture_to_first_write_ms") -> list[float]:
        return [t[key] for t in self.timings if key in t]


def _ms(a: float, b: float) -> float:
    return (b - a) * 1e3


class DeployLoop:
    def __init__(
        self,
        arm: So101Driver,
        cam: FrameSource,
        pipe: Callable[[np.ndarray], np.ndarray],
        client: PolicyClient,
        dcfg: DeployConfig,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        on_decision: Callable[[dict[str, float]], None] | None = None,
    ):
        self.arm, self.cam, self.pipe, self.client, self.dcfg = arm, cam, pipe, client, dcfg
        self.clock, self.sleep = clock, sleep
        self.on_decision = on_decision
        self.period = 1.0 / dcfg.control_hz

    def _read_frame(self) -> Frame | None:
        try:
            return self.cam.read()
        except CameraFault as e:
            log.critical("camera: %s", e)
            return None

    def _sleep_to(self, t0: float) -> None:
        self.sleep(max(0.0, self.period - (self.clock() - t0)))

    def go_home(self) -> dict[str, Any]:
        """Drive to ``home_qpos`` through the full safety filter at control_hz, reading a fresh
        frame every tick. MOVES THE ARM. Residuals under ``home_tol_deg`` count as reached and are
        logged (a P=16 servo under gravity may settle a degree or two off). May raise
        DeadmanTripped / SerialFault like any step."""
        target = np.asarray(self.dcfg.home_qpos, dtype=np.float64)
        t_end = self.clock() + self.dcfg.home_timeout_s
        steps, reached, rejected = 0, False, False
        while self.clock() < t_end:
            t0 = self.clock()
            frame = self._read_frame()
            out = self.arm.step(target, frame_time=None if frame is None else frame.capture_time)
            steps += 1
            if out.safety.status == "rejected":
                rejected = True
                break
            if np.all(np.abs(out.measured - target) <= self.dcfg.home_tol_rad):
                reached = True
                break
            self._sleep_to(t0)
        residual = np.degrees(np.abs(self.arm.read_state() - target))
        if not reached:
            log.warning(
                "home not reached (rejected=%s): residual deg %s", rejected, residual.round(1)
            )
        return {
            "reached": reached,
            "rejected": rejected,
            "steps": steps,
            "residual_deg": [round(float(v), 2) for v in residual],
        }

    def run_trial(self, instruction: str, max_steps: int | None = None) -> TrialResult:
        """One policy trial until ``trial_max_steps`` writes or a stop. Never raises for a stop:
        the reason is in the result, and the arm is holding (or limp after a serial fault)."""
        res = TrialResult(t_start=self.clock())
        n_max = self.dcfg.trial_max_steps if max_steps is None else int(max_steps)
        try:
            self._loop(res, instruction, n_max)
            if not res.stop_reason:
                res.stop_reason = "max_steps"
            self.arm.hold()  # stop where it is; the next command starts from the measured pose
        except DeadmanTripped:
            res.stop_reason = res.stop_reason or "deadman"
        except SerialFault as e:
            res.stop_reason, res.stop_detail = "serial_fault", str(e)
        except KeyboardInterrupt:
            res.stop_reason = "operator_abort"
            self.arm.trip("operator Ctrl-C")
        except Exception as e:  # any client bug must still leave the arm holding
            log.exception("deploy loop failed")
            res.stop_reason = "client_error"
            self.arm.trip(f"client error: {type(e).__name__}: {e}")
        if self.arm.deadman.tripped:
            res.stop_detail = res.stop_detail or self.arm.deadman.reason
            response = self.arm.deadman.response
            res.deadman_response = None if response is None else response.value
        res.t_end = self.clock()
        return res

    def _loop(self, res: TrialResult, instruction: str, n_max: int) -> None:
        dcfg = self.dcfg
        while res.steps < n_max:
            # -- decision tick: fresh frame, liveness check, query --------------------------------
            frame = self._read_frame()
            self.arm.check_frame(None if frame is None else frame.capture_time)
            assert frame is not None  # check_frame raised otherwise
            t_capture = frame.capture_time
            state = self.arm.read_state()
            image = self.pipe(frame.image)
            t_pre = self.clock()
            try:
                reply = self.client.act(image, state.astype(np.float32), instruction)
            except ServerFault as e:
                res.stop_reason = f"server_{e.kind}"
                raise self.arm.trip(f"policy server {e.kind}: {e}") from e
            res.decisions += 1
            timing = {
                "decision": res.decisions,
                "capture_to_preprocessed_ms": _ms(t_capture, t_pre),
                "preprocessed_to_send_ms": _ms(t_pre, reply.t_send),
                "rtt_ms": _ms(reply.t_send, reply.t_recv),
                "infer_ms": reply.infer_ms,
                "server_ms": reply.server_ms,
                "network_ms": _ms(reply.t_send, reply.t_recv) - reply.server_ms,
            }
            # -- execute the chunk open loop, one fresh frame per tick ----------------------------
            for i, action in enumerate(reply.actions[: dcfg.exec_steps]):
                if res.steps >= n_max:
                    break
                t0 = self.clock()  # each write tick is paced from its own start, so action 2
                frame = self._read_frame()  # lands one period after action 1, not after the query
                now = self.clock()
                if now - t_capture > dcfg.max_chunk_age_s:
                    if i == 0:
                        res.stop_reason = "stale_chunk"
                        raise self.arm.trip(
                            f"chunk stale before its first action: {_ms(t_capture, now):.0f} ms > "
                            f"{dcfg.max_chunk_age_s * 1e3:.0f} ms"
                        )
                    res.stale_chunk_drops += 1
                    break
                frame_time = None if frame is None else frame.capture_time
                if frame_time is not None:
                    res.max_frame_age_ms = max(res.max_frame_age_ms, _ms(frame_time, now))
                out = self.arm.step(action, frame_time=frame_time)
                t_write = self.clock()
                res.steps += 1
                self._record_safety(res, out.safety)
                if i == 0:
                    timing["recv_to_first_write_ms"] = _ms(reply.t_recv, t_write)
                    timing["capture_to_first_write_ms"] = _ms(t_capture, t_write)
                    res.timings.append(timing)
                    if self.on_decision is not None:
                        self.on_decision(timing)
                if out.safety.status == "rejected" and dcfg.stop_on_reject:
                    res.stop_reason = "safety_reject"
                    res.stop_detail = "; ".join(iv.reason for iv in out.safety.interventions)
                    return
                self._sleep_to(t0)

    @staticmethod
    def _record_safety(res: TrialResult, safety: Any) -> None:
        res.safety_status[safety.status] += 1
        for iv in safety.interventions:
            res.interventions_by_kind[iv.kind] += 1
            group = "pose" if iv.joint is None else ("gripper" if iv.joint == "gripper" else "arm")
            res.interventions_by_group[group] += 1
            if len(res.first_interventions) < MAX_LOGGED_INTERVENTIONS:
                res.first_interventions.append(
                    {
                        "step": res.steps,
                        "kind": iv.kind,
                        "joint": iv.joint,
                        "requested_deg": iv.requested_deg,
                        "applied_deg": iv.applied_deg,
                        "reason": iv.reason,
                    }
                )


# ------------------------------------------------------------------------------------------------
# Trials, records, summary
# ------------------------------------------------------------------------------------------------
def stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "max": round(float(a.max()), 3),
    }


LATENCY_NOTE = (
    "host-stamped capture (LeRobot stamps after its driver read) to bus write; excludes sensor "
    "exposure and USB / driver buffering"
)


def trial_object(task: str, cell: str, trial_idx: int) -> protocol.ObjectSpec:
    """Manipulated object: red cube on seen / shifted; unseen_objects alternates the held-out
    objects by trial index (logged per trial)."""
    objs = protocol.manipulated_objects(cell)
    return objs[trial_idx % len(objs)]


def trial_record(
    result: TrialResult,
    *,
    task: str,
    cell: str,
    trial_idx: int,
    instruction: str,
    info: Mapping[str, Any],
    start_utc: str,
    end_utc: str,
    outcome: str,
    outcome_source: str,
    home: Mapping[str, Any] | None,
    camera_calibrated: bool,
) -> dict[str, Any]:
    return {
        "task": task,
        "cell": cell,
        "trial_idx": trial_idx,
        "instruction": instruction,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
        "policy_id": info.get("policy_id"),
        "server_describe": info.get("describe"),
        "start_utc": start_utc,
        "end_utc": end_utc,
        "outcome": outcome,  # success | failure | aborted (not scored)
        "outcome_source": outcome_source,  # operator | scripted
        "steps": result.steps,
        "decisions": result.decisions,
        "stop_reason": result.stop_reason,
        "stop_detail": result.stop_detail,
        "deadman_response": result.deadman_response,
        "stale_chunk_drops": result.stale_chunk_drops,
        "home": dict(home) if home is not None else None,
        "safety": {
            "status_counts": dict(result.safety_status),
            "interventions_by_kind": dict(result.interventions_by_kind),
            "interventions_by_group": dict(result.interventions_by_group),
            "first_interventions": result.first_interventions,
        },
        "latency": {
            "capture_to_first_write_ms": stats(result.latency_ms()),
            "rtt_ms": stats(result.latency_ms("rtt_ms")),
            "infer_ms": stats(result.latency_ms("infer_ms")),
            "max_frame_age_ms": round(result.max_frame_age_ms, 3),
            "per_decision_capture_to_first_write_ms": [round(v, 3) for v in result.latency_ms()],
            "note": LATENCY_NOTE,
        },
        "camera_calibrated": camera_calibrated,
    }


def summarize(trials: Sequence[Mapping[str, Any]], latency_target_ms: float) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for t in trials:
        c = cells.setdefault(
            f"{t['task']}/{t['cell']}",
            {"task": t["task"], "cell": t["cell"], "n": 0, "successes": 0, "aborted": 0},
        )
        c.setdefault("stop_reasons", Counter())[t["stop_reason"]] += 1
        if t["outcome"] == "aborted":
            c["aborted"] += 1
            continue
        c["n"] += 1
        c["successes"] += int(t["outcome"] == "success")
    for c in cells.values():
        lo, hi = protocol.wilson_interval(c["successes"], c["n"])
        c["success_rate"] = c["successes"] / c["n"] if c["n"] else None
        c["wilson_95"] = [round(lo, 4), round(hi, 4)]
        c["stop_reasons"] = dict(c["stop_reasons"])
    lat = [v for t in trials for v in t["latency"]["per_decision_capture_to_first_write_ms"]]
    lat_stats = stats(lat)
    p95 = lat_stats["p95"]
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
        "cells": cells,
        "capture_to_first_write_ms": lat_stats,
        "latency_target_ms": latency_target_ms,
        "latency_p95_under_target": None if p95 is None else bool(p95 < latency_target_ms),
        "latency_note": LATENCY_NOTE,
    }


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Operator(Protocol):
    source: str  # "operator" | "scripted": recorded as each trial's outcome_source

    def confirm(self, msg: str) -> bool: ...

    def outcome(self, result: TrialResult) -> str: ...

    def ack_stop(self, reason: str) -> str | None: ...

    def wait(self, msg: str) -> None: ...


class ConsoleOperator:
    """The person at the arm. ``confirm``: ENTER = yes, q = quit."""

    source = "operator"

    def __init__(self, input_fn: Callable[[str], str] = input):
        self.input = input_fn

    def confirm(self, msg: str) -> bool:
        return self.input(f"{msg} [ENTER / q] ").strip().lower() not in ("q", "quit")

    def outcome(self, result: TrialResult) -> str:
        while True:
            ans = self.input(f"trial ended ({result.stop_reason}). success? [y/n] ").strip().lower()
            if ans in ("y", "n"):
                return "success" if ans == "y" else "failure"

    def ack_stop(self, reason: str) -> str | None:
        ans = self.input(
            f"STOPPED: {reason}\narm is holding; type 'resume' to continue, else quit: "
        )
        return "operator typed resume" if ans.strip() == "resume" else None

    def wait(self, msg: str) -> None:
        self.input(msg)


class ScriptedOperator:
    """Mock runs only: says yes to every prompt, scores success iff the trial ran to
    ``trial_max_steps`` without a stop, and acknowledges HOLD stops so the session continues."""

    source = "scripted"

    def confirm(self, msg: str) -> bool:
        return True

    def outcome(self, result: TrialResult) -> str:
        return "success" if result.stop_reason == "max_steps" else "failure"

    def ack_stop(self, reason: str) -> str | None:
        return "scripted acknowledgement (mock run)"

    def wait(self, msg: str) -> None:
        return None


def run_trials(
    loop: DeployLoop,
    operator: Operator,
    *,
    task: str,
    cell: str,
    n_trials: int,
    info: Mapping[str, Any],
    camera_calibrated: bool,
    log_row: Callable[..., None] | None = None,
    max_steps: int | None = None,
    trials: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Home -> trial -> outcome, ``n_trials`` times. Returns (trial records, why the session
    ended). A serial fault / torque-off, an operator abort or a declined resume ends the session.
    Records are appended to ``trials`` as they finish, so a Ctrl-C at a prompt keeps them."""
    if task not in protocol.TASKS or cell not in protocol.CELLS:
        raise ValueError(f"unknown task/cell {task}/{cell}")
    arm = loop.arm
    trials = [] if trials is None else trials
    for k in range(n_trials):
        obj = trial_object(task, cell, k)
        instruction = protocol.instruction(task, obj)
        tag = f"[{task}/{cell} {k + 1}/{n_trials}]"
        if not operator.confirm(f"{tag} clear the workspace; ENTER moves the arm HOME"):
            return trials, "operator_quit"
        start = utc_now()
        try:
            home: dict[str, Any] = loop.go_home()
        except (DeadmanTripped, SerialFault) as e:
            home = {"reached": False, "error": str(e)}
        if home["reached"]:
            if not operator.confirm(f"{tag} place the {obj.label} (grid slot {k}); ENTER starts"):
                return trials, "operator_quit"
            if log_row is not None:
                loop.on_decision = lambda t, k=k: log_row(kind="decision", trial_idx=k, **t)
            result = loop.run_trial(instruction, max_steps)
        else:
            result = TrialResult(stop_reason="home_failed", stop_detail=str(home))
            if not arm.deadman.tripped:
                arm.hold()
        aborted = result.stop_reason in ("operator_abort", "home_failed")
        outcome = "aborted" if aborted else operator.outcome(result)
        rec = trial_record(
            result,
            task=task,
            cell=cell,
            trial_idx=k,
            instruction=instruction,
            info=info,
            start_utc=start,
            end_utc=utc_now(),
            outcome=outcome,
            outcome_source=operator.source,
            home=home,
            camera_calibrated=camera_calibrated,
        )
        trials.append(rec)
        if log_row is not None:
            log_row(kind="trial", **rec)
        log.info("%s %s: %s (%d steps)", tag, outcome, result.stop_reason, result.steps)
        if result.stop_reason == "operator_abort":
            return trials, "operator_abort"
        if arm.deadman.tripped:
            if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
                return trials, f"torque_off: {arm.deadman.reason}"
            ack = operator.ack_stop(arm.deadman.reason)
            if ack is None:
                return trials, f"stopped: {arm.deadman.reason}"
            arm.deadman.reset(ack)
            arm.safety.reset_reference()
    return trials, "completed"


def safe_shutdown(arm: So101Driver, operator: Operator) -> None:
    """End of a real-arm session (also after Ctrl-C): hold the measured pose so the arm stays up,
    let the operator support it, then disable torque (the arm goes limp). A second Ctrl-C at the
    prompt goes straight to torque off. Leaving the driver's ``with`` block disables torque again.
    """
    try:
        arm.hold()
    except SerialFault:
        pass  # the driver already cut torque
    try:
        operator.wait("arm HOLDING; support it, then press ENTER to disable torque: ")
    except (KeyboardInterrupt, EOFError):
        pass
    arm.torque_off()
