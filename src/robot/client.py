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
   tick that writes a chunk's first action, the decision tick before a query and a tick whose
   action is dropped as stale, and the driver checks that frame's capture time before anything
   else happens on the tick. A camera that faults (a lost frame) trips the deadman (HOLD) on that
   tick. One that stalls (returns the same stamp) trips at the first check after the frame is
   older than ``deadman_frame_timeout_ms``: within one tick while a chunk executes, but only
   after the reply while a query is in flight, so the worst case is the timeout + preprocessing
   + RTT (<= ``request_timeout_ms``), ~700 ms with real.yaml. No write ever uses a frame older
   than ``deadman_frame_timeout_ms``.
2. Chunk staleness. ``max_chunk_age_ms`` bounds the time from the chunk's SOURCE frame capture to
   each write. A later action that is too old is dropped with the rest of the chunk and the
   policy is re-queried (counted in ``stale_chunk_drops``); a chunk whose FIRST action is already
   too old trips the deadman (HOLD), since the loop can no longer act on current observations.
3. Server timeout. ``request_timeout_ms`` is the gRPC deadline, and also checked on the loop's
   own clock. A timeout, an unreachable server, a server error or a malformed reply (wrong shape,
   non-finite actions) trips the deadman (HOLD). Nothing is written after a trip except the
   driver's single hold write.

Pacing. Writes are at least one period apart. The policy is queried synchronously between chunks,
right after the previous chunk's last write, and that query counts against the next write's
period: the gap at a chunk boundary is max(period, decision tick + query), not their sum. A query
slower than the period therefore stretches the boundary gap (``boundary_gap_ms`` per decision,
``write_hz`` per trial); overlapping the query with the last action is left for WO-08.

A serial error ends everything: the driver has already cut torque, and nothing is written after
it. A servo fault (over-temperature / over-current, ``src/robot/servo_health.py``) is the same:
torque is already off, and the operator is told to support the arm. A HOLD trip ends the trial;
resuming needs an operator acknowledgement (``Deadman.reset``). Ctrl-C during motion (a trial or
the move home) holds (a deadman trip). On the real arm the session runs under
``prompts.operator_signals``: Ctrl-C outside a prompt only sets a flag that ``DeployLoop`` checks
between ticks (``SessionStop``), so it never lands inside a bus transaction; a hang-up (SIGHUP /
SIGTERM: the Terminal closed) stops motion the same way (stop reason ``hangup``) and every later
prompt takes the closed-stdin path. Without the handler (mock) a Ctrl-C mid-tick still holds; the
port is released first in case it landed inside a bus transaction.

Prompts with torque on (``ConsoleOperator`` through ``src/robot/prompts.py``'s ``TorqueGuard``,
owner 2026-11-08): every prompt answered while the arm holds under torque (between trials, the
placement, the score, the 'resume' after a HOLD stop, the final support prompt) reads servo health
before the wait and after the answer, and waits at most ``deploy.prompt_torque_timeout_s``
(default 60 s; the real arm refuses null or <= 0). With no answer: "SUPPORT THE ARM NOW",
``deploy.prompt_support_grace_s`` (default 10 s, ENTER = now), deadman TORQUE_OFF, torque off, and
the session ends (``prompt_timeout``; no park). Mock runs use ``ScriptedOperator`` (instant).

Scoring (``run_trials`` / ``_score``, judging rule real-judge-2 in scripts/wo07_real_eval.py):
a trial counts once its policy started (``TrialResult.policy_started``: the first action from the
policy server was handed to ``So101Driver.step``, i.e. the safety filter). A fault after that
(servo, serial, a safety-layer or deadman stop, server failure, Ctrl-C during motion, client
error, torque off unconfirmed) is a failure with ``fault`` recorded and is never judged by the
operator; an attempt whose policy never started is ``not_started`` (retried on a resume, never
scored). ``summarize``: primary metric = successes / started trials.

Shutdown (``safe_shutdown`` -> ``Shutdown``). Healthy (``healthy=True``: the session completed, or
the operator quit / pressed Ctrl-C at a prompt with the arm idle, and the deadman has not tripped)
AND parking enabled (``robot.park.enabled``, default false until the owner verifies the rest pose
and park path): park at ``robot.park.qpos`` through the safety filter, then torque off. Otherwise
(parking off, any fault, an abort during motion, a declined resume): hold, the operator supports
the arm and presses ENTER (``supported``, ``support_confirmed: true``), then torque off. Only an
explicit ENTER confirms: Ctrl-C there is ignored (the arm keeps holding, the bounded wait keeps
running); a closed stdin, a hang-up or the timeout releases torque after the grace period and
the end is ``released_unconfirmed`` with the cause (``support_confirmed: false``). After a serial or servo
fault torque is already off, nothing is written, and the acknowledgement is recorded with how it
was given (``torque_was_off``).

Torque off unconfirmed (``So101Driver.torque_off_unconfirmed``, latched by the driver when a
torque-off write fails): no further motion is possible in the process. The shutdown result is
``torque_off_unconfirmed`` and the operator is told to switch off the inline power switch; every
trial record from then on carries ``torque_off_unconfirmed: true`` and that stop reason, applied
last so nothing after it can overwrite it (a started trial is a failure, fault kind
``torque_off_unconfirmed``); ``summarize`` reports it for the session.

Latency, all from the injected ``clock`` (``time.perf_counter`` on the arm; the camera stamps with
the same clock): frame stamp -> preprocessed (state read + image pipeline) -> send -> recv ->
first write, plus the server's ``infer_ms`` / ``server_ms``. The logged metric is
``frame_return_to_first_write_ms``; when the query beats the period it includes ``pace_wait_ms``,
the wait for the write tick. It is NOT the end-to-end (capture -> command) latency of the WO-07
benchmark: LeRobot 0.4.4 stamps a frame with ``time.perf_counter()`` in its read thread after
``VideoCapture.read()`` returned and after the BGR->RGB conversion
(``lerobot/cameras/opencv/camera_opencv.py:443-451``), and OpenCV 4.11's AVFoundation camera
backend keeps no capture timestamp (``CvCaptureCAM::getProperty`` answers only width, height,
fps and format; ``CAP_PROP_POS_MSEC`` returns 0 for a live camera; the delegate never reads the
sample buffer's presentation time). Exposure, sensor readout, USB transfer, AVFoundation
buffering and decode are therefore excluded. ``docs/bringup/latency-measurement.md`` measures that
offset once (optical event -> frame stamp); with ``deploy.capture_offset_ms`` set, ``summarize``
adds an ``end_to_end_estimate_ms`` labelled with it, and never otherwise.
"""

from __future__ import annotations

import functools
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
from src.robot.driver import (
    SERVO_FAULT_PROMPT,
    TORQUE_OFF_UNCONFIRMED,
    TORQUE_OFF_UNCONFIRMED_MSG,
    DeadmanTripped,
    ParkSettings,
    SerialFault,
    ServoFault,
    So101Driver,
    park_from_config,
)
from src.robot.prompts import (
    CTRL_C,
    ENTER,
    EOF,
    HANGUP,
    TIMEOUT,
    Hangup,
    Interrupts,
    TorqueGuard,
    ack_record,
    ignoring_ctrl_c,
)
from src.robot.safety import DeadmanResponse
from src.robot.serve import ACT_METHOD, INFO_METHOD, WIRE_VERSION, WireError, pack, unpack
from src.robot.units import NUM_JOINTS

log = logging.getLogger(__name__)

MAX_LOGGED_INTERVENTIONS = 20
# LeRobot frame stamp (after the driver returned the frame) -> first bus write. Excludes capture:
# never call it end-to-end (module doc, docs/bringup/latency-measurement.md).
LATENCY_KEY = "frame_return_to_first_write_ms"


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
    policy_checkpoint: str | None = None  # server.policy_checkpoint: Info's policy_id must match
    capture_offset_ms: float | None = None  # optical event -> frame stamp, measured; None = not
    park: ParkSettings | None = None  # healthy-shutdown park; None = robot.park.enabled false

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
        checkpoint = (cfg.get("server") or {}).get("policy_checkpoint")
        offset = d.get("capture_offset_ms")
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
            policy_checkpoint=None if checkpoint is None else str(checkpoint),
            capture_offset_ms=None if offset is None else float(offset),
            park=park_from_config(cfg),
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
        if not np.all(np.isfinite(actions)):  # a server fault, not a policy safety rejection
            raise ServerFault("bad_reply", "actions contain non-finite values")
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
    policy_id = info.get("policy_id")
    if not policy_id or str(policy_id).strip().lower() in ("none", "null"):
        out.append(f"policy_id: server sent none ({policy_id!r})")
    elif dcfg.policy_checkpoint is not None and policy_id != dcfg.policy_checkpoint:
        out.append(
            f"policy_id: server {policy_id!r} != server.policy_checkpoint "
            f"{dcfg.policy_checkpoint!r}"
        )
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
    # servo_fault | home_failed | torque_off_unconfirmed (set by run_trials, from the driver) |
    # server_{timeout,unavailable,error,bad_reply} | operator_abort | hangup | client_error |
    # session_lost | operator_quit / prompt_timeout / stdin_closed (placement-prompt attempts)
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
    t_first_write: float | None = None
    t_last_write: float | None = None
    # policy started: the trial's first action, received from the policy server, was handed to
    # So101Driver.step (deadman frame check -> SafetyFilter -> bus). Loop clock and UTC.
    policy_started_t: float | None = None
    policy_started_utc: str | None = None

    @property
    def policy_started(self) -> bool:
        return self.policy_started_t is not None

    def latency_ms(self, key: str = LATENCY_KEY) -> list[float]:
        return [t[key] for t in self.timings if key in t]

    @property
    def write_hz(self) -> float | None:
        """Realised policy-write rate over the trial (sim: exactly control_hz)."""
        if self.t_first_write is None or self.t_last_write is None or self.steps < 2:
            return None
        span = self.t_last_write - self.t_first_write
        return (self.steps - 1) / span if span > 0 else None


def _ms(a: float, b: float) -> float:
    return (b - a) * 1e3


def _say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except OSError:
        pass  # the terminal is gone (hang-up)


class SessionStop(Exception):
    """A Ctrl-C (flag) or a hang-up seen between ticks (``DeployLoop.stop_requested``)."""

    def __init__(self, kind: str, reason: str):
        super().__init__(reason)
        self.kind, self.reason = kind, reason


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
        interrupts: Interrupts | None = None,
    ):
        self.arm, self.cam, self.pipe, self.client, self.dcfg = arm, cam, pipe, client, dcfg
        self.clock, self.sleep = clock, sleep
        self.on_decision = on_decision
        # runs once per trial, right after its first action went to the safety filter (the
        # durable "policy started" marker; run_trials)
        self.on_policy_start: Callable[[TrialResult], None] | None = None
        # the session's signal flags (src/robot/prompts.operator_signals; hardware only): a
        # Ctrl-C or a hang-up stops motion between ticks, never inside a bus transaction
        self.interrupts = interrupts
        self.period = 1.0 / dcfg.control_hz

    def stop_requested(self) -> SessionStop | None:
        """A Ctrl-C (flag) or a hang-up since the last prompt answer, as the stop to raise."""
        i = self.interrupts
        if i is None:
            return None
        if i.hangup is not None:
            return SessionStop("hangup", f"{i.hangup}: the operator's terminal closed")
        if i.ctrl_c:
            return SessionStop("operator_abort", "operator Ctrl-C")
        return None

    def clear_ctrl_c(self) -> None:
        """Drop a Ctrl-C pressed while nothing moved (a prompt was answered since)."""
        if self.interrupts is not None:
            self.interrupts.ctrl_c = False

    def _check_stop(self) -> None:
        stop = self.stop_requested()
        if stop is not None:
            raise stop

    def _read_frame(self) -> Frame | None:
        try:
            return self.cam.read()
        except CameraFault as e:
            log.critical("camera: %s", e)
            return None

    def _sleep_to(self, t0: float) -> None:
        self.sleep(max(0.0, self.period - (self.clock() - t0)))

    def frame_time(self) -> float | None:
        """A fresh frame's stamp, or None on a lost frame (the deadman then trips)."""
        frame = self._read_frame()
        return None if frame is None else frame.capture_time

    def go_home(self) -> dict[str, Any]:
        """Drive to ``home_qpos`` through the full safety filter at control_hz, reading a fresh
        frame every tick. MOVES THE ARM. Residuals under ``home_tol_deg`` count as reached and are
        logged (a P=16 servo under gravity may settle a degree or two off). May raise
        DeadmanTripped / SerialFault like any step."""
        target = np.asarray(self.dcfg.home_qpos, dtype=np.float64)
        t_end = self.clock() + self.dcfg.home_timeout_s
        steps, reached, rejected = 0, False, False
        while self.clock() < t_end:
            self._check_stop()  # SessionStop: the caller holds the arm (between ticks)
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
        except ServoFault as e:  # torque already off: no hold, no park
            res.stop_reason, res.stop_detail = "servo_fault", str(e)
        except DeadmanTripped:
            res.stop_reason = res.stop_reason or "deadman"
        except SerialFault as e:
            res.stop_reason, res.stop_detail = "serial_fault", str(e)
        except SessionStop as e:  # a Ctrl-C / hang-up flag, seen between ticks: hold
            res.stop_reason, res.stop_detail = e.kind, e.reason
            ignoring_ctrl_c(functools.partial(self.arm.trip, e.reason), self.arm, _say)
        except KeyboardInterrupt:  # no operator_signals handler (mock): the press landed mid-tick
            res.stop_reason = "operator_abort"
            ignoring_ctrl_c(functools.partial(self.arm.trip, "operator Ctrl-C"), self.arm, _say)
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
        t_tick: float | None = None  # start of the last write tick; the next starts a period later
        while res.steps < n_max:
            self._check_stop()
            # -- decision tick: right after the last write, so the query counts against the next
            # write's period. Fresh frame, liveness check, query --------------------------------
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
                "frame_return_to_preprocessed_ms": _ms(t_capture, t_pre),
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
                wait = 0.0 if t_tick is None else max(0.0, t_tick + self.period - self.clock())
                self.sleep(wait)  # writes at least one period apart, the first of a chunk too
                self._check_stop()
                t0 = self.clock()
                frame = self._read_frame()
                # liveness first, on every tick: a lost frame on a stale-drop tick still trips
                self.arm.check_frame(None if frame is None else frame.capture_time)
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
                first = res.policy_started_t is None
                if first:  # the first policy action goes to the filter
                    res.policy_started_t, res.policy_started_utc = self.clock(), utc_now()
                try:
                    out = self.arm.step(action, frame_time=frame_time)
                    t_write = self.clock()
                finally:
                    if first:  # after the write and its stamp: never delays or skews either
                        self._policy_started(res)
                t_tick = t0
                res.steps += 1
                self._record_safety(res, out.safety)
                if i == 0:
                    timing["pace_wait_ms"] = wait * 1e3
                    timing["recv_to_first_write_ms"] = _ms(reply.t_recv, t_write)
                    timing[LATENCY_KEY] = _ms(t_capture, t_write)
                    if res.t_last_write is not None:  # last write of the previous chunk -> this
                        timing["boundary_gap_ms"] = _ms(res.t_last_write, t_write)
                    res.timings.append(timing)
                    if self.on_decision is not None:
                        self.on_decision(timing)
                if res.t_first_write is None:
                    res.t_first_write = t_write
                res.t_last_write = t_write
                if out.safety.status == "rejected" and dcfg.stop_on_reject:
                    res.stop_reason = "safety_reject"
                    res.stop_detail = "; ".join(iv.reason for iv in out.safety.interventions)
                    return

    def _policy_started(self, res: TrialResult) -> None:
        if self.on_policy_start is None:
            return
        try:
            self.on_policy_start(res)
        except Exception:  # a logging failure must not stop the trial
            log.exception("recording the policy start failed")

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


SCORED_OUTCOMES = ("success", "failure")
PRIMARY_METRIC = (
    "success_rate = successes / trials whose policy started (each slot scored once); a fault "
    "after the start (servo, serial, safety / deadman stop, server, torque off unconfirmed, "
    "Ctrl-C during motion) and an unjudged score prompt count as failures"
)
DIAGNOSTIC_NOTE = (
    "SECONDARY diagnostic, not the result: success rate over started trials that did not end in "
    "a fault. The primary metric (success_rate) counts every faulted trial as a failure."
)

LATENCY_NOTE = (
    "frame_return_to_first_write_ms: LeRobot's frame stamp (perf_counter in its read thread, "
    "after VideoCapture.read() returned and the BGR->RGB conversion) to the first bus write. "
    "EXCLUDES capture: exposure, sensor readout, USB transfer, AVFoundation buffering, decode. "
    "Not end-to-end; see docs/bringup/latency-measurement.md"
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
    servo_health: Mapping[str, Any] | None = None,
    fault: Mapping[str, Any] | None = None,
    unjudged: Mapping[str, Any] | None = None,
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
        "outcome": outcome,  # success | failure | not_started (an attempt, never scored)
        "outcome_source": outcome_source,  # operator | scripted | fault | unjudged | not_scored
        "policy_started": result.policy_started,
        "policy_started_utc": result.policy_started_utc,
        "fault": None if fault is None else dict(fault),  # {kind, stop_reason, reason, where}
        "unjudged": None if unjudged is None else dict(unjudged),  # {how, reason}
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
            LATENCY_KEY: stats(result.latency_ms()),
            "rtt_ms": stats(result.latency_ms("rtt_ms")),
            "infer_ms": stats(result.latency_ms("infer_ms")),
            "boundary_gap_ms": stats(result.latency_ms("boundary_gap_ms")),
            "write_hz": None if result.write_hz is None else round(result.write_hz, 3),
            "max_frame_age_ms": round(result.max_frame_age_ms, 3),
            f"per_decision_{LATENCY_KEY}": [round(v, 3) for v in result.latency_ms()],
            "note": LATENCY_NOTE,
        },
        "servo_health": None if servo_health is None else dict(servo_health),  # session so far
        "camera_calibrated": camera_calibrated,
    }


def summarize(
    trials: Sequence[Mapping[str, Any]],
    latency_target_ms: float,
    capture_offset_ms: float | None = None,
) -> dict[str, Any]:
    """Per cell. PRIMARY end-to-end metric: ``successes / n`` with ``n`` = trials whose policy
    started (``policy_started``), each scored once; a started trial that ended in a fault is a
    failure. Attempts whose policy never started are counted (``not_started``) but never scored.
    ``diagnostic_excluding_faulted`` is a SECONDARY diagnostic (the success rate over started
    trials without a fault), and ``faults_by_kind`` counts the faults."""
    cells: dict[str, dict[str, Any]] = {}
    for t in trials:
        c = cells.setdefault(
            f"{t['task']}/{t['cell']}",
            {
                "task": t["task"],
                "cell": t["cell"],
                "n": 0,
                "successes": 0,
                "not_started": 0,
                "faulted": 0,
                "unjudged": 0,
            },
        )
        c.setdefault("stop_reasons", Counter())[t["stop_reason"]] += 1
        c.setdefault("faults_by_kind", Counter())
        if t["outcome"] not in SCORED_OUTCOMES:
            c["not_started"] += 1
            continue
        c["n"] += 1
        c["successes"] += int(t["outcome"] == "success")
        if t.get("fault"):
            c["faulted"] += 1
            c["faults_by_kind"][t["fault"]["kind"]] += 1
        c["unjudged"] += int(t.get("outcome_source") == "unjudged")
    for c in cells.values():
        lo, hi = protocol.wilson_interval(c["successes"], c["n"])
        c["success_rate"] = c["successes"] / c["n"] if c["n"] else None
        c["wilson_95"] = [round(lo, 4), round(hi, 4)]
        c["stop_reasons"] = dict(c["stop_reasons"])
        c["faults_by_kind"] = dict(c["faults_by_kind"])
        clean = c["n"] - c["faulted"]
        c["diagnostic_excluding_faulted"] = {
            "n": clean,
            "successes": c["successes"],
            "success_rate": c["successes"] / clean if clean else None,
            "note": DIAGNOSTIC_NOTE,
        }
    lat = [v for t in trials for v in t["latency"][f"per_decision_{LATENCY_KEY}"]]
    lat_stats = stats(lat)
    p95 = lat_stats["p95"]
    out: dict[str, Any] = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_hash": protocol.protocol_hash(),
        "cells": cells,
        LATENCY_KEY: lat_stats,
        "write_hz": stats([h for t in trials if (h := t["latency"].get("write_hz")) is not None]),
        "primary_metric": PRIMARY_METRIC,
        "latency_target_ms": latency_target_ms,
        # p95 of a metric that excludes capture: necessary, not sufficient, for the WO-07 target
        "frame_return_p95_under_target": None if p95 is None else bool(p95 < latency_target_ms),
        "latency_note": LATENCY_NOTE,
        "end_to_end_estimate_ms": None,
        # any trial whose torque-off write failed: the session's numbers stand, the arm did not
        # end in a confirmed safe state (support it, cut the supply at the inline switch)
        "torque_off_unconfirmed": any(bool(t.get(TORQUE_OFF_UNCONFIRMED)) for t in trials),
    }
    if capture_offset_ms is not None:
        e2e = stats([v + capture_offset_ms for v in lat])
        out["end_to_end_estimate_ms"] = {
            **e2e,
            "capture_offset_ms": capture_offset_ms,
            "method": "frame_return_to_first_write_ms + deploy.capture_offset_ms (measured once, "
            "docs/bringup/latency-measurement.md)",
            "p95_under_target": (
                None if e2e["p95"] is None else bool(e2e["p95"] < latency_target_ms)
            ),
        }
    return out


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class PromptTimeout(RuntimeError):
    """A prompt answered with torque on got no answer within ``prompt_torque_timeout_s``: torque
    has been released (support message, grace, deadman TORQUE_OFF, torque off;
    ``TorqueGuard.release``). The session ends; nothing after it holds, parks or moves."""


class Operator(Protocol):
    """The person at the arm (``ConsoleOperator``) or a mock script (``ScriptedOperator``).

    ``confirm`` / ``outcome`` / ``ack_stop`` are answered with torque on: they raise
    KeyboardInterrupt (Ctrl-C), EOFError (closed stdin) or ``PromptTimeout``, and ServoFault /
    SerialFault from the servo-health reads around the wait. ``acknowledge`` (torque already off)
    and ``support`` (the final support prompt, torque on) never raise for Ctrl-C / EOF / a
    timeout: they return how the prompt ended (``enter`` | ``ctrl_c`` | ``eof`` | ``timeout``),
    and only ``enter`` confirms. ``released`` is set once a timeout or a closed stdin released
    torque."""

    source: str  # "operator" | "scripted": recorded as each judged trial's outcome_source

    def confirm(self, msg: str) -> bool: ...

    def outcome(self, result: TrialResult) -> str: ...

    def ack_stop(self, reason: str) -> str | None: ...

    def acknowledge(self, msg: str) -> str: ...

    def support(self, msg: str) -> str: ...

    def released(self) -> dict[str, Any] | None: ...


class ConsoleOperator:
    """The person at the arm, through a ``TorqueGuard`` (src/robot/prompts.py): every prompt
    answered with torque on reads servo health before and after the wait and waits at most
    ``deploy.prompt_torque_timeout_s``. ``confirm``: ENTER = yes, q = quit."""

    source = "operator"

    def __init__(self, guard: TorqueGuard):
        self.guard = guard

    def _ask(self, prompt: str) -> str:
        reply = self.guard.ask(prompt)
        if reply.how == CTRL_C:
            raise KeyboardInterrupt
        if reply.how == HANGUP:
            raise Hangup(f"terminal hung up at {prompt.strip()!r}")
        if reply.how == EOF:
            raise EOFError(f"stdin closed at {prompt.strip()!r}")
        if reply.how == TIMEOUT:
            t = self.guard.wait.timeout_s
            limit = "" if t is None else f" for {t:.0f} s"
            raise PromptTimeout(f"no answer{limit} at {prompt.strip()!r}")
        assert reply.how == ENTER, reply.how
        return reply.answer

    def confirm(self, msg: str) -> bool:
        return self._ask(f"{msg} [ENTER / q] ") not in ("q", "quit")

    def outcome(self, result: TrialResult) -> str:
        while True:
            ans = self._ask(f"trial ended ({result.stop_reason}). success? [y/n] ")
            if ans in ("y", "n"):
                return "success" if ans == "y" else "failure"

    def ack_stop(self, reason: str) -> str | None:
        ans = self._ask(
            f"STOPPED: {reason}\narm is holding; type 'resume' to continue, else quit: "
        )
        return "operator typed resume" if ans == "resume" else None

    def acknowledge(self, msg: str) -> str:
        return self.guard.acknowledge(msg)

    def support(self, msg: str) -> str:
        return self.guard.support(msg)

    def released(self) -> dict[str, Any] | None:
        return self.guard.released


class ScriptedOperator:
    """Mock runs only: says yes to every prompt at once, scores success iff the trial ran to
    ``trial_max_steps`` without a stop, acknowledges HOLD stops so the session continues, and
    answers ENTER at the support / acknowledgement prompts (a scripted ENTER: ``outcome_source``
    and the run's ``mode`` say it was a mock)."""

    source = "scripted"

    def confirm(self, msg: str) -> bool:
        return True

    def outcome(self, result: TrialResult) -> str:
        return "success" if result.stop_reason == "max_steps" else "failure"

    def ack_stop(self, reason: str) -> str | None:
        return "scripted acknowledgement (mock run)"

    def acknowledge(self, msg: str) -> str:
        return ENTER

    def support(self, msg: str) -> str:
        return ENTER

    def released(self) -> dict[str, Any] | None:
        return None


PlacementFn = Callable[[int], tuple[str, dict[str, Any]]]
"""trial_idx -> (what to tell the operator to place where, the placement for the record)."""

# stop reason -> fault kind, for a trial whose policy started (``trial_fault``)
FAULT_KIND = {
    "servo_fault": "servo",  # over-temperature / over-current: torque off at once
    "serial_fault": "serial",  # bus error: torque off
    "safety_reject": "safety",  # a joint / workspace violation stopped the trial
    "deadman": "safety",  # lost camera frame, stall (tracking error): HOLD
    "stale_chunk": "safety",  # the chunk's first action too old: HOLD
    "operator_abort": "operator_abort",  # Ctrl-C while the arm moved: HOLD
    "client_error": "client_error",  # a bug in the client: HOLD
    "hangup": "hangup",  # SIGHUP / SIGTERM (the terminal closed) during motion: HOLD
    "session_lost": "session_lost",  # the process ended before the trial was recorded
    "prompt_timeout": "prompt_timeout",  # a torque-on prompt timed out (not-started attempts)
    TORQUE_OFF_UNCONFIRMED: TORQUE_OFF_UNCONFIRMED,
}
"""``server_*`` (timeout, unavailable, error, bad reply: the deadman held) is kind ``server``."""


def trial_fault(result: TrialResult, where: str = "trial") -> dict[str, Any] | None:
    """The fault that ended a started trial, or None (it ran to ``trial_max_steps``)."""
    if result.stop_reason == "max_steps":
        return None
    kind = (
        "server"
        if result.stop_reason.startswith("server_")
        else FAULT_KIND.get(result.stop_reason, "safety")
    )
    return {
        "kind": kind,
        "stop_reason": result.stop_reason,
        "reason": result.stop_detail,
        "where": where,
    }


def _torque_off_end(arm: So101Driver) -> str:
    if arm.torque_off_unconfirmed is not None:
        return f"{TORQUE_OFF_UNCONFIRMED}: {arm.torque_off_unconfirmed}"
    return f"torque_off: {arm.deadman.reason}"


def _ready(operator: Operator, arm: So101Driver, msg: str) -> str | None:
    """A torque-on prompt between trials: None = go on, else why the session ends."""
    try:
        return None if operator.confirm(msg) else "operator_quit"
    except PromptTimeout as e:
        return f"prompt_timeout: {e}"
    except (ServoFault, SerialFault):  # a health read around the wait: torque is now off
        return _torque_off_end(arm)


Scored = tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]
"""(outcome, outcome_source, fault, unjudged)."""


def _score(
    loop: DeployLoop, operator: Operator, result: TrialResult, settle: dict[str, Any] | None
) -> Scored:
    """Judging rule real-judge-2 (scripts/wo07_real_eval.py). A trial whose policy never started
    is an attempt (``not_started``, never scored). A started trial that ended in a fault, or
    faults during the settle / at the score prompt, is a failure (``fault``) and the operator is
    not asked. A clean one settles, the servos are read, then the operator judges it; a score
    prompt that ends without a judgement (Ctrl-C, EOF, timeout) is a failure (``unjudged``,
    doubt = n)."""
    arm = loop.arm
    if not result.policy_started:
        if settle is not None:
            settle["skipped"] = "policy never started"
        return "not_started", "not_scored", None, None
    fault = trial_fault(result)
    if fault is None and arm.torque_off_unconfirmed is not None:  # stamped last, below
        fault = {
            "kind": TORQUE_OFF_UNCONFIRMED,
            "stop_reason": result.stop_reason,
            "where": "trial",
        }
    if fault is not None:
        if settle is not None:
            settle["skipped"] = f"fault: {fault['kind']}"
        return "failure", "fault", fault, None
    stage = "the settling interval"
    try:
        if settle is not None:
            t_settle = loop.clock()
            loop.sleep(float(settle["settle_s"]))  # hold still before scoring
            settle["measured_s"] = round(loop.clock() - t_settle, 3)
            stop = loop.stop_requested()  # a Ctrl-C (flag) / hang-up during the settle
            if stop is not None and stop.kind == "hangup":
                raise Hangup(stop.reason)
            if stop is not None:
                raise KeyboardInterrupt
            try:
                arm.check_servos()  # the arm held under torque while it settled
            except (ServoFault, SerialFault) as e:  # torque now off: the same physical event
                settle["fault"] = str(e)
                return "failure", "fault", _late_fault(result, e, "settle"), None
        stage = "the score prompt"
        return operator.outcome(result), operator.source, None, None
    except (ServoFault, SerialFault) as e:  # heat built up while the operator judged
        return "failure", "fault", _late_fault(result, e, "score_prompt"), None
    except KeyboardInterrupt:
        how, why = CTRL_C, f"operator Ctrl-C during {stage}"
    except EOFError as e:
        if isinstance(e, Hangup):
            how, why = HANGUP, f"terminal hung up during {stage}"
        else:
            how, why = EOF, f"stdin closed (EOF) during {stage}"
    except PromptTimeout as e:
        how, why = TIMEOUT, f"{e}; torque released"
    return "failure", "unjudged", None, {"how": how, "reason": f"{why}: doubt = n (real-judge-2)"}


def _late_fault(result: TrialResult, e: BaseException, where: str) -> dict[str, Any]:
    """A servo / serial fault after the trial itself ended (settle read, score prompt)."""
    result.stop_reason = "servo_fault" if isinstance(e, ServoFault) else "serial_fault"
    when = "the settling interval" if where == "settle" else "the score prompt"
    result.stop_detail = f"after {when}: {e}"
    fault = trial_fault(result, where)
    assert fault is not None
    return fault


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
    on_trial_start: Callable[[int], None] | None = None,
    indices: Sequence[int] | None = None,
    placement_fn: PlacementFn | None = None,
    record_extra: Mapping[str, Any] | None = None,
    settle_s: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Home -> trial -> outcome for each trial index (``indices``, default ``range(n_trials)``).
    Returns (trial records, why the session ended). Records are appended to ``trials`` as they
    finish, so a Ctrl-C at a prompt keeps them. ``on_trial_start(k)`` runs right before each
    policy trial (the mock pilot resets its path); ``placement_fn(k)`` supplies the placement
    prompt and record (scripts/wo07_real_eval.py); ``record_extra`` is merged into every record.
    ``settle_s``: after a clean trial the arm holds still this long (``loop.sleep``), the servos
    are read once, and only then is the operator asked to score it; recorded as ``settle``.

    Scoring (``_score``, judging rule real-judge-2): ``policy_started`` (the trial's first action
    from the policy server was handed to the safety filter) decides whether an attempt counts.
    Not started (home failed / faulted / Ctrl-C, a server failure or a stale first chunk before
    the first action, a failed torque-off before it, the placement prompt ended by q / timeout /
    Ctrl-C / closed stdin / a servo read): ``not_started``, kept in the attempts log, never
    scored, the slot is retried on a resume. Started: scored exactly once. A fault after the
    start (servo, serial, safety-layer / deadman stop, server failure, operator Ctrl-C during
    motion, a hang-up, client error, torque off unconfirmed; in the trial, at the settle read or
    around the score prompt) is a failure with ``fault: {kind, stop_reason, reason, where}``,
    never judged by the operator, and a later 'resume' does not change it. A score prompt that
    ends without a judgement (Ctrl-C, EOF, timeout) is a failure with ``unjudged: {how, reason}``.

    Durable start: the moment the policy starts, a ``trial_started`` row (the record as it would
    stand if the process died there: a failure, fault kind ``session_lost``) goes to ``log_row``;
    a resume (scripts/wo07_real_eval.load_prior) turns one without its ``trial`` row into that
    failure, so a started slot is never re-rolled. An exception that ends the session between
    the start and the record (a crash, a Ctrl-C without the flag handler) appends that record
    before it propagates.

    The session ends on: q (``operator_quit``), a torque-on prompt timeout (``prompt_timeout``:
    torque released), a servo / serial fault (``torque_off: ...``), a failed torque-off, a
    Ctrl-C during motion (``operator_abort``), a hang-up (``hangup: ...``), Ctrl-C at the score
    prompt (``operator_ctrl_c``), a declined resume after a HOLD stop. EOF at the score prompt
    raises EOFError after the trial is recorded; Ctrl-C / EOF at the other prompts propagate. The
    operator supports the arm after a fault in the shutdown (``safe_shutdown``) that follows
    every session end."""
    if task not in protocol.TASKS or cell not in protocol.CELLS:
        raise ValueError(f"unknown task/cell {task}/{cell}")
    arm = loop.arm
    trials = [] if trials is None else trials
    for k in range(n_trials) if indices is None else indices:
        if arm.torque_off_unconfirmed is not None:  # no motion is possible: never start a trial
            return trials, _torque_off_end(arm)
        obj = trial_object(task, cell, k)
        instruction = protocol.instruction(task, obj)
        tag = f"[{task}/{cell} {k + 1}/{n_trials}]"
        if placement_fn is None:
            place_msg, placement = f"place the {obj.label} (grid slot {k})", None
        else:
            place_msg, placement = placement_fn(k)
        end = _ready(operator, arm, f"{tag} clear the workspace; ENTER moves the arm HOME")
        if end is not None:
            return trials, end
        loop.clear_ctrl_c()  # a press while nothing moved: the operator answered since
        start = utc_now()
        slot: dict[str, Any] = {"pending": None, "recorded": False}
        home: dict[str, Any] = {"reached": False}

        def record(
            result: TrialResult,
            home: Mapping[str, Any] | None,
            scored: Scored,
            settle: Mapping[str, Any] | None = None,
            *,
            kind: str = "trial",
            k: int = k,
            instruction: str = instruction,
            start: str = start,
            placement: Mapping[str, Any] | None = placement,
            slot: dict[str, Any] = slot,
        ) -> dict[str, Any]:
            outcome, source, fault, unjudged = scored
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
                outcome_source=source,
                home=home,
                camera_calibrated=camera_calibrated,
                servo_health=None if arm.monitor is None else arm.monitor.summary(),
                fault=fault,
                unjudged=unjudged,
            )
            if placement is not None:
                rec["placement"] = dict(placement)
            if settle is not None:
                rec["settle"] = dict(settle)
            if record_extra:
                rec.update(record_extra)
            stamp_torque_off_unconfirmed(rec, arm)  # last: nothing after it overwrites the flag
            if kind == "trial":
                trials.append(rec)
                slot["recorded"] = True
            if log_row is not None:
                log_row(kind=kind, **rec)
            return rec

        def on_policy_start(
            res: TrialResult,
            home: dict[str, Any] = home,
            slot: dict[str, Any] = slot,
            record: RecordFn = record,
        ) -> None:
            slot["pending"] = record(
                res, home, ("failure", "fault", dict(SESSION_LOST), None), kind="trial_started"
            )

        loop.on_policy_start = on_policy_start
        try:
            result: TrialResult | None = None
            try:
                home.update(loop.go_home())
            except ServoFault as e:  # torque already off
                home.update(reached=False, error=str(e))
                result = TrialResult(stop_reason="servo_fault", stop_detail=str(e))
            except (DeadmanTripped, SerialFault) as e:
                home.update(reached=False, error=str(e))
            except SessionStop as e:  # Ctrl-C / hang-up flag between ticks: hold, never park
                why = f"{e.reason} during the move home"
                ignoring_ctrl_c(functools.partial(arm.trip, why), arm, _say)
                home.update(reached=False, error=e.reason)
                result = TrialResult(stop_reason=e.kind, stop_detail=f"{e.reason} during home")
            except KeyboardInterrupt:  # arm in motion: hold, never park (port released by trip)
                arm.trip("operator Ctrl-C during the move home")
                home.update(reached=False, error="operator Ctrl-C")
                result = TrialResult(stop_reason="operator_abort", stop_detail="Ctrl-C during home")
            except Exception as e:  # a client bug while the arm moves: hold (deadman), then raise
                log.exception("move home failed")
                arm.trip(f"client error during the move home: {type(e).__name__}: {e}")
                raise
            if result is None and home["reached"]:
                end = _placement(operator, arm, f"{tag} {place_msg}; ENTER starts", home, record)
                if end is not None:
                    return trials, end
                loop.clear_ctrl_c()
                if log_row is not None:
                    loop.on_decision = lambda t, k=k: log_row(
                        kind="decision", cell=cell, trial_idx=k, **t
                    )
                if on_trial_start is not None:
                    on_trial_start(k)
                result = loop.run_trial(instruction, max_steps)
            elif result is None:
                result = TrialResult(stop_reason="home_failed", stop_detail=str(home))
                if not arm.deadman.tripped:
                    arm.hold()
            settle: dict[str, Any] | None = None
            if settle_s is not None:
                settle = {"settle_s": float(settle_s), "measured_s": None}
            scored = _score(loop, operator, result, settle)
            rec = record(result, home, scored, settle)
        except BaseException as e:
            pending = slot["pending"]
            if pending is not None and not slot["recorded"]:  # started, never recorded
                pending["fault"] = {
                    **SESSION_LOST,
                    "reason": f"{type(e).__name__} after the policy started, before the trial "
                    f"was recorded: {e}",
                }
                stamp_torque_off_unconfirmed(pending, arm)
                trials.append(pending)
                if log_row is not None:
                    log_row(kind="trial", **pending)
            raise
        finally:
            loop.on_policy_start = None
        unjudged = scored[3]
        log.info("%s %s: %s (%d steps)", tag, rec["outcome"], rec["stop_reason"], result.steps)
        if unjudged is not None:
            if unjudged["how"] == HANGUP:  # not an operator choice: the session ends as an error
                raise Hangup(unjudged["reason"])
            if unjudged["how"] == EOF:
                raise EOFError(unjudged["reason"])
            if unjudged["how"] == CTRL_C:
                return trials, "operator_ctrl_c"
            if unjudged["how"] == TIMEOUT:
                return trials, f"prompt_timeout: {unjudged['reason']}"
        if arm.torque_off_unconfirmed is not None:
            return trials, _torque_off_end(arm)
        if result.stop_reason == "operator_abort":
            return trials, "operator_abort"
        if result.stop_reason == "hangup":
            return trials, f"hangup: {result.stop_detail}"
        if arm.deadman.tripped:
            if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
                return trials, _torque_off_end(arm)
            try:
                ack = operator.ack_stop(arm.deadman.reason)
            except PromptTimeout as e:
                return trials, f"prompt_timeout: {e}"
            except (ServoFault, SerialFault):
                return trials, _torque_off_end(arm)
            if ack is None:
                return trials, f"stopped: {arm.deadman.reason}"
            arm.deadman.reset(ack)
            arm.safety.reset_reference()
    return trials, "completed"


SESSION_LOST = {
    "kind": "session_lost",
    "stop_reason": "session_lost",
    "reason": "the process ended after the policy started, before the trial was recorded",
    "where": "trial",
}
"""The fault of a started trial whose record was never written (``trial_started`` marker)."""

RecordFn = Callable[..., dict[str, Any]]


def _placement(
    operator: Operator, arm: So101Driver, msg: str, home: Mapping[str, Any], record: RecordFn
) -> str | None:
    """The placement prompt ("ENTER starts"), arm at home under torque. None = start the policy.
    Any other end (q, a timeout, a servo / serial fault from the health reads, Ctrl-C, a closed
    stdin, a hang-up) appends a ``not_started`` attempt first (``stop_reason`` says which, the
    fault if any), then returns the session end or re-raises."""

    def attempt(stop: str, detail: str, fault_kind: bool = False) -> None:
        res = TrialResult(stop_reason=stop, stop_detail=f"placement prompt: {detail}")
        fault = trial_fault(res, "placement_prompt") if fault_kind else None
        record(res, home, ("not_started", "not_scored", fault, None))

    try:
        end = _ready(operator, arm, msg)
    except KeyboardInterrupt:
        attempt("operator_ctrl_c", "operator Ctrl-C")
        raise
    except EOFError as e:
        attempt("hangup" if isinstance(e, Hangup) else "stdin_closed", str(e) or "EOF")
        raise
    if end is None:
        return None
    if end == "operator_quit":
        attempt("operator_quit", end)
    elif end.startswith("prompt_timeout"):
        attempt("prompt_timeout", end, fault_kind=True)
    elif end.startswith(TORQUE_OFF_UNCONFIRMED):
        attempt(TORQUE_OFF_UNCONFIRMED, end, fault_kind=True)
    else:  # torque_off: a servo / serial fault from the health reads around the wait
        attempt("servo_fault" if arm.servo_fault is not None else "serial_fault", end, True)
    return end


def stamp_torque_off_unconfirmed(rec: dict[str, Any], arm: So101Driver) -> dict[str, Any]:
    """Every trial record says whether torque off was confirmed. When the driver latched a failed
    torque-off, the stop reason becomes ``torque_off_unconfirmed`` (the trial's own reason is kept
    as ``stop_reason_before``). A started trial is then a failure with fault kind
    ``torque_off_unconfirmed`` (the earlier fault kept as ``fault_before``); an attempt whose
    policy never started stays ``not_started``. Call it last on a record."""
    detail = arm.torque_off_unconfirmed
    rec[TORQUE_OFF_UNCONFIRMED] = detail is not None
    if detail is not None:
        if rec.get("stop_reason") != TORQUE_OFF_UNCONFIRMED:
            rec["stop_reason_before"] = rec.get("stop_reason")
        rec["stop_reason"] = TORQUE_OFF_UNCONFIRMED
        rec["torque_off_unconfirmed_detail"] = detail
        if rec.get("policy_started"):
            before = rec.get("fault")
            if not (before and before.get("kind") == TORQUE_OFF_UNCONFIRMED):
                rec["fault"] = {
                    "kind": TORQUE_OFF_UNCONFIRMED,
                    "stop_reason": TORQUE_OFF_UNCONFIRMED,
                    "reason": detail,
                    "where": (before or {}).get("where", "trial"),
                    "fault_before": before,
                }
            else:
                rec["fault"] = {**before, "stop_reason": TORQUE_OFF_UNCONFIRMED, "reason": detail}
            rec["outcome"], rec["outcome_source"] = "failure", "fault"
        else:
            rec["outcome"], rec["outcome_source"] = "not_started", "not_scored"
    return rec


def final_session_end(arm: So101Driver, end: str) -> str:
    """The session end to record: a latched torque-off failure wins over any other end."""
    if arm.torque_off_unconfirmed is not None and not end.startswith(TORQUE_OFF_UNCONFIRMED):
        return f"{TORQUE_OFF_UNCONFIRMED}: {arm.torque_off_unconfirmed} (session end was {end})"
    return end


HEALTHY_ENDS = ("completed", "operator_quit", "operator_ctrl_c")
"""Session ends with the arm idle at a prompt: eligible for a park (if the deadman is clear)."""

SUPPORT_PROMPT = "arm HOLDING under torque; support it, then press ENTER to disable torque: "
REST_PROMPT = "press ENTER once the arm rests safely (supported or on the table): "
REST_SERVO_PROMPT = "press ENTER once the arm rests safely and the servo supply is off: "
POWER_OFF_PROMPT = "press ENTER once the servo supply is switched OFF at the inline power switch: "


@dataclass(frozen=True)
class Shutdown:
    """How a real-arm session ended (``safe_shutdown``).

    ``end``: parked | supported | released_unconfirmed | torque_was_off | torque_off_unconfirmed.
    ``support_confirmed`` is True only after an explicit ENTER at a prompt that asked the operator
    to support the arm or check it rests safely (``supported``; ``torque_was_off`` acknowledged);
    None after a completed park (the arm rests at the verified park pose; nobody was asked).
    ``released_unconfirmed``: torque went off after a torque-on prompt timed out, stdin closed or
    the terminal hung up, with ``cause`` timeout | eof | hangup; nobody confirmed support.
    ``ack``: the acknowledgement wait at the end, {prompt, given, how}. A shutdown that raised
    is recorded by the scripts as ``shutdown: "error: ..."`` with support_confirmed false."""

    end: str
    support_confirmed: bool | None  # None: parked, nobody was asked
    cause: str | None = None
    ack: dict[str, Any] | None = None
    released: dict[str, Any] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "end": self.end,
            "support_confirmed": self.support_confirmed,
            "cause": self.cause,
            "ack": self.ack,
            "released": self.released,
        }


def _torque_off_unconfirmed(arm: So101Driver, operator: Operator) -> Shutdown:
    """A torque-off write failed (driver latch): retry once (harmless), tell the operator to cut
    the power at the inline switch, and wait for the acknowledgement (recorded with how it was
    given). Nothing else is written."""
    arm.torque_off()  # retry: the latch stays set whatever it returns
    msg = TORQUE_OFF_UNCONFIRMED_MSG.format(detail=arm.torque_off_unconfirmed)
    log.critical(msg)
    _say(f"\n{msg}")
    if arm.servo_fault is not None:
        _say(SERVO_FAULT_PROMPT.format(reason=arm.servo_fault))
    how = operator.acknowledge(POWER_OFF_PROMPT)
    return Shutdown(
        TORQUE_OFF_UNCONFIRMED,
        support_confirmed=False,
        ack=ack_record(POWER_OFF_PROMPT, how, confirms="servo supply switched off"),
        released=operator.released(),
    )


def _support_after_fault(arm: So101Driver, operator: Operator, reason: str) -> dict[str, Any]:
    """Torque is off after a serial or servo fault: the arm is limp and may have dropped. Tell
    the operator to support it (and, after a servo fault, to switch off the supply) and wait for
    the acknowledgement. Returns it, {prompt, given, how}; only ENTER is ``given``."""
    if arm.servo_fault is not None:
        msg, prompt = SERVO_FAULT_PROMPT.format(reason=arm.servo_fault), REST_SERVO_PROMPT
    else:
        msg = (
            f"FAULT ({reason}). Torque is OFF: the arm is limp and may have dropped. Support it / "
            "lower it onto the table."
        )
        prompt = REST_PROMPT
    log.critical(msg)
    _say(f"\n{msg}")
    return ack_record(prompt, operator.acknowledge(prompt))


def _torque_already_off(arm: So101Driver, reason: str, operator: Operator) -> Shutdown:
    """torque_was_off (support confirmed only by an ENTER at the rest prompt), or
    torque_off_unconfirmed if the retry (or an earlier write) failed."""
    if arm.torque_off_unconfirmed is not None or not arm.torque_off():  # retry, harmless
        return _torque_off_unconfirmed(arm, operator)
    _say(f"torque is already OFF ({reason}): the arm is limp; check it is resting safely")
    ack = _support_after_fault(arm, operator, reason)
    return Shutdown("torque_was_off", support_confirmed=bool(ack["given"]), ack=ack)


def _released(arm: So101Driver, operator: Operator, released: dict[str, Any]) -> Shutdown:
    """A torque-on prompt timed out or stdin closed: the guard already told the operator to
    support the arm, waited the grace period and turned torque off. Nobody confirmed support."""
    if arm.torque_off_unconfirmed is not None or not arm.torque_off():  # retry, harmless
        return _torque_off_unconfirmed(arm, operator)
    cause = str(released.get("cause"))
    msg = (
        f"torque was RELEASED after {cause} at {released.get('prompt')!r}: support was NOT "
        "confirmed; check the arm rests safely"
    )
    log.critical(msg)
    _say(msg)
    return Shutdown("released_unconfirmed", support_confirmed=False, cause=cause, released=released)


def safe_shutdown(
    arm: So101Driver,
    operator: Operator,
    *,
    healthy: bool = False,
    park: ParkSettings | None = None,
    frame_time_fn: Callable[[], float | None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
) -> Shutdown:
    """End of a real-arm session (``Shutdown``).

    * Torque off unconfirmed (an earlier torque-off write failed, driver latch): nothing is
      written; torque-off retried; the operator is told to cut power at the inline switch.
    * Torque already released after a torque-on prompt timed out or stdin closed
      (``operator.released()``): ``released_unconfirmed``; nothing is asked again.
    * Deadman latched TORQUE_OFF (serial or servo fault): nothing is written; torque-off retried;
      the operator supports the arm and acknowledges (``_support_after_fault``).
    * ``healthy`` (see ``HEALTHY_ENDS``) with ``park`` (``robot.park.enabled``) and a frame
      source, deadman clear: park at ``park.qpos`` through the safety filter, then torque off. A
      park that fails falls through.
    * Otherwise: hold the measured pose so the arm stays up, read servo health, and ask the
      operator to support it (``operator.support``). Only ENTER confirms (``supported``, then
      torque off). Ctrl-C there is ignored (the arm keeps holding; the bounded wait keeps
      running), also when it lands in the hold or the servo read; a closed stdin, a hang-up or
      the timeout releases torque after the grace period: ``released_unconfirmed`` with the
      cause.
    ``stop_fn`` (the Ctrl-C / hang-up flag) stops the park, like Ctrl-C.
    Any torque-off that fails on the way returns ``torque_off_unconfirmed``. Leaving the driver's
    ``with`` block disables torque again.
    """
    if arm.torque_off_unconfirmed is not None:
        return _torque_off_unconfirmed(arm, operator)
    released = operator.released()
    if released is not None:
        return _released(arm, operator, released)
    if arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF:
        return _torque_already_off(arm, arm.deadman.reason, operator)
    if healthy and park is not None and frame_time_fn is not None and not arm.deadman.tripped:
        _say("session ended normally: parking the arm (it moves), then torque off")
        if arm.park(park, frame_time_fn, stop_fn=stop_fn):
            _say("arm PARKED; torque OFF")
            return Shutdown("parked", support_confirmed=None)
        if arm.torque_off_unconfirmed is not None:
            return _torque_off_unconfirmed(arm, operator)
        if arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF:
            return _torque_already_off(arm, arm.deadman.reason, operator)
        _say("park did not complete: the arm holds; support it")
    try:
        ignoring_ctrl_c(arm.hold, arm, _say)  # a panicked Ctrl-C must not skip the hold
        how = operator.support(SUPPORT_PROMPT)  # servo health read before the wait
    except (ServoFault, SerialFault):  # torque is now off (or not confirmed off)
        return _torque_already_off(arm, arm.deadman.reason, operator)
    if how != ENTER:  # eof / hangup / timeout: torque was released after the grace period
        if arm.torque_off_unconfirmed is not None or not arm.torque_off():
            return _torque_off_unconfirmed(arm, operator)
        return Shutdown(
            "released_unconfirmed",
            support_confirmed=False,
            cause=how,
            ack=ack_record(SUPPORT_PROMPT, how),
            released=operator.released(),
        )
    if not ignoring_ctrl_c(arm.torque_off, arm, _say):
        return _torque_off_unconfirmed(arm, operator)
    return Shutdown("supported", support_confirmed=True, ack=ack_record(SUPPORT_PROMPT, how))


def shutdown_session(loop: DeployLoop, operator: Operator, end: str) -> Shutdown:
    """``safe_shutdown`` for a session that ended with ``end``: park only on a healthy end
    (``HEALTHY_ENDS``, deadman clear), never on a fault or an abort during motion."""
    healthy = end in HEALTHY_ENDS and not loop.arm.deadman.tripped
    loop.clear_ctrl_c()  # a press before the shutdown must not stop the park at once
    return safe_shutdown(
        loop.arm,
        operator,
        healthy=healthy,
        park=loop.dcfg.park,
        frame_time_fn=loop.frame_time,
        stop_fn=lambda: loop.stop_requested() is not None,
    )


def shutdown_or_record(
    results: dict[str, Any], loop: DeployLoop, operator: Operator, end: str
) -> Shutdown:
    """``shutdown_session``; if it raises, the run record still says so (``shutdown: error``,
    support not confirmed, the session end) before the exception propagates."""
    try:
        return shutdown_session(loop, operator, end)
    except BaseException as e:
        results.update(
            session_end=final_session_end(loop.arm, end),
            shutdown=f"error: {type(e).__name__}: {e}",
            shutdown_detail=None,
            support_confirmed=False,
            torque_off_unconfirmed=loop.arm.torque_off_unconfirmed,
        )
        raise


def shutdown_results(sd: Shutdown) -> dict[str, Any]:
    """The run-record fields for a shutdown (pilot and real eval)."""
    return {
        "shutdown": sd.end,
        "shutdown_detail": sd.record(),
        "support_confirmed": sd.support_confirmed,
    }


def latency_lines(summary: Mapping[str, Any]) -> list[str]:
    """Console lines for the latency part of ``summarize`` (never labelled end-to-end unless an
    optical capture offset was measured)."""
    lat = summary[LATENCY_KEY]
    target = summary["latency_target_ms"]
    lines = [
        (
            f"  frame return -> first write ms (EXCLUDES capture): median {lat['median']}  "
            f"p95 {lat['p95']}  (n {lat['n']}; WO-07 target < {target:.0f} is end-to-end)"
        )
    ]
    e2e = summary.get("end_to_end_estimate_ms")
    if e2e:
        lines.append(
            f"  end-to-end estimate ms (+ measured capture offset {e2e['capture_offset_ms']}): "
            f"median {e2e['median']}  p95 {e2e['p95']}"
        )
    else:
        lines.append("  end-to-end latency: NOT measured (docs/bringup/latency-measurement.md)")
    return lines
