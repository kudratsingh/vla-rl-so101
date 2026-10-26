"""Real-arm safety (CLAUDE.md hard rule 8): every command passes through here before the servo bus.

All angles are canonical radians (``src.robot.units``); config values are degrees / metres / ms.

``SafetyFilter.filter(target, measured, now)`` runs these checks in this order. Each step can only
shrink the motion, so the returned command satisfies all of them at once:

0. Deadman latched -> no command at all (status ``deadman``); the driver holds or cuts torque.
1. Joint reading must be 6 finite values, else the deadman trips (TORQUE_OFF: the bus is suspect).
2. Stall/collision guard: if the last command and the measured position differ by more than
   ``max_tracking_error_deg`` on an ARM joint, the deadman trips (HOLD at the measured pose, which
   zeroes the servo's position error). A blocked STS3215 pushing at a large error overheats.
   The gripper is exempt: squeezing an object is a deliberate tracking error, and LeRobot's
   ``configure()`` already caps gripper torque/current in firmware.
   Then the target must be 6 finite values, else it is rejected (hold).
3. Joint limits: clip into [lo, hi] (``joint_limits_deg``, which can only narrow the URDF limits
   shrunk by ``urdf_limit_margin_deg``). If the reference pose is already outside (e.g. the folded
   rest pose at start-up) the bound for that joint is widened to the reference: never further out.
4. Per-step clip and velocity cap, per joint, relative to the reference (last sent command, or the
   measured pose on the first step / after a reset):
   ``|cmd - ref| <= min(max_joint_step_deg, max_velocity_deg_s * dt)`` with
   ``dt = min(time since last command, 1 / control_hz)``. A loop running faster than control_hz
   therefore gets smaller steps; a slower loop never gets bigger ones.
5. Workspace box (TCP from the injected ``fk``, robot base frame): the command is REJECTED and the
   last safe command is held if the requested target's TCP or the clipped command's TCP is outside
   ``workspace_box_m`` and farther from the box than the reference pose's TCP. With the arm inside
   the box this is simply "outside -> rejected"; from an outside start pose it only admits motion
   that does not move away from the box. Not projected onto the box: projecting a TCP needs IK on
   a 5-DoF arm and would silently execute a motion the policy never asked for. FK errors fail closed.

Every intervention is logged (``logging``, logger ``src.robot.safety``) with its reason and is
returned in ``SafetyResult.interventions`` for run logs.

``Deadman``: latched until an operator ``reset``. Trips on a missing/stale camera frame (older than
``deadman_frame_timeout_ms``; also on a frame stamped in the future, which means mismatched clocks)
-> HOLD; on any serial error -> TORQUE_OFF; on a stall (above) -> HOLD. Frame timestamps must come
from ``time.perf_counter`` (what LeRobot's OpenCVCamera stamps frames with).
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from src.config import Config
from src.robot.units import ARM_JOINTS, JOINT_NAMES, NUM_JOINTS, URDF_LIMITS_RAD, FloatArray

log = logging.getLogger(__name__)

FkFn = Callable[[FloatArray], npt.ArrayLike]
"""Canonical joint radians (6,) -> TCP xyz (3,) in metres, robot base frame."""

# Ceilings no config may exceed (STS3215 servos overheat under aggressive commands).
HARD_MAX_STEP_DEG = 10.0
HARD_MAX_VELOCITY_DEG_S = 180.0
DEFAULT_URDF_MARGIN_DEG = 5.0
DEFAULT_TRACKING_ERROR_DEG = 20.0
_WS_EPS_M = 1e-6
_FUTURE_FRAME_TOLERANCE_S = 0.05


# ------------------------------------------------------------------------------------------------
# Deadman
# ------------------------------------------------------------------------------------------------
class DeadmanResponse(str, Enum):
    HOLD = "hold"  # bus healthy: command the measured pose, stop the loop
    TORQUE_OFF = "torque_off"  # bus suspect: disable torque (arm goes limp), stop the loop


_SEVERITY = {DeadmanResponse.HOLD: 1, DeadmanResponse.TORQUE_OFF: 2}


class Deadman:
    """Latched stop. Once tripped, nothing is sent until an operator calls ``reset``."""

    def __init__(self, frame_timeout_s: float, clock: Callable[[], float] = time.perf_counter):
        if not (math.isfinite(frame_timeout_s) and frame_timeout_s > 0):
            raise ValueError(f"frame_timeout_s must be > 0, got {frame_timeout_s}")
        self.frame_timeout_s = float(frame_timeout_s)
        self.clock = clock
        self._reasons: list[str] = []
        self._response: DeadmanResponse | None = None

    @property
    def tripped(self) -> bool:
        return self._response is not None

    @property
    def response(self) -> DeadmanResponse | None:
        return self._response

    @property
    def reason(self) -> str:
        return "; ".join(self._reasons)

    def trip(self, reason: str, response: DeadmanResponse) -> None:
        """Latch. A later, more severe trip escalates HOLD -> TORQUE_OFF; never de-escalates."""
        self._reasons.append(reason)
        if self._response is None or _SEVERITY[response] > _SEVERITY[self._response]:
            self._response = response
        log.critical("DEADMAN TRIPPED -> %s: %s", self._response.value, reason)

    def check_frame(self, capture_time: float | None, now: float | None = None) -> bool:
        """True iff the camera frame is fresh. Trips (HOLD) on a missing, stale or future frame."""
        if self.tripped:
            return False
        t = self.clock() if now is None else now
        if capture_time is None:
            self.trip("no camera frame", DeadmanResponse.HOLD)
            return False
        age = t - capture_time
        if age > self.frame_timeout_s:
            self.trip(
                f"stale camera frame: age {age * 1e3:.0f} ms > {self.frame_timeout_s * 1e3:.0f} ms",
                DeadmanResponse.HOLD,
            )
            return False
        if age < -_FUTURE_FRAME_TOLERANCE_S:
            self.trip(
                f"camera frame stamped {-age * 1e3:.0f} ms in the future: camera and control "
                "loop use different clocks (must be time.perf_counter)",
                DeadmanResponse.HOLD,
            )
            return False
        return True

    def serial_error(self, exc: BaseException) -> None:
        self.trip(f"serial error: {type(exc).__name__}: {exc}", DeadmanResponse.TORQUE_OFF)

    def reset(self, operator_ack: str) -> None:
        """Clear the latch. Only call on an explicit human action (e.g. typed confirmation)."""
        if not operator_ack.strip():
            raise ValueError("deadman reset needs a non-empty operator acknowledgement")
        log.warning("deadman reset by operator (%r); cleared: %s", operator_ack, self.reason)
        self._reasons.clear()
        self._response = None


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------
def resolve_joint_limits_deg(safety_cfg: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """Canonical joint limits (degrees): URDF limits shrunk by the margin, intersected with
    ``joint_limits_deg`` (which can only narrow them, never widen)."""
    margin = float(safety_cfg.get("urdf_limit_margin_deg", DEFAULT_URDF_MARGIN_DEG))
    if not (0.0 <= margin < 45.0):
        raise ValueError(f"urdf_limit_margin_deg out of range: {margin}")
    configured: Mapping[str, Any] = safety_cfg.get("joint_limits_deg") or {}
    unknown = set(configured) - set(JOINT_NAMES)
    if unknown:
        raise ValueError(f"joint_limits_deg has unknown joints {sorted(unknown)}")
    out: dict[str, tuple[float, float]] = {}
    for name in JOINT_NAMES:
        lo_rad, hi_rad = URDF_LIMITS_RAD[name]
        lo, hi = math.degrees(lo_rad) + margin, math.degrees(hi_rad) - margin
        if name in configured:
            c_lo, c_hi = (float(v) for v in configured[name])
            if not c_lo < c_hi:
                raise ValueError(f"joint_limits_deg[{name}] must be [lo, hi] with lo < hi")
            lo, hi = max(lo, c_lo), min(hi, c_hi)
        if not lo < hi:
            raise ValueError(f"empty joint range for {name}: [{lo:.1f}, {hi:.1f}] deg")
        out[name] = (lo, hi)
    return out


@dataclass(frozen=True)
class SafetyConfig:
    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]
    workspace_lo_m: tuple[float, float, float]
    workspace_hi_m: tuple[float, float, float]
    max_joint_step_rad: float
    max_velocity_rad_s: float
    control_hz: float
    max_tracking_error_rad: float
    frame_timeout_s: float

    def __post_init__(self) -> None:
        if len(self.lower_rad) != NUM_JOINTS or len(self.upper_rad) != NUM_JOINTS:
            raise ValueError("joint limits need one [lo, hi] per joint")
        for name, lo, hi in zip(JOINT_NAMES, self.lower_rad, self.upper_rad, strict=True):
            u_lo, u_hi = URDF_LIMITS_RAD[name]
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                raise ValueError(f"bad limits for {name}: [{lo}, {hi}]")
            if lo < u_lo - 1e-9 or hi > u_hi + 1e-9:
                raise ValueError(f"limits for {name} exceed the URDF limits [{u_lo}, {u_hi}] rad")
        for a, b in zip(self.workspace_lo_m, self.workspace_hi_m, strict=True):
            if not (math.isfinite(a) and math.isfinite(b) and a < b):
                raise ValueError(
                    f"bad workspace box {self.workspace_lo_m} .. {self.workspace_hi_m}"
                )
        step_deg = math.degrees(self.max_joint_step_rad)
        vel_deg = math.degrees(self.max_velocity_rad_s)
        if not 0.0 < step_deg <= HARD_MAX_STEP_DEG:
            raise ValueError(f"max_joint_step_deg must be in (0, {HARD_MAX_STEP_DEG}]: {step_deg}")
        if not 0.0 < vel_deg <= HARD_MAX_VELOCITY_DEG_S:
            raise ValueError(
                f"max_velocity_deg_s must be in (0, {HARD_MAX_VELOCITY_DEG_S}]: {vel_deg}"
            )
        if not self.control_hz > 0:
            raise ValueError("control_hz must be > 0")
        if not self.max_tracking_error_rad > 0:
            raise ValueError("max_tracking_error_deg must be > 0")
        if not self.frame_timeout_s > 0:
            raise ValueError("deadman_frame_timeout_ms must be > 0")

    @classmethod
    def from_config(cls, cfg: Config) -> SafetyConfig:
        robot = cfg["robot"]
        s = robot["safety"]
        limits = resolve_joint_limits_deg(s)
        box = s["workspace_box_m"]
        lo3 = (float(box["x"][0]), float(box["y"][0]), float(box["z"][0]))
        hi3 = (float(box["x"][1]), float(box["y"][1]), float(box["z"][1]))
        return cls(
            lower_rad=tuple(math.radians(limits[n][0]) for n in JOINT_NAMES),
            upper_rad=tuple(math.radians(limits[n][1]) for n in JOINT_NAMES),
            workspace_lo_m=lo3,
            workspace_hi_m=hi3,
            max_joint_step_rad=math.radians(float(s["max_joint_step_deg"])),
            max_velocity_rad_s=math.radians(float(s["max_velocity_deg_s"])),
            control_hz=float(robot["control_hz"]),
            max_tracking_error_rad=math.radians(
                float(s.get("max_tracking_error_deg", DEFAULT_TRACKING_ERROR_DEG))
            ),
            frame_timeout_s=float(s["deadman_frame_timeout_ms"]) / 1000.0,
        )


# ------------------------------------------------------------------------------------------------
# Filter
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Intervention:
    kind: str  # invalid_action | joint_limit | step_clip | velocity_cap | workspace_reject
    reason: str
    joint: str | None = None
    requested_deg: float | None = None
    applied_deg: float | None = None


@dataclass(frozen=True)
class SafetyResult:
    command: FloatArray | None  # (6,) rad, safe to write; None when the deadman is tripped
    status: str  # "pass" | "clipped" | "rejected" | "deadman"
    interventions: tuple[Intervention, ...] = ()


_LOG_LEVEL = {"step_clip": logging.INFO, "velocity_cap": logging.INFO}


class SafetyFilter:
    def __init__(self, config: SafetyConfig, fk: FkFn, deadman: Deadman):
        self.config = config
        self.fk = fk
        self.deadman = deadman
        self._lo = np.asarray(config.lower_rad, dtype=np.float64)
        self._hi = np.asarray(config.upper_rad, dtype=np.float64)
        self._ws_lo = np.asarray(config.workspace_lo_m, dtype=np.float64)
        self._ws_hi = np.asarray(config.workspace_hi_m, dtype=np.float64)
        self._ref: FloatArray | None = None
        self._last_t: float | None = None
        self.counts: Counter[str] = Counter()

    @property
    def reference(self) -> FloatArray | None:
        return None if self._ref is None else self._ref.copy()

    def reset_reference(self) -> None:
        """Forget the last command; the next step is measured from the measured pose."""
        self._ref = None
        self._last_t = None

    def filter(self, target: npt.ArrayLike, measured: npt.ArrayLike, now: float) -> SafetyResult:
        if self.deadman.tripped:
            return SafetyResult(None, "deadman")
        if not math.isfinite(now):
            self.deadman.trip(f"invalid control-loop time {now!r}", DeadmanResponse.HOLD)
            return SafetyResult(None, "deadman")
        q_meas = np.asarray(measured, dtype=np.float64).reshape(-1)
        if q_meas.shape != (NUM_JOINTS,) or not np.all(np.isfinite(q_meas)):
            self.deadman.trip(f"invalid joint reading {q_meas}", DeadmanResponse.TORQUE_OFF)
            return SafetyResult(None, "deadman")
        ref = q_meas.copy() if self._ref is None else self._ref.copy()

        # 2. stall / collision guard (arm joints only)
        if self._ref is not None:
            err = np.abs(self._ref[:5] - q_meas[:5])
            j = int(np.argmax(err))
            if err[j] > self.config.max_tracking_error_rad:
                self.deadman.trip(
                    f"tracking error on {ARM_JOINTS[j]}: commanded {math.degrees(self._ref[j]):.1f}"
                    f" deg, measured {math.degrees(q_meas[j]):.1f} deg (> "
                    f"{math.degrees(self.config.max_tracking_error_rad):.1f} deg): blocked "
                    "joint, collision or stall",
                    DeadmanResponse.HOLD,
                )
                return SafetyResult(None, "deadman")

        q_req = np.asarray(target, dtype=np.float64).reshape(-1)
        if q_req.shape != (NUM_JOINTS,) or not np.all(np.isfinite(q_req)):
            iv = Intervention(
                "invalid_action", f"target must be {NUM_JOINTS} finite values: {q_req}"
            )
            return self._reject(ref, now, [iv])

        interventions: list[Intervention] = []

        # 3. joint limits, never further out than the reference
        lo = np.minimum(self._lo, ref)
        hi = np.maximum(self._hi, ref)
        q_lim = np.clip(q_req, lo, hi)
        for k in (int(i) for i in np.flatnonzero(q_lim != q_req)):
            interventions.append(
                self._joint_iv(
                    "joint_limit",
                    f"outside [{math.degrees(lo[k]):.1f}, {math.degrees(hi[k]):.1f}] deg",
                    k,
                    q_req[k],
                    q_lim[k],
                )
            )

        # 4. per-step clip and velocity cap
        nominal_dt = 1.0 / self.config.control_hz
        dt = nominal_dt if self._last_t is None else now - self._last_t
        dt_eff = min(max(dt, 0.0), nominal_dt)
        s_step = self.config.max_joint_step_rad
        s_vel = self.config.max_velocity_rad_s * dt_eff
        bound = min(s_step, s_vel)
        kind = "velocity_cap" if s_vel < s_step * (1 - 1e-9) else "step_clip"
        q_cmd = np.clip(q_lim, ref - bound, ref + bound)
        for k in (int(i) for i in np.flatnonzero(q_cmd != q_lim)):
            interventions.append(
                self._joint_iv(
                    kind,
                    f"|step| limited to {math.degrees(bound):.2f} deg (dt {dt_eff * 1e3:.0f} ms)",
                    k,
                    q_lim[k],
                    q_cmd[k],
                )
            )

        # 5. workspace box
        violation = self._workspace_violation(q_lim, q_cmd, ref)
        if violation is not None:
            return self._reject(
                ref, now, [*interventions, Intervention("workspace_reject", violation)]
            )

        for iv in interventions:
            self._log(iv)
        self._ref = q_cmd.copy()
        self._last_t = now
        return SafetyResult(q_cmd, "clipped" if interventions else "pass", tuple(interventions))

    # -- helpers --------------------------------------------------------------------------------
    def _reject(self, hold: FloatArray, now: float, ivs: list[Intervention]) -> SafetyResult:
        """Hold the last safe command (or the measured pose if nothing was sent yet)."""
        for iv in ivs:
            self._log(iv)
        log.warning("safety: command REJECTED, holding last safe pose")
        self._ref = hold.copy()
        self._last_t = now
        return SafetyResult(hold.copy(), "rejected", tuple(ivs))

    @staticmethod
    def _joint_iv(kind: str, reason: str, j: int, req: float, applied: float) -> Intervention:
        return Intervention(
            kind,
            reason,
            joint=JOINT_NAMES[j],
            requested_deg=math.degrees(float(req)),
            applied_deg=math.degrees(float(applied)),
        )

    def _log(self, iv: Intervention) -> None:
        self.counts[iv.kind] += 1
        where = (
            f" {iv.joint}: {iv.requested_deg:.2f} -> {iv.applied_deg:.2f} deg" if iv.joint else ""
        )
        log.log(
            _LOG_LEVEL.get(iv.kind, logging.WARNING), "safety %s%s (%s)", iv.kind, where, iv.reason
        )

    def _tcp(self, q: FloatArray) -> FloatArray:
        p = np.asarray(self.fk(q.copy()), dtype=np.float64).reshape(-1)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError(f"fk returned {p!r}, expected 3 finite values")
        return p

    def box_distance(self, p: FloatArray) -> float:
        """Euclidean distance (m) from point p to the workspace box; 0 inside."""
        d = np.maximum(np.maximum(self._ws_lo - p, 0.0), p - self._ws_hi)
        return float(np.linalg.norm(d))

    def _workspace_violation(
        self, q_target: FloatArray, q_cmd: FloatArray, ref: FloatArray
    ) -> str | None:
        try:
            d_ref = self.box_distance(self._tcp(ref))
            checks = (("requested target", self._tcp(q_target)), ("next command", self._tcp(q_cmd)))
        except Exception as e:  # noqa: BLE001 - any FK failure must fail closed (reject)
            return f"forward kinematics failed ({type(e).__name__}: {e})"
        for label, p in checks:
            d = self.box_distance(p)
            if d > d_ref + _WS_EPS_M:
                return (
                    f"{label} TCP ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m is {d * 100:.1f} cm outside "
                    f"workspace_box_m (reference pose {d_ref * 100:.1f} cm outside)"
                )
        return None


def make_safety(
    cfg: Config, fk: FkFn, clock: Callable[[], float] = time.perf_counter
) -> SafetyFilter:
    """SafetyFilter + its Deadman from ``configs/real.yaml`` (``robot.control_hz``, ``robot.safety``)."""
    sc = SafetyConfig.from_config(cfg)
    return SafetyFilter(sc, fk, Deadman(sc.frame_timeout_s, clock=clock))


def load_sim_fk(device: str = "cpu") -> FkFn:
    """The shared SO-101 forward kinematics (``src/env/kinematics.py``, lands from WO-01).

    Adapter for WO-01's ``ArmKinematics.from_urdf().fk(q (B,>=5)) -> (B,4,4)`` (base_link ->
    tcp_link). Raises with instructions when that module is not on this branch yet: without FK
    there is no workspace check, and without the workspace check nothing is sent to the arm.
    """
    try:
        torch = importlib.import_module("torch")
        kin_mod = importlib.import_module("src.env.kinematics")
        kin = kin_mod.ArmKinematics.from_urdf(device=device)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "Forward kinematics for the workspace check live in src/env/kinematics.py, which lands "
            "from the WO-01 branch (wo-01-sim-env). Merge WO-01 first; the arm is never driven "
            f"without the workspace check. ({type(e).__name__}: {e})"
        ) from e

    def fk(q: FloatArray) -> FloatArray:
        qt = torch.as_tensor(np.asarray(q, dtype=np.float32)[None, :5], device=device)
        with torch.no_grad():
            pose = kin.fk(qt)
        return np.asarray(pose[0, :3, 3].cpu().numpy(), dtype=np.float64)

    return fk
