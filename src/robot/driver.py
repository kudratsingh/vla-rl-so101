"""Thin glue over LeRobot 0.4.4's ``SO101Follower``: canonical radians in/out, safety on every write.

    with So101Driver.from_config(cfg, fk=load_sim_fk()) as arm:
        frame = camera.read()
        out = arm.step(policy_action_rad, frame_time=frame.capture_time)
        ...
        arm.park(ParkSettings.from_config(cfg), frame_time_fn)   # healthy end only

* ``step`` = deadman frame check -> read joints -> ``SafetyFilter`` -> bus write (-> scheduled
  servo-health read). The only way a policy action reaches the bus.
* On a deadman trip it performs the response (HOLD: write the measured pose as the goal, so the
  servos stop pushing; TORQUE_OFF: disable torque) and raises ``DeadmanTripped``: the loop stops.
  A caller may catch it, ask the operator, ``deadman.reset(...)`` and continue.
* Any exception from the bus is a serial fault: deadman TORQUE_OFF, then ``SerialFault`` is raised.
* ``check_frame`` (camera liveness without a write) and ``trip`` (any other failure, e.g. the
  policy server) end the same way: response carried out, ``DeadmanTripped``.
* Servo health (``src/robot/servo_health.py``): with a ``ServoMonitor``, ``step`` reads
  Present_Temperature / Present_Current on a schedule after its write. Over a threshold: torque
  OFF at once (not HOLD, which keeps the hot servo energized; not a park, which adds motion),
  deadman latched TORQUE_OFF, ``ServoFault`` raised carrying ``SERVO_FAULT_PROMPT`` for the
  operator (support the arm).
* Healthy shutdown: ``park(settings, ...)`` drives to the park pose through the full safety
  filter, then disables torque. It refuses (returns False, nothing moves) once the deadman has
  tripped, so a fault path never parks; the caller then falls back to hold + "support the arm".
* Leaving the ``with`` block always disables torque (LeRobot's ``disable_torque_on_disconnect``;
  explicitly first if leaving on an exception). Without a park first the arm goes limp and drops
  onto the table: keep it low, or support it.
* ``connect`` never calibrates (calibration is interactive: ``scripts/wo06_calibrate.py``) and
  refuses an arm whose motors do not match the calibration file. Every torque enable (LeRobot's
  ``configure()`` re-enables torque at the end of ``connect``) is preceded by the rest-pose check
  below, then Goal_Position := Present_Position, so a servo never drives towards a stale goal.
* Torque-enable refusal (``check_rest_pose``): with torque still off, every joint's resting
  position is checked against (a) its calibrated range in raw ticks, from BOTH the calibration
  file and the servo EEPROM (Min/Max_Position_Limit, which must agree), and (b) the safety joint
  limits (``src/robot/safety.py``) widened by ``RestTolerance.safety_limits_deg`` (6 deg: those
  limits sit 5 deg inside the calibrated range, so a gripper resting closed or a joint resting
  on its stop passes). Outside either: ``TorqueEnableRefused`` with an operator message, torque
  stays OFF. A servo outside its EEPROM limits can be driven to the limit when torque comes on;
  a pose far outside the safety limits means the calibration or ``robot.units`` is wrong.
* Port-busy backstop (``release_port``): a Ctrl-C (KeyboardInterrupt) that lands inside
  scservo_sdk's packet loop leaves ``port_handler.is_using`` True, and every later transaction
  then fails with "[TxRxResult] Port is in use!" (``scservo_sdk/protocol_packet_handler.py:73-75``
  sets the flag in ``txPacket``; only ``rxPacket`` / the TxOnly paths clear it), so the hold or
  torque-off that should follow the Ctrl-C never reaches the servos. ``hold``, ``torque_off``,
  ``trip`` and ``park`` clear it first, as lerobot's own ``disconnect(disable_torque=True)`` does
  (``lerobot/motors/motors_bus.py:543-545``: ``clearPort()``, ``is_using = False``), and drop a
  half-read status packet from the serial input buffer.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt

from src.config import Config
from src.robot.safety import DeadmanResponse, FkFn, SafetyFilter, SafetyResult, make_safety
from src.robot.servo_health import REGISTERS, ServoMonitor, monitor_from_config
from src.robot.units import (
    JOINT_NAMES,
    LEROBOT_DEG_PER_TICK,
    FloatArray,
    UnitMap,
    raw_to_canonical,
)

log = logging.getLogger(__name__)


class FollowerLike(Protocol):
    """The subset of LeRobot's ``SOFollower`` used here (a fake implements it in tests)."""

    bus: Any

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_calibrated(self) -> bool: ...

    def connect(self, calibrate: bool = True) -> None: ...

    def disconnect(self) -> None: ...

    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...


class DeadmanTripped(RuntimeError):
    """The deadman latched; its response (hold / torque off) has been carried out."""


class SerialFault(RuntimeError):
    """The servo bus failed; torque-off has been attempted."""


class NotCalibrated(RuntimeError):
    pass


class ServoFault(DeadmanTripped):
    """Over-temperature / over-current: torque is already OFF (deadman latched TORQUE_OFF)."""


class TorqueEnableRefused(RuntimeError):
    """A joint rests outside its calibrated range or the safety limits: torque was not enabled."""


SERVO_FAULT_PROMPT = (
    "SERVO FAULT ({reason}). Torque is OFF on every motor: the arm is limp and may have dropped. "
    "Support it / lower it onto the table, switch off the servo supply, let the servos cool for "
    "at least 10 minutes, and find the cause (blocked joint, squeezing gripper, long hold) "
    "before the next session."
)

TORQUE_REFUSAL_MSG = (
    "TORQUE ENABLE REFUSED: torque stays OFF. {problems}\nWith torque off, move the joint(s) by "
    "hand back inside the range (support the arm), then reconnect. If a joint is inside its "
    "physical range but reported outside, the calibration is wrong: re-run `uv run "
    "scripts/wo06_calibrate.py --preset real calibrate --device follower`, then `check "
    "--write-limits`."
)


@dataclass(frozen=True)
class StepResult:
    measured: FloatArray  # (6,) rad, read just before the command
    safety: SafetyResult  # .command is what was written to the bus


@dataclass(frozen=True)
class RestTolerance:
    """How far (deg) a joint may rest outside a range at torque enable before it is refused."""

    calibrated_range_deg: float = 1.0
    safety_limits_deg: float = 6.0

    @classmethod
    def from_config(cls, cfg: Config) -> RestTolerance:
        t = cfg["robot"]["safety"].get("torque_enable_tol_deg") or {}
        d = cls()
        return cls(
            calibrated_range_deg=float(t.get("calibrated_range", d.calibrated_range_deg)),
            safety_limits_deg=float(t.get("safety_limits", d.safety_limits_deg)),
        )


@dataclass(frozen=True)
class ParkSettings:
    """Healthy-shutdown park pose and tolerance (``robot.park`` in configs/real.yaml)."""

    qpos: tuple[float, ...]
    tol_rad: float
    timeout_s: float

    def __post_init__(self) -> None:
        if len(self.qpos) != len(JOINT_NAMES) or not all(map(math.isfinite, self.qpos)):
            raise ValueError(f"robot.park.qpos must be {len(JOINT_NAMES)} finite radians")
        if not (self.tol_rad > 0 and self.timeout_s > 0):
            raise ValueError("robot.park tol_deg and timeout_s must be > 0")

    @classmethod
    def from_config(cls, cfg: Config) -> ParkSettings:
        """``robot.park``; ``qpos: null`` means ``deploy.home_qpos`` (sim REST_QPOS)."""
        p = cfg["robot"].get("park") or {}
        q = p.get("qpos")
        qpos = cfg["deploy"]["home_qpos"] if q is None else q
        return cls(
            qpos=tuple(float(v) for v in qpos),
            tol_rad=math.radians(float(p.get("tol_deg", 3.0))),
            timeout_s=float(p.get("timeout_s", 20.0)),
        )


def follower_from_config(cfg: Config) -> Any:
    """LeRobot ``SO101Follower`` for ``configs/real.yaml`` ``robot`` (no cameras: see camera.py)."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    r = cfg["robot"]
    if r.get("type") != "so101_follower":
        raise ValueError(f"robot.type must be so101_follower, got {r.get('type')!r}")
    port = str(r["port"])
    if "XXXX" in port:
        raise ValueError(
            "robot.port is still the placeholder; run `uv run scripts/wo06_calibrate.py "
            "--preset real find-port --write`"
        )
    if not r.get("use_degrees", True):
        raise ValueError("robot.use_degrees must be true (src/robot/units.py assumes degrees)")
    cal_dir = r.get("calibration_dir")
    fcfg = SO101FollowerConfig(
        port=port,
        id=str(r["id"]),
        calibration_dir=Path(cal_dir).expanduser() if cal_dir else None,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        max_relative_target=None,  # src/robot/safety.py is the single limiter
        cameras={},
    )
    return SO101Follower(fcfg)


def release_port(bus: Any) -> bool:
    """Clear a servo port left busy by an interrupted transaction (module doc). True if it was
    busy. Safe between transactions: the control loop is single-threaded, so ``is_using`` can only
    still be True here when a transaction was interrupted."""
    ph = getattr(bus, "port_handler", None)
    if ph is None or not getattr(ph, "is_using", False):
        return False
    log.critical("servo port was left busy by an interrupted transaction: clearing it")
    try:
        ph.clearPort()  # flush pending output, as lerobot's disconnect does
    except Exception as e:  # noqa: BLE001 - best effort; is_using must be cleared regardless
        log.warning("clearPort failed: %s", e)
    ph.is_using = False
    ser = getattr(ph, "ser", None)
    if ser is not None:
        try:
            ser.reset_input_buffer()  # drop a half-read status packet
        except Exception as e:  # noqa: BLE001
            log.warning("reset_input_buffer failed: %s", e)
    return True


def install_hold_on_torque_enable(bus: Any, guard: Callable[[Any], None] | None = None) -> None:
    """Make every ``bus.enable_torque`` first run ``guard(motors)`` (which raises to refuse, torque
    still off), then write Goal_Position := Present_Position (raw ticks).

    A Feetech servo drives to its Goal_Position register the moment torque is enabled; LeRobot's
    connect()/configure() enable torque without touching that register. Instance-level wrap of
    lerobot 0.4.4's ``FeetechMotorsBus.enable_torque``; idempotent (a later call replaces the guard).
    """
    bus._vla_torque_guard = guard
    if getattr(bus, "_vla_hold_on_enable", False):
        return
    original = bus.enable_torque

    def enable_torque(motors: Any = None, num_retry: int = 0) -> None:
        check = getattr(bus, "_vla_torque_guard", None)
        if check is not None:
            check(motors)
        present = bus.sync_read("Present_Position", motors, normalize=False)
        bus.sync_write("Goal_Position", present, normalize=False)
        original(motors, num_retry=num_retry)

    bus.enable_torque = enable_torque
    bus._vla_hold_on_enable = True


class So101Driver:
    def __init__(
        self,
        follower: FollowerLike,
        units: UnitMap,
        safety: SafetyFilter,
        clock: Callable[[], float] = time.perf_counter,
        *,
        monitor: ServoMonitor | None = None,
        rest_tol: RestTolerance | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.follower = follower
        self.units = units
        self.safety = safety
        self.deadman = safety.deadman
        self.clock = clock
        self.sleep = sleep
        self.monitor = monitor
        self.rest_tol = rest_tol or RestTolerance()
        self.servo_fault: str | None = None  # reason of the latest ServoFault
        self._connected = False

    @classmethod
    def from_config(cls, cfg: Config, fk: FkFn) -> So101Driver:
        return cls(
            follower_from_config(cfg),
            UnitMap.from_config(cfg),
            make_safety(cfg, fk),
            monitor=monitor_from_config(cfg),
            rest_tol=RestTolerance.from_config(cfg),
        )

    # -- lifecycle ------------------------------------------------------------------------------
    def connect(self) -> None:
        calib = getattr(self.follower, "calibration", None)
        if calib is not None and not calib:
            raise NotCalibrated(
                "no calibration file for this arm id; run `uv run scripts/wo06_calibrate.py "
                "--preset real calibrate --device follower`"
            )
        install_hold_on_torque_enable(self.follower.bus, guard=self.check_rest_pose)
        try:
            self.follower.connect(calibrate=False)
        except BaseException:
            log.critical("follower connect failed: torque off, closing the port")
            self.torque_off()
            try:
                self.follower.bus.disconnect(disable_torque=False)
            except Exception as e:  # noqa: BLE001 - best effort; the original error is raised
                log.warning("closing the port failed too: %s", e)
            raise
        self._connected = True
        if not self.follower.is_calibrated:
            self.torque_off()
            self.disconnect()
            raise NotCalibrated(
                "motor calibration does not match the calibration file; re-run "
                "`uv run scripts/wo06_calibrate.py --preset real calibrate --device follower`"
            )
        self.safety.reset_reference()
        self._log_firmware_limits()
        log.info("SO-101 follower connected")

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            self.follower.disconnect()  # disables torque (disable_torque_on_disconnect=True)
        except Exception as e:  # noqa: BLE001 - must fall through to a direct torque-off
            log.critical("disconnect failed (%s); attempting torque off directly", e)
            self.torque_off()

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            log.critical("leaving robot session on %s: torque off", exc_type.__name__)
            self.torque_off()
        self.disconnect()

    # -- torque-enable refusal ------------------------------------------------------------------
    def check_rest_pose(self, motors: Any = None) -> None:
        """Torque-enable guard; runs with torque off. Raises TorqueEnableRefused when a joint rests
        outside its calibrated range (file and EEPROM, raw ticks) by more than
        ``rest_tol.calibrated_range_deg``, or outside the safety joint limits by more than
        ``rest_tol.safety_limits_deg``. Always checks all six motors, whatever ``motors`` is."""
        bus = self.follower.bus
        file_cal = getattr(bus, "calibration", None) or {}
        try:
            eeprom = bus.read_calibration()
            raw = bus.sync_read("Present_Position", None, normalize=False)
        except Exception as e:
            msg = TORQUE_REFUSAL_MSG.format(problems=f"Could not read the servos: {e}.")
            log.critical(msg)
            raise TorqueEnableRefused(msg) from e
        problems: list[str] = []
        tol_ticks = self.rest_tol.calibrated_range_deg / LEROBOT_DEG_PER_TICK
        for name in JOINT_NAMES:
            fc, ec = file_cal.get(name), eeprom.get(name)
            if fc is None or ec is None or name not in raw:
                problems.append(f"{name}: no calibration or no position reading")
                continue
            f_rng = (int(fc.range_min), int(fc.range_max))
            e_rng = (int(ec.range_min), int(ec.range_max))
            if f_rng != e_rng:
                problems.append(
                    f"{name}: servo EEPROM limits {e_rng} differ from the calibration file {f_rng}"
                )
            lo, hi = max(f_rng[0], e_rng[0]), min(f_rng[1], e_rng[1])
            r = int(raw[name])
            out = max(lo - r, r - hi, 0)
            if out > tol_ticks:
                problems.append(
                    f"{name} rests at raw {r}, {out * LEROBOT_DEG_PER_TICK:.1f} deg outside its "
                    f"calibrated range [{lo}, {hi}] ticks"
                )
        if not problems:
            q = raw_to_canonical(self.units, raw, file_cal)
            sc = self.safety.config
            tol = math.radians(self.rest_tol.safety_limits_deg)
            for j, name in enumerate(JOINT_NAMES):
                lo_r, hi_r = sc.lower_rad[j], sc.upper_rad[j]
                out_r = max(lo_r - float(q[j]), float(q[j]) - hi_r, 0.0)
                if out_r > tol:
                    problems.append(
                        f"{name} rests at {math.degrees(q[j]):.1f} deg, "
                        f"{math.degrees(out_r):.1f} deg outside the safety limits "
                        f"[{math.degrees(lo_r):.1f}, {math.degrees(hi_r):.1f}] deg (tolerance "
                        f"{self.rest_tol.safety_limits_deg:.1f} deg)"
                    )
        if problems:
            msg = TORQUE_REFUSAL_MSG.format(problems="; ".join(problems) + ".")
            log.critical(msg)
            raise TorqueEnableRefused(msg)

    def _log_firmware_limits(self) -> None:
        """Log the firmware's own temperature cut-out next to the monitor threshold (best effort)."""
        if self.monitor is None:
            return
        try:
            limit = self.follower.bus.sync_read("Max_Temperature_Limit", None, normalize=False)
        except Exception as e:  # noqa: BLE001 - informational only
            log.warning("could not read Max_Temperature_Limit: %s", e)
            return
        trip = self.monitor.config.max_temperature_c
        log.warning(
            "servo firmware Max_Temperature_Limit (C): %s; monitor trips at %.0f C", limit, trip
        )
        if limit and trip >= min(float(v) for v in limit.values()):
            log.critical(
                "servo_monitor.max_temperature_c is not below the firmware limit: lower it"
            )

    # -- bus primitives -------------------------------------------------------------------------
    def torque_off(self) -> bool:
        """Disable torque on every motor. True on success; on failure the operator must cut power."""
        release_port(self.follower.bus)
        try:
            self.follower.bus.disable_torque(num_retry=5)
            log.warning("torque disabled on all motors")
            return True
        except Exception as e:  # noqa: BLE001 - report every failure as "cut power"
            log.critical("TORQUE OFF FAILED (%s: %s): CUT THE ARM'S POWER NOW", type(e).__name__, e)
            return False

    def _serial_fault(self, what: str, e: BaseException) -> SerialFault:
        self.deadman.serial_error(e)
        self.torque_off()
        return SerialFault(f"{what} failed: {type(e).__name__}: {e}")

    def read_state(self) -> FloatArray:
        """Measured joint positions, canonical radians (6,). Serial errors -> torque off + raise."""
        try:
            obs = self.follower.get_observation()
            return self.units.to_canonical(obs)
        except Exception as e:
            raise self._serial_fault("joint read", e) from e

    def _write(self, q: FloatArray) -> None:
        try:
            self.follower.send_action(self.units.to_lerobot(q))
        except Exception as e:
            raise self._serial_fault("joint write", e) from e

    def hold(self, measured: FloatArray | None = None) -> None:
        """Goal := measured pose (zero position error, servos stop pushing). Bypasses the filter:
        holding where the arm already is cannot move it. Refused (nothing read or written) while
        the deadman is latched TORQUE_OFF: the bus is suspect and torque must stay off."""
        if self.deadman.tripped and self.deadman.response is DeadmanResponse.TORQUE_OFF:
            log.critical("hold refused: torque is OFF after %s", self.deadman.reason)
            return
        release_port(self.follower.bus)
        q = self.read_state() if measured is None else measured
        self._write(q)
        self.safety.reset_reference()
        log.warning(
            "holding at measured pose (deg): %s",
            [round(math.degrees(float(v)), 1) for v in q],
        )

    def _respond_to_deadman(self, measured: FloatArray | None) -> DeadmanTripped:
        if self.deadman.response is DeadmanResponse.TORQUE_OFF:
            self.torque_off()
        else:
            try:
                self.hold(measured)
            except SerialFault:
                pass  # already escalated to torque off
        return DeadmanTripped(f"deadman ({self.deadman.response}): {self.deadman.reason}")

    def trip(self, reason: str, response: DeadmanResponse = DeadmanResponse.HOLD) -> DeadmanTripped:
        """Latch the deadman for a failure outside the bus and camera (policy server, operator
        Ctrl-C, client bug), carry out its response, and return the exception for the caller to
        raise: ``raise arm.trip("server timeout")``. A Ctrl-C may have interrupted a bus
        transaction, so the port is released before the response is written."""
        release_port(self.follower.bus)
        self.deadman.trip(reason, response)
        return self._respond_to_deadman(None)

    def check_frame(self, frame_time: float | None, now: float | None = None) -> None:
        """The deadman's camera check without a write (e.g. before a policy query): on a missing,
        stale or future frame, hold and raise DeadmanTripped."""
        if not self._connected:
            raise RuntimeError("driver not connected")
        if self.deadman.tripped:
            raise DeadmanTripped(f"deadman latched: {self.deadman.reason}")
        if not self.deadman.check_frame(frame_time, self.clock() if now is None else now):
            raise self._respond_to_deadman(None)

    # -- servo health ---------------------------------------------------------------------------
    def _raise_servo_fault(self, reason: str) -> ServoFault:
        """Torque off at once (no hold, no park), latch TORQUE_OFF, tell the operator."""
        self.servo_fault = reason
        self.deadman.trip(f"servo fault: {reason}", DeadmanResponse.TORQUE_OFF)
        self.torque_off()
        msg = SERVO_FAULT_PROMPT.format(reason=reason)
        log.critical(msg)
        return ServoFault(msg)

    def _read_health(self, register: str) -> None:
        assert self.monitor is not None
        t0 = self.clock()
        try:
            raw = self.follower.bus.sync_read(register, None, normalize=False)
        except Exception as e:
            raise self._serial_fault(f"{register} read", e) from e
        t1 = self.clock()
        self.monitor.note_read_ms((t1 - t0) * 1e3)
        fault = self.monitor.update(register, raw, t1)
        if fault is not None:
            raise self._raise_servo_fault(fault)

    def check_servos(self) -> None:
        """Read temperature and current now (before a park, or while holding at a prompt).
        Raises ServoFault (torque already off) or SerialFault. No-op without a monitor, before
        connect, or once torque is off."""
        if self.monitor is None or not self._connected:
            return
        if self.deadman.tripped and self.deadman.response is DeadmanResponse.TORQUE_OFF:
            return
        for reg in REGISTERS:
            self._read_health(reg)

    # -- control --------------------------------------------------------------------------------
    def step(
        self, action: npt.ArrayLike, *, frame_time: float | None, now: float | None = None
    ) -> StepResult:
        """One control tick. ``frame_time``: perf_counter stamp of the frame the action came from
        (None = no frame -> deadman). Raises DeadmanTripped (ServoFault included) / SerialFault;
        never sends unsafely."""
        if not self._connected:
            raise RuntimeError("driver not connected")
        t = self.clock() if now is None else now
        if self.deadman.tripped:
            raise DeadmanTripped(f"deadman latched: {self.deadman.reason}")
        if not self.deadman.check_frame(frame_time, t):
            raise self._respond_to_deadman(None)
        measured = self.read_state()
        result = self.safety.filter(action, measured, t)
        if result.command is None:
            raise self._respond_to_deadman(measured)
        self._write(result.command)
        if self.monitor is not None:
            reg = self.monitor.due()
            if reg is not None:  # after the write: the read never delays this tick's command
                self._read_health(reg)
        return StepResult(measured=measured, safety=result)

    def move_to(
        self,
        target: npt.ArrayLike,
        frame_time_fn: Callable[[], float | None],
        tol_rad: float = math.radians(1.0),
        timeout_s: float = 20.0,
        stop_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """Drive to ``target`` through the full safety filter at control_hz (e.g. a home pose).
        True when every joint is within ``tol_rad``. Stops (False) on a rejection, a timeout or
        ``stop_fn()`` returning True (checked before every tick)."""
        q = np.asarray(target, dtype=np.float64)
        period = 1.0 / self.safety.config.control_hz
        t_end = self.clock() + timeout_s
        while self.clock() < t_end:
            if stop_fn is not None and stop_fn():
                log.warning("move_to stopped by the operator")
                return False
            t0 = self.clock()
            out = self.step(q, frame_time=frame_time_fn())
            if out.safety.status == "rejected":
                log.warning("move_to stopped: target rejected by safety")
                return False
            if np.all(np.abs(out.measured - q) <= tol_rad):
                return True
            self.sleep(max(0.0, period - (self.clock() - t0)))
        return False

    # -- healthy shutdown -----------------------------------------------------------------------
    def park(
        self,
        settings: ParkSettings,
        frame_time_fn: Callable[[], float | None],
        stop_fn: Callable[[], bool] | None = None,
    ) -> bool:
        """HEALTHY SHUTDOWN ONLY: drive to ``settings.qpos`` through the full safety filter at
        control_hz, then disable torque. MOVES THE ARM. True when parked with torque off.

        Refuses (False, nothing written) when the deadman has tripped: a fault path never parks.
        Reads servo temperature/current first. False on a rejection, a timeout, ``stop_fn()``
        becoming True or a Ctrl-C; the arm is then holding (or limp after a fault) and the caller
        falls back to hold + "support the arm" + torque off.
        """
        if not self._connected:
            return False
        if self.deadman.tripped:
            log.warning("park refused: deadman latched (%s)", self.deadman.reason)
            return False
        release_port(self.follower.bus)
        try:
            self.check_servos()
            self.safety.reset_reference()  # first step measured from the pose the arm holds
            reached = self.move_to(
                settings.qpos,
                frame_time_fn,
                tol_rad=settings.tol_rad,
                timeout_s=settings.timeout_s,
                stop_fn=stop_fn,
            )
        except (DeadmanTripped, SerialFault) as e:
            log.critical("park stopped: %s", e)
            return False
        except KeyboardInterrupt:
            self.trip("operator Ctrl-C during park")
            return False
        try:
            if not reached:
                log.warning("park pose not reached: holding")
                self.hold()
                return False
            q = self.read_state()
        except SerialFault as e:  # torque already off
            log.critical("park ended on a serial fault: %s", e)
            return False
        log.warning("parked (deg): %s; torque off", [round(math.degrees(float(v)), 1) for v in q])
        return self.torque_off()
