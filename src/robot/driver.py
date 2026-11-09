"""Thin glue over LeRobot 0.4.4's ``SO101Follower``: canonical radians in/out, safety on every write.

    with So101Driver.from_config(cfg, fk=load_sim_fk()) as arm:
        frame = camera.read()
        out = arm.step(policy_action_rad, frame_time=frame.capture_time)

* ``step`` = deadman frame check -> read joints -> ``SafetyFilter`` -> bus write. The only way a
  policy action reaches the bus.
* On a deadman trip it performs the response (HOLD: write the measured pose as the goal, so the
  servos stop pushing; TORQUE_OFF: disable torque) and raises ``DeadmanTripped``: the loop stops.
  A caller may catch it, ask the operator, ``deadman.reset(...)`` and continue.
* Any exception from the bus is a serial fault: deadman TORQUE_OFF, then ``SerialFault`` is raised.
* ``check_frame`` (camera liveness without a write) and ``trip`` (any other failure, e.g. the
  policy server) end the same way: response carried out, ``DeadmanTripped``.
* Leaving the ``with`` block always disables torque (LeRobot's ``disable_torque_on_disconnect``;
  explicitly first if leaving on an exception). The arm then goes limp and drops onto the table:
  keep it low, or support it, when a session ends.
* ``connect`` never calibrates (calibration is interactive: ``scripts/wo06_calibrate.py``) and
  refuses an arm whose motors do not match the calibration file. Every torque enable (LeRobot's
  ``configure()`` re-enables torque at the end of ``connect``) is preceded by
  Goal_Position := Present_Position, so a servo never drives towards a stale goal register.
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
from src.robot.units import FloatArray, UnitMap

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


@dataclass(frozen=True)
class StepResult:
    measured: FloatArray  # (6,) rad, read just before the command
    safety: SafetyResult  # .command is what was written to the bus


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


def install_hold_on_torque_enable(bus: Any) -> None:
    """Make every ``bus.enable_torque`` first write Goal_Position := Present_Position (raw ticks).

    A Feetech servo drives to its Goal_Position register the moment torque is enabled; LeRobot's
    connect()/configure() enable torque without touching that register. Instance-level wrap of
    lerobot 0.4.4's ``FeetechMotorsBus.enable_torque``; idempotent.
    """
    if getattr(bus, "_vla_hold_on_enable", False):
        return
    original = bus.enable_torque

    def enable_torque(motors: Any = None, num_retry: int = 0) -> None:
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
    ):
        self.follower = follower
        self.units = units
        self.safety = safety
        self.deadman = safety.deadman
        self.clock = clock
        self._connected = False

    @classmethod
    def from_config(cls, cfg: Config, fk: FkFn) -> So101Driver:
        return cls(follower_from_config(cfg), UnitMap.from_config(cfg), make_safety(cfg, fk))

    # -- lifecycle ------------------------------------------------------------------------------
    def connect(self) -> None:
        calib = getattr(self.follower, "calibration", None)
        if calib is not None and not calib:
            raise NotCalibrated(
                "no calibration file for this arm id; run `uv run scripts/wo06_calibrate.py "
                "--preset real calibrate --device follower`"
            )
        install_hold_on_torque_enable(self.follower.bus)
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

    # -- bus primitives -------------------------------------------------------------------------
    def torque_off(self) -> bool:
        """Disable torque on every motor. True on success; on failure the operator must cut power."""
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
        raise: ``raise arm.trip("server timeout")``."""
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

    # -- control --------------------------------------------------------------------------------
    def step(
        self, action: npt.ArrayLike, *, frame_time: float | None, now: float | None = None
    ) -> StepResult:
        """One control tick. ``frame_time``: perf_counter stamp of the frame the action came from
        (None = no frame -> deadman). Raises DeadmanTripped / SerialFault; never sends unsafely."""
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
        return StepResult(measured=measured, safety=result)

    def move_to(
        self,
        target: npt.ArrayLike,
        frame_time_fn: Callable[[], float | None],
        tol_rad: float = math.radians(1.0),
        timeout_s: float = 20.0,
    ) -> bool:
        """Drive to ``target`` through the full safety filter at control_hz (e.g. a home pose).
        True when every joint is within ``tol_rad``. Stops (False) on a rejection or timeout."""
        q = np.asarray(target, dtype=np.float64)
        period = 1.0 / self.safety.config.control_hz
        t_end = self.clock() + timeout_s
        while self.clock() < t_end:
            t0 = self.clock()
            out = self.step(q, frame_time=frame_time_fn())
            if out.safety.status == "rejected":
                log.warning("move_to stopped: target rejected by safety")
                return False
            if np.all(np.abs(out.measured - q) <= tol_rad):
                return True
            time.sleep(max(0.0, period - (self.clock() - t0)))
        return False
