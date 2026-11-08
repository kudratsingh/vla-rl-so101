"""Fakes for the robot-side tests. No serial port, no camera, no lerobot hardware access."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import load_config
from src.robot.camera import CameraFault, Frame
from src.robot.units import JOINT_NAMES, URDF_LIMITS_RAD


def toy_fk(q: np.ndarray) -> np.ndarray:
    """TEST ONLY - NOT the SO-101 kinematics (that is src/env/kinematics.py from WO-01).

    Pan + a planar 2-link arm: at q = 0 the TCP is at (0.26, 0, 0.10) m; positive shoulder_lift
    or elbow_flex lowers it, so a large lift drives the TCP below the table (z < 0).
    """
    pan, lift, elbow = float(q[0]), float(q[1]), float(q[2])
    l1, l2, base_h = 0.12, 0.14, 0.10
    r = l1 * math.cos(lift) + l2 * math.cos(lift + elbow)
    z = base_h - l1 * math.sin(lift) - l2 * math.sin(lift + elbow)
    return np.array([r * math.cos(pan), r * math.sin(pan), z])


def real_cfg() -> dict[str, Any]:
    return load_config("real")


@dataclass(frozen=True)
class FakeCal:
    """lerobot's ``MotorCalibration`` fields (no lerobot import: CI installs no robot extra)."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


MID_TICK = 2048
GRIPPER_RANGE = (2000, 3500)  # raw ticks, closed .. open (arbitrary, like a real calibration)
PORT_BUSY = "[TxRxResult] Port is in use!"  # scservo_sdk getTxRxResult(COMM_PORT_BUSY)
CURRENT_MA_PER_COUNT = 6.5


def full_sweep_calibration() -> dict[str, FakeCal]:
    """A calibration whose sweep reached the URDF limits (src/robot/units.py's assumption):
    arm joints mid = 2048 ticks, wrist_roll lerobot's fixed 0..4095, gripper GRIPPER_RANGE."""
    cal: dict[str, FakeCal] = {}
    for i, name in enumerate(JOINT_NAMES):
        if name == "gripper":
            lo, hi = GRIPPER_RANGE
        elif name == "wrist_roll":
            lo, hi = 0, 4095
        else:
            half = round(math.degrees(URDF_LIMITS_RAD[name][1]) * 4095 / 360)
            lo, hi = MID_TICK - half, MID_TICK + half
        cal[name] = FakeCal(i + 1, 0, 0, lo, hi)
    return cal


class FakeSerial:
    def __init__(self) -> None:
        self.input_resets = 0

    def reset_input_buffer(self) -> None:
        self.input_resets += 1


class FakePortHandler:
    """scservo_sdk ``PortHandler`` subset: ``is_using`` (the busy flag), ``clearPort``, ``ser``."""

    def __init__(self) -> None:
        self.is_using = False
        self.ser = FakeSerial()
        self.clears = 0

    def clearPort(self) -> None:
        self.clears += 1


class FakeBus:
    """Records torque/goal traffic like lerobot's FeetechMotorsBus. Raw position ticks: arm
    2048 + int(deg*4095/360) (lerobot DEGREES with mid 2048), gripper range_min + pct% of the range.

    Every transaction fails with scservo_sdk's "Port is in use!" while ``port_handler.is_using``
    is set (a Ctrl-C that interrupted a transaction leaves it set). Temperature (deg C) and current
    (mA, read back as 6.5 mA counts) per motor come from ``follower.temperature_c`` /
    ``follower.current_ma``. ``eeprom``: the servos' Min/Max_Position_Limit (default = the file).
    """

    def __init__(self, follower: FakeFollower):
        self.follower = follower
        self.torque_enabled = False
        self.calls: list[tuple[str, Any]] = []
        self.fail_disable = False
        self.fail_health_read = False
        self.port_handler = FakePortHandler()
        self.calibration: dict[str, FakeCal] = full_sweep_calibration()
        self.eeprom: dict[str, FakeCal] | None = None
        self.max_temperature_limit = 70

    def _check_port(self, what: str) -> None:
        if self.port_handler.is_using:
            raise ConnectionError(f"Failed to {what} {PORT_BUSY}")

    def raw_positions(self) -> dict[str, int]:
        out = {}
        for m, v in self.follower.positions_lerobot().items():
            if m == "gripper":
                c = self.calibration[m]
                out[m] = c.range_min + round(v / 100 * (c.range_max - c.range_min))
            else:
                out[m] = MID_TICK + int(v * 4095 / 360)
        return out

    def sync_read(self, data_name: str, motors: Any = None, *, normalize: bool = True) -> Any:
        self._check_port(f"sync read '{data_name}'")
        self.calls.append(("sync_read", data_name))
        if data_name == "Present_Temperature":
            if self.fail_health_read:
                raise ConnectionError("Failed to sync read 'Present_Temperature'")
            return {m: int(v) for m, v in self.follower.temperature_c.items()}
        if data_name == "Present_Current":
            if self.fail_health_read:
                raise ConnectionError("Failed to sync read 'Present_Current'")
            return {m: round(v / CURRENT_MA_PER_COUNT) for m, v in self.follower.current_ma.items()}
        if data_name == "Max_Temperature_Limit":
            return {m: self.max_temperature_limit for m in JOINT_NAMES}
        return self.raw_positions()

    def read_calibration(self) -> dict[str, FakeCal]:
        self._check_port("read calibration")
        self.calls.append(("read_calibration", None))
        return dict(self.calibration if self.eeprom is None else self.eeprom)

    def sync_write(self, data_name: str, values: Any, *, normalize: bool = True) -> None:
        self._check_port(f"sync write '{data_name}'")
        self.calls.append(("sync_write", (data_name, dict(values))))

    def enable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self._check_port("write 'Torque_Enable'")
        self.calls.append(("enable_torque", None))
        self.torque_enabled = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(("disconnect", disable_torque))
        self.follower.connected = False

    def disable_torque(self, motors: Any = None, num_retry: int = 0) -> None:
        self._check_port("write 'Torque_Enable'")
        self.calls.append(("disable_torque", None))
        if self.fail_disable:
            raise ConnectionError("bus gone")
        self.torque_enabled = False


class FakeFollower:
    """Stands in for lerobot's SO101Follower. Positions in LeRobot units (deg, gripper 0-100).

    ``track``: the arm reaches every goal instantly (perfect servo).
    ``interrupt_next``: "read" / "write" -> the next get_observation / send_action is interrupted
    by a Ctrl-C mid-transaction: the port is left busy and KeyboardInterrupt is raised.
    ``temperature_c`` / ``current_ma``: what the servo-health registers read (deg C, mA).
    """

    def __init__(self, start: dict[str, float] | None = None, track: bool = True):
        self.pos = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
        self.pos["gripper.pos"] = 50.0
        if start:
            self.pos.update(start)
        self.track = track
        self.bus = FakeBus(self)
        self.calibration = self.bus.calibration  # lerobot: robot and bus share the file's dict
        self._calibrated = True
        self.connected = False
        self.sent: list[dict[str, float]] = []
        self.fail_read = False
        self.fail_write = False
        self.fail_connect = False
        self.interrupt_next: str | None = None
        self.temperature_c: dict[str, float] = {n: 35.0 for n in JOINT_NAMES}
        self.current_ma: dict[str, float] = {n: 100.0 for n in JOINT_NAMES}

    def _maybe_interrupt(self, kind: str) -> None:
        if self.interrupt_next == kind:
            self.interrupt_next = None
            self.bus.port_handler.is_using = True  # txPacket set it; rxPacket never ran
            raise KeyboardInterrupt

    def positions_lerobot(self) -> dict[str, float]:
        return {k.removesuffix(".pos"): v for k, v in self.pos.items()}

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        assert calibrate is False, "the driver must never auto-calibrate"
        self.connected = True
        self.bus.disable_torque()  # like SOFollower.configure(): torque off ... torque on
        if self.fail_connect:
            raise ConnectionError("Failed to write 'P_Coefficient'")
        self.bus.enable_torque()

    def disconnect(self) -> None:
        self.bus.disable_torque(num_retry=5)  # disable_torque_on_disconnect=True
        self.connected = False

    def get_observation(self) -> dict[str, Any]:
        self._maybe_interrupt("read")
        self.bus._check_port("sync read 'Present_Position'")
        if self.fail_read:
            raise ConnectionError("Failed to sync read 'Present_Position'")
        return dict(self.pos)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._maybe_interrupt("write")
        self.bus._check_port("sync write 'Goal_Position'")
        if self.fail_write:
            raise ConnectionError("Failed to sync write 'Goal_Position'")
        self.sent.append(dict(action))
        if self.track:
            self.pos.update(action)
        return dict(action)


class FakeClock:
    """Shared fake ``perf_counter``: ``sleep`` advances it instantly. Share ONE instance between
    the driver, the deadman, the camera, the server and the loop, or stamps look stale / future."""

    def __init__(self, t0: float = 100.0):
        self.t = float(t0)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += max(0.0, float(dt))

    def sleep(self, dt: float) -> None:
        self.advance(dt)


class FakeCam:
    """640x480 RGB frames stamped with ``clock``. Failure switches:

    ``fail`` / ``fail_after`` / ``fail_reads`` (all reads / reads > n / these read numbers):
    raise CameraFault (a lost frame);
    ``stall_after`` (reads): keep returning the last frame, old stamp and all (a frozen stream);
    ``interrupt_after`` (reads): raise KeyboardInterrupt (the operator's Ctrl-C).
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        h, w = 480, 640
        yy, xx = np.mgrid[0:h, 0:w]
        self.image = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
        self.reads = 0
        self.fail = False
        self.fail_after: int | None = None
        self.fail_reads: set[int] = set()
        self.stall_after: int | None = None
        self.interrupt_after: int | None = None
        self.last: Frame | None = None
        self.last_fresh_time: float | None = None

    def read(self) -> Frame:
        self.reads += 1
        if self.interrupt_after is not None and self.reads > self.interrupt_after:
            raise KeyboardInterrupt
        if (
            self.fail
            or self.reads in self.fail_reads
            or (self.fail_after is not None and self.reads > self.fail_after)
        ):
            raise CameraFault("timed out waiting for a frame")
        if self.stall_after is not None and self.reads > self.stall_after and self.last:
            return self.last
        self.last = Frame(self.image, self.clock())
        self.last_fresh_time = self.last.capture_time
        return self.last
