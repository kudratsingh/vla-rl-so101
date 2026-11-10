"""Servo temperature / current monitor and its fault rule (project review 2026-11-07, 3.3 gap 2).

The driver (``So101Driver.step``) calls ``ServoMonitor.due()`` once per control tick, after the
tick's bus write, and on a due tick reads ONE register from all six servos with one
``sync_read(..., normalize=False)``. Registers alternate, so with ``read_every_ticks`` = k each
register is refreshed every 2k ticks:

    k = 4 at 15 Hz: one extra sync_read every 266.7 ms (3.75 Hz), each register every 533 ms.

Bus-time cost. A tick already does two bus transactions (``Present_Position`` sync_read, the
``Goal_Position`` sync_write). The monitor adds one sync_read on one tick in k. On the wire a
6-servo sync_read of a 1-2 byte register is ~14 instruction + 6 x 7-8 status bytes, about 60
bytes x 10 bits at 1 Mbaud = ~0.6 ms; the rest is USB-serial turnaround on the Mac (a few ms,
NOT measured on this arm). Every monitor read is timed with the driver clock and summarised
(``summary()['read_ms']``); a read slower than ``bus_budget_ms`` (10 ms, 15% of the 66.7 ms
period) is logged as a warning, and the driver's pacing absorbs it because the read happens
after the write, inside the tick's sleep budget.

Registers, units and scaling (verified against the installed lerobot 0.4.4,
``lerobot/motors/feetech/tables.py`` ``STS_SMS_SERIES_CONTROL_TABLE``; the STS3215 uses it):

* ``Present_Temperature``: address 63, 1 byte, degrees Celsius (raw value = degC).
* ``Present_Current``: address 69, 2 bytes, not in lerobot's sign-magnitude encoding table, so a
  raw read returns the register word. Scale ``current_ma_per_count`` = 6.5 mA per count (Feetech
  STS series memory table, the document lerobot's tables.py cites). Bit 15 is masked off before
  scaling: if the firmware uses it as a direction bit the magnitude is still right, and a real
  reading never reaches 2^15 counts (213 A). Consistent with lerobot's SO-101 ``configure()``
  writing ``Protection_Current`` = 250 "50% of max current" for the gripper
  (``lerobot/robots/so_follower/so_follower.py:169``): 500 counts x 6.5 mA = 3.25 A, the order of
  the STS3215's stall current.

Fault rule. Temperature: one reading at or above ``max_temperature_c`` trips (temperature is
slow and not noisy). Current: a motor whose readings stay above ``max_current_ma`` for at least
``current_trip_s`` trips (acceleration spikes are brief; a blocked or squeezing servo is not).
"Stay above" needs at least two consecutive over-threshold readings with no gap between two
current reads longer than ``max(2 * register_period_s, current_trip_s)``: after a longer
unmonitored gap (an operator prompt, the move home) the window restarts at the new reading, so
two isolated spikes minutes apart never count as sustained. ``check_servos()`` re-reads a motor
found over the threshold after ``current_trip_s`` before it returns.
The response belongs to the driver: TORQUE OFF at once with the operator told to support the
arm. Not HOLD (keeps the hot servo energized) and not a powered park (adds motion during the
fault). The thresholds in ``configs/real.yaml`` (55 C, 1500 mA for 1 s) are PROVISIONAL: kept by
the owner on 2026-11-08 for the first sessions, to be confirmed with the logged temperatures.

Not covered: the arm is not monitored continuously between control loops (an operator prompt
with the arm holding), because the serial bus has a single owner (one thread) and a background
reader would contend for the port with the control loop. The driver's ``check_servos()`` reads
both registers once on demand (plus the over-current re-read above): before a park, before
and after each torque-on teleop prompt (scene, keep, resume; the wait is bounded,
``teleop.prompt_torque_timeout_s``), before the final teleop support prompt, and after the
real-eval settling interval. The real-eval / pilot prompts (``src/robot/client.py``) wait
without a time limit and are not read during the wait: an accepted gap (docs/STATUS.md).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.config import Config

log = logging.getLogger(__name__)

TEMPERATURE = "Present_Temperature"
CURRENT = "Present_Current"
REGISTERS: tuple[str, str] = (TEMPERATURE, CURRENT)
CURRENT_MAGNITUDE_MASK = 0x7FFF


@dataclass(frozen=True)
class ServoHealthConfig:
    read_every_ticks: int
    max_temperature_c: float
    max_current_ma: float
    current_trip_s: float
    current_ma_per_count: float
    bus_budget_ms: float
    control_hz: float

    def __post_init__(self) -> None:
        if self.read_every_ticks < 1:
            raise ValueError("servo_monitor.read_every_ticks must be >= 1")
        for name in ("max_temperature_c", "max_current_ma", "current_ma_per_count", "control_hz"):
            v = float(getattr(self, name))
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"servo_monitor.{name} must be > 0, got {v}")
        if not 0.0 <= self.current_trip_s < 60.0:
            raise ValueError("servo_monitor.current_trip_s must be in [0, 60)")
        if not self.bus_budget_ms > 0:
            raise ValueError("servo_monitor.bus_budget_ms must be > 0")
        if self.max_temperature_c > 70.0:
            raise ValueError(
                "servo_monitor.max_temperature_c above 70 C: that is the STS3215 factory "
                "Max_Temperature_Limit, the firmware's own cut-out"
            )

    @property
    def register_period_s(self) -> float:
        """Time between two readings of the same register."""
        return len(REGISTERS) * self.read_every_ticks / self.control_hz

    @classmethod
    def from_config(cls, cfg: Config) -> ServoHealthConfig | None:
        """``robot.servo_monitor`` from configs/real.yaml; None when the block is absent."""
        robot = cfg["robot"]
        m: Mapping[str, Any] | None = robot.get("servo_monitor")
        if not m:
            return None
        return cls(
            read_every_ticks=int(m["read_every_ticks"]),
            max_temperature_c=float(m["max_temperature_c"]),
            max_current_ma=float(m["max_current_ma"]),
            current_trip_s=float(m["current_trip_s"]),
            current_ma_per_count=float(m["current_ma_per_count"]),
            bus_budget_ms=float(m["bus_budget_ms"]),
            control_hz=float(robot["control_hz"]),
        )


class ServoMonitor:
    """Schedules the reads and applies the fault rule. Pure logic: the driver does the bus I/O."""

    def __init__(self, config: ServoHealthConfig):
        self.config = config
        self._tick = 0
        self._next = 0  # index into REGISTERS
        self._over_since: dict[str, float] = {}
        self._over_n: dict[str, int] = {}  # consecutive over-threshold current readings
        self._last_current_t: float | None = None
        self.max_temperature_c: dict[str, float] = {}
        self.max_current_ma: dict[str, float] = {}
        self.last: dict[str, dict[str, float]] = {}
        self._reads = 0
        self._read_ms_sum = 0.0
        self._read_ms_max = 0.0
        self._slow_reads = 0

    def due(self) -> str | None:
        """Call once per control tick. The register to read on this tick, or None."""
        self._tick += 1
        if self._tick % self.config.read_every_ticks:
            return None
        reg = REGISTERS[self._next]
        self._next = (self._next + 1) % len(REGISTERS)
        return reg

    @property
    def max_current_gap_s(self) -> float:
        """Longest gap between two current reads that still counts as continuous monitoring."""
        return max(2 * self.config.register_period_s, self.config.current_trip_s)

    def pending_over_current(self) -> bool:
        """A motor's last current reading was over the threshold without tripping (yet)."""
        return bool(self._over_since)

    def note_read_ms(self, ms: float) -> None:
        self._reads += 1
        self._read_ms_sum += ms
        self._read_ms_max = max(self._read_ms_max, ms)
        if ms > self.config.bus_budget_ms:
            self._slow_reads += 1
            if self._slow_reads == 1:
                log.warning(
                    "servo monitor read took %.1f ms > bus budget %.1f ms: it delays the next "
                    "tick; raise robot.servo_monitor.read_every_ticks",
                    ms,
                    self.config.bus_budget_ms,
                )

    def current_ma(self, raw: int) -> float:
        return (int(raw) & CURRENT_MAGNITUDE_MASK) * self.config.current_ma_per_count

    def update(self, register: str, raw: Mapping[str, Any], now: float) -> str | None:
        """Feed one raw sync_read ({motor: value}). Returns the fault reason, or None."""
        c = self.config
        if register == TEMPERATURE:
            temps = {m: float(v) for m, v in raw.items()}
            self.last[TEMPERATURE] = temps
            for m, t in temps.items():
                self.max_temperature_c[m] = max(self.max_temperature_c.get(m, -math.inf), t)
            hot = {m: t for m, t in temps.items() if t >= c.max_temperature_c}
            if hot:
                worst = max(hot, key=lambda m: hot[m])
                return (
                    f"over-temperature: {worst} at {hot[worst]:.0f} C (>= "
                    f"{c.max_temperature_c:.0f} C); all temperatures {temps}"
                )
            return None
        if register == CURRENT:
            amps = {m: self.current_ma(v) for m, v in raw.items()}
            self.last[CURRENT] = amps
            for m, a in amps.items():
                self.max_current_ma[m] = max(self.max_current_ma.get(m, -math.inf), a)
            gap = math.inf if self._last_current_t is None else now - self._last_current_t
            self._last_current_t = now
            if gap > self.max_current_gap_s:  # unmonitored in between: nothing was "sustained"
                self._over_since.clear()
                self._over_n.clear()
            for m, a in amps.items():
                if a <= c.max_current_ma:
                    self._over_since.pop(m, None)
                    self._over_n.pop(m, None)
                    continue
                since = self._over_since.setdefault(m, now)
                n = self._over_n[m] = self._over_n.get(m, 0) + 1
                if now - since >= c.current_trip_s and (n >= 2 or c.current_trip_s == 0):
                    return (
                        f"over-current: {m} at {a:.0f} mA (> {c.max_current_ma:.0f} mA) for "
                        f"{now - since:.2f} s over {n} consecutive readings (>= "
                        f"{c.current_trip_s:.2f} s); all currents (mA) "
                        f"{ {k: round(v) for k, v in amps.items()} }"
                    )
            return None
        raise ValueError(f"not a monitored register: {register!r}")

    def summary(self) -> dict[str, Any]:
        n = self._reads
        return {
            "reads": n,
            "read_ms": {
                "mean": round(self._read_ms_sum / n, 3) if n else None,
                "max": round(self._read_ms_max, 3) if n else None,
                "over_budget": self._slow_reads,
                "budget": self.config.bus_budget_ms,
            },
            "read_every_ticks": self.config.read_every_ticks,
            "register_period_s": round(self.config.register_period_s, 4),
            "max_temperature_c": dict(self.max_temperature_c),
            "max_current_ma": {k: round(v, 1) for k, v in self.max_current_ma.items()},
        }


def monitor_from_config(cfg: Config) -> ServoMonitor | None:
    """A ServoMonitor for ``robot.servo_monitor``, or None when the block is absent."""
    mcfg = ServoHealthConfig.from_config(cfg)
    return None if mcfg is None else ServoMonitor(mcfg)


# ------------------------------------------------------------------------------------------------
# Supply voltage over the bus (WO-06 checklist step 2b; owner kit list 2026-11-10)
# ------------------------------------------------------------------------------------------------
# lerobot 0.4.4 ``STS_SMS_SERIES_CONTROL_TABLE``: Present_Voltage address 62, 1 byte, read-only;
# Max_Voltage_Limit 14 and Min_Voltage_Limit 15 (EEPROM), 1 byte each. Unit 0.1 V per count
# (Feetech STS memory table; unverified on this arm).
VOLTAGE = "Present_Voltage"
VOLTAGE_LIMITS: tuple[str, str] = ("Min_Voltage_Limit", "Max_Voltage_Limit")
VOLTS_PER_COUNT = 0.1


def read_voltages(bus: Any) -> dict[str, dict[str, int]]:
    """Raw ``Present_Voltage`` and the EEPROM voltage limits of every motor: three read-only
    ``sync_read(..., normalize=False)`` calls. Writes nothing; torque is left as it is."""
    out: dict[str, dict[str, int]] = {}
    for reg in (VOLTAGE, *VOLTAGE_LIMITS):
        raw = bus.sync_read(reg, None, normalize=False)
        out[reg] = {str(m): int(v) for m, v in raw.items()}
    return out


def voltage_report(
    raw: Mapping[str, Mapping[str, int]],
    supply_v: float,
    tolerance_frac: float,
    other_supply_v: float | None = None,
) -> dict[str, Any]:
    """Per motor: the reading and the servo's own limits in volts, and the problems: a reading
    more than ``tolerance_frac`` off the supply label (and, when it sits nearer the OTHER arm's
    supply, a crossed-supply warning), or outside the servo's own limits (it would alarm)."""
    motors: dict[str, dict[str, float]] = {}
    problems: list[str] = []
    lo_reg, hi_reg = VOLTAGE_LIMITS
    for m, c in raw[VOLTAGE].items():
        v = round(c * VOLTS_PER_COUNT, 2)
        lo = raw.get(lo_reg, {}).get(m)
        hi = raw.get(hi_reg, {}).get(m)
        row: dict[str, float] = {"present_v": v}
        if lo is not None:
            row["min_limit_v"] = round(lo * VOLTS_PER_COUNT, 2)
        if hi is not None:
            row["max_limit_v"] = round(hi * VOLTS_PER_COUNT, 2)
        motors[m] = row
        if abs(v - supply_v) > tolerance_frac * supply_v:
            msg = f"{m}: reads {v:.1f} V, the supply label says {supply_v:g} V"
            if other_supply_v is not None and abs(v - other_supply_v) < abs(v - supply_v):
                msg += (
                    f" (nearer the other arm's {other_supply_v:g} V supply: CROSSED SUPPLIES? "
                    "switch off now)"
                )
            problems.append(msg)
        if lo is not None and hi is not None and not row["min_limit_v"] <= v <= row["max_limit_v"]:
            problems.append(
                f"{m}: {v:.1f} V is outside the servo's own limits "
                f"[{row['min_limit_v']:g}, {row['max_limit_v']:g}] V (voltage alarm)"
            )
    return {
        "supply_v": supply_v,
        "tolerance_frac": tolerance_frac,
        "motors": motors,
        "problems": problems,
        "ok": not problems,
    }
