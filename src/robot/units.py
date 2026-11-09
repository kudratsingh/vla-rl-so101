"""LeRobot 0.4.4 ``so101_follower`` units <-> the canonical joint vector (radians). Explicit on purpose.

Canonical vector (policy action and observed state everywhere in this repo, sim and real)::

    float32[6] = [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

absolute joint positions in RADIANS of the sim robot model, TheRobotStudio/SO-ARM100
``Simulation/SO101/so101_new_calib.urdf`` (Apache-2.0) @ ``URDF_SHA``; the gripper value is that
URDF's ``gripper`` joint angle (lower limit = closed, larger = more open).

LeRobot side (``SO101Follower`` / ``SO101Leader`` with ``use_degrees=True``, the 0.4.4 default):
a dict ``{"<motor>.pos": float}`` in, and out of, ``get_observation`` / ``send_action``.
From ``lerobot/motors/motors_bus.py`` (``_normalize`` / ``_unnormalize``):

* arm motors, ``MotorNormMode.DEGREES``: ``deg = (raw - mid) * 360 / 4095`` with
  ``mid = (range_min + range_max) / 2`` from the calibration file. So LeRobot's 0 deg is the
  middle of the range of motion recorded at calibration (``wrist_roll`` has a fixed 0..4095 range,
  so its 0 is the pose held at the "middle of range" calibration step). Not clipped by LeRobot.
* gripper, ``MotorNormMode.RANGE_0_100``: ``pct = (raw - range_min) / (range_max - range_min) * 100``,
  0 = fully closed, 100 = fully open (SO-ARM100 ``Simulation/SO101/README.md``, "Gripper Note").
  LeRobot clips commands to 0..100.

The URDF's "new calibration" convention: "Each joint's virtual zero is set to the middle of its
joint range" (SO-ARM100 ``Simulation/SO101/README.md``) - the same zero LeRobot's degrees mode uses.
ManiSkill 3.0.1's ``LeRobotRealAgent`` (``mani_skill/agents/robots/lerobot/manipulator.py``) maps
LeRobot degrees to its SO-100 model with a plain ``deg2rad`` (plus an SO-100-specific elbow offset).

Mapping used here (all parameters in ``configs/real.yaml`` ``robot.units``)::

    arm joint j:  q_rad = deg2rad(sign_j * lerobot_deg_j + offset_deg_j)      default sign +1, offset 0
    gripper:      q_rad = closed_rad + pct / 100 * (open_rad - closed_rad)     URDF limits -0.174533..1.74533

Assumptions to verify on the real arm (``scripts/wo06_calibrate.py check`` and ``verify``):
signs/offsets (+1/0 is the documented convention, not yet measured), and that the calibration sweep
reached both mechanical stops of every joint so LeRobot's range middle equals the URDF's.
LeRobot's 360/4095 (not /4096) degrees-per-tick is a 0.02 % scale error; ignored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from src.config import Config

FloatArray = npt.NDArray[np.float64]

JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_JOINTS: tuple[str, ...] = JOINT_NAMES[:5]
NUM_JOINTS = len(JOINT_NAMES)
GRIPPER_INDEX = 5

URDF_SHA = "a758567c3978dfeefe282ede0500085a48fe8f78"
URDF_SOURCE = f"TheRobotStudio/SO-ARM100@{URDF_SHA}:Simulation/SO101/so101_new_calib.urdf"
# <limit lower upper> of each revolute joint in URDF_SOURCE, radians.
URDF_LIMITS_RAD: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}

# STS3215: 4096 ticks per turn; LeRobot normalizes with max_res = 4096 - 1.
LEROBOT_DEG_PER_TICK = 360.0 / 4095.0


def _keys(names: Sequence[str]) -> list[str]:
    return [f"{n}.pos" for n in names]


@dataclass(frozen=True)
class UnitMap:
    """Explicit LeRobot <-> canonical conversion. Construct with ``from_config``."""

    arm_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    arm_offset_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    gripper_closed_rad: float = URDF_LIMITS_RAD["gripper"][0]
    gripper_open_rad: float = URDF_LIMITS_RAD["gripper"][1]

    def __post_init__(self) -> None:
        if len(self.arm_sign) != 5 or any(s not in (1.0, -1.0) for s in self.arm_sign):
            raise ValueError(f"arm_sign must be 5 values of +1/-1, got {self.arm_sign}")
        if len(self.arm_offset_deg) != 5 or not all(math.isfinite(o) for o in self.arm_offset_deg):
            raise ValueError(f"arm_offset_deg must be 5 finite values, got {self.arm_offset_deg}")
        if not (math.isfinite(self.gripper_closed_rad) and math.isfinite(self.gripper_open_rad)):
            raise ValueError("gripper_closed_rad / gripper_open_rad must be finite")
        if self.gripper_closed_rad == self.gripper_open_rad:
            raise ValueError("gripper_closed_rad must differ from gripper_open_rad")

    @classmethod
    def from_config(cls, cfg: Config) -> UnitMap:
        u: dict[str, Any] = dict(cfg.get("robot", {}).get("units") or {})
        default = cls()
        return cls(
            arm_sign=tuple(float(s) for s in u.get("arm_sign", default.arm_sign)),
            arm_offset_deg=tuple(float(o) for o in u.get("arm_offset_deg", default.arm_offset_deg)),
            gripper_closed_rad=float(u.get("gripper_closed_rad", default.gripper_closed_rad)),
            gripper_open_rad=float(u.get("gripper_open_rad", default.gripper_open_rad)),
        )

    # -- gripper ------------------------------------------------------------------------------
    def gripper_pct_to_rad(self, pct: float) -> float:
        span = self.gripper_open_rad - self.gripper_closed_rad
        return self.gripper_closed_rad + float(pct) / 100.0 * span

    def gripper_rad_to_pct(self, rad: float) -> float:
        span = self.gripper_open_rad - self.gripper_closed_rad
        return (float(rad) - self.gripper_closed_rad) / span * 100.0

    # -- arm ------------------------------------------------------------------------------------
    def arm_lerobot_deg_to_rad(self, index: int, deg: float) -> float:
        return math.radians(self.arm_sign[index] * float(deg) + self.arm_offset_deg[index])

    def arm_rad_to_lerobot_deg(self, index: int, rad: float) -> float:
        # sign is +/-1, so dividing by it == multiplying by it
        return (math.degrees(float(rad)) - self.arm_offset_deg[index]) * self.arm_sign[index]

    # -- full vectors ---------------------------------------------------------------------------
    def to_canonical(self, lerobot: Mapping[str, Any]) -> FloatArray:
        """LeRobot observation/action dict -> canonical float64[6] radians (extra keys ignored)."""
        missing = [k for k in _keys(JOINT_NAMES) if k not in lerobot]
        if missing:
            raise KeyError(f"LeRobot dict lacks {missing}; got keys {sorted(lerobot)}")
        q = np.empty(NUM_JOINTS, dtype=np.float64)
        for i, name in enumerate(ARM_JOINTS):
            q[i] = self.arm_lerobot_deg_to_rad(i, float(lerobot[f"{name}.pos"]))
        q[GRIPPER_INDEX] = self.gripper_pct_to_rad(float(lerobot["gripper.pos"]))
        if not np.all(np.isfinite(q)):
            raise ValueError(f"non-finite joint reading {dict(lerobot)}")
        return q

    def to_lerobot(self, q: npt.ArrayLike) -> dict[str, float]:
        """Canonical radians[6] -> LeRobot action dict (degrees for the arm, 0-100 gripper)."""
        v = np.asarray(q, dtype=np.float64).reshape(-1)
        if v.shape != (NUM_JOINTS,):
            raise ValueError(f"expected {NUM_JOINTS} joint values, got shape {np.shape(q)}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"non-finite joint command {v}")
        out = {
            f"{name}.pos": self.arm_rad_to_lerobot_deg(i, v[i]) for i, name in enumerate(ARM_JOINTS)
        }
        out["gripper.pos"] = self.gripper_rad_to_pct(v[GRIPPER_INDEX])
        return out

    # -- calibration-derived ranges --------------------------------------------------------------
    def calibrated_limits_deg(
        self, calibration: Mapping[str, Any]
    ) -> dict[str, tuple[float, float]]:
        """Canonical joint range (degrees) implied by a LeRobot calibration file.

        ``calibration`` is the parsed ``<id>.json`` ({motor: {range_min, range_max, ...}}). Arm
        joints: LeRobot degrees span ``(range - mid) * 360/4095`` = +/- half the recorded span.
        Gripper: 0..100 maps to closed..open by construction.
        """
        out: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(ARM_JOINTS):
            c = calibration[name]
            half = (float(c["range_max"]) - float(c["range_min"])) / 2.0 * LEROBOT_DEG_PER_TICK
            a = math.degrees(self.arm_lerobot_deg_to_rad(i, -half))
            b = math.degrees(self.arm_lerobot_deg_to_rad(i, half))
            out[name] = (min(a, b), max(a, b))
        g = sorted((math.degrees(self.gripper_closed_rad), math.degrees(self.gripper_open_rad)))
        out["gripper"] = (g[0], g[1])
        return out


def _cal(c: Any, key: str) -> int:
    """A calibration field from a lerobot ``MotorCalibration`` or the parsed ``<id>.json`` dict."""
    return int(c[key] if isinstance(c, Mapping) else getattr(c, key))


def raw_to_lerobot(name: str, raw: float, calibration: Any) -> float:
    """One ``Present_Position`` raw tick value -> LeRobot units, as lerobot 0.4.4's
    ``MotorsBus._normalize`` (``lerobot/motors/motors_bus.py:838-865``) with its SO-101 norm modes
    (arm DEGREES, gripper RANGE_0_100, drive mode applied to the gripper), EXCEPT that the gripper
    is NOT clamped to [range_min, range_max]: lerobot clamps it, which hides a gripper resting
    outside its calibrated range. ``calibration``: that motor's MotorCalibration (or json dict)."""
    lo, hi = _cal(calibration, "range_min"), _cal(calibration, "range_max")
    if hi == lo:
        raise ValueError(f"invalid calibration for {name}: range_min == range_max")
    if name == "gripper":
        pct = (float(raw) - lo) / (hi - lo) * 100.0
        return 100.0 - pct if _cal(calibration, "drive_mode") else pct
    return (float(raw) - (lo + hi) / 2.0) * LEROBOT_DEG_PER_TICK


def raw_to_canonical(
    units: UnitMap, raw: Mapping[str, float], calibration: Mapping[str, Any]
) -> FloatArray:
    """Raw ``Present_Position`` ticks of all six motors -> canonical radians (unclamped)."""
    return units.to_canonical(
        {f"{n}.pos": raw_to_lerobot(n, raw[n], calibration[n]) for n in JOINT_NAMES}
    )


def urdf_limits_deg() -> dict[str, tuple[float, float]]:
    return {k: (math.degrees(lo), math.degrees(hi)) for k, (lo, hi) in URDF_LIMITS_RAD.items()}


def recorded_span_deg(calibration: Mapping[str, Any]) -> dict[str, float]:
    """Recorded range of motion per motor in degrees (gripper too), for the bring-up check."""
    return {
        name: (float(calibration[name]["range_max"]) - float(calibration[name]["range_min"]))
        * LEROBOT_DEG_PER_TICK
        for name in JOINT_NAMES
    }
