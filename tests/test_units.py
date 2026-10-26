"""LeRobot <-> canonical radian conversions, grounded in lerobot 0.4.4's own normalization code."""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_fakes import real_cfg

from src.robot.units import (
    JOINT_NAMES,
    LEROBOT_DEG_PER_TICK,
    URDF_LIMITS_RAD,
    UnitMap,
    recorded_span_deg,
)

LO = np.array([URDF_LIMITS_RAD[n][0] for n in JOINT_NAMES])
HI = np.array([URDF_LIMITS_RAD[n][1] for n in JOINT_NAMES])


def _maps() -> list[UnitMap]:
    return [
        UnitMap(),
        UnitMap(arm_sign=(1, -1, 1, -1, 1), arm_offset_deg=(0, 6.8, -3.0, 0, 90.0)),
        UnitMap(gripper_closed_rad=0.2, gripper_open_rad=-0.5),
    ]


@pytest.mark.parametrize("um", _maps())
def test_round_trip_both_ways(um: UnitMap) -> None:
    rng = np.random.default_rng(0)
    for q in rng.uniform(LO, HI, size=(50, 6)):
        back = um.to_canonical(um.to_lerobot(q))
        np.testing.assert_allclose(back, q, atol=1e-12)
    for _ in range(50):
        d = {f"{n}.pos": float(rng.uniform(-120, 120)) for n in JOINT_NAMES[:5]}
        d["gripper.pos"] = float(rng.uniform(0, 100))
        again = um.to_lerobot(um.to_canonical(d))
        assert again.keys() == d.keys()
        for k, v in d.items():
            assert again[k] == pytest.approx(v, abs=1e-9)


def test_default_mapping_values() -> None:
    um = UnitMap.from_config(real_cfg())
    assert um == UnitMap()  # real.yaml ships the documented defaults
    d = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
    d.update({"shoulder_lift.pos": 90.0, "elbow_flex.pos": -45.0, "gripper.pos": 0.0})
    q = um.to_canonical(d)
    np.testing.assert_allclose(q[:5], [0.0, math.pi / 2, -math.pi / 4, 0.0, 0.0])
    assert q[5] == pytest.approx(URDF_LIMITS_RAD["gripper"][0])  # LeRobot 0 = closed = URDF lower
    assert um.gripper_pct_to_rad(100.0) == pytest.approx(URDF_LIMITS_RAD["gripper"][1])
    assert um.gripper_rad_to_pct(sum(URDF_LIMITS_RAD["gripper"]) / 2) == pytest.approx(50.0)
    assert q.dtype == np.float64 and q.shape == (6,)


def test_bad_inputs() -> None:
    um = UnitMap()
    with pytest.raises(KeyError):
        um.to_canonical({"shoulder_pan.pos": 0.0})
    d = {f"{n}.pos": 0.0 for n in JOINT_NAMES}
    d["wrist_flex.pos"] = float("nan")
    with pytest.raises(ValueError):
        um.to_canonical(d)
    with pytest.raises(ValueError):
        um.to_lerobot(np.zeros(5))
    with pytest.raises(ValueError):
        um.to_lerobot(np.array([0, 0, 0, np.inf, 0, 0]))
    with pytest.raises(ValueError):
        UnitMap(arm_sign=(1, 1, 1, 1, 2))


def _lerobot_bus(calibration: dict[str, tuple[int, int]]) -> object:
    """lerobot 0.4.4's own FeetechMotorsBus, exactly as SOFollower(use_degrees=True) builds it.
    Constructing it does not open the port (scservo_sdk PortHandler only stores the name)."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {
        n: Motor(
            i + 1, "sts3215", MotorNormMode.DEGREES if n != "gripper" else MotorNormMode.RANGE_0_100
        )
        for i, n in enumerate(JOINT_NAMES)
    }
    cal = {
        n: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=lo, range_max=hi)
        for i, (n, (lo, hi)) in enumerate(calibration.items())
    }
    return FeetechMotorsBus(port="/dev/null-never-opened", motors=motors, calibration=cal)


def test_mapping_matches_lerobot_normalization() -> None:
    """raw ticks -> lerobot 0.4.4 _normalize -> our canonical radians, and back via _unnormalize."""
    calib = {
        "shoulder_pan": (800, 3300),
        "shoulder_lift": (900, 3200),
        "elbow_flex": (950, 3150),
        "wrist_flex": (870, 3100),
        "wrist_roll": (0, 4095),
        "gripper": (2000, 3250),
    }
    bus = _lerobot_bus(calib)
    um = UnitMap()
    ids = {n: i + 1 for i, n in enumerate(JOINT_NAMES)}

    def canonical_from_raw(raw: dict[str, int]) -> np.ndarray:
        norm = bus._normalize({ids[n]: v for n, v in raw.items()})  # type: ignore[attr-defined]
        return um.to_canonical({f"{n}.pos": norm[ids[n]] for n in JOINT_NAMES})

    mid_raw = {n: (lo + hi) // 2 for n, (lo, hi) in calib.items()}
    mid_raw["gripper"] = calib["gripper"][0]  # gripper fully closed
    q = canonical_from_raw(mid_raw)
    # middle of the recorded range == URDF zero (new_calib convention); 1 tick = 0.088 deg
    np.testing.assert_allclose(q[:5], 0.0, atol=math.radians(LEROBOT_DEG_PER_TICK))
    assert q[5] == pytest.approx(URDF_LIMITS_RAD["gripper"][0])

    top = {n: hi for n, (lo, hi) in calib.items()}
    q_top = canonical_from_raw(top)
    span = recorded_span_deg(
        {n: {"range_min": lo, "range_max": hi} for n, (lo, hi) in calib.items()}
    )
    for i, n in enumerate(JOINT_NAMES[:5]):
        assert math.degrees(q_top[i]) == pytest.approx(span[n] / 2, abs=1e-9)
    assert q_top[5] == pytest.approx(URDF_LIMITS_RAD["gripper"][1])  # LeRobot 100 = open

    # canonical -> LeRobot dict -> lerobot _unnormalize -> raw ticks (int truncation: <= 1 tick)
    q_cmd = np.array([0.3, -0.5, 0.7, -0.2, 1.0, 0.4])
    lr = um.to_lerobot(q_cmd)
    raw = bus._unnormalize({ids[n]: lr[f"{n}.pos"] for n in JOINT_NAMES})  # type: ignore[attr-defined]
    q_back = canonical_from_raw({n: raw[ids[n]] for n in JOINT_NAMES})
    np.testing.assert_allclose(q_back[:5], q_cmd[:5], atol=math.radians(2 * LEROBOT_DEG_PER_TICK))
    assert q_back[5] == pytest.approx(q_cmd[5], abs=2 * (HI[5] - LO[5]) / (3250 - 2000))


def test_calibrated_limits() -> None:
    um = UnitMap(arm_sign=(1, -1, 1, 1, 1), arm_offset_deg=(0, 10.0, 0, 0, 0))
    calib = {n: {"range_min": 1000, "range_max": 3000} for n in JOINT_NAMES}
    lim = um.calibrated_limits_deg(calib)
    half = 1000 * LEROBOT_DEG_PER_TICK
    assert lim["shoulder_pan"] == pytest.approx((-half, half))
    assert lim["shoulder_lift"] == pytest.approx((-half + 10.0, half + 10.0))  # sign flip, offset
    assert lim["gripper"] == pytest.approx(
        tuple(math.degrees(v) for v in URDF_LIMITS_RAD["gripper"])
    )
