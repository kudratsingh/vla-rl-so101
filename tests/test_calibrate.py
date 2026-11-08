"""Calibration wrapper (scripts/wo06_calibrate.py): torque never enabled, always a fresh calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from src.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location(
    "wo06_calibrate", REPO_ROOT / "scripts" / "wo06_calibrate.py"
)
assert _spec is not None and _spec.loader is not None
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)


class CalBus:
    """Records the bus traffic of a calibration (lerobot 0.4.4 FeetechMotorsBus subset)."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.torque = True  # e.g. left on by an earlier session

    def connect(self) -> None:
        self.calls.append("connect")

    def disable_torque(self) -> None:
        self.calls.append("disable_torque")
        self.torque = False

    def enable_torque(self, *a: Any, **k: Any) -> None:
        raise AssertionError("calibration must never enable torque")

    def write_calibration(self, calibration: Any) -> None:
        assert not self.torque, "homing offsets written with torque on"
        self.calls.append("write_calibration")

    def disconnect(self, disable_torque: bool = True) -> None:
        self.calls.append(f"disconnect(disable_torque={disable_torque})")


class CalDevice:
    """Stands in for SO101Follower/SO101Leader: ``calibrate`` mimics LeRobot's two branches."""

    def __init__(self, path: Path, interrupt: bool = False) -> None:
        self.calibration_fpath = path
        self.calibration: dict[str, Any] = json.loads(path.read_text()) if path.is_file() else {}
        self.bus = CalBus()
        self.interrupt = interrupt
        self.branch = ""

    def connect(self, calibrate: bool = True) -> None:
        raise AssertionError("device.connect() runs configure(), which re-enables torque")

    def calibrate(self) -> None:
        assert not self.bus.torque
        if self.calibration:
            self.branch = "reuse"  # LeRobot: "Press ENTER to use provided calibration file"
            self.bus.write_calibration(self.calibration)
            return
        self.branch = "fresh"
        if self.interrupt:
            raise KeyboardInterrupt
        self.calibration = {"shoulder_pan": {"range_min": 1000, "range_max": 3000}}
        self.calibration_fpath.write_text(json.dumps(self.calibration))


def test_calibration_is_fresh_torque_off_and_backs_up(tmp_path: Path) -> None:
    path = tmp_path / "so101_follower.json"
    path.write_text(json.dumps({"shoulder_pan": {"range_min": 0, "range_max": 1}}))
    dev = CalDevice(path)
    prompts: list[str] = []
    assert cal.run_calibration(dev, ask=prompts.append) == path
    assert dev.branch == "fresh"  # the old file was never offered for reuse
    assert dev.bus.calls == ["connect", "disable_torque", "disconnect(disable_torque=True)"]
    assert "support the arm" in prompts[0]
    backups = list(tmp_path.glob("so101_follower.*.bak.json"))
    assert (
        len(backups) == 1 and json.loads(backups[0].read_text())["shoulder_pan"]["range_max"] == 1
    )
    assert json.loads(path.read_text())["shoulder_pan"]["range_max"] == 3000

    reuse = CalDevice(path)
    cal.run_calibration(reuse, reuse_file=True, ask=lambda _: "")
    assert reuse.bus.calls == [
        "connect",
        "disable_torque",
        "write_calibration",
        "disconnect(disable_torque=True)",
    ]
    with pytest.raises(SystemExit, match="no calibration file"):
        cal.run_calibration(CalDevice(tmp_path / "none.json"), reuse_file=True, ask=lambda _: "")


def test_interrupted_calibration_keeps_old_file_and_disconnects(tmp_path: Path) -> None:
    path = tmp_path / "so101_leader.json"
    path.write_text(json.dumps({"gripper": {"range_min": 5, "range_max": 6}}))
    dev = CalDevice(path, interrupt=True)
    with pytest.raises(SystemExit, match="re-run calibrate"):
        cal.run_calibration(dev, ask=lambda _: "")
    assert dev.bus.calls[-1] == "disconnect(disable_torque=True)"
    assert json.loads(path.read_text())["gripper"]["range_max"] == 6
