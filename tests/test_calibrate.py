"""Calibration wrapper (scripts/wo06_calibrate.py): torque never enabled, always a fresh calibration."""

from __future__ import annotations

import argparse
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


class VerifyBus(CalBus):
    is_calibrated = True

    def sync_read(self, data_name: str) -> Any:
        raise KeyboardInterrupt  # the operator stops verify at the first reading


def test_verify_asks_to_support_the_arm_before_torque_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robot_fakes import real_cfg

    path = tmp_path / "so101_follower.json"
    path.write_text(json.dumps({"shoulder_pan": {"range_min": 0, "range_max": 1}}))
    dev = CalDevice(path)
    dev.bus = VerifyBus()
    monkeypatch.setattr(cal, "_make_device", lambda args, cfg: dev)

    def ask(prompt: str) -> str:
        dev.bus.calls.append(f"ask: {prompt}")
        return ""

    cal.cmd_verify(argparse.Namespace(device="follower"), real_cfg(), ask=ask)
    assert "support the arm" in dev.bus.calls[0]
    assert dev.bus.calls[1:] == ["connect", "disable_torque", "disconnect(disable_torque=True)"]


class VoltBus:
    """A bus that answers the three voltage registers; any write or torque change fails."""

    def __init__(self, present: dict[str, int], lo: int = 40, hi: int = 140) -> None:
        self.regs = {
            "Present_Voltage": present,
            "Min_Voltage_Limit": {m: lo for m in present},
            "Max_Voltage_Limit": {m: hi for m in present},
        }
        self.calls: list[str] = []

    def connect(self) -> None:
        self.calls.append("connect")

    def sync_read(self, reg: str, motors: Any = None, normalize: bool = True) -> dict[str, int]:
        assert normalize is False and motors is None
        self.calls.append(f"read {reg}")
        return dict(self.regs[reg])

    def disable_torque(self, *a: Any, **k: Any) -> None:
        raise AssertionError("the voltage read never changes torque")

    enable_torque = disable_torque

    def disconnect(self, disable_torque: bool = True) -> None:
        assert disable_torque is False
        self.calls.append("disconnect")


def test_voltage_reads_present_voltage_read_only(capsys: pytest.CaptureFixture[str]) -> None:
    """Owner kit list 2026-11-10: the follower's 12 V STS3215 on the 12 V supply, the leader's
    7.4 V servos on the 5 V supply; step 2b reads Present_Voltage over the bus."""
    from src.config import load_config

    cfg = load_config("real")
    assert (
        cfg["power"]["follower"]["supply_v"] == 12.0 and cfg["power"]["leader"]["supply_v"] == 5.0
    )
    names = ["shoulder_pan", "gripper"]
    args = argparse.Namespace(device="follower")
    bus = VoltBus({n: 121 for n in names})
    rep = cal.cmd_voltage(args, cfg, bus=bus)
    assert rep["ok"] and rep["motors"]["gripper"] == {
        "present_v": 12.1,
        "min_limit_v": 4.0,
        "max_limit_v": 14.0,
    }
    assert bus.calls == [
        "connect",
        "read Present_Voltage",
        "read Min_Voltage_Limit",
        "read Max_Voltage_Limit",
        "disconnect",
    ]
    assert "voltage check: OK" in capsys.readouterr().out
    # the leader supply on the follower: crossed
    crossed = cal.cmd_voltage(args, cfg, bus=VoltBus({n: 50 for n in names}))
    assert not crossed["ok"] and "CROSSED SUPPLIES" in crossed["problems"][0]
    # a 12 V supply on the leader's 7.4 V servos (limits 4.0-8.0 V): off-label and over the limit
    lead = cal.cmd_voltage(
        argparse.Namespace(device="leader"), cfg, bus=VoltBus({n: 120 for n in names}, hi=80)
    )
    assert not lead["ok"] and any("outside the servo's own limits" in p for p in lead["problems"])
    assert "voltage check: FAIL" in capsys.readouterr().out
