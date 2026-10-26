#!/usr/bin/env python
"""WO-06: find the serial port, calibrate the SO-101 follower (and leader), check and verify.

Owner, on the Mac (zsh), arm on its 5 V 4 A supply, USB to the Mac, motor IDs already set at
assembly (``uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>``):

    uv run scripts/wo06_calibrate.py --preset real find-port --write              # unplug/replug when asked
    uv run scripts/wo06_calibrate.py --preset real calibrate --device follower    # follow the prompts
    uv run scripts/wo06_calibrate.py --preset real check --write-limits           # reads the file only
    uv run scripts/wo06_calibrate.py --preset real verify                         # torque OFF, move by hand

Leader arm only if you own one (set ``teleop.leader`` in configs/real.yaml first):

    uv run scripts/wo06_calibrate.py --preset real find-port --device leader --write
    uv run scripts/wo06_calibrate.py --preset real calibrate --device leader

Wraps lerobot 0.4.4 ``lerobot-find-port`` / ``lerobot-calibrate`` (same calls, same files) and adds
the checks that tie LeRobot's calibration to the sim's joint radians (``src/robot/units.py``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG_DIR, add_config_args, load_config
from src.robot.safety import (
    DEFAULT_URDF_MARGIN_DEG,
    SafetyConfig,
    load_sim_fk,
    resolve_joint_limits_deg,
)
from src.robot.units import (
    ARM_JOINTS,
    JOINT_NAMES,
    UnitMap,
    recorded_span_deg,
    urdf_limits_deg,
)
from src.robot.yamlio import update_yaml_file

FOLLOWER_STEPS = """
WHAT YOU DO PHYSICALLY (follower):
  1. Arm on the table, clamped, power on, USB connected. Keep the workspace clear.
  2. If asked "Press ENTER to use provided calibration file ... or type 'c'": type c, ENTER.
  3. "Move ... to the middle of its range of motion": put EVERY joint at the middle of its travel
     (LeRobot SO-101 docs picture: upper arm and forearm at mid travel, wrist straight, gripper
     half open). Hold wrist_roll centred: this pose BECOMES wrist_roll's zero. Press ENTER.
  4. Move each joint EXCEPT wrist_roll slowly through its ENTIRE range, touching BOTH mechanical
     end stops; close the gripper fully and open it fully. A short sweep shifts the joint's zero:
     LeRobot's 0 deg is the middle of the recorded range, the sim's 0 rad the middle of the URDF's.
  5. Press ENTER. The file lands in robot.calibration_dir/<robot.id>.json. Then run `check`.
"""

LEADER_STEPS = """
WHAT YOU DO PHYSICALLY (leader): same as the follower - middle pose, ENTER, sweep every joint
except wrist_roll end stop to end stop, trigger fully closed and open, ENTER.
"""

VERIFY_STEPS = """
VERIFY (torque is OFF: support the arm, it will sag). Readings are canonical (sim) degrees.
  1. Calibration middle pose -> every arm joint within ~5 deg of 0.
  2. Lift the gripper straight up       -> TCP z increases.
     Move it away from the base          -> TCP x increases.
     Swing it to the robot's left        -> TCP y increases (base frame, +y = robot's left).
  3. Gripper fully closed -> ~ -10 deg; fully open -> ~ +100 deg.
  4. With WO-01 merged: scripts/wo06_camera_match.py --with-arm renders the sim arm at these
     angles next to the camera image; the two arms must overlap.
  A joint moving the wrong way: flip its sign in robot.units.arm_sign. A constant offset: put it
  in robot.units.arm_offset_deg (degrees). Ctrl-C to stop.
"""


def _cfg_path(args: argparse.Namespace) -> Path:
    return Path(args.config) if args.config else CONFIG_DIR / f"{args.preset}.yaml"


def _calibration_file(dev_cfg: dict[str, Any], kind: str) -> Path:
    default = f"~/.cache/huggingface/lerobot/calibration/{kind}"
    return Path(dev_cfg.get("calibration_dir") or default).expanduser() / f"{dev_cfg['id']}.json"


def _leader_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    leader = (cfg.get("teleop") or {}).get("leader")
    if not leader:
        sys.exit(
            "No leader arm configured (teleop.leader is null in configs/real.yaml). Set it to\n"
            "  {type: so101_leader, id: so101_leader, port: /dev/tty.usbmodemYYYY,\n"
            "   calibration_dir: ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader}\n"
            "if you own a leader arm."
        )
    return dict(leader)


# ------------------------------------------------------------------------------------------------
def cmd_find_port(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from lerobot.scripts.lerobot_find_port import (
        find_available_ports,
    )  # lists /dev/tty*, opens none

    if args.device == "leader":
        _leader_cfg(cfg)
    before = set(find_available_ports())
    input(f"Unplug the USB cable of the {args.device} board, then press ENTER ")
    time.sleep(0.5)
    gone = sorted(before - set(find_available_ports()))
    if len(gone) != 1:
        sys.exit(f"Expected exactly one port to disappear, got {gone}. Try again.")
    port = gone[0]
    print(f"The {args.device} board is on {port}. Plug the USB cable back in.")
    keys = ["robot", "port"] if args.device == "follower" else ["teleop", "leader"]
    if args.write:
        if args.device == "follower":
            update_yaml_file(_cfg_path(args), keys, port)
        else:
            leader = _leader_cfg(cfg)
            leader["port"] = port
            update_yaml_file(_cfg_path(args), keys, leader)
        print(f"wrote {'.'.join(keys)} in {_cfg_path(args)}")
    else:
        print("re-run with --write to store it in the config")


def cmd_calibrate(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if args.device == "follower":
        from src.robot.driver import follower_from_config, install_hold_on_torque_enable

        print(FOLLOWER_STEPS)
        device = follower_from_config(cfg)
        install_hold_on_torque_enable(device.bus)  # LeRobot connect() enables torque
    else:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        print(LEADER_STEPS)
        lc = _leader_cfg(cfg)
        cal_dir = lc.get("calibration_dir")
        device = SO101Leader(
            SO101LeaderConfig(
                port=str(lc["port"]),
                id=str(lc["id"]),
                calibration_dir=Path(cal_dir).expanduser() if cal_dir else None,
                use_degrees=True,
            )
        )
    device.connect(calibrate=False)  # exactly what lerobot-calibrate does
    try:
        device.calibrate()
    finally:
        device.disconnect()
    print(f"saved {device.calibration_fpath}")
    if args.device == "follower":
        cmd_check(args, cfg)


def cmd_check(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    path = _calibration_file(cfg["robot"], "robots/so_follower")
    if not path.is_file():
        sys.exit(f"no calibration file at {path}; run `calibrate --device follower` first")
    calib = json.loads(path.read_text())
    units = UnitMap.from_config(cfg)
    spans = recorded_span_deg(calib)
    urdf = urdf_limits_deg()
    from_calib = units.calibrated_limits_deg(calib)
    margin = float(cfg["robot"]["safety"].get("urdf_limit_margin_deg", DEFAULT_URDF_MARGIN_DEG))
    print(f"calibration: {path}")
    print(f"{'joint':<14}{'recorded':>10}{'URDF':>8}   canonical range from calibration (deg)")
    warnings = []
    for name in JOINT_NAMES:
        u_span = urdf[name][1] - urdf[name][0]
        lo, hi = from_calib[name]
        print(f"{name:<14}{spans[name]:>9.1f}°{u_span:>7.1f}°   [{lo:7.1f}, {hi:7.1f}]")
        tol = 15.0 if name == "gripper" else 10.0
        if name != "wrist_roll" and abs(spans[name] - u_span) > tol:
            warnings.append(
                f"{name}: recorded span {spans[name]:.0f} deg vs URDF {u_span:.0f} deg - "
                + (
                    "the gripper map (0-100 -> URDF range) will be off"
                    if name == "gripper"
                    else "joint zero may differ from the sim's; re-calibrate with a full sweep"
                )
            )
    for w in warnings:
        print("WARNING", w)
    # arm joints: the recorded mechanical range shrunk by the margin (it only ever narrows the
    # URDF-minus-margin limits that safety.py applies anyway). The gripper keeps the URDF limits.
    limits = {
        n: [round(from_calib[n][0] + margin, 1), round(from_calib[n][1] - margin, 1)]
        for n in ARM_JOINTS
    }
    safety_cfg = dict(cfg["robot"]["safety"], joint_limits_deg=limits)
    effective = resolve_joint_limits_deg(safety_cfg)
    print("effective safety limits (deg):")
    for n in JOINT_NAMES:
        print(f"  {n:<14}[{effective[n][0]:7.1f}, {effective[n][1]:7.1f}]")
    if args.write_limits:
        update_yaml_file(_cfg_path(args), ["robot", "safety", "joint_limits_deg"], limits)
        print(f"wrote robot.safety.joint_limits_deg in {_cfg_path(args)}")
    else:
        print("re-run with --write-limits to store them in the config")


def cmd_verify(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from src.robot.driver import follower_from_config

    units = UnitMap.from_config(cfg)
    sc = SafetyConfig.from_config(cfg)
    try:
        fk = load_sim_fk()
    except RuntimeError as e:
        fk = None
        print(f"(no TCP readout: {e})")
    follower = follower_from_config(cfg)
    if not follower.calibration:
        sys.exit("no calibration file; run `calibrate --device follower` first")
    bus = follower.bus
    print(VERIFY_STEPS)
    bus.connect()  # port only: no configure(), no torque enable
    try:
        bus.disable_torque()
        if not bus.is_calibrated:
            sys.exit("motors do not match the calibration file; re-run calibrate")
        while True:
            raw = bus.sync_read("Present_Position")
            q = units.to_canonical({f"{m}.pos": v for m, v in raw.items()})
            line = "  ".join(f"{n[:8]}={math.degrees(v):7.1f}" for n, v in zip(JOINT_NAMES, q))
            if fk is not None:
                p = [float(v) for v in fk(q)]
                inside = all(
                    lo <= v <= hi
                    for v, lo, hi in zip(p, sc.workspace_lo_m, sc.workspace_hi_m, strict=True)
                )
                line += f"  TCP=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) {'in' if inside else 'OUT'}"
            print(line, flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        bus.disconnect(disable_torque=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(parser)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find-port", help="which /dev/tty.* is the arm (unplug/replug)")
    p.add_argument("--device", choices=["follower", "leader"], default="follower")
    p.add_argument("--write", action="store_true", help="store the port in the config")
    p = sub.add_parser("calibrate", help="LeRobot calibration (interactive, physical)")
    p.add_argument("--device", choices=["follower", "leader"], default="follower")
    p.add_argument("--write-limits", action="store_true", help="after a follower calibration")
    p = sub.add_parser("check", help="compare the calibration with the URDF; derive limits")
    p.add_argument("--write-limits", action="store_true")
    sub.add_parser("verify", help="stream canonical joint angles + TCP with torque off")
    args = parser.parse_args()
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    cfg = load_config(args.preset, args.config)
    {"find-port": cmd_find_port, "calibrate": cmd_calibrate, "check": cmd_check}.get(
        args.cmd, cmd_verify
    )(args, cfg)


if __name__ == "__main__":
    main()
