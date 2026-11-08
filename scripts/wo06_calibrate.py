#!/usr/bin/env python
"""WO-06: find the serial port, calibrate the SO-101 follower (and leader), check and verify.

Owner, on the Mac (zsh), arm on its own supply, USB to the Mac, motor IDs already set at
assembly (``uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>``).
Full ordered procedure: docs/bringup/WO-06-checklist.md.

    uv run scripts/wo06_calibrate.py --preset real find-port --write              # unplug/replug when asked
    uv run scripts/wo06_calibrate.py --preset real calibrate --device follower    # follow the prompts
    uv run scripts/wo06_calibrate.py --preset real check --write-limits           # reads the file only
    uv run scripts/wo06_calibrate.py --preset real verify                         # torque OFF, move by hand

Leader arm only if you own one (set ``teleop.leader`` in configs/real.yaml first):

    uv run scripts/wo06_calibrate.py --preset real find-port --device leader --write
    uv run scripts/wo06_calibrate.py --preset real calibrate --device leader
    uv run scripts/wo06_calibrate.py --preset real verify --device leader         # torque OFF

Wraps lerobot 0.4.4 ``lerobot-find-port`` / ``lerobot-calibrate`` (same files) and adds the checks
that tie LeRobot's calibration to the sim's joint radians (``src/robot/units.py``). Unlike stock
``lerobot-calibrate``, ``calibrate`` never enables torque: it opens the bus only (no
``configure()``, which ends by enabling follower torque), always runs a FRESH calibration (the old
file is backed up first; LeRobot's "ENTER = use provided file" branch rewrites homing offsets with
torque on), and closes with torque off. ``--reuse-file`` instead writes the existing file to the
motors, torque off.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections.abc import Callable
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

SHORT_NAMES = ("pan", "lift", "elbow", "wflex", "wroll", "grip")  # verify columns
SUPPORT_PROMPT = (
    "Torque goes OFF now (if it was on, the arm drops): support the arm, then press ENTER "
)

FOLLOWER_STEPS = """
WHAT YOU DO PHYSICALLY (follower):
  1. Arm on the table, clamped, power on, USB connected. Keep the workspace clear. Torque stays
     OFF for the whole command (the arm is limp): support it whenever it is off the table.
  2. (No "use provided calibration file" question: this wrapper always calibrates afresh.)
  3. "Move ... to the middle of its range of motion": put EVERY joint at the middle of its travel
     (LeRobot SO-101 docs picture: upper arm and forearm at mid travel, wrist straight, gripper
     half open). Hold wrist_roll centred: this pose BECOMES wrist_roll's zero. Press ENTER.
  4. Move each joint EXCEPT wrist_roll slowly through its ENTIRE range, touching BOTH mechanical
     end stops; close the gripper fully and open it fully. A short sweep shifts the joint's zero:
     LeRobot's 0 deg is the middle of the recorded range, the sim's 0 rad the middle of the URDF's.
  5. Press ENTER. The file lands in robot.calibration_dir/<robot.id>.json. Then `check` runs.
"""

LEADER_STEPS = """
WHAT YOU DO PHYSICALLY (leader): same as the follower - middle pose, ENTER, sweep every joint
except wrist_roll end stop to end stop, trigger fully closed and open, ENTER. The leader never
has torque.
"""

VERIFY_STEPS = """
VERIFY (torque is OFF: support the arm, it will sag). Readings are canonical (sim) degrees.
  1. Calibration middle pose -> every arm joint within ~5 deg of 0.
  2. Lift the gripper straight up       -> TCP z increases.
     Move it away from the base          -> TCP x increases.
     Swing it to the robot's left        -> TCP y increases (base frame, +y = robot's left).
  3. Gripper fully closed -> ~ -10 deg; fully open -> ~ +100 deg.
  4. scripts/wo06_camera_match.py --with-arm renders the sim arm at these angles next to the
     camera image; the two arms must overlap.
  A joint moving the wrong way: flip its sign in robot.units.arm_sign. A constant offset: put it
  in robot.units.arm_offset_deg (degrees). Ctrl-C to stop.
"""

LEADER_VERIFY_STEPS = """
VERIFY LEADER (torque is OFF; the leader never has torque). Readings use the follower's
robot.units, exactly as teleop does. Put the leader in the same pose as the follower (by eye) and
compare with `verify` on the follower: every joint should agree within ~5 deg; trigger closed
~ -10 deg, open ~ +100 deg. robot.units is shared by both arms, so a sign or offset that differs
between them cannot be fixed there: recalibrate the arm that disagrees. Ctrl-C to stop.
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


def _make_device(args: argparse.Namespace, cfg: dict[str, Any]) -> Any:
    """LeRobot ``SO101Follower`` / ``SO101Leader`` for ``args.device``, not connected."""
    if args.device == "follower":
        from src.robot.driver import follower_from_config, install_hold_on_torque_enable

        device = follower_from_config(cfg)
        install_hold_on_torque_enable(device.bus)  # belt and braces: nothing here enables torque
        return device
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    lc = _leader_cfg(cfg)
    if "XXXX" in str(lc.get("port")) or "YYYY" in str(lc.get("port")):
        sys.exit("teleop.leader.port is a placeholder; run `find-port --device leader --write`")
    cal_dir = lc.get("calibration_dir")
    return SO101Leader(
        SO101LeaderConfig(
            port=str(lc["port"]),
            id=str(lc["id"]),
            calibration_dir=Path(cal_dir).expanduser() if cal_dir else None,
            use_degrees=True,
        )
    )


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


def run_calibration(
    device: Any, *, reuse_file: bool = False, ask: Callable[[str], str] = input
) -> Path:
    """Torque-off LeRobot calibration of ``device`` (follower or leader); returns the file.

    Bus-level connect only (``SOFollower.connect`` runs ``configure()``, whose
    ``torque_disabled()`` block re-enables torque on exit), torque off, then LeRobot's own
    ``calibrate()`` with its cached calibration cleared, so it always takes the fresh path (middle
    pose + sweep) and never offers to write the old file to the motors. The old file is copied to
    ``<id>.<time>.bak.json`` first; LeRobot replaces ``<id>.json`` only after the sweep, so an
    interrupted run leaves the old file and motors that no longer match it (the driver then refuses
    to connect). ``reuse_file``: write the existing file to the motors instead, also torque off.
    Always disconnects with torque off.
    """
    path = Path(device.calibration_fpath)
    if reuse_file and not device.calibration:
        sys.exit(f"--reuse-file: no calibration file at {path}")
    ask(SUPPORT_PROMPT)
    bus = device.bus
    bus.connect()  # opens the port and pings the motors; no configure(), no torque enable
    try:
        bus.disable_torque()
        if reuse_file:
            bus.write_calibration(device.calibration)
            print(f"wrote {path} to the motors (torque off)")
        else:
            if device.calibration:
                backup = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.json")
                shutil.copy2(path, backup)
                print(f"backed up the previous calibration to {backup}")
                device.calibration = {}  # LeRobot then skips "ENTER = use provided file"
            device.calibrate()
    except KeyboardInterrupt:
        sys.exit(
            f"\ncalibration interrupted; {path} was not replaced but the motors may no longer "
            "match it: re-run calibrate"
        )
    finally:
        bus.disconnect(disable_torque=True)
    return path


def cmd_calibrate(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    print(LEADER_STEPS if args.device == "leader" else FOLLOWER_STEPS)
    path = run_calibration(_make_device(args, cfg), reuse_file=args.reuse_file)
    print(f"saved {path}")
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


def cmd_verify(
    args: argparse.Namespace, cfg: dict[str, Any], ask: Callable[[str], str] = input
) -> None:
    units = UnitMap.from_config(cfg)
    sc = SafetyConfig.from_config(cfg)
    try:
        fk = load_sim_fk()
    except RuntimeError as e:
        fk = None
        print(f"(no TCP readout: {e})")
    device = _make_device(args, cfg)
    if not device.calibration:
        sys.exit(f"no calibration file; run `calibrate --device {args.device}` first")
    bus = device.bus
    print(LEADER_VERIFY_STEPS if args.device == "leader" else VERIFY_STEPS)
    if args.device == "follower":
        ask(SUPPORT_PROMPT)  # torque can be on after a crashed session (the leader never has any)
    bus.connect()  # port only: no configure(), no torque enable
    try:
        bus.disable_torque()
        if not bus.is_calibrated:
            sys.exit("motors do not match the calibration file; re-run calibrate")
        while True:
            raw = bus.sync_read("Present_Position")
            q = units.to_canonical({f"{m}.pos": v for m, v in raw.items()})
            line = "  ".join(f"{n}={math.degrees(v):7.1f}" for n, v in zip(SHORT_NAMES, q))
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
    p = sub.add_parser("calibrate", help="fresh LeRobot calibration, torque off (physical)")
    p.add_argument("--device", choices=["follower", "leader"], default="follower")
    p.add_argument("--write-limits", action="store_true", help="after a follower calibration")
    p.add_argument(
        "--reuse-file",
        action="store_true",
        help="write the existing calibration file to the motors (torque off) instead of "
        "calibrating afresh; only to restore motors that lost a known-good calibration",
    )
    p = sub.add_parser("check", help="compare the calibration with the URDF; derive limits")
    p.add_argument("--write-limits", action="store_true")
    p = sub.add_parser("verify", help="stream canonical joint angles + TCP with torque off")
    p.add_argument("--device", choices=["follower", "leader"], default="follower")
    args = parser.parse_args()
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    cfg = load_config(args.preset, args.config)
    {"find-port": cmd_find_port, "calibrate": cmd_calibrate, "check": cmd_check}.get(
        args.cmd, cmd_verify
    )(args, cfg)


if __name__ == "__main__":
    main()
