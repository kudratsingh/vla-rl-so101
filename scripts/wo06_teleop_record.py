#!/usr/bin/env python
"""WO-06 item 4: record real teleop episodes (5 per task, sanity only - not training data).

Leader arm (so101_leader) -> canonical radians -> src/robot/safety.py -> follower, at control_hz,
with the front camera through the policy's image pipeline. Writes one LeRobotDataset (v3.0,
lerobot 0.4.4) with the SAME schema as WO-01's sim demos: ``observation.images.front``
(crop_to x crop_to RGB video), ``observation.state`` and ``action`` (float32[6] canonical radians;
action = the safe command actually sent), per-frame ``task`` = language instruction.

Needs: a leader arm in ``teleop.leader`` (otherwise exits with a message), both arms calibrated
(scripts/wo06_calibrate.py), and the shared FK from WO-01 for the workspace check.

    uv run scripts/wo06_teleop_record.py --preset real                  # 5 episodes x 3 tasks
    uv run scripts/wo06_teleop_record.py --preset real --task Push --episodes 2
    uv run scripts/wo06_teleop_record.py --preset real --push           # also upload (private)

Per episode: arrange the scene, press ENTER, the follower first eases onto the leader's pose
(rate-limited, not recorded), then recording runs for ``teleop.episode_time_s`` or until Ctrl-C;
keep or discard it. A lost frame / stall holds the arm and asks you; a serial error cuts torque.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args, load_config
from src.robot.camera import (
    CameraFault,
    OpenCVFrameSource,
    PolicyImagePipeline,
    sim_mismatches,
)
from src.robot.driver import DeadmanTripped, So101Driver, StepResult
from src.robot.safety import DeadmanResponse, load_sim_fk
from src.robot.units import JOINT_NAMES, UnitMap

IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
SYNC_TOL_RAD = np.radians(3.0)
SYNC_TIMEOUT_S = 10.0

NO_LEADER = """No leader arm configured (teleop.leader is null in configs/real.yaml).

Teleop recording drives the follower from an SO-101 LEADER arm. If you own only a follower,
skip WO-06 item 4 (5 sanity episodes per task; nothing trains on them) and note it in
docs/STATUS.md - the rest of WO-06 (calibration, safety, camera match) does not need a leader.
With a leader: set teleop.leader to
  {type: so101_leader, id: so101_leader, port: /dev/tty.usbmodemYYYY,
   calibration_dir: ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader}
then run scripts/wo06_calibrate.py find-port/calibrate --device leader."""


def features(side: int) -> dict[str, dict[str, Any]]:
    names = list(JOINT_NAMES)
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (side, side, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (6,), "names": names},
        ACTION_KEY: {"dtype": "float32", "shape": (6,), "names": names},
    }


def make_leader(lc: dict[str, Any]) -> Any:
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    if lc.get("type", "so101_leader") != "so101_leader":
        sys.exit(f"teleop.leader.type must be so101_leader, got {lc.get('type')!r}")
    cal_dir = lc.get("calibration_dir")
    leader = SO101Leader(
        SO101LeaderConfig(
            port=str(lc["port"]),
            id=str(lc["id"]),
            calibration_dir=Path(cal_dir).expanduser() if cal_dir else None,
            use_degrees=True,
        )
    )
    if not leader.calibration:
        sys.exit("leader has no calibration file: run wo06_calibrate.py calibrate --device leader")
    return leader


class Session:
    def __init__(self, arm: So101Driver, leader: Any, cam: OpenCVFrameSource, units: UnitMap):
        self.arm, self.leader, self.cam, self.units = arm, leader, cam, units
        self.period = 1.0 / arm.safety.config.control_hz

    def tick(self) -> tuple[Any, StepResult]:
        """One control step. Returns (camera frame or None, step result)."""
        try:
            frame = self.cam.read()
        except CameraFault as e:
            print(f"camera: {e}")
            frame = None
        try:
            target = self.units.to_canonical(self.leader.get_action())
        except Exception as e:
            self.arm.deadman.trip(f"leader read failed: {e}", DeadmanResponse.HOLD)
            self.arm.hold()
            raise DeadmanTripped(self.arm.deadman.reason) from e
        out = self.arm.step(target, frame_time=None if frame is None else frame.capture_time)
        return frame, out

    def sleep_to(self, t0: float) -> None:
        time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def sync_to_leader(self) -> bool:
        """Ease the follower onto the leader's pose through the rate limiter (not recorded)."""
        t_end = time.perf_counter() + SYNC_TIMEOUT_S
        while time.perf_counter() < t_end:
            t0 = time.perf_counter()
            _, out = self.tick()
            assert out.safety.command is not None
            leader_q = self.units.to_canonical(self.leader.get_action())
            if np.all(np.abs(out.measured - leader_q) <= SYNC_TOL_RAD):
                return True
            self.sleep_to(t0)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--task", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--episodes", type=int, default=None, help="per task (default config)")
    parser.add_argument("--overwrite", action="store_true", help="replace the local dataset")
    parser.add_argument("--push", action="store_true", help="upload to the Hub (private)")
    args = parser.parse_args()
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    cfg = load_config(args.preset, args.config)
    tcfg = cfg.get("teleop") or {}
    if not tcfg.get("leader"):
        print(NO_LEADER)
        sys.exit(2)
    instructions: dict[str, str] = dict(tcfg["instructions"])
    tasks = args.task or list(instructions)
    unknown = [t for t in tasks if t not in instructions]
    if unknown:
        sys.exit(f"no instruction for {unknown} in teleop.instructions")
    n_eps = int(args.episodes or tcfg["episodes_per_task"])
    ep_time = float(tcfg["episode_time_s"])
    sim_cfg = load_config("sim")
    for m in sim_mismatches(cfg, sim_cfg):
        print(f"WARNING camera config mismatch, {m}")

    fk = load_sim_fk()  # exits with instructions if WO-01's FK is not on this branch yet
    pipe = PolicyImagePipeline.from_config(cfg)
    units = UnitMap.from_config(cfg)
    repo_id = str(tcfg["hub_repo"])
    root = REPO_ROOT / str(tcfg["dataset_root"]) / repo_id.split("/")[-1]
    if root.exists():
        if not args.overwrite:
            sys.exit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    leader = make_leader(dict(tcfg["leader"]))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(cfg["robot"]["control_hz"]),
        features=features(pipe.out_size),
        root=root,
        robot_type="so101_follower",
        use_videos=True,
        video_backend="torchcodec",
        vcodec="libsvtav1",
    )
    saved = 0
    leader.connect(calibrate=False)  # leader configure() leaves its torque off
    try:
        if not leader.is_calibrated:
            sys.exit("leader motors do not match its calibration file; recalibrate the leader")
        with (
            OpenCVFrameSource(cfg["cameras"]["front"]) as cam,
            So101Driver.from_config(cfg, fk) as arm,
        ):
            sess = Session(arm, leader, cam, units)
            for task in tasks:
                k = 0
                while k < n_eps:
                    msg = f"[{task} {k + 1}/{n_eps}] arrange the scene, hold the leader, ENTER: "
                    input(msg)
                    try:
                        if not sess.sync_to_leader():
                            print("follower did not reach the leader pose in time; retry")
                            continue
                        t_start = time.perf_counter()
                        print(f"recording ({ep_time:.0f} s max, Ctrl-C to end) ...")
                        try:
                            while time.perf_counter() - t_start < ep_time:
                                t0 = time.perf_counter()
                                frame, out = sess.tick()
                                assert frame is not None and out.safety.command is not None
                                dataset.add_frame(
                                    {
                                        IMAGE_KEY: pipe(frame.image),
                                        STATE_KEY: out.measured.astype(np.float32),
                                        ACTION_KEY: out.safety.command.astype(np.float32),
                                        "task": instructions[task],
                                    }
                                )
                                sess.sleep_to(t0)
                        except KeyboardInterrupt:
                            print("\nepisode ended")
                    except DeadmanTripped as e:
                        print(f"\nSTOPPED: {e}\nEpisode discarded.")
                        dataset.clear_episode_buffer()
                        if arm.deadman.response is not DeadmanResponse.HOLD:
                            raise  # torque is off: end the session
                        ans = input(
                            "arm is holding; type 'resume' to clear the deadman, else quit: "
                        )
                        if ans.strip() != "resume":
                            raise
                        arm.deadman.reset("operator typed resume")
                        arm.safety.reset_reference()
                        continue
                    n_frames = int(dataset.episode_buffer["size"])
                    keep = input(f"keep this episode ({n_frames} frames)? [Y/n] ").strip().lower()
                    if n_frames > 0 and keep in ("", "y", "yes"):
                        dataset.save_episode()
                        saved += 1
                        k += 1
                    else:
                        dataset.clear_episode_buffer()
    finally:
        dataset.finalize()
        if leader.is_connected:
            leader.disconnect()
        print(f"saved {saved} episodes to {root}")
    if args.push and saved:
        dataset.push_to_hub(private=True)
        print(f"pushed to https://huggingface.co/datasets/{repo_id} (private)")


if __name__ == "__main__":
    main()
