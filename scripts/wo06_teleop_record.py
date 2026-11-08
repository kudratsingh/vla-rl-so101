#!/usr/bin/env python
"""WO-06 item 4: record real teleop episodes (5 per task, sanity only - not training data).

Leader arm (so101_leader) -> canonical radians -> src/robot/safety.py -> follower, at control_hz,
with the front camera through the policy's image pipeline. Writes one LeRobotDataset (v3.0,
lerobot 0.4.4) with the SAME schema as WO-01's sim demos: ``observation.images.front``
(crop_to x crop_to RGB video), ``observation.state`` and ``action`` (float32[6] canonical radians;
action = the safe command actually sent), per-frame ``task`` = the protocol instruction
(``src.eval.protocol.instruction(task, RED_CUBE)``, the sim demos' string), also recorded with the
protocol hash in ``meta/vla_rl_so101.json``.

Needs: a leader arm in ``teleop.leader`` (otherwise exits with a message), both arms calibrated
(scripts/wo06_calibrate.py), and the shared FK from WO-01 for the workspace check.

    uv run scripts/wo06_teleop_record.py --preset real --task Push --rehearsal   # 1 throwaway episode
    uv run scripts/wo06_teleop_record.py --preset real                           # 5 episodes x 3 tasks
    uv run scripts/wo06_teleop_record.py --preset real --push                    # also upload (private)

Torque is ON from connect until the session ends. Per episode: arrange the scene, press ENTER, the
follower first eases onto the leader's pose clipped to the safety limits (rate-limited, not
recorded), then recording runs for ``teleop.episode_time_s`` or until Ctrl-C, then the follower
holds its pose; keep, redo or quit. ``q`` or Ctrl-C at any prompt (or the last episode) ends the
session the healthy way: the follower PARKS at ``robot.park.qpos`` through the safety filter (it
moves; Ctrl-C stops the park), then torque goes off. If the park cannot complete, or the session
ends after a deadman stop (no ``resume``), it holds, asks the operator to support the arm, then
torque off. A lost frame / stall holds the arm and asks you; a serial error cuts torque; a servo
over-temperature / over-current (``robot.servo_monitor``) cuts torque AT ONCE and tells you to
support the arm (no hold, no park). Outside the prompts Ctrl-C never interrupts a bus or dataset
call: it is a flag checked between control ticks (ends the sync or the episode), and ignored
while an episode saves. ``--verbose`` also prints the safety filter's per-tick INFO clips.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args, load_config
from src.eval.protocol import (
    PROTOCOL_VERSION,
    RED_CUBE,
    TASKS,
    instruction,
    protocol_hash,
)
from src.robot.camera import (
    CameraFault,
    OpenCVFrameSource,
    PolicyImagePipeline,
    sim_mismatches,
)
from src.robot.driver import (
    DeadmanTripped,
    ParkSettings,
    SerialFault,
    ServoFault,
    So101Driver,
    StepResult,
)
from src.robot.safety import DeadmanResponse, load_sim_fk
from src.robot.units import JOINT_NAMES, UnitMap

IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
SYNC_TOL_RAD = np.radians(3.0)
SYNC_TIMEOUT_S = 10.0
QUIT = "q"
SUPPORT_PROMPT = (
    "Ending the session. The follower holds its pose under torque. Put a hand under the forearm "
    "(or check it rests on the table), then press ENTER: torque goes OFF, the arm goes limp "
)

NO_LEADER = """No leader arm configured (teleop.leader is null in configs/real.yaml).

Teleop recording drives the follower from an SO-101 LEADER arm. If you own only a follower,
skip WO-06 item 4 (5 sanity episodes per task; nothing trains on them) and note it in
docs/STATUS.md - the rest of WO-06 (calibration, safety, camera match) does not need a leader.
With a leader: set teleop.leader to
  {type: so101_leader, id: so101_leader, port: /dev/tty.usbmodemYYYY,
   calibration_dir: ~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader}
then run scripts/wo06_calibrate.py find-port/calibrate/verify --device leader."""


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


def protocol_instructions(tasks: Sequence[str]) -> dict[str, str]:
    """Per-frame ``task`` strings: the protocol's seen-object instruction (= the sim demos')."""
    return {t: instruction(t, RED_CUBE) for t in tasks}


def prompt_input(prompt: str) -> str:
    """``input()`` with Ctrl-C raising KeyboardInterrupt, also inside ``ctrl_c_as_flag``."""
    prev = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return input(prompt)
    finally:
        signal.signal(signal.SIGINT, prev)


def ask(prompt: str) -> str:
    """``input()``, lower-cased; Ctrl-C / EOF answer ``q`` (end the session the safe way)."""
    try:
        return prompt_input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return QUIT


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
    def __init__(
        self,
        arm: So101Driver,
        leader: Any,
        cam: OpenCVFrameSource,
        units: UnitMap,
        park: ParkSettings | None = None,
    ):
        self.arm, self.leader, self.cam, self.units = arm, leader, cam, units
        self.park = park  # healthy-end park pose; None = hold + support prompt only
        self.period = 1.0 / arm.safety.config.control_hz
        self.target: np.ndarray | None = None  # last leader pose, canonical rad
        self.stop = False  # Ctrl-C under ctrl_c_as_flag; the loops check it between ticks

    def frame_time(self) -> float | None:
        try:
            return float(self.cam.read().capture_time)
        except CameraFault as e:
            print(f"camera: {e}")
            return None

    def tick(self) -> tuple[Any, StepResult]:
        """One control step. Returns (camera frame or None, step result)."""
        try:
            frame = self.cam.read()
        except CameraFault as e:
            print(f"camera: {e}")
            frame = None
        try:
            self.target = self.units.to_canonical(self.leader.get_action())
        except Exception as e:
            self.arm.deadman.trip(f"leader read failed: {e}", DeadmanResponse.HOLD)
            self.arm.hold()
            raise DeadmanTripped(self.arm.deadman.reason) from e
        out = self.arm.step(self.target, frame_time=None if frame is None else frame.capture_time)
        return frame, out

    def sleep_to(self, t0: float) -> None:
        time.sleep(max(0.0, self.period - (time.perf_counter() - t0)))

    def sync_to_leader(self, timeout_s: float = SYNC_TIMEOUT_S) -> bool:
        """Ease the follower onto the leader's pose through the rate limiter (not recorded).

        Converged when every joint is within SYNC_TOL_RAD of the leader pose CLIPPED to the
        safety joint limits: the follower never goes past them, so a fully closed or open trigger
        or a leader joint at its end stop must not block the sync. Like the filter, a joint already
        outside a limit (gripper resting closed, arm folded on its stops) has that limit widened to
        the current command: the filter holds it there rather than move it further out. Prints why
        it waits on a workspace rejection and which joints lag on a timeout. Ctrl-C (``stop``)
        aborts it: False.
        """
        sc = self.arm.safety.config
        lo, hi = np.asarray(sc.lower_rad), np.asarray(sc.upper_rad)
        t_end = time.perf_counter() + timeout_s
        warned = False
        goal = measured = None
        while time.perf_counter() < t_end:
            if self.stop:
                print("\nsync aborted")
                return False
            t0 = time.perf_counter()
            _, out = self.tick()
            cmd = out.safety.command
            assert cmd is not None and self.target is not None
            goal = np.clip(self.target, np.minimum(lo, cmd), np.maximum(hi, cmd))
            measured = out.measured
            if np.all(np.abs(measured - goal) <= SYNC_TOL_RAD):
                return True
            ws = [iv.reason for iv in out.safety.interventions if iv.kind == "workspace_reject"]
            if ws and not warned:
                warned = True
                print(
                    f"safety rejects the leader pose ({ws[0]}). The follower holds: move the "
                    f"leader's gripper into workspace_box_m x {sc.workspace_lo_m[0]}..."
                    f"{sc.workspace_hi_m[0]}, y {sc.workspace_lo_m[1]}..{sc.workspace_hi_m[1]}, "
                    f"z {sc.workspace_lo_m[2]}..{sc.workspace_hi_m[2]} m (robot base frame)."
                )
            self.sleep_to(t0)
        if goal is not None and measured is not None:
            lag = [
                f"{n} {np.degrees(m):.1f} vs {np.degrees(g):.1f}"
                for n, m, g in zip(JOINT_NAMES, measured, goal, strict=True)
                if abs(m - g) > SYNC_TOL_RAD
            ]
            print(f"not in sync after {timeout_s:.0f} s (follower vs target, deg): {lag}")
        return False


@contextmanager
def ctrl_c_as_flag(sess: Session) -> Iterator[None]:
    """Ctrl-C only sets ``sess.stop`` (``prompt_input`` turns it back into KeyboardInterrupt).

    A KeyboardInterrupt raised mid-tick can land inside scservo_sdk's pure-Python packet loop,
    which leaves the port marked busy (every later bus call fails, the hold too: the arm drops),
    or inside ``add_frame``, which leaves the episode buffer ragged (``save_episode`` raises).
    """

    def on_sigint(signum: int, frame: Any) -> None:
        sess.stop = True

    prev = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


def discard_episode(dataset: Any) -> None:
    try:
        dataset.clear_episode_buffer()
    except Exception:  # noqa: BLE001 - e.g. after a failed save_episode popped buffer keys
        dataset.episode_buffer = dataset.create_episode_buffer()


def end_session(arm: So101Driver, sess: Session | None = None) -> None:
    """End an operator-ended session (``q`` / Ctrl-C at a prompt, the last episode, no ``resume``).

    Healthy (deadman clear) with ``sess.park`` set: park through the safety filter, then torque
    off (``So101Driver.park``; Ctrl-C during the park stops it). Otherwise, or if the park does
    not complete: hold at the measured pose, then let the operator support the arm before torque
    goes off, when the driver's ``with`` block closes normally. Only ENTER (or a closed stdin)
    gets there: Ctrl-C at this prompt is ignored, a panicked second press must not drop the arm.
    """
    if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
        return  # torque is already off
    if sess is not None and sess.park is not None and not arm.deadman.tripped:
        print("ending the session: the follower PARKS (it moves; Ctrl-C stops it), then torque off")
        sess.stop = False
        if arm.park(sess.park, sess.frame_time, stop_fn=lambda: sess.stop):
            print("follower parked; torque OFF")
            return
        if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
            return  # a servo / serial fault during the park: torque is already off
        print("park did not complete: the follower holds")
    try:
        arm.hold()
    except SerialFault:
        return  # escalated to torque off already
    while True:
        try:
            prompt_input(SUPPORT_PROMPT)
            return
        except KeyboardInterrupt:
            print(
                "\nCtrl-C ignored: the follower still holds. Support the arm, then press ENTER "
                "(or switch off the servo supply)."
            )
        except EOFError:
            print("\nWARNING stdin closed: torque goes OFF without confirmation, the arm drops")
            return


def record_session(
    sess: Session,
    dataset: Any,
    tasks: Sequence[str],
    instructions: dict[str, str],
    n_eps: int,
    ep_time: float,
    pipe: Callable[[Any], Any],
) -> int:
    """The interactive episode loop; returns the number of saved episodes. Ends through
    ``end_session`` unless an exception (serial fault, torque-off deadman) ends it first."""
    with ctrl_c_as_flag(sess):
        return _episodes(sess, dataset, tasks, instructions, n_eps, ep_time, pipe)


def _episodes(
    sess: Session,
    dataset: Any,
    tasks: Sequence[str],
    instructions: dict[str, str],
    n_eps: int,
    ep_time: float,
    pipe: Callable[[Any], Any],
) -> int:
    arm = sess.arm
    saved = 0
    for task in tasks:
        k = 0
        while k < n_eps:
            msg = f"[{task} {k + 1}/{n_eps}] arrange the scene, hold the leader, ENTER (q = end): "
            if ask(msg) == QUIT:
                end_session(arm, sess)
                return saved
            sess.stop = False  # drop a Ctrl-C pressed while saving / holding
            try:
                arm.check_servos()  # heat built up while the arm held at the prompt
                if not sess.sync_to_leader():
                    arm.hold()
                    print("follower holds; retry")
                    continue
                t_start = time.perf_counter()
                print(f"recording ({ep_time:.0f} s max, Ctrl-C to end) ...")
                while time.perf_counter() - t_start < ep_time and not sess.stop:
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
                if sess.stop:
                    print("\nepisode ended")
                arm.hold()  # zero position error while the prompts wait (servo heat)
            except DeadmanTripped as e:
                print(f"\nSTOPPED: {e}\nEpisode discarded.")
                if isinstance(e, ServoFault):
                    print("SUPPORT THE ARM NOW: torque is OFF (servo fault).")
                dataset.clear_episode_buffer()
                if arm.deadman.response is not DeadmanResponse.HOLD:
                    raise  # torque is off: end the session
                ans = ask("arm is holding; type 'resume' to clear the deadman, else end: ")
                if ans != "resume":
                    end_session(arm, sess)
                    return saved
                arm.deadman.reset("operator typed resume")
                arm.safety.reset_reference()
                continue
            n_frames = int(dataset.episode_buffer["size"])
            keep = ask(f"keep this episode ({n_frames} frames)? [Y]es / [n]o, redo / [q]uit: ")
            if keep == QUIT:
                dataset.clear_episode_buffer()
                end_session(arm, sess)
                return saved
            if n_frames > 0 and keep in ("", "y", "yes"):
                print("saving episode (encoding video, a few seconds; Ctrl-C is ignored) ...")
                try:
                    dataset.save_episode()
                except Exception as e:  # noqa: BLE001 - the arm holds; never exit through __exit__
                    print(f"saving failed ({type(e).__name__}: {e}); episode discarded, redo it")
                    discard_episode(dataset)
                    continue
                saved += 1
                k += 1
            else:
                dataset.clear_episode_buffer()
    end_session(arm, sess)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--task", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--episodes", type=int, default=None, help="per task (default config)")
    parser.add_argument("--overwrite", action="store_true", help="replace the local dataset")
    parser.add_argument("--push", action="store_true", help="upload to the Hub (private)")
    parser.add_argument(
        "--verbose", action="store_true", help="also log the safety filter's per-tick INFO clips"
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="1 episode per task into a throwaway <dataset>-rehearsal folder (replaced each run, "
        "never pushed); the real dataset is not touched",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s", force=True
    )
    if args.verbose:
        logging.getLogger("src.robot.safety").setLevel(logging.INFO)
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    if args.rehearsal and args.push:
        parser.error("--rehearsal is never pushed")
    cfg = load_config(args.preset, args.config)
    tcfg = cfg.get("teleop") or {}
    if not tcfg.get("leader"):
        print(NO_LEADER)
        sys.exit(2)
    config_instructions: dict[str, str] = dict(tcfg["instructions"])
    tasks = args.task or list(config_instructions)
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        sys.exit(f"unknown tasks {unknown}; protocol tasks are {list(TASKS)}")
    instructions = protocol_instructions(tasks)
    for t in tasks:
        if config_instructions.get(t) != instructions[t]:
            print(
                f"NOTE {t}: recording the protocol instruction {instructions[t]!r} "
                f"(teleop.instructions says {config_instructions.get(t)!r})"
            )
    n_eps = 1 if args.rehearsal else int(args.episodes or tcfg["episodes_per_task"])
    ep_time = float(tcfg["episode_time_s"])
    sim_cfg = load_config("sim")
    for m in sim_mismatches(cfg, sim_cfg):
        print(f"WARNING camera config mismatch, {m}")

    fk = load_sim_fk()  # exits with instructions if WO-01's FK is not on this branch yet
    pipe = PolicyImagePipeline.from_config(cfg)
    units = UnitMap.from_config(cfg)
    repo_id = str(tcfg["hub_repo"])
    root = REPO_ROOT / str(tcfg["dataset_root"]) / repo_id.split("/")[-1]
    if args.rehearsal:
        root = root.with_name(root.name + "-rehearsal")
    if root.exists():
        if not (args.overwrite or args.rehearsal):
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
    provenance = {
        "source": "scripts/wo06_teleop_record.py",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "instructions": instructions,
        "config_instructions": {t: config_instructions.get(t) for t in tasks},
        "rehearsal": bool(args.rehearsal),
    }
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "vla_rl_so101.json").write_text(json.dumps(provenance, indent=2) + "\n")
    leader.connect(calibrate=False)  # leader configure() leaves its torque off
    try:
        if not leader.is_calibrated:
            sys.exit("leader motors do not match its calibration file; recalibrate the leader")
        with (
            OpenCVFrameSource(cfg["cameras"]["front"]) as cam,
            So101Driver.from_config(cfg, fk) as arm,  # torque ON from here, holding the pose
        ):
            sess = Session(arm, leader, cam, units, park=ParkSettings.from_config(cfg))
            record_session(sess, dataset, tasks, instructions, n_eps, ep_time, pipe)
    finally:
        dataset.finalize()
        if leader.is_connected:
            leader.disconnect()
        saved = int(dataset.num_episodes)
        print(f"saved {saved} episodes to {root}")
    if args.push and saved:
        dataset.push_to_hub(private=True)
        print(f"pushed to https://huggingface.co/datasets/{repo_id} (private)")


if __name__ == "__main__":
    main()
