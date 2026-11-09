#!/usr/bin/env python
"""WO-06 item 4 / WO-07: record real teleop episodes (leader arm, front + wrist cameras).

Leader arm (so101_leader) -> canonical radians -> src/robot/safety.py -> follower, at control_hz,
with BOTH cameras on every tick. Writes one LeRobotDataset (v3.0, lerobot 0.4.4) in the frozen
real-arm convention ``src/robot/convention.py`` (``CONVENTION_ID`` + ``convention_hash()``, cited
in ``meta/vla_rl_so101.json``): ``observation.images.front`` and ``observation.images.wrist``
(uncropped 480 x 640 RGB video), ``observation.state`` (the measured pose) and ``action`` (the safe
command actually sent; absolute targets), both float32[6] canonical radians, per-frame ``task`` =
the protocol instruction (``src.eval.protocol.instruction(task, RED_CUBE)``), also recorded with
the protocol hash. Every frame goes through ``convention.recording_frame``: a frame without both
cameras is never written, and the session refuses to start unless ``convention.check_config``
passes (both cameras configured at 640 x 480 / 30 fps, control_hz 15, the gripper encoding).

Needs: a leader arm in ``teleop.leader`` (otherwise exits with a message), both arms calibrated
(scripts/wo06_calibrate.py), and the shared FK from WO-01 for the workspace check.

    .venv/bin/python scripts/wo06_teleop_record.py --preset real --task Push --rehearsal  # 1 episode
    .venv/bin/python scripts/wo06_teleop_record.py --preset real                  # 5 x 3 tasks
    .venv/bin/python scripts/wo06_teleop_record.py --preset real --push           # also upload
    .venv/bin/python scripts/wo06_teleop_record.py --preset real --task PickPlace --demo-schedule
    .venv/bin/python scripts/wo06_teleop_record.py --preset real --task PickPlace --demo-schedule \
        --resume                                                          # continue the 50 demos

WO-07 demos (``--demo-schedule``; owner 2026-11-09, final2 item 6 / amendments D1, D2, D4): the 50
PickPlace demos of protocol real-0.1 (``src/eval/protocol_real.py`` ``demo_schedule()``): 10 at each
of the five frozen seen slots B2 B5 E2 E5 C4, the red cube, the bowl FIXED on its mark at
(x 0.195, y +0.205) m for every demo (``docs/assets/placement-grid-a3.pdf`` marks the slots and the
bowl). Never the other 11 inner slots, the 20-slot ring or the unseen objects (purple / green cube),
and none of them in a camera-match frame either. Each episode's prompt names its slot and yaw (a
PLANNED pose); a redo keeps the same placement; ``meta/vla_rl_so101.json`` maps every saved episode
to its placement and cites the protocol hash, the session id, the time, the follower id and the
calibration copied to ``meta/calibration_follower.json`` (what ``scripts/wo07_export_bundle.py
demo`` needs; it refuses a dataset recorded without ``--demo-schedule``).

Durable and resumable (review F1 / F5). Every kept episode is saved, its placement written to
``meta/vla_rl_so101.json``, then the dataset is finalized and reopened (``src/robot/recording.py``
``DurableDataset``): a power loss or a kill costs at most the episode being saved. A session that
ended early (a servo fault, a timeout, ``q``, a closed Terminal) continues with ``--resume``: it
recovers the dataset (``recover_dataset``), refuses unless the dataset cites the same protocol
hash, convention hash, demo schedule and ``robot.units`` and the follower loads the same
calibration, and continues the frozen schedule at the next demo index (the session id stays the
dataset's; each sitting is listed under ``sittings``).

Camera roles (review conv-4): before torque goes on, one frame of each camera is written as a
labelled sheet (``meta/camera_check_<n>.png``, opened in Preview) and the operator types ``yes``
only if FRONT shows the front view and WRIST the gripper (``src/robot/camera.py``
``confirm_camera_roles``); OpenCV indices follow USB enumeration on macOS. Recorded per sitting.

(``.venv/bin/python``, not ``uv run``: under ``uv run`` a Ctrl-Z also stops ``uv`` and the session
ends; ``src/robot/prompts.py``.)

Torque is ON from connect until the session ends. Per episode: arrange the scene, press ENTER, the
follower first eases onto the leader's pose clipped to the safety limits (rate-limited, not
recorded), then recording runs for ``teleop.episode_time_s`` or until Ctrl-C, then the follower
holds its pose; keep, redo or quit. ``q`` or Ctrl-C at any prompt (or the last episode) ends the
session: the follower holds, the operator supports the arm, ENTER turns torque off. With
``robot.park.enabled: true`` (default false until the owner verifies the rest pose and the park
path, WO-06 checklist step 5) a healthy end PARKS at ``robot.park.qpos`` through the safety filter
first (it moves; Ctrl-C stops the park). If the park cannot complete, or the session ends after a
deadman stop (no ``resume``), it holds, asks the operator to support the arm, then torque off.
After a hang-up (SIGHUP / SIGTERM) or a lost terminal foreground has been recorded it never parks;
one during the park stops it before the next goal write (owner decision 2026-11-08). It then
holds, waits the grace period without stdin and turns torque off: ``released_unconfirmed``,
``support_confirmed: false``, ``park: skipped_after_hangup`` / ``interrupted_by_hangup`` (or
``..._lost_foreground``) and ``hangup_signal`` in ``meta/vla_rl_so101.json``.

Bounded torque-on waits (``teleop.prompt_torque_timeout_s``, default 60 s; null = wait forever),
shared with the WO-07 pilot and real eval (``src/robot/prompts.py``): every prompt answered while
the follower holds under torque (scene, keep, resume) reads the servos' temperature and current
first, waits at most that long, and reads them again after the answer; the final support prompt
reads them once before its wait. With no answer in time it tells the operator to support the arm,
waits ``teleop.prompt_support_grace_s`` (default 10 s; ENTER = now), turns torque off and ends the
session (saved episodes are kept; an unsaved one is discarded). At the final support prompt only
ENTER confirms: Ctrl-C is ignored (the follower keeps holding, the bounded wait keeps running); a
closed stdin prints a loud warning, waits the grace period without stdin, then turns torque off.
Ctrl-Z never suspends the session (it prints that it was ignored); losing the terminal's
foreground (the process resumed with ``bg``) is handled like a closed Terminal, ``lost_foreground``.
Those ends are recorded as ``released_unconfirmed`` (``session_end`` and ``support_confirmed:
false`` in ``meta/vla_rl_so101.json``), never as supported. The wait is ``select()`` on stdin in
this one thread: the servo bus keeps a single owner, and nothing reads it in the background.

A failed torque-off write ("TORQUE OFF UNCONFIRMED", ``So101Driver.torque_off_unconfirmed``)
blocks every further goal-position and torque-enable write; the script tells the operator to
switch off the inline power switch and ends.

Any other error (the dataset, a frame outside the convention, a bug) holds the follower, discards the
episode and goes through the same support prompt before it is re-raised.

A lost frame / stall holds the arm and asks you; a serial error cuts torque; a servo
over-temperature / over-current (``robot.servo_monitor``) cuts torque AT ONCE and tells you to
support the arm (no hold, no park). Outside the prompts Ctrl-C never interrupts a bus or dataset
call: it is a flag checked between control ticks (ends the sync or the episode), and ignored
while an episode saves. ``--verbose`` also prints the safety filter's per-tick INFO clips.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import shutil
import subprocess
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
from src.data.recorder import datasets_write_compat
from src.eval import protocol_real as PR
from src.eval.protocol import (
    PROTOCOL_VERSION,
    RED_CUBE,
    TASKS,
    instruction,
    protocol_hash,
)
from src.robot import convention as conv
from src.robot import recording
from src.robot.camera import (
    CameraFault,
    CameraRig,
    CameraRoleError,
    RigFrame,
    confirm_camera_roles,
    sim_mismatches,
)
from src.robot.driver import (
    TORQUE_OFF_UNCONFIRMED,
    TORQUE_OFF_UNCONFIRMED_MSG,
    DeadmanTripped,
    MotionStopped,
    ParkSettings,
    SerialFault,
    ServoFault,
    So101Driver,
    StepResult,
    park_from_config,
)
from src.robot.prompts import (
    CTRL_C,
    ENTER,
    INTERRUPTS,
    STDIN_GONE,
    TIMEOUT,
    LineReader,
    TorqueGuard,
    TorqueWait,
    job_control,
    operator_signals,
    prompt_input,
    require_terminal,
    timed_input,
)
from src.robot.safety import DeadmanResponse, load_sim_fk
from src.robot.units import JOINT_NAMES, UnitMap
from src.runs import new_run_id

log = logging.getLogger(__name__)

STATE_KEY = conv.STATE_KEY
ACTION_KEY = conv.ACTION_KEY
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


def features() -> dict[str, dict[str, Any]]:
    """The dataset schema: ``convention.dataset_features`` (both cameras, state, action)."""
    return conv.dataset_features()


def create_dataset(repo_id: str, root: Path) -> Any:
    """The recording LeRobotDataset (v3.0) in the frozen convention (``src/robot/recording.py``,
    shared with the evaluation capture). ``root`` must not exist."""
    return recording.create_dataset(repo_id, root)


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def protocol_instructions(tasks: Sequence[str]) -> dict[str, str]:
    """Per-frame ``task`` strings: the protocol's seen-object instruction (= the sim demos')."""
    return {t: instruction(t, RED_CUBE) for t in tasks}


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
        cam: CameraRig,
        units: UnitMap,
        park: ParkSettings | None = None,
        torque_wait: TorqueWait | None = None,
        read: LineReader = timed_input,
    ):
        self.arm, self.leader, self.units = arm, leader, units
        self.cam = cam  # every camera of the convention (front + wrist), read together
        self.park = park  # healthy-end park pose; None = hold + support prompt only
        # bounded torque-on prompts (src/robot/prompts.py); torque_wait None = wait forever
        self.guard = TorqueGuard(arm, torque_wait, read=read)
        self.session_end: str | None = None  # end_session's result (provenance file)
        # what the end-of-session park did (provenance ``park``; ``PARK_OUTCOMES``); None = no
        # park was due (parking disabled, a deadman stop, torque already off)
        self.park_outcome: str | None = None
        self.period = 1.0 / arm.safety.config.control_hz
        self.target: np.ndarray | None = None  # last leader pose, canonical rad
        self.stop = False  # Ctrl-C under ctrl_c_as_flag; the loops check it between ticks
        # saved episode index -> its demo-schedule placement (``--demo-schedule``)
        self.placements: dict[str, dict[str, Any]] = {}
        # after every saved episode, before the dataset is finalized: write the provenance
        self.on_saved: Callable[[], None] | None = None

    @property
    def torque_released(self) -> bool:
        """Torque was turned off after a prompt timed out or stdin closed (support unconfirmed)."""
        return self.guard.released is not None

    def frame_time(self) -> float | None:
        try:
            return float(self.cam.read().capture_time)
        except CameraFault as e:
            print(f"camera: {e}")
            return None

    def stop_requested(self) -> bool:
        """Checked before every tick, and immediately before every goal write of a tick and of
        the end-of-session park (``So101Driver.step``'s ``stop_fn``): a
        Ctrl-C (``stop``), a RECORDED hang-up (SIGHUP / SIGTERM, ``INTERRUPTS.hangup``: also
        while the terminal still reports this process as its foreground), or the terminal's
        foreground lost (recorded, or polled here: ``Interrupts.terminal_gone``). On a hang-up
        or a loss ``ctrl_c_as_flag`` also sets ``stop``; the recorded flags are checked too, so
        clearing ``stop`` (``end_session`` does, before the park) never clears them. Review F3:
        the poll used to sit inside ``tick``, which then still sent one goal after detecting
        the loss."""
        return INTERRUPTS.terminal_gone() is not None or self.stop

    def tick(self) -> tuple[RigFrame | None, StepResult]:
        """One control step. Returns (frame of every camera or None, step result). A fault on
        either camera is no frame: the deadman trips in ``arm.step``. The loops call
        ``stop_requested`` first; ``arm.step`` calls it again after the camera, leader and servo
        reads, immediately before its goal write, and raises ``MotionStopped`` (nothing written)
        when it is True: a hang-up, a lost foreground or a Ctrl-C recorded while this tick waited
        on a frame never sends one more leader-following goal (review F2, round 8)."""
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
        out = self.arm.step(
            self.target,
            frame_time=None if frame is None else frame.capture_time,
            stop_fn=self.stop_requested,
        )
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
            if self.stop_requested():
                print("\nsync aborted")
                return False
            t0 = time.perf_counter()
            try:
                _, out = self.tick()
            except MotionStopped:  # recorded mid-tick: no goal written this tick
                print("\nsync aborted")
                return False
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
    SIGHUP / SIGTERM (the Terminal closed) also set ``sess.stop``, and every later prompt ends
    as ``hangup``: the closed-stdin path (grace period, torque off, ``released_unconfirmed``)
    instead of the process dying with the arm held under torque (``operator_signals``). A lost
    terminal foreground (SIGTTIN / SIGTTOU, or ``tcgetpgrp`` polled every tick) does the same,
    as ``lost_foreground``. Ctrl-Z is ignored (``job_control``, which ``main`` also enters
    around the whole torque-on block).
    """

    def on_signal() -> None:
        sess.stop = True

    with operator_signals(on_signal):
        yield


def _torque_off_unconfirmed(arm: So101Driver) -> bool:
    """Tell the operator to cut power if a torque-off write failed (driver latch). True if so."""
    if arm.torque_off_unconfirmed is None:
        return False
    print(f"\n{TORQUE_OFF_UNCONFIRMED_MSG.format(detail=arm.torque_off_unconfirmed)}", flush=True)
    return True


def torque_on_ask(sess: Session, msg: str) -> str:
    """``ask`` at a prompt where the follower holds under torque (``TorqueGuard.ask``): servo
    health before and after the wait, and the wait bounded by ``teleop.prompt_torque_timeout_s``
    (module doc). A timeout releases torque and answers ``q``; Ctrl-C / EOF answer ``q``. Raises
    ServoFault / SerialFault from the health reads."""
    try:
        reply = sess.guard.ask(msg)
    except ServoFault:
        print("SUPPORT THE ARM NOW: torque is OFF (servo fault).")
        raise
    if reply.how == CTRL_C or reply.how in STDIN_GONE:
        print()
        return QUIT
    if reply.how == TIMEOUT:
        return QUIT  # torque already released (support message, grace, torque off)
    return reply.answer


def save_durably(dataset: Any, hook: Callable[[int], None]) -> int:
    """Save the buffered episode; ``hook(index)`` records it; a ``recording.DurableDataset`` is
    then finalized and reopened (module doc). Returns the episode's index."""
    if isinstance(dataset, recording.DurableDataset):
        return dataset.save_episode(before_checkpoint=hook)
    index = int(getattr(dataset, "num_episodes", 0))
    with datasets_write_compat():
        dataset.save_episode()
    hook(index)
    return index


def discard_episode(dataset: Any) -> None:
    try:
        dataset.clear_episode_buffer()
    except Exception:  # noqa: BLE001 - e.g. after a failed save_episode popped buffer keys
        dataset.episode_buffer = dataset.create_episode_buffer()


def end_session(arm: So101Driver, sess: Session | None = None) -> str:
    """End an operator-ended session (``q`` / Ctrl-C at a prompt, the last episode, no ``resume``).
    Returns how it ended (also ``sess.session_end``, written to the provenance file): parked |
    supported | released_unconfirmed | torque_was_off | torque_off_unconfirmed.

    Healthy (deadman clear) with ``sess.park`` set (``robot.park.enabled``): park through the
    safety filter, then torque off (``So101Driver.park``; Ctrl-C during the park stops it).
    Owner decision 2026-11-08: a hang-up (SIGHUP / SIGTERM) or a lost foreground recorded before
    the park would start skips it (no park goal written); one during the park stops it before
    its next goal write (``_park``, ``sess.park_outcome``). Either way the arm holds, the support
    prompt ends at once as ``hangup`` / ``lost_foreground``, the grace period runs without
    stdin and torque goes off: ``released_unconfirmed``, never "parked" or "supported".
    Otherwise, or if the park does not complete: hold at the measured pose, then let the operator
    support the arm before torque goes off, when the driver's ``with`` block closes normally.
    The servos are read once after the hold, before the wait. Only ENTER confirms the support
    (``supported``): Ctrl-C at this prompt is ignored, a panicked second press must not drop the
    arm, and the bounded wait keeps running. A closed stdin or a hung-up terminal (loud message,
    the grace period without stdin; the grace applies with ``prompt_torque_timeout_s: null``
    too) or the bounded wait plus its grace running out turns torque off without a
    confirmation: ``released_unconfirmed`` (the cause in the provenance file's
    ``support_release``), never "supported". A failed torque-off (driver latch)
    tells the operator to cut power at the inline switch.
    """
    end = _end_session(arm, sess)
    if sess is not None:
        sess.session_end = end
    return end


PARK_OUTCOMES = (
    "parked",
    "skipped_after_hangup",
    "skipped_after_lost_foreground",
    "interrupted_by_hangup",
    "interrupted_by_lost_foreground",
    "stopped_by_ctrl_c",
    "not_completed",
)
"""``Session.park_outcome``: ``skipped_after_*`` = a hang-up (SIGHUP / SIGTERM) or a lost
foreground was recorded before the park would start, so it never started (no park goal written);
``interrupted_by_*`` = one arrived during the park, which stopped before its next goal write;
``not_completed`` = a timeout, a rejected pose or a fault. Every outcome but ``parked`` goes on to
the hold and the support prompt (after a hang-up or a lost foreground: the grace period without
stdin, torque off, ``released_unconfirmed``)."""


def _park(arm: So101Driver, sess: Session, park: ParkSettings) -> bool:
    """The end-of-session park (owner decision 2026-11-08): never started once a hang-up or a
    lost foreground is recorded (``Interrupts.terminal_gone``: the recorded SIGHUP / SIGTERM
    decides, whatever ``tcgetpgrp`` says); during the park ``Session.stop_requested`` runs
    before every goal write (``So101Driver.move_to``) and stops it on a Ctrl-C, a hang-up or a
    lost foreground. True only when it parked with torque off. Sets ``sess.park_outcome``."""
    gone = INTERRUPTS.terminal_gone()
    if gone is not None:
        sess.park_outcome = f"skipped_after_{gone[0]}"
        print(f"\n{gone[1]}: NOT parking; the follower holds", flush=True)
        return False
    print("ending the session: the follower PARKS (it moves; Ctrl-C stops it), then torque off")
    sess.stop = False  # drop a Ctrl-C pressed at the last prompt (the recorded flags stay)
    if arm.park(park, sess.frame_time, stop_fn=sess.stop_requested):
        # the driver checked stop_requested after the last goal write too, before the torque
        # off: a hang-up after that finds the arm parked with torque off
        sess.park_outcome = "parked"
        return True
    gone = INTERRUPTS.terminal_gone()
    if gone is not None:
        sess.park_outcome = f"interrupted_by_{gone[0]}"
    else:
        sess.park_outcome = "stopped_by_ctrl_c" if sess.stop else "not_completed"
    return False


def _end_session(arm: So101Driver, sess: Session | None) -> str:
    if _torque_off_unconfirmed(arm):
        return TORQUE_OFF_UNCONFIRMED
    if sess is not None and sess.torque_released:
        return "released_unconfirmed"  # a prompt timed out / stdin closed: torque is off
    if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
        return "torque_was_off"
    if sess is not None and sess.park is not None and not arm.deadman.tripped:
        if _park(arm, sess, sess.park):
            print("follower parked; torque OFF")
            return "parked"
        if _torque_off_unconfirmed(arm):
            return TORQUE_OFF_UNCONFIRMED
        if arm.deadman.response is DeadmanResponse.TORQUE_OFF:
            return "torque_was_off"  # a servo / serial fault during the park
        if sess.park_outcome is not None and sess.park_outcome.startswith("interrupted_by_"):
            print("park interrupted: the follower holds")
        elif sess.park_outcome is not None and not sess.park_outcome.startswith("skipped_"):
            print("park did not complete: the follower holds")
    guard = sess.guard if sess is not None else TorqueGuard(arm, TorqueWait(None), read=timed_input)
    try:
        arm.hold()
        how = guard.support(SUPPORT_PROMPT)  # servo health read before the wait
    except (ServoFault, SerialFault) as e:
        if isinstance(e, ServoFault):
            print("SUPPORT THE ARM NOW: torque is OFF (servo fault).")
        return TORQUE_OFF_UNCONFIRMED if _torque_off_unconfirmed(arm) else "torque_was_off"
    if how == ENTER:
        return "supported"  # torque goes off when the driver's `with` block closes
    _torque_off_unconfirmed(arm)
    return TORQUE_OFF_UNCONFIRMED if arm.torque_off_unconfirmed else "released_unconfirmed"


def session_record(arm: So101Driver, sess: Session | None) -> dict[str, Any]:
    """How the session ended, for the provenance file (``meta/vla_rl_so101.json``):
    ``session_end``; ``support_confirmed`` (None after a completed park or no session, True only
    after ENTER at the support prompt); ``support_release`` (``TorqueGuard.released``: the cause
    timeout | eof | hangup | lost_foreground); ``park`` (``PARK_OUTCOMES``, None when no park
    was due); ``hangup_signal`` (SIGHUP | SIGTERM once one was received, else None);
    ``lost_foreground`` (why, once the foreground was lost); ``torque_off_unconfirmed``."""
    end = None if sess is None else sess.session_end
    return {
        "session_end": end,
        "support_confirmed": None if end in (None, "parked") else end == "supported",
        "support_release": None if sess is None else sess.guard.released,
        "park": None if sess is None else sess.park_outcome,
        "hangup_signal": INTERRUPTS.hangup,
        "lost_foreground": INTERRUPTS.lost_foreground,
        "torque_off_unconfirmed": arm.torque_off_unconfirmed,
    }


def record_session(
    sess: Session,
    dataset: Any,
    tasks: Sequence[str],
    instructions: dict[str, str],
    n_eps: int,
    ep_time: float,
    placements: Sequence[dict[str, Any]] | None = None,
) -> int:
    """The interactive episode loop; returns the number of saved episodes. ``placements``
    (``--demo-schedule``): episode k of the task is placed as ``placements[k]`` (prompted; a redo
    keeps it) and recorded in ``sess.placements`` under its dataset episode index. Ends through
    ``end_session`` unless an exception (serial fault, torque-off deadman) ends it first. Any
    other exception (a dataset or convention error, a bug) holds the follower and goes
    through ``end_session``'s support prompt before it propagates (``_unexpected_error``): the
    driver's ``with`` block would otherwise drop the arm at once."""
    with ctrl_c_as_flag(sess):
        try:
            return _episodes(sess, dataset, tasks, instructions, n_eps, ep_time, placements)
        except (DeadmanTripped, SerialFault):
            raise  # torque already off (or not confirmed): nothing to hold or support
        except Exception as e:
            _unexpected_error(sess, dataset, e)
            raise


def _unexpected_error(sess: Session, dataset: Any, e: BaseException) -> None:
    """Hold (a HOLD deadman trip), discard the episode, then the bounded support prompt."""
    log.exception("teleop stopped on an unexpected error")
    print(f"\nSTOPPED: unexpected {type(e).__name__}: {e}; episode discarded", flush=True)
    try:
        discard_episode(dataset)
    except Exception as d:  # noqa: BLE001 - never skip the hold and the support prompt
        print(f"discarding the episode failed too ({type(d).__name__}: {d})")
    arm = sess.arm
    torque_off = arm.deadman.tripped and arm.deadman.response is DeadmanResponse.TORQUE_OFF
    if arm.torque_off_unconfirmed is None and not torque_off:
        try:
            arm.trip(f"teleop error: {type(e).__name__}: {e}", DeadmanResponse.HOLD)
        except SerialFault:
            pass  # the hold escalated to torque off; end_session reports it
    try:
        end_session(arm, sess)
    except Exception as f:  # noqa: BLE001 - the original error is the one to report
        print(f"ending the session failed ({type(f).__name__}: {f})")


def _episodes(
    sess: Session,
    dataset: Any,
    tasks: Sequence[str],
    instructions: dict[str, str],
    n_eps: int,
    ep_time: float,
    placements: Sequence[dict[str, Any]] | None = None,
) -> int:
    arm = sess.arm
    saved = 0
    for task in tasks:
        k = 0
        while k < n_eps:
            scene = (
                "arrange the scene" if placements is None else PR.placement_prompt(placements[k])
            )
            msg = f"[{task} {k + 1}/{n_eps}] {scene}, hold the leader, ENTER (q = end): "
            if torque_on_ask(sess, msg) == QUIT:  # servo health before and after the wait
                end_session(arm, sess)
                return saved
            sess.stop = False  # drop a Ctrl-C pressed while saving / holding
            try:
                if not sess.sync_to_leader():
                    arm.hold()
                    print("follower holds; retry")
                    continue
                t_start = time.perf_counter()
                t_first: float | None = None  # the episode's first write (timing_s origin)
                print(f"recording ({ep_time:.0f} s max, Ctrl-C to end) ...")
                while time.perf_counter() - t_start < ep_time and not sess.stop_requested():
                    t0 = time.perf_counter()
                    try:
                        frame, out = sess.tick()
                    except MotionStopped:  # recorded mid-tick: no goal, no frame this tick
                        break
                    t_write = time.perf_counter()  # the goal was written inside the tick
                    t_first = t_write if t_first is None else t_first
                    assert frame is not None and out.safety.command is not None
                    row = conv.recording_frame(  # refuses a frame without both cameras
                        frame.images,
                        out.measured,
                        out.safety.command,
                        instructions[task],
                        write_time_s=t_write - t_first,
                        frame_time_s=float(frame.capture_time) - t_first,
                    )
                    with datasets_write_compat():
                        dataset.add_frame(row)
                    sess.sleep_to(t0)
                if sess.stop:
                    print("\nepisode ended")
                arm.hold()  # zero position error while the prompts wait (servo heat)
            except DeadmanTripped as e:
                print(f"\nSTOPPED: {e}\nEpisode discarded.")
                if isinstance(e, ServoFault):
                    print("SUPPORT THE ARM NOW: torque is OFF (servo fault).")
                discard_episode(dataset)
                if arm.deadman.response is not DeadmanResponse.HOLD:
                    raise  # torque is off: end the session
                ans = torque_on_ask(
                    sess, "arm is holding; type 'resume' to clear the deadman, else end: "
                )
                if ans != "resume":
                    end_session(arm, sess)
                    return saved
                arm.deadman.reset("operator typed resume")
                arm.safety.reset_reference()
                continue
            n_frames = int(dataset.episode_buffer["size"])
            keep = torque_on_ask(
                sess, f"keep this episode ({n_frames} frames)? [Y]es / [n]o, redo / [q]uit: "
            )
            if keep == QUIT:
                discard_episode(dataset)
                end_session(arm, sess)
                return saved
            if n_frames > 0 and keep in ("", "y", "yes"):
                print("saving episode (encoding video, a few seconds; Ctrl-C is ignored) ...")

                def on_saved(index: int, k: int = k) -> None:
                    if placements is not None:  # only the demo schedule records the placement
                        sess.placements[str(index)] = dict(placements[k])
                    if sess.on_saved is not None:
                        sess.on_saved()  # before the checkpoint: a kill keeps the map

                try:
                    save_durably(dataset, on_saved)
                except recording.CheckpointFailed:  # saved, but not durable: stop recording
                    saved += 1
                    raise
                except Exception as e:  # noqa: BLE001 - the arm holds; never exit through __exit__
                    print(f"saving failed ({type(e).__name__}: {e}); episode discarded, redo it")
                    discard_episode(dataset)
                    continue
                saved += 1
                k += 1
            else:
                discard_episode(dataset)
    end_session(arm, sess)
    return saved


def recording_provenance(
    cfg: dict[str, Any],
    units: UnitMap,
    instructions: dict[str, str],
    config_instructions: dict[str, str | None],
    *,
    rehearsal: bool,
    demo_schedule: bool,
) -> dict[str, Any]:
    """``meta/vla_rl_so101.json`` at the start of a session (``main`` adds the calibration copy,
    the session end and the episode placements)."""
    return {
        "source": "scripts/wo06_teleop_record.py",
        "convention": conv.citation(),
        "units": dict(conv.DECLARED_UNITS),
        "unit_map": dataclasses.asdict(units),
        "cameras": {n: dict(cfg["cameras"][n]) for n in conv.CAMERA_NAMES},
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "instructions": instructions,
        "config_instructions": config_instructions,
        "rehearsal": rehearsal,
        # export inputs (scripts/wo07_export_bundle.py demo): nothing there is guessed
        "purpose": "demonstration",
        "session_id": new_run_id("WO-06", "teleop"),
        "recorded_at": _utc_now(),
        "robot_id": str(cfg["robot"]["id"]),
        "demo_schedule": demo_schedule,
        "protocol": PR.citation() if demo_schedule else None,
        "episodes": {},
        "sittings": [],
    }


PROVENANCE_FILE = "meta/vla_rl_so101.json"


def resume_problems(
    prov: dict[str, Any], units: UnitMap, num_episodes: int, placements: Sequence[dict[str, Any]]
) -> list[str]:
    """Why a recorded demo dataset may not be continued (empty = it may): the same protocol,
    convention, demo schedule and ``robot.units``, and its placements = demos 0..N-1 in order."""
    out = []
    if not prov.get("demo_schedule"):
        out.append("--resume continues a --demo-schedule recording only")
    if prov.get("rehearsal"):
        out.append("a rehearsal recording is never continued")
    if (prov.get("protocol") or {}).get("hash") != PR.protocol_hash():
        out.append(f"recorded under protocol hash {(prov.get('protocol') or {}).get('hash')}")
    if (prov.get("convention") or {}).get("hash") != conv.convention_hash():
        out.append(f"recorded under convention {(prov.get('convention') or {}).get('hash')}")
    if prov.get("unit_map") != json.loads(json.dumps(dataclasses.asdict(units))):
        out.append(f"robot.units changed: recorded {prov.get('unit_map')}")
    eps = prov.get("episodes") or {}
    if sorted(int(i) for i in eps) != list(range(num_episodes)):
        out.append(f"episodes {sorted(eps, key=int)} != the dataset's 0..{num_episodes - 1}")
    idx = [int(eps[str(i)].get("demo_idx", -1)) for i in range(num_episodes) if str(i) in eps]
    if idx != list(range(len(idx))):
        out.append(f"recorded demo indices {idx} are not 0..{len(idx) - 1} in order")
    for i in range(min(num_episodes, len(placements))):
        if str(i) in eps and eps[str(i)] != placements[i]:
            out.append(f"episode {i}'s placement is not demo {i} of the frozen schedule")
            break
    return out


def show_file(path: Path) -> None:
    """Open a picture in Preview (macOS); harmless where ``open`` does not exist."""
    try:
        subprocess.run(["open", str(path)], check=False)
    except OSError:
        pass


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
        "--demo-schedule",
        action="store_true",
        help="WO-07 demos: protocol real-0.1's frozen demo schedule (PickPlace, 5 seen slots x 10, "
        "fixed bowl); --episodes N records its first N",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted --demo-schedule recording at its next demo (module doc)",
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
    if args.resume and (args.rehearsal or args.overwrite or not args.demo_schedule):
        parser.error("--resume continues a --demo-schedule recording (no --rehearsal/--overwrite)")
    require_terminal("wo06_teleop_record")  # before the leader, the dataset or the follower
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
    placements: list[dict[str, Any]] | None = None
    if args.demo_schedule:
        if tasks != [PR.TASK]:
            sys.exit(f"--demo-schedule records {PR.TASK} only: pass --task {PR.TASK}")
        placements = PR.demo_schedule()
        n_eps = 1 if args.rehearsal else int(args.episodes or len(placements))
        if not 1 <= n_eps <= len(placements):
            sys.exit(f"--episodes must be in [1, {len(placements)}] with --demo-schedule")
    ep_time = float(tcfg["episode_time_s"])
    problems = conv.check_config(cfg)
    if problems:
        sys.exit(
            f"refusing to record: the config departs from {conv.CONVENTION_ID} "
            f"{conv.CONVENTION_VERSION} (src/robot/convention.py):\n  " + "\n  ".join(problems)
        )
    sim_cfg = load_config("sim")
    for m in sim_mismatches(cfg, sim_cfg):
        print(f"WARNING front camera config vs sim (pose / fov for the WO-08 sim match), {m}")

    fk = load_sim_fk()  # exits with instructions if WO-01's FK is not on this branch yet
    units = UnitMap.from_config(cfg)
    repo_id = str(tcfg["hub_repo"])
    root = REPO_ROOT / str(tcfg["dataset_root"]) / repo_id.split("/")[-1]
    if args.rehearsal:
        root = root.with_name(root.name + "-rehearsal")
    start = 0  # the first demo index of this sitting
    recovery: dict[str, Any] | None = None
    if args.resume:
        if not (root / "meta" / "info.json").is_file():
            sys.exit(f"--resume: no recording at {root}")
        recovery = recording.recover_dataset(root)  # a kill mid-save: quarantine, rebuild meta
        provenance = json.loads((root / PROVENANCE_FILE).read_text())
        dropped = recording.prune_provenance(provenance, int(recovery["episodes"]))
        assert placements is not None
        problems = resume_problems(provenance, units, int(recovery["episodes"]), placements)
        if recovery["episodes"] == 0:
            problems.append("no episode was saved: record it anew with --overwrite")
        if problems:
            sys.exit("--resume refused; nothing was opened:\n  " + "\n  ".join(problems))
        start = int(recovery["episodes"])
        if start >= len(placements):
            sys.exit(f"--resume: all {len(placements)} demos are recorded")
        n_eps = min(int(args.episodes or len(placements) - start), len(placements) - start)
        placements = placements[start:]
        recovery["pruned_placements"] = dropped
        print(f"resuming {root} at demo {start + 1}/{start + len(placements)}: {recovery}")
    elif root.exists():
        if not (args.overwrite or args.rehearsal):
            sys.exit(f"{root} exists; pass --overwrite to replace it, or --resume to continue")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    leader = make_leader(dict(tcfg["leader"]))
    if args.resume:
        dataset = recording.DurableDataset(recording.open_dataset(repo_id, root), repo_id)
    else:
        dataset = recording.DurableDataset(create_dataset(repo_id, root), repo_id)
        provenance = recording_provenance(
            cfg,
            units,
            instructions,
            {t: config_instructions.get(t) for t in tasks},
            rehearsal=bool(args.rehearsal),
            demo_schedule=placements is not None,
        )
    cite = conv.citation()
    print(f"recording in {cite['id']} {cite['version']} ({cite['module']}, sha256 {cite['hash']})")
    sitting: dict[str, Any] = {
        "started_utc": _utc_now(),
        "first_episode": start,
        "cameras": {n: dict(cfg["cameras"][n]) for n in conv.CAMERA_NAMES},
        "recovery": recovery,
    }
    provenance.setdefault("sittings", []).append(sitting)
    prov_path = root / PROVENANCE_FILE
    recording.write_json(prov_path, provenance)
    leader.connect(calibrate=False)  # leader configure() leaves its torque off
    try:
        if not leader.is_calibrated:
            sys.exit("leader motors do not match its calibration file; recalibrate the leader")
        arm = So101Driver.from_config(cfg, fk)
        # the follower's calibration as loaded (lerobot read <calibration_dir>/<id>.json)
        if args.resume:
            data, _src = recording.calibration_bytes(arm.follower)
            if hashlib.sha256(data).hexdigest() != (provenance.get("calibration") or {}).get(
                "sha256"
            ):
                sys.exit(
                    "--resume refused: the follower loads another calibration than the recorded "
                    "demos (its sha256 differs); torque was never enabled"
                )
        else:
            provenance["calibration"] = recording.copy_calibration(arm.follower, root)
        recording.write_json(prov_path, provenance)
        sess: Session | None = None
        # Ctrl-Z ignored, foreground watched: from before torque on until the session's record
        # is written, the TORQUE OFF UNCONFIRMED instruction included (review F3)
        with job_control():
            try:
                with CameraRig.from_config(cfg, conv.CAMERA_NAMES) as cam:  # front + wrist
                    n_check = len(provenance["sittings"])
                    try:  # before torque: which index is which camera (review conv-4)
                        sitting["camera_check"] = confirm_camera_roles(
                            cam,
                            root / "meta" / f"camera_check_{n_check}.png",
                            prompt_input,
                            indices={n: cfg["cameras"][n].get("index") for n in conv.CAMERA_NAMES},
                            show=show_file,
                        )
                    except CameraRoleError as e:
                        sys.exit(str(e))
                    recording.write_json(prov_path, provenance)
                    with arm:  # torque ON from here, holding the pose
                        sess = Session(
                            arm,
                            leader,
                            cam,
                            units,
                            park=park_from_config(cfg),  # None unless robot.park.enabled
                            torque_wait=TorqueWait.from_config(cfg, "teleop", allow_none=True),
                        )
                        sess.placements = dict(provenance.get("episodes") or {})  # a resume
                        live = sess

                        def save_provenance() -> None:
                            provenance["episodes"] = dict(live.placements)
                            recording.write_json(prov_path, provenance)

                        sess.on_saved = save_provenance
                        record_session(
                            sess, dataset, tasks, instructions, n_eps, ep_time, placements
                        )
            finally:
                _torque_off_unconfirmed(arm)  # also after the driver's own torque-off on exit
                ended = session_record(arm, sess)
                provenance.update(ended)
                sitting.update(ended)
                if sess is not None:
                    provenance["episodes"] = dict(sess.placements)
                recording.write_json(prov_path, provenance)
    finally:
        dataset.close()
        if leader.is_connected:
            leader.disconnect()
        saved = int(dataset.num_episodes)
        print(f"saved {saved} episodes to {root}")
    if args.push and saved:
        dataset.push_to_hub(private=True)
        print(f"pushed to https://huggingface.co/datasets/{repo_id} (private)")


if __name__ == "__main__":
    main()
