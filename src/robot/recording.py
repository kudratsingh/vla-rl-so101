"""Real-arm recordings in the frozen convention: the shared LeRobotDataset helpers, the follower's
calibration copy, and the evaluation capture (final2 hardware floor; WO-07 item 6b).

Demos (``scripts/wo06_teleop_record.py``) and evaluation trials (``scripts/wo07_real_eval.py``) are
written by the same code: ``create_dataset`` (LeRobot v3.0, ``convention.dataset_features``: both
cameras, state, action) and ``convention.recording_frame`` per tick (state = the pose measured just
before the write, action = the safety filter's command written on that tick, both camera images
read on that tick). Nothing here moves the arm or opens a device.

Evaluation capture (``EvalCapture``), one dataset per evaluation session, one episode per trial:

* ``RigFrameSource`` hands the deploy loop a ``CameraRig`` (front + wrist) as its frame source: the
  policy camera's image, the OLDEST camera's stamp (the deadman covers every camera) and every
  camera's image (the gRPC path, whose policy takes one camera). The in-process SmolVLA / ACT path
  (``src/robot/local_policy.py``) reads the ``CameraRig`` itself: its ``RigFrame`` carries the same
  images and stamps. ``DeployLoop.on_write`` passes each policy write's frame and step result to
  ``EvalCapture.on_write``. Frames are captured only for policy writes (the move home and the
  settle are not recorded, as the demo recorder does not record its sync).
* ``begin`` opens a trial's episode buffer; ``finish`` (from the trial record, after scoring)
  saves the episode when the policy started and at least one frame was written, else discards
  it, and returns the trial record's ``capture`` block: ``episode_index`` (None when nothing was
  saved, with the reason), ``frames``, ``trial_id`` and the end-state stills. It also reads one
  frame of every camera at that moment (the arm holding where the trial ended, after the settle
  and the score) and writes it as PNG stills for the blinded clip review.
* A capture fault NEVER reaches the arm or the score: ``on_write`` and ``finish`` catch it, the
  episode is discarded, and the record says why (``capture.error``).
* ``meta/vla_rl_so101.json`` cites the convention and the protocol and maps every saved episode
  index to its trial id (written after each save); ``meta/calibration_follower.json`` is the
  follower's LeRobot calibration file as loaded for the session.
* Resume: ``EvalCapture.open(resume=True)`` reopens the session's dataset (LeRobot's own resume
  path, ``lerobot_record.py --resume``) and appends. A dataset left unfinalized by a killed run
  (SIGKILL, power loss: its Parquet footers were never written) cannot be reopened: the resume
  refuses the capture and says so; every normal end, an error and a hang-up finalize it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.robot import convention as conv
from src.robot.camera import CameraFault, CameraRig, Frame, ImageU8, RigFrame

log = logging.getLogger(__name__)

PROVENANCE = "meta/vla_rl_so101.json"
CALIBRATION_COPY = "meta/calibration_follower.json"
END_STATE_DIR = "end_state"


def create_dataset(repo_id: str, root: Path) -> Any:
    """A new recording LeRobotDataset (v3.0) in the frozen convention. ``root`` must not exist."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.recorder import datasets_write_compat

    with datasets_write_compat():
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=conv.FPS,
            features=conv.dataset_features(),
            root=root,
            robot_type=conv.ROBOT_TYPE,
            use_videos=True,
            # LeRobot's record default: 4 PNG writer threads per camera, off the control tick
            image_writer_threads=4 * len(conv.CAMERAS),
            video_backend=conv.VIDEO_BACKEND,
            vcodec=conv.VIDEO_CODEC,
        )


def open_dataset(repo_id: str, root: Path) -> Any:
    """Reopen a finalized recording dataset to append episodes (LeRobot's resume path)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.recorder import install_torchcodec_compat

    install_torchcodec_compat()
    ds = LeRobotDataset(
        repo_id, root=root, video_backend=conv.VIDEO_BACKEND, vcodec=conv.VIDEO_CODEC
    )
    problems = conv.check_dataset_info(ds.meta.info)
    if problems:
        raise conv.ConventionError(f"{root}: not in the convention: {problems}")
    ds.start_image_writer(num_processes=0, num_threads=4 * len(conv.CAMERAS))
    return ds


def write_json(path: Path, value: Any) -> None:
    """Atomic JSON write (temp file + rename): a kill never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def calibration_bytes(follower: Any) -> tuple[bytes, str]:
    """The follower's LeRobot calibration as loaded: the file's bytes when LeRobot read one
    (``calibration_fpath``), else the in-memory calibration in LeRobot's file format (fakes).
    Returns (bytes, source)."""
    path = getattr(follower, "calibration_fpath", None)
    if path is not None and Path(path).is_file():
        return Path(path).read_bytes(), str(path)
    cal = getattr(follower, "calibration", None)
    if not cal:
        raise FileNotFoundError("the follower has no calibration loaded")
    keys = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
    body = {m: {k: int(getattr(c, k)) for k in keys} for m, c in cal.items()}
    return (json.dumps(body, indent=4) + "\n").encode(), "in-memory follower calibration (no file)"


def copy_calibration(follower: Any, dataset_root: Path) -> dict[str, Any]:
    """Write the calibration into the dataset's meta (``CALIBRATION_COPY``); its provenance."""
    data, source = calibration_bytes(follower)
    dest = dataset_root / CALIBRATION_COPY
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return {
        "file": CALIBRATION_COPY,
        "sha256": hashlib.sha256(data).hexdigest(),
        "source": source,
        "robot_id": getattr(follower, "id", None)
        or getattr(getattr(follower, "config", None), "id", None),
    }


# ------------------------------------------------------------------------------------------------
# The deploy loop's frame source: every camera, read together
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RigCapturedFrame(Frame):
    """A ``Frame`` (the policy camera's image, the oldest stamp) plus every camera's image."""

    images: dict[str, ImageU8]
    capture_times: dict[str, float]


class RigFrameSource:
    """A ``CameraRig`` as the deploy loop's ``FrameSource``. A fault on ANY camera is a lost frame
    (``CameraFault``): the deadman trips and nothing is written or recorded with a camera missing.
    """

    def __init__(self, rig: CameraRig, policy_camera: str = "front"):
        if policy_camera not in rig.names:
            raise ValueError(f"policy camera {policy_camera!r} is not in the rig {rig.names}")
        self.rig, self.policy_camera = rig, policy_camera

    def read(self) -> RigCapturedFrame:
        f: RigFrame = self.rig.read()
        return RigCapturedFrame(
            image=f.images[self.policy_camera],
            capture_time=f.capture_time,
            images=dict(f.images),
            capture_times=dict(f.capture_times),
        )


# ------------------------------------------------------------------------------------------------
# Evaluation capture
# ------------------------------------------------------------------------------------------------
class EvalCapture:
    """One evaluation session's LeRobotDataset (module doc)."""

    def __init__(
        self,
        dataset: Any,
        root: Path,
        provenance: dict[str, Any],
        *,
        end_state: Callable[[], Mapping[str, ImageU8] | None] | None = None,
    ):
        self.dataset = dataset
        self.root = Path(root)  # the capture dir: dataset/, end_state/
        self.provenance = provenance
        self.provenance.setdefault("episodes", {})
        self.end_state = end_state
        self._open: dict[str, Any] | None = None  # trial_id, task, frames, error
        self.closed = False

    @property
    def dataset_root(self) -> Path:
        return Path(self.dataset.root)

    @classmethod
    def open(
        cls,
        root: Path,
        repo_id: str,
        provenance: dict[str, Any],
        *,
        resume: bool,
        follower: Any,
        end_state: Callable[[], Mapping[str, ImageU8] | None] | None = None,
    ) -> EvalCapture:
        ds_root = Path(root) / "dataset"
        if resume and ds_root.exists():
            try:
                ds = open_dataset(repo_id, ds_root)
            except Exception as e:
                raise RuntimeError(
                    f"cannot reopen the session's capture dataset {ds_root} ({type(e).__name__}: "
                    f"{e}); a run killed mid-session leaves it unfinalized. The trial records are "
                    "intact; the capture of this session cannot be continued"
                ) from e
            prov = json.loads((ds_root / PROVENANCE).read_text())
            if prov.get("session_id") != provenance.get("session_id"):
                raise RuntimeError(f"{ds_root} belongs to session {prov.get('session_id')!r}")
            prov.setdefault("runs", []).extend(provenance.get("runs", []))
            cap = cls(ds, Path(root), prov, end_state=end_state)
        else:
            if ds_root.exists():
                raise FileExistsError(f"{ds_root} exists: a new session never reuses a dataset")
            ds_root.parent.mkdir(parents=True, exist_ok=True)
            ds = create_dataset(repo_id, ds_root)
            provenance["calibration"] = copy_calibration(follower, ds_root)
            cap = cls(ds, Path(root), provenance, end_state=end_state)
        cap._write_provenance()
        return cap

    def _write_provenance(self) -> None:
        write_json(self.dataset_root / PROVENANCE, self.provenance)

    def begin(self, trial_id: str, task: str, meta: Mapping[str, Any] | None = None) -> None:
        """Start a trial's episode (an unfinished one is discarded first)."""
        self._discard()
        self._open = {
            "trial_id": trial_id,
            "task": task,
            "frames": 0,
            "error": None,
            "meta": dict(meta or {}),
        }

    def on_write(self, frame: Frame | RigFrame | None, out: Any) -> None:
        """One policy write: (both camera images, measured state, written command). Never raises.
        ``frame`` is a ``RigCapturedFrame`` (``RigFrameSource``: the gRPC path) or a ``RigFrame``
        (a ``CameraRig`` read directly: the in-process SmolVLA / ACT path); both carry every
        camera's image, the same frame the policy was given."""
        cur = self._open
        if cur is None or cur["error"] is not None:
            return
        try:
            images = getattr(frame, "images", None)
            if images is None:
                raise conv.ConventionError("the frame source does not deliver every camera")
            row = conv.recording_frame(images, out.measured, out.safety.command, cur["task"])
            from src.data.recorder import datasets_write_compat

            with datasets_write_compat():
                self.dataset.add_frame(row)
            cur["frames"] += 1
        except Exception as e:
            log.exception("eval capture: frame not recorded")
            cur["error"] = f"{type(e).__name__}: {e}"

    def _discard(self) -> None:
        if self._open is not None and self._open["frames"]:
            try:
                self.dataset.clear_episode_buffer()
            except Exception:  # noqa: BLE001 - e.g. after a failed save popped buffer keys
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self._open = None

    def _stills(self, episode_index: int) -> dict[str, str] | str:
        if self.end_state is None:
            return "no end-state source"
        try:
            images = self.end_state()
        except CameraFault as e:
            return f"camera fault: {e}"
        if images is None:
            return "no frame"
        import cv2

        out = {}
        d = self.root / END_STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        for name, img in images.items():
            p = d / f"episode_{episode_index:06d}_{name}.png"
            cv2.imwrite(str(p), cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR))
            out[name] = p.relative_to(self.root).as_posix()
        return out

    def finish(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Save (policy started, >= 1 frame, no capture error) or discard the open episode; the
        trial record's ``capture`` block. Never raises."""
        cur, self._open = self._open, None
        if cur is None:
            return {"episode_index": None, "frames": 0, "reason": "no capture was open"}
        base = {"trial_id": cur["trial_id"], "frames": cur["frames"], "dataset": "dataset"}
        why = None
        if not rec.get("policy_started"):
            why = "not started: an attempt, never an episode"
        elif cur["error"] is not None:
            why = f"capture error: {cur['error']}"
        elif cur["frames"] == 0:
            why = "no policy write was recorded"
        if why is not None:
            self._open = cur
            self._discard()
            return {**base, "episode_index": None, "reason": why, "error": cur["error"]}
        try:
            from src.data.recorder import datasets_write_compat

            index = int(self.dataset.num_episodes)
            with datasets_write_compat():
                self.dataset.save_episode()
        except Exception as e:
            log.exception("eval capture: episode not saved")
            self._open = cur
            self._discard()
            return {
                **base,
                "episode_index": None,
                "reason": "save failed",
                "error": f"{type(e).__name__}: {e}",
            }
        stills = self._stills(index)
        self.provenance["episodes"][str(index)] = {
            "trial_id": cur["trial_id"],
            "frames": cur["frames"],
            "end_state": stills,
            **cur["meta"],
        }
        try:
            self._write_provenance()
        except OSError as e:
            log.error("eval capture: provenance not written: %s", e)
        return {**base, "episode_index": index, "end_state": stills}

    def close(self) -> None:
        """Discard an unfinished episode, finalize the dataset (Parquet footers), write the
        provenance. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self._discard()
        try:
            self.dataset.finalize()
        finally:
            self._write_provenance()
