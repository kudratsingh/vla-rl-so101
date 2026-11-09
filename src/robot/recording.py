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
* ``begin`` opens a trial's episode buffer. ``on_settled`` (``DeployLoop.on_settled``, called at
  the end of the settling interval, BEFORE the score prompt) reads one frame of every camera:
  the end state the rubric judges, kept for the PNG stills of the blinded clip review (a trial
  that ended in a fault has no settle: its stills are read when it is saved, and say so).
  ``finish`` runs AFTER the scored trial record is logged and flushed (``scripts/wo07_real_eval.py``
  logs the capture block in a follow-up ``capture`` row): it saves the episode when the policy
  started and at least one frame was written, else discards it, and returns the ``capture``
  block: ``episode_index`` (None when nothing was saved, with the reason), ``frames``,
  ``trial_id``, the achieved write rate and the end-state stills.
* Every frame carries ``timing_s`` (convention 1.1): the real times of its command write and of
  its camera grab, on the loop's clock. The dataset's ``timestamp`` is LeRobot's nominal ``frame_index / fps``; an
  in-process SmolVLA writes ~5 actions, then pauses for the next chunk, so the capture block
  records ``write_rate_hz`` (frames - 1 over the write-time span) next to the nominal fps.
* A capture fault NEVER reaches the arm or the score: ``on_write`` and ``finish`` catch it, the
  episode is discarded, and the record says why (``capture.error``).
* ``meta/vla_rl_so101.json`` cites the convention and the protocol and maps every saved episode
  index to its trial id (written after each save); ``meta/calibration_follower.json`` is the
  follower's LeRobot calibration file as loaded for the session.
* Durable after every episode (``DurableDataset``, review F1): LeRobot writes Parquet through
  open writers whose footers exist only after ``finalize()``, and buffers episode metadata, so a
  dataset that stays open for a whole session is unreadable after a kill (SIGKILL, power loss).
  Every saved episode is therefore finalized and the dataset reopened through LeRobot's own
  resume path (``lerobot_record.py --resume``: the next episode goes to new files). A kill can
  only cost the episode being saved: ``recover_dataset`` moves what it left (a footerless data
  file, an unreferenced video, stray frame PNGs) to a quarantine folder and rewrites
  ``meta/info.json`` / ``meta/stats.json`` from the episodes that were finalized. Used by the
  evaluation capture and the teleop demo recorder alike; ``EvalCapture.open(resume=True)`` and
  ``check_capture_resumable`` (run before the arm is connected) recover first.
* Staged before the score is logged (owner 2026-11-09, the capture / resume gap): the episode is
  saved AFTER the scored record, so a kill between the two used to leave a scored trial with no
  recording (LeRobot keeps state, action and timing in memory until ``save_episode``; only the
  frame PNGs are on disk, and the recovery quarantines them). ``EvalCapture.stage`` therefore
  runs in the record hook, BEFORE the record is logged: it waits for LeRobot's PNG writer, writes
  the episode's state / action / timing (``frames.npz``), hard-links the frame PNGs (a copy if a
  link fails) and the end-state stills into ``<capture>/staged/<trial id>/``, and writes
  ``staged.json`` last (the directory is renamed into place: a journal is whole or absent). A
  successful save deletes it. On a resume, ``EvalCapture.recover_staged`` rebuilds the episode
  from it (the same frames, PNG is lossless: the same pixels the save would have encoded) and
  saves it under the NEXT free episode index, mapped to its trial id in the provenance like any
  other episode (indices follow save order, never trial order). Limits: what was never staged
  cannot be recovered: a trial the process died in before its record (``session_lost``; staged
  at the session end only when the process lived to run it), a kill while staging (no
  ``staged.json``), a capture error, or a lost / damaged journal. Such a trial is flagged
  ``recording_missing`` (scripts/wo07_real_eval.py ``reconcile_captures``) and its export is
  refused as complete (``src/eval/export_bundle.py``). A recovery that fails while the journal is
  whole (e.g. a transient save error) is NOT flagged: the journal stays and the next resume
  retries it (``recovery_failed``). A started trial that no policy write reached (frames 0, no
  capture error: the first write faulted) has no recording to miss (``recording_required``).
* Retracted starts (PR #33, ``DeployLoop._first_goal_check``): an attempt whose first goal was
  blocked after its ``trial_started`` marker never started (``not_started``): no policy write
  reached the capture, nothing is staged (``stage`` needs a started record), ``finish`` discards
  its open buffer and it never needs a recording. A journal records its attempt (cell,
  trial_idx, start_utc, attempt): ``recover_staged`` discards one that names another attempt of
  the slot (``STAGED_OTHER_ATTEMPT``), and the resume's reconcile discards any journal of a slot
  none of whose attempts started (``STAGED_NOT_STARTED``), so a not-started attempt's frames can
  never be attached to a later attempt of the same slot.
* Power loss (S32-D2): a saved episode is synced to stable storage (``src.runs.sync_tree``:
  every dataset file the save wrote except the frame PNG folder, then every directory;
  F_FULLFSYNC on macOS) BEFORE its staged journal is deleted and before ``finish`` returns, so
  the scored record's ``capture`` block (and the synced record.json that holds it) never names
  an episode the page cache still held. Between trials, never during motion; ``sync_s`` in the
  block says what it cost. A failed sync keeps the journal and says so (``sync_error``).
  ``write_json`` (provenance, journals) is atomic and synced. Should an episode a record names
  still be gone after a power loss, the resume's reconcile treats the trial as having none
  (scripts/wo07_real_eval.py ``reconcile_captures``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.robot import convention as conv
from src.robot.camera import CameraFault, CameraRig, Frame, ImageU8, RigFrame
from src.runs import sync_path, sync_tree, write_durable

log = logging.getLogger(__name__)

PROVENANCE = "meta/vla_rl_so101.json"
CALIBRATION_COPY = "meta/calibration_follower.json"
END_STATE_DIR = "end_state"
STAGED_DIR = "staged"  # <capture>/staged/<trial id>/: an episode not yet saved (module doc)
STAGED_INDEX = "staged.json"  # written last: its presence marks a whole journal
STAGED_FRAMES = "frames.npz"
RECOVERED_FROM_STAGE = (
    "staged capture: the run ended after the scored record and before the episode was saved; "
    "rebuilt from the journal written before the record (same frames, same pixels)"
)


NO_POLICY_WRITE = "no policy write was recorded"  # nothing reached the capture: no episode
STAGED_UNREADABLE = "staged journal unreadable"  # damaged: flagged, never retried or guessed
STAGED_OTHER_ATTEMPT = "staged journal of another attempt of the slot (discarded)"
STAGED_NOT_STARTED = "staged journal of a slot no attempt of which started (discarded)"
JOURNAL_KEYS = ("cell", "trial_idx", "start_utc", "attempt")  # the attempt a journal belongs to
SCORED_OUTCOMES = ("success", "failure")


def recording_required(t: Mapping[str, Any]) -> bool:
    """A scored trial (its policy started) needs its episode: the export refuses without one
    (owner 2026-11-09). Except a trial no policy write ever reached the capture of (frames 0,
    no capture error: e.g. the first write faulted on the bus before the frame was taken): it
    cannot have a recording, and is listed as an uncaptured started trial instead. The ONE rule
    of ``scripts/wo07_real_eval.py`` (reconcile / gaps) and ``src/eval/export_bundle.py``."""
    if not t.get("policy_started") or t.get("outcome") not in SCORED_OUTCOMES:
        return False
    cap = t.get("capture") or {}
    if cap.get("episode_index") is not None:
        return True
    # the save's verdict, or the stage's (a kill between the record and the save: the stage ran
    # in the record hook, after the trial's last write)
    return not any(
        b.get("reason") == NO_POLICY_WRITE and not b.get("frames") and not b.get("error")
        for b in (cap, cap.get("stage") or {})
    )


def journal_mismatch(journal: Mapping[str, Any], expect: Mapping[str, Any] | None) -> str:
    """What differs between a staged journal's attempt (the ``JOURNAL_KEYS`` it records) and
    ``expect``'s; empty when they match (a journal written before ``attempt`` was recorded is
    matched on the rest)."""
    if expect is None:
        return ""
    diff = [
        f"{k} {journal[k]!r} != {expect.get(k)!r}"
        for k in JOURNAL_KEYS
        if k in journal and journal[k] != expect.get(k)
    ]
    return "; ".join(diff)


def staged_name(trial_id: str) -> str:
    """The staged journal's directory name of a trial id (``<session>/<cell>/<idx>``)."""
    return str(trial_id).replace("/", "__")


def staged_path(capture_root: Path, trial_id: str) -> Path:
    return Path(capture_root) / STAGED_DIR / staged_name(trial_id)


def has_staged(capture_root: Path, trial_id: str) -> bool:
    """A whole staged journal exists for this trial (``staged.json`` is written last)."""
    return (staged_path(capture_root, trial_id) / STAGED_INDEX).is_file()


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


class CheckpointFailed(RuntimeError):
    """The episode WAS saved (``index``) but finalizing / reopening the dataset failed."""

    def __init__(self, index: int, error: BaseException):
        super().__init__(f"episode {index} saved, then the checkpoint failed: {error}")
        self.index = index


class DurableDataset:
    """A recording ``LeRobotDataset`` that is finalized after every saved episode (module doc).

    Everything else is the wrapped dataset's (attribute reads and writes pass through), so the
    recorders call ``add_frame`` / ``episode_buffer`` / ``num_episodes`` as before."""

    def __init__(self, dataset: Any, repo_id: str):
        object.__setattr__(self, "_ds", dataset)
        object.__setattr__(self, "repo_id", repo_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_ds"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._ds, name, value)

    @property
    def inner(self) -> Any:
        return self._ds

    def save_episode(self, before_checkpoint: Callable[[int], None] | None = None) -> int:
        """Save the buffered episode, run ``before_checkpoint(index)`` (the provenance entry),
        then finalize and reopen. Returns the episode's index. A failure of the save itself
        raises before anything is checkpointed (the caller discards the buffer)."""
        from src.data.recorder import datasets_write_compat

        index = int(self._ds.num_episodes)
        with datasets_write_compat():
            self._ds.save_episode()
        try:
            if before_checkpoint is not None:
                before_checkpoint(index)
            self.checkpoint()
        except Exception as e:
            raise CheckpointFailed(index, e) from e
        return index

    def checkpoint(self) -> None:
        """Close every Parquet writer (footers), then reopen through LeRobot's resume path."""
        ds = self._ds
        ds.finalize()
        ds.stop_image_writer()
        object.__setattr__(self, "_ds", open_dataset(self.repo_id, Path(ds.root)))

    def close(self) -> None:
        """Finalize (Parquet footers) and stop the PNG writer threads. Idempotent."""
        self._ds.finalize()
        stop = getattr(self._ds, "stop_image_writer", None)
        if stop is not None:
            stop()


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def recover_dataset(root: Path, quarantine: Path | None = None) -> dict[str, Any]:
    """Make a recording dataset a kill left mid-save readable again (module doc): unreadable
    episode-metadata files, data / video files no finalized episode references, and the frame
    PNG folder are moved to ``quarantine``; ``meta/info.json`` totals and ``meta/stats.json`` are
    rebuilt from the finalized episodes. Returns what it did (``moved`` empty = nothing to do).
    Raises ``RuntimeError`` when a file a finalized episode needs is itself unreadable."""
    import numpy as np
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.utils import write_info, write_stats

    root = Path(root)
    quarantine = (
        Path(quarantine) if quarantine else root.parent / f"{root.name}.recovered-{_utc_stamp()}"
    )
    info = json.loads((root / "meta" / "info.json").read_text())
    moved: list[str] = []

    def stash(path: Path) -> None:
        rel = path.relative_to(root)
        dest = quarantine / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        moved.append(rel.as_posix())

    episodes: list[dict[str, Any]] = []
    for f in sorted((root / "meta" / "episodes").rglob("*.parquet")):
        try:
            episodes += pq.read_table(f).to_pylist()
        except Exception:  # noqa: BLE001 - a footerless file: its episode was never finalized
            stash(f)
    episodes.sort(key=lambda e: int(e["episode_index"]))
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    keep = {
        str(info["data_path"]).format(
            chunk_index=e["data/chunk_index"], file_index=e["data/file_index"]
        )
        for e in episodes
    }
    for e in episodes:
        for k in video_keys:
            keep.add(
                str(info["video_path"]).format(
                    video_key=k,
                    chunk_index=e[f"videos/{k}/chunk_index"],
                    file_index=e[f"videos/{k}/file_index"],
                )
            )
    for sub in ("data", "videos"):
        for f in sorted((root / sub).rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(root).as_posix()
            if rel not in keep:
                stash(f)
            elif f.suffix == ".parquet":
                try:
                    pq.read_metadata(f)
                except Exception as e:
                    raise RuntimeError(
                        f"{rel} holds finalized episodes but is unreadable ({e}): not recoverable"
                    ) from e
    if (root / "images").exists():
        moved.append("images/")
        quarantine.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root / "images"), str(quarantine / "images"))
    n, frames = len(episodes), sum(int(e["length"]) for e in episodes)
    if moved or info.get("total_episodes") != n or info.get("total_frames") != frames:
        info.update(total_episodes=n, total_frames=frames, splits={"train": f"0:{n}"})
        write_info(info, root)
        stats_path = root / "meta" / "stats.json"
        if n:
            per_episode = []
            for ep in episodes:
                st: dict[str, dict[str, Any]] = {}
                for key, value in ep.items():
                    if key.startswith("stats/"):
                        _, feat, stat = key.split("/", 2)
                        st.setdefault(feat, {})[stat] = np.asarray(value)
                per_episode.append(st)
            write_stats(aggregate_stats(per_episode), root)
        elif stats_path.exists():
            stash(stats_path)
    return {
        "moved": moved,
        "quarantine": str(quarantine) if moved else None,
        "episodes": n,
        "frames": frames,
    }


def prune_provenance(prov: dict[str, Any], num_episodes: int) -> list[str]:
    """Drop provenance entries of episode indices the dataset no longer holds (a save that a
    kill interrupted, removed by ``recover_dataset``). Returns the dropped indices."""
    eps = prov.get("episodes") or {}
    gone = sorted((k for k in eps if int(k) >= num_episodes), key=int)
    for k in gone:
        del eps[k]
    return gone


def write_json(path: Path, value: Any) -> None:
    """Atomic, synced JSON write (``src.runs.write_durable``: temp file + sync + rename +
    directory sync): neither a kill nor a power loss leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_durable(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


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
        dataset: DurableDataset,
        root: Path,
        provenance: dict[str, Any],
        *,
        end_state: Callable[[], Mapping[str, ImageU8] | None] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.dataset = dataset
        self.root = Path(root)  # the capture dir: dataset/, end_state/
        self.provenance = provenance
        self.provenance.setdefault("episodes", {})
        self.end_state = end_state
        self.clock = clock  # the deploy loop's: write times of the fake clock are exact
        self._open: dict[str, Any] | None = None  # trial_id, task, frames, error, times, stills
        self.closed = False
        self.broken: str | None = None  # a failed checkpoint: nothing more is captured

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
        clock: Callable[[], float] = time.perf_counter,
    ) -> EvalCapture:
        ds_root = Path(root) / "dataset"
        if resume and ds_root.exists():
            prov, recovery = recover_capture(root, provenance.get("session_id"))
            if recovery["episodes"] == 0:  # killed during its first save: start the dataset over
                stash = Path(root) / f"dataset.empty-{_utc_stamp()}"
                shutil.move(str(ds_root), str(stash))
                ds = DurableDataset(create_dataset(repo_id, ds_root), repo_id)
                prov["calibration"] = copy_calibration(follower, ds_root)
                recovery["restarted_empty_dataset"] = str(stash)
            else:
                try:
                    ds = DurableDataset(open_dataset(repo_id, ds_root), repo_id)
                except Exception as e:
                    raise RuntimeError(
                        f"cannot reopen the session's capture dataset {ds_root} after recovery "
                        f"({type(e).__name__}: {e})"
                    ) from e
            prov.setdefault("runs", []).extend(provenance.get("runs", []))
            if recovery["moved"] or recovery.get("pruned"):
                prov.setdefault("recoveries", []).append(recovery)
            cap = cls(ds, Path(root), prov, end_state=end_state, clock=clock)
        else:
            if ds_root.exists():
                raise FileExistsError(f"{ds_root} exists: a new session never reuses a dataset")
            ds_root.parent.mkdir(parents=True, exist_ok=True)
            ds = DurableDataset(create_dataset(repo_id, ds_root), repo_id)
            provenance["calibration"] = copy_calibration(follower, ds_root)
            cap = cls(ds, Path(root), provenance, end_state=end_state, clock=clock)
        cap._write_provenance()
        return cap

    def _write_provenance(self) -> None:
        write_json(self.dataset_root / PROVENANCE, self.provenance)

    @property
    def open_episode(self) -> tuple[str, int] | None:
        """(trial id, frames) of the episode being recorded, None when none is open."""
        cur = self._open
        return None if cur is None else (str(cur["trial_id"]), int(cur["frames"]))

    def begin(self, trial_id: str, task: str, meta: Mapping[str, Any] | None = None) -> None:
        """Start a trial's episode (an unfinished one is discarded first)."""
        self._discard()
        self._open = {
            "trial_id": trial_id,
            "task": task,
            "frames": 0,
            "error": None,
            "meta": dict(meta or {}),
            "t0": None,
            "t_last": None,
            "end_images": None,
            "end_state_when": None,
            "recovered": None,
        }

    def on_write(self, frame: Frame | RigFrame | None, out: Any) -> None:
        """One policy write: (both camera images, measured state, written command). Never raises.
        ``frame`` is a ``RigCapturedFrame`` (``RigFrameSource``: the gRPC path) or a ``RigFrame``
        (a ``CameraRig`` read directly: the in-process SmolVLA / ACT path); both carry every
        camera's image, the same frame the policy was given."""
        cur = self._open
        if cur is None or cur["error"] is not None:
            return
        if self.broken is not None:
            cur["error"] = f"capture stopped after a failed checkpoint: {self.broken}"
            return
        try:
            t = float(self.clock())
            if cur["t0"] is None:
                cur["t0"] = t
            images = getattr(frame, "images", None)
            if images is None:
                raise conv.ConventionError("the frame source does not deliver every camera")
            grabbed = getattr(frame, "capture_time", None)
            row = conv.recording_frame(
                images,
                out.measured,
                out.safety.command,
                cur["task"],
                write_time_s=t - cur["t0"],
                frame_time_s=(float(grabbed) if grabbed is not None else t) - cur["t0"],
            )
            from src.data.recorder import datasets_write_compat

            with datasets_write_compat():
                self.dataset.add_frame(row)
            cur["frames"] += 1
            cur["t_last"] = t
        except Exception as e:
            log.exception("eval capture: frame not recorded")
            cur["error"] = f"{type(e).__name__}: {e}"

    def _grab_end_state(self) -> Mapping[str, ImageU8] | str:
        if self.end_state is None:
            return "no end-state source"
        try:
            images = self.end_state()
        except CameraFault as e:
            return f"camera fault: {e}"
        except Exception as e:  # noqa: BLE001 - never reaches the arm or the score
            return f"{type(e).__name__}: {e}"
        return "no frame" if images is None else dict(images)

    def on_settled(self) -> None:
        """``DeployLoop.on_settled``: the end state the rubric judges, read at the end of the
        settling interval, before the score prompt. Never raises."""
        cur = self._open
        if cur is None:
            return
        cur["end_images"] = self._grab_end_state()
        cur["end_state_when"] = "end of the settling interval, before the score prompt"

    def _discard(self) -> None:
        if self._open is not None and self._open["frames"]:
            try:
                self.dataset.clear_episode_buffer()
            except Exception:  # noqa: BLE001 - e.g. after a failed save popped buffer keys
                self.dataset.episode_buffer = self.dataset.create_episode_buffer()
        self._open = None

    def _stills(
        self, episode_index: int, images: Mapping[str, ImageU8] | str
    ) -> dict[str, str] | str:
        if isinstance(images, str):
            return images
        import cv2

        out = {}
        d = self.root / END_STATE_DIR
        d.mkdir(parents=True, exist_ok=True)
        for name, img in images.items():
            p = d / f"episode_{episode_index:06d}_{name}.png"
            cv2.imwrite(str(p), cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR))
            out[name] = p.relative_to(self.root).as_posix()
        return out

    # -- the staged journal (module doc) ----------------------------------------------------------
    def stage(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Journal the open episode to ``<capture>/staged/<trial id>/`` BEFORE its scored record
        is logged (the record hook), so a kill before the save cannot lose it. Only a started
        trial with >= 1 frame and no capture error is staged. Never raises: returns
        ``{"staged": <rel dir> | None, ...}`` for the record."""
        cur = self._open
        if cur is None:
            return {"staged": None, "reason": "no capture was open"}
        if not rec.get("policy_started"):
            return {"staged": None, "reason": "not started: nothing to save"}
        if cur["error"] is not None:
            return {
                "staged": None,
                "reason": f"capture error: {cur['error']}",
                "frames": int(cur["frames"]),
                "error": cur["error"],
            }
        if cur["frames"] == 0:  # the trial is over (record hook): no write will follow
            return {"staged": None, "reason": NO_POLICY_WRITE, "frames": 0}
        if self.broken is not None:
            return {"staged": None, "reason": f"capture stopped: {self.broken}"}
        t0 = time.perf_counter()
        final = staged_path(self.root, cur["trial_id"])
        tmp = final.with_name(f".tmp-{final.name}")
        try:
            wait = getattr(self.dataset, "_wait_image_writer", None)
            if wait is not None:
                wait()  # every frame PNG is on disk
            buf = self.dataset.episode_buffer
            n = int(buf["size"])
            if n != int(cur["frames"]):
                raise RuntimeError(
                    f"episode buffer holds {n} frames, the capture counted {cur['frames']}"
                )
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            np.savez(
                tmp / STAGED_FRAMES,
                state=np.stack(buf[conv.STATE_KEY]).astype(np.float32),
                action=np.stack(buf[conv.ACTION_KEY]).astype(np.float32),
                timing=np.stack(buf[conv.TIMING_KEY]).astype(np.float32),
            )
            linked = 0
            for name, key in conv.IMAGE_KEYS.items():
                paths = [Path(p) for p in buf[key]]
                if len(paths) != n:
                    raise RuntimeError(f"{key}: {len(paths)} frame files for {n} frames")
                d = tmp / "images" / name
                d.mkdir(parents=True)
                for i, src in enumerate(paths):
                    dst = d / f"frame-{i:06d}.png"
                    try:
                        os.link(src, dst)
                        linked += 1
                    except OSError:
                        shutil.copyfile(src, dst)
            end = cur["end_images"]
            if isinstance(end, Mapping):
                import cv2

                (tmp / END_STATE_DIR).mkdir()
                for name, img in end.items():
                    cv2.imwrite(
                        str(tmp / END_STATE_DIR / f"{name}.png"),
                        cv2.cvtColor(np.asarray(img, np.uint8), cv2.COLOR_RGB2BGR),
                    )
            tasks = sorted({str(t) for t in buf["task"]})
            if len(tasks) != 1:
                raise RuntimeError(f"an episode has one task string, got {tasks}")
            write_json(
                tmp / STAGED_INDEX,
                {
                    "trial_id": cur["trial_id"],
                    "task": tasks[0],
                    "frames": n,
                    "cameras": list(conv.CAMERA_NAMES),
                    "meta": cur["meta"],
                    "t0": cur["t0"],
                    "t_last": cur["t_last"],
                    "end_state": "end_state" if isinstance(end, Mapping) else end,
                    "end_state_when": cur["end_state_when"],
                    "record": {k: rec.get(k) for k in JOURNAL_KEYS},
                    "staged_utc": _utc_stamp(),
                },
            )
            if final.exists():
                shutil.rmtree(final)
            os.rename(tmp, final)
        except Exception as e:  # never reaches the arm or the score
            log.exception("eval capture: staging failed")
            shutil.rmtree(tmp, ignore_errors=True)
            return {"staged": None, "error": f"{type(e).__name__}: {e}"}
        return {
            "staged": final.relative_to(self.root).as_posix(),
            "frames": n,
            "hardlinked_frames": linked,
            "stage_s": round(time.perf_counter() - t0, 3),
        }

    def drop_staged(self, trial_id: str) -> None:
        shutil.rmtree(staged_path(self.root, trial_id), ignore_errors=True)

    def recover_staged(
        self, trial_id: str, expect: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Rebuild and save a staged episode a kill left unsaved (module doc); its ``capture``
        block (``episode_index`` None with the reason when it cannot be recovered). ``expect``:
        the trial record it is recovered for; a journal of another attempt (``JOURNAL_KEYS``,
        those the journal records) is discarded, never attached. Never raises."""
        d = staged_path(self.root, trial_id)
        if not (d / STAGED_INDEX).is_file():
            why = "no staged journal" if not d.exists() else "staged journal incomplete"
            return {"trial_id": trial_id, "episode_index": None, "reason": why}
        try:
            from PIL import Image

            meta = json.loads((d / STAGED_INDEX).read_text())
            if meta.get("trial_id") != trial_id:
                raise RuntimeError(f"journal names trial {meta.get('trial_id')!r}")
            other = journal_mismatch(meta.get("record") or {}, expect)
            if other:
                self.drop_staged(trial_id)
                return {
                    "trial_id": trial_id,
                    "episode_index": None,
                    "reason": STAGED_OTHER_ATTEMPT,
                    "error": other,
                }
            n = int(meta["frames"])
            with np.load(d / STAGED_FRAMES) as z:
                state, action, timing = z["state"], z["action"], z["timing"]
            if not (len(state) == len(action) == len(timing) == n):
                raise RuntimeError("frames.npz does not hold every frame")
            self.begin(trial_id, str(meta["task"]), meta.get("meta") or {})
            cur = self._open
            assert cur is not None
            from src.data.recorder import datasets_write_compat

            for i in range(n):
                images = {}
                for name in conv.CAMERA_NAMES:
                    with Image.open(d / "images" / name / f"frame-{i:06d}.png") as im:
                        images[name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
                row = conv.recording_frame(
                    images,
                    state[i],
                    action[i],
                    str(meta["task"]),
                    write_time_s=float(timing[i, 0]),
                    frame_time_s=float(timing[i, 1]),
                )
                with datasets_write_compat():
                    self.dataset.add_frame(row)
                cur["frames"] += 1
            cur["t0"], cur["t_last"] = meta.get("t0"), meta.get("t_last")
            end = meta.get("end_state")
            if end == "end_state":
                stills = {}
                for name in conv.CAMERA_NAMES:
                    p = d / END_STATE_DIR / f"{name}.png"
                    if p.is_file():
                        with Image.open(p) as im:
                            stills[name] = np.asarray(im.convert("RGB"), dtype=np.uint8)
                cur["end_images"] = stills or "staged stills missing"
            else:
                cur["end_images"] = end if isinstance(end, str) else "no end-state stills staged"
            cur["end_state_when"] = meta.get("end_state_when")
            cur["recovered"] = RECOVERED_FROM_STAGE
        except Exception as e:  # a damaged journal: flagged, never guessed
            log.exception("eval capture: staged episode not recoverable")
            self._discard()
            return {
                "trial_id": trial_id,
                "episode_index": None,
                "reason": STAGED_UNREADABLE,
                "error": f"{type(e).__name__}: {e}",
            }
        return self.finish({"policy_started": True})

    def finish(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Save (policy started, >= 1 frame, no capture error) or discard the open episode; the
        trial record's ``capture`` block. Called after the scored record is logged. Never
        raises. A saved episode's staged journal is deleted; a failed save keeps it (a resume
        recovers it)."""
        cur, self._open = self._open, None
        if cur is None:
            return {"episode_index": None, "frames": 0, "reason": "no capture was open"}
        base = {"trial_id": cur["trial_id"], "frames": cur["frames"], "dataset": "dataset"}
        if cur.get("recovered"):
            base["recovered"] = cur["recovered"]
        why = None
        if not rec.get("policy_started"):
            why = "not started: an attempt, never an episode"
        elif cur["error"] is not None:
            why = f"capture error: {cur['error']}"
        elif cur["frames"] == 0:
            why = NO_POLICY_WRITE
        if why is not None:
            self._open = cur
            self._discard()
            return {**base, "episode_index": None, "reason": why, "error": cur["error"]}
        end_images = cur["end_images"]
        when = cur["end_state_when"]
        if end_images is None:  # no settle (a fault ended the trial): read it now, say so
            end_images = self._grab_end_state()
            when = "after the trial record (no settling interval: the trial ended in a fault)"
        span = None
        if cur["t0"] is not None and cur["t_last"] is not None:
            span = float(cur["t_last"]) - float(cur["t0"])
        timing = {
            "nominal_fps": conv.FPS,
            "write_time_span_s": None if span is None else round(span, 4),
            "write_rate_hz": (
                round((cur["frames"] - 1) / span, 3) if span and cur["frames"] > 1 else None
            ),
        }
        entry = {"trial_id": cur["trial_id"], "frames": cur["frames"], **timing, **cur["meta"]}
        if cur.get("recovered"):
            entry["recovered"] = cur["recovered"]

        def provenance_entry(index: int) -> None:  # before the checkpoint: a kill keeps the map
            self.provenance["episodes"][str(index)] = entry
            self._write_provenance()

        t_save = time.time()
        try:
            index = self.dataset.save_episode(before_checkpoint=provenance_entry)
        except CheckpointFailed as e:  # saved, but the dataset could not be made durable
            log.exception("eval capture: checkpoint failed")
            self.broken = str(e)
            return {**base, **timing, "episode_index": e.index, "error": str(e)}
        except Exception as e:
            log.exception("eval capture: episode not saved")
            self._open = cur
            try:
                self._discard()
            except Exception:
                log.exception("eval capture: discarding the unsaved episode failed")
            staged = has_staged(self.root, cur["trial_id"])
            return {
                **base,
                **timing,
                "episode_index": None,
                "reason": "save failed" + ("; staged: a resume recovers it" if staged else ""),
                "error": f"{type(e).__name__}: {e}",
            }
        t_sync = time.perf_counter()
        durable: dict[str, Any] = {}
        try:  # on stable storage before the journal goes and the record names it (module doc)
            durable["synced_files"] = sync_tree(
                self.dataset_root, skip=frozenset({"images"}), since=t_save
            )
            self.drop_staged(cur["trial_id"])  # saved, synced, in the provenance: done
        except OSError as e:
            log.error("eval capture: episode %d saved but not synced: %s", index, e)
            durable["sync_error"] = f"{type(e).__name__}: {e} (the staged journal is kept)"
        durable["sync_s"] = round(time.perf_counter() - t_sync, 4)
        stills = self._stills(index, end_images)
        entry.update(end_state=stills, end_state_when=when)
        try:
            self._write_provenance()
            if isinstance(stills, dict):
                for rel in stills.values():
                    sync_path(self.root / rel)
                sync_path(self.root / END_STATE_DIR)
        except OSError as e:
            log.error("eval capture: provenance / stills not written: %s", e)
        return {
            **base,
            **timing,
            "episode_index": index,
            "end_state": stills,
            "end_state_when": when,
            **durable,
        }

    def close(self) -> None:
        """Discard an unfinished episode, finalize the dataset (Parquet footers), write the
        provenance. Idempotent."""
        if self.closed:
            return
        self.closed = True
        self._discard()
        try:
            self.dataset.close()
        finally:
            self._write_provenance()


def recover_capture(root: Path, session_id: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """(provenance, recovery report) of a session's capture dir: ``recover_dataset`` on its
    dataset (quarantine under ``<capture>/recovered-<utc>``), provenance entries of dropped
    episodes pruned. Refuses a capture of another session."""
    ds_root = Path(root) / "dataset"
    prov = json.loads((ds_root / PROVENANCE).read_text())
    if prov.get("session_id") != session_id:
        raise RuntimeError(f"{ds_root} belongs to session {prov.get('session_id')!r}")
    report = recover_dataset(ds_root, Path(root) / f"recovered-{_utc_stamp()}")
    report["pruned"] = prune_provenance(prov, int(report["episodes"]))
    if report["moved"] or report["pruned"]:
        write_json(ds_root / PROVENANCE, prov)
    return prov, report


def check_capture_resumable(root: Path, repo_id: str, session_id: Any) -> dict[str, Any] | None:
    """Before the arm is connected (a resumed hardware session): recover the capture and make
    sure LeRobot reopens it. None when the session has no capture yet. Raises on failure."""
    ds_root = Path(root) / "dataset"
    if not ds_root.exists():
        return None
    _, report = recover_capture(root, session_id)
    if report["episodes"]:
        ds = open_dataset(repo_id, ds_root)
        ds.stop_image_writer()
    return report
