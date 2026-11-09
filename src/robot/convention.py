"""The frozen observation and action convention of the real SO-101 track (SmolVLA and ACT, WO-07).

ONE place for what a real demo, a training batch and an inference input contain. The teleop
recorder (``scripts/wo06_teleop_record.py``) cites it by ``CONVENTION_ID`` and ``convention_hash()``
in ``meta/vla_rl_so101.json``; the trainer and the deploy loop import the same names. Changing any
value in ``spec()`` changes the hash and fails ``tests/test_convention.py`` on purpose: bump
``CONVENTION_VERSION`` and re-pin instead (recorded demos keep the hash they were recorded under).

Joint vector, state and action (``observation.state`` and ``action``, float32[6])
    Order ``JOINT_NAMES`` = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
    gripper (``src/robot/units.py``; the LeRobot ``so101_follower`` motor order,
    ``lerobot/robots/so_follower/so_follower.py:55-60``). Units: RADIANS of the canonical frame,
    the joint angles of the vendored sim model ``so101_new_calib.urdf`` (``units.URDF_SOURCE``),
    for every joint. Declared to the evaluator project (vishwaroop ``docs/SO101_EXPORT.md`` §2) as
    ``rad`` per joint name: ``DECLARED_UNITS``.

    Why radians and not LeRobot's degrees or RANGE_M100_100 units:
    * smolvla_base and ACT normalize state and action with MEAN_STD (``smolvla_base``
      ``config.json`` ``normalization_mapping``; ``lerobot/policies/smolvla/
      configuration_smolvla.py:35-40``; ``lerobot/policies/act/configuration_act.py:89-94``), and a
      fine-tune normalizes with the FINE-TUNE dataset's own stats (``lerobot/scripts/
      lerobot_train.py:262-280`` overrides the pretrained normalizer with ``dataset.meta.stats``).
      Every candidate unit is a per-joint affine map of the others (deg = rad * 180/pi; LeRobot
      0-100 gripper = affine in the URDF gripper angle; RANGE_M100_100 = affine in the raw tick),
      with a positive scale under the default ``robot.units.arm_sign`` (+1 per joint), so after
      normalization the policy sees the SAME numbers whichever unit the dataset stores. A joint
      with ``arm_sign`` -1 is MIRRORED, not rescaled: what the stored radians mean depends on
      ``robot.units`` (canonical q = deg2rad(sign * deg + offset)). Those values are calibration of
      the canonical frame, not part of this hash: every recording stores them (``unit_map`` in
      ``meta/vla_rl_so101.json``), the trainer copies them into the checkpoint's provenance
      (``src/policy/lerobot_ckpt.py`` ``PROVENANCE_FILE``, inside the checkpoint hash) and the
      deploy path refuses a checkpoint whose ``unit_map`` or convention hash differs from the live
      ones (``src/robot/local_policy.py``).
    * The pretraining data is not one unit anyway: ``lerobot/svla_so100_pickplace`` (SmolVLA
      paper data) stores pre-2025-calibration degrees (wrist_roll mean 85.8, range 56..123), and
      ``lerobot/svla_so101_pickplace`` stores RANGE_M100_100 values (shoulder_lift reaches -100.0,
      elbow_flex 99.4; both ``meta/stats.json`` on the Hub, read 2026-11-09).
    * Our driver, safety filter, deploy loop and sim all work in canonical radians already
      (``So101Driver``, ``SafetyFilter``, ``DeployLoop``, ``src/env``): storing them means no
      conversion between the dataset, the policy and the safety filter. The only conversions are
      at the bus, below.

    Gripper: the URDF ``gripper`` joint angle, linear in LeRobot's RANGE_0_100 reading:
    LeRobot 0 (closed) = ``GRIPPER_CLOSED_RAD`` = -0.174533, LeRobot 100 (open) =
    ``GRIPPER_OPEN_RAD`` = 1.74533 (the URDF limits). Larger = more open.

    Conversions, both ways, applied ONLY at the servo bus (exact inverses, ``UnitMap``):
    * ``lerobot_to_canonical`` (``UnitMap.to_canonical``): LeRobot ``{"<motor>.pos"}`` (arm in
      DEGREES, ``use_degrees=True``; gripper 0-100) -> radians. Applied in
      ``So101Driver.read_state`` (``src/robot/driver.py``; the measured state that is recorded and
      fed to the policy) and to the leader arm's reading in the teleop recorder
      (``Session.tick``).
    * ``canonical_to_lerobot`` (``UnitMap.to_lerobot``): radians -> the LeRobot action dict.
      Applied in ``So101Driver._write``, AFTER the safety filter, on every goal write.
    Nothing converts between the dataset, the trainer, the policy and the safety filter.

Action semantics
    ``action`` = ABSOLUTE joint position targets (not deltas), canonical radians, one per control
    tick. In a teleop demo it is the safety filter's command actually written on that tick (the
    leader pose clipped and rate-limited, ``StepResult.safety.command``); ``observation.state`` is
    the follower pose measured just before that write. At inference the policy's (un-normalized)
    output is the same quantity and goes through ``action_targets`` into ``So101Driver.step``
    (deadman -> ``SafetyFilter`` -> bus). Rate: ``FPS`` = 15 Hz = ``robot.control_hz`` = sim
    ``CONTROL_HZ`` (the merged safety tuning: 4 deg/step = 60 deg/s; the WO-08 sim plan keeps the
    4 deg/step limit). LeRobot's own recipe records at 30 fps (``lerobot_record.py:161``); 15 Hz
    is a deliberate departure, to keep the safety-validated tick and sim parity; changing it is a
    new convention version. Chunk sizes (actions predicted per query, ``CHUNK_SIZE``): SmolVLA
    50 (smolvla_base ``chunk_size``, the paper's §4.3), ACT 50 (= 3.3 s at 15 Hz, the horizon in
    seconds of LeRobot's ACT default of 100 steps at 30 fps, ``configuration_act.py:86``). How
    many of them are executed per query is a deploy setting, never more than the chunk.

    Timing (``TIMING_KEY`` = ``timing_s``, float32[2] = [write, camera_frame], from version 1.1):
    seconds since the episode's first command write, on the recorder's clock, at which this
    frame's command was written and at which its (oldest) camera image was grabbed. LeRobot's
    ``timestamp`` is always ``frame_index / fps`` (nominal); a demo ticks at 15 Hz, but an
    evaluation rollout writes in bursts (an in-process SmolVLA runs a few actions, then pauses
    for the next chunk), so the real timing lives here. Not an ``observation.*`` / ``action`` key:
    LeRobot never makes it a policy input. (A [1]-shaped scalar feature does not survive
    LeRobot 0.4.4's ``save_episode`` with numpy 2, hence the pair.)

Cameras
    Two, both in every frame of every demo and every eval trial (final2 hardware floor; a frame
    without both is refused): ``front`` (Logitech C920x at the sim front pose, elevated 45 deg:
    the stand-in for the paper's "top" view) and ``wrist`` (the kit's 32 mm module). Streams
    640 x 480 at 30 fps (``CAMERA_FPS``); recorded UNCROPPED at the full 480 x 640 as uint8 RGB
    (LeRobot's ``OpenCVCamera`` default color mode is RGB,
    ``lerobot/cameras/opencv/configuration_opencv.py:62``), HWC, as AV1 video
    (``VIDEO_CODEC``, LeRobot's default) decoded with torchcodec (CLAUDE.md rule 6). Feature keys
    ``observation.images.front`` and ``observation.images.wrist``. Policy input: float32 CHW in
    [0, 1] (exactly what ``LeRobotDataset`` returns and what LeRobot's
    ``prepare_observation_for_inference`` builds, ``lerobot/policies/utils.py:127-133``); SmolVLA
    then resizes with padding to 512 x 512 itself. smolvla_base expects ``camera1..3``
    (its ``config.json`` ``input_features``); ``SMOLVLA_RENAME_MAP`` renames front -> camera1,
    wrist -> camera2, exactly as the LeRobot SmolVLA guide does for its top + wrist dataset
    (``docs/source/smolvla.mdx`` on lerobot main @ 8b00dab, 2026-10-29: ``--rename_map='{
    "observation.images.top": "observation.images.camera1", "observation.images.wrist":
    "observation.images.camera2"}'``, for training and for rollout; the v0.4.4 guide has no
    rename map). Pass it as ``--rename_map`` / ``TrainPipelineConfig.rename_map``; LeRobot saves
    it in the checkpoint's preprocessor, which then renames at inference too.

Language and dataset
    Per-frame ``task`` = the protocol instruction of the trial's task and object
    (``src.eval.protocol.instruction``, the one function the recorder, ``run_trials`` and the real
    protocol's schedules call; its templates ``INSTRUCTIONS`` are hashed here); demos are PickPlace
    with the red cube: ``DEMO_TASK_STRING``.
    Dataset: LeRobot ``codebase_version`` v3.0 (what lerobot 0.4.4 writes), ``robot_type``
    ``so101_follower``, fps = ``FPS``.

The ONE preprocessing function: ``policy_observation``. The inference path calls it on a live
``RigFrame`` and the measured state; the recorder check calls it on the frames read back from the
recorded dataset (``policy_observation_from_sample``). Both produce the batch that LeRobot's policy
preprocessor (rename, tokenize, normalize) receives during training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from src.config import Config
from src.eval.protocol import INSTRUCTIONS, RED_CUBE, instruction
from src.robot.units import (
    GRIPPER_INDEX,
    JOINT_NAMES,
    NUM_JOINTS,
    URDF_LIMITS_RAD,
    URDF_SOURCE,
    FloatArray,
    UnitMap,
)

CONVENTION_ID = "so101-real-obs-action"
CONVENTION_VERSION = "1.1"  # 1.1: + timing_s (review conv-3 / F6)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"
TIMING_KEY = "timing_s"
TIMING_NAMES = ("write", "camera_frame")
JOINT_UNIT = "rad"
DECLARED_UNITS: dict[str, str] = {name: JOINT_UNIT for name in JOINT_NAMES}
GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD = URDF_LIMITS_RAD["gripper"]
ACTION_SEMANTICS = "absolute_joint_position"

FPS = 15  # Hz: control rate = dataset fps = robot.control_hz = sim CONTROL_HZ
CAMERA_FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
CHUNK_SIZE: dict[str, int] = {"smolvla": 50, "act": 50}

ROBOT_TYPE = "so101_follower"
CODEBASE_VERSION = "v3.0"
VIDEO_CODEC = "libsvtav1"
VIDEO_BACKEND = "torchcodec"
DEMO_TASK = "PickPlace"
DEMO_TASK_STRING = instruction(DEMO_TASK, RED_CUBE)


@dataclass(frozen=True)
class CameraSpec:
    name: str  # configs/real.yaml cameras.<name>
    key: str  # LeRobot feature name
    smolvla_key: str  # smolvla_base input feature it is renamed to
    mount: str


CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec(
        "front",
        "observation.images.front",
        "observation.images.camera1",
        "Logitech C920x at the sim front pose (elevated 45 deg; the paper's 'top' stand-in)",
    ),
    CameraSpec(
        "wrist",
        "observation.images.wrist",
        "observation.images.camera2",
        "SO-101 kit 32 mm module on the gripper",
    ),
)
CAMERA_NAMES: tuple[str, ...] = tuple(c.name for c in CAMERAS)
IMAGE_KEYS: dict[str, str] = {c.name: c.key for c in CAMERAS}
SMOLVLA_RENAME_MAP: dict[str, str] = {c.key: c.smolvla_key for c in CAMERAS}
IMAGE_SHAPE_HWC = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


class ConventionError(ValueError):
    """Data that does not follow this convention (refused, never repaired)."""


def spec() -> dict[str, Any]:
    """Every frozen value, JSON-serializable. ``convention_hash`` hashes exactly this."""
    return {
        "id": CONVENTION_ID,
        "version": CONVENTION_VERSION,
        "joint_names": list(JOINT_NAMES),
        "state_key": STATE_KEY,
        "action_key": ACTION_KEY,
        "units": dict(DECLARED_UNITS),
        "frame": URDF_SOURCE,
        "urdf_limits_rad": {k: list(v) for k, v in URDF_LIMITS_RAD.items()},
        "gripper": {
            "index": GRIPPER_INDEX,
            "closed_rad": GRIPPER_CLOSED_RAD,
            "open_rad": GRIPPER_OPEN_RAD,
            "lerobot": "RANGE_0_100, 0 = closed, linear",
        },
        "lerobot_side": {"arm": "DEGREES (use_degrees=True)", "gripper": "RANGE_0_100"},
        "action": {
            "semantics": ACTION_SEMANTICS,
            "recorded": "safety filter command written on the tick",
            "state": "measured just before the write",
            "fps": FPS,
            "chunk_size": dict(CHUNK_SIZE),
        },
        "cameras": [
            {"name": c.name, "key": c.key, "smolvla_key": c.smolvla_key, "mount": c.mount}
            for c in CAMERAS
        ],
        "image": {
            "shape_hwc": list(IMAGE_SHAPE_HWC),
            "stream_fps": CAMERA_FPS,
            "color": "RGB",
            "recorded": "uint8 HWC, uncropped",
            "policy_input": "float32 CHW in [0, 1]",
        },
        "timing": {
            "key": TIMING_KEY,
            "dtype": "float32",
            "names": list(TIMING_NAMES),
            "meaning": "s since the episode's first command write (recorder clock): this frame's "
            "command write, its oldest camera grab",
        },
        "smolvla_rename_map": dict(SMOLVLA_RENAME_MAP),
        "task": {"key": TASK_KEY, "templates": dict(INSTRUCTIONS), "demo": DEMO_TASK_STRING},
        "dataset": {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": ROBOT_TYPE,
            "video_codec": VIDEO_CODEC,
            "video_backend": VIDEO_BACKEND,
        },
    }


def convention_hash() -> str:
    """SHA-256 of the canonical JSON of ``spec()`` (docstring edits do not change it)."""
    blob = json.dumps(spec(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def citation() -> dict[str, str]:
    """What a recording, a training run or a deploy record cites."""
    return {
        "module": "src/robot/convention.py",
        "id": CONVENTION_ID,
        "version": CONVENTION_VERSION,
        "hash": convention_hash(),
    }


# ------------------------------------------------------------------------------------------------
# Units at the bus (the only two conversions)
# ------------------------------------------------------------------------------------------------
def lerobot_to_canonical(units: UnitMap, lerobot: Mapping[str, Any]) -> FloatArray:
    """LeRobot ``{"<motor>.pos"}`` (arm deg, gripper 0-100) -> canonical radians[6]."""
    return units.to_canonical(lerobot)


def canonical_to_lerobot(units: UnitMap, q: npt.ArrayLike) -> dict[str, float]:
    """Canonical radians[6] -> LeRobot action dict (arm deg, gripper 0-100)."""
    return units.to_lerobot(q)


def check_unit_map(units: UnitMap) -> list[str]:
    """The gripper encoding is part of the convention; arm sign / offset are calibration."""
    out = []
    if not math.isclose(units.gripper_closed_rad, GRIPPER_CLOSED_RAD, abs_tol=1e-9):
        out.append(f"gripper_closed_rad {units.gripper_closed_rad} != {GRIPPER_CLOSED_RAD}")
    if not math.isclose(units.gripper_open_rad, GRIPPER_OPEN_RAD, abs_tol=1e-9):
        out.append(f"gripper_open_rad {units.gripper_open_rad} != {GRIPPER_OPEN_RAD}")
    return out


# ------------------------------------------------------------------------------------------------
# Dataset schema
# ------------------------------------------------------------------------------------------------
def dataset_features() -> dict[str, dict[str, Any]]:
    """``LeRobotDataset.create(features=...)``: both cameras, state, action, timing."""
    names = list(JOINT_NAMES)
    feats: dict[str, dict[str, Any]] = {
        c.key: {
            "dtype": "video",
            "shape": IMAGE_SHAPE_HWC,
            "names": ["height", "width", "channels"],
        }
        for c in CAMERAS
    }
    feats[STATE_KEY] = {"dtype": "float32", "shape": (NUM_JOINTS,), "names": names}
    feats[ACTION_KEY] = {"dtype": "float32", "shape": (NUM_JOINTS,), "names": names}
    feats[TIMING_KEY] = {"dtype": "float32", "shape": (2,), "names": list(TIMING_NAMES)}
    return feats


def check_dataset_info(info: Mapping[str, Any]) -> list[str]:
    """Where a dataset's ``meta/info.json`` departs from the convention (empty = conforms)."""
    out: list[str] = []
    if info.get("codebase_version") != CODEBASE_VERSION:
        out.append(f"codebase_version {info.get('codebase_version')!r} != {CODEBASE_VERSION!r}")
    if info.get("fps") != FPS:
        out.append(f"fps {info.get('fps')!r} != {FPS}")
    if info.get("robot_type") != ROBOT_TYPE:
        out.append(f"robot_type {info.get('robot_type')!r} != {ROBOT_TYPE!r}")
    feats = info.get("features") or {}
    for key in (STATE_KEY, ACTION_KEY):
        f = feats.get(key) or {}
        if f.get("dtype") != "float32" or list(f.get("shape") or []) != [NUM_JOINTS]:
            out.append(
                f"{key} must be float32[{NUM_JOINTS}], got {f.get('dtype')} {f.get('shape')}"
            )
        if list(f.get("names") or []) != list(JOINT_NAMES):
            out.append(f"{key} names {f.get('names')} != {list(JOINT_NAMES)}")
    wt = feats.get(TIMING_KEY) or {}
    if wt.get("dtype") != "float32" or list(wt.get("names") or []) != list(TIMING_NAMES):
        out.append(
            f"{TIMING_KEY} must be float32{list(TIMING_NAMES)} (convention 1.1), got {wt or 'missing'}"
        )
    for c in CAMERAS:
        f = feats.get(c.key) or {}
        if f.get("dtype") != "video" or list(f.get("shape") or []) != list(IMAGE_SHAPE_HWC):
            out.append(f"{c.key} must be video {list(IMAGE_SHAPE_HWC)}, got {f or 'missing'}")
    extra = [
        k for k in feats if k.startswith("observation.images.") and k not in SMOLVLA_RENAME_MAP
    ]
    if extra:
        out.append(f"unexpected cameras {extra}")
    return out


def check_config(cfg: Config) -> list[str]:
    """Where ``configs/real.yaml`` would record something else (empty = conforms)."""
    out: list[str] = []
    hz = cfg.get("robot", {}).get("control_hz")
    if hz != FPS:
        out.append(f"robot.control_hz {hz!r} != {FPS}")
    cams = cfg.get("cameras") or {}
    for name in CAMERA_NAMES:
        c = cams.get(name)
        if not c or c.get("enabled", True) is False or c.get("index") is None:
            out.append(f"cameras.{name} is missing or disabled: both cameras record every frame")
            continue
        got = (c.get("width"), c.get("height"), c.get("fps"))
        if got != (IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FPS):
            out.append(
                f"cameras.{name} streams {got[0]}x{got[1]} at {got[2]} fps, convention "
                f"{IMAGE_WIDTH}x{IMAGE_HEIGHT} at {CAMERA_FPS}"
            )
    out += [f"robot.units.{p}" for p in check_unit_map(UnitMap.from_config(cfg))]
    return out


# ------------------------------------------------------------------------------------------------
# Frames: recording and the one preprocessing function
# ------------------------------------------------------------------------------------------------
def _joints(what: str, v: Any) -> npt.NDArray[np.float32]:
    a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    a = a.astype(np.float32).reshape(-1)
    if a.shape != (NUM_JOINTS,) or not np.all(np.isfinite(a)):
        raise ConventionError(f"{what} must be {NUM_JOINTS} finite values, got {np.shape(v)}")
    return a


def _missing_cameras(images: Mapping[str, Any]) -> None:
    missing = [n for n in CAMERA_NAMES if images.get(n) is None]
    extra = sorted(set(images) - set(CAMERA_NAMES))
    if missing or extra:
        raise ConventionError(
            f"a frame needs exactly the cameras {list(CAMERA_NAMES)}: missing {missing}, "
            f"unexpected {extra}"
        )


def _hwc_u8(name: str, img: Any) -> npt.NDArray[np.uint8]:
    a = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else np.asarray(img)
    if a.dtype != np.uint8 or a.shape != IMAGE_SHAPE_HWC:
        raise ConventionError(
            f"{name}: expected uint8 {IMAGE_SHAPE_HWC} RGB, got {a.dtype} {a.shape}"
        )
    return np.ascontiguousarray(a)


def recording_frame(
    images: Mapping[str, Any],
    state: Any,
    action: Any,
    task: str,
    *,
    write_time_s: float,
    frame_time_s: float,
) -> dict[str, Any]:
    """The ``LeRobotDataset.add_frame`` dict of one tick. Refuses a frame missing either camera.
    ``write_time_s`` / ``frame_time_s``: seconds since the episode's first command write at this
    tick's write (>= 0) and at its oldest camera grab (``TIMING_KEY``)."""
    _missing_cameras(images)
    if not task:
        raise ConventionError("empty task string")
    if not (math.isfinite(write_time_s) and write_time_s >= 0.0 and math.isfinite(frame_time_s)):
        raise ConventionError(
            f"timing must be finite with write_time_s >= 0, got {write_time_s!r}, {frame_time_s!r}"
        )
    frame: dict[str, Any] = {IMAGE_KEYS[n]: _hwc_u8(n, images[n]) for n in CAMERA_NAMES}
    frame[STATE_KEY] = _joints("state", state)
    frame[ACTION_KEY] = _joints("action", action)
    frame[TIMING_KEY] = np.array([write_time_s, frame_time_s], np.float32)
    frame[TASK_KEY] = task
    return frame


def _chw01(name: str, img: Any) -> torch.Tensor:
    """uint8 HWC (a live frame) or float CHW in [0, 1] (a decoded dataset frame) -> float32 CHW."""
    t = img if isinstance(img, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(img))
    t = t.detach().cpu()
    if t.dtype == torch.uint8:
        if tuple(t.shape) != IMAGE_SHAPE_HWC:
            raise ConventionError(
                f"{name}: uint8 image must be {IMAGE_SHAPE_HWC}, got {tuple(t.shape)}"
            )
        return t.permute(2, 0, 1).to(torch.float32).div(255.0).contiguous()
    chw = (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    if not t.is_floating_point() or tuple(t.shape) != chw:
        raise ConventionError(
            f"{name}: float image must be {chw} in [0, 1], got {t.dtype} {tuple(t.shape)}"
        )
    if t.numel() and (float(t.min()) < 0.0 or float(t.max()) > 1.0):
        raise ConventionError(f"{name}: float image values outside [0, 1]")
    return t.to(torch.float32).contiguous()


def policy_observation(images: Mapping[str, Any], state: Any, task: str) -> dict[str, Any]:
    """THE preprocessing function: one observation -> the raw policy batch (batch size 1).

    ``images``: {"front", "wrist"} -> uint8 HWC RGB (live) or float CHW in [0, 1] (dataset).
    ``state``: canonical radians[6] in ``JOINT_NAMES`` order. Returns
    ``{observation.images.front: [1,3,480,640], observation.images.wrist: ..., observation.state:
    [1,6] float32, task: [task]}`` under the dataset's feature names, the batch LeRobot's policy
    preprocessor (``SMOLVLA_RENAME_MAP`` rename, tokenizer, MEAN_STD normalizer) takes.
    """
    _missing_cameras(images)
    if not task:
        raise ConventionError("empty task string")
    batch: dict[str, Any] = {IMAGE_KEYS[n]: _chw01(n, images[n]).unsqueeze(0) for n in CAMERA_NAMES}
    batch[STATE_KEY] = torch.from_numpy(_joints("state", state)).unsqueeze(0)
    batch[TASK_KEY] = [task]
    return batch


def policy_observation_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    """``policy_observation`` of one ``LeRobotDataset`` item (training data, recorder check)."""
    images = {n: sample[IMAGE_KEYS[n]] for n in CAMERA_NAMES if IMAGE_KEYS[n] in sample}
    return policy_observation(images, sample[STATE_KEY], str(sample[TASK_KEY]))


def action_targets(action: Any) -> FloatArray:
    """A policy's un-normalized action (6,) or chunk (N, 6) -> float64 (N, 6) canonical radians,
    absolute targets for ``So101Driver.step`` (which applies the safety filter)."""
    a = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    a = a.astype(np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] != NUM_JOINTS or a.shape[0] == 0 or not np.all(np.isfinite(a)):
        raise ConventionError(
            f"action must be finite (N, {NUM_JOINTS}), got shape {np.shape(action)}"
        )
    return a
