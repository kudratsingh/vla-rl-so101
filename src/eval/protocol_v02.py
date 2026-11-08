"""Evaluation protocol 0.2: 0.1-draft's tasks, cells, seeds and episode counts, a composite hash
that pins the whole evaluation, and the real arm's per-step joint clip applied in sim eval.

Why a new version (project review 2026-11-07, section 3.2): the 0.1-draft hash is the SHA-256 of
``src/eval/protocol.py`` alone, so it does not change when the success checks, spawn sampling,
horizon, rates, camera, image size, chunk rule or sim backend change. ``protocol.py`` is left
byte-for-byte untouched: every number already reported cites its hash ``e9ce5bb47a88``.

Same as 0.1-draft (imported from ``src/eval/protocol.py``, not copied): TASKS, CELLS, SIM_SEEDS,
SIM_EPISODES_PER_CELL, REAL_TRIALS_PER_CELL, spawn regions, object sets, instructions, Wilson z;
and the episode-seed formula (``src/eval/sim_eval.py`` ``episode_seeds``), so a 0.2 run plays the
same episode seeds as the 0.1-draft run of the same (task, cell, seed).

Changed in 0.2: every executed action passes a per-joint step clip
``|a_t - a_{t-1}| <= bound`` (``a_{-1}`` = the joint reading at reset), with
``bound = min(max_joint_step_deg, max_velocity_deg_s / CONTROL_HZ)`` read from
``configs/real.yaml`` ``robot.safety``: the nominal-rate bound ``src/robot/safety.py`` applies on
the arm (4 deg/step at 15 Hz). The real filter's joint-limit clip and workspace-box rejection are
not mirrored in sim.

Hash (``protocol_hash()``): SHA-256 of the canonical JSON of ``hash_components()``:

- ``files``: SHA-256 of the BYTES of ``FILES`` (this file, protocol.py, the env package: scene,
  spawn sampling, success checks, camera, robot, randomization, CPU vector env) and of the robot
  model (URDF + meshes). Bytes, not extracted constants: spawn sampling in ``tasks.py`` is code
  (rejection sampling, inline yaw ranges), so only its bytes pin it. A comment edit in one of these
  files therefore changes the hash too: a false alarm costs one re-freeze, a silent change costs a
  wrong comparison.
- ``settings``: the canonical run settings below (horizon, image size, control mode, sim backend,
  chunk execution rule, camera, randomization off). A run whose config deviates from them is not a
  0.2 run: ``deviations()`` lists the differences and the eval marks it ``protocol_complete:
  false``.
- ``action_clip``: the clip as read from ``configs/real.yaml`` (changing it there changes the hash).
- ``episode_seed_probe``: the first episode seed of every (seed, task, cell), pinning the formula.
- ``sim_stack``: the versions of the simulator packages pinned in ``uv.lock``.

Control/physics rates (``CONTROL_HZ`` in ``src/env/__init__.py``, ``SIM_FREQ`` in ``tasks.py``)
are covered by those files' bytes. Not covered: the policy side (checkpoint, crop, precision),
cited by the run's config hash, and the machine (GPU, driver), in the run record.

Status: not frozen. Its first reported number freezes it; pin the hash in a test then.
"""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.eval.protocol import (  # noqa: F401  (re-exported: 0.2 keeps these unchanged)
    CELLS,
    CONFIDENCE_Z,
    INSTRUCTIONS,
    REAL_TRIALS_PER_CELL,
    SIM_EPISODES_PER_CELL,
    SIM_SEEDS,
    TASKS,
    instruction,
    manipulated_objects,
    spawn_regions,
    wilson_interval,
)

PROTOCOL_VERSION = "0.2"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

FILES: tuple[str, ...] = (
    "src/eval/protocol.py",
    "src/eval/protocol_v02.py",
    "src/env/__init__.py",
    "src/env/tasks.py",
    "src/env/success.py",
    "src/env/cameras.py",
    "src/env/robot.py",
    "src/env/randomization.py",
    "src/env/vector.py",
)
ROBOT_MODEL_DIR = "src/env/assets/so101"
ROBOT_MODEL_GLOBS: tuple[str, ...] = ("*.urdf", "**/*.stl")
REAL_CONFIG = "configs/real.yaml"
SIM_STACK: tuple[str, ...] = ("gymnasium", "mani-skill", "sapien", "torch")

# Canonical run settings. A run must match every entry to count as a 0.2 run.
SETTINGS: dict[str, Any] = {
    "max_episode_steps": 150,  # horizon, control steps (10 s at 15 Hz)
    "image_size": 224,  # policy camera, square
    "control_mode": "pd_joint_pos",  # absolute joint targets (src/env/__init__.py make_env)
    "sim_backend": "physx_cuda",  # GPU sim + NVIDIA renderer, where every reported eval ran
    "camera": {"pose": "front_left_45", "fov_deg": 40.0},  # resolved by src/env/cameras.py
    "randomize": False,  # domain randomization off
    "exec_steps": None,  # whole action chunk executed open loop, then re-query
    "decoding": "greedy",
    "episode_end": "first success step, env termination/truncation, or max_episode_steps",
}
CAMERA_TOL_M = 1e-3


@dataclass(frozen=True)
class ActionClip:
    """Per-joint step clip, all six joints (gripper included, as on the arm)."""

    max_joint_step_deg: float
    max_velocity_deg_s: float
    control_hz: float

    @property
    def bound_deg(self) -> float:
        return min(self.max_joint_step_deg, self.max_velocity_deg_s / self.control_hz)

    @property
    def bound_rad(self) -> float:
        return math.radians(self.bound_deg)


def action_clip(root: Path = REPO_ROOT) -> ActionClip:
    """The clip from ``configs/real.yaml`` ``robot.safety`` at the sim control rate."""
    from src.env import CONTROL_HZ  # cheap: the package defers ManiSkill imports

    safety = yaml.safe_load((root / REAL_CONFIG).read_text())["robot"]["safety"]
    return ActionClip(
        float(safety["max_joint_step_deg"]), float(safety["max_velocity_deg_s"]), CONTROL_HZ
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashed_files(root: Path = REPO_ROOT) -> list[str]:
    """Repo-relative paths whose bytes enter the hash (sorted); a missing file raises."""
    model = root / ROBOT_MODEL_DIR
    found = {p for g in ROBOT_MODEL_GLOBS for p in model.glob(g) if p.is_file()}
    if not found:
        raise FileNotFoundError(f"no robot model files under {model}")
    paths = list(FILES) + [p.relative_to(root).as_posix() for p in found]
    return sorted(paths)


def _sim_stack(root: Path) -> dict[str, list[str]]:
    lock = tomllib.loads((root / "uv.lock").read_text())
    out: dict[str, list[str]] = {name: [] for name in SIM_STACK}
    for pkg in lock.get("package", []):
        if pkg.get("name") in out:
            out[pkg["name"]].append(str(pkg.get("version")))
    missing = [k for k, v in out.items() if not v]
    if missing:
        raise KeyError(f"uv.lock pins none of {missing}")
    return {k: sorted(v) for k, v in out.items()}


def _episode_seed_probe() -> list[int]:
    from src.eval.sim_eval import episode_seeds

    return [
        episode_seeds(s, t, c, 1)[0]
        for s in SIM_SEEDS
        for t in range(len(TASKS))
        for c in range(len(CELLS))
    ]


def hash_components(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Everything the hash covers (recorded in eval.json so a changed hash can be traced)."""
    clip = action_clip(root)
    return {
        "version": PROTOCOL_VERSION,
        "files": {rel: _sha256(root / rel) for rel in hashed_files(root)},
        "settings": SETTINGS,
        "action_clip": {
            "source": f"{REAL_CONFIG} robot.safety",
            "max_joint_step_deg": clip.max_joint_step_deg,
            "max_velocity_deg_s": clip.max_velocity_deg_s,
            "control_hz": clip.control_hz,
            "bound_deg": clip.bound_deg,
        },
        "episode_seed_probe": _episode_seed_probe(),
        "sim_stack": _sim_stack(root),
    }


def protocol_hash(root: Path = REPO_ROOT) -> str:
    blob = json.dumps(hash_components(root), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def deviations(cfg: Mapping[str, Any], *, sim_device: str, exec_steps: int | None) -> list[str]:
    """How a run's effective config differs from SETTINGS (empty = a 0.2-conforming run)."""
    from src.env.cameras import resolve_front_camera

    env = cfg.get("env", {})
    out = []
    for key in ("max_episode_steps", "image_size"):
        got = env.get(key)
        if got is None or int(got) != SETTINGS[key]:
            out.append(f"env.{key} = {got!r}, protocol {SETTINGS[key]!r}")
    if bool(env.get("randomize", False)) != SETTINGS["randomize"]:
        out.append(f"env.randomize = {env.get('randomize')!r}, protocol {SETTINGS['randomize']!r}")
    backend = "physx_cuda" if sim_device.startswith("cuda") else "physx_cpu"
    if backend != SETTINGS["sim_backend"]:
        out.append(f"sim backend {backend}, protocol {SETTINGS['sim_backend']}")
    if exec_steps != SETTINGS["exec_steps"]:
        out.append(f"exec_steps = {exec_steps!r}, protocol {SETTINGS['exec_steps']!r}")
    run_cam = resolve_front_camera(env.get("camera"))
    ref_cam = resolve_front_camera(dict(SETTINGS["camera"]))
    far = max(
        abs(a - b)
        for a, b in zip(run_cam.eye + run_cam.target, ref_cam.eye + ref_cam.target, strict=True)
    )
    if far > CAMERA_TOL_M or abs(run_cam.fov_deg - ref_cam.fov_deg) > 1e-6:
        out.append(
            f"camera eye/target/fov {run_cam.eye}/{run_cam.target}/{run_cam.fov_deg}, protocol "
            f"{ref_cam.eye}/{ref_cam.target}/{ref_cam.fov_deg}"
        )
    return out
