"""Domain-randomization settings. OFF unless cfg ``env.randomize: true``; then the defaults below
apply, overridden by keys under cfg ``env.randomization``.

Object *positions* are always sampled per episode from the cell's spawn region
(src/eval/protocol.py) — that is the task distribution, not a randomization. The hooks here are:

- object_color_jitter: max per-channel RGB offset of manipulated/base objects (colour name kept)
- object_physics: friction and mass jitter of manipulated/base objects
- lighting: ambient + directional light intensity/direction per sub-scene
- camera_*: per-episode camera pose jitter around the calibrated pose, drawn from the episode
  seed's RNG so a seed replays its camera (+ fov jitter at load)
- robot_color: random colour of the 3D-printed robot parts per sub-scene
- action_latency_max_steps: each episode draws a latency in [0, max] control steps
- reconfigure_every: rebuild the scene every K resets so load-time draws (colours, lights, fov)
  are re-sampled; 0 = only at construction
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class RandomizationConfig:
    enabled: bool = False
    object_color_jitter: float = 0.0
    object_physics: bool = False
    lighting: bool = False
    camera_pos_jitter_m: float = 0.0
    camera_target_jitter_m: float = 0.0
    camera_roll_jitter_rad: float = 0.0
    camera_fov_jitter_deg: float = 0.0
    robot_color: bool = False
    action_latency_max_steps: int = 0
    reconfigure_every: int = 0

    @property
    def camera_jitter(self) -> bool:
        return self.enabled and (
            self.camera_pos_jitter_m > 0
            or self.camera_target_jitter_m > 0
            or self.camera_roll_jitter_rad > 0
        )


ON_DEFAULTS: dict[str, Any] = {
    "object_color_jitter": 0.08,
    "object_physics": True,
    "lighting": True,
    "camera_pos_jitter_m": 0.02,
    "camera_target_jitter_m": 0.01,
    "camera_roll_jitter_rad": 0.02,
    "camera_fov_jitter_deg": 1.5,
    "robot_color": True,
    "action_latency_max_steps": 2,
    "reconfigure_every": 10,
}


def randomization_from_cfg(
    randomize: bool, overrides: dict[str, Any] | None
) -> RandomizationConfig:
    if not randomize:
        return RandomizationConfig()
    known = {f.name for f in fields(RandomizationConfig)} - {"enabled"}
    overrides = overrides or {}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"unknown env.randomization keys: {sorted(unknown)}")
    values = {**ON_DEFAULTS, **overrides}
    if not 0 <= int(values["action_latency_max_steps"]) <= 2:
        raise ValueError("action_latency_max_steps must be in [0, 2]")
    return RandomizationConfig(enabled=True, **values)
