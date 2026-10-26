"""Evaluation protocol: the single source of truth for tasks, cells, spawn regions and object sets.

DRAFT until WO-02 freezes it. After the freeze, never edit this file to change a number: add a
new protocol version instead (CLAUDE.md rule 4). Every reported number cites `protocol_hash()`.

Frames: positions are metres in the robot base frame (x forward from the base, y to the robot's
left, z up; table surface at z = 0). The env (src/env) reads regions and objects from here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

PROTOCOL_VERSION = "0.1-draft"

TASKS: tuple[str, ...] = ("PickPlace", "Push", "Stack")
CELLS: tuple[str, ...] = ("seen", "shifted", "unseen_objects")

SIM_EPISODES_PER_CELL = 100
REAL_TRIALS_PER_CELL = 50
SIM_SEEDS: tuple[int, ...] = (0, 1, 2)
CONFIDENCE_Z = 1.96  # 95% Wilson interval

# Instruction templates; {obj} / {base} are ObjectSpec.label values.
INSTRUCTIONS: dict[str, str] = {
    "PickPlace": "pick up the {obj} and place it in the bowl",
    "Push": "push the {obj} to the target",
    "Stack": "stack the {obj} on the {base}",
}


@dataclass(frozen=True)
class Region:
    """Axis-aligned rectangle of object-centre positions on the table."""

    center: tuple[float, float]
    half_extent: tuple[float, float]

    def scaled(self, factor: float) -> Region:
        return Region(self.center, (self.half_extent[0] * factor, self.half_extent[1] * factor))

    def contains(self, x: float, y: float) -> bool:
        return (
            abs(x - self.center[0]) <= self.half_extent[0]
            and abs(y - self.center[1]) <= self.half_extent[1]
        )


@dataclass(frozen=True)
class ObjectSpec:
    """A manipulable object. `size` is the cube edge / cylinder diameter and height (m)."""

    label: str  # used verbatim in instructions
    shape: str  # "cube" | "cylinder"
    rgba: tuple[float, float, float, float]
    size: float = 0.03


# Demo ("seen") spawn region: object centres sampled uniformly inside it.
SEEN_REGION = Region(center=(0.205, 0.0), half_extent=(0.05, 0.07))
SHIFT_SCALE = 1.5
# "shifted": centres sampled inside SHIFTED_REGION but OUTSIDE SEEN_REGION.
SHIFTED_REGION = SEEN_REGION.scaled(SHIFT_SCALE)

# Objects. Seen = the only objects that appear in demos/SFT data.
RED_CUBE = ObjectSpec("red cube", "cube", (0.85, 0.10, 0.10, 1.0))
BLUE_CUBE = ObjectSpec("blue cube", "cube", (0.10, 0.25, 0.85, 1.0))
# Held out: never used in demos. Replace the manipulated object (Stack keeps the blue base cube).
GREEN_CYLINDER = ObjectSpec("green cylinder", "cylinder", (0.10, 0.65, 0.20, 1.0))
PURPLE_CUBE = ObjectSpec("purple cube", "cube", (0.55, 0.20, 0.75, 1.0))

SEEN_OBJECTS: dict[str, ObjectSpec] = {"obj": RED_CUBE, "base": BLUE_CUBE}
UNSEEN_OBJECTS: tuple[ObjectSpec, ObjectSpec] = (GREEN_CYLINDER, PURPLE_CUBE)


def spawn_regions(cell: str) -> tuple[Region, Region | None]:
    """(sample region, exclusion region) for a cell."""
    if cell == "shifted":
        return SHIFTED_REGION, SEEN_REGION
    if cell in ("seen", "unseen_objects"):
        return SEEN_REGION, None
    raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")


def manipulated_objects(cell: str) -> tuple[ObjectSpec, ...]:
    """Candidate manipulated objects for a cell (one is chosen per sub-env)."""
    if cell == "unseen_objects":
        return UNSEEN_OBJECTS
    if cell in ("seen", "shifted"):
        return (SEEN_OBJECTS["obj"],)
    raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")


def instruction(task: str, obj: ObjectSpec, base: ObjectSpec = BLUE_CUBE) -> str:
    return INSTRUCTIONS[task].format(obj=obj.label, base=base.label)


def protocol_hash() -> str:
    """SHA-256 of this file's bytes."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def wilson_interval(successes: int, n: int, z: float = CONFIDENCE_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (0, 1) when n == 0."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
