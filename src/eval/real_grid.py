"""Real-arm placement grid (WO-07): printed slots that stand in for the sim's spawn sampling.

Derived, never hand-typed, from the protocol's spawn regions (``src/eval/protocol.py``, read
only: SEEN_REGION, SHIFTED_REGION = SEEN_REGION x SHIFT_SCALE 1.5) and the scene's placement
constraints (``src/env/tasks.py``, read only: BASE_KEEPOUT, MIN_SEP, PUSH_DIST, PUSH_REACH). Both
are the same in protocol 0.1-draft and 0.2 (0.2 imports them from protocol.py).

Lattice. SHIFTED_REGION is cut into a 6 x 6 grid of equal rectangles, 2.5 cm along x by 3.5 cm
along y. Because SHIFT_SCALE is 1.5, the inner 4 x 4 block is exactly SEEN_REGION. A slot is a
rectangle's centre:

* ``seen`` and ``unseen_objects`` slots: the 16 inner rectangles (inside SEEN_REGION),
* ``shifted`` slots: the 20 ring rectangles (inside SHIFTED_REGION, outside SEEN_REGION: the
  sim's shifted sampling region).

Equal-area rectangles sampled uniformly = a stratified version of the sim's uniform sampling, at
the resolution an operator can place by hand.

Slot ids: a row letter A-F along +x (A nearest the robot base) and a column number 1-6 along -y
(1 at the robot's left, +y; viewed from behind the base looking forward, column 1 is on the left),
e.g. ``C3``. ``docs/assets/placement-grid-a3.pdf`` (``scripts/make_placement_grid.py``) prints
these ids; ``scripts/wo07_real_eval.py`` samples them with ``sample_plan``.

Per trial, ``sample_plan`` picks the manipulated object's slot (cycling a seeded permutation of
the cell's valid object slots, so every slot is used about equally often) and the second slot
(PickPlace bowl, Push target, Stack base cube) uniformly among the slots that satisfy the sim's
constraints for that object slot: the same cell's slots, centre distances (MIN_SEP; Push 6-10 cm
along any direction, the pusher start within PUSH_REACH), and keep-outs from the robot base. Also
a cube yaw in 15 deg steps within the sim's +/-45 deg. The same seed gives the same plan, so an SFT
and an SFT+RL session with one seed are paired trial by trial (owner 2026-11-08: both checkpoints
use the same slot sequence). The plan is prefix-stable for any ``n``: the object order comes
from its own stream (``[seed, task, cell, 0]``, whole permutations appended in turn) and trial
k's second slot and yaw from a stream of their own (``[seed, task, cell, 1, k]``), so trial k's
placement depends only on (seed, task, cell, k), never on the session's trial count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.eval import protocol as P

N_ROWS = 6  # along x
N_COLS = 6  # along y
ROW_LETTERS = "ABCDEF"
YAW_STEP_DEG = 15
YAW_MAX_DEG = 45  # sim: cube yaw uniform in [-pi/4, pi/4] (src/env/tasks.py)
PUSH_PUSHER_OFFSET_M = 0.035  # sim: the pusher starts ~3.5 cm behind the object

if not math.isclose(P.SHIFT_SCALE, 1.5):
    raise RuntimeError(
        "real_grid assumes SHIFT_SCALE == 1.5 (seen = the inner 4 x 4 of a 6 x 6 lattice); "
        "re-derive the lattice for a new protocol version"
    )


@dataclass(frozen=True)
class Slot:
    id: str
    row: int  # 0..5 along +x
    col: int  # 0..5 along -y
    x: float  # m, robot base frame
    y: float
    cell: str  # "seen" | "shifted"
    half_x: float  # rectangle half size, m
    half_y: float


def cell_size_m() -> tuple[float, float]:
    hx, hy = P.SHIFTED_REGION.half_extent
    return 2 * hx / N_ROWS, 2 * hy / N_COLS


def all_slots() -> list[Slot]:
    """The 36 lattice rectangles, row-major (A1 .. F6)."""
    (cx, cy), (hx, hy) = P.SHIFTED_REGION.center, P.SHIFTED_REGION.half_extent
    dx, dy = cell_size_m()
    out = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x = round(cx - hx + (r + 0.5) * dx, 6)
            y = round(cy + hy - (c + 0.5) * dy, 6)
            cell = "seen" if P.SEEN_REGION.contains(x, y) else "shifted"
            out.append(Slot(f"{ROW_LETTERS[r]}{c + 1}", r, c, x, y, cell, dx / 2, dy / 2))
    return out


def cell_slots(cell: str) -> list[Slot]:
    """Slots a protocol cell samples from (``unseen_objects`` uses the seen region)."""
    if cell not in P.CELLS:
        raise ValueError(f"unknown cell {cell!r}; expected one of {P.CELLS}")
    want = "shifted" if cell == "shifted" else "seen"
    return [s for s in all_slots() if s.cell == want]


GOAL_KIND = {"PickPlace": "bowl", "Push": "target mark", "Stack": "blue cube"}


def _constraints() -> dict[str, Any]:
    from src.env import tasks as T  # sim constants, read only (heavy import: SAPIEN, ManiSkill)

    return {
        "keepout": T.BASE_KEEPOUT,
        "min_sep": dict(T.MIN_SEP),
        "push_dist": tuple(T.PUSH_DIST),
        "push_reach": T.PUSH_REACH,
    }


def _ok(task: str, o: Slot, g: Slot, k: dict[str, Any]) -> bool:
    po, pg = np.array([o.x, o.y]), np.array([g.x, g.y])
    if o.id == g.id:
        return False
    d = float(np.linalg.norm(pg - po))
    r_o, r_g = float(np.linalg.norm(po)), float(np.linalg.norm(pg))
    if task == "PickPlace":  # sim keep-outs: object + 2.5 cm, bowl + 6.5 cm
        return bool(
            d >= k["min_sep"]["PickPlace"]
            and r_o >= k["keepout"] + 0.025
            and r_g >= k["keepout"] + 0.065
        )
    if task == "Stack":
        return bool(d >= k["min_sep"]["Stack"] and min(r_o, r_g) >= k["keepout"] + 0.025)
    if task == "Push":
        lo, hi = k["push_dist"]
        if not (lo <= d <= hi) or r_o < k["keepout"] + 0.025:
            return False
        u = (pg - po) / d
        return bool(float(np.linalg.norm(po - PUSH_PUSHER_OFFSET_M * u)) <= k["push_reach"])
    raise ValueError(f"unknown task {task!r}")


def valid_pairs(task: str, cell: str) -> dict[str, list[str]]:
    """object slot id -> second-slot ids allowed with it (object slots with none are dropped)."""
    k = _constraints()
    slots = cell_slots(cell)
    out: dict[str, list[str]] = {}
    for o in slots:
        goals = [g.id for g in slots if _ok(task, o, g, k)]
        if goals:
            out[o.id] = goals
    return out


def sample_plan(task: str, cell: str, n: int, seed: int) -> list[dict[str, Any]]:
    """``n`` placements for (task, cell), deterministic in ``seed`` (module doc)."""
    if task not in P.TASKS:
        raise ValueError(f"unknown task {task!r}")
    by_id = {s.id: s for s in all_slots()}
    pairs = valid_pairs(task, cell)
    if not pairs:
        raise RuntimeError(f"no valid placements for {task}/{cell}")
    objs = sorted(pairs)
    key = [int(seed), P.TASKS.index(task), P.CELLS.index(cell)]
    rng_order = np.random.default_rng([*key, 0])  # independent streams: prefix-stable (module doc)
    order: list[str] = []
    while len(order) < int(n):
        order += [objs[i] for i in rng_order.permutation(len(objs))]
    yaws = list(range(-YAW_MAX_DEG, YAW_MAX_DEG + 1, YAW_STEP_DEG))
    plan = []
    for k in range(int(n)):
        rng = np.random.default_rng([*key, 1, k])
        o = by_id[order[k]]
        goals = pairs[o.id]
        g = by_id[goals[int(rng.integers(len(goals)))]]
        plan.append(
            {
                "trial_idx": k,
                "slot_seed": int(seed),
                "object_slot": o.id,
                "object_xy_m": [o.x, o.y],
                "object_yaw_deg": int(yaws[int(rng.integers(len(yaws)))]),
                "goal_kind": GOAL_KIND[task],
                "goal_slot": g.id,
                "goal_xy_m": [g.x, g.y],
            }
        )
    return plan


def placement_prompt(obj_label: str, placement: dict[str, Any]) -> str:
    yaw = int(placement["object_yaw_deg"])
    turn = (
        "square to the grid" if yaw == 0 else f"turned {abs(yaw)} deg {'CCW' if yaw > 0 else 'CW'}"
    )
    return (
        f"place the {obj_label} centred on slot {placement['object_slot']} ({turn}) and the "
        f"{placement['goal_kind']} centred on slot {placement['goal_slot']}"
    )
