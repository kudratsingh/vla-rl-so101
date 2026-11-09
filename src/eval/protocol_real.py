"""Real-arm evaluation protocol ``real-0.1`` (WO-07, SmolVLA + ACT on the SO-101): the frozen seen
slots, the fixed bowl, the held-out objects, the seeded trial schedules, the rubric, and one
composite hash over all of them.

Owner decisions 2026-11-09 (final2 handoff item 6 and the pivot amendments D1-D5). This is a NEW
protocol version: ``src/eval/protocol.py`` (0.1-draft) and ``src/eval/protocol_v02.py`` (0.2) stay
byte-identical and keep governing the sim. Only PickPlace is defined: Push and Stack wait until the
PickPlace baseline is in.

Seen cell (D1). Five slots of the 16-slot inner block of ``src/eval/real_grid.py`` (the grid is
unchanged): its four corner slots ``B2 B5 E2 E5`` and one of its four central slots. The 4 x 4 block
has no centre slot, so the central one is chosen by ``central_slot()``: of ``C3 C4 D3 D4``, the slot
whose bowl distance is nearest the mean bowl distance of the four corners, so the middle training
position is also in the middle of the carry distances the corners span (``C4``). With the corners
on the inner block's boundary every ring slot lies outside the seen hull: the shifted cell is an
extrapolation in every direction. The other 11 inner slots, the ring and the held-out objects are
never used by demos or camera-match frames.

Bowl (D2). ONE fixed position for every demo and every trial of every cell, so shifted tests object
position and unseen tests object identity with everything else held constant. Derived by
``derive_bowl()`` from the scene's placement constraints (``src/env/tasks.py``, read only), against
all 36 lattice slots (stronger than the 25 used: it also stays valid if a seen slot is ever
re-chosen):

* every slot centre at least ``MIN_SEP["PickPlace"]`` (11 cm) from the bowl centre,
* the bowl centre at least ``BASE_KEEPOUT + 0.065`` from the base (the sim's bowl keep-out),
* the bowl centre within ``PUSH_REACH`` (29 cm; the arm's top-down reach, the only reach constant
  in tasks.py),

each with a 5 mm placement-tolerance margin, on a 5 mm lattice (a position the operator can mark).
Among the positions that pass, the one nearest the grid centre (shortest carry; keeps the bowl beside
the grid in the front camera's view); the +y / -y tie goes to +y, the front camera's side: the arm
then reaches the bowl from the side away from the camera, so the camera sees the release. Result
``BOWL_XY_M = (0.195, 0.205)``: 11.75 cm from the nearest slot, 28.3 cm from the base. The sim's
oracle IK reaches it at release, carry and retreat heights (probed 2026-11-09, < 0.1 mm residual,
tilt <= 20 deg). Nothing else fits in front of the grid: a bowl ahead of it would sit beyond 37 cm.

Objects (D3). Seen: the red cube (``protocol.RED_CUBE``). Unseen, replacing protocol 0.1's green
cylinder on the real arm only: the purple cube (``protocol.PURPLE_CUBE``) and a green cube
(``GREEN_CUBE``, the cylinder's green): 3 cm cubes of the red cube's material, mass and finish, so
only the colour differs.

Schedules (D3, D4, D5), all seeded and frozen, materialized in the hash:

* ``eval_schedule()``: 150 trials. Seen 50 = 10 at each seen slot (every 15 deg yaw of
  -45..45 once, three more drawn); shifted 50 over the 20 ring slots, 3 at every other slot around
  the ring and 2 at the rest (fixed); unseen 50 at the five seen slots, 25 purple + 25 green, five
  per slot per object (five distinct yaws). The SESSION ORDER is one seeded permutation of all 150
  trials across the cells, so drift (lighting, servo heat, the operator) never lines up with a cell.
  ``trial_idx`` is the trial's stable index inside its cell. Both policies run this same schedule.
* ``demo_schedule()``: the 50 PickPlace demos, 10 at each seen slot, red cube, the fixed bowl.
* ``dev_schedule()``: 30 development trials (10 seen, 10 shifted, 10 unseen) for checkpoint
  selection and the 100-demo failure analysis, NEVER reported. Disjoint from the evaluation
  schedule by construction: its yaws are the half-step grid (+-7.5, +-22.5, +-37.5 deg), so no
  development placement equals any evaluation placement.

Every coordinate is a PLANNED pose (the grid slot centre and yaw the operator is told to use), never
a measurement: the keys say ``planned_`` and ``POSE_SOURCE`` says so.

Scoring (CLAUDE.md rule 4): binary success under ``RUBRIC`` (judging rule ``real-judge-2``, the
settle interval and the fault rule) is the headline. The paper's partial-credit score may be logged
as a secondary column (``RUBRIC["secondary"]``), never as the headline. Unseen is reported per
object and pooled; the pooled 50 is the headline.

Hash (``protocol_hash()``): SHA-256 of the canonical JSON of ``hash_components()``: the version,
slot ids and the rules that chose them, the bowl and the constraint values it was derived from, the
objects, the three schedules in full, the grid geometry (every slot's id, cell and centre), the
judging rule and rubric, the horizon and settle interval, and the observation/action convention
hash (``src/robot/convention.py``). Values, not file bytes, so a comment edit never moves it.
Pinned in ``tests/test_protocol_real.py``: freeze before the first demo (D4).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.eval import protocol as P
from src.eval import real_grid as G

PROTOCOL_VERSION = "real-0.1"
TASK = "PickPlace"
TASK_ID = "so101-real-pickplace"
CELLS: tuple[str, ...] = P.CELLS  # seen, shifted, unseen_objects
TRIALS_PER_CELL = P.REAL_TRIALS_PER_CELL  # 50
POSE_SOURCE = "planned: grid slot centre and yaw from this schedule, not measured"

# -- seen slots (D1) --------------------------------------------------------------------------
CORNER_SLOTS: tuple[str, ...] = ("B2", "B5", "E2", "E5")  # corners of the 4 x 4 inner block
CENTRAL_CANDIDATES: tuple[str, ...] = ("C3", "C4", "D3", "D4")  # the block has no centre slot
SEEN_SLOTS: tuple[str, ...] = ("B2", "B5", "E2", "E5", "C4")
SEEN_RULE = (
    "the four corner slots of the 16-slot inner block, plus the central slot (of C3 C4 D3 D4) "
    "whose bowl distance is nearest the mean bowl distance of the corners"
)

# -- bowl (D2) --------------------------------------------------------------------------------
BOWL_XY_M: tuple[float, float] = (0.195, 0.205)
BOWL_MARGIN_M = 0.005  # placement tolerance on every constraint
BOWL_LATTICE_M = 0.005
BOWL_SEARCH_M = ((0.0, 0.30), (-0.30, 0.30))  # x, y ranges searched (front half plane)
BOWL_RULE = (
    "on a 5 mm lattice, every lattice slot centre >= MIN_SEP[PickPlace] + 5 mm away, >= "
    "BASE_KEEPOUT + 0.065 + 5 mm and <= PUSH_REACH - 5 mm from the base; nearest the grid centre; "
    "tie +y (the front camera's side)"
)
BOWL_SPEC: dict[str, Any] = {
    "label": "bowl",
    "planned_xy_m": list(BOWL_XY_M),
    "yaw": "not applicable (round bowl)",
    "inner_floor_diameter_mm": "90 +- 5 (sim: 2 x BOWL_FLOOR_RADIUS 45 mm, src/env/success.py)",
    "rim_height_mm": "<= 36 (sim wall 30 mm + floor 6 mm; the oracle carries the cube 16 mm above)",
    "colour": "off-white (sim BOWL_RGBA 0.93 0.93 0.90)",
    "fixed": "the same bowl at the same marked position for every demo and every trial",
}

# -- objects (D3) -----------------------------------------------------------------------------
GREEN_CUBE = P.ObjectSpec("green cube", "cube", (0.10, 0.65, 0.20, 1.0))  # the cylinder's green
SEEN_OBJECT = P.RED_CUBE
UNSEEN_OBJECTS: tuple[P.ObjectSpec, ...] = (P.PURPLE_CUBE, GREEN_CUBE)
OBJECT_MATERIAL = "same material, mass and finish as the red demo cube; only the colour differs"


def object_spec(o: P.ObjectSpec) -> dict[str, Any]:
    return {
        "label": o.label,
        "shape": o.shape,
        "edge_mm": round(o.size * 1000.0, 3),
        "nominal_rgba": list(o.rgba),
        "material": OBJECT_MATERIAL,
    }


def objects_for(cell: str) -> tuple[P.ObjectSpec, ...]:
    if cell == "unseen_objects":
        return UNSEEN_OBJECTS
    if cell in ("seen", "shifted"):
        return (SEEN_OBJECT,)
    raise ValueError(f"unknown cell {cell!r}; expected one of {CELLS}")


_BY_LABEL = {o.label: o for o in (SEEN_OBJECT, *UNSEEN_OBJECTS)}


def object_by_label(label: str) -> P.ObjectSpec:
    return _BY_LABEL[label]


# -- schedules (D3-D5) ------------------------------------------------------------------------
EVAL_SEED = 20261109
DEMO_SEED = 20261110
DEV_SEED = 20261111
YAWS_DEG: tuple[int, ...] = tuple(range(-G.YAW_MAX_DEG, G.YAW_MAX_DEG + 1, G.YAW_STEP_DEG))
DEV_YAWS_DEG: tuple[float, ...] = (-37.5, -22.5, -7.5, 7.5, 22.5, 37.5)  # half-step: disjoint
DEMOS_PER_SLOT = 10
DEV_TRIALS = {"seen": 10, "shifted": 10, "unseen_objects": 10}

# -- trial horizon, settling, rubric ----------------------------------------------------------
CONTROL_HZ = 15  # = src/robot/convention.FPS
TRIAL_MAX_STEPS = (
    450  # 30 s at 15 Hz = the teleop demo cap (configs/real.yaml teleop.episode_time_s)
)
SETTLE_S = 3.0
JUDGING_RULE_VERSION = "real-judge-2"
JUDGING_GENERAL = (
    "Judge ONCE, after the trial has ended and the arm has then held still for the settling "
    "interval of {settle_s:g} s (eval.settle_s; this prompt only appears then), by eye, touching "
    "nothing. y only if ALL of the task's conditions hold NOW; a success that was undone before "
    "the end is n. Doubt = n. A trial counts once its policy started (its first action from the "
    "policy server reached the safety filter) and is then scored exactly once, never redone. A "
    "fault after the start (servo, serial, a safety-layer or deadman stop: joint / workspace "
    "violation, lost frame, stall, stale chunk; a policy-server failure; Ctrl-C during motion; a "
    "client error; torque off unconfirmed; inside the trial, at the settle read or while this "
    "prompt waits) makes it a failure without a judgement (the fault is recorded; a later "
    "'resume' does not change it). A score prompt that ends without a judgement (Ctrl-C, closed "
    "input, no answer in deploy.prompt_torque_timeout_s) scores it n. An attempt whose policy "
    "never started is not scored and is redone with the same placement."
)
JUDGING_RULE: dict[str, str] = {  # mirrors src/env/success.py, read only
    "PickPlace": "the object rests inside the bowl (over the bowl floor, below the rim), the bowl "
    "is upright, and the object touches neither gripper jaw",
    "Push": "the object's centre is inside the 5 cm target disc, the object is on the table, it "
    "was never lifted off the table during the trial, and the gripper is not holding it",
    "Stack": "the object rests on top of the blue cube (centred within ~1.5 cm, flat), the blue "
    "cube is still on the table, and the object touches neither gripper jaw",
}
RUBRIC_ID = "so101-real-pickplace-rubric"
RUBRIC_VERSION = "1"
TASK_VERSION = f"{RUBRIC_ID}-{RUBRIC_VERSION}"  # vishwaroop task_version = rubric version


def judging_text(settle_s: float = SETTLE_S) -> str:
    return JUDGING_GENERAL.format(settle_s=settle_s)


RUBRIC: dict[str, Any] = {
    "rubric_id": RUBRIC_ID,
    "version": RUBRIC_VERSION,
    "protocol_version": PROTOCOL_VERSION,
    "task": TASK,
    "task_id": TASK_ID,
    "instruction_template": P.INSTRUCTIONS[TASK],
    "scoring": "binary",
    "headline": "successes / trials whose policy started, per cell (unseen: per object and "
    "pooled; the pooled 50 is the headline); Wilson 95% interval",
    "judging_rule": JUDGING_RULE_VERSION,
    "success": JUDGING_RULE[TASK],
    "when": judging_text(SETTLE_S),
    "settle_s": SETTLE_S,
    "horizon_steps": TRIAL_MAX_STEPS,
    "control_hz": CONTROL_HZ,
    "rules": {
        "R1_end_state": "success iff the success condition holds after the settling interval",
        "R2_fault": "a trial whose policy started and that ended in a fault (servo, serial, "
        "safety / deadman, server, operator abort, client error, torque off unconfirmed, session "
        "lost) is a failure, from its recorded fault, without a judgement",
        "R3_unjudged": "a score prompt that ended without a judgement is a failure (doubt = n)",
        "R4_not_started": "an attempt whose policy never started is not a trial: never scored, "
        "redone with the same placement",
    },
    "secondary": {
        "partial_credit": "optional SECONDARY column (SmolVLA paper's pick 0.5 + place 0.5 style "
        "score), only from a blinded clip review (scripts/wo07_blind_review.py "
        "--partial-credit); never the headline",
    },
    "labels": "live operator scores keep outcome_source operator; automatic fault outcomes keep "
    "outcome_source fault; only a blinded clip review (scripts/wo07_blind_review.py) is "
    "declared blinded_review",
}


def _slot(sid: str) -> G.Slot:
    return {s.id: s for s in G.all_slots()}[sid]


def ring_order() -> list[str]:
    """The 20 ring slots clockwise as seen from above (+x up the page): row A left to right,
    column 6 up, row F right to left, column 1 down."""
    ring = {s.id for s in G.cell_slots("shifted")}
    rows, n = G.ROW_LETTERS, G.N_COLS
    path = [f"{rows[0]}{c}" for c in range(1, n + 1)]
    path += [f"{r}{n}" for r in rows[1:]]
    path += [f"{rows[-1]}{c}" for c in range(n - 1, 0, -1)]
    path += [f"{r}1" for r in reversed(rows[1:-1])]
    if len(path) != len(ring) or set(path) != ring:
        raise RuntimeError("ring_order does not cover the 20 ring slots")
    return path


def shifted_counts() -> dict[str, int]:
    """3 trials at every other ring slot (starting at A1), 2 at the rest: 50 over 20, fixed."""
    out = {sid: 3 if i % 2 == 0 else 2 for i, sid in enumerate(ring_order())}
    assert sum(out.values()) == TRIALS_PER_CELL
    return out


def _bowl_ok(x: float, y: float, slots: Sequence[G.Slot], margin: float) -> bool:
    from src.env import tasks as T  # read only

    c = np.array([x, y])
    d = min(float(np.linalg.norm(c - np.array([s.x, s.y]))) for s in slots)
    r = float(np.linalg.norm(c))
    return (
        d >= T.MIN_SEP["PickPlace"] + margin
        and r >= T.BASE_KEEPOUT + 0.065 + margin
        and r <= T.PUSH_REACH - margin
    )


def bowl_constraints() -> dict[str, float]:
    from src.env import tasks as T

    return {
        "min_sep_m": float(T.MIN_SEP["PickPlace"]),
        "bowl_keepout_m": float(T.BASE_KEEPOUT + 0.065),
        "reach_m": float(T.PUSH_REACH),
        "margin_m": BOWL_MARGIN_M,
        "lattice_m": BOWL_LATTICE_M,
    }


def derive_bowl() -> tuple[float, float]:
    """The bowl position by ``BOWL_RULE`` (module doc). Raises if no position passes."""
    k = bowl_constraints()
    m = BOWL_MARGIN_M
    pts = np.array([(s.x, s.y) for s in G.all_slots()])
    (x0, x1), (y0, y1) = BOWL_SEARCH_M
    st = BOWL_LATTICE_M
    xs = np.round(x0 + st * np.arange(round((x1 - x0) / st) + 1), 4)
    ys = np.round(y0 + st * np.arange(round((y1 - y0) / st) + 1), 4)
    cand = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    d = np.linalg.norm(cand[:, None, :] - pts[None], axis=-1).min(axis=1)
    r = np.linalg.norm(cand, axis=1)
    ok = (d >= k["min_sep_m"] + m) & (r >= k["bowl_keepout_m"] + m) & (r <= k["reach_m"] - m)
    if not ok.any():
        raise RuntimeError("no fixed bowl position satisfies every slot (see BOWL_RULE)")
    good = cand[ok]
    near = np.round(np.linalg.norm(good - np.array(P.SEEN_REGION.center), axis=1), 9)
    i = int(np.lexsort((-good[:, 1], near))[0])  # nearest the grid centre; tie -> +y
    return float(good[i, 0]), float(good[i, 1])


def bowl_report(slot_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Distances of the fixed bowl to every slot used (default: seen + ring) and to the base."""
    ids = list(slot_ids) if slot_ids is not None else [*SEEN_SLOTS, *ring_order()]
    b = np.array(BOWL_XY_M)
    dist = {
        sid: round(float(np.linalg.norm(b - np.array([_slot(sid).x, _slot(sid).y]))), 4)
        for sid in ids
    }
    k = bowl_constraints()
    return {
        "bowl_xy_m": list(BOWL_XY_M),
        "min_slot_distance_m": min(dist.values()),
        "nearest_slot": min(dist, key=lambda s: dist[s]),
        "base_distance_m": round(float(np.linalg.norm(b)), 4),
        "constraints": k,
        "valid": all(_bowl_ok(*BOWL_XY_M, [_slot(s)], 0.0) for s in ids),
        "distances_m": dist,
    }


def central_slot() -> str:
    """``SEEN_RULE``'s central slot (ties: the first in CENTRAL_CANDIDATES order)."""
    b = np.array(BOWL_XY_M)

    def bd(sid: str) -> float:
        s = _slot(sid)
        return float(np.linalg.norm(b - np.array([s.x, s.y])))

    mean = sum(bd(s) for s in CORNER_SLOTS) / len(CORNER_SLOTS)
    return min(
        CENTRAL_CANDIDATES, key=lambda s: (round(abs(bd(s) - mean), 9), CENTRAL_CANDIDATES.index(s))
    )


def instruction(obj: P.ObjectSpec) -> str:
    return P.instruction(TASK, obj)


def _item(
    cell: str, trial_idx: int, obj: P.ObjectSpec, sid: str, yaw: float, kind: str
) -> dict[str, Any]:
    s = _slot(sid)
    return {
        "kind": kind,  # eval | demo | dev
        "cell": cell,
        "trial_idx": trial_idx,
        "object": obj.label,
        "object_slot": sid,
        "planned_object_xy_m": [s.x, s.y],
        "planned_object_yaw_deg": float(yaw),
        "planned_bowl_xy_m": list(BOWL_XY_M),
        "goal_kind": "bowl",
        "instruction": instruction(obj),
        "pose_source": POSE_SOURCE,
    }


def _yaws(rng: np.random.Generator, n: int, grid: Sequence[float]) -> list[float]:
    """n yaws: whole seeded permutations of ``grid`` (each value once per pass), then a draw
    without replacement for the remainder."""
    out: list[float] = []
    while len(out) + len(grid) <= n:
        out += [float(grid[i]) for i in rng.permutation(len(grid))]
    rest = n - len(out)
    if rest:
        out += [float(grid[i]) for i in rng.choice(len(grid), size=rest, replace=False)]
    return out


def _cells(
    seed: int, kind: str, yaw_grid: Sequence[float], counts: dict[str, dict[str, int]]
) -> list[dict[str, Any]]:
    """Canonical per-cell lists (trial_idx in this order): counts[cell][slot]; unseen splits each
    slot's count over the two objects."""
    rng = np.random.default_rng([seed, 1])
    items: list[dict[str, Any]] = []
    for cell in CELLS:
        k = 0
        for sid, n in counts[cell].items():
            if cell == "unseen_objects":
                for obj in UNSEEN_OBJECTS:
                    for yaw in _yaws(rng, n, yaw_grid):
                        items.append(_item(cell, k, obj, sid, yaw, kind))
                        k += 1
            else:
                for yaw in _yaws(rng, n, yaw_grid):
                    items.append(_item(cell, k, SEEN_OBJECT, sid, yaw, kind))
                    k += 1
    return items


def _ordered(items: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    perm = np.random.default_rng([seed, 2]).permutation(len(items))
    out = []
    for pos, i in enumerate(perm):
        out.append({"schedule_pos": pos, **items[int(i)]})
    return out


def eval_schedule() -> list[dict[str, Any]]:
    """The 150 evaluation trials in session order (module doc)."""
    counts = {
        "seen": {sid: TRIALS_PER_CELL // len(SEEN_SLOTS) for sid in SEEN_SLOTS},
        "shifted": shifted_counts(),
        # five per slot per object
        "unseen_objects": {
            sid: TRIALS_PER_CELL // len(SEEN_SLOTS) // len(UNSEEN_OBJECTS) for sid in SEEN_SLOTS
        },
    }
    return _ordered(_cells(EVAL_SEED, "eval", YAWS_DEG, counts), EVAL_SEED)


def dev_schedule() -> list[dict[str, Any]]:
    """30 development trials, never reported; yaws on the half-step grid (disjoint)."""
    ring = ring_order()
    counts = {
        "seen": {sid: DEV_TRIALS["seen"] // len(SEEN_SLOTS) for sid in SEEN_SLOTS},
        "shifted": {sid: 1 for sid in ring[1::2]},  # the ring slots that get 2 in the eval
        "unseen_objects": {sid: 1 for sid in SEEN_SLOTS},  # x 2 objects
    }
    return _ordered(_cells(DEV_SEED, "dev", DEV_YAWS_DEG, counts), DEV_SEED)


def demo_schedule() -> list[dict[str, Any]]:
    """The 50 PickPlace demos: 10 per seen slot, red cube, fixed bowl, seeded order."""
    rng = np.random.default_rng([DEMO_SEED, 1])
    items = []
    k = 0
    for sid in SEEN_SLOTS:
        for yaw in _yaws(rng, DEMOS_PER_SLOT, YAWS_DEG):
            items.append(_item("seen", k, SEEN_OBJECT, sid, yaw, "demo"))
            k += 1
    out = _ordered(items, DEMO_SEED)
    for d in out:
        d["demo_idx"] = d.pop("schedule_pos")
        del d["trial_idx"]
    return out


SCHEDULES = {"eval": eval_schedule, "dev": dev_schedule}


def schedule(name: str) -> list[dict[str, Any]]:
    if name not in SCHEDULES:
        raise ValueError(f"unknown schedule {name!r}; expected one of {sorted(SCHEDULES)}")
    return SCHEDULES[name]()


def placement_prompt(item: dict[str, Any]) -> str:
    yaw = float(item["planned_object_yaw_deg"])
    turn = (
        "square to the grid"
        if yaw == 0
        else f"turned {abs(yaw):g} deg {'CCW' if yaw > 0 else 'CW'}"
    )
    return (
        f"place the {item['object']} centred on slot {item['object_slot']} ({turn}); the bowl "
        f"stays on its fixed mark (x {BOWL_XY_M[0]:.3f}, y {BOWL_XY_M[1]:+.3f} m)"
    )


def _digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def schedule_sha256(name: str) -> str:
    return _digest(schedule(name))


def grid_geometry() -> list[dict[str, Any]]:
    return [{"id": s.id, "cell": s.cell, "x": s.x, "y": s.y} for s in G.all_slots()]


def hash_components() -> dict[str, Any]:
    """Everything ``protocol_hash`` covers (recorded with every trial so a change can be traced)."""
    from src.robot.convention import citation

    return {
        "version": PROTOCOL_VERSION,
        "task": {"name": TASK, "id": TASK_ID, "version": TASK_VERSION},
        "seen": {"slots": list(SEEN_SLOTS), "rule": SEEN_RULE, "corners": list(CORNER_SLOTS)},
        "shifted": {"slots": ring_order(), "counts": shifted_counts()},
        "bowl": {**BOWL_SPEC, "rule": BOWL_RULE, "constraints": bowl_constraints()},
        "objects": {
            "seen": object_spec(SEEN_OBJECT),
            "unseen": [object_spec(o) for o in UNSEEN_OBJECTS],
        },
        "schedules": {
            "eval": eval_schedule(),
            "demo": demo_schedule(),
            "dev": dev_schedule(),
            "seeds": {"eval": EVAL_SEED, "demo": DEMO_SEED, "dev": DEV_SEED},
            "yaws_deg": list(YAWS_DEG),
            "dev_yaws_deg": list(DEV_YAWS_DEG),
        },
        "grid": grid_geometry(),
        "judging": {
            "version": JUDGING_RULE_VERSION,
            "general": JUDGING_GENERAL,
            "task": JUDGING_RULE[TASK],
        },
        "rubric": RUBRIC,
        "horizon": {"trial_max_steps": TRIAL_MAX_STEPS, "control_hz": CONTROL_HZ},
        "settle_s": SETTLE_S,
        "pose_source": POSE_SOURCE,
        "convention": citation(),
    }


def protocol_hash() -> str:
    return _digest(hash_components())


def citation() -> dict[str, str]:
    return {
        "module": "src/eval/protocol_real.py",
        "version": PROTOCOL_VERSION,
        "hash": protocol_hash(),
    }


def rubric() -> dict[str, Any]:
    """rubric.json for an export (versioned; carries the protocol hash it belongs to)."""
    return {**RUBRIC, "protocol_hash": protocol_hash()}


def cell_tables(trials: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Binary success per reporting row: seen, shifted, unseen per object and pooled (the
    headline). ``trials``: one scored record per (cell, trial_idx); ``outcome`` success/failure."""
    rows: dict[str, list[bool]] = {
        "seen": [],
        "shifted": [],
        **{f"unseen_objects/{o.label}": [] for o in UNSEEN_OBJECTS},
        "unseen_objects (pooled, headline)": [],
    }
    for t in trials:
        if t.get("outcome") not in ("success", "failure"):
            continue
        ok = t["outcome"] == "success"
        cell = t["cell"]
        if cell == "unseen_objects":
            obj = (t.get("placement") or {}).get("object")
            rows[f"unseen_objects/{obj}"].append(ok)
            rows["unseen_objects (pooled, headline)"].append(ok)
        else:
            rows[cell].append(ok)
    out = {}
    for name, v in rows.items():
        n, s = len(v), sum(v)
        lo, hi = P.wilson_interval(s, n)
        out[name] = {
            "n": n,
            "successes": s,
            "success_rate": s / n if n else None,
            "wilson_95": [round(lo, 4), round(hi, 4)],
        }
    return out
