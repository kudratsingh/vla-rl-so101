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

Objects (D3; owner kit list 2026-11-10, ``docs/bringup/kit.md``). Every object is a cube from ONE
set, hand2mind 1-inch wood cubes (set of 100, six colours): edge 25.4 mm NOMINAL (1 inch) until the
owner's measurement lands in ``MEASURED``, wood, same mass and finish, the colours as supplied in
the set (never painted to the sim RGBA). Seen (PickPlace): the set's RED cube. Unseen, replacing
protocol 0.1's green cylinder on the real arm only: the set's PURPLE and GREEN cubes, so only the
colour differs. The set's BLUE cube is recorded as the seen Stack base (``STACK_BASE_OBJECT``): Stack
is deferred, so it is unused in real-0.1. The sim keeps its 3 cm cubes (protocol 0.2 frozen).

Bowl. The LE TAUCI 8 oz bouillon cup, white (set of 6: one is the bowl, the others spares). Its
outer rim diameter, inside floor diameter, table-to-rim height and table-to-inside-floor height are
PENDING measurements (``MEASURED``); ``src/eval/bowl_fit.py`` reads them for the release window,
the cube-to-rim, carry-over-rim and lower-over-floor clearances and the release height, and falls
back to the sim bowl's geometry, labelled as a fallback, while they are pending.
``measurement_problems()`` range-checks them (mm, plausible for a 1-inch cube and an 8 oz cup). Filling ``MEASURED`` is the one edit before the geometry rerun
(``scripts/wo07_bowl_check.py``).

Scene (owner kit list 2026-11-10; hash inputs, ``scene_spec()``): table surface all white; one 17-inch LED
clamp desk lamp set ONCE to 5000 K and never changed (its position recorded at the camera match,
WO-06 step 7); exposure and white balance locked on both cameras at that light
(``configs/real.yaml`` ``cameras.<name>.controls``); the front camera a Logitech C920x on a NEEWER
UA057 22-inch magic arm with desk clamp at the sim front pose (``src/env/cameras.py``
``front_left_45``); the wrist camera the kit's 32 mm USB module on the gripper mount. RISK, white on
white: the white cup rim on the white table is low contrast for both cameras. The camera-match step
must show the rim visible in BOTH cameras under the 5000 K lamp with exposure locked; the table is
not changed for it. The sim scene stays wood-coloured (protocol 0.2 frozen); the WO-08 two-camera
sim protocol should use a white table, a white cup and 25.4 mm cubes to match this scene.

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

Scoring (CLAUDE.md rule 4): binary success under ``RUBRIC`` (judging rule ``real-judge-3``, the
settle interval and the fault rule) is the headline. real-judge-3 (owner 2026-11-09, after PR #33)
moves the start of a trial from "its first action reached the safety filter" (real-judge-2) to
"the bus write of its first policy goal was begun": a first goal blocked before its write (the
pre-write stop check, a stop or a stale frame / chunk after the start marker, a marker that could
not be saved) never started and is redone; a write that began and then failed counts as started. The paper's partial-credit score may be logged
as a secondary column (``RUBRIC["secondary"]``), never as the headline. Unseen is reported per
object and pooled; the pooled 50 is the headline.

Hash (``protocol_hash()``): SHA-256 of the canonical JSON of ``hash_components()``: the version,
slot ids and the rules that chose them, the bowl (its product, measurements and position) and the
constraint values it was derived from, the objects (the set, nominal and measured edge), the scene
facts, the three schedules in full, the grid geometry (every slot's id, cell and centre), the
judging rule and rubric, the horizon and settle interval, and the observation/action convention
hash (``src/robot/convention.py``). Values, not file bytes, so a comment edit never moves it.
Pinned in ``tests/test_protocol_real.py``: freeze before the first demo (D4).

Freeze status (``FREEZE_STATUS``): PENDING. The owner is checking bowl option B' (0.205, 0.185) m
on the bench (unpowered fit check); until he confirms, ``BOWL_XY_M``, ``RUBRIC_VERSION`` and the
placement-grid PDF stay as they are, the hash pinned in the test is the current computed value
(it moved with real-judge-3), and ``freeze_problem()`` makes every hardware session and the demo
schedule refuse (scripts/wo07_real_eval.py ``real_hardware_problems``, scripts/wo06_teleop_record.py
``--demo-schedule``), next to ``bowl_fit.bowl_box_problem``. ``freeze_problem()`` also refuses
while any prop measurement in ``MEASURED`` is None or out of range (``measurement_problems``),
even with ``FREEZE_STATUS`` set to ``FROZEN``.
The freeze fills ``MEASURED``, sets ``FREEZE_STATUS`` to ``FROZEN``, moves the bowl, bumps the
rubric and regenerates the PDF and the hash together.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from src.eval import protocol as P
from src.eval import real_grid as G

PROTOCOL_VERSION = "real-0.1"
FROZEN = "frozen"
FREEZE_STATUS = "pending: bowl fit check"  # module doc; FROZEN once the owner confirms the bowl
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
# -- props: the owner's bench measurements (owner kit list 2026-11-10, docs/bringup/kit.md) -----
CUBE_SET = "hand2mind 1-inch wood cubes (set of 100, six colours)"
CUBE_EDGE_NOMINAL_MM = 25.4  # 1 inch: NOMINAL, the set's stated size, until MEASURED has it
BOWL_PRODUCT = "LE TAUCI 8 oz bouillon cup, white (set of 6; one is the bowl, the rest spares)"


@dataclass(frozen=True)
class PropMeasurements:
    """The owner's measured prop dimensions, mm. None = PENDING (not measured yet).

    THE one place for them: ``freeze_problem`` refuses while any is None, the hash covers them,
    and ``src/eval/bowl_fit.py`` (release window, cube-to-rim and carry-over-rim clearances, read by
    ``scripts/wo07_bowl_check.py``) reads them, so the geometry rerun after the measurements is the
    one edit of ``MEASURED`` below."""

    red_cube_edge_mm: float | None = None  # calipers across two faces of the red cube
    bowl_outer_rim_diameter_mm: float | None = None  # across the outside of the rim
    bowl_inside_floor_diameter_mm: float | None = None  # the flat inside floor
    bowl_table_to_rim_height_mm: float | None = None  # table top to the rim's top edge
    # table top to the top of the inside floor (a footed cup's floor sits above the table): the
    # height the cube is released onto (bowl_fit.geometry floor_height_m)
    bowl_table_to_inside_floor_height_mm: float | None = None


MEASURED = PropMeasurements()  # PENDING: the owner sends the five numbers (2026-11-10)


def pending_measurements() -> list[str]:
    """The names of the prop measurements still None (empty once all are measured)."""
    return [f.name for f in fields(MEASURED) if getattr(MEASURED, f.name) is None]


def measurement_problems(m: PropMeasurements | None = None) -> list[str]:
    """Out-of-range or inconsistent measured values (mm; a value in inches or cm, or a cup the
    1-inch cube cannot sit in). Pending (None) values are not checked here."""
    m = MEASURED if m is None else m
    out: list[str] = []
    edge = m.red_cube_edge_mm
    rim, floor = m.bowl_outer_rim_diameter_mm, m.bowl_inside_floor_diameter_mm
    h_rim, h_floor = m.bowl_table_to_rim_height_mm, m.bowl_table_to_inside_floor_height_mm
    if edge is not None and not 20.0 <= edge <= 30.0:
        out.append(f"red_cube_edge_mm {edge} not in [20, 30] mm (a 1-inch cube is ~25.4 mm)")
    diag = (CUBE_EDGE_NOMINAL_MM if edge is None else edge) * 2.0**0.5
    if floor is not None and not diag < floor <= 200.0:
        out.append(
            f"bowl_inside_floor_diameter_mm {floor} not in ({diag:.1f}, 200] mm (more than the "
            "cube's diagonal, or the cube cannot lie flat on it at every yaw)"
        )
    if rim is not None and not 30.0 <= rim <= 250.0:
        out.append(f"bowl_outer_rim_diameter_mm {rim} not in [30, 250] mm")
    if rim is not None and floor is not None and rim <= floor:
        out.append(f"bowl_outer_rim_diameter_mm {rim} <= the inside floor diameter {floor} mm")
    if h_rim is not None and not 20.0 <= h_rim <= 120.0:
        out.append(f"bowl_table_to_rim_height_mm {h_rim} not in [20, 120] mm")
    if h_floor is not None and not 0.0 <= h_floor <= 40.0:
        out.append(f"bowl_table_to_inside_floor_height_mm {h_floor} not in [0, 40] mm")
    if h_rim is not None and h_floor is not None and h_floor >= h_rim:
        out.append(f"bowl_table_to_inside_floor_height_mm {h_floor} >= the rim height {h_rim} mm")
    return out


def cube_edge_mm() -> tuple[float, str]:
    """(edge, source): the measured red-cube edge, else the set's nominal 25.4 mm."""
    if MEASURED.red_cube_edge_mm is not None:
        return float(MEASURED.red_cube_edge_mm), "measured (red cube)"
    return CUBE_EDGE_NOMINAL_MM, "NOMINAL 1 inch, pending the owner's measurement"


def props_spec() -> dict[str, Any]:
    edge, src = cube_edge_mm()
    return {
        "measured_mm": asdict(MEASURED),
        "pending": pending_measurements(),
        "cube_set": CUBE_SET,
        "cube_edge_nominal_mm": CUBE_EDGE_NOMINAL_MM,
        "cube_edge_mm": edge,
        "cube_edge_source": src,
        "bowl_product": BOWL_PRODUCT,
    }


def bowl_spec() -> dict[str, Any]:
    return {
        "label": "bowl",
        "product": BOWL_PRODUCT,
        "planned_xy_m": list(BOWL_XY_M),
        "yaw": "not applicable for a round cup (any handle orientation is recorded at the fit "
        "check)",
        "outer_rim_diameter_mm": MEASURED.bowl_outer_rim_diameter_mm,  # None = PENDING
        "inside_floor_diameter_mm": MEASURED.bowl_inside_floor_diameter_mm,  # None = PENDING
        "table_to_rim_height_mm": MEASURED.bowl_table_to_rim_height_mm,  # None = PENDING
        "table_to_inside_floor_height_mm": MEASURED.bowl_table_to_inside_floor_height_mm,
        "colour": "white",
        "fixed": "the same cup at the same marked position for every demo and every trial",
    }


# -- objects (D3) -----------------------------------------------------------------------------
# The set's colours; the RGBA is the sim's reference value only (not a hash input: the real cube
# is the set's colour as supplied). Size = the nominal edge; object_spec reports cube_edge_mm().
_EDGE_M = CUBE_EDGE_NOMINAL_MM / 1000.0
SEEN_OBJECT = P.ObjectSpec(P.RED_CUBE.label, "cube", P.RED_CUBE.rgba, _EDGE_M)
PURPLE_CUBE = P.ObjectSpec(P.PURPLE_CUBE.label, "cube", P.PURPLE_CUBE.rgba, _EDGE_M)
GREEN_CUBE = P.ObjectSpec("green cube", "cube", (0.10, 0.65, 0.20, 1.0), _EDGE_M)
UNSEEN_OBJECTS: tuple[P.ObjectSpec, ...] = (PURPLE_CUBE, GREEN_CUBE)
STACK_BASE_OBJECT = P.ObjectSpec(P.BLUE_CUBE.label, "cube", P.BLUE_CUBE.rgba, _EDGE_M)
STACK_BASE_NOTE = "seen Stack base; Stack is deferred, so unused in real-0.1"
OBJECT_MATERIAL = (
    "wood, from the same set as the red demo cube: same edge, mass and finish; only the colour "
    "differs"
)


def object_spec(o: P.ObjectSpec) -> dict[str, Any]:
    edge, src = cube_edge_mm()
    return {
        "label": o.label,
        "shape": o.shape,
        "set": CUBE_SET,
        "colour": f"{o.label.split()[0]}, as supplied in the set",
        "edge_mm": edge,
        "edge_source": src,
        "material": OBJECT_MATERIAL,
    }


# -- scene (owner kit list 2026-11-10) ----------------------------------------------------------
WHITE_ON_WHITE_RISK = (
    "low contrast: the white cup rim on the all-white table, for both cameras. The camera match "
    "(WO-06 step 7) must show the rim visible in BOTH cameras under the 5000 K lamp with "
    "exposure and white balance locked; the table is not changed"
)


def scene_spec() -> dict[str, Any]:
    from src.env.cameras import DEFAULT_FOV_DEG, named_pose  # read only

    eye, target = named_pose("front_left_45")
    return {
        "table_surface": "all white",
        "lighting": "one 17-inch LED clamp desk lamp (2700-6500 K) set ONCE to 5000 K and never "
        "changed; its position recorded at the camera match",
        "camera_controls": "exposure and white balance locked on both cameras at that light "
        "(configs/real.yaml cameras.<name>.controls), never auto",
        "front_camera": {
            "device": "Logitech C920x",
            "mount": "NEEWER UA057 22-inch magic arm with desk clamp",
            "pose": "front_left_45 (the sim front pose, src/env/cameras.py)",
            "eye_m": list(eye),
            "target_m": list(target),
            "sim_vertical_fov_deg": DEFAULT_FOV_DEG,
        },
        "wrist_camera": {
            "device": "the SO-ARM101 Pro kit's 32 mm USB camera module",
            "mount": "the SO-101 gripper mount (3D-printed from the LeRobot SO-101 files if the "
            "kit has none)",
        },
        "risk": WHITE_ON_WHITE_RISK,
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
# A fixed number of policy COMMANDS, not a fixed wall-clock time (review F4): 450 writes are 30 s
# at the nominal 15 Hz, but an in-process SmolVLA on the Mac writes ~6.2-6.6 per second (~70 s per
# trial) and ACT ~13 (~35 s); docs/STATUS.md "PR 2: review round 1" has the bench budget.
TRIAL_MAX_STEPS = (
    450  # 30 s at 15 Hz = the teleop demo cap (configs/real.yaml teleop.episode_time_s)
)
SETTLE_S = 3.0
JUDGING_RULE_VERSION = "real-judge-3"
JUDGING_GENERAL = (
    "Judge ONCE, after the trial has ended and the arm has then held still for the settling "
    "interval of {settle_s:g} s (eval.settle_s; this prompt only appears then), by eye, touching "
    "nothing. y only if ALL of the task's conditions hold NOW; a success that was undone before "
    "the end is n. Doubt = n. A trial counts once its policy started (the bus write of its first "
    "action from the policy server was begun, even if that write then failed; a first goal "
    "blocked before its write, also one blocked after the start was marked, never started) and "
    "is then scored exactly once, never redone. A "
    "fault after the start (servo, serial, a safety-layer or deadman stop: joint / workspace "
    "violation, lost frame, stall, stale chunk; a policy-server failure; Ctrl-C during motion; a "
    "client error; torque off unconfirmed; inside the trial, at the settle read or while this "
    "prompt waits) makes it a failure without a judgement (the fault is recorded; a later "
    "'resume' does not change it). A score prompt that ends without a judgement (Ctrl-C, closed "
    "input, no answer in deploy.prompt_torque_timeout_s) scores it n. An attempt whose policy "
    "never started is not scored and is redone with the same placement."
)
# The success conditions of src/env/success.py (read only), with the REAL props' sizes (owner kit
# list 2026-11-10). Only PickPlace is judged in real-0.1; Push and Stack are deferred.
JUDGING_RULE: dict[str, str] = {
    "PickPlace": "the object rests inside the bowl (over the bowl floor, below the rim), the bowl "
    "is upright, and the object touches neither gripper jaw",
    "Push": "(deferred: not judged in real-0.1) the object's centre is inside the 2-inch (50.8 mm) "
    "matte black target disc, the object is on the table, it was never lifted off the table "
    "during the trial, and the gripper is not holding it",
    "Stack": "(deferred: not judged in real-0.1) the object rests on top of the blue cube (centred "
    f"within half the cube edge, ~{cube_edge_mm()[0] / 2:.1f} mm, flat), the blue cube is still on "
    "the table, and the object touches neither gripper jaw",
}
RUBRIC_ID = "so101-real-pickplace-rubric"
RUBRIC_VERSION = "1"
TASK_VERSION = f"{RUBRIC_ID}-{RUBRIC_VERSION}"  # vishwaroop task_version = rubric version


def freeze_problem() -> str | None:
    """Why no demo or hardware session may use this protocol yet (None once it is frozen and
    every prop measurement is in)."""
    pending = pending_measurements()
    bad = measurement_problems()
    if FREEZE_STATUS == FROZEN and not pending and not bad:
        return None
    parts = []
    if FREEZE_STATUS != FROZEN:
        parts.append(
            f"protocol {PROTOCOL_VERSION} is not frozen ({FREEZE_STATUS}): the bowl position, the "
            "rubric version, the placement-grid PDF and the hash wait for the owner's "
            "confirmation (src/eval/protocol_real.py FREEZE_STATUS)"
        )
    if pending:
        parts.append(
            f"protocol {PROTOCOL_VERSION} prop measurements PENDING: {', '.join(pending)} "
            "(src/eval/protocol_real.py MEASURED; the owner's bench measurements)"
        )
    if bad:
        parts.append(
            f"protocol {PROTOCOL_VERSION} prop measurements out of range: {'; '.join(bad)} "
            "(src/eval/protocol_real.py MEASURED, in mm)"
        )
    return "; ".join(parts)


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
        "bowl": {**bowl_spec(), "rule": BOWL_RULE, "constraints": bowl_constraints()},
        "objects": {
            "seen": object_spec(SEEN_OBJECT),
            "unseen": [object_spec(o) for o in UNSEEN_OBJECTS],
            "stack_base": {**object_spec(STACK_BASE_OBJECT), "use": STACK_BASE_NOTE},
        },
        "props": props_spec(),
        "scene": scene_spec(),
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
