"""Does the real-0.1 bowl fit the real safety filter's TCP workspace box? (owner 2026-11-09)

``scripts/wo07_bowl_check.py`` verifies a bowl position offline through ``src/robot/safety.py``;
this module holds the two geometric checks it shares with the hardware preflights, so a session or
a demo recording never starts at a bowl the filter would reject on every carry:

* ``box_margin``: the bowl centre's smallest margin to the ``workspace_box_m`` x / y faces, after
  the operator's 5 mm placement tolerance (``protocol_real.BOWL_MARGIN_M``) and the oracle's
  TCP-to-bowl offset at lower / release / retreat (>= 0 fits);
* ``release_window``: how much of the release region (the TCP disc over the bowl floor, radius
  floor radius less the cube's planar half-diagonal) stays inside the box after the tolerance: a learned or
  teleop release off centre that leaves the box is rejected by the filter, so this is the share of
  in-bowl releases the filter allows (the region's radius is ``release_radius_m``: the floor
  radius less the cube's planar half-DIAGONAL, so the whole cube is on the floor at any yaw).

``bowl_box_problem`` is the preflight (``scripts/wo07_real_eval.py`` ``real_hardware_problems``,
``scripts/wo06_teleop_record.py --demo-schedule``). It reads the protocol's bowl and the config's
box; it changes neither (nothing here is a protocol hash input, and the filter is never loosened).

Prop geometry (owner kit list 2026-11-10). ``geometry()`` is the cube and cup geometry every check
here and in ``scripts/wo07_bowl_check.py`` uses, read from ONE place,
``protocol_real.MEASURED`` (the owner's measurements): the cube half-edge (measured, else the
hand2mind set's nominal 25.4 mm), the cup's inside floor radius, outer rim radius, table-to-rim
height and table-to-inside-floor height. While a cup measurement is pending the sim bowl's value stands in
(``src/env/success.py``, read only) and its ``sources`` entry says ``PENDING ... sim fallback``,
so no number derived from it is the real cup's. From it:

* ``release_radius_m``: the TCP disc over the floor that keeps the whole cube inside it at any
  yaw (floor radius less the cube's planar half-diagonal, edge / sqrt 2; the release-window
  input; owner review of PR #32: half the edge holds only for a yaw-aligned cube);
* ``cube_rim_clearance_m``: the outer rim to the nearest cube's corner at any yaw, the bowl placed
  the 5 mm tolerance closer (cube only: the open jaws reach further, not modelled);
* ``carry_rim_clearance_m``: how far the cube's bottom passes above the rim at the sim oracle's
  carry height (``src/data/oracle.py`` ``CARRY_Z``, the TCP ~3 mm above the cube centre after
  the descend); < 0 means the oracle's carry would drag the cube through the rim;
* ``lower_floor_clearance_m``: the cube's bottom above the cup's inside floor at the end of the
  oracle's lower (its target: the cube bottom ``BOWL_FLOOR_THICKNESS + 12 mm`` above the table,
  ``src/data/oracle.py`` "lower"); < 0 means the lower would drive the cube into the floor.

``geometry_problems`` turns these (and ``protocol_real.measurement_problems``) into the bowl
check's verdict: any entry fails the check (``scripts/wo07_bowl_check.py``), whatever the
kinematic phases say, because the kinematic scene models neither the rim nor the floor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.eval import protocol_real as PR

# The sim oracle's largest TCP-to-bowl-centre distance at lower / release / retreat, measured by
# scripts/wo07_bowl_check.py at the frozen bowl over all 25 slots x 7 yaws (2026-11-09: 9.3 mm).
# The largest of the measured positions; the feasibility search uses it for every candidate.
TCP_TO_BOWL_OFFSET_M = 0.0093
TOLERANCE_M = PR.BOWL_MARGIN_M  # the operator's placement tolerance (5 mm), as derive_bowl
# the oracle's descend stops the TCP this far above the cube centre (src/data/oracle.py "descend")
TCP_ABOVE_CUBE_CENTRE_M = 0.003
# the oracle's lower stops the cube bottom this far above the sim floor (src/data/oracle.py "lower")
LOWER_DROP_M = 0.012
PENDING_FALLBACK = "PENDING measurement: sim bowl fallback (src/env/success.py)"


@dataclass(frozen=True)
class PropGeometry:
    """Cube and cup geometry (m) for the bowl checks, with where each value came from."""

    cube_half_m: float
    floor_radius_m: float
    rim_radius_m: float  # outer rim
    rim_height_m: float  # table top to the rim's top edge
    floor_height_m: float  # table top to the top of the inside floor (where the cube lands)
    sources: dict[str, str]
    problems: tuple[str, ...] = ()  # protocol_real.measurement_problems of the measurements used

    @property
    def measured(self) -> bool:
        return not any(v.startswith("PENDING") or "NOMINAL" in v for v in self.sources.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "cube_half_m": round(self.cube_half_m, 5),
            "floor_radius_m": round(self.floor_radius_m, 5),
            "rim_radius_m": round(self.rim_radius_m, 5),
            "rim_height_m": round(self.rim_height_m, 5),
            "floor_height_m": round(self.floor_height_m, 5),
            "sources": dict(self.sources),
            "all_measured": self.measured,
            "measurement_problems": list(self.problems),
        }


def geometry(m: PR.PropMeasurements | None = None) -> PropGeometry:
    """The prop geometry from ``protocol_real.MEASURED`` (or ``m``); sim fallbacks labelled."""
    from src.env import success as S  # read only

    m = PR.MEASURED if m is None else m
    sim_rim_r = (
        S.BOWL_FLOOR_RADIUS
        + S.BOWL_WALL_HEIGHT * math.tan(math.radians(S.BOWL_WALL_TILT_DEG))
        + S.BOWL_WALL_THICKNESS
    )
    sim_rim_h = S.BOWL_WALL_HEIGHT + S.BOWL_FLOOR_THICKNESS
    src: dict[str, str] = {}

    def pick(key: str, mm: float | None, fallback_m: float, scale: float) -> float:
        if mm is None:
            src[key] = PENDING_FALLBACK
            return fallback_m
        src[key] = "measured"
        return float(mm) * scale / 1000.0

    if m.red_cube_edge_mm is None:
        cube_half = PR.CUBE_EDGE_NOMINAL_MM / 2000.0
        src["cube_half_m"] = "NOMINAL 25.4 mm edge (hand2mind 1 inch), pending measurement"
    else:
        cube_half = float(m.red_cube_edge_mm) / 2000.0
        src["cube_half_m"] = "measured"
    return PropGeometry(
        cube_half_m=cube_half,
        floor_radius_m=pick(
            "floor_radius_m", m.bowl_inside_floor_diameter_mm, S.BOWL_FLOOR_RADIUS, 0.5
        ),
        rim_radius_m=pick("rim_radius_m", m.bowl_outer_rim_diameter_mm, sim_rim_r, 0.5),
        rim_height_m=pick("rim_height_m", m.bowl_table_to_rim_height_mm, sim_rim_h, 1.0),
        floor_height_m=pick(
            "floor_height_m",
            m.bowl_table_to_inside_floor_height_mm,
            S.BOWL_FLOOR_THICKNESS,
            1.0,
        ),
        sources=src,
        problems=tuple(PR.measurement_problems(m)),
    )


def cube_half_diagonal_m(g: PropGeometry | None = None) -> float:
    """The cube's planar half-diagonal (centre to a vertical edge, edge / sqrt 2): the footprint
    radius at an arbitrary yaw. Half the edge is right only vertically or for a yaw-aligned cube."""
    g = geometry() if g is None else g
    return g.cube_half_m * math.sqrt(2.0)


def release_radius_m(g: PropGeometry | None = None) -> float:
    """TCP disc radius over the cup floor that keeps the whole cube on it at any yaw: the floor
    radius less the cube's half-diagonal (25.4 mm cube, 90 mm floor: 27.0 mm, not 32.3 mm)."""
    g = geometry() if g is None else g
    return g.floor_radius_m - cube_half_diagonal_m(g)


def cube_rim_clearance_m(slot_distance_m: float, g: PropGeometry | None = None) -> float:
    """Outer rim to the nearest cube's corner (any yaw) with the bowl the 5 mm tolerance closer."""
    g = geometry() if g is None else g
    return slot_distance_m - g.rim_radius_m - cube_half_diagonal_m(g) - TOLERANCE_M


def carry_rim_clearance_m(g: PropGeometry | None = None) -> float:
    """Cube bottom above the rim at the sim oracle's carry height (< 0: through the rim)."""
    from src.data.oracle import CARRY_Z  # read only

    g = geometry() if g is None else g
    # half the edge is right here: a vertical distance (the cube lies flat, centre to bottom face)
    return CARRY_Z - TCP_ABOVE_CUBE_CENTRE_M - g.cube_half_m - g.rim_height_m


def lower_floor_clearance_m(g: PropGeometry | None = None) -> float:
    """Cube bottom above the cup's inside floor at the end of the oracle's lower (< 0: into it)."""
    from src.env import success as S  # read only

    g = geometry() if g is None else g
    return S.BOWL_FLOOR_THICKNESS + LOWER_DROP_M - g.floor_height_m


def geometry_problems(g: PropGeometry | None = None) -> list[str]:
    """Why the prop geometry fails the bowl check whatever the bowl position (empty = none):
    out-of-range measurements, no release disc, the carry through the rim, the lower into the
    floor. The position-dependent cube-to-rim clearance is checked per bowl."""
    g = geometry() if g is None else g
    out = [f"measurement: {p}" for p in g.problems]
    if release_radius_m(g) <= 0:
        out.append(
            f"release radius {1000 * release_radius_m(g):+.1f} mm: the inside floor is narrower "
            "than the cube's diagonal"
        )
    c = carry_rim_clearance_m(g)
    if c < 0:
        out.append(
            f"carry through the rim: the cube's bottom passes {100 * c:+.2f} cm over the rim at the "
            "oracle's carry height (src/data/oracle.py CARRY_Z)"
        )
    f = lower_floor_clearance_m(g)
    if f < 0:
        out.append(
            f"lower into the floor: the oracle's lower ends with the cube's bottom {100 * f:+.2f} "
            "cm over the cup's inside floor"
        )
    return out


def workspace_box(cfg: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """``robot.safety.workspace_box_m`` as {axis: (lo, hi)} in metres (what make_safety reads)."""
    box = cfg["robot"]["safety"]["workspace_box_m"]
    return {a: (float(box[a][0]), float(box[a][1])) for a in ("x", "y", "z")}


def face_margins(
    bowl: Sequence[float], box: Mapping[str, Sequence[float]], tcp_off: float
) -> dict[str, float]:
    """Margin (m) of the bowl centre to each x / y face after the tolerance and ``tcp_off``."""
    x, y = float(bowl[0]), float(bowl[1])
    pad = TOLERANCE_M + float(tcp_off)
    return {
        "x_lo": round(x - float(box["x"][0]) - pad, 4),
        "x_hi": round(float(box["x"][1]) - x - pad, 4),
        "y_lo": round(y - float(box["y"][0]) - pad, 4),
        "y_hi": round(float(box["y"][1]) - y - pad, 4),
    }


def box_margin(bowl: Sequence[float], box: Mapping[str, Sequence[float]], tcp_off: float) -> float:
    """Smallest of ``face_margins`` (>= 0: the oracle's release stays inside the box)."""
    return min(face_margins(bowl, box, tcp_off).values())


def release_window(
    bowl: Sequence[float], box: Mapping[str, Sequence[float]], radius_m: float, n: int = 241
) -> dict[str, float]:
    """The release region (TCP disc of ``radius_m`` around the bowl centre) against the box
    shrunk by the placement tolerance: ``inside_fraction`` (area share, a deterministic n x n
    grid) and ``inscribed_radius_m`` (the largest centred disc wholly inside, capped at
    ``radius_m``). 1.0 / radius_m: no in-bowl release is ever rejected."""
    c = np.asarray(bowl, dtype=np.float64)[:2]
    lo = np.array([float(box["x"][0]), float(box["y"][0])]) + TOLERANCE_M
    hi = np.array([float(box["x"][1]), float(box["y"][1])]) - TOLERANCE_M
    u = np.linspace(-radius_m, radius_m, n)
    gx, gy = np.meshgrid(u, u)
    disc = gx**2 + gy**2 <= radius_m**2
    px, py = gx + c[0], gy + c[1]
    inside = (px >= lo[0]) & (px <= hi[0]) & (py >= lo[1]) & (py <= hi[1])
    face = float(min(c[0] - lo[0], hi[0] - c[0], c[1] - lo[1], hi[1] - c[1]))
    return {
        "radius_m": round(float(radius_m), 4),
        "inside_fraction": round(float((disc & inside).sum() / disc.sum()), 3),
        "inscribed_radius_m": round(max(0.0, min(float(radius_m), face)), 4),
    }


def bowl_box_problem(cfg: Mapping[str, Any], bowl: Sequence[float] | None = None) -> str | None:
    """Why the bowl (default: the protocol's) cannot be used on the arm, None when it fits."""
    b = tuple(PR.BOWL_XY_M if bowl is None else bowl)
    box = workspace_box(cfg)
    m = box_margin(b, box, TCP_TO_BOWL_OFFSET_M)
    if m >= 0:
        return None
    return (
        f"the protocol bowl ({b[0]:.3f}, {b[1]:.3f}) m is {-100 * m:.2f} cm outside the safety "
        f"filter's workspace box {box['x']} x {box['y']} m (after the {1000 * TOLERANCE_M:.0f} mm "
        f"placement tolerance and the oracle's {1000 * TCP_TO_BOWL_OFFSET_M:.1f} mm TCP offset): "
        "every carry to it would be rejected (scripts/wo07_bowl_check.py). The bowl is pending "
        "the owner's decision; the filter is never loosened"
    )
