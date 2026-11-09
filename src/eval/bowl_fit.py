"""Does the real-0.1 bowl fit the real safety filter's TCP workspace box? (owner 2026-11-09)

``scripts/wo07_bowl_check.py`` verifies a bowl position offline through ``src/robot/safety.py``;
this module holds the two geometric checks it shares with the hardware preflights, so a session or
a demo recording never starts at a bowl the filter would reject on every carry:

* ``box_margin``: the bowl centre's smallest margin to the ``workspace_box_m`` x / y faces, after
  the operator's 5 mm placement tolerance (``protocol_real.BOWL_MARGIN_M``) and the oracle's
  TCP-to-bowl offset at lower / release / retreat (>= 0 fits);
* ``release_window``: how much of the release region (the TCP disc over the bowl floor, radius
  floor radius less the cube half-size) stays inside the box after the tolerance: a learned or
  teleop release off centre that leaves the box is rejected by the filter, so this is the share of
  in-bowl releases the filter allows.

``bowl_box_problem`` is the preflight (``scripts/wo07_real_eval.py`` ``real_hardware_problems``,
``scripts/wo06_teleop_record.py --demo-schedule``). It refuses every bowl but the one the owner
approved for the freeze (``APPROVED_BOWL_XY_M``, B'), and that one too unless it is the
protocol's bowl and fits the config's box. It changes neither (nothing here is a protocol hash
input, and the filter is never loosened).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.eval import protocol_real as PR

# The bowl the owner approved for the real-0.1 freeze: option B' (2026-11-09; the unpowered fit
# check at the bowl and C1, D1, B2, E2). A session or a demo at any other bowl is refused, so a
# protocol edit that moves the bowl cannot reach the arm without moving this pin too.
APPROVED_BOWL_XY_M: tuple[float, float] = (0.205, 0.185)
# The sim oracle's largest TCP-to-bowl-centre distance at lower / release / retreat, measured by
# scripts/wo07_bowl_check.py at the sim-rule bowl (0.195, 0.205) over all 25 slots x 7 yaws
# (2026-11-09: 9.3 mm; 8.8 mm at B').
# The largest of the measured positions; the feasibility search uses it for every candidate.
TCP_TO_BOWL_OFFSET_M = 0.0093
TOLERANCE_M = PR.BOWL_MARGIN_M  # the operator's placement tolerance (5 mm), as derive_bowl


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
    """Why the bowl (default: the protocol's) cannot be used on the arm, None when it fits: only
    the approved bowl, only while it is the protocol's, and only inside the box."""
    b = tuple(float(v) for v in (PR.BOWL_XY_M if bowl is None else bowl))
    if b != APPROVED_BOWL_XY_M or tuple(PR.BOWL_XY_M) != APPROVED_BOWL_XY_M:
        return (
            f"the bowl ({b[0]:.3f}, {b[1]:.3f}) m is not the approved real-0.1 bowl "
            f"{APPROVED_BOWL_XY_M} m (src/eval/bowl_fit.py APPROVED_BOWL_XY_M) or the protocol's "
            f"bowl {tuple(PR.BOWL_XY_M)} m differs from it: a session or a demo only runs at the "
            "bowl the owner approved for the freeze"
        )
    box = workspace_box(cfg)
    m = box_margin(b, box, TCP_TO_BOWL_OFFSET_M)
    if m >= 0:
        return None
    return (
        f"the protocol bowl ({b[0]:.3f}, {b[1]:.3f}) m is {-100 * m:.2f} cm outside the safety "
        f"filter's workspace box {box['x']} x {box['y']} m (after the {1000 * TOLERANCE_M:.0f} mm "
        f"placement tolerance and the oracle's {1000 * TCP_TO_BOWL_OFFSET_M:.1f} mm TCP offset): "
        "every carry to it would be rejected (scripts/wo07_bowl_check.py). The filter is never "
        "loosened"
    )
