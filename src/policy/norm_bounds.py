"""Action / proprio normalisation bounds for the SO-101 -- NOT taken from the demos.

OFT maps each joint target to 256 bins over [low, high] and clips outside it, so the bounds are
the policy's whole reachable action space. Bounds from 30 demos would make the "shifted
positions" cell unreachable and stop RL from exploring past the demos. Instead:

  arm joints (pan, lift, elbow, wrist_flex): joint limits ∩ the range over which the TCP can be
      inside the task workspace (FK over uniformly sampled joint configurations, from the URDF),
      with the gripper approach axis within `max_tilt_deg` of straight down;
  wrist_roll: a π-wide window (a parallel gripper is 180°-symmetric, so any grasp yaw is
      reachable inside it; object yaw is randomised in ±45°);
  gripper: the commanded open/closed range;
  then ∪ extra configurations (the episode start pose, optionally the demo actions), + margin,
  ∩ joint limits; and every joint must resolve to <= `max_deg_per_bin` (default 1°/bin).

Proprio uses the same bounds (same joint space). `bounds_from_cfg` reads `policy.norm`.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.config import REPO_ROOT, Config
from src.policy.oft.actions import N_BINS, NormBounds

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ARM_CHAIN = JOINTS[:5]


@dataclass(frozen=True)
class WorkspaceSpec:
    x: tuple[float, float]  # TCP x range (m, robot base frame)
    y: tuple[float, float]
    z: tuple[float, float]
    max_tilt_deg: float  # max angle between the TCP approach (z) axis and straight down
    wrist_roll: tuple[float, float]
    gripper: tuple[float, float]
    include_qpos: tuple[tuple[float, ...], ...] = ()
    margin_rad: float = 0.05
    trim_quantile: float = (
        0.005  # drop the rarest contorted FK solutions (e.g. reaching over the base)
    )
    samples: int = 400_000
    seed: int = 0
    tip_link: str = "tcp_link"
    base_link: str = "base_link"


@dataclass
class BoundsReport:
    bounds: NormBounds
    joint_limits: dict[str, tuple[float, float]]
    kept_fraction: float
    deg_per_bin: list[float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "low": list(self.bounds.low),
            "high": list(self.bounds.high),
            "joint_limits": {k: list(v) for k, v in self.joint_limits.items()},
            "kept_fraction": self.kept_fraction,
            "deg_per_bin": self.deg_per_bin,
            "notes": self.notes,
        }


# ------------------------------------------------------------------------------- URDF
def _floats(s: str | None, default: str) -> list[float]:
    return [float(v) for v in (s or default).split()]


def _rpy_matrix(r: float, p: float, y: float) -> torch.Tensor:
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return torch.tensor(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=torch.float64,
    )


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    origin: torch.Tensor  # (4, 4)
    axis: torch.Tensor  # (3,)
    limits: tuple[float, float] | None


def _parse(urdf: Path) -> tuple[dict[str, _Joint], dict[str, tuple[str, str]]]:
    root = ET.parse(urdf).getroot()
    joints: dict[str, _Joint] = {}
    links: dict[str, tuple[str, str]] = {}  # child link -> (joint, parent link)
    for j in root.findall("joint"):
        name = str(j.get("name"))
        o = j.find("origin")
        tf = torch.eye(4, dtype=torch.float64)
        tf[:3, :3] = _rpy_matrix(*_floats(None if o is None else o.get("rpy"), "0 0 0"))
        tf[:3, 3] = torch.tensor(_floats(None if o is None else o.get("xyz"), "0 0 0"))
        ax = j.find("axis")
        axis = torch.tensor(
            _floats(None if ax is None else ax.get("xyz"), "1 0 0"), dtype=torch.float64
        )
        lim = j.find("limit")
        limits = None
        if lim is not None and j.get("type") in ("revolute", "prismatic"):
            limits = (float(lim.get("lower", "0")), float(lim.get("upper", "0")))
        joints[name] = _Joint(name, str(j.get("type")), tf, axis / axis.norm(), limits)
        parent, child = j.find("parent"), j.find("child")
        if parent is None or child is None:
            raise ValueError(f"{urdf}: joint {name} lacks parent/child")
        links[str(child.get("link"))] = (name, str(parent.get("link")))
    return joints, links


def urdf_joint_limits(urdf: Path, names: Sequence[str] = JOINTS) -> dict[str, tuple[float, float]]:
    joints, _ = _parse(urdf)
    out = {}
    for n in names:
        lim = joints[n].limits
        if lim is None:
            raise ValueError(f"{urdf}: joint {n} has no limits")
        out[n] = lim
    return out


def _axis_angle(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """(N,) angles about a unit axis -> (N, 4, 4) rotations (Rodrigues)."""
    k = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=torch.float64,
    )
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    rot = torch.eye(3, dtype=torch.float64) + s * k + (1 - c) * (k @ k)
    out = torch.eye(4, dtype=torch.float64).repeat(q.shape[0], 1, 1)
    out[:, :3, :3] = rot
    return out


def forward_kinematics(
    urdf: Path, q: torch.Tensor, chain: Sequence[str] = ARM_CHAIN, *, base: str, tip: str
) -> torch.Tensor:
    """(N, len(chain)) joint values -> (N, 4, 4) tip pose in the base frame."""
    joints, links = _parse(urdf)
    path: list[str] = []
    link = tip
    while link != base:
        if link not in links:
            raise ValueError(f"{urdf}: no kinematic path from {base} to {tip}")
        jname, link = links[link]
        path.append(jname)
    path.reverse()
    idx = {n: i for i, n in enumerate(chain)}
    tf = torch.eye(4, dtype=torch.float64).repeat(q.shape[0], 1, 1)
    for jname in path:
        j = joints[jname]
        tf = tf @ j.origin
        if j.kind in ("revolute", "continuous"):
            if jname not in idx:
                raise ValueError(f"joint {jname} on the {base}->{tip} path is not in {chain}")
            tf = tf @ _axis_angle(j.axis, q[:, idx[jname]].double())
        elif j.kind != "fixed":
            raise ValueError(f"unsupported joint type {j.kind} ({jname})")
    return tf


# ------------------------------------------------------------------------------ bounds
def workspace_bounds(
    urdf: Path,
    spec: WorkspaceSpec,
    *,
    extra_actions: torch.Tensor | None = None,
    max_deg_per_bin: float = 1.0,
) -> BoundsReport:
    limits = urdf_joint_limits(urdf)
    g = torch.Generator().manual_seed(spec.seed)
    lo = torch.tensor([limits[n][0] for n in ARM_CHAIN], dtype=torch.float64)
    hi = torch.tensor([limits[n][1] for n in ARM_CHAIN], dtype=torch.float64)
    q = lo + (hi - lo) * torch.rand(spec.samples, len(ARM_CHAIN), generator=g, dtype=torch.float64)
    tf = forward_kinematics(urdf, q, base=spec.base_link, tip=spec.tip_link)
    p, approach = tf[:, :3, 3], tf[:, :3, 2]
    keep = (
        (p[:, 0] >= spec.x[0])
        & (p[:, 0] <= spec.x[1])
        & (p[:, 1] >= spec.y[0])
        & (p[:, 1] <= spec.y[1])
        & (p[:, 2] >= spec.z[0])
        & (p[:, 2] <= spec.z[1])
        & (-approach[:, 2] >= math.cos(math.radians(spec.max_tilt_deg)))
    )
    if int(keep.sum()) < 100:
        raise ValueError(f"only {int(keep.sum())} FK samples reach the workspace; check the spec")
    kept = q[keep][:, :4]
    t = spec.trim_quantile
    low = [float(v) for v in torch.quantile(kept, t, dim=0)] + [spec.wrist_roll[0], spec.gripper[0]]
    high = [float(v) for v in torch.quantile(kept, 1 - t, dim=0)] + [
        spec.wrist_roll[1],
        spec.gripper[1],
    ]
    notes = [f"arm joints: FK-workspace quantiles [{t}, {1 - t}] of {int(keep.sum())} samples"]
    extra = [torch.tensor(qp, dtype=torch.float64) for qp in spec.include_qpos]
    if extra_actions is not None:
        extra.append(extra_actions.reshape(-1, len(JOINTS)).double())
    for e in extra:
        e = e.reshape(-1, len(JOINTS))
        low = [min(a, float(b)) for a, b in zip(low, e.min(0).values, strict=True)]
        high = [max(a, float(b)) for a, b in zip(high, e.max(0).values, strict=True)]
    low = [max(v - spec.margin_rad, limits[n][0]) for v, n in zip(low, JOINTS, strict=True)]
    high = [min(v + spec.margin_rad, limits[n][1]) for v, n in zip(high, JOINTS, strict=True)]
    bounds = NormBounds.from_lists(low, high)
    dpb = bounds.deg_per_bin(N_BINS)
    bad = [f"{n}: {d:.3f}°/bin" for n, d in zip(JOINTS, dpb, strict=True) if d > max_deg_per_bin]
    if bad:
        raise ValueError(f"action resolution worse than {max_deg_per_bin}°/bin: {bad}")
    return BoundsReport(bounds, limits, float(keep.float().mean()), dpb, notes)


def demo_coverage(bounds: NormBounds, actions: torch.Tensor) -> dict[str, Any]:
    """How much of the demo action data falls outside `bounds` (would be clipped)."""
    a = actions.reshape(-1, bounds.dim).float()
    lo, hi = torch.tensor(bounds.low), torch.tensor(bounds.high)
    outside = (a < lo) | (a > hi)
    return {
        "frac_outside": float(outside.any(-1).float().mean()),
        "per_joint_frac_outside": [float(v) for v in outside.float().mean(0)],
        "demo_min": [float(v) for v in a.min(0).values],
        "demo_max": [float(v) for v in a.max(0).values],
    }


def spec_from_cfg(norm: Config) -> WorkspaceSpec:
    def pair(key: str) -> tuple[float, float]:
        v = norm[key]
        return float(v[0]), float(v[1])

    return WorkspaceSpec(
        x=pair("workspace_x"),
        y=pair("workspace_y"),
        z=pair("workspace_z"),
        max_tilt_deg=float(norm.get("max_tilt_deg", 75.0)),
        wrist_roll=pair("wrist_roll"),
        gripper=pair("gripper"),
        include_qpos=tuple(tuple(float(x) for x in q) for q in norm.get("include_qpos", [])),
        margin_rad=float(norm.get("margin_rad", 0.05)),
        trim_quantile=float(norm.get("trim_quantile", 0.005)),
        samples=int(norm.get("samples", 400_000)),
        seed=int(norm.get("seed", 0)),
        tip_link=str(norm.get("tip_link", "tcp_link")),
        base_link=str(norm.get("base_link", "base_link")),
    )


def bounds_from_cfg(
    cfg: Config, extra_actions: torch.Tensor | None = None
) -> tuple[NormBounds, NormBounds]:
    """(action, proprio) bounds from `policy.norm` (source: workspace | fixed)."""
    norm = cfg["policy"]["norm"]
    source = norm.get("source", "workspace")
    if source == "fixed":
        b = NormBounds.from_lists(norm["low"], norm["high"])
    elif source == "workspace":
        urdf = Path(norm["urdf"])
        urdf = urdf if urdf.is_absolute() else REPO_ROOT / urdf
        report = workspace_bounds(
            urdf,
            spec_from_cfg(norm),
            extra_actions=extra_actions,
            max_deg_per_bin=float(norm.get("max_deg_per_bin", 1.0)),
        )
        b = report.bounds
    else:
        raise ValueError(f"policy.norm.source must be workspace|fixed, got {source!r}")
    return b, b
