#!/usr/bin/env python
"""WO-07 / real-0.1: verify the fixed bowl OFFLINE through the REAL safety filter (owner 2026-11-09).

No arm, no device, no sim physics. The PickPlace oracle (``src/data/oracle.py``, the scripted
policy of the sim demos: hover, descend, close, lift, carry, lower, release, retreat, with its own
waypoint heights, tilts and 0.065 rad/step joint cap) runs KINEMATICALLY: every joint command it
issues is assumed tracked exactly, the TCP is the shared FK (``src/env/kinematics.py``), the cube
rides in the gripper from the lift to the release with the offset it had at the grasp, and the
gripper reads at least 0.15 rad while it holds the 3 cm cube. Every command is passed through
``src/robot/safety.py`` ``SafetyFilter`` built from ``configs/real.yaml`` (``make_safety``: the
URDF joint limits less ``urdf_limit_margin_deg``, the 4 deg/step and 60 deg/s caps, the
``workspace_box_m`` TCP box) with the sim FK, measured = the previous command (the reference is
reset per command, so each one is judged on its own step from the pose before it).

Per bowl candidate: every slot used by real-0.1 (5 seen + 20 ring) at every yaw of the frozen
schedules (-45..45 deg, 15 deg steps); the table is phase x slot -> ``pass`` / ``clip`` (with
the joint and kind) / ``reject`` (workspace), worst over the yaws. The gripper's closing command
(-0.17 rad, the sim's GRIPPER_CLOSED) is clipped to the -5 deg floor on EVERY grasp (the owner's
2026-11-08 setting: the gripper never closes past it); that clip is reported separately as
``clip(gripper floor)`` since it is not a property of the bowl position.

``--search`` adds the constrained re-derivation: the analytic rule of ``protocol_real.derive_bowl``
(slot separation >= MIN_SEP + 5 mm against all 36 slots, base keep-out + 5 mm, reach - 5 mm) PLUS
the safety filter's workspace box, shrunk by the 5 mm placement tolerance and by the oracle's
measured TCP-to-bowl offset at lower / release / retreat. When nothing passes it reports the
closest options and what each violates, each verified through the filter. Nothing here loosens
the safety filter.

    uv run scripts/wo07_bowl_check.py --preset real                    # the frozen bowl
    uv run scripts/wo07_bowl_check.py --preset real --search           # + constrained search
    uv run scripts/wo07_bowl_check.py --preset real --bowl 0.205 0.187 # any candidate
    uv run scripts/wo07_bowl_check.py --preset tiny                    # CPU smoke: 2 slots, 1 yaw

Writes ``runs/<id>/record.json`` (results.bowl_check) and prints the tables.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG_DIR, Config, add_config_args, load_config
from src.data import oracle as O
from src.env import success as S
from src.env.kinematics import ArmKinematics
from src.eval import protocol_real as PR
from src.eval import real_grid as G
from src.robot.safety import FkFn, SafetyFilter, load_sim_fk, make_safety
from src.runs import RUNS_DIR, Run, new_run_id

PHASES = ("hover", "descend", "close", "lift", "carry", "lower", "release", "retreat")
BOWL_PHASES = ("lower", "release", "retreat")
CUBE_HALF_M = 0.015  # the 3 cm cubes of real-0.1
GRIP_HOLD_RAD = 0.15  # the gripper reads at least this while it holds the cube
MAX_STEPS = 700
BOX_MARGIN_M = PR.BOWL_MARGIN_M  # the operator's placement tolerance (5 mm), as derive_bowl
STATUS_ORDER = {"pass": 0, "clip(gripper floor)": 1, "clip": 2, "reject": 3, "not reached": 4}


# ------------------------------------------------------------------------------------------------
def _quat_wxyz(r: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) rotation -> (N, 4) unit quaternion (w, x, y, z)."""
    m = r.double()
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cands = torch.stack(
        [
            1 + tr,
            1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
        ],
        -1,
    )
    k = cands.argmax(-1)
    out = torch.zeros(len(m), 4, dtype=torch.float64)
    for i in range(len(m)):
        a = m[i]
        if k[i] == 0:
            w = math.sqrt(float(cands[i, 0])) / 2
            q = [w, (a[2, 1] - a[1, 2]) / (4 * w), (a[0, 2] - a[2, 0]) / (4 * w)]
            q.append((a[1, 0] - a[0, 1]) / (4 * w))
        elif k[i] == 1:
            x = math.sqrt(float(cands[i, 1])) / 2
            q = [(a[2, 1] - a[1, 2]) / (4 * x), x, (a[0, 1] + a[1, 0]) / (4 * x)]
            q.append((a[0, 2] + a[2, 0]) / (4 * x))
        elif k[i] == 2:
            y = math.sqrt(float(cands[i, 2])) / 2
            q = [(a[0, 2] - a[2, 0]) / (4 * y), (a[0, 1] + a[1, 0]) / (4 * y), y]
            q.append((a[1, 2] + a[2, 1]) / (4 * y))
        else:
            z = math.sqrt(float(cands[i, 3])) / 2
            q = [(a[1, 0] - a[0, 1]) / (4 * z), (a[0, 2] + a[2, 0]) / (4 * z)]
            q += [(a[1, 2] + a[2, 1]) / (4 * z), z]
        out[i] = torch.tensor([float(v) for v in q], dtype=torch.float64)
    return (out / out.norm(dim=-1, keepdim=True)).float()


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), z, z, torch.sin(yaw / 2)], -1)


class KinematicScene:
    """The oracle's observation without physics: commands tracked exactly, the cube carried."""

    def __init__(self, kin: ArmKinematics, obj_xy: torch.Tensor, yaw: torch.Tensor, bowl: Any):
        n = len(obj_xy)
        self.kin = kin
        self.q = torch.tensor(O._SEEDS[0].tolist() + [0.0]).repeat(n, 1).float()  # REST_QPOS
        self.obj = torch.cat([obj_xy, torch.full((n, 1), CUBE_HALF_M)], 1).float()
        self.obj_q = _yaw_quat(yaw.float())
        self.goal = torch.tensor([bowl[0], bowl[1], 0.0]).repeat(n, 1).float()
        self.attached = torch.zeros(n, dtype=torch.bool)
        self.released = torch.zeros(n, dtype=torch.bool)
        self.off = torch.zeros(n, 3)

    def tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.kin.fk(self.q[:, :5])
        return t[:, :3, 3], t[:, :3, :3]

    def obs(self) -> dict[str, Any]:
        p, r = self.tcp()
        n = len(p)
        return {
            "agent": {"qpos": self.q.clone()},
            "extra": {
                "tcp_pose": torch.cat([p, _quat_wxyz(r)], 1),
                "obj_pose": torch.cat([self.obj, self.obj_q], 1),
                "goal_pose": torch.cat([self.goal, _yaw_quat(torch.zeros(n))], 1),
                "obj_half_height": torch.full((n,), CUBE_HALF_M),
            },
        }

    def step(self, q_cmd: torch.Tensor, names: Sequence[str]) -> None:
        """The arm reaches ``q_cmd``; the cube rides from the lift to the release; the gripper
        reads >= GRIP_HOLD_RAD while it holds the cube."""
        q = q_cmd.clone().float()
        p, _ = self.tcp()
        for i, nm in enumerate(names):
            holding = nm in ("close", "lift", "carry", "lower") or (
                nm == "release" and float(q[i, 5]) < 0.3
            )
            if nm == "lift" and not bool(self.attached[i]) and not bool(self.released[i]):
                self.attached[i] = True
                self.off[i] = self.obj[i] - p[i]
            if bool(self.attached[i]) and not holding:
                self.attached[i] = False
                self.released[i] = True
                self.obj[i, 2] = self.goal[i, 2] + S.BOWL_FLOOR_THICKNESS + CUBE_HALF_M
            if holding:
                q[i, 5] = max(float(q[i, 5]), GRIP_HOLD_RAD)
        self.q = q
        p2, _ = self.tcp()
        for i in range(len(q)):
            if bool(self.attached[i]):
                self.obj[i] = p2[i] + self.off[i]


def _classify(res: Any) -> tuple[str, list[str]]:
    if res.status == "rejected":
        return "reject", [iv.reason for iv in res.interventions if iv.kind == "workspace_reject"]
    if res.status == "clipped":
        kinds = sorted({f"{iv.kind}:{iv.joint}" for iv in res.interventions})
        if kinds == ["joint_limit:gripper"]:
            return "clip(gripper floor)", kinds
        return "clip", kinds
    return "pass", []


def _worse(a: str, b: str) -> str:
    return a if STATUS_ORDER[a] >= STATUS_ORDER[b] else b


def verify(
    cfg: Config,
    bowl: tuple[float, float],
    slot_ids: Sequence[str],
    yaws_deg: Sequence[float],
    fk: FkFn,
    ik_limits: str = "safety",
) -> dict[str, Any]:
    """Oracle x safety filter for one bowl position (module doc). ``ik_limits``: ``urdf`` = the
    sim oracle unchanged (its IK uses the URDF joint range); ``safety`` = the same oracle with its
    IK bounded by the safety filter's joint limits (URDF less the margin), i.e. a motion that a
    controller respecting the real limits would command."""
    kin = ArmKinematics.from_urdf()
    by_id = {s.id: s for s in G.all_slots()}
    rows = [(sid, float(y)) for sid in slot_ids for y in yaws_deg]
    xy = torch.tensor([[by_id[s].x, by_id[s].y] for s, _ in rows])
    yaw = torch.tensor([math.radians(y) for _, y in rows])
    scene = KinematicScene(kin, xy, yaw, bowl)
    orc = O.Oracle("PickPlace", len(rows))
    filt: SafetyFilter = make_safety(cfg, fk)
    if ik_limits == "safety":
        lo = torch.tensor(filt.config.lower_rad[:5], dtype=orc.kin.lower.dtype)
        hi = torch.tensor(filt.config.upper_rad[:5], dtype=orc.kin.upper.dtype)
        orc.kin.lower = torch.maximum(orc.kin.lower, lo)
        orc.kin.upper = torch.minimum(orc.kin.upper, hi)
    elif ik_limits != "urdf":
        raise ValueError(f"ik_limits {ik_limits!r}: urdf | safety")
    orc.reset(scene.obs())
    box_lo = np.array(filt.config.workspace_lo_m)
    box_hi = np.array(filt.config.workspace_hi_m)
    cells: dict[tuple[str, str], str] = {}
    detail: dict[tuple[str, str], set[str]] = {}
    clip_deg: dict[str, float] = {}  # "<phase> <kind>:<joint>" -> largest |requested - applied|
    clip_slots: dict[str, set[str]] = {}
    outside_m: dict[str, float] = {}  # phase -> largest TCP distance outside the box (rejects)
    tcp_bowl_off = 0.0
    box_margin = {"x_lo": 1.0, "x_hi": 1.0, "y_lo": 1.0, "y_hi": 1.0, "z_lo": 1.0, "z_hi": 1.0}
    bowl_box_margin = dict(box_margin)
    reached = torch.zeros(len(rows), dtype=torch.bool)
    t = 0.0
    ref = [scene.q[i].double().numpy().copy() for i in range(len(rows))]
    for _ in range(MAX_STEPS):
        names = [orc.names[int(i)] for i in orc.phase.tolist()]
        if all(n == "done" for n in names):
            break
        q_cmd = orc.act(scene.obs())
        t += 1.0 / float(filt.config.control_hz)
        p_cmd = kin.fk(q_cmd[:, :5])[:, :3, 3]
        for i, nm in enumerate(names):
            if nm == "done":
                continue
            filt.reset_reference()  # judged from the last filtered command (module doc)
            res = filt.filter(q_cmd[i].double().numpy(), ref[i], t)
            if res.command is not None:
                ref[i] = res.command
            status, why = _classify(res)
            key = (rows[i][0], nm)
            cells[key] = _worse(status, cells.get(key, "pass"))
            if why:
                detail.setdefault(key, set()).update(why)
            for iv in res.interventions:
                if iv.joint is not None and iv.requested_deg is not None:
                    tag = f"{nm} {iv.kind}:{iv.joint}"
                    mag = abs(float(iv.requested_deg) - float(iv.applied_deg or 0.0))
                    clip_deg[tag] = max(clip_deg.get(tag, 0.0), mag)
                    clip_slots.setdefault(tag, set()).add(rows[i][0])
            if status == "reject":
                d_out = filt.box_distance(p_cmd[i].double().numpy())
                outside_m[nm] = max(outside_m.get(nm, 0.0), d_out)
            pc = p_cmd[i].double().numpy()
            m = {
                "x_lo": pc[0] - box_lo[0],
                "x_hi": box_hi[0] - pc[0],
                "y_lo": pc[1] - box_lo[1],
                "y_hi": box_hi[1] - pc[1],
                "z_lo": pc[2] - box_lo[2],
                "z_hi": box_hi[2] - pc[2],
            }
            for k, v in m.items():
                box_margin[k] = min(box_margin[k], float(v))
                if nm in BOWL_PHASES:
                    bowl_box_margin[k] = min(bowl_box_margin[k], float(v))
            if nm in BOWL_PHASES:
                d = float(np.linalg.norm(pc[:2] - np.array(bowl)))
                tcp_bowl_off = max(tcp_bowl_off, d)
            if nm == "retreat":
                reached[i] = True
        scene.step(q_cmd, names)
    for i, (sid, _) in enumerate(rows):
        for nm in PHASES:
            cells.setdefault((sid, nm), "not reached")
        if not reached[i]:
            cells[(sid, "retreat")] = _worse("not reached", cells[(sid, "retreat")])
    table = {sid: {nm: cells[(sid, nm)] for nm in PHASES} for sid in slot_ids}
    worst = {
        nm: max((table[s][nm] for s in slot_ids), key=STATUS_ORDER.__getitem__) for nm in PHASES
    }
    reasons = {
        f"{s}/{nm}": sorted(v)[:3]
        for (s, nm), v in detail.items()
        if cells[(s, nm)] != "clip(gripper floor)"
    }
    bowl_ok = all(STATUS_ORDER[worst[nm]] <= 1 for nm in ("carry", *BOWL_PHASES))
    in_bowl = (scene.obj[:, :2] - torch.tensor(bowl)).norm(dim=-1) < S.BOWL_FLOOR_RADIUS
    placed = sorted({rows[i][0] for i in range(len(rows)) if bool(scene.released[i] & in_bowl[i])})
    return {
        "bowl_xy_m": [round(bowl[0], 4), round(bowl[1], 4)],
        "ik_limits": ik_limits,
        "cube_released_over_bowl_floor": f"{int((scene.released & in_bowl).sum())}/{len(rows)}",
        "slots_never_placed": sorted(set(slot_ids) - set(placed)),
        "slots": list(slot_ids),
        "yaws_deg": [float(y) for y in yaws_deg],
        "table": table,
        "worst_by_phase": worst,
        "reasons": dict(sorted(reasons.items())[:40]),
        "clips": {
            tag: {"max_deg": round(v, 2), "slots": sorted(clip_slots[tag])}
            for tag, v in sorted(clip_deg.items())
        },
        "reject_max_outside_box_m": {k: round(v, 4) for k, v in outside_m.items()},
        "passes_carry_and_bowl_phases": bowl_ok,
        "tcp_to_bowl_max_m": round(tcp_bowl_off, 4),
        "tcp_box_margin_m": {k: round(v, 4) for k, v in box_margin.items()},
        "tcp_box_margin_bowl_phases_m": {k: round(v, 4) for k, v in bowl_box_margin.items()},
        "box_m": {"lo": box_lo.tolist(), "hi": box_hi.tolist()},
    }


# ------------------------------------------------------------------------------------------------
def analytic(bowl: Sequence[float]) -> dict[str, float]:
    """derive_bowl's margins (m; >= 0 passes) against all 36 slots."""
    k = PR.bowl_constraints()
    pts = np.array([(s.x, s.y) for s in G.all_slots()])
    c = np.array(bowl, dtype=np.float64)
    d = float(np.linalg.norm(pts - c, axis=1).min())
    r = float(np.linalg.norm(c))
    m = PR.BOWL_MARGIN_M
    return {
        "min_slot_distance_m": round(d, 4),
        "base_distance_m": round(r, 4),
        "sep_margin_m": round(d - (k["min_sep_m"] + m), 4),
        "keepout_margin_m": round(r - (k["bowl_keepout_m"] + m), 4),
        "reach_margin_m": round(k["reach_m"] - m - r, 4),
    }


def central_slot_for(bowl: Sequence[float]) -> dict[str, Any]:
    """``protocol_real.central_slot``'s rule for any bowl: of C3 C4 D3 D4, the slot whose bowl
    distance is nearest the corners' mean (ties: candidate order)."""
    by_id = {s.id: s for s in G.all_slots()}
    b = np.array(bowl, dtype=np.float64)

    def bd(sid: str) -> float:
        return float(np.linalg.norm(b - np.array([by_id[sid].x, by_id[sid].y])))

    mean = sum(bd(s) for s in PR.CORNER_SLOTS) / len(PR.CORNER_SLOTS)
    pick = min(
        PR.CENTRAL_CANDIDATES,
        key=lambda s: (round(abs(bd(s) - mean), 9), PR.CENTRAL_CANDIDATES.index(s)),
    )
    return {
        "central_slot": pick,
        "corner_mean_m": round(mean, 4),
        "candidates_m": {s: round(bd(s), 4) for s in PR.CENTRAL_CANDIDATES},
    }


def box_margin(bowl: Sequence[float], cfg: Config, tcp_off: float) -> float:
    """Smallest margin (m) of the bowl centre to the workspace box's x / y faces, after the 5 mm
    placement tolerance and the oracle's TCP-to-bowl offset (>= 0 passes)."""
    box = cfg["robot"]["safety"]["workspace_box_m"]
    x, y = float(bowl[0]), float(bowl[1])
    pad = BOX_MARGIN_M + tcp_off
    return round(
        min(
            x - float(box["x"][0]) - pad,
            float(box["x"][1]) - x - pad,
            y - float(box["y"][0]) - pad,
            float(box["y"][1]) - y - pad,
        ),
        4,
    )


def search(cfg: Config, tcp_off: float) -> dict[str, Any]:
    """derive_bowl's lattice with the box constraint added (module doc), and the closest options
    when nothing passes: (a) the rule as frozen (separation kept), smallest box violation; (b) the
    box kept, largest separation; (c) both kept except reach (or keep-out)."""
    (x0, x1), (y0, y1) = PR.BOWL_SEARCH_M
    st = PR.BOWL_LATTICE_M
    xs = np.round(x0 + st * np.arange(round((x1 - x0) / st) + 1), 4)
    ys = np.round(y0 + st * np.arange(round((y1 - y0) / st) + 1), 4)
    centre = np.array(PR.P.SEEN_REGION.center)
    rows = []
    for x in xs:
        for y in ys:
            a = analytic((x, y))
            rows.append(
                {
                    "xy": (float(x), float(y)),
                    **a,
                    "box_margin_m": box_margin((x, y), cfg, tcp_off),
                    "grid_centre_m": round(float(np.linalg.norm(np.array([x, y]) - centre)), 4),
                }
            )
    keys = ("sep_margin_m", "keepout_margin_m", "reach_margin_m", "box_margin_m")

    def ok(r: dict[str, Any], skip: Sequence[str] = ()) -> bool:
        return all(r[k] >= -1e-9 for k in keys if k not in skip)

    passing = sorted((r for r in rows if ok(r)), key=lambda r: (r["grid_centre_m"], -r["xy"][1]))
    out: dict[str, Any] = {"tcp_offset_m": tcp_off, "box_pad_m": BOX_MARGIN_M + tcp_off}
    out["passing"] = passing[:5]
    if passing:
        return out
    a = sorted(
        (r for r in rows if ok(r, ("box_margin_m",))),
        key=lambda r: (-r["box_margin_m"], r["grid_centre_m"]),
    )
    b = sorted(
        (r for r in rows if ok(r, ("sep_margin_m",))),
        key=lambda r: (-r["sep_margin_m"], r["grid_centre_m"], -r["xy"][1]),
    )
    c = sorted(
        (r for r in rows if ok(r, ("reach_margin_m", "keepout_margin_m"))),
        key=lambda r: (-min(r["reach_margin_m"], r["keepout_margin_m"]), r["grid_centre_m"]),
    )
    out["closest"] = {
        "a_rule_kept_box_violated": a[:2],
        "b_box_kept_separation_short": b[:2],
        "c_sep_and_box_kept_reach_or_keepout_violated": c[:2],
    }
    return out


def _fmt_table(v: dict[str, Any]) -> str:
    short = {
        "pass": "pass",
        "clip(gripper floor)": "pass*",
        "clip": "CLIP",
        "reject": "REJECT",
        "not reached": "n/r",
    }
    lines = ["| slot | " + " | ".join(PHASES) + " |", "|---" * (len(PHASES) + 1) + "|"]
    for sid in v["slots"]:
        lines.append(f"| {sid} | " + " | ".join(short[v["table"][sid][nm]] for nm in PHASES) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--bowl", type=float, nargs=2, action="append", default=None)
    parser.add_argument("--search", action="store_true")
    parser.add_argument(
        "--ik-limits", nargs="+", choices=("safety", "urdf"), default=["safety", "urdf"]
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset not in ("real", "tiny"):
        parser.error("--preset real (or tiny: a CPU smoke)")
    logging.getLogger("src.robot.safety").setLevel(logging.ERROR)  # every verdict is tabled
    config = args.config or (CONFIG_DIR / "real.yaml" if args.preset == "tiny" else None)
    cfg = load_config(args.preset, config)
    tiny = args.preset == "tiny"
    slots = ["C4", "A1"] if tiny else [*PR.SEEN_SLOTS, *PR.ring_order()]
    yaws = [0.0] if tiny else [float(y) for y in PR.YAWS_DEG]
    bowls = [tuple(b) for b in args.bowl] if args.bowl else [PR.BOWL_XY_M]
    fk = load_sim_fk()
    out: dict[str, Any] = {"frozen_bowl_xy_m": list(PR.BOWL_XY_M), "results": []}
    with Run(
        work_order="WO-07",
        script=__file__,
        cfg=cfg,
        device="cpu",
        runs_dir=args.runs_dir,
        run_id=new_run_id("WO-07", f"bowl-check-{args.preset}"),
    ) as run:
        for b, lim in [(b, lim) for b in bowls for lim in args.ik_limits]:
            v = verify(cfg, (float(b[0]), float(b[1])), slots, yaws, fk, lim)
            v["analytic"] = analytic(b)
            v["central_slot"] = central_slot_for(b)
            v["box_margin_m"] = box_margin(b, cfg, v["tcp_to_bowl_max_m"])
            out["results"].append(v)
            print(
                f"\nbowl {v['bowl_xy_m']} (oracle IK limits: {lim}): {v['analytic']}, box margin "
                f"{v['box_margin_m']}; cube over the bowl floor {v['cube_released_over_bowl_floor']}"
            )
            print(_fmt_table(v))
            print(f"worst by phase {v['worst_by_phase']}; central slot {v['central_slot']}")
            print(
                f"TCP-to-bowl max {v['tcp_to_bowl_max_m']} m; box margins at the bowl "
                f"{v['tcp_box_margin_bowl_phases_m']}"
            )
            for tag, c in v["clips"].items():
                print(
                    f"  clip {tag}: max {c['max_deg']} deg at {len(c['slots'])} slots {c['slots']}"
                )
            print(f"  rejects: max TCP outside the box {v['reject_max_outside_box_m']}")
        if args.search:
            # the offset measured at the FIRST bowl (the frozen one by default): a far candidate
            # the IK cannot reach exactly would inflate it for every position
            off = out["results"][0]["tcp_to_bowl_max_m"]
            out["search"] = search(cfg, off)
            print(f"\nsearch (TCP offset {off} m): passing {out['search']['passing']}")
            for k, r in (out["search"].get("closest") or {}).items():
                print(f"  {k}: {r}")
        out["central_slot"] = PR.central_slot()
        run.results["bowl_check"] = out
    return out


if __name__ == "__main__":
    main()
