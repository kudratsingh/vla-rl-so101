"""Scripted oracle per task: privileged object state + numeric IK -> absolute joint targets.

No mplib / pinocchio (Mac-safe). Batched over sub-envs. Each task is a small per-env state
machine of TCP waypoints; between waypoints the TCP moves on a straight line (per-step IK, warm
started) and every commanded joint moves at most MAX_JOINT_STEP rad per control step (the
real-arm safety cap is 4°/step, configs/real.yaml), so demos are executable on the real arm.

Observation keys used: obs["agent"]["qpos"], obs["extra"]{tcp_pose, obj_pose, goal_pose,
obj_half_height} — the privileged state the env exposes for this purpose only.

Speed profiles (`OracleProfile`, `data.oracle_profile`): "v1" is the oracle that recorded the
0.1-era demo sets (KudratSingh22/vla-rl-so101-demos, -demos-25): 1.5 cm/step nominal approach
(the joint cap holds the real TCP to ~0.9 cm/step), 1 cm/step descent, so the last 10 cm before
the grasp take ~1.5 policy chunks of 8 steps; it is the default. "v2" (opt-in via
`data.oracle_profile: v2`, configs/demos_v2.yaml; review 2026-11-07 section 3.2) runs
the last `approach_slow_radius` (6 cm) of the approach and the descent to the grasp / push height
at 0.5 cm/step: ~2.6-3.0 chunks in the last 10 cm (scripts/wo01_oracle_eval.py near_decisions).
Everything else (waypoints, carry, release, retries) is identical; v1 reproduces the 0.1 oracle
bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from src.env import success as S
from src.env.kinematics import ArmKinematics
from src.env.robot import GRIPPER_CLOSED, GRIPPER_OPEN

MAX_JOINT_STEP = 0.065  # rad per control step (< 4°)
GRIPPER_RELEASE = 0.45
GRASP_CLEARANCE = 0.004  # shift toward the fixed jaw while descending (its tip clears the face)
HOVER = 0.065  # TCP height above the object centre before descending
CARRY_Z = 0.072  # TCP height while carrying (cube bottom clears the 3.6 cm bowl rim by ~1.6 cm)
RETREAT_Z = 0.11
# approach tilts tried in order when top-down is unreachable (+ = gripper tip leans away from the
# base, needed far out / high; - = leans toward the base, needed close to the base)
TILTS_DEG = (0.0, 10.0, 20.0, 30.0, 45.0, -10.0, -20.0, -30.0, -45.0)
PUSHER_OFFSET = 0.012  # closed-finger pad centre sits this far from the TCP along -closing
PUSHER_HALF = 0.007  # half thickness of the closed fingers along the push direction
PHASE_TIMEOUT = 45
MAX_RETRIES = 2


@dataclass(frozen=True)
class OracleProfile:
    """TCP speeds (m per control step) for the approach to the object; see the module docstring."""

    hover_speed: float = 0.015  # transit to the point HOVER above the object (push: behind it)
    approach_slow_radius: float = 0.0  # last metres of that transit run at approach_speed
    approach_speed: float = 0.015
    descend_speed: float = 0.01  # hover point -> grasp / push height


ORACLE_PROFILES: dict[str, OracleProfile] = {
    "v1": OracleProfile(),
    "v2": OracleProfile(approach_slow_radius=0.06, approach_speed=0.005, descend_speed=0.005),
}
# v1 by default so configs that do not set data.oracle_profile (sim.yaml, sft_a4.yaml) keep
# recording the 0.1 oracle under their unchanged config hash; configs/demos_v2.yaml opts in to v2.
DEFAULT_PROFILE = "v1"


_SEEDS = torch.tensor(
    [
        [0.0, -1.30, 0.60, 1.30, 0.0],
        [0.0, 0.00, 0.70, 1.00, 0.0],
        [0.0, -0.50, 1.20, 0.50, 0.0],
        [0.0, -0.30, 0.30, 1.50, 0.0],
    ]
)


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def _unit(v: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return out


def _horiz(v: torch.Tensor) -> torch.Tensor:
    out = v.clone()
    out[..., 2] = 0
    return _unit(out)


def _approach(pos: torch.Tensor, tilt: torch.Tensor) -> torch.Tensor:
    """Approach (TCP z) tilted outward by `tilt` rad from straight down, in the radial plane."""
    az = torch.atan2(pos[:, 1], pos[:, 0])
    return torch.stack(
        [torch.cos(az) * torch.sin(tilt), torch.sin(az) * torch.sin(tilt), -torch.cos(tilt)], -1
    )


def _face_closing(yaw: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Horizontal cube-face normal (from yaw, 4-fold) closest to `ref` (sign included)."""
    cands = torch.stack([yaw + k * math.pi / 2 for k in range(4)], -1)  # (N, 4)
    vecs = torch.stack([torch.cos(cands), torch.sin(cands), torch.zeros_like(cands)], -1)
    best = (vecs * ref[:, None, :]).sum(-1).argmax(-1)
    return vecs[torch.arange(len(yaw)), best]


class Oracle:
    """Batched scripted policy. `reset(obs)` at episode start, then `act(obs)` every step."""

    # phase ids per task (index into the phase list); "done" holds still
    PHASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "PickPlace": (
            "hover",
            "descend",
            "close",
            "lift",
            "carry",
            "lower",
            "release",
            "retreat",
            "done",
        ),
        "Stack": (
            "hover",
            "descend",
            "close",
            "lift",
            "carry",
            "lower",
            "release",
            "retreat",
            "done",
        ),
        "Push": ("hover", "descend", "push", "retreat", "done"),
    }

    def __init__(
        self,
        task: str,
        num_envs: int,
        device: torch.device | str = "cpu",
        profile: str | OracleProfile = DEFAULT_PROFILE,
    ) -> None:
        if task not in self.PHASES:
            raise ValueError(f"unknown task {task!r}")
        if isinstance(profile, str):
            if profile not in ORACLE_PROFILES:
                raise ValueError(
                    f"unknown oracle profile {profile!r}; one of {list(ORACLE_PROFILES)}"
                )
            profile = ORACLE_PROFILES[profile]
        self.profile = profile
        self.task = task
        self.n = num_envs
        # The state machine is small and branchy: run it on CPU, return actions on `out_device`.
        self.device = torch.device("cpu")
        self.out_device = torch.device(device)
        self.kin = ArmKinematics.from_urdf(device=self.device)
        self.names = self.PHASES[task]
        self.seeds = _SEEDS.to(self.device)

    # ------------------------------------------------------------------ helpers
    def _state(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        ex = obs["extra"]

        def f(t: Any) -> torch.Tensor:
            return torch.as_tensor(t).detach().to(self.device).float()

        st = {
            "qpos": f(obs["agent"]["qpos"]),
            "tcp": f(ex["tcp_pose"]),
            "obj": f(ex["obj_pose"]),
            "goal": f(ex["goal_pose"]),
            "half": f(ex["obj_half_height"]),
        }
        st["tcp_R"] = _rotmat(st["tcp"][:, 3:])
        return st

    def _tilt_of(self, tcp_R: torch.Tensor) -> torch.Tensor:
        a = tcp_R[:, :, 2]
        return torch.acos((-a[:, 2]).clamp(-1, 1))

    def _feasible_tilt(
        self, pos: torch.Tensor, closing: torch.Tensor, q0: torch.Tensor
    ) -> torch.Tensor:
        """Smallest outward tilt in TILTS_DEG for which the pose is reachable (per row)."""
        n = pos.shape[0]
        tilts = torch.tensor([math.radians(t) for t in TILTS_DEG], device=self.device)
        k = len(tilts)
        p = pos.repeat(k, 1)
        tl = tilts.repeat_interleave(n)
        a = _approach(p, tl)
        c = closing.repeat(k, 1)
        c = _unit(c - (c * a).sum(-1, keepdim=True) * a)
        _, err = self.kin.ik_multi(p, a, c, q0.repeat(k, 1), self.seeds, iters=40)
        err = err.reshape(k, n)
        ok = err < 2e-3
        best = torch.where(ok.any(0), ok.float().argmax(0), err.argmin(0))
        return tilts[best]

    # ------------------------------------------------------------------ API
    def reset(self, obs: dict[str, Any]) -> None:
        st = self._state(obs)
        n, dev = self.n, self.device
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.t_phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.retries = torch.zeros(n, dtype=torch.long, device=dev)
        self.q_cmd = st["qpos"].clone()
        self.prev_qpos = st["qpos"].clone()
        self.q_ik = st["qpos"][:, :5].clone()
        self.grip_goal = st["qpos"][:, 5].clone()
        # per-env segment: start/goal TCP position, tilt, closing; progress s in [0, 1]
        self.p0 = st["tcp"][:, :3].clone()
        self.p1 = self.p0.clone()
        self.tilt0 = self._tilt_of(st["tcp_R"])
        self.tilt1 = self.tilt0.clone()
        self.c0 = st["tcp_R"][:, :, 1].clone()
        self.c1 = self.c0.clone()
        self.speed = torch.full((n,), 0.01, device=dev)
        # per segment: the last `slow_radius` metres run at min(speed, slow_speed)
        self.slow_radius = torch.zeros(n, device=dev)
        self.slow_speed = torch.full((n,), math.inf, device=dev)
        self.s = torch.ones(n, device=dev)
        self.grasp_off = torch.zeros(n, 3, device=dev)  # obj - tcp while grasped
        self.push_dir = torch.zeros(n, 3, device=dev)
        self._enter(torch.ones(n, dtype=torch.bool, device=dev), st)

    def _set_segment(
        self,
        mask: torch.Tensor,
        st: dict[str, torch.Tensor],
        goal: torch.Tensor,
        closing: torch.Tensor,
        speed: float,
        tilt: torch.Tensor | None = None,
        *,
        slow_radius: float = 0.0,
        slow_speed: float = math.inf,
    ) -> None:
        if not bool(mask.any()):
            return
        self.p0[mask] = st["tcp"][mask, :3]
        self.tilt0[mask] = self._tilt_of(st["tcp_R"][mask])
        self.c0[mask] = st["tcp_R"][mask, :, 1]
        self.p1[mask] = goal[mask]
        c = closing[mask]
        c = torch.where(((c * self.c0[mask]).sum(-1, keepdim=True) < 0), -c, c)
        self.c1[mask] = c
        if tilt is None:
            tilt = self._feasible_tilt(goal[mask], c, self.q_cmd[mask, :5])
        else:
            tilt = tilt[mask]
        self.tilt1[mask] = tilt
        self.speed[mask] = speed
        self.slow_radius[mask] = slow_radius
        self.slow_speed[mask] = slow_speed
        self.s[mask] = 0.0

    def _enter(self, mask: torch.Tensor, st: dict[str, torch.Tensor]) -> None:
        """Set up the segment for every env in `mask` according to its (new) phase."""
        self.t_phase[mask] = 0
        name = [self.names[int(i)] for i in self.phase.tolist()]
        obj_p, goal_p = st["obj"][:, :3], st["goal"][:, :3]
        obj_yaw, goal_yaw = _yaw(st["obj"][:, 3:]), _yaw(st["goal"][:, 3:])
        tcp_p = st["tcp"][:, :3]
        cur_c = st["tcp_R"][:, :, 1]
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        zero_tilt = torch.zeros(self.n, device=self.device)
        pr = self.profile
        slow = (pr.approach_slow_radius, pr.approach_speed)

        def sel(nm: str) -> torch.Tensor:
            return mask & torch.tensor([x == nm for x in name], device=self.device)

        if self.task in ("PickPlace", "Stack"):
            m = sel("hover")
            if bool(m.any()):
                c = _face_closing(obj_yaw, _horiz(cur_c))  # least wrist roll; sign = current
                g = obj_p - GRASP_CLEARANCE * c + up * HOVER
                self._set_segment(
                    m, st, g, c, pr.hover_speed, slow_radius=slow[0], slow_speed=slow[1]
                )
                self.grip_goal[m] = GRIPPER_OPEN
            m = sel("descend")
            if bool(m.any()):
                g = self.p1.clone()
                g[:, 2] = obj_p[:, 2] + 0.003
                self._set_segment(m, st, g, self.c1, pr.descend_speed, zero_tilt)
            m = sel("close")
            if bool(m.any()):
                self.grip_goal[m] = GRIPPER_CLOSED
            m = sel("lift")
            if bool(m.any()):
                g = tcp_p.clone()
                g[:, 2] = CARRY_Z
                self._set_segment(m, st, g, cur_c, 0.012)
            m = sel("carry")
            if bool(m.any()):
                self.grasp_off[m] = obj_p[m] - tcp_p[m]
                c = cur_c if self.task == "PickPlace" else _face_closing(goal_yaw, cur_c)
                g = goal_p - self.grasp_off
                g[:, 2] = CARRY_Z
                self._set_segment(m, st, g, c, 0.015)
            m = sel("lower")
            if bool(m.any()):
                off = obj_p - tcp_p
                g = goal_p - off
                if self.task == "PickPlace":
                    g[:, 2] = goal_p[:, 2] + S.BOWL_FLOOR_THICKNESS + st["half"] + 0.012 - off[:, 2]
                else:
                    g[:, 2] = goal_p[:, 2] + 0.015 + st["half"] + 0.004 - off[:, 2]
                self._set_segment(m, st, g, cur_c, 0.008)
            m = sel("release")
            if bool(m.any()):
                self.grip_goal[m] = GRIPPER_RELEASE
            m = sel("retreat")
            if bool(m.any()):
                g = tcp_p - 0.004 * cur_c
                g[:, 2] = RETREAT_Z
                self._set_segment(m, st, g, cur_c, 0.01)
        else:  # Push
            d = _horiz(goal_p - obj_p)
            m = sel("hover")
            if bool(m.any()):
                self.push_dir[m] = d[m]
                c = _horiz(torch.cross(up.expand_as(d), d, dim=-1))
                c = torch.where((c * cur_c).sum(-1, keepdim=True) < 0, -c, c)  # least wrist roll
                g = self._pusher_tcp(obj_p, st["half"], d, c, back=0.02)
                g[:, 2] = obj_p[:, 2] + HOVER
                self._set_segment(
                    m, st, g, c, pr.hover_speed, slow_radius=slow[0], slow_speed=slow[1]
                )
                self.grip_goal[m] = GRIPPER_CLOSED
            m = sel("descend")
            if bool(m.any()):
                g = self.p1.clone()
                g[:, 2] = 0.016
                # tilt only if top-down is out of reach
                self._set_segment(m, st, g, self.c1, pr.descend_speed)
            m = sel("retreat")
            if bool(m.any()):
                g = tcp_p - 0.02 * self.push_dir
                g[:, 2] = 0.08
                self._set_segment(m, st, g, cur_c, 0.01)

    def _pusher_tcp(
        self, obj_p: torch.Tensor, half: torch.Tensor, d: torch.Tensor, c: torch.Tensor, back: float
    ) -> torch.Tensor:
        """TCP position that puts the closed-finger pad just behind the object along d."""
        return obj_p - (half[:, None] + PUSHER_HALF + back) * d + PUSHER_OFFSET * c

    def act(self, obs: dict[str, Any]) -> torch.Tensor:
        st = self._state(obs)
        name = [self.names[int(i)] for i in self.phase.tolist()]

        # closed-loop push: retarget the segment every step from the live object pose
        if self.task == "Push":
            m = torch.tensor([x == "push" for x in name], device=self.device)
            if bool(m.any()):
                obj_p, goal_p = st["obj"][:, :3], st["goal"][:, :3]
                d = _horiz(goal_p - obj_p)
                up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
                c = _horiz(torch.cross(up.expand_as(d), d, dim=-1))
                c = torch.where(((c * self.c1).sum(-1, keepdim=True) < 0), -c, c)
                remaining = (goal_p - obj_p)[:, :2].norm(dim=-1)
                step = remaining.clamp(max=0.006)
                g = self._pusher_tcp(obj_p, st["half"], d, c, back=-0.002) + step[:, None] * d
                g[:, 2] = 0.016
                self.p0[m] = st["tcp"][m, :3]
                self.p1[m] = g[m]
                self.c0[m] = st["tcp_R"][m, :, 1]
                self.c1[m] = c[m]
                self.tilt0[m] = self.tilt1[m]  # keep the tilt chosen for the descent
                self.s[m] = 0.0
                self.speed[m] = 0.006
                self.slow_radius[m] = 0.0

        # advance along the segment. Segments with a slow zone (v2 approach) track the actual
        # TCP: the command leads the TCP's projection on the segment by at most one step and never
        # moves back, so when the joint cap makes the arm lag, entering the last `slow_radius`
        # metres (measured at the TCP) really slows the arm instead of letting it catch up.
        seg = self.p1 - self.p0
        seg_len = seg.norm(dim=-1).clamp(min=1e-6)
        s_tcp = (((st["tcp"][:, :3] - self.p0) * seg).sum(-1) / seg_len**2).clamp(0.0, 1.0)
        slow = (1.0 - s_tcp) * seg_len < self.slow_radius
        speed = torch.where(slow, torch.minimum(self.speed, self.slow_speed), self.speed)
        s_next = self.s + speed / seg_len
        led = torch.minimum(s_next, torch.maximum(self.s, s_tcp + speed / seg_len))
        s_next = torch.where(self.slow_radius > 0, led, s_next)
        self.s = torch.clamp(s_next, max=1.0)
        s = self.s[:, None]
        pos = self.p0 + s * (self.p1 - self.p0)
        tilt = self.tilt0 + self.s * (self.tilt1 - self.tilt0)
        a = _approach(pos, tilt)
        c = _unit(self.c0 + s * (self.c1 - self.c0))
        c = _unit(c - (c * a).sum(-1, keepdim=True) * a)
        q_ik, _ = self.kin.ik(pos, a, c, self.q_ik, iters=12)
        self.q_ik = q_ik
        q_prev = self.q_cmd.clone()
        self.q_cmd[:, :5] = q_prev[:, :5] + (q_ik - q_prev[:, :5]).clamp(
            -MAX_JOINT_STEP, MAX_JOINT_STEP
        )
        self.q_cmd[:, 5] = q_prev[:, 5] + (self.grip_goal - q_prev[:, 5]).clamp(
            -MAX_JOINT_STEP, MAX_JOINT_STEP
        )

        self._advance_phases(st, name)
        return self.q_cmd.clone().to(self.out_device)

    def _advance_phases(self, st: dict[str, torch.Tensor], name: list[str]) -> None:
        self.t_phase += 1
        qpos = st["qpos"]
        tracked = (self.q_cmd[:, :5] - self.q_ik).abs().max(-1).values < 0.01
        moved = (qpos[:, :5] - self.prev_qpos[:, :5]).abs().max(-1).values
        self.prev_qpos = qpos.clone()
        # settled: on target, or stopped moving (blocked by contact / unreachable) for a while
        settled = ((qpos[:, :5] - self.q_cmd[:, :5]).abs().max(-1).values < 0.02) | (
            (moved < 0.004) & (self.t_phase >= 8)
        )
        grip_done = (self.q_cmd[:, 5] - self.grip_goal).abs() < 1e-4
        seg_done = (self.s >= 1.0) & tracked & settled
        nxt = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        restart = torch.zeros_like(nxt)
        obj_p, goal_p, half = st["obj"][:, :3], st["goal"][:, :3], st["half"]
        for i, nm in enumerate(name):
            t = int(self.t_phase[i])
            if nm in ("hover", "descend", "lift", "carry", "lower", "retreat"):
                done = bool(seg_done[i]) or t >= PHASE_TIMEOUT
                if nm == "lift" and done and float(obj_p[i, 2]) < float(half[i]) + 0.02:
                    restart[i] = True  # dropped / missed: re-grasp
                    continue
                nxt[i] = done
            elif nm in ("close", "release"):
                nxt[i] = (bool(grip_done[i]) and t >= 4) or t >= PHASE_TIMEOUT
                if nm == "close" and nxt[i] and float(qpos[i, 5]) < 0.05:
                    nxt[i] = False
                    restart[i] = True  # closed on nothing
            elif nm == "push":
                remaining = float((goal_p[i] - obj_p[i])[:2].norm())
                d = _horiz(goal_p[i : i + 1] - obj_p[i : i + 1])[0]
                pad = st["tcp"][i, :3] - PUSHER_OFFSET * self.c1[i]  # finger-pad centre
                rel = (pad - obj_p[i])[:2]
                lateral = abs(float(rel[0] * d[1] - rel[1] * d[0]))
                behind = float(-(rel * d[:2]).sum())
                if remaining < 0.006:
                    nxt[i] = True
                elif t > 6 and (behind < 0.0 or behind > 0.06 or lateral > 0.02):
                    restart[i] = True  # lost the object: back off and re-approach
                elif t >= 3 * PHASE_TIMEOUT:
                    nxt[i] = True
            elif nm == "done":
                if t >= 15 and not self._looks_solved(i, st) and int(self.retries[i]) < MAX_RETRIES:
                    restart[i] = True
        restart &= self.retries < MAX_RETRIES
        self.retries += restart.long()
        self.phase = torch.where(restart, torch.zeros_like(self.phase), self.phase + nxt.long())
        self.phase.clamp_(max=len(self.names) - 1)
        changed = nxt | restart
        if bool(changed.any()):
            self._enter(changed, st)

    def finished(self) -> torch.Tensor:
        """(N,) bool: the script has run to its end (released / retreated, holding still)."""
        return self.phase == len(self.names) - 1

    def contact_phase(self) -> int:
        """Index of the first phase that touches the object (close / push)."""
        return self.names.index("push" if self.task == "Push" else "close")

    def _looks_solved(self, i: int, st: dict[str, torch.Tensor]) -> bool:
        o, g, h = st["obj"][i, :3], st["goal"][i, :3], float(st["half"][i])
        if self.task == "PickPlace":
            return float((o - g)[:2].norm()) < S.BOWL_FLOOR_RADIUS and float(o[2] - g[2]) < 0.06
        if self.task == "Stack":
            return (
                float((o - g)[:2].norm()) < S.STACK_XY_TOL
                and abs(float(o[2] - g[2]) - 2 * h) < 0.006
            )
        return float((o - g)[:2].norm()) < S.TARGET_RADIUS


def make_oracle(task: str, env: Any, profile: str | OracleProfile = DEFAULT_PROFILE) -> Oracle:
    """Oracle for `task` sized to `env` (ManiSkill env or the CPU serial batch from make_env)."""
    base = env.unwrapped
    return Oracle(task, int(base.num_envs), getattr(base, "device", "cpu"), profile)
