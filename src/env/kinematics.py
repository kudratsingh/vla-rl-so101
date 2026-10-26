"""Batched forward/inverse kinematics for the SO-101 arm (torch; no pinocchio/mplib, Mac-safe).

FK is parsed from the vendored URDF (`base_link` -> `tcp_link`). IK is damped least squares
(Levenberg-Marquardt) on position + approach-axis + closing-axis residuals with a finite-difference
Jacobian. The arm has 5 DOF, so a TCP pose is specified as (position, approach direction,
closing direction) and the closing term can be down-weighted or switched off.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch

from src.env.robot import URDF_PATH

ARM_CHAIN = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


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


def _origin_tf(joint: ET.Element) -> torch.Tensor:
    o = joint.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    t = torch.eye(4, dtype=torch.float64)
    t[:3, :3] = _rpy_matrix(*rpy)
    t[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return t


@dataclass
class ArmKinematics:
    """FK/IK for base_link -> tcp_link. All tensors float32 on `device` unless noted."""

    origins: torch.Tensor  # (5, 4, 4) fixed transform before each revolute joint
    tcp_offset: torch.Tensor  # (4, 4) wrist_roll child (gripper_link) -> tcp_link
    lower: torch.Tensor  # (5,)
    upper: torch.Tensor  # (5,)

    @classmethod
    def from_urdf(cls, path: Path = URDF_PATH, device: str | torch.device = "cpu") -> ArmKinematics:
        root = ET.parse(path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        origins, lower, upper = [], [], []
        for name in ARM_CHAIN:
            j = joints[name]
            axis = j.find("axis")
            assert axis is not None and axis.get("xyz", "").split() == ["0", "0", "1"], name
            origins.append(_origin_tf(j))
            lim = j.find("limit")
            assert lim is not None
            lower.append(float(lim.get("lower", "0")))
            upper.append(float(lim.get("upper", "0")))
        tcp = _origin_tf(joints["tcp_joint"])
        dev = torch.device(device)
        return cls(
            origins=torch.stack(origins).float().to(dev),
            tcp_offset=tcp.float().to(dev),
            lower=torch.tensor(lower, device=dev),
            upper=torch.tensor(upper, device=dev),
        )

    def fk(self, q: torch.Tensor) -> torch.Tensor:
        """q (B, >=5) arm joint angles -> (B, 4, 4) TCP pose in the robot base frame."""
        b = q.shape[0]
        t = torch.eye(4, device=q.device, dtype=q.dtype).expand(b, 4, 4)
        c, s = torch.cos(q[:, :5]), torch.sin(q[:, :5])
        for i in range(5):
            rz = torch.zeros(b, 4, 4, device=q.device, dtype=q.dtype)
            rz[:, 0, 0], rz[:, 0, 1], rz[:, 1, 0], rz[:, 1, 1] = c[:, i], -s[:, i], s[:, i], c[:, i]
            rz[:, 2, 2] = 1.0
            rz[:, 3, 3] = 1.0
            t = t @ self.origins[i] @ rz
        return t @ self.tcp_offset

    def _residual(
        self,
        q: torch.Tensor,
        pos: torch.Tensor,
        approach: torch.Tensor,
        closing: torch.Tensor,
        w_approach: float,
        w_closing: torch.Tensor,
    ) -> torch.Tensor:
        t = self.fk(q)
        r_p = t[:, :3, 3] - pos
        r_a = (t[:, :3, 2] - approach) * w_approach
        # closing axis is sign-symmetric for a parallel gripper: use the closer of +/- target
        y = t[:, :3, 1]
        sign = torch.sign((y * closing).sum(-1, keepdim=True)).clamp(min=0) * 2 - 1
        r_c = (y - sign * closing) * w_closing[:, None]
        return torch.cat([r_p, r_a, r_c], dim=-1)

    def ik(
        self,
        pos: torch.Tensor,
        approach: torch.Tensor,
        closing: torch.Tensor | None,
        q0: torch.Tensor,
        *,
        w_approach: float = 0.05,
        w_closing: float | torch.Tensor = 0.02,
        iters: int = 40,
        damping: float = 1e-4,
        tol: float = 5e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve for arm joints (B, 5). Returns (q, position error in m (B,)).

        pos (B, 3), approach (B, 3) unit vector of the TCP z axis, closing (B, 3) unit vector of the
        TCP y axis (sign ignored) or None to leave roll free. q0 (B, >=5) warm start.
        """
        b = pos.shape[0]
        q = q0[:, :5].clone()
        if closing is None:
            closing = torch.zeros_like(pos)
            w_c = torch.zeros(b, device=pos.device)
        elif isinstance(w_closing, torch.Tensor):
            w_c = w_closing.to(pos.device).expand(b)
        else:
            w_c = torch.full((b,), float(w_closing), device=pos.device)
        h = 1e-4
        eye = torch.eye(5, device=pos.device, dtype=pos.dtype)
        for _ in range(iters):
            qs = torch.cat([q[None], q[None] + h * eye[:, None, :]], dim=0)  # (6, B, 5)
            r_all = self._residual(
                qs.reshape(-1, 5),
                pos.repeat(6, 1),
                approach.repeat(6, 1),
                closing.repeat(6, 1),
                w_approach,
                w_c.repeat(6),
            ).reshape(6, b, -1)
            r = r_all[0]
            jac = ((r_all[1:] - r[None]) / h).permute(1, 2, 0)  # (B, 9, 5)
            jt = jac.transpose(1, 2)
            lhs = jt @ jac + damping * eye
            dq = torch.linalg.solve(lhs, -(jt @ r[..., None]))[..., 0]
            dq = dq.clamp(-0.3, 0.3)
            q = torch.max(torch.min(q + dq, self.upper), self.lower)
            if bool((r[:, :3].norm(dim=-1) < tol).all()) and float(dq.abs().max()) < 1e-4:
                break
        err = (self.fk(q)[:, :3, 3] - pos).norm(dim=-1)
        return q, err

    def ik_multi(
        self,
        pos: torch.Tensor,
        approach: torch.Tensor,
        closing: torch.Tensor | None,
        q0: torch.Tensor,
        seeds: torch.Tensor,
        **kwargs: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IK from q0 and from each of `seeds` (K, 5); keep the lowest-error solution per row
        (ties broken by joint distance to q0)."""
        b, k = pos.shape[0], seeds.shape[0] + 1
        starts = torch.cat([q0[None, :, :5], seeds[:, None, :5].expand(-1, b, -1)], dim=0)

        def rep(t: torch.Tensor) -> torch.Tensor:
            return t[None].expand(k, *t.shape).reshape(k * b, -1)

        q, err = self.ik(
            rep(pos),
            rep(approach),
            None if closing is None else rep(closing),
            starts.reshape(k * b, 5),
            **kwargs,  # type: ignore[arg-type]  # iters/tol/... forwarded verbatim
        )
        q, err = q.reshape(k, b, 5), err.reshape(k, b)
        dist = (q - q0[None, :, :5]).abs().sum(-1)
        score = err.clamp(min=1e-3) * 1e3 + dist * 0.01
        best = score.argmin(0)
        idx = torch.arange(b, device=pos.device)
        return q[best, idx], err[best, idx]
