"""Action-normalisation bounds from joint limits ∩ reachable workspace (not from demos)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.config import REPO_ROOT, load_config
from src.policy.norm_bounds import (
    JOINTS,
    WorkspaceSpec,
    bounds_from_cfg,
    demo_coverage,
    forward_kinematics,
    urdf_joint_limits,
    workspace_bounds,
)

# A 5R arm with the SO-101's joint names: pan about z, then three pitch joints about y, roll about
# the tool axis, tool pointing along the last link; a separate gripper joint.
URDF = """<robot name="toy">
  <link name="base_link"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
  <link name="gripper_link"/><link name="tcp_link"/><link name="jaw"/>
  <joint name="shoulder_pan" type="revolute"><parent link="base_link"/><child link="l1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-1.9" upper="1.9"/></joint>
  <joint name="shoulder_lift" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.7" upper="1.7"/></joint>
  <joint name="elbow_flex" type="revolute"><parent link="l2"/><child link="l3"/>
    <origin xyz="0.12 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.7" upper="1.7"/></joint>
  <joint name="wrist_flex" type="revolute"><parent link="l3"/><child link="l4"/>
    <origin xyz="0.12 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.7" upper="1.7"/></joint>
  <joint name="wrist_roll" type="revolute"><parent link="l4"/><child link="gripper_link"/>
    <origin xyz="0.03 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/><limit lower="-2.8" upper="2.8"/></joint>
  <joint name="tcp_joint" type="fixed"><parent link="gripper_link"/><child link="tcp_link"/>
    <origin xyz="0.07 0 0" rpy="0 1.5707963 0"/></joint>
  <joint name="gripper" type="revolute"><parent link="gripper_link"/><child link="jaw"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-0.2" upper="1.7"/></joint>
</robot>"""

SPEC = WorkspaceSpec(
    x=(0.12, 0.28),
    y=(-0.1, 0.1),
    z=(0.0, 0.2),
    max_tilt_deg=60.0,
    wrist_roll=(-math.pi / 2, math.pi / 2),
    gripper=(-0.17, 0.6),
    include_qpos=((0.0, -0.5, 0.8, 0.8, 0.0, 0.0),),
    samples=100_000,
)


@pytest.fixture()
def urdf(tmp_path: Path) -> Path:
    p = tmp_path / "toy.urdf"
    p.write_text(URDF)
    return p


def test_fk_and_limits(urdf: Path) -> None:
    lim = urdf_joint_limits(urdf)
    assert list(lim) == list(JOINTS) and lim["wrist_roll"] == (-2.8, 2.8)
    tf = forward_kinematics(urdf, torch.zeros(1, 5), base="base_link", tip="tcp_link")
    # straight arm along x at height 0.1; tcp z axis = tool x after the fixed pitch of +90°
    assert torch.allclose(
        tf[0, :3, 3], torch.tensor([0.34, 0.0, 0.1], dtype=torch.float64), atol=1e-6
    )
    assert torch.allclose(
        tf[0, :3, 2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), atol=1e-6
    )


def test_workspace_bounds_tighter_than_limits_and_resolve_1deg(urdf: Path) -> None:
    rep = workspace_bounds(urdf, SPEC)
    b, lim = rep.bounds, rep.joint_limits
    for i, name in enumerate(JOINTS):
        assert lim[name][0] <= b.low[i] < b.high[i] <= lim[name][1]
    # pan only needs about +-atan(0.1 / 0.12) plus margin, far inside the +-1.9 limit
    assert b.high[0] < 1.2 and b.low[0] > -1.2
    assert b.low[4] == pytest.approx(-math.pi / 2 - 0.05) and b.high[5] == pytest.approx(0.65)
    assert max(rep.deg_per_bin) <= 1.0 and 0 < rep.kept_fraction < 1
    # the episode start pose is always representable
    q0 = torch.tensor(SPEC.include_qpos[0])
    assert bool(((q0 >= torch.tensor(b.low)) & (q0 <= torch.tensor(b.high))).all())
    cov = demo_coverage(b, torch.zeros(10, 8, 6))
    assert cov["frac_outside"] == 0.0


def test_resolution_check_and_demo_union(urdf: Path) -> None:
    wide = WorkspaceSpec(**{**SPEC.__dict__, "wrist_roll": (-2.8, 2.8)})
    with pytest.raises(ValueError, match="°/bin"):
        workspace_bounds(urdf, wide)
    demo = torch.zeros(4, 6)
    demo[0, 0] = 1.5  # a demo action outside the workspace range widens the bound
    rep = workspace_bounds(urdf, SPEC, extra_actions=demo)
    assert rep.bounds.high[0] >= 1.5


def test_sft_config_bounds_on_real_urdf() -> None:
    """configs/sft.yaml on the vendored SO-101 URDF (lands with WO-01): <= 1°/bin everywhere."""
    if not (REPO_ROOT / "src/env/assets/so101/so101.urdf").is_file():
        pytest.skip("SO-101 URDF not on this branch yet (WO-01)")
    action, proprio = bounds_from_cfg(load_config("sim", "configs/sft.yaml"))
    assert action == proprio and max(action.deg_per_bin()) <= 1.0
