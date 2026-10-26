"""SO-101 follower arm as a ManiSkill 3 agent (uid ``so101``).

Sources / licenses:
- Robot model: TheRobotStudio/SO-ARM100 @ a758567c3978dfeefe282ede0500085a48fe8f78,
  ``Simulation/SO101/so101_new_calib.urdf`` + meshes, Apache-2.0. Vendored with edits in
  ``src/env/assets/so101/`` (see its README).
- Agent structure (controller set, jaw-friction urdf_config, ``is_grasping`` contact test) adapted
  from ManiSkill 3.0.1 ``mani_skill/agents/robots/so100/so_100.py`` (haosulab/ManiSkill, tag
  v3.0.1, Apache-2.0).

Joint order (= LeRobot ``so101_follower`` motor names, = action/state order everywhere):
``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper``, all in radians in
the "new calibration" convention (0 = middle of range). Gripper: ~-0.17 closed, larger = more open.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, deepcopy_dict
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from src.env import JOINT_NAMES

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "so101"
URDF_PATH = ASSET_DIR / "so101.urdf"

ARM_JOINTS = JOINT_NAMES[:5]

# Gripper joint values (rad). Closed command squeezes past contact; a 3 cm cube stops the jaw
# at ~0.15-0.2 rad.
GRIPPER_CLOSED = -0.17
GRIPPER_OPEN = 0.6

# Episode start ("rest") pose: gripper raised ~18 cm above the workspace centre, tilted down.
REST_QPOS = np.array([0.0, -1.30, 0.60, 1.30, 0.0, 0.0])

# Joint PD drive (PhysX implicit drive, units N·m/rad, N·m·s/rad, N·m).
ARM_STIFFNESS, ARM_DAMPING, ARM_FORCE_LIMIT = 1500.0, 20.0, 10.0
GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_FORCE_LIMIT = 300.0, 15.0, 0.5
# Max per-step |delta| for pd_joint_delta_pos (rad). Matches the real-arm safety cap of 4°/step.
DELTA_LIMIT = 0.07


@register_agent()
class SO101(BaseAgent):  # type: ignore[misc]
    uid = "so101"
    urdf_path = str(URDF_PATH)
    urdf_config: ClassVar[dict[str, Any]] = {
        "_materials": {
            "gripper": {"static_friction": 1.5, "dynamic_friction": 1.5, "restitution": 0.0}
        },
        "link": {
            "gripper_link": {"material": "gripper", "patch_radius": 0.1, "min_patch_radius": 0.1},
            "moving_jaw_so101_v1_link": {
                "material": "gripper",
                "patch_radius": 0.1,
                "min_patch_radius": 0.1,
            },
        },
    }
    keyframes: ClassVar[dict[str, Keyframe]] = {
        "rest": Keyframe(qpos=REST_QPOS.copy(), pose=sapien.Pose()),
        "zero": Keyframe(qpos=np.zeros(6), pose=sapien.Pose()),
    }

    @property
    def _controller_configs(self) -> dict[str, Any]:
        stiffness = [ARM_STIFFNESS] * 5 + [GRIPPER_STIFFNESS]
        damping = [ARM_DAMPING] * 5 + [GRIPPER_DAMPING]
        force_limit = [ARM_FORCE_LIMIT] * 5 + [GRIPPER_FORCE_LIMIT]
        pd_joint_pos = PDJointPosControllerConfig(
            list(JOINT_NAMES),
            lower=None,
            upper=None,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            list(JOINT_NAMES),
            lower=-DELTA_LIMIT,
            upper=DELTA_LIMIT,
            stiffness=stiffness,
            damping=damping,
            force_limit=force_limit,
            use_delta=True,
            use_target=False,  # delta from the measured qpos (ManiSkill naming convention)
            normalize_action=False,
        )
        configs = {"pd_joint_pos": pd_joint_pos, "pd_joint_delta_pos": pd_joint_delta_pos}
        return deepcopy_dict(copy.deepcopy(configs))  # type: ignore[no-any-return]

    def _after_loading_articulation(self) -> None:
        super()._after_loading_articulation()
        self.fixed_jaw_link = self.robot.links_map["gripper_link"]
        self.moving_jaw_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        self.tcp_link = self.robot.links_map["tcp_link"]
        names = [j.name for j in self.robot.active_joints]
        assert tuple(names) == JOINT_NAMES, f"unexpected joint order {names}"

    @property
    def tcp_pose(self) -> Pose:
        """Grasp centre (between the jaw pads); z = approach axis, y = jaw closing axis."""
        return self.tcp_link.pose

    @property
    def tcp_pos(self) -> torch.Tensor:
        pos: torch.Tensor = self.tcp_link.pose.p
        return pos

    def is_touching(self, obj: Actor, min_force: float = 1e-2) -> torch.Tensor:
        """(N,) bool: either jaw link is in contact with `obj`."""
        f1 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.fixed_jaw_link, obj), dim=1
        )
        f2 = torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(self.moving_jaw_link, obj), dim=1
        )
        touching: torch.Tensor = (f1 >= min_force) | (f2 >= min_force)
        return touching

    def is_grasping(
        self, obj: Actor, min_force: float = 0.5, max_angle: float = 85.0
    ) -> torch.Tensor:
        """(N,) bool: both jaws push on `obj` from opposite sides (ManiSkill so100 test)."""
        l_forces = self.scene.get_pairwise_contact_forces(self.fixed_jaw_link, obj)
        r_forces = self.scene.get_pairwise_contact_forces(self.moving_jaw_link, obj)
        lforce = torch.linalg.norm(l_forces, dim=1)
        rforce = torch.linalg.norm(r_forces, dim=1)
        # Closing direction = tcp y axis (fixed jaw pushes the object along +y, moving jaw along -y).
        closing = self.tcp_link.pose.to_transformation_matrix()[..., :3, 1]
        # pairwise forces act on the jaw links: the fixed jaw is pushed along -closing, the moving
        # jaw along +closing when an object sits between them.
        langle = common.compute_angle_between(-closing, l_forces)
        rangle = common.compute_angle_between(closing, r_forces)
        lflag = (lforce >= min_force) & (torch.rad2deg(langle) <= max_angle)
        rflag = (rforce >= min_force) & (torch.rad2deg(rangle) <= max_angle)
        grasping: torch.Tensor = lflag & rflag
        return grasping

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        qvel = self.robot.get_qvel()[:, :-1]
        static: torch.Tensor = torch.max(torch.abs(qvel), 1)[0] <= threshold
        return static
