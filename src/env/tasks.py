"""SO-101 tabletop scene + the three tasks: SO101PickPlace-v1, SO101Push-v1, SO101Stack-v1.

World frame = robot base frame (base_link at the origin, arm facing +x, table top at z = 0).
Spawn regions and object sets come from src/eval/protocol.py; success checks from success.py.

Domain-randomization hooks (lighting, robot colour, camera jitter) follow the pattern of
ManiSkill 3.0.1 ``mani_skill/envs/tasks/digital_twins/so100_arm/grasp_cube.py``
(haosulab/ManiSkill, tag v3.0.1, Apache-2.0).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.camera import noised_look_at
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from src.env import CONTROL_HZ
from src.env import success as S
from src.env.cameras import CAMERA_NAME, resolve_front_camera
from src.env.randomization import RandomizationConfig
from src.env.robot import REST_QPOS, SO101
from src.eval import protocol as P

SIM_FREQ = 300  # 20 physics substeps per control step
DEFAULT_MAX_EPISODE_STEPS = 150

TABLE_RGBA = (0.62, 0.50, 0.38, 1.0)
BOWL_RGBA = (0.93, 0.93, 0.90, 1.0)
TARGET_RGBA = (0.08, 0.08, 0.08, 1.0)
OBJ_DENSITY = 600.0  # wood, ~16 g per 3 cm cube
OBJ_FRICTION = 0.6
TABLE_FRICTION = 0.4
PARK_Z = -0.30  # hidden shelf under the table for inactive candidate objects

# Placement constraints (m): min centre distances and keep-out radius around the robot base.
BASE_KEEPOUT = 0.085
MIN_SEP = {"PickPlace": 0.11, "Push": 0.0, "Stack": 0.08}
PUSH_DIST = (0.06, 0.10)
# The pusher starts ~3.5 cm behind the object (away from the target); keep that point within the
# arm's top-down reach so every sampled push is physically feasible.
PUSH_REACH = 0.29


def _q_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N, 4) wxyz quaternions."""
    return torch.stack(
        [torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1
    )


def _sample_points(
    rng: np.random.RandomState,
    n: int,
    region: P.Region,
    exclude: P.Region | None,
    min_sep: float,
    keepout: list[float],
    tries: int = 2000,
) -> np.ndarray:
    """n points uniformly in `region` (outside `exclude`), pairwise >= min_sep apart, and each
    point i at least keepout[i] from the robot base. Rejection sampling (per point, then restart).
    """
    cx, cy = region.center
    hx, hy = region.half_extent
    for _ in range(tries):
        pts: list[np.ndarray] = []
        for i in range(n):
            for _ in range(200):
                p = np.array([cx + rng.uniform(-hx, hx), cy + rng.uniform(-hy, hy)])
                if exclude is not None and exclude.contains(float(p[0]), float(p[1])):
                    continue
                if np.linalg.norm(p) < keepout[i]:
                    continue
                if any(np.linalg.norm(p - q) < min_sep for q in pts):
                    continue
                pts.append(p)
                break
            else:
                break
        if len(pts) == n:
            return np.stack(pts)
    raise RuntimeError("could not sample a valid placement; region too small for constraints")


class SO101TabletopEnv(BaseEnv):  # type: ignore[misc]
    """Shared scene: table, SO-101 at the origin, `front` camera, manipulated object(s)."""

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["so101"]
    SUPPORTED_REWARD_MODES: ClassVar[list[str]] = ["sparse", "none"]
    TASK: ClassVar[str] = ""
    agent: SO101

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "so101",
        cell: str = "seen",
        image_size: int = 128,
        camera: dict[str, Any] | None = None,
        randomization: RandomizationConfig | None = None,
        robot_init_qpos_noise: float = 0.02,
        **kwargs: Any,
    ) -> None:
        if cell not in P.CELLS:
            raise ValueError(f"unknown cell {cell!r}; expected one of {P.CELLS}")
        self.cell = cell
        self.image_size = int(image_size)
        self.camera_spec = resolve_front_camera(camera)
        self.dr = randomization or RandomizationConfig()
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.region, self.exclude = P.spawn_regions(cell)
        self.candidate_specs: tuple[P.ObjectSpec, ...] = P.manipulated_objects(cell)
        self._instructions: list[str] = []
        if self.dr.enabled and self.dr.reconfigure_every > 0:
            kwargs.setdefault("reconfiguration_freq", self.dr.reconfigure_every)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------ config
    @property
    def _default_sim_config(self) -> SimConfig:
        # 25 position iterations: with the default 15 the solver does not converge while the jaws
        # squeeze a cube, which leaves the wrist several degrees off its target.
        return SimConfig(
            sim_freq=SIM_FREQ,
            control_freq=CONTROL_HZ,
            scene_config=SceneConfig(solver_position_iterations=25),
        )

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        fov = math.radians(self.camera_spec.fov_deg)
        if self.dr.enabled and self.dr.camera_fov_jitter_deg > 0:
            noise = self._batched_episode_rng.uniform(-1.0, 1.0) * self.dr.camera_fov_jitter_deg
            fov = fov + np.deg2rad(noise)
        if self.dr.camera_jitter:
            pose = sapien.Pose()
            mount = self.camera_mount
        else:
            pose = sapien_utils.look_at(self.camera_spec.eye, self.camera_spec.target)
            mount = None
        size = self.image_size
        return [CameraConfig(CAMERA_NAME, pose, size, size, fov, 0.01, 10.0, mount=mount)]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        pose = sapien_utils.look_at(self.camera_spec.eye, self.camera_spec.target)
        return CameraConfig(
            "render_camera", pose, 384, 384, math.radians(self.camera_spec.fov_deg), 0.01, 10.0
        )

    # ------------------------------------------------------------------ loading
    def _load_agent(self, options: dict[str, Any]) -> None:
        separate = self.dr.enabled and self.dr.robot_color
        super()._load_agent(options, sapien.Pose(), build_separate=separate)

    def _load_lighting(self, options: dict[str, Any]) -> None:
        if self.dr.enabled and self.dr.lighting:
            rng = self._batched_episode_rng
            ambient = rng.uniform(0.15, 0.45, size=(3,))
            scale = rng.uniform(0.6, 1.4)
            tilt = rng.uniform(-0.4, 0.4, size=(2,))
            for i, sub in enumerate(self.scene.sub_scenes):
                sub.render_system.ambient_light = ambient[i].tolist()
                s = float(scale[i])
                self.scene.add_directional_light(
                    [1.0 + tilt[i, 0], 1.0 + tilt[i, 1], -1.0], [s, s, s], scene_idxs=[i]
                )
                self.scene.add_directional_light([0, 0, -1], [s * 0.8] * 3, scene_idxs=[i])
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
            self.scene.add_directional_light([1, 1, -1], [1, 1, 1])
            self.scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])

    def _physx_material(self, friction: float) -> Any:
        return sapien.pysapien.physx.PhysxMaterial(
            static_friction=friction, dynamic_friction=friction, restitution=0.0
        )

    def _build_object(self, spec: P.ObjectSpec, name: str, park_xy: tuple[float, float]) -> Actor:
        """One actor per spec; per-sub-scene copies when colour/physics randomization is on."""
        h = spec.size / 2
        per_env = self.dr.enabled and (self.dr.object_color_jitter > 0 or self.dr.object_physics)
        n = self.num_envs if per_env else 1
        rng = self._batched_episode_rng
        rgba = np.tile(np.array(spec.rgba), (n, 1))
        friction = np.full(n, OBJ_FRICTION)
        density = np.full(n, OBJ_DENSITY)
        if per_env and self.dr.object_color_jitter > 0:
            j = self.dr.object_color_jitter
            rgba[:, :3] = np.clip(rgba[:, :3] + rng.uniform(-j, j, size=(3,)), 0.0, 1.0)
        if per_env and self.dr.object_physics:
            friction = rng.uniform(0.4, 0.9)
            density = rng.uniform(400.0, 900.0)
        parts = []
        for i in range(n):
            builder = self.scene.create_actor_builder()
            mat = self._physx_material(float(friction[i]))
            vis = sapien.render.RenderMaterial(base_color=rgba[i].tolist())
            if spec.shape == "cube":
                builder.add_box_collision(
                    half_size=[h] * 3, material=mat, density=float(density[i])
                )
                builder.add_box_visual(half_size=[h] * 3, material=vis)
            elif spec.shape == "cylinder":
                axis_z = sapien.Pose(q=euler2quat(0, -np.pi / 2, 0))  # SAPIEN cylinders run along x
                builder.add_cylinder_collision(
                    pose=axis_z, radius=h, half_length=h, material=mat, density=float(density[i])
                )
                builder.add_cylinder_visual(pose=axis_z, radius=h, half_length=h, material=vis)
            else:
                raise ValueError(f"unknown shape {spec.shape!r}")
            builder.initial_pose = sapien.Pose(p=[park_xy[0], park_xy[1], PARK_Z + h])
            if per_env:
                builder.set_scene_idxs([i])
                part = builder.build(name=f"{name}-{i}")
                self.remove_from_state_dict_registry(part)
                parts.append(part)
            else:
                actor: Actor = builder.build(name=name)
                return actor
        merged = Actor.merge(parts, name=name)
        self.add_to_state_dict_registry(merged)
        return merged

    def _load_scene(self, options: dict[str, Any]) -> None:
        # table: large static slab, top at z = 0 (also hides the parking shelf below it)
        b = self.scene.create_actor_builder()
        b.add_box_collision(
            pose=sapien.Pose(p=[0.1, 0, -0.02]),
            half_size=[1.5, 1.5, 0.02],
            material=self._physx_material(TABLE_FRICTION),
        )
        b.add_box_visual(
            pose=sapien.Pose(p=[0.1, 0, -0.02]),
            half_size=[1.5, 1.5, 0.02],
            material=sapien.render.RenderMaterial(base_color=list(TABLE_RGBA)),
        )
        b.initial_pose = sapien.Pose()
        self.table = b.build_static(name="table")
        b = self.scene.create_actor_builder()
        b.add_box_collision(pose=sapien.Pose(p=[0.1, 0, PARK_Z - 0.01]), half_size=[0.4, 0.4, 0.01])
        b.initial_pose = sapien.Pose()
        self.shelf = b.build_static(name="park_shelf")

        if self.dr.camera_jitter:
            mb = self.scene.create_actor_builder()
            mb.initial_pose = sapien.Pose()
            self.camera_mount = mb.build_kinematic("camera_mount")

        self.candidates = [
            self._build_object(spec, f"obj{k}", (0.1, -0.2 + 0.1 * k))
            for k, spec in enumerate(self.candidate_specs)
        ]
        self._load_task_objects()
        if self.dr.enabled and self.dr.robot_color:
            self._randomize_robot_color()

        n = self.num_envs
        self._active = torch.zeros(n, dtype=torch.long, device=self.device)
        self._obj_half = torch.full((n,), self.candidate_specs[0].size / 2, device=self.device)
        self._latency = torch.zeros(n, dtype=torch.long, device=self.device)
        self._act_hist = torch.zeros(n, 3, 6, device=self.device)
        self._instructions = [""] * n

    def _randomize_robot_color(self) -> None:
        rng = self._batched_episode_rng
        colors = rng.uniform(0.1, 1.0, size=(3,))
        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                comp = obj.entity.find_component_by_type(RenderBodyComponent)
                if comp is None:
                    continue
                for shape in comp.render_shapes:
                    for part in shape.parts:
                        if float(np.mean(part.material.base_color[:3])) < 0.2:
                            continue  # keep the black servos black
                        part.material.set_base_color(colors[i].tolist() + [1.0])

    def _load_task_objects(self) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ episodes
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            idx_np = common.to_numpy(env_idx)
            rngs = self._batched_episode_rng[idx_np]

            qpos = torch.tensor(REST_QPOS, dtype=torch.float32).repeat(b, 1)
            noise = torch.tensor(
                rngs.normal(0.0, self.robot_init_qpos_noise, size=(6,)), dtype=torch.float32
            )
            noise[:, 5] = 0.0
            qpos = qpos + noise
            self.agent.robot.set_pose(Pose.create_from_pq(torch.zeros(b, 3)))
            self.agent.robot.set_qpos(qpos)

            # which candidate object is active in each sub-env this episode
            k = len(self.candidates)
            active = (
                torch.tensor(rngs.randint(0, k), dtype=torch.long)
                if k > 1
                else torch.zeros(b, dtype=torch.long)
            )
            self._active[env_idx] = active
            sizes = torch.tensor([s.size / 2 for s in self.candidate_specs])
            self._obj_half[env_idx] = sizes[active]

            self._init_task(env_idx, rngs)  # must call _place_active exactly once

            for j, e in enumerate(idx_np.tolist()):
                spec = self.candidate_specs[int(active[j])]
                self._instructions[e] = P.instruction(self.TASK, spec, P.BLUE_CUBE)

            # action latency: hold the start pose for the delayed steps
            if self.dr.enabled and self.dr.action_latency_max_steps > 0:
                lat = rngs.randint(0, self.dr.action_latency_max_steps + 1)
                self._latency[env_idx] = torch.tensor(lat, dtype=torch.long)
            hold = qpos if self.control_mode == "pd_joint_pos" else torch.zeros_like(qpos)
            self._act_hist[env_idx] = hold[:, None, :].repeat(1, 3, 1)

            if self.dr.camera_jitter:
                eye = torch.tensor(self.camera_spec.eye).repeat(b, 1)
                eye = eye + (torch.rand(b, 3) * 2 - 1) * self.dr.camera_pos_jitter_m
                pose = noised_look_at(
                    eye,
                    target=torch.tensor(self.camera_spec.target),
                    look_at_noise=self.dr.camera_target_jitter_m,
                    view_axis_rot_noise=self.dr.camera_roll_jitter_rad,
                )
                self.camera_mount.set_pose(pose)

    def _place_active(self, env_idx: torch.Tensor, xy: torch.Tensor, yaw: torch.Tensor) -> None:
        """Put each sub-env's active candidate at (xy, yaw) on the table and park the others on
        the hidden shelf. Called once per episode init with all of env_idx (reset-mask writes)."""
        b = len(env_idx)
        active = self._active[env_idx]
        for c_i, cand in enumerate(self.candidates):
            h = self.candidate_specs[c_i].size / 2
            p = torch.tensor([0.1, -0.2 + 0.1 * c_i, PARK_Z + h]).repeat(b, 1)
            q = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(b, 1)
            mask = active == c_i
            p[mask] = torch.cat([xy[mask], torch.full((int(mask.sum()), 1), h)], dim=1)
            q[mask] = _q_yaw(yaw[mask])
            cand.set_pose(Pose.create_from_pq(p, q))

    def _init_task(self, env_idx: torch.Tensor, rngs: Any) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ stepping
    def _step_action(self, action: Any) -> Any:
        if self.dr.enabled and self.dr.action_latency_max_steps > 0 and action is not None:
            a = common.to_tensor(action, device=self.device).float()
            if a.ndim == 1:
                a = a[None].expand(self.num_envs, -1)
            self._act_hist = torch.cat([a[:, None], self._act_hist[:, :-1]], dim=1)
            action = self._act_hist[torch.arange(self.num_envs, device=self.device), self._latency]
        return super()._step_action(action)

    # ------------------------------------------------------------------ state
    def _select(self, per_candidate: list[torch.Tensor]) -> torch.Tensor:
        if len(per_candidate) == 1:
            return per_candidate[0]
        stacked = torch.stack(per_candidate)
        return stacked[self._active, torch.arange(self.num_envs, device=self.device)]

    def obj_pose(self) -> torch.Tensor:
        return self._select([c.pose.raw_pose for c in self.candidates])

    def obj_vel(self) -> tuple[torch.Tensor, torch.Tensor]:
        lin = self._select([c.linear_velocity for c in self.candidates])
        ang = self._select([c.angular_velocity for c in self.candidates])
        return lin, ang

    def obj_touching(self) -> torch.Tensor:
        return self._select([self.agent.is_touching(c) for c in self.candidates])

    def obj_grasped(self) -> torch.Tensor:
        return self._select([self.agent.is_grasping(c) for c in self.candidates])

    def goal_pose(self) -> torch.Tensor:
        raise NotImplementedError

    def get_instructions(self) -> list[str]:
        return list(self._instructions)

    def _get_obs_agent(self) -> dict[str, torch.Tensor]:
        # qvel is not observable on the real arm at useful quality; keep qpos (+ controller target)
        obs = {"qpos": self.agent.robot.get_qpos()}
        return obs

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Privileged state for the scripted oracle (NOT for the policy; see policy_obs)."""
        return {
            "tcp_pose": self.agent.tcp_pose.raw_pose,
            "obj_pose": self.obj_pose(),
            "goal_pose": self.goal_pose(),
            "obj_half_height": self._obj_half.clone(),
            "is_grasped": info["is_grasped"],
        }

    def compute_sparse_reward(
        self, obs: Any, action: torch.Tensor, info: dict[str, Any]
    ) -> torch.Tensor:
        reward: torch.Tensor = info["success"].float()
        return reward


@register_env("SO101PickPlace-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class SO101PickPlaceEnv(SO101TabletopEnv):
    TASK = "PickPlace"

    def _load_task_objects(self) -> None:
        b = self.scene.create_actor_builder()
        mat = self._physx_material(0.5)
        vis = sapien.render.RenderMaterial(base_color=list(BOWL_RGBA))
        r0, th = S.BOWL_FLOOR_RADIUS, S.BOWL_FLOOR_THICKNESS
        floor = sapien.Pose(p=[0, 0, th / 2], q=euler2quat(0, -np.pi / 2, 0))
        b.add_cylinder_collision(
            pose=floor, radius=r0 + 0.004, half_length=th / 2, material=mat, density=2500
        )
        b.add_cylinder_visual(pose=floor, radius=r0 + 0.004, half_length=th / 2, material=vis)
        tilt = math.radians(S.BOWL_WALL_TILT_DEG)
        hgt, t = S.BOWL_WALL_HEIGHT, S.BOWL_WALL_THICKNESS
        slant = hgt / math.cos(tilt)
        r_mid = r0 + t / 2 + 0.5 * hgt * math.tan(tilt)
        n_seg = 20
        seg_half = math.pi * (2 * r_mid) / n_seg / 2 + 0.002
        for i in range(n_seg):
            yaw = 2 * math.pi * i / n_seg
            # box local frame: x = radial thickness, y = tangent, z = up; tilt outward about y
            q = euler2quat(yaw, tilt, 0, "rzyx")  # Rz(yaw) @ Ry(tilt): top leans outward
            pose = sapien.Pose(p=[r_mid * math.cos(yaw), r_mid * math.sin(yaw), th + hgt / 2], q=q)
            half = [t / 2, seg_half, slant / 2]
            b.add_box_collision(pose=pose, half_size=half, material=mat, density=2500)
            b.add_box_visual(pose=pose, half_size=half, material=vis)
        b.initial_pose = sapien.Pose(p=[0.25, 0.2, 0])
        self.bowl = b.build(name="bowl")

    def _init_task(self, env_idx: torch.Tensor, rngs: Any) -> None:
        b = len(env_idx)
        pts = np.stack(
            [
                _sample_points(
                    rngs[j],
                    2,
                    self.region,
                    self.exclude,
                    MIN_SEP["PickPlace"],
                    [BASE_KEEPOUT + 0.025, BASE_KEEPOUT + 0.065],
                )
                for j in range(b)
            ]
        )
        pts_t = torch.tensor(pts, dtype=torch.float32)
        yaw = torch.tensor(rngs.uniform(-np.pi / 4, np.pi / 4), dtype=torch.float32)
        self._place_active(env_idx, pts_t[:, 0], yaw)
        bowl_yaw = torch.tensor(rngs.uniform(-np.pi, np.pi), dtype=torch.float32)
        p = torch.cat([pts_t[:, 1], torch.zeros(b, 1)], dim=1)
        self.bowl.set_pose(Pose.create_from_pq(p, _q_yaw(bowl_yaw)))

    def goal_pose(self) -> torch.Tensor:
        return self.bowl.pose.raw_pose  # type: ignore[no-any-return]

    def evaluate(self) -> dict[str, torch.Tensor]:
        pose = self.obj_pose()
        lin, ang = self.obj_vel()
        bowl_up_z = self.bowl.pose.to_transformation_matrix()[:, 2, 2]
        out = S.pick_place(
            pose[:, :3], lin, ang, self.bowl.pose.p, bowl_up_z, self.obj_touching(), self._obj_half
        )
        out["is_grasped"] = self.obj_grasped()
        return out


@register_env("SO101Push-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class SO101PushEnv(SO101TabletopEnv):
    TASK = "Push"

    def _load_task_objects(self) -> None:
        b = self.scene.create_actor_builder()
        vis = sapien.render.RenderMaterial(base_color=list(TARGET_RGBA))
        b.add_cylinder_visual(
            pose=sapien.Pose(p=[0, 0, 0.0005], q=euler2quat(0, -np.pi / 2, 0)),
            radius=S.TARGET_RADIUS,
            half_length=0.0005,
            material=vis,
        )
        b.initial_pose = sapien.Pose(p=[0.25, 0.0, 0])
        self.target = b.build_kinematic(name="target")
        self._ever_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _init_task(self, env_idx: torch.Tensor, rngs: Any) -> None:
        b = len(env_idx)
        obj_xy, goal_xy = [], []
        for j in range(b):
            rng = rngs[j]
            for _ in range(1000):
                o = _sample_points(rng, 1, self.region, self.exclude, 0.0, [BASE_KEEPOUT + 0.025])[
                    0
                ]
                d, a = rng.uniform(*PUSH_DIST), rng.uniform(-np.pi, np.pi)
                u = np.array([math.cos(a), math.sin(a)])
                g = o + d * u
                if np.linalg.norm(o - 0.035 * u) > PUSH_REACH:
                    continue
                if not self.region.contains(float(g[0]), float(g[1])):
                    continue
                if self.exclude is not None and self.exclude.contains(float(g[0]), float(g[1])):
                    continue
                break
            else:
                raise RuntimeError("could not sample a push target")
            obj_xy.append(o)
            goal_xy.append(g)
        yaw = torch.tensor(rngs.uniform(-np.pi / 4, np.pi / 4), dtype=torch.float32)
        self._place_active(env_idx, torch.tensor(np.stack(obj_xy), dtype=torch.float32), yaw)
        g = torch.tensor(np.stack(goal_xy), dtype=torch.float32)
        self.target.set_pose(Pose.create_from_pq(torch.cat([g, torch.zeros(b, 1)], dim=1)))
        self._ever_lifted[env_idx] = False

    def goal_pose(self) -> torch.Tensor:
        return self.target.pose.raw_pose  # type: ignore[no-any-return]

    def evaluate(self) -> dict[str, torch.Tensor]:
        pose = self.obj_pose()
        lin, ang = self.obj_vel()
        self._ever_lifted |= S.is_lifted(pose[:, :3], self._obj_half)
        grasped = self.obj_grasped()
        out = S.push(
            pose[:, :3], lin, ang, self.target.pose.p, grasped, self._ever_lifted, self._obj_half
        )
        out["is_grasped"] = grasped
        out["ever_lifted"] = self._ever_lifted.clone()
        return out


@register_env("SO101Stack-v1", max_episode_steps=DEFAULT_MAX_EPISODE_STEPS)
class SO101StackEnv(SO101TabletopEnv):
    TASK = "Stack"

    def _load_task_objects(self) -> None:
        self.base_spec = P.BLUE_CUBE
        self.base = self._build_object(self.base_spec, "base", (0.1, 0.2))

    def _init_task(self, env_idx: torch.Tensor, rngs: Any) -> None:
        b = len(env_idx)
        pts = np.stack(
            [
                _sample_points(
                    rngs[j],
                    2,
                    self.region,
                    self.exclude,
                    MIN_SEP["Stack"],
                    [BASE_KEEPOUT + 0.025, BASE_KEEPOUT + 0.025],
                )
                for j in range(b)
            ]
        )
        pts_t = torch.tensor(pts, dtype=torch.float32)
        yaw = torch.tensor(rngs.uniform(-np.pi / 4, np.pi / 4, size=(2,)), dtype=torch.float32)
        self._place_active(env_idx, pts_t[:, 0], yaw[:, 0])
        h = self.base_spec.size / 2
        p = torch.cat([pts_t[:, 1], torch.full((b, 1), h)], dim=1)
        self.base.set_pose(Pose.create_from_pq(p, _q_yaw(yaw[:, 1])))

    def goal_pose(self) -> torch.Tensor:
        return self.base.pose.raw_pose  # type: ignore[no-any-return]

    def evaluate(self) -> dict[str, torch.Tensor]:
        pose = self.obj_pose()
        lin, ang = self.obj_vel()
        out = S.stack(
            pose[:, :3],
            lin,
            ang,
            self.base.pose.p,
            self.obj_touching(),
            self._obj_half,
            self.base_spec.size / 2,
        )
        out["is_grasped"] = self.obj_grasped()
        return out


TASK_ENV_IDS: dict[str, str] = {
    "PickPlace": "SO101PickPlace-v1",
    "Push": "SO101Push-v1",
    "Stack": "SO101Stack-v1",
}
