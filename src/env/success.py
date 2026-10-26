"""Programmatic success checks (the ONLY reward in this project, CLAUDE.md rule 5).

Pure tensor functions so they are testable without a simulator. Every check requires the object
to be released (no jaw contact / not grasped) and at rest, so holding the object over the goal,
flinging it through the goal or pinning it in place does not count.

All inputs are batched over sub-envs (N, ...); every returned value is a bool tensor (N,).
"""

from __future__ import annotations

import torch

# Thresholds (metres, m/s).
REST_LIN_VEL = 0.03  # object counts as "at rest" below this linear speed
REST_ANG_VEL = 0.5  # rad/s
ON_TABLE_TOL = 0.006  # centre height may exceed half height by this much and still be "on table"
LIFT_TOL = 0.012  # centre above half height + this => "lifted" (Push latch)

# Bowl geometry (src/env/tasks.py builds it from these).
BOWL_FLOOR_RADIUS = 0.045
BOWL_FLOOR_THICKNESS = 0.006
BOWL_WALL_HEIGHT = 0.03
BOWL_WALL_TILT_DEG = 25.0
BOWL_WALL_THICKNESS = 0.005

# Push target (flat marked disc on the table).
TARGET_RADIUS = 0.025

# Stack: A's centre must be over B's top face, within this radius of B's axis.
STACK_XY_TOL = 0.015
STACK_Z_TOL = 0.006


def at_rest(lin_vel: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
    rest: torch.Tensor = (lin_vel.norm(dim=-1) < REST_LIN_VEL) & (
        ang_vel.norm(dim=-1) < REST_ANG_VEL
    )
    return rest


def is_lifted(obj_pos: torch.Tensor, half_height: torch.Tensor | float) -> torch.Tensor:
    return obj_pos[:, 2] > half_height + LIFT_TOL


def pick_place(
    obj_pos: torch.Tensor,
    obj_lin_vel: torch.Tensor,
    obj_ang_vel: torch.Tensor,
    bowl_pos: torch.Tensor,
    bowl_up_z: torch.Tensor,
    touching: torch.Tensor,
    half_height: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Object inside the (upright) bowl, released and at rest.

    Inside = centre horizontally over the bowl floor and vertically between floor and rim; the
    3 cm cube cannot be slid over the 3 cm rim, so knocking it in is not a cheap path.
    """
    rel = obj_pos - bowl_pos
    in_xy = rel[:, :2].norm(dim=-1) <= BOWL_FLOOR_RADIUS
    in_z = (rel[:, 2] > BOWL_FLOOR_THICKNESS) & (rel[:, 2] <= BOWL_WALL_HEIGHT + half_height)
    upright = bowl_up_z > 0.9
    in_bowl = in_xy & in_z & upright
    rest = at_rest(obj_lin_vel, obj_ang_vel)
    success = in_bowl & ~touching & rest
    return {"success": success, "obj_in_bowl": in_bowl, "obj_at_rest": rest}


def push(
    obj_pos: torch.Tensor,
    obj_lin_vel: torch.Tensor,
    obj_ang_vel: torch.Tensor,
    goal_pos: torch.Tensor,
    grasped: torch.Tensor,
    ever_lifted: torch.Tensor,
    half_height: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Object centre inside the target disc, on the table, never lifted this episode, not grasped,
    at rest. `ever_lifted` is an episode latch kept by the env (pick-and-place is not a push)."""
    dist = (obj_pos - goal_pos)[:, :2].norm(dim=-1)
    in_target = dist <= TARGET_RADIUS
    on_table = obj_pos[:, 2] <= half_height + ON_TABLE_TOL
    rest = at_rest(obj_lin_vel, obj_ang_vel)
    success = in_target & on_table & ~grasped & ~ever_lifted & rest
    return {"success": success, "obj_in_target": in_target, "obj_at_rest": rest}


def stack(
    a_pos: torch.Tensor,
    a_lin_vel: torch.Tensor,
    a_ang_vel: torch.Tensor,
    b_pos: torch.Tensor,
    touching_a: torch.Tensor,
    half_a: torch.Tensor | float,
    half_b: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """A resting on top of B (B still on the table), A released and at rest."""
    rel = a_pos - b_pos
    xy_ok = rel[:, :2].norm(dim=-1) <= STACK_XY_TOL
    z_ok = (rel[:, 2] - (half_a + half_b)).abs() <= STACK_Z_TOL
    b_on_table = b_pos[:, 2] <= half_b + ON_TABLE_TOL
    on_top = xy_ok & z_ok & b_on_table
    rest = at_rest(a_lin_vel, a_ang_vel)
    success = on_top & ~touching_a & rest
    return {"success": success, "obj_on_base": on_top, "obj_at_rest": rest}
