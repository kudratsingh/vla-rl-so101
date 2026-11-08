"""Recovery SFT data selection (docs/plans/recovery-sft.md): failure taxonomy, branch states.

From a `RolloutTrace` (src/eval/sim_eval.py) of failed SFT episodes:

1. `classify` names each failure's mode and the step of its decisive event, per task:
   PickPlace/Stack: dropped (grasped, then lost away from the goal), misplaced (released near
   the goal, not a success), knocked (object pushed > 2 cm without a grasp), stalled (arm still
   for the last 16 steps of a timed-out episode), never_grasped (none of these).
   Push: lifted (the latch, success impossible afterwards), no_contact (object moved < 1 cm),
   missed (moved, not into the target).
2. `choose_branch` picks the chunk boundary `back` boundaries before that event at which the
   state is still recoverable (`recoverable`): object within reach, bowl upright, Stack base flat
   on the table, Push never lifted yet. No such boundary -> the failure is unrecoverable.
3. `interleave` caps each mode and orders the branches round-robin across modes, so any prefix
   of the recorded corrections (the per-task cap at load time) keeps the mode mix.

Pure tensor functions over one episode's series, so they are testable without a simulator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from src.env import success as S

REACH = 0.31  # m from the base: beyond it the top-down oracle grasp is not reachable
MOVED = 0.02  # m of object travel that counts as "knocked" (PickPlace/Stack)
PUSH_MOVED = 0.01  # m: below this the pusher never moved the object
NEAR_GOAL = {"PickPlace": 2 * S.BOWL_FLOOR_RADIUS, "Stack": 0.04}
STALL_STEPS = 16  # two 8-action chunks
STALL_TOL = 0.01  # rad
STACK_BASE_HALF = 0.015  # src/eval/protocol.py BLUE_CUBE size / 2
UPRIGHT = 0.9

MODES: dict[str, tuple[str, ...]] = {
    "PickPlace": ("dropped", "misplaced", "knocked", "stalled", "never_grasped"),
    "Stack": ("dropped", "misplaced", "knocked", "stalled", "never_grasped"),
    "Push": ("lifted", "no_contact", "missed"),
}

Series = Mapping[str, Tensor]  # one env: key -> (T + 1, ...) over steps 0..T


def episode_series(steps: Sequence[Mapping[str, Tensor]], row: int) -> dict[str, Tensor]:
    """One env's series from a trace's batched per-step dicts (keys present at every step;
    "action", absent at the reset step, is dropped)."""
    keys = [k for k in steps[0] if all(k in s for s in steps)]
    return {k: torch.stack([s[k][row] for s in steps]) for k in keys}


def _up_z(quat: Tensor) -> Tensor:
    """z component of the body z axis, from (..., 4) wxyz quaternions."""
    _, x, y, _ = quat.unbind(-1)
    out: Tensor = 1 - 2 * (x * x + y * y)
    return out


def _flag(s: Series, key: str, length: int) -> Tensor:
    if key not in s:
        return torch.zeros(length + 1, dtype=torch.bool)
    return s[key][: length + 1].reshape(length + 1, -1)[:, 0].bool()


def classify(task: str, s: Series, length: int, max_steps: int) -> tuple[str, int]:
    """(failure mode, event step) of a failed episode of `length` steps."""
    obj = s["obj_pose"][: length + 1, :3]
    goal = s["goal_pose"][: length + 1, :3]
    tcp = s["tcp_pose"][: length + 1, :3]
    qpos = s["qpos"][: length + 1]
    grasped = _flag(s, "is_grasped", length)
    moved = (obj[:, :2] - obj[0, :2]).norm(dim=-1)
    tcp_obj = (tcp - obj).norm(dim=-1)
    if task == "Push":
        lifted = _flag(s, "ever_lifted", length)
        if bool(lifted.any()):
            return "lifted", int(lifted.float().argmax())
        if float(moved.max()) < PUSH_MOVED:
            return "no_contact", int(tcp_obj.argmin())
        return "missed", int((obj[:, :2] - goal[:, :2]).norm(dim=-1).argmin())
    if bool(grasped.any()):
        t_last = int(torch.nonzero(grasped).max())
        near = float((obj[t_last, :2] - goal[t_last, :2]).norm()) < NEAR_GOAL[task]
        return ("misplaced" if near else "dropped"), t_last
    if float(moved.max()) > MOVED:
        return "knocked", int(torch.nonzero(moved > MOVED / 2).min())
    still = (qpos - qpos[-1]).abs().amax(-1) < STALL_TOL
    if length >= max_steps and bool(still[-STALL_STEPS:].all()):
        onset = length
        while onset > 0 and bool(still[onset - 1]):
            onset -= 1
        return "stalled", onset
    return "never_grasped", int(tcp_obj.argmin())


def recoverable(task: str, s: Series, t: int) -> bool:
    """Can the oracle still succeed from step t? Checks the irreversible cases only."""
    obj = s["obj_pose"][t]
    goal = s["goal_pose"][t]
    if float(obj[:2].norm()) > REACH or float(obj[2]) < -0.01:
        return False
    if task == "PickPlace":
        return float(_up_z(goal[3:])) > UPRIGHT
    if task == "Stack":
        flat = abs(float(_up_z(goal[3:]))) > UPRIGHT
        return flat and float(goal[2]) <= STACK_BASE_HALF + S.ON_TABLE_TOL
    if task == "Push":
        return not ("ever_lifted" in s and bool(s["ever_lifted"][t]))
    raise ValueError(f"unknown task {task!r}")


def choose_branch(
    task: str, s: Series, boundary_steps: Sequence[int], event: int, length: int, back: int
) -> int | None:
    """Index into `boundary_steps` of the branch state: `back` boundaries before the last one at
    or before `event` (at least the reset state), moved earlier until `recoverable`. None if no
    boundary qualifies."""
    cands = [i for i, b in enumerate(boundary_steps) if b <= min(event, length - 1)]
    if not cands:
        return None
    k = max(cands[-1] - back, 0)
    while k >= 0:
        if recoverable(task, s, int(boundary_steps[k])):
            return k
        k -= 1
    return None


@dataclass(frozen=True)
class Branch:
    """One correction to try: restore `seed`'s episode at boundary `boundary` (step `step`)."""

    task: str
    seed: int
    mode: str
    event_step: int
    boundary: int
    step: int
    in_hand: bool
    grip_cmd: float
    batch: int  # rollout batch index and row of the source episode (snapshot lookup)
    row: int


def interleave(branches: Sequence[Branch], cap_per_mode: int) -> list[Branch]:
    """At most `cap_per_mode` per mode (first in seed order), ordered round-robin over modes."""
    by_mode: dict[str, list[Branch]] = {}
    for b in sorted(branches, key=lambda x: (x.seed, x.step)):
        by_mode.setdefault(b.mode, [])
        if len(by_mode[b.mode]) < cap_per_mode:
            by_mode[b.mode].append(b)
    order = [m for m in MODES.get(branches[0].task, ()) if m in by_mode] if branches else []
    order += sorted(m for m in by_mode if m not in order)
    out: list[Branch] = []
    for i in range(max((len(v) for v in by_mode.values()), default=0)):
        out += [by_mode[m][i] for m in order if i < len(by_mode[m])]
    return out


def yield_table(tried: Sequence[Branch], ok: Sequence[bool]) -> dict[str, dict[str, float]]:
    """Per mode: branches tried, oracle successes, yield."""
    out: dict[str, dict[str, float]] = {}
    for b, good in zip(tried, ok, strict=True):
        row = out.setdefault(b.mode, {"tried": 0, "ok": 0, "yield": math.nan})
        row["tried"] += 1
        row["ok"] += int(good)
    for row in out.values():
        row["yield"] = row["ok"] / row["tried"] if row["tried"] else math.nan
    return out
