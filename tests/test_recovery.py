"""Failure taxonomy + branch-state choice (src/data/recovery.py) on synthetic episode series."""

from __future__ import annotations

import pytest
import torch

from src.data.recovery import (
    Branch,
    choose_branch,
    classify,
    episode_series,
    interleave,
    recoverable,
    yield_table,
)

T = 40
UPRIGHT_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])


def series(task: str = "PickPlace") -> dict[str, torch.Tensor]:
    """Object at (0.2, 0), goal 10 cm to the side, arm moving, nothing grasped."""
    obj = torch.cat([torch.tensor([0.2, 0.0, 0.015]), UPRIGHT_Q]).repeat(T + 1, 1)
    goal = torch.cat([torch.tensor([0.2, 0.1, 0.0]), UPRIGHT_Q]).repeat(T + 1, 1)
    tcp = obj.clone()
    tcp[:, 2] = torch.linspace(0.2, 0.05, T + 1)
    qpos = torch.linspace(0, 1, T + 1)[:, None].repeat(1, 6)
    s = {
        "obj_pose": obj,
        "goal_pose": goal,
        "tcp_pose": tcp,
        "qpos": qpos,
        "is_grasped": torch.zeros(T + 1, dtype=torch.bool),
    }
    if task == "Push":
        s["ever_lifted"] = torch.zeros(T + 1, dtype=torch.bool)
    return s


def test_pick_place_modes() -> None:
    s = series()
    assert classify("PickPlace", s, T, T)[0] == "never_grasped"
    s["obj_pose"][20:, 0] += 0.05  # knocked away without a grasp
    assert classify("PickPlace", s, T, T) == ("knocked", 20)
    s = series()
    s["is_grasped"][10:25] = True  # grasped, lost far from the bowl
    assert classify("PickPlace", s, T, T) == ("dropped", 24)
    s["obj_pose"][24, :3] = s["goal_pose"][24, :3]  # ... or released over the bowl
    assert classify("PickPlace", s, T, T)[0] == "misplaced"
    s = series()
    s["qpos"][20:] = s["qpos"][20]  # arm frozen from step 20 to the timeout
    assert classify("PickPlace", s, T, T) == ("stalled", 20)
    assert classify("PickPlace", s, T, max_steps=150)[0] == "never_grasped"  # not timed out


def test_push_modes_and_latch() -> None:
    s = series("Push")
    assert classify("Push", s, T, T)[0] == "no_contact"
    s["obj_pose"][10:, 1] += torch.linspace(0, 0.05, T - 9)
    mode, event = classify("Push", s, T, T)
    assert mode == "missed" and event == T  # closest to the goal at the end
    s["ever_lifted"][30:] = True
    assert classify("Push", s, T, T) == ("lifted", 30)
    assert recoverable("Push", s, 29) and not recoverable("Push", s, 30)


def test_choose_branch_backs_off_to_a_recoverable_state() -> None:
    s = series()
    bounds = [0, 8, 16, 24, 32]
    assert choose_branch("PickPlace", s, bounds, event=20, length=T, back=1) == 1
    assert choose_branch("PickPlace", s, bounds, event=20, length=T, back=0) == 2
    assert choose_branch("PickPlace", s, bounds, event=3, length=T, back=1) == 0  # reset state
    s["goal_pose"][8:, 3:] = torch.tensor([0.7071, 0.7071, 0.0, 0.0])  # bowl tipped from step 8
    assert choose_branch("PickPlace", s, bounds, event=20, length=T, back=0) == 0
    s["obj_pose"][:, 0] = 0.5  # object out of reach the whole episode
    assert choose_branch("PickPlace", s, bounds, event=20, length=T, back=0) is None
    s = series("Stack")
    s["goal_pose"][:, 2] = 0.05  # base cube lifted off the table
    assert not recoverable("Stack", s, 0)


def test_interleave_caps_and_round_robins() -> None:
    def br(mode: str, seed: int) -> Branch:
        return Branch("PickPlace", seed, mode, 5, 0, 0, False, 0.6, 0, 0)

    bs = [br("knocked", s) for s in (5, 1, 3)] + [br("dropped", s) for s in (4, 2)]
    out = interleave(bs, cap_per_mode=2)
    assert [(b.mode, b.seed) for b in out] == [
        ("dropped", 2),
        ("knocked", 1),
        ("dropped", 4),
        ("knocked", 3),
    ]
    table = yield_table(out, [True, False, True, True])
    assert table["dropped"] == {"tried": 2, "ok": 2, "yield": 1.0}
    assert interleave([], 3) == []


def test_episode_series_rows() -> None:
    steps = [{"qpos": torch.full((2, 6), float(t))} for t in range(3)]
    steps[1]["action"] = torch.zeros(2, 6)
    s = episode_series(steps, 1)
    assert set(s) == {"qpos"} and s["qpos"][:, 0].tolist() == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="unknown task"):
        recoverable("Lift", series(), 0)
