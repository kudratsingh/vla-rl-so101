"""Smoke tests for the spend gates (src/rl/gates.py) and the movement diagnostic
(src/rl/diagnostics.py), on hand-made numbers."""

import math

import pytest
import torch

from src.rl.diagnostics import argmax_flip_frac, entropy_from_sampled_log_probs, movement_stats
from src.rl.gates import RolloutGates, TaskTally, pooled_rate


def _gates(**kw: object) -> RolloutGates:
    args: dict[str, object] = {
        "tasks": ["PickPlace", "Push"],
        "window": 2,
        "check_step": 4,
        "min_gain": 0.03,
        "rule": "any_task",
        "max_drop": 0.0,
        "entropy_min": 0.0,
    }
    args.update(kw)
    return RolloutGates(**args)  # type: ignore[arg-type]


def _feed(g: RolloutGates, step: int, pick: float, push: float, ent: float = 1.0) -> None:
    """10 groups per task per step, every group at the given mean success."""
    tallies = {
        "PickPlace": TaskTally.from_group_rates([pick] * 10),
        "Push": TaskTally.from_group_rates([push] * 10),
    }
    g.record(step, tallies, entropy_sum=ent * 100, entropy_tokens=100)


def test_pooled_rate_uses_groups_as_the_unit() -> None:
    wr = pooled_rate([TaskTally.from_group_rates([0.0, 0.5]), TaskTally.from_group_rates([1.0])])
    assert wr is not None and wr.groups == 3 and wr.rate == pytest.approx(0.5)
    assert wr.se == pytest.approx(math.sqrt(0.25 / 3))  # sample var of (0, .5, 1) = .25
    assert pooled_rate([]) is None


def test_flat_gate_any_task_and_mean_rules() -> None:
    """Windows 1-2 vs 3-4: PickPlace +3.1 with Push -5 passes any_task, fails mean; +2.9 fails."""
    cases = (
        ("any_task", (0.531, 0.45), None),
        ("any_task", (0.529, 0.529), "flat"),
        ("mean", (0.531, 0.45), "flat"),
        ("mean", (0.54, 0.54), None),
    )
    for rule, (pick, push), want in cases:
        g = _gates(rule=rule)
        for step in (1, 2):
            _feed(g, step, 0.5, 0.5)
        for step in (3, 4):
            _feed(g, step, pick, push)
        assert g.evaluate(3).reason is None  # not the check step
        res = g.evaluate(4)
        assert res.reason == want, (rule, pick, push)
        assert ("event/flat_run" in res.metrics) is (want == "flat")
        assert res.metrics["gate/success_window"] == pytest.approx((pick + push) / 2)
    with pytest.raises(ValueError, match="overlap"):
        _gates(check_step=3)  # < 2 x window


def test_decline_and_entropy_gates_and_state_round_trip() -> None:
    g = _gates(check_step=0, max_drop=0.05, entropy_min=0.5)
    for step, rate in enumerate([0.2, 0.2, 0.3, 0.3, 0.27, 0.27], start=1):
        _feed(g, step, rate, rate)
        assert g.evaluate(step).reason is None  # -3 points from the best: within max_drop
    state = g.state_dict()
    _feed(g, 7, 0.2, 0.2)
    res = g.evaluate(7)  # window 6-7 = 0.235 < 0.30 - 0.05, best window (3-4) is >= 2 steps back
    assert res.reason == "decline" and res.metrics["gate/best_success_step"] == 4.0
    # restored state behaves identically
    h = _gates(check_step=0, max_drop=0.05, entropy_min=0.5)
    h.load_state_dict(state)
    _feed(h, 7, 0.2, 0.2)
    assert h.evaluate(7).metrics == res.metrics
    e = _gates(check_step=0, entropy_min=0.5)
    _feed(e, 1, 0.2, 0.2, ent=0.6)
    _feed(e, 2, 0.2, 0.2, ent=0.3)
    res = e.evaluate(2)
    assert res.reason == "entropy" and res.metrics["gate/entropy_window"] == pytest.approx(0.45)


def test_movement_stats_on_hand_values() -> None:
    # 4 tokens, reference log-prob 0; ratios 1.5, 0.5 (A = +1), 1.1, 0.7 (A = -1); one padding
    ratio = torch.tensor([1.5, 0.5, 1.1, 0.7, 9.0], dtype=torch.float64)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
    valid = torch.tensor([True, True, True, True, False])
    out = movement_stats(
        ratio.log(), torch.zeros(5), adv, valid, clip_low=0.2, clip_high=0.28
    )  # fmt: skip
    d = ratio[:4].log()
    assert out["tokens"] == 4 and out["tokens_pos_adv"] == 2
    assert out["mean_abs_dlogp"] == pytest.approx(float(d.abs().mean()))
    assert out["mean_dlogp_pos_adv"] == pytest.approx(float(d[:2].mean()))
    assert out["mean_dlogp_neg_adv"] == pytest.approx(float(d[2:].mean()))
    assert out["mean_adv_dlogp"] == pytest.approx(float((d[:2].sum() - d[2:].sum()) / 4))
    assert out["k3_kl"] == pytest.approx(float(((ratio[:4] - 1) - d).mean()))
    # clipped: A>0 r=1.5 > 1.28, A<0 r=0.7 < 0.8 -> 2/4; outside either way adds r=0.5 -> 3/4
    assert out["clip_frac"] == pytest.approx(0.5)
    assert out["frac_outside_clip"] == pytest.approx(0.75)
    zero = movement_stats(torch.zeros(5), torch.zeros(5), adv, valid, clip_low=0.2, clip_high=0.28)
    assert zero["k3_kl"] == zero["mean_abs_dlogp"] == zero["clip_frac"] == 0.0
    greedy = torch.tensor([0, 1, 2, 3, 4])
    assert argmax_flip_frac(greedy, torch.tensor([0, 1, 0, 3, 0]), valid) == pytest.approx(0.25)
    s, n = entropy_from_sampled_log_probs(torch.tensor([-1.0, -2.0, -9.0]), valid[2:].clone())
    assert (s, n) == (pytest.approx(3.0), 2)  # -log p of the two valid sampled tokens
    with pytest.raises(ValueError, match="no valid"):
        movement_stats(torch.zeros(2), torch.zeros(2), torch.ones(2), torch.zeros(2, dtype=bool),
                       clip_low=0.2, clip_high=0.28)  # fmt: skip
