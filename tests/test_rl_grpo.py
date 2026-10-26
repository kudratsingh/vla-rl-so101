"""Hand-computed checks of the GRPO math in src/rl/grpo.py (WO-03 benchmark items a, b, c)."""

import math

import pytest
import torch

from src.rl.grpo import (
    clip_higher_policy_loss,
    dynamic_sampling_keep,
    group_advantages,
    seq_mean_token_mean_weights,
)

EPS = 1e-6


def test_group_advantage_hand_computed() -> None:
    rewards = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    adv = group_advantages(rewards, eps=EPS)
    # group 0: mean 0.5, deviations +-0.5, unbiased var = 4 * 0.25 / (4 - 1) = 1/3,
    #          std = sqrt(1/3) = 0.5773503 -> A = +-0.5 / (0.5773503 + 1e-6) = +-0.8660239
    g0 = 0.5 / (math.sqrt(1.0 / 3.0) + EPS)
    assert adv[0].tolist() == pytest.approx([g0, -g0, -g0, g0], abs=1e-6)
    assert g0 == pytest.approx(0.8660239, abs=1e-6)
    # group 1: mean 0.25, deviations 0.75, -0.25 x3, sum sq = 0.5625 + 3 * 0.0625 = 0.75,
    #          var = 0.75 / 3 = 0.25, std = 0.5 -> A = 0.75 / 0.500001 = 1.499997,
    #          -0.25 / 0.500001 = -0.499999
    assert adv[1].tolist() == pytest.approx([1.499997, -0.499999, -0.499999, -0.499999], abs=1e-6)
    # all-equal groups: std = 0 -> A = 0 / (0 + 1e-6) = 0 exactly (no gradient, dilutes the mean)
    assert adv[2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert adv[3].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_group_advantage_rejects_singleton_groups() -> None:
    with pytest.raises(ValueError):
        group_advantages(torch.ones(3, 1))


def test_clip_higher_hand_computed() -> None:
    # clip range [1 - 0.2, 1 + 0.28] = [0.8, 1.28]; ratio = exp(logp - old) with old = 0.
    # per token loss = max(-A r, -A clamp(r, 0.8, 1.28)); d loss / d logp = -A r if unclipped else 0
    #   A=+1  r=0.50: max(-0.50, -0.80) = -0.50  unclipped  grad -0.50
    #   A=+1  r=1.00: -1.00                      unclipped  grad -1.00
    #   A=+1  r=1.25: max(-1.25, -1.25) = -1.25  unclipped  grad -1.25  (symmetric 0.2 would clip)
    #   A=+1  r=1.50: max(-1.50, -1.28) = -1.28  CLIPPED    grad  0
    #   A=-1  r=0.50: max( 0.50,  0.80) =  0.80  CLIPPED    grad  0
    #   A=-1  r=1.00:  1.00                      unclipped  grad +1.00
    #   A=-1  r=1.25:  1.25                      unclipped  grad +1.25
    #   A=-1  r=1.50: max( 1.50,  1.28) =  1.50  unclipped  grad +1.50
    ratios = torch.tensor([0.5, 1.0, 1.25, 1.5, 0.5, 1.0, 1.25, 1.5], dtype=torch.float64)
    adv = torch.tensor([1.0] * 4 + [-1.0] * 4, dtype=torch.float64)
    logp = ratios.log().requires_grad_(True)
    old = torch.zeros_like(logp)
    w = torch.full_like(logp, 1.0 / 8.0)
    out = clip_higher_policy_loss(logp, old, adv, w, clip_low=0.2, clip_high=0.28)
    out.loss.backward()
    assert logp.grad is not None
    # loss = (-0.5 - 1 - 1.25 - 1.28 + 0.8 + 1 + 1.25 + 1.5) / 8 = 0.52 / 8 = 0.065
    assert float(out.loss) == pytest.approx(0.065, abs=1e-12)
    expected_grad = [-0.5, -1.0, -1.25, 0.0, 0.0, 1.0, 1.25, 1.5]
    assert (logp.grad * 8).tolist() == pytest.approx(expected_grad, abs=1e-12)
    # 2 of 8 tokens clipped (A=+1 r=1.5, A=-1 r=0.5)
    assert float(out.clip_frac) == pytest.approx(0.25)
    # approx_kl = mean(old - logp) = -2 (ln .5 + ln 1 + ln 1.25 + ln 1.5) / 8 = -ln(0.9375) / 4
    assert float(out.approx_kl) == pytest.approx(-math.log(0.9375) / 4, abs=1e-12)


def test_clip_loss_ignores_padding_tokens() -> None:
    logp = torch.tensor([math.log(1.5), float("nan")], requires_grad=True)
    out = clip_higher_policy_loss(
        logp, torch.zeros(2), torch.ones(2), torch.tensor([1.0, 0.0]), clip_low=0.2, clip_high=0.28
    )
    out.loss.backward()
    assert logp.grad is not None
    assert float(out.loss) == pytest.approx(-1.28)  # only the valid, clipped token
    assert logp.grad.tolist() == [0.0, 0.0]
    assert math.isfinite(float(out.approx_kl))


def test_dynamic_sampling_drops_all_equal_groups() -> None:
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],  # all fail    -> drop
            [1.0, 1.0, 1.0, 1.0],  # all succeed -> drop
            [1.0, 0.0, 1.0, 0.0],  # mixed       -> keep
            [0.0, 0.0, 0.0, 1.0],  # mixed       -> keep
        ]
    )
    assert dynamic_sampling_keep(rewards).tolist() == [False, False, True, True]
    # upstream's accuracy filter (mean in [0.1, 0.9]) gives the same answer for binary K <= 10
    mean = rewards.mean(dim=1)
    assert ((mean >= 0.1) & (mean <= 0.9)).tolist() == [False, False, True, True]


def test_seq_mean_token_mean_matches_upstream_aggregation() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])
    per_token_loss = torch.tensor([[1.0, 3.0, 99.0], [5.0, 99.0, 99.0]])
    w = seq_mean_token_mean_weights(mask)
    # w = mask / (tokens in trajectory * trajectories) = [[1/4, 1/4, 0], [1/2, 0, 0]]
    assert w.tolist() == [[0.25, 0.25, 0.0], [0.5, 0.0, 0.0]]
    # upstream dp_rob: mean over trajectories of each trajectory's token-mean = ((1+3)/2 + 5)/2
    assert float((w * per_token_loss).sum()) == pytest.approx(3.5)
    # (DAPO token-mean would be (1 + 3 + 5) / 3 = 3.0 -- not what upstream does)
    with pytest.raises(ValueError):
        seq_mean_token_mean_weights(torch.tensor([[True], [False]]))
