"""Paired comparison of two rollout arms on the same groups (the clip probe, owner 2026-11-08).

Written while `src/eval/paired.py` (recovery SFT, PR #25) was not on main; both now exist and
serve different inputs. `src/eval/paired.py` reads two protocol eval run dirs (`episodes.jsonl`,
independent protocol episodes, one-sided McNemar as the plan's test, protocol hash checked). This
module takes two in-memory (G, K) RL rollout reward tensors whose row g and column k are the same
prompt (env reset seed) and the same sample slot (sampling seed) in both arms; its unit is the
group, and its two-sided McNemar is a count, not the test. Only the binomial tail is shared.

The unit for the success difference is the GROUP: its K samples share a scene, so episodes within
a group are not independent. `success_diff_se` is the SE of the mean per-group difference over
groups. The episode table and its exact McNemar p treat episodes as independent pairs and are
therefore optimistic; they are reported as counts, not as the test.
"""

from __future__ import annotations

import math

from torch import Tensor

from src.rl.grpo import dynamic_sampling_keep


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs (binomial, p = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2.0**n
    return min(1.0, 2.0 * tail)


def paired_group_rewards(off: Tensor, on: Tensor) -> dict[str, float]:
    """Per-group success difference (on - off) with its SE over groups, the episode-level paired
    count table (both / off only / on only / neither succeeded) with its exact McNemar p, and the
    paired table of groups dropped by dynamic sampling (all-equal groups)."""
    if off.shape != on.shape or off.dim() != 2:
        raise ValueError(f"need two (G, K) reward tensors of one shape, got {off.shape} {on.shape}")
    a, b = off.bool(), on.bool()
    g = off.shape[0]
    diff = on.float().mean(dim=1) - off.float().mean(dim=1)
    se = float(diff.std() / math.sqrt(g)) if g > 1 else 0.0
    off_only, on_only = int((a & ~b).sum()), int((~a & b).sum())
    drop_a, drop_b = ~dynamic_sampling_keep(off), ~dynamic_sampling_keep(on)
    return {
        "groups": float(g),
        "episodes_per_arm": float(off.numel()),
        "success_off": float(off.float().mean()),
        "success_on": float(on.float().mean()),
        "success_diff_on_minus_off": float(diff.mean()),
        "success_diff_se": se,
        "episodes_both_success": float((a & b).sum()),
        "episodes_off_only": float(off_only),
        "episodes_on_only": float(on_only),
        "episodes_neither": float((~a & ~b).sum()),
        "episodes_mcnemar_exact_p": mcnemar_exact_p(off_only, on_only),
        "groups_dropped_off": float(drop_a.float().mean()),
        "groups_dropped_on": float(drop_b.float().mean()),
        "groups_dropped_both": float((drop_a & drop_b).sum()),
        "groups_dropped_off_only": float((drop_a & ~drop_b).sum()),
        "groups_dropped_on_only": float((~drop_a & drop_b).sum()),
        "groups_kept_both": float((~drop_a & ~drop_b).sum()),
    }
