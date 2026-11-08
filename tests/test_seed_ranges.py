"""Seed-disjointness guard (src/eval/seed_ranges.py): every range the project uses is disjoint."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from src.eval import protocol as P
from src.eval import seed_ranges as SR
from src.eval.sim_eval import episode_seeds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_standard_ranges_are_disjoint_and_cover_their_seeds() -> None:
    import wo01_build_env

    assert SR.DEMO_SEED_BASE == wo01_build_env.DEMO_SEED_BASE
    ranges = SR.standard_ranges()
    SR.check_disjoint(ranges)
    names = {r.name for r in ranges}
    assert {"protocol[0]", "selection[7]", "demo[0]", "recovery", "control", "rl_eval"} <= names
    # every protocol episode seed (and the evaluate() batch padding) is inside "protocol"
    for s in P.SIM_SEEDS:
        for t in range(len(P.TASKS)):
            for c in range(len(P.CELLS)):
                seeds = episode_seeds(s, t, c, 2 * P.SIM_EPISODES_PER_CELL)
                SR.assert_seeds_in(seeds, "protocol", ranges)
    SR.assert_seeds_in(episode_seeds(7, 2, 0, 100), "selection", ranges)
    SR.assert_seeds_in([SR.DEMO_SEED_BASE + k for k in range(100)], "demo", ranges)
    for task in P.TASKS:
        SR.assert_seeds_in(SR.recovery_seeds(task, 500), "recovery", ranges)
        SR.assert_seeds_in(SR.control_seeds(task, 500), "control", ranges)
    assert not set(SR.recovery_seeds("Push", 10)) & set(SR.recovery_seeds("Stack", 10))
    SR.assert_seeds_in([3, 10_000_000 + 3_600, 20_000_000], "rl_train", ranges)
    SR.assert_seeds_in([4_000_000_000 + 63], "rl_eval", ranges)


def test_guard_rejects_overlaps_and_strays() -> None:
    with pytest.raises(ValueError, match="selection seeds"):
        SR.standard_ranges(selection_seeds=(0,))
    assert SR.MAX_RL_SEED == 10
    for bad in (10, 11, 39, -1):  # 10: train prompts 100M.. = protocol seed 0's episodes
        with pytest.raises(ValueError, match=r"must be in \[0, 10\)"):
            SR.check_rl_seed(bad)
    for ok in (0, 3, 9):
        SR.check_rl_seed(ok)
    with pytest.raises(ValueError, match="outside"):
        SR.assert_seeds_in([episode_seeds(0, 0, 0, 1)[0]], "recovery", SR.standard_ranges())
    leak = SR.SeedRange("leak", 100_000_000, 100_000_050)  # e.g. a sweep on protocol seed 0
    SR.assert_seeds_in([100_000_001], "leak", [leak])  # alone: fine
    with pytest.raises(ValueError, match="overlapping seed ranges: protocol"):
        SR.assert_seeds_in([100_000_001], "leak", [*SR.standard_ranges(), leak])
    with pytest.raises(ValueError, match="stride"):
        SR.recovery_seeds("Push", 10, start=SR.TASK_STRIDE)
