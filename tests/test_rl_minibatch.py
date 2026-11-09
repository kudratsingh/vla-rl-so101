"""PPO mini-batches are equal whole-group splits (owner 2026-11-08): a short tail of 1-7 groups
never gets its own full Adam step, no group is split, every group is used once per epoch."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.config import load_config
from src.rl.policy_api import build_policy
from src.rl.toy_env import make_toy_env_factory, policy_obs
from src.rl.trainer import GRPOConfig, GRPOTrainer, group_trajectories, ppo_minibatches


@pytest.mark.parametrize(
    ("n_groups", "sizes"),
    [(1, [1]), (7, [7]), (8, [8]), (9, [5, 4]), (11, [6, 5]), (16, [8, 8]), (17, [6, 6, 5])],
)
def test_equal_whole_group_minibatches(n_groups: int, sizes: list[int]) -> None:
    mbs = ppo_minibatches(n_groups, 8)  # configs/rl.yaml: ppo_mini_batch 64 / K 8
    assert [len(g) for g in mbs] == sizes
    assert len(mbs) == -(-n_groups // 8)  # ceil: the old optimizer-step count is unchanged
    assert max(sizes) - min(sizes) <= 1
    assert [g for mb in mbs for g in mb] == list(range(n_groups))  # every group once, in order
    traj = torch.cat([group_trajectories(mb, 8) for mb in mbs])
    assert traj.tolist() == list(range(n_groups * 8))  # whole groups: e = g * K + k
    assert ppo_minibatches(0, 8) == []
    with pytest.raises(ValueError):
        ppo_minibatches(4, 0)


def test_update_steps_on_equal_whole_groups(tmp_path: Path) -> None:
    """11 kept groups (refill cap hit) at 8 groups per mini-batch: 2 Adam steps per epoch on
    6 + 5 whole groups, each group exactly once per epoch, never 8 + a 3-group tail."""
    cfg = load_config("tiny")  # K = 4
    # 8 groups per mini-batch; no scoring forwards (they gather too)
    cfg["rl"].update(ppo_mini_batch=32, ppo_epochs=2, diag_every=0, recompute_old_log_probs=False)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    tr = GRPOTrainer(
        rl=GRPOConfig.from_config(cfg),
        policy=build_policy(cfg, "cpu"),
        env_factory=make_toy_env_factory(cfg),
        obs_fn=policy_obs,
        tasks=cfg["env"]["tasks"],
        max_episode_steps=int(cfg["toy"]["max_episode_steps"]),
        run_dir=tmp_path,
        seed=0,
    )
    batch, _ = tr.collect()
    assert batch is not None and batch.num_groups > 11
    keep = torch.zeros(batch.num_groups, dtype=torch.bool)
    keep[:11] = True
    batch = batch.select_groups(keep)

    steps: list[set[int]] = [set()]
    gather = tr._gather

    def spy(b: Any, e: torch.Tensor, s: torch.Tensor) -> Any:
        steps[-1].update(e.tolist())
        return gather(b, e, s)

    step = tr.optimizer.step

    def spy_step(*a: Any, **kw: Any) -> Any:
        steps.append(set())
        return step(*a, **kw)

    tr._gather = spy  # type: ignore[method-assign]
    tr.optimizer.step = spy_step  # type: ignore[method-assign]
    stats = tr.update(batch)
    steps = steps[:-1]  # the set opened by the last step stays empty
    assert stats["train/minibatch_groups"] == [6, 5]
    assert stats["train/optimizer_steps"] == 4.0  # 2 epochs x 2 mini-batches
    groups = [{e // 4 for e in s} for s in steps]
    assert [len(g) for g in groups] == [6, 5, 6, 5]
    for s, g in zip(steps, groups, strict=True):
        assert s == {gi * 4 + k for gi in g for k in range(4)}  # no group split
    for epoch in (groups[:2], groups[2:]):
        assert sorted(gi for g in epoch for gi in g) == list(range(11))  # each group once
