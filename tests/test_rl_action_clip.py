"""rl.rollout_action_clip_deg: off (null) by default, so RL rollouts send the policy's actions
unchanged (the WO-04 runs); when set, executed actions get protocol 0.2's per-joint step clip."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest
import torch

from src.config import load_config
from src.rl.policy_api import PolicyObs, RolloutStep
from src.rl.rollout import collect_rollouts
from src.rl.trainer import GRPOConfig

N, CHUNK = 4, 3


class JointEnv:
    """6-joint env: the obs state is the last commanded joint target; records executed actions."""

    def __init__(self) -> None:
        self.q = torch.zeros(N, 6)
        self.actions: list[torch.Tensor] = []
        self.t = 0

    @property
    def unwrapped(self) -> Any:
        return self

    def get_instructions(self) -> list[str]:
        return ["x"] * N

    def _obs(self) -> dict[str, torch.Tensor]:
        return {"image": torch.zeros(N, 2, 2, 3, dtype=torch.uint8), "state": self.q.clone()}

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        self.q.zero_()
        self.t = 0
        return self._obs(), {}

    def step(self, action: torch.Tensor) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self.actions.append(action.clone())
        self.q = action.clone()
        self.t += 1
        done = torch.full((N,), self.t >= 6)
        return (
            self._obs(),
            torch.zeros(N),
            torch.zeros(N, dtype=torch.bool),
            done,
            {"success": torch.zeros(N, dtype=torch.bool)},
        )


class JumpPolicy:
    """Always commands every joint to 1 rad (a 57 deg jump from the reset pose)."""

    def sample(self, obs: PolicyObs, instructions: Sequence[str], temperature: float) -> Any:
        return RolloutStep(
            actions=torch.ones(N, CHUNK, 6),
            tokens=torch.zeros(N, CHUNK, 6, dtype=torch.long),
            log_probs=torch.zeros(N, CHUNK, 6),
        )

    def eval(self) -> object:
        return self


def _run(clip: float | None) -> tuple[JointEnv, dict[str, int]]:
    env = JointEnv()
    stats: dict[str, int] = {}
    collect_rollouts(
        JumpPolicy(),  # type: ignore[arg-type]
        env,
        lambda o: o,
        prompt_seeds=[0, 1],
        group_size=2,
        temperature=1.0,
        max_episode_steps=6,
        task="t",
        action_clip=clip,
        clip_stats=stats,
    )
    return env, stats


def test_default_is_off_and_matches_the_wo04_rollouts() -> None:
    for name in ("rl", "tiny"):
        assert GRPOConfig.from_config(load_config(name)).rollout_action_clip_deg is None
    env, stats = _run(None)
    assert all(torch.equal(a, torch.ones(N, 6)) for a in env.actions)  # unclipped
    assert stats == {"clipped_steps": 0, "active_steps": 0}


def test_clip_limits_each_executed_step_like_protocol_02() -> None:
    bound = math.radians(4.0)
    env, stats = _run(bound)
    prev = torch.zeros(N, 6)
    for a in env.actions:
        assert torch.allclose((a - prev).abs(), torch.full((N, 6), bound))  # 4 deg per step
        prev = a
    assert stats == {"clipped_steps": 6 * N, "active_steps": 6 * N}


def test_config_key() -> None:
    cfg = load_config("tiny")
    cfg["rl"]["rollout_action_clip_deg"] = 4
    rl = GRPOConfig.from_config(cfg)
    assert rl.rollout_action_clip_deg == 4.0
    assert rl.action_clip_rad == pytest.approx(math.radians(4.0))
    cfg["rl"]["rollout_action_clip_deg"] = None
    assert GRPOConfig.from_config(cfg).action_clip_rad is None
    cfg["rl"]["rollout_action_clip_deg"] = 0
    with pytest.raises(ValueError, match="rollout_action_clip_deg"):
        GRPOConfig.from_config(cfg)
