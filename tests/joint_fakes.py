"""Test doubles for the RL action clip: a 6-joint env whose policy-obs state is the joint reading
(as src.env.policy_obs gives, so protocol 0.2's clip applies) and a small stochastic RLPolicy
that jumps far more than 4 deg per step. No ManiSkill, no renderer: they run in CI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.rl.policy_api import PolicyObs, RolloutStep, TokenLogProbs

JOINTS = 6
GOAL_RAD = 0.15  # success: joint 0 past this (>= 3 clipped steps of 4 deg, 1 unclipped jump)


class JointEnv:
    """Batched 6-joint env: the joints go exactly to the commanded target; reset pose 0 for every
    seed (so groups share their scene); success once joint 0 > `goal_rad` (inf: never). Records every action.
    """

    def __init__(self, num_envs: int, max_steps: int = 8, goal_rad: float = GOAL_RAD) -> None:
        self.n, self.max_steps, self.goal = num_envs, max_steps, goal_rad
        self.q = torch.zeros(num_envs, JOINTS)
        self.t = 0
        self.actions: list[Tensor] = []
        self.resets: list[list[int]] = []

    @property
    def unwrapped(self) -> Any:
        return self

    def get_instructions(self) -> list[str]:
        return ["reach"] * self.n

    def _obs(self) -> dict[str, Tensor]:
        return {"image": torch.zeros(self.n, 2, 2, 3, dtype=torch.uint8), "state": self.q.clone()}

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]:
        self.resets.append([int(s) for s in seed] if seed is not None else [])
        self.q = torch.zeros(self.n, JOINTS)
        self.t = 0
        return self._obs(), {}

    def step(self, action: Tensor) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self.actions.append(action.detach().clone())
        self.q = action.detach().clone().float()
        self.t += 1
        return (
            self._obs(),
            torch.zeros(self.n),
            torch.zeros(self.n, dtype=torch.bool),
            torch.full((self.n,), self.t >= self.max_steps),
            {"success": self.q[:, 0] > self.goal},
        )

    def close(self) -> None:
        pass


def joint_obs(obs: Any) -> dict[str, Tensor]:
    return {"image": obs["image"], "state": obs["state"]}


class JointPolicy(nn.Module):
    """Linear state -> logits over N_BINS joint targets in [-0.5, 0.5] rad, C actions per chunk.
    Records every sample() output and every token tensor log_prob() is asked to score."""

    N_BINS = 5

    def __init__(self, chunk_len: int = 2) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.net = nn.Linear(JOINTS, chunk_len * JOINTS * self.N_BINS)
        with torch.no_grad():
            self.net.weight.mul_(0.1)
            self.net.bias.zero_()
        self.register_buffer("centres", torch.linspace(-0.5, 0.5, self.N_BINS))
        self.samples: list[RolloutStep] = []
        self.scored: list[Tensor] = []

    def _logits(self, obs: PolicyObs) -> Tensor:
        state = obs["state"].float()
        return self.net(state).view(state.shape[0], self.chunk_len, JOINTS, self.N_BINS)

    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> RolloutStep:
        logits = self._logits(obs)
        if temperature > 0:
            logp_all = F.log_softmax(logits / temperature, dim=-1)
            tokens = torch.multinomial(logp_all.reshape(-1, self.N_BINS).exp(), 1)
            tokens = tokens.view(logits.shape[:-1])
        else:
            logp_all = F.log_softmax(logits, dim=-1)
            tokens = logits.argmax(dim=-1)
        log_probs = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        step = RolloutStep(actions=self.centres[tokens], tokens=tokens, log_probs=log_probs)
        self.samples.append(
            RolloutStep(step.actions.clone(), step.tokens.clone(), step.log_probs.detach().clone())
        )
        return step

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: Tensor, temperature: float
    ) -> TokenLogProbs:
        self.scored.append(tokens.detach().clone())
        logp_all = F.log_softmax(self._logits(obs) / temperature, dim=-1)
        log_probs = logp_all.gather(-1, tokens.long().unsqueeze(-1)).squeeze(-1)
        return TokenLogProbs(log_probs, -(logp_all.exp() * logp_all).sum(-1))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "policy.pt")
