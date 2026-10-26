"""The policy interface shared by SFT (WO-02), eval (WO-02) and GRPO (WO-03/04).

`VLAPolicy` is a superset of WO-03's `src.rl.policy_api.RLPolicy`, implemented *structurally*
(no import of src.rl): `sample` returns an object with `.actions/.tokens/.log_probs` (B, C, A),
`log_prob` returns `(log_probs, entropy)` (B, C, A), plus `trainable_parameters/train/eval/save`.
Token (c, a) is OFT's flat action slot c * A + a (chunk-major), as a bin index in [0, 256).

Observations follow the env contract (src/env `policy_obs`): {"image": uint8 (B, H, W, 3),
"state": float32 (B, 6) joint positions, rad}. Actions are absolute joint targets (rad).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from torch import Tensor, nn

PolicyObs = Mapping[str, Tensor]


@dataclass(frozen=True)
class ActionSample:
    """Same fields as src.rl.policy_api.RolloutStep."""

    actions: Tensor  # (B, C, A) float32, env units
    tokens: Tensor  # (B, C, A) int64 in [0, n_bins)
    log_probs: Tensor  # (B, C, A) float32 at the sampling temperature


class TokenScores(NamedTuple):
    """Same fields as src.rl.policy_api.TokenLogProbs."""

    log_probs: Tensor  # (B, C, A), differentiable w.r.t. trainable params
    entropy: Tensor  # (B, C, A)


@dataclass(frozen=True)
class SFTBatch:
    images: Tensor  # (B, 3, H, W) float in [0, 1]  or  (B, H, W, 3) uint8
    states: Tensor  # (B, 6) rad
    actions: Tensor  # (B, C, 6) rad, the chunk starting at this frame (last action repeated)
    instructions: list[str]


@runtime_checkable
class VLAPolicy(Protocol):
    chunk_len: int
    action_dim: int

    # --- deployment / eval ------------------------------------------------------------------
    def act(self, obs: Mapping[str, np.ndarray | Tensor], instruction: str) -> np.ndarray:
        """One observation (image (H, W, 3) uint8, state (6,)) -> greedy chunk (C, 6), rad."""
        ...

    def act_batch(self, obs: PolicyObs, instructions: Sequence[str]) -> Tensor:
        """Greedy (B, C, 6) chunks, rad."""
        ...

    # --- SFT ---------------------------------------------------------------------------------
    def sft_loss(self, batch: SFTBatch) -> tuple[Tensor, dict[str, float]]: ...

    # --- RL (src.rl.policy_api.RLPolicy) -----------------------------------------------------
    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> ActionSample: ...

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: Tensor, temperature: float
    ) -> TokenScores: ...

    def trainable_parameters(self) -> list[nn.Parameter]: ...

    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def save(self, path: Path) -> None: ...
