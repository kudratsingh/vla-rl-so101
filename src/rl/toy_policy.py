"""2-layer MLP stand-in for the VLA, with LoRA via peft, implementing `RLPolicy` (tiny preset).

It mirrors the OpenVLA-OFT action head shape the RL loop sees: one decision = an action chunk of
C steps x A dims, each a token over n_bins bins; actions are the bin centres. The frozen base MLP
is randomly initialised (the stand-in for a pre-trained model); only the LoRA A/B matrices on both
linear layers train, so the trainer's LoRA-only path is exercised. LoRA dropout is 0 (see the
determinism requirement in policy_api.py). The instruction is accepted and ignored.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn

from src.config import Config
from src.rl.policy_api import PolicyObs, RolloutStep, TokenLogProbs
from src.rl.toy_env import ACTION_DIM, STATE_DIM


class ToyPolicy(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        chunk_len: int,
        n_bins: int,
        lora_rank: int,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.chunk_len = chunk_len
        self.action_dim = ACTION_DIM
        self.n_bins = n_bins
        base = nn.Sequential(
            nn.Linear(STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk_len * ACTION_DIM * n_bins),
        )
        lora = LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, target_modules=["0", "2"]
        )
        self.net = cast(nn.Module, get_peft_model(base, lora))
        # bin i of n covers [-1 + 2i/n, -1 + 2(i+1)/n]; its centre is the executed action value
        centres = -1.0 + (2.0 * torch.arange(n_bins, dtype=torch.float32) + 1.0) / n_bins
        self.register_buffer("bin_centres", centres)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return cast(Tensor, self.bin_centres).device

    def _logits(self, obs: PolicyObs) -> Tensor:
        state = obs["state"].to(self.device, torch.float32)
        out = cast(Tensor, self.net(state))
        return out.view(state.shape[0], self.chunk_len, self.action_dim, self.n_bins)

    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> RolloutStep:
        logits = self._logits(obs)
        if temperature > 0:
            logp_all = F.log_softmax(logits / temperature, dim=-1)
            flat = logp_all.reshape(-1, self.n_bins).exp()
            tokens = torch.multinomial(flat, 1).view(logits.shape[:-1])
        else:
            logp_all = F.log_softmax(logits, dim=-1)
            tokens = logits.argmax(dim=-1)
        log_probs = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        actions = cast(Tensor, self.bin_centres)[tokens]
        return RolloutStep(actions=actions, tokens=tokens, log_probs=log_probs)

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: Tensor, temperature: float
    ) -> TokenLogProbs:
        if temperature <= 0:
            raise ValueError("log_prob needs the (positive) rollout temperature")
        logp_all = F.log_softmax(self._logits(obs) / temperature, dim=-1)
        tokens = tokens.to(self.device, torch.long)
        log_probs = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(dim=-1)
        return TokenLogProbs(log_probs=log_probs, entropy=entropy)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.net.save_pretrained(str(path))  # type: ignore[operator]


def build_toy_policy(cfg: Config, device: str) -> ToyPolicy:
    toy = cfg["toy"]
    return ToyPolicy(
        hidden=int(toy["hidden"]),
        chunk_len=int(toy["chunk_len"]),
        n_bins=int(toy["n_bins"]),
        lora_rank=int(cfg["policy"]["lora_rank"]),
        device=device,
    )
