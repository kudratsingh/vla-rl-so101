"""What the GRPO loop needs from a policy. WO-02's VLA wrapper implements `RLPolicy` structurally.

Action representation (OpenVLA-OFT under SimpleVLA-RL's head, @7c51662): one policy *decision*
(query) looks at one observation and decodes a whole action chunk in ONE forward pass (parallel
decoding): C=8 env steps x A=6 dims = 48 discrete tokens, each one of n_bins=256 bins over the
normalised [-1, 1] range (vocab ids 31744..31999; only the 256-wide logit slice [..., -320:-64]
is used). The RL loss and the clip-higher ratio are per token. The loop sees bin indices:

    tokens, log_probs, entropy : (B, C, A)   B = decisions in the batch; token (c, a) is OFT's
                                             flat position c * A + a (chunk-major, as dp_rob.py
                                             reshapes), i.e. OFT's (B, C*A) viewed as (B, C, A)
    actions                    : (B, C, A) float32, env units (absolute joint targets, radians)

The rollout stores the raw policy observation + instruction of every decision (CPU) and the
trainer calls `log_prob` on them later; one decision executes up to C env steps (fewer at
episode end, the rest is masked). The wrapper re-runs its own preprocessing each call.

Contract for implementers (each violation is a silent GRPO bug, so the trainer measures them):
  * `log_prob` must score tokens with the same tempered distribution `sample` drew them from:
    log_softmax(logits[..., 256 action bins] / temperature), same token->bin mapping. Upstream
    divides by the rollout temperature in both places (dp_rob.py `logits.div(temperature)`).
    Measured by `train/first_minibatch_max_abs_log_ratio` (old vs new log-prob before the first
    optimizer step: must be ~0) and, with recompute on, `train/logprob_mismatch`.
  * Preprocessing must be deterministic (center crop yes, random-crop augmentation no) and
    `log_prob` deterministic in both train() and eval() mode: LoRA dropout 0, no other dropout.
  * A batch may mix instructions of different token lengths (several tasks); the wrapper pads
    (upstream asserts one shared prompt length per micro-batch, so pad or bucket inside).
  * temperature == 0 means greedy (argmax) sampling, used for eval; log_probs are then of the
    untempered distribution and are not used.
  * Only LoRA (adapter) parameters may require grad (ADR-0002). `save` writes the adapter;
    loading a checkpoint is the builder's job (policy.checkpoint), not the loop's.
  * Inputs may arrive on any device (rollout buffers live on CPU); the policy moves them to its
    own device. Outputs are on the policy's device.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol, cast, runtime_checkable

from torch import Tensor, nn

from src.config import Config

# {"image": uint8 (B, H, W, 3), "state": float32 (B, 6)} -- the shape of src.env.policy_obs(obs).
PolicyObs = Mapping[str, Tensor]


@dataclass(frozen=True)
class RolloutStep:
    """One decision for a batch of B observations."""

    actions: Tensor  # (B, C, A) float32, env units; executed one env step at a time
    tokens: Tensor  # (B, C, A) int64 in [0, n_bins)
    log_probs: Tensor  # (B, C, A) float32, log pi_T(token | obs, instruction) at sampling temp


class TokenLogProbs(NamedTuple):
    log_probs: Tensor  # (B, C, A), differentiable w.r.t. trainable params
    entropy: Tensor  # (B, C, A), entropy of the tempered per-token distribution (diagnostic)


@runtime_checkable
class RLPolicy(Protocol):
    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> RolloutStep: ...

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: Tensor, temperature: float
    ) -> TokenLogProbs: ...

    def trainable_parameters(self) -> list[nn.Parameter]: ...

    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def save(self, path: Path) -> None:
        """Write the trainable (adapter) weights to directory `path`."""
        ...


# policy.base -> "module:builder". Builders are imported lazily so `tiny` needs no VLA deps.
# Builder signature: (cfg: Config, device: str) -> RLPolicy.
POLICY_BUILDERS: dict[str, str] = {
    "toy": "src.rl.toy_policy:build_toy_policy",
    "openvla_oft": "src.policy.openvla_oft:build_rl_policy",  # WO-02
    "pi05": "src.policy.pi05:build_rl_policy",  # ADR-0001 fallback
}


def build_policy(cfg: Config, device: str) -> RLPolicy:
    base = str(cfg["policy"]["base"])
    if base not in POLICY_BUILDERS:
        raise KeyError(
            f"no RL policy builder for policy.base={base!r}; known: {list(POLICY_BUILDERS)}"
        )
    module_name, fn_name = POLICY_BUILDERS[base].split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"policy.base={base!r} needs `{module_name}.{fn_name}(cfg, device) -> RLPolicy` "
            "(see src/rl/policy_api.py); it is not importable here"
        ) from exc
    builder = cast(Callable[[Config, str], RLPolicy], getattr(module, fn_name))
    policy = builder(cfg, device)
    if not isinstance(policy, RLPolicy):
        raise TypeError(f"{module_name}.{fn_name} returned {type(policy)!r}, not an RLPolicy")
    return policy
