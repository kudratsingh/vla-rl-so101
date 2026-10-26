"""GRPO math (ADR-0002): group advantages, clip-higher policy loss, dynamic-sampling filter.

Pure functions on tensors; no model, env or optimizer in here. `tests/test_rl_grpo.py` checks each
one against hand-computed values.

Adapted from PRIME-RL/SimpleVLA-RL, commit 7c51662df27b586f9e8a1ab35fcf849f2b8852f9, MIT License
(Copyright (c) 2025 PRIME-RL). The upstream files also carry verl's Apache-2.0 header
(Copyright 2024 Bytedance Ltd. and/or its affiliates; core_algos.py: Copyright 2022 The
HuggingFace Team). Only the math was taken; no ray / FSDP / DataProto.

  group_advantages             <- verl/trainer/ppo/core_algos.py  compute_grpo_outcome_advantage
  clip_higher_policy_loss      <- verl/trainer/ppo/core_algos.py  compute_policy_loss
  seq_mean_token_mean_weights  <- verl/workers/actor/dp_rob.py    RobDataParallelPPOActor.update_policy
                                  (the pg_loss * mask_sum / response_mask_sum / grad_accum scaling)
  dynamic_sampling_keep        <- verl/trainer/ppo/ray_trainer.py RayTrainer.filter (filter_accuracy)

Intentional differences from upstream, all no-ops for this project's settings unless noted:
  * Groups are a (G, K) tensor (trajectory g*K + k belongs to prompt g) instead of a uid lookup.
    Upstream treats a group of size 1 as mean 0 / std 1; we reject K < 2 (GRPO needs a baseline).
  * Dynamic sampling drops groups whose rewards are all equal (max == min). Upstream keeps groups
    with mean success in [0.1, 0.9] (run_openvla_oft_rl_libero.sh). For binary rewards and
    K <= 10 these are the same set; for K > 10 upstream would also drop 1-success groups.
  * The loss is a weighted token sum so the update can be micro-batched arbitrarily with an
    identical gradient; the weights reproduce upstream's aggregation exactly (see below).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ADV_EPS = 1e-6  # upstream compute_grpo_outcome_advantage(epsilon=1e-6)


def group_advantages(rewards: Tensor, eps: float = ADV_EPS) -> Tensor:
    """GRPO outcome advantage: A[g, k] = (r[g, k] - mean_g) / (std_g + eps).

    rewards: (G, K) scalar episode returns (binary success here), K samples of the same prompt.
    std is the *unbiased* (Bessel, divide by K-1) std, as upstream's `torch.std`.

    All-equal group (all 0 or all 1): std = 0, so A = 0 / (0 + eps) = 0 exactly. Such a group
    gives zero policy gradient (the surrogate is -A * ratio), but it still counts in the
    per-mini-batch trajectory mean, i.e. it dilutes the step. Dynamic sampling removes it.
    Binary group with p = mean success: A_success = (1 - p) / std, A_fail = -p / std,
    std = sqrt(K / (K - 1) * p * (1 - p)).
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError(f"rewards must be (G, K) with K >= 2, got {tuple(rewards.shape)}")
    rewards = rewards.float()
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)  # unbiased, matches upstream
    return (rewards - mean) / (std + eps)


def dynamic_sampling_keep(rewards: Tensor) -> Tensor:
    """(G, K) rewards -> (G,) bool: keep groups whose rewards are not all equal (DAPO-style)."""
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be (G, K), got {tuple(rewards.shape)}")
    return rewards.amax(dim=1) != rewards.amin(dim=1)


def seq_mean_token_mean_weights(token_mask: Tensor) -> Tensor:
    """Per-token loss weights reproducing upstream's aggregation over one PPO mini-batch.

    token_mask: (M, ...) bool, M trajectories of one mini-batch; True = token of an action the env
    actually executed while the episode was running (dp_rob.py: `steps < finish_step`).

    Upstream (dp_rob.py update_policy, ppo_micro_batch_size == 1) computes, per trajectory i, the
    masked token-mean of the per-token PPO loss, and divides by gradient_accumulation = M:

        L = (1 / M) * sum_i [ (1 / n_i) * sum_{t valid in i} loss_{i,t} ]       (seq-mean-token-mean)

    i.e. each trajectory weighs the same regardless of its length. This is NOT DAPO's token-mean
    (sum over all tokens / total tokens). Returned weights w_{i,t} = mask_{i,t} / (n_i * M), so
    L = sum(w * loss) and the weights of a mini-batch sum to 1.
    """
    if token_mask.ndim < 2:
        raise ValueError("token_mask must be (M, ...) with a trajectory dim first")
    flat = token_mask.reshape(token_mask.shape[0], -1).float()
    counts = flat.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("every trajectory needs >= 1 valid token (an executed action)")
    weights = flat / (counts * flat.shape[0])
    return weights.reshape(token_mask.shape)


@dataclass(frozen=True)
class PolicyLoss:
    """`loss` carries the gradient; the rest are detached weighted sums (sum(w * x)).

    Because a mini-batch's weights sum to 1, summing these over its micro-batches gives the
    mini-batch's weighted means: clip_frac and approx_kl exactly as upstream logs them.
    """

    loss: Tensor
    clip_frac: Tensor
    approx_kl: Tensor


def clip_higher_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    weights: Tensor,
    clip_low: float,
    clip_high: float,
) -> PolicyLoss:
    """PPO clipped surrogate with decoupled clip range [1 - clip_low, 1 + clip_high] (DAPO).

    All inputs share one shape (per token). Per token, with ratio = exp(logp - old_logp):
        pg1 = -A * ratio
        pg2 = -A * clamp(ratio, 1 - clip_low, 1 + clip_high)
        loss_t = max(pg1, pg2)       # pessimistic bound; gradient is 0 where pg2 > pg1
    so for A > 0 the ratio can rise only to 1 + clip_high (0.28 > 0.2: "clip-higher" leaves
    room to raise low-probability tokens), and for A < 0 fall only to 1 - clip_low. No dual-clip,
    no KL term, no entropy bonus (all absent upstream with kl_coef=0, entropy_coeff=0).

    clip_frac counts tokens where the clipped branch is active (pg2 > pg1), as upstream.
    approx_kl is upstream's `ppo_kl` = mean(old_logp - logp); a diagnostic only.
    Tokens with weight 0 (padding) are zeroed with `where`, so garbage there cannot leak NaNs.
    """
    if not (log_probs.shape == old_log_probs.shape == advantages.shape == weights.shape):
        raise ValueError(
            "shape mismatch: "
            f"{tuple(log_probs.shape)} {tuple(old_log_probs.shape)} "
            f"{tuple(advantages.shape)} {tuple(weights.shape)}"
        )
    valid = weights > 0
    neg_log_ratio = old_log_probs - log_probs
    log_ratio = torch.where(valid, -neg_log_ratio, torch.zeros_like(log_probs))
    ratio = torch.exp(log_ratio)
    pg1 = -advantages * ratio
    pg2 = -advantages * torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    per_token = torch.maximum(pg1, pg2)
    loss = (torch.where(valid, per_token, torch.zeros_like(per_token)) * weights).sum()
    with torch.no_grad():
        clipped = (pg2 > pg1) & valid
        clip_frac = (clipped.float() * weights).sum()
        approx_kl = (torch.where(valid, neg_log_ratio, torch.zeros_like(log_probs)) * weights).sum()
    return PolicyLoss(loss=loss, clip_frac=clip_frac.detach(), approx_kl=approx_kl.detach())
