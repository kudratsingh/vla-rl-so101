"""Fixed-batch policy-movement diagnostic (review 3.1): how far did one whole update move the policy?

The trainer scores a held set of trajectories of the update batch (whole groups, the first
`rl.diag_trajectories`) with a no-grad eval-mode forward BEFORE the update and again AFTER all of
its optimizer steps (every epoch and mini-batch), against the same reference log-probs the PPO
ratio uses (`old`: recomputed, or the sampler's). This module turns those per-token log-probs into
numbers; it has no model, env or optimizer.

Why not the in-update aggregates: upstream's `ppo_kl` / `clip_frac` are measured inside the loss
before each optimizer step, so the first mini-batch of the first epoch is exactly on-policy (ratio
1) and every later one has moved by at most a few Adam steps on other prompts. Their update mean
is at most a fraction of the movement and rounds to 0 whether or not the policy learns (WO-04:
RMS |dlogp| ~0.013 vs a clip edge of ~0.22). `movement_stats` on the post-update scores is the
total movement on tokens the update trained on.

Per token, with d = logp_now - logp_ref and r = exp(d) (all means are over valid tokens):
  mean_abs_dlogp, max_abs_dlogp      |d|
  mean_dlogp_pos_adv / _neg_adv      signed d on tokens of A > 0 / A < 0 trajectories: an update
                                     tends to raise the first and lower the second, but shared
                                     structure can move both one way on a given batch
  mean_abs_dlogp_pos_adv / _neg_adv  |d| by advantage sign
  mean_adv_dlogp                     mean(A x d): the first-order surrogate gain the update
                                     maximises; > 0 for a correct update, ~0 if it did not move
  k3_kl                              (r - 1) - d >= 0: estimator of KL(ref || now) on tokens
                                     sampled from ref (Schulman's k3)
  clip_frac                          PPO-clipped tokens: A > 0 and r > 1 + clip_high, or
                                     A < 0 and r < 1 - clip_low (where the next step's gradient
                                     would be zero)
  frac_outside_clip                  r outside [1 - clip_low, 1 + clip_high], either sign of A
  argmax_flip_frac (optional)        tokens whose greedy (argmax) bin changed: what the protocol's
                                     greedy eval would see
"""

from __future__ import annotations

from torch import Tensor


def movement_stats(
    log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    valid: Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> dict[str, float]:
    """Movement of `log_probs` from `ref_log_probs` on the `valid` tokens (all one shape).

    `advantages` is per token (the trajectory's advantage broadcast). Tokens with A == 0 count in
    the overall means but in neither sign split. Empty splits are reported as 0.0.
    """
    if not (log_probs.shape == ref_log_probs.shape == advantages.shape == valid.shape):
        raise ValueError("log_probs, ref_log_probs, advantages and valid must share one shape")
    v = valid.bool()
    if not bool(v.any()):
        raise ValueError("no valid tokens to measure")
    d = (log_probs.double() - ref_log_probs.double())[v]
    adv = advantages.double()[v]
    ratio = d.exp()
    pos, neg = adv > 0, adv < 0

    def mean(x: Tensor, m: Tensor | None = None) -> float:
        x = x if m is None else x[m]
        return float(x.mean()) if x.numel() else 0.0

    clipped = (pos & (ratio > 1.0 + clip_high)) | (neg & (ratio < 1.0 - clip_low))
    outside = (ratio > 1.0 + clip_high) | (ratio < 1.0 - clip_low)
    return {
        "mean_abs_dlogp": mean(d.abs()),
        "max_abs_dlogp": float(d.abs().max()),
        "mean_dlogp_pos_adv": mean(d, pos),
        "mean_dlogp_neg_adv": mean(d, neg),
        "mean_abs_dlogp_pos_adv": mean(d.abs(), pos),
        "mean_abs_dlogp_neg_adv": mean(d.abs(), neg),
        "mean_adv_dlogp": mean(adv * d),
        "k3_kl": mean((ratio - 1.0) - d),
        "clip_frac": mean(clipped.double()),
        "frac_outside_clip": mean(outside.double()),
        "tokens": float(d.numel()),
        "tokens_pos_adv": float(pos.sum()),
        "tokens_neg_adv": float(neg.sum()),
    }


def argmax_flip_frac(greedy_before: Tensor, greedy_after: Tensor, valid: Tensor) -> float:
    """Fraction of valid tokens whose greedy bin changed between two scorings."""
    if not (greedy_before.shape == greedy_after.shape == valid.shape):
        raise ValueError("greedy token tensors and valid must share one shape")
    v = valid.bool()
    if not bool(v.any()):
        raise ValueError("no valid tokens to measure")
    return float((greedy_before != greedy_after)[v].double().mean())


def entropy_from_sampled_log_probs(log_probs: Tensor, valid: Tensor) -> tuple[float, int]:
    """(sum of -log p, token count) over valid sampled tokens.

    For tokens drawn from a distribution p, E[-log p(token)] is p's entropy, so the mean is an
    unbiased Monte-Carlo estimate of the rollout policy's per-token entropy (at the sampling
    temperature) that costs no forward pass. Returned as a sum + count so windows can pool it.
    """
    v = valid.bool()
    return float(-(log_probs.double()[v]).sum()), int(v.sum())
