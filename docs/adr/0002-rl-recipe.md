# ADR-0002: RL recipe

Status: accepted
Date: 2026-10-26
Work order: WO-03

## Decision

GRPO with a binary environment success reward, as in SimpleVLA-RL: 8 samples per prompt,
group-normalized advantage, clip-higher (ε_low 0.2, ε_high 0.28), rollout temperature ≈1.6, no KL
penalty, dynamic sampling that drops all-equal-reward groups. LoRA parameters only. Single GPU:
rollout and update phases alternate on the same card; parallel env count set by VRAM budget.

## Why

- No critic → lowest memory, which is the binding constraint on one card.
- The recipe has published +40–50 point gains from a few-shot SFT start; we are reproducing, not
  inventing.
- Binary env reward cannot be shaped-hacked; success-check bugs are fixed in the check.

## Alternatives considered

- PPO with a learned value head (VLA-RL, RLinf PPO) — extra memory, more hyperparameters; only if
  GRPO stalls.
- Learned process reward model — out of scope by rule 5.

## What would reopen this

GRPO shows no improvement after 2 full runs on a base with 20–60% success; or rollout throughput
makes a run >12 h on an A100 (then consider RLinf's pipelined mode or an H100).
