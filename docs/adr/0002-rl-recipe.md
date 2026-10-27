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

## Amendment 1 (2026-10-27, owner, after the WO-04 probes)

Two probes (seed 0, 3 GRPO steps each, WO-02 attempt-4 `step_004000` base, then-current
`configs/rl.yaml`: K = 8, 16 prompts, temperature 1.6, sampler log-probs reused, horizon 150,
rollouts on all three tasks, 300 steps). Logs and `record.json`: Hub
`KudratSingh22/vla-rl-so101-rl`, `wo04/seed0/<pod>/`.

| | A100-SXM4-80GB (pod 5rdd45j801bk5b) | H100 80GB HBM3 (pod nz8fp6rckh1yfe) |
|---|---|---|
| s/step, median (max) | 716.4 (722.6) | 611.5 (612.7) |
| rollout per 32-env batch / update | 55.9 s / 45.6 s | 48.9 s / 24.8 s |
| greedy eval per batch | 52.7 s | 46.7 s |
| SFT baseline, 64 greedy eps/task (PickPlace / Push / Stack) | 0.109 / 0.141 / 0.000 | 0.109 / 0.125 / 0.000 |
| groups dropped (all-equal reward) | 93% | 92% |
| first-minibatch \|log ratio\| max / mean | 0.1958 / 0.01325 | 0.1488 / 0.01309 |
| VRAM: torch peak allocated; device in use | 39.2 GB; 46.7 of 85.0 GB | 39.1 GB; 46.5 of 85.1 GB |
| projected 300-step run | 60.4 h = $108.04 @ $1.79/h | 51.5 h = $205.64 @ $3.99/h |
| kill-switch stop (step 75) | 15.2 h = $27.19 | 13.0 h = $51.76 |

Reading: (1) with 92–93% of groups all-equal, every step hit the dynamic-sampling refill cap, so
rollout time dominated and almost no signal reached the update; (2) the sampler-vs-update log
ratio is ~3–4× the 0.05 bar, so reusing sampler log-probs is not safe; (3) Stack is 0/64, so its
groups are always all-zero; (4) the A100 is under the owner's "H100 only if within ~10% of the
A100 $" rule ($108 vs $206 per seed).

Owner decisions (2026-10-27), now in `configs/rl.yaml`:

- **Temperature 1.0** (was ≈1.6): more groups with mixed outcomes at a ~10–14% base.
- **RL on PickPlace + Push; Stack eval-only** (`rl.train_tasks`). Evals still cover all three
  tasks; the kill switch compares the trained-task mean (`eval/success_train`) — the agent's
  reading of "Stack eval-only", flagged for the owner in PR #11.
- **`recompute_old_log_probs: true`** (measured log ratio above).
- **RL horizon 120** (`rl.max_episode_steps`) for training rollouts and in-training evals; the
  protocol eval (`scripts/wo02_eval.py`, `docs/eval_protocol.md`) keeps 150.
- **150 steps; kill switch at step 38** (ceil(0.25 × 150); 64 greedy episodes/task, stop if the
  trained-task mean rose < 3 points over step 0).
- **K = 16 rule:** if the confirming probe still drops > 60% of groups, raise
  `samples_per_prompt` to 16 before changing anything else. (With 32 envs per batch that is 2
  groups per batch; keeping 128 trajectories per update means `rollout_batch` 8 — the agent's
  proposal, to be confirmed with the owner when it triggers.)
- **GPU:** A100-80GB.

A confirming A100 probe with these settings runs before any seed; its numbers are appended
below. The seeds wait for the owner's PR #11 review.
