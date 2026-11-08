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

A confirming A100 probe with these settings runs before any seed; its numbers are below. The
seeds wait for the owner's PR #11 review.

### Confirming probe (2026-10-27, A100-SXM4-80GB, pod gzxhwhjhdtc1kc, US-KS-2, ~$1.5)

Settings above with K = 8 × 16 prompts, seed 0, 3 GRPO steps (`wo04/seed0/gzxhwhjhdtc1kc/` on
the Hub). The host is cgroup v1 (256 visible cores, 27.2-CPU quota); the job ran with 28 threads
(the script's cap now reads the v1 quota too).

| | value |
|---|---|
| SFT baseline, 64 greedy eps/task, horizon 120 (PickPlace / Push / Stack) | 0.125 / 0.141 / 0.016 (trained-task mean 0.133) |
| groups dropped | **72%** (was 93%) |
| s/step median (max) | 736.8 (757.4): rollout 44.5 s per 32-env batch × 12 (refill cap), update 202.7 s (incl. recompute) |
| first-minibatch \|log ratio\| max / mean | 0.1318 / 0.00012 (mean was 0.01325 without recompute) |
| VRAM | torch peak 39.3 GB allocated; device 46.8 of 85.1 GB |
| projected 150-step run | 31.0 h = $55.57 per seed; kill-switch stop at step 38 = 8.0 h = $14.23 |

Reading: temperature 1.0 lifted the surviving groups from 7% to 28%, so the update now gets real
signal (update time 45.6 → 202.7 s is that signal, not overhead), but every step still hit the
refill cap. Recompute cut the mean sampler/update disagreement ~100×; the max (0.13) is a few
tokens (bf16 batch-shape numerics), bounded by the 0.2/0.28 clip.

**K = 16 rule fired (72% > 60%)**, applied in `configs/rl.yaml`: `samples_per_prompt: 16`,
`rollout_batch: 8` (128 trajectories per update, as before; 2 groups per 32-env batch). Cost
model updated to the measured rollout rate (86 env actions/s) and the refill cap (3.0 episodes per
trained one): worst case 13.1 min/step, 150 steps = 33.0 h = **$59 per seed** (`--estimate`);
if 50–60% of K=16 groups survive, ~10–11.5 min/step = ~$46–52. The seed-0 pod's own probe (job
stage `probe`, before its approval gate) measures the K=16 drop rate before any spend beyond ~$1.5.

## Amendment 2 (2026-10-28, owner; outcomes 2026-10-30)

Seeds 1 and 2 of the two-task run stopped at the step-38 mean check (0.0 and -1.6 points); seed 0
passed by one episode (+3.1). In all three PickPlace rose 5-9 points and Push fell 2-9. An estimate
from the aggregate logs put PickPlace's share of trained groups at ~48% (at most ~63%), so it did
not dominate the updates.

Owner decisions: no per-task advantage normalization (groups already share a prompt); the kill
switch passes if ANY trained task is +3 at step 38 (`rl.early_stop_rule: any_task`); run
PickPlace-only, one seed, then multi-task with stratified batches only if it clears +10 points on
the final protocol eval (seen, 300 episodes, >= 27.0%). Added for that run: checkpoints every 25
steps, a stop at step 100 if below step 50, protocol eval of the best mid-run checkpoint.

Outcome: PickPlace seen 20.7% [16.5, 25.6], **+3.7 points, bar missed**, so stratified batches are
not run. Two-task seed 0: 22.0% [17.7, 27.0], +5.0. `docs/results.md` has both tables. The peak
stop did not fire (step 100 28.1% vs step 50 25.0%). The best checkpoint was step 150, so the job
ran no separate mid-run eval; a later protocol eval of two-task step 50 (21.3%) and PickPlace-only
step 100 (22.7%) found no peak.

Both runs logged approx_kl ~3e-5 and clip fraction 0.000. Correction (2026-11-08, review
`docs/reviews/2026-11-07-project-review.md` section 3.1): with one PPO epoch the logged value is
the mean over 2 mini-batches. Mini-batch 1 is exactly on-policy (KL 0), so the logged ~3e-5 is
half of mini-batch 2's ~6e-5. That value was measured after one Adam step on other prompts, and
nothing was measured on the trained prompts after the full update. Clip fraction is ~0 by
construction at this step size: RMS |delta log p| is about 0.013, against a clip edge of about
0.22. This is weak evidence that one step at lr 5e-5 moves the policy little, not a measure of the
per-update change. The run did take few steps: 150 updates x 2 optimizer steps at lr 5e-5 = 300
steps, against 8000 SFT steps at lr 5e-4. Small step size remains a hypothesis for the slow gain,
not a verified cause; the logged KL/clip is consistent with it but does not settle it.

## Amendment 3 (proposed 2026-11-08, PR 1 `fix/pre-run-hardening`; waits for owner review)

Applies to the next GRPO run only. No paid run has used it.

- **Shape:** 16 prompts x K=8 (`rollout_batch: 16`, `samples_per_prompt: 8`), the same 128
  trajectories per update as K=16 x 8 but twice the prompts.
- **lr 1e-4, `ppo_epochs: 2`** (4 Adam steps per update at `ppo_mini_batch: 64`). The lr follows
  the published LoRA recipes (RIPT-VLA, RLinf-VLA use 1e-4). It does not rest on the logged
  KL/clip, which is only weak evidence (above). An offline lr sweep (`scripts/wo03_rl.py --lr-sweep`)
  measures policy movement on one fixed batch before any further lr change.
- **Gates** replace the 64-episode kill switch and the 32-episode peak stop. They use windows of
  `rollout/success_rate` (per trained task) and rollout entropy (`rl.gate_*`). They decide whether
  to keep spending. Only the protocol eval decides whether the policy improved. Periodic greedy
  evals run on the trained tasks only and are sanity logs. Thresholds: `gate_min_gain` 0.03 is
  the owner's +3 bar, but it now applies to sampled rollout success, a different quantity, so it
  needs the owner's confirmation. The flat gate's gain is a per-task least-squares trend over
  steps 1-38, i.e. the change from the SFT policy (the first proposal's baseline window, steps
  1-10, had already taken 0-9 updates). Proposed with it, for the owner: `gate_flat_z` 0.5 (stop
  only if gain + 0.5 x SE < 3 points), which in simulation stops 40-73% of flat runs and 6-9% of
  runs where one task truly gains 5 points (`src/rl/gates.py`). `gate_max_drop` 0.04 and
  `gate_entropy_min` 0.5 are the agent's judgment calls and have not been measured. The decline
  gate also needs the drop to exceed
  `gate_decline_z` (3) x its group-level SE on two consecutive non-overlapping windows, so
  sampling noise does not stop an improving run (z from a CPU simulation, `src/rl/gates.py`).
- **Fixed-batch diagnostic** (`src/rl/diagnostics.py`): it scores the first 32 trajectories of
  each update batch before and after all optimizer steps. It logs |delta log p|, delta log p split
  by advantage sign, k3 KL, clip fraction and `grad_norm`. The mini-batch-averaged
  `train/approx_kl` and `train/clip_frac` are removed.
- **Resume** from optimizer, scheduler, step and RNG state (`--resume`); checkpoints every 25
  steps.
- **RL seeds >= 10 are rejected**: their training seeds (`rl_seed * 10_000_000 + k`) would land in
  the protocol eval range (100M+) (`src/eval/seed_ranges.py`).

Unverified on GPU: the gate behaviour, the diagnostic and sweep forward cost, the CUDA RNG
restore, and the throughput for the 16 x 8 / 2-epoch shape.

This meets half of the "what would reopen this" condition above: GRPO has shown no clear
improvement on a base below the 20-60% band. The decision is with the owner.
