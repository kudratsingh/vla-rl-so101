# Plan

## What we are building

A post-training pipeline that takes a pre-trained VLA, fine-tunes it on ≤10 demonstrations per
task (SFT), then improves it with reinforcement learning against a binary success reward in a
GPU-parallel simulator (GRPO), on one GPU — and then deploys the result to a real SO-101 arm and
measures whether the RL gain survives contact with reality.

## Why this is worth doing

- Published VLA-RL recipes (SimpleVLA-RL, πRL/RLinf, RLinf-VLA) report +40–50 point gains from a
  few-shot SFT start, but all of them trained on 8×80 GB GPUs. No single-GPU reproduction with
  numbers exists.
- Only one of those papers reports real-robot results (SimpleVLA-RL: 17.5% → 38.5% real, zero real
  demos). Real-world validation of VLA-RL on cheap hardware is rare.
- The owner's direction is physical AI; this project is the post-training + sim-to-real core of it.

## Phases

| Phase | WOs | Output |
|---|---|---|
| A. Sim pipeline | 01–03 | Env, demos, SFT checkpoint with nonzero success, RL loop verified on CPU |
| B. RL result | 04–05 | 3-seed SFT vs SFT+RL table in sim; ablations; public writeup v1 |
| C. Real | 06–08 | Arm bring-up, real eval tables, gap-closing retrain |
| D. Release | 09 | README, model card, reproducibility |

Phase B is a complete, shippable result on its own.

## Tasks (sim and real must match)

1. `PickPlace` — pick the named object, place in the bowl.
2. `Push` — push the cube to the marked target.
3. `Stack` — stack cube A on cube B.

Each task has a programmatic success check in `src/env/success.py`. Language instruction is a
template per task with object names filled in.

## Base model decision (ADR-0001)

Default: **OpenVLA-OFT** with LoRA — it is what the open RL recipes target and its action-chunk
inference (≈108 Hz on LIBERO) keeps rollouts cheap.
Fallback: **π0.5** via openpi LoRA (>22.5 GB VRAM) through RLinf's πRL path if OpenVLA-OFT's
SFT lands below 20% on seen positions after WO-02.
Smoke-test model: **SmolVLA** (450M) in WO-01 only, to prove the data path.

Decide at the end of WO-02; record in the ADR.

## RL recipe (ADR-0002)

GRPO as in SimpleVLA-RL: 8 samples per prompt, binary outcome reward from the env success
check, group-normalized advantages, clip-higher (ε_low 0.2, ε_high 0.28), rollout temperature
≈1.6, no KL penalty, dynamic sampling (drop groups whose rewards are all-equal). Single GPU:
rollouts and updates alternate on the same card; LoRA parameters only; rollout batch sized to
VRAM. Start from the WO-02 SFT checkpoint.

## Evaluation protocol (frozen in `src/eval/protocol.py`)

- Cells: seen positions · shifted positions (objects re-sampled outside the SFT demo range) ·
  unseen objects (same task, new object meshes/colours).
- Sim: 100 episodes per cell per task, 3 seeds. Real: 50 trials per cell per task, one physical
  setup, randomized object placement from a printed grid.
- Report success rate with a 95% Wilson interval, SFT-only vs SFT+RL side by side.
- Protocol hash = SHA-256 of the protocol file; every number cites it.

## Non-goals

- No learned reward models, no shaped rewards, no human-in-the-loop RL in this project.
- No serving beyond the deploy loop; no web UI.
- No new simulator; ManiSkill3 only. No Isaac Sim.
- No dexterous, deformable or bimanual tasks.
- No full fine-tuning of 7B models; LoRA only.

## Budget

Compute target ≈ $200 (RunPod, Oct 2026 list: RTX 4090 $0.89/h, A100-80GB $1.79/h,
H100 PCIe $2.89/h). WO-01–02 on a 4090; WO-04–05 on an A100-80GB. Hardware ≈ $300–600
(SO-101 follower or pair, 2 webcams, clamps). The owner approves every run; the agent estimates
cost in every training PR.

## Timeline (owner's)

Nine weeks, one WO per week, WO-06 starts when the arm arrives. Order the arm during WO-01.

## Follow-ons (not in scope now)

- Residual RL on the frozen VLA using object poses (Microsoft CoRL 2026 recipe, no code
  released) — reuses this repo's arm, sim and SFT checkpoint.
- The same brain-over-skills loop on the Microduck's command block.
