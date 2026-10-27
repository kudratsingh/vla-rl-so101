# vla-rl-so101 — agent brief

Single-GPU reinforcement-learning post-training of a pre-trained vision-language-action (VLA)
model, trained in ManiSkill3 simulation and deployed to a real SO-101 arm. The owner is Kudrat
Singh. You (the coding agent) write all the code. The owner launches every paid run.

Read in this order before any work: this file → `docs/PLAN.md` → `docs/ENVIRONMENT.md` →
the current work order in `docs/work-orders/`.

## Goal in one sentence

Reproduce the published few-shot VLA-RL result (≤10 demos + GRPO ≈ full-data SFT; e.g. LIBERO
48.9% → 96.9% in SimpleVLA-RL) on ONE GPU, then show the same recipe improves real success on a
$300 arm — with SFT-only vs SFT+RL tables on seen positions, shifted positions and unseen objects.

## Operating mode: fast

- Build order is the work orders, WO-01 → WO-09. Do not gold-plate.
- One work order = one branch = one PR. A WO whose inputs don't depend on an unfinished WO may run
  in parallel (owner, 2026-10-26: e.g. WO-03 on the toy env, WO-06 robot-side code). A WO is done
  only when its completion benchmark (below) is met and recorded in `docs/STATUS.md`.
- **No independent review pass by default.** Self-check against the WO's benchmark, run the test
  suite, open the PR. The owner reviews only WO-03 and WO-04 (the RL loop) because a silent bug
  there burns GPU dollars.
- Minimal tests: one smoke test per module that proves it runs end to end on CPU with tiny
  settings. No test-count targets, no coverage targets.
- Prefer vendoring: copy the environment and RL code you need into `src/` with a header comment
  naming the source repo, commit and license. Do not maintain forks of upstream repos.
- When blocked, write the blocker and your best-guess resolution into `docs/STATUS.md` and
  continue with the parts that are not blocked. Ask the owner only for decisions that spend money
  or cannot be undone.

## Hard rules

1. **No paid runs without the owner's explicit, per-run approval.** You produce the config, a CPU
   dry run on the `tiny` preset, and a cost estimate (GPU type × hours × RunPod price). On the
   owner's explicit "go" for that run, the agent launches it via the RunPod connector and stops /
   deletes the pod when it ends (owner, 2026-10-26). Every paid run starts with a short probe;
   the full run waits behind an approval gate in the pod job script until the owner gives an
   explicit go on the posted probe numbers (step time, projected hours and $). Every run gets a run record (`runs/<id>/record.json`: config hash, git commit, GPU,
   start/end, dollars).
2. **Pin everything.** Versions in `docs/ENVIRONMENT.md` and `pyproject.toml` are frozen. Upgrading
   any pinned dependency is its own work order with a stated reason.
3. **Few-shot regime.** SFT uses ≤25 demos per task (amended from ≤10 by the owner on 2026-10-27
   after three ≤10-demo OpenVLA-OFT SFT attempts stayed below 20% — still few-shot; ADR-0001). RL only starts from a checkpoint with
   nonzero success (target 20–60% on seen positions). RL cannot rescue a 0%-success base; if SFT
   is at 0%, fix SFT, do not launch RL.
4. **Eval protocol is fixed before training.** `src/eval/protocol.py` defines: tasks, 100 episodes
   per cell in sim, 50 trials per cell on the real arm, cells = {seen positions, shifted positions,
   unseen objects}, 3 seeds in sim. Every reported number cites its protocol hash. Never change
   the protocol to make a number look better; add a new protocol version instead.
5. **Reward is the environment's success check only.** No shaped rewards, no learned reward
   models in this project. If a success check can be gamed (gripper closed without the object,
   object knocked into the goal), fix the check, not the policy.
6. **Decode video with torchcodec, never PyAV.** PyAV fallback drops GPU utilization to 3–30%.
7. **Secrets and weights stay out of git.** `.env`, `runs/`, `checkpoints/`, `data/` are ignored.
   Datasets go to the Hugging Face Hub under the owner's account; checkpoints to Hub or RunPod
   volume.
8. **Real-arm safety.** Joint limits and workspace bounds enforced in `src/robot/safety.py` before
   any policy action reaches the bus. Policy actions clipped per joint; max velocity capped.
   Deadman: the deploy loop stops on any lost camera frame or serial error.
9. **Honest reporting.** Single-run numbers are labelled single-run. Sim numbers are never
   presented as real numbers. Negative results go in `docs/results.md` too.
10. **Mac is for control, cloud is for training.** The owner's machines are Apple Silicon Macs
    (zsh, Homebrew Python). Training, ManiSkill3 rendering and VLA inference run on a rented
    NVIDIA GPU. Commands for the owner are written for macOS; commands for the GPU box for Linux.

## Build order and completion benchmarks

| WO | Name | Done when |
|---|---|---|
| 01 | Sim env + oracle + demos | One command builds the ManiSkill3 SO-101 scene with 3 tasks; scripted oracle ≥95% on each; 10 demos/task saved as LeRobotDataset; CPU smoke passes |
| 02 | SFT + eval harness | OpenVLA-OFT (or π0.5 LoRA) few-shot SFT reaches 20–60% on seen positions; eval harness emits the full table (3 cells × 3 seeds) from one command |
| 03 | RL loop, dry | GRPO loop runs end to end on CPU `tiny` preset; rollout → reward → advantage → update verified on a synthetic env where the optimal policy is known |
| 04 | RL, real sim | SFT+RL beats SFT by ≥20 points on seen positions on at least 2 of 3 tasks, 3 seeds; shifted-position gain reported; $/run logged |
| 05 | Ablations + writeup v1 | Demo-count (1/5/10) and samples-per-prompt ablations; unseen-object eval; public README + blog draft "single-GPU VLA RL" |
| 06 | Hardware bring-up | Arm calibrated; teleop works; real camera frames match sim camera (same intrinsics, pose, crop); randomization retrain if needed |
| 07 | Deploy + real eval | Policy served over gRPC; 50 real trials per cell, SFT vs SFT+RL; 10-trial unscripted video |
| 08 | Close the gap | Diagnose WO-07 failures (latency, lighting, actuator); retrain; real success improves on ≥2/3 tasks |
| 09 | Final writeup | README comparison table, model card, reproduce commands, ADR index, results.md with every number and its protocol hash |

A sim-only milestone (WO-01–05) is shippable on its own if hardware is late.

## Repository layout

```
CLAUDE.md                 this file
docs/PLAN.md              phases, scope, non-goals
docs/ENVIRONMENT.md       pinned versions, GPU tiers, install
docs/STATUS.md            running log: current WO, blockers, decisions, spend
docs/results.md           every number with run id + protocol hash
docs/adr/                 decisions (template in 0000-template.md)
docs/work-orders/         WO-01 … WO-09
configs/                  tiny.yaml (CPU), sim.yaml, rl.yaml, real.yaml
src/env/                  ManiSkill3 scene, tasks, success checks, randomization
src/data/                 oracle, demo recording, LeRobotDataset I/O
src/policy/               VLA wrappers (SmolVLA, OpenVLA-OFT, π0.5), LoRA, action heads
src/rl/                   GRPO loop, rollout workers, advantage, update
src/eval/                 protocol.py, sim eval, real eval, report
src/robot/                SO-101 driver glue, calibration, safety, gRPC serving
scripts/                  one entry point per WO step
runs/                     run records (ignored except record.json)
tests/                    smoke tests only
```

## Conventions

- Python 3.11, `uv` for deps, `ruff` + `black`, strict `mypy` on `src/`.
- Conventional Commits. PR template: what, benchmark result, cost, what's next.
- Every script takes `--config` and `--preset {tiny,sim,rl,real}`; `tiny` must run on CPU in <2 min.
- Logs: JSONL to `runs/<id>/metrics.jsonl`; W&B optional via `WANDB_MODE`.
- Model names come from config (`policy.base`), never hardcoded.
- Cost line in every PR that touches training: `estimated: <GPU> × <h> = $<x>`.

## Owner's standing notes to the agent

- Speed over polish. Fast iteration and completion is the priority; the owner studies the project
  on the side.
- Do not add process (extra reviews, extra docs, extra tests) that this file does not call for.
- If a WO looks like it will exceed one week of agent effort, split it and say so in STATUS.md.
- Re-price recurring costs when the project changes shape (e.g. if rollouts dominate, say so with
  numbers and options before the next paid run).
