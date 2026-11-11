# vla-rl-so101 — agent brief

SmolVLA (with ACT as the baseline) fine-tuned on leader-arm teleop demos for a real SO-101 arm,
plus a filtered behaviour cloning experiment in ManiSkill3 simulation. The OpenVLA-OFT / GRPO track
is deferred until after full evaluation of the real arm and the SmolVLA/ACT baseline
(owner, 2026-11-10; amendment to `docs/adr/0003-pivot-to-smolvla.md`). The owner is Kudrat
Singh. You (the coding agent) write all the code. The owner launches every paid run.

Read in this order before any work: this file → `docs/PLAN.md` → `docs/ENVIRONMENT.md` →
the current work order in `docs/work-orders/`.

## Goal in one sentence

Fine-tune SmolVLA and ACT on the same 50 real PickPlace demos (10 at each of five seen positions,
wrist + front cameras) on one 24 GB GPU, run both on the Mac (MPS), and report 50 binary
trials per cell (seen positions, shifted positions, unseen objects) for both policies under a
frozen, hashed real protocol — then test in sim whether filtered behaviour cloning improves the
SmolVLA checkpoint before trying it on the arm.

## Operating mode: fast

- Build order is the work orders, WO-06 → WO-09 (WO-01–05 are closed: the OFT sim track, ADR-0003).
  Do not gold-plate.
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
3. **Few-shot is a sim experiment; the real arm is data-first** (owner, 2026-11-09, ADR-0003).
   In sim, SFT uses ≤25 demos per task (amended from ≤10 on 2026-10-27, ADR-0001). On the real
   arm, start with 50 PickPlace demos (10 at each of five frozen seen positions, both cameras);
   go to 100 only if a failure analysis of development trials shows that more data addresses the
   observed failures. A post-training step (WO-08 filtered behaviour cloning) may start from a
   checkpoint with ≥10% success on seen positions on at least 2 of 3 tasks (owner amendment
   2026-10-27, unchanged by the pivot; the 20–60% band stays the target). Post-training cannot
   rescue a 0%-success base; if the base is at 0%, fix the base, do not launch it.
4. **Eval protocol is fixed before training.** `src/eval/protocol.py` defines: tasks, 100 episodes
   per cell in sim, 50 trials per cell on the real arm, cells = {seen positions, shifted positions,
   unseen objects}, 3 seeds in sim. Every reported number cites its protocol hash. Never change
   the protocol to make a number look better; add a new protocol version instead. The real arm
   has its own protocol version: the five seen slot ids, the fixed bowl position and the seeded
   trial schedule are frozen and hashed before the first demo; both policies run the same
   schedule; binary success is the headline (ADR-0003).
5. **Reward (and the WO-08 filter) is the environment's success check only.** No shaped rewards, no learned reward
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
10. **Mac is for control and real-arm inference, cloud is for training.** The owner's machines are
    Apple Silicon Macs (zsh, Homebrew Python). Training and GPU-sim ManiSkill3 rendering run on a
    rented NVIDIA GPU; SmolVLA and ACT inference for the real arm runs on the Mac over MPS
    (ADR-0003). Commands for the owner are written for macOS; commands for the GPU box for Linux.

## Build order and completion benchmarks

WO-01–05 (sim env, OpenVLA-OFT SFT, GRPO loop, GRPO in sim, ablations) are closed by the pivot
(ADR-0003): WO-01 and WO-03 met, WO-02 closed below its band, WO-04's benchmark unmet (recorded as a
partial result in `docs/results.md`), WO-05 not started. Their files stay in `docs/work-orders/`.

| WO | Name | Done when |
|---|---|---|
| 06 | Bring-up | Checklist steps 0–10 complete under supervision: calibration verified, per-session drift check, teleop with the leader arm, both cameras matched and exposure/white balance locked, MPS inference latency measured on the mock rig |
| 07 | Real demos + SmolVLA/ACT baseline | 50 PickPlace demos (10 per seen position, both cameras); SmolVLA and ACT fine-tuned on the same demos; 50 binary trials per cell (seen, shifted, unseen) for each; immutable export |
| 08 | Filtered BC in sim | SmolVLA fine-tuned on the 25 sim demos per task; reward-filtered BC against the sim success check; both arms from the same checkpoint, matched budget, held-out seeds; sim gain measured; transfer tested on the arm only after the sim gain is in |
| 09 | Writeup | README table, model cards, reproduce commands, ADR index, results.md with every number and protocol hash |

Push and Stack wait until the PickPlace baseline is in. The LeRobot guide reports 25 demos were
"not enough" and 50 worked; start at 50. Going to 100 is conditional on a failure analysis from
development trials showing that more data addresses the observed failures (the Cowling 2026 SmolVLA
result fell at 100). Reserve the 50-per-cell evaluation for selected checkpoints.

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
src/policy/               VLA wrappers (SmolVLA, OpenVLA-OFT), LoRA, action heads
src/rl/                   GRPO loop (OFT track; kept, unused since ADR-0003), rollout workers
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
