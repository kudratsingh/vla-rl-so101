# Status

Updated: 2026-10-26

## Current work order

Independent WOs run in parallel (owner decision 2026-10-26, CLAUDE.md "Operating mode"). Any WO
started before its predecessors merge is listed here, marked **out of order**, before work begins.

| WO | State | Branch / PR | Notes |
|---|---|---|---|
| 01 | done (benchmark met), PR open | `wo-01-sim-env` | See benchmarks. Demos on the Hub (`KudratSingh22/vla-rl-so101-demos`, private, built at commit `0dffe51`). |
| 02 | in progress, **out of order**; blocked on an owner decision | `wo-02-sft` | OpenVLA-OFT LoRA SFT, 4 attempts, all below the 20-60% band on seen positions: a1 (10 demos, lr 5e-5, 2k) best 8/4/2%; a2 (lr 5e-4, 2k) 20/6/4%; a3 (8k steps) 22/4/4%; receding-horizon recheck worse; a4 (25 demos/task, rule 3 amended to <=25) 18/16/6%. Per the owner's path: pi0.5 next, plan + cost to the owner before any spend (ADR-0001). |
| 03 | done, merged after owner review | #4 (merged) | **Out of order** (started before WO-01/02). CPU only, toy env. |
| 06 | part 1 merged, part 2 waits for the arm | #5 (merged) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. |
| 04 | prep in progress, **out of order** (started 2026-10-27 before WO-02 merged) | `wo-04-rl-sim` (stacked on `wo-02-sft`) | CPU only: owner's WO-04 settings, `--preset rl` integration smoke (tiny OFT + real env), pod job with probe gate. No paid RL run until WO-02 has an in-band checkpoint and the owner gives a go on the probe numbers. |
| 05, 07, 08, 09 | not started | | |

## Benchmarks met

| WO | Met on | Evidence (run id / PR) |
|---|---|---|
| 01 | 2026-10-26 | Oracle, seen cell, 100 episodes/task, Mac CPU: PickPlace 100%, Push 99%, Stack 100% (run `20261026T194952Z-wo01-sim-8ad7`, which also recorded the 10 demos/task: 30 episodes, 1968 frames, pushed privately to `KudratSingh22/vla-rl-so101-demos`). SmolVLA smoke, 2000 steps x batch 8 on Mac MPS, 10 seen episodes/task: PickPlace 50%, Push 20%, Stack 0% (single run; run `20261026T195329Z-wo01-sim-b326`). `tiny` preset build: 6.9 s on CPU. |
| 02 | | |
| 03 | 2026-10-26 | tiny toy GRPO, seeds 0/1/2: greedy eval 0.301/0.102/0.191 -> 0.988/1.000/0.996 of optimum 1.0 in 100 steps (>=0.95 by step 20/20/25), bit-identical reruns, ~3.5 s CPU each; runs 20261026T193248Z-wo03-tiny-s0-be5126, -193252Z-…-s1-cf3b6d, -193255Z-…-s2-d11eb5 (reruns -fe1afa/-812773/-8c72b6); branch `wo-03-rl-loop` @ 051f1e6 |
| 04 | | |
| 05 | | |
| 06 | | |
| 07 | | |
| 08 | | |
| 09 | | |

## Spend

| Run id | WO | GPU | Hours | $ | Approved by owner |
|---|---|---|---|---|---|
| pod `iqrmjox6ws0e66` (WO-02 SFT attempt 1: probe + 2000 steps + seen sweep) | 02 | RTX 4090, $0.89/h | 1.05 | 0.94 | yes, 2026-10-26 |
| pod `efe7plxr06gaqc` (WO-02 SFT attempt 2: probe + 2000 steps + seen sweep) | 02 | RTX 4090, $0.89/h | 1.30 | 1.17 | yes, 2026-10-27 (go on probe 1.547 s/step) |
| pod `l3485bxmdj0fce` (WO-02 SFT attempt 3: probe + 8000 steps + sweep) | 02 | RTX 4090, $0.89/h | 4.45 | 4.01 | yes, 2026-10-27 (go on probe 1.575 s/step) |
| pod `cm7d9mlccy3xjm` (WO-02 receding-horizon recheck, eval only) | 02 | RTX 4090, $0.89/h | 0.35 | 0.31 | yes, 2026-10-27 (approved $0.5) |
| pod `63yklak55c2zvf` (WO-02 attempt 4, approval gate expired before the go) | 02 | RTX 4090, $0.89/h | 0.58 | 0.52 | yes, 2026-10-27 |
| pod `u87r9sjnwig58d` (WO-02 attempt 4: 25 demos/task, 8000 steps + sweep) | 02 | A100-80GB, $1.79/h | 2.81 | ~5.0 (3.35 billed at 09:51 UTC) | yes, 2026-10-27 (go on probe 0.774 s/step; pod cap $5.5) |

Running total: ~$10.3-12.0 (RunPod billing; last pod still settling); WO-02 cap $15.

## Blockers and decisions pending the owner

- WO-02: OpenVLA-OFT few-shot SFT stays below the 20-60% band after 4 attempts (best 18/16/6% at
  25 demos/task). Next per the owner's path: pi0.5 (openpi LoRA on an A100, separate environment,
  RLinf flow-matching RL replacing WO-03/04's token GRPO) — needs the owner's approval of a plan
  and cost before any spend. WO-04 RL runs are blocked until an in-band SFT checkpoint exists.

## Owner decisions in effect

- 2026-10-26 WO-02 SFT is LoRA only, on the sim.yaml budget (2000 steps), RTX 4090 tier, pinned
  CUDA 12.4.1 image; WO-02 spend cap $15; target 20-60% on seen positions (not max success).
  Below 20% -> up to two more SFT attempts, then the pi0.5 switch (ADR-0001); above 60% -> cut
  steps or demos.
- 2026-10-26 Every paid run: probe first, then an explicit go on the posted probe numbers before
  the full run (job-script approval gate). WO-04's RL runs (~$135) must not start without it.
- 2026-10-26 WO-04 settings: LoRA lr 5e-5 (not 1e-5); kill switch evaluates 64 episodes/task at
  step 0 and at the 25% check and stops only if the improvement is under 3 points; ppo_mini_batch 64.
- 2026-10-27 WO-04 GPU: prefer the H100 if its measured $ per seed is within ~10% of the A100's
  (wall-clock matters); otherwise the A100. Decided from probe numbers on both, not estimates.
- 2026-10-27 Leader arm ordered: WO-06 keeps the teleop recording step (5 real episodes/task).

## Decisions taken by the agent (one line each, link ADR if any)

- 2026-10-26 Python 3.11 kept; LeRobot pinned to 0.4.4, the last release supporting 3.11 (0.5+ needs ≥3.12).
- 2026-10-26 PyTorch 2.6.0 instead of 2.5.x: every LeRobot ≥0.3 needs torchcodec ≥0.2.1, which pairs with torch 2.6.
- 2026-10-26 ManiSkill 3.0.1 (stable) instead of a 3.0.0bX beta; SAPIEN 3.0.3 (macOS wheels).
- 2026-10-26 LeRobotDataset v3.0 (lerobot 0.4.4 native), one multi-task dataset with a per-episode
  sidecar for task family/seed/cell, instead of v2 per-task directories — LeRobot trains from one repo_id.
- 2026-10-26 SO-101 model vendored from TheRobotStudio/SO-ARM100 (Apache-2.0). lerobot-sim2real has
  no license, so nothing is copied from it; "squint" could not be found.
- 2026-10-26 Action space: absolute joint-position targets in radians, LeRobot so101_follower joint
  names/order — matches the real arm's interface, so WO-07 needs no IK on a 5-DoF arm.
- 2026-10-26 Oracle uses IK (pytorch_kinematics), not mplib: mplib has no macOS wheels.
- 2026-10-26 Mac runs ManiSkill CPU sim + rendering via MoltenVK (`brew install molten-vk`);
  torchcodec needs `brew install ffmpeg@7`. `scripts/mac_env.sh` / Makefile set the env vars.
- 2026-10-26 SmolVLA smoke (WO-01 item 5) runs on the Mac (MPS), as ENVIRONMENT.md allows — no paid run.
- 2026-10-26 OpenVLA-OFT needs torch 2.2 + a transformers 4.40 fork that conflicts with SmolVLA's
  transformers; to be resolved in WO-02 as a separate uv extra declared conflicting.
- 2026-10-26 WO-03 loss aggregation = upstream SimpleVLA-RL dp_rob.py: per-trajectory token-mean, then mean over the mini-batch (not DAPO token-mean); advantage = (r - mean) / (unbiased std + 1e-6).
- 2026-10-26 WO-03 dynamic sampling drops all-equal groups and refills per env batch, capped at `max_rollout_rounds` (upstream: whole rounds, uncapped).
- 2026-10-26 WO-03 rl preset made cost-aware (owner budget ~$200): 16 prompts x 8 per step and sampler log-probs reused as old log-probs; estimated 25 h / ~$45 per seed vs ~$227 at the upstream 64 prompts (`scripts/wo03_rl.py --preset rl --estimate`; estimates only).
- 2026-10-26 Policy camera 224x224 (OpenVLA-OFT input size), 40 deg vertical fov at 0.70 m
  (real webcam must cover >= 40 deg vertical; C920-class qualifies); real.yaml matches.
- 2026-10-26 Sim 300 Hz / control 15 Hz; PhysX position iterations 25; jaw collision meshes split
  into convex slabs (upstream hull filled the jaw gap).
- 2026-10-26 make_env defaults to one env on CPU (64 serial envs exhausted MoltenVK descriptor pools).
- 2026-10-26 lerobot 0.4.4 hands torchcodec 0.2.1 a file object; src/data/recorder.py shims it to
  open local files by path. Candidate for a pin-upgrade WO later.
