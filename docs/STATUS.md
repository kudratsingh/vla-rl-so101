# Status

Updated: 2026-10-26

## Current work order

Independent WOs run in parallel (owner decision 2026-10-26, CLAUDE.md "Operating mode"). Any WO
started before its predecessors merge is listed here, marked **out of order**, before work begins.

| WO | State | Branch / PR | Notes |
|---|---|---|---|
| 01 | done (benchmark met), PR open | `wo-01-sim-env` | See benchmarks. Demos on the Hub (`KudratSingh22/vla-rl-so101-demos`, private, built at commit `0dffe51`). |
| 02 | in progress, **out of order** (started before WO-01 merged) | `wo-02-sft` | Uses WO-01's code (merged into its branch) and the Hub demos above. LoRA SFT (sim.yaml budget: 2000 steps) running on RunPod pod `iqrmjox6ws0e66` since 2026-10-26 20:18 UTC; owner-approved, WO-02 spend cap $15. If WO-01 review changes env/oracle/camera, the demos and this SFT are redone. |
| 03 | done, merged after owner review | #4 (merged) | **Out of order** (started before WO-01/02). CPU only, toy env. |
| 06 | part 1 merged, part 2 waits for the arm | #5 (merged) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. |
| 04 | not started; blocked on WO-02's SFT checkpoint | | Owner settings recorded below; applied as WO-04's first commit. Noted here before it starts. |
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
| pod `iqrmjox6ws0e66` (WO-02 probe + SFT + eval) | 02 | RTX 4090, $0.89/h | running | — | yes, 2026-10-26 (WO-02 cap $15) |

Running total: $0 settled; WO-02 pod running (max $5.34 by its 6 h self-delete cap).

## Blockers and decisions pending the owner

- (none)

## Owner decisions in effect

- 2026-10-26 WO-02 SFT is LoRA only, on the sim.yaml budget (2000 steps), RTX 4090 tier, pinned
  CUDA 12.4.1 image; WO-02 spend cap $15; target 20-60% on seen positions (not max success).
  Below 20% -> up to two more SFT attempts, then the pi0.5 switch (ADR-0001); above 60% -> cut
  steps or demos.
- 2026-10-26 Every paid run: probe first, then an explicit go on the posted probe numbers before
  the full run (job-script approval gate). WO-04's RL runs (~$135) must not start without it.
- 2026-10-26 WO-04 settings: LoRA lr 5e-5 (not 1e-5); kill switch evaluates 64 episodes/task at
  step 0 and at the 25% check and stops only if the improvement is under 3 points; ppo_mini_batch 64.

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
