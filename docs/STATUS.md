# Status

Updated: 2026-10-27

## Current work order

Independent WOs run in parallel (owner decision 2026-10-26, CLAUDE.md "Operating mode"). Any WO
started before its predecessors merge is listed here, marked **out of order**, before work begins.

| WO | State | Branch / PR | Notes |
|---|---|---|---|
| 01 | done, merged | #7 (merged) | See benchmarks. Demos on the Hub (`KudratSingh22/vla-rl-so101-demos`, private, built at commit `0dffe51`). |
| 02 | closed by owner decision, **below the original target band**; PR open | `wo-02-sft` | OpenVLA-OFT LoRA SFT, 4 attempts, all below the 20-60% band on seen positions (ADR-0001). Owner (2026-10-27): keep OFT, amend rule 3, close with the full protocol table of attempt 4 `step_004000` (25 demos/task): seen PickPlace 17.0%, Push 10.0%, Stack 2.7% (`docs/results.md`). |
| 03 | done, merged after owner review | #4 (merged) | **Out of order** (started before WO-01/02). CPU only, toy env. |
| 06 | part 1 merged, part 2 waits for the arm | #5 (merged) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. |
| 04 | seed 0 running, **out of order** (started before WO-02 merged) | #11 (owner-approved), `wo-04-seeds` | Owner-amended GRPO (ADR-0002 amendment 1): T 1.0, K 16 x 8 prompts, PickPlace + Push trained (Stack eval-only), horizon 120, 150 steps, kill switch at 38. Seed 0 on pod `9d8p7vyfyv6ca4` (A100), auto-gate: probe <= 60% groups dropped and <= $60 projected. Seeds 1-2 only after seed 0 passes the step-38 check. |
| 05, 07, 08, 09 | not started | | |

## Benchmarks met

| WO | Met on | Evidence (run id / PR) |
|---|---|---|
| 01 | 2026-10-26 | Oracle, seen cell, 100 episodes/task, Mac CPU: PickPlace 100%, Push 99%, Stack 100% (run `20261026T194952Z-wo01-sim-8ad7`, which also recorded the 10 demos/task: 30 episodes, 1968 frames, pushed privately to `KudratSingh22/vla-rl-so101-demos`). SmolVLA smoke, 2000 steps x batch 8 on Mac MPS, 10 seen episodes/task: PickPlace 50%, Push 20%, Stack 0% (single run; run `20261026T195329Z-wo01-sim-b326`). `tiny` preset build: 6.9 s on CPU. |
| 02 | **not met** (closed by owner decision 2026-10-27) | Full protocol, attempt 4 `step_004000`, 3 cells x 3 seeds x 100 episodes, GPU sim, protocol `0.1-draft` `e9ce5bb47a88`: seen 17.0 / 10.0 / 2.7% (PickPlace / Push / Stack), **below the original 20-60% band**; meets the amended rule 3 (>= 10% on 2 of 3 tasks), Push exactly at 10.0%. Run `20261027T190836Z-wo02eval-sim-cdd3` (pod `jqtaavsbwvemin`). Eval harness emits the full table from one command (`scripts/wo02_eval.py`). |
| 03 | 2026-10-26 | tiny toy GRPO, seeds 0/1/2: greedy eval 0.301/0.102/0.191 -> 0.988/1.000/0.996 of optimum 1.0 in 100 steps (>=0.95 by step 20/20/25), bit-identical reruns, ~3.5 s CPU each; runs 20261026T193248Z-wo03-tiny-s0-be5126, -193252Z-…-s1-cf3b6d, -193255Z-…-s2-d11eb5 (reruns -fe1afa/-812773/-8c72b6); branch `wo-03-rl-loop` @ 051f1e6 |
| 04 | | |
| 05 | | |
| 06 | | |
| 07 | | |
| 08 | | |
| 09 | | |

## Spend

RunPod pod billing (list-pod-billing, 2026-10-27 ~20:15 UTC; the last two finished pods were still
settling, so their rows use wall-clock x rate).

| Run id | WO | GPU | Hours | $ | Approved by owner |
|---|---|---|---|---|---|
| pod `iqrmjox6ws0e66` (WO-02 SFT attempt 1: probe + 2000 steps + seen sweep) | 02 | RTX 4090, $0.89/h | 1.05 | 0.94 | yes, 2026-10-26 |
| pod `efe7plxr06gaqc` (WO-02 SFT attempt 2: probe + 2000 steps + seen sweep) | 02 | RTX 4090, $0.89/h | 1.30 | 1.17 | yes, 2026-10-27 (go on probe 1.547 s/step) |
| pod `l3485bxmdj0fce` (WO-02 SFT attempt 3: probe + 8000 steps + sweep) | 02 | RTX 4090, $0.89/h | 4.45 | 4.01 | yes, 2026-10-27 (go on probe 1.575 s/step) |
| pod `cm7d9mlccy3xjm` (WO-02 receding-horizon recheck, eval only) | 02 | RTX 4090, $0.89/h | 0.35 | 0.31 | yes, 2026-10-27 (approved $0.5) |
| pod `63yklak55c2zvf` (WO-02 attempt 4, approval gate expired before the go) | 02 | RTX 4090, $0.89/h | 0.58 | 0.52 | yes, 2026-10-27 |
| pod `u87r9sjnwig58d` (WO-02 attempt 4: 25 demos/task, 8000 steps + sweep) | 02 | A100-80GB, $1.79/h | 2.81 | 5.05 | yes, 2026-10-27 (go on probe 0.774 s/step; pod cap $5.5) |
| pod `19ke4869g8zmsg` (WO-02 full eval, approval gate expired before the go) | 02 | RTX 4090, $0.89/h | 0.58 | 0.52 | yes, 2026-10-27 |
| pod `jqtaavsbwvemin` (WO-02 full protocol eval of a4 `step_004000`) | 02 | RTX 4090, $0.89/h | 1.17 | ~1.04 | yes, 2026-10-27 (pre-approved, cap $2.23) |
| pod `5rdd45j801bk5b` (WO-04 probe, original settings) | 04 | A100-80GB, $1.79/h | 0.80 | 1.45 | yes, 2026-10-27 |
| pod `nz8fp6rckh1yfe` (WO-04 probe, original settings) | 04 | H100 SXM, $3.99/h | 0.86 | 3.44 | yes, 2026-10-27 |
| pod `gzxhwhjhdtc1kc` (WO-04 confirming probe, amended settings) | 04 | A100-80GB, $1.79/h | 0.83 | ~1.49 | yes, 2026-10-27 (approved $1.4) |
| pod `9d8p7vyfyv6ca4` (WO-04 seed 0: probe -> auto-gate -> 150-step run -> protocol eval) | 04 | A100-80GB, $1.79/h | running | <= 65 (36.5 h hard cap) | yes, 2026-10-27 (auto-gate: <= 60% dropped, <= $60 projected) |

Running total (finished pods): WO-02 ~$13.56 of its $15 cap; WO-04 ~$6.38; total ~$19.94.

## Blockers and decisions pending the owner

- (none) Seeds 1-2 wait for seed 0's step-38 kill-switch check (owner rule), not for a decision.

## Owner decisions in effect

- 2026-10-26 WO-02 SFT is LoRA only, on the sim.yaml budget (2000 steps), RTX 4090 tier, pinned
  CUDA 12.4.1 image; WO-02 spend cap $15; target 20-60% on seen positions (not max success).
  Below 20% -> up to two more SFT attempts, then the pi0.5 switch (ADR-0001); above 60% -> cut
  steps or demos.
- 2026-10-26 Checkpoint selection: in band = 20-60% seen success on >= 2 of 3 tasks. If the final
  checkpoint lands above 60%, take the earliest swept checkpoint inside the band before any
  retrain (scripts/pod/wo02_job.sh `select` stage).
- 2026-10-26 Every paid run: probe first, then an explicit go on the posted probe numbers before
  the full run (job-script approval gate), unless the owner pre-approves a dollar amount or an
  automatic gate condition for that run.
- 2026-10-26 WO-04 settings: LoRA lr 5e-5 (not 1e-5); kill switch evaluates 64 episodes/task at
  step 0 and at the 25% check and stops only if the improvement is under 3 points; ppo_mini_batch 64.
- 2026-10-27 WO-04 GPU: prefer the H100 if its measured $ per seed is within ~10% of the A100's
  (wall-clock matters); otherwise the A100. Probes: A100 $108 vs H100 $206 per seed -> A100.
- 2026-10-27 Leader arm ordered: WO-06 keeps the teleop recording step (5 real episodes/task).
- 2026-10-27 No pi0.5 switch. Rule 3 amended: RL may start from a checkpoint with >= 10% on at
  least 2 of 3 tasks (CLAUDE.md, ADR-0001). WO-02 closes with the full eval table of attempt 4
  `step_004000`, labelled "below the original target band".
- 2026-10-27 WO-04 amendment (ADR-0002 amendment 1): temperature 1.0; RL on PickPlace + Push,
  Stack eval-only; recompute_old_log_probs on; RL horizon 120 (protocol eval stays 150); 150
  steps; kill switch at step 38 on the trained-task mean; if a probe drops > 60% of groups, K 16
  first (fired: confirming probe 72% -> K 16 x 8 prompts, 128 trajectories per update).
- 2026-10-27 WO-04 seeds: seed 0 passes its gate automatically if its K=16 probe drops <= 60% of
  groups and projects <= $60; otherwise stop and report. Seeds 1-2 only after seed 0 passes the
  step-38 check.

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
- 2026-10-26 WO-02 OpenVLA-OFT on stock transformers 4.57.6 + timm 0.9.16 with a bidirectional 4-D
  mask hook, not the moojink fork (bit-identical on CPU; no uv conflicts). See ADR-0001.
- 2026-10-26 WO-02 token layout keeps OFT's `▁` (29871) before the action slots in SFT, eval and RL.
- 2026-10-26 WO-02 SFT loss = CE on the 256-bin action slice (the distribution RL samples).
- 2026-10-26 WO-02 action/proprio bounds from joint limits ∩ reachable workspace (≤1°/bin), not demo
  min/max; wrist roll widened to ±2.1 rad to cover the oracle demos (0 frames clipped, 0.944°/bin).
- 2026-10-26 WO-02 proprio on; the new proprio projector is trained during SFT and frozen during RL;
  the 7.56B pretrained weights stay frozen (LoRA r32 on every Linear incl. lm_head).
- 2026-10-26 RunPod Vulkan: the mounted GLX ICD fails headless; an EGL ICD at
  /usr/share/vulkan/icd.d/nvidia_icd.json fixes GPU and CPU sim rendering (ENVIRONMENT.md).
- 2026-10-27 Pod jobs size torch/OpenMP threads from the cgroup CPU quota (v2 `cpu.max`, v1
  `cfs_quota_us`): hosts expose 128-256 cores with a 14-27 CPU quota.
- 2026-10-27 WO-04 kill switch compares the trained-task mean (`eval/success_train`); Stack is
  evaluated and reported but cannot trip or pass it (owner confirmed on PR #11).
- 2026-10-27 WO-04 K=16 keeps 128 trajectories per update (8 prompts), not 256 (owner confirmed).
