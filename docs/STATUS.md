# Status

Updated: 2026-10-26

## Current work order

WO-01 — in progress on `wo-01-sim-env`. Env/tasks/oracle and dataset/SmolVLA scripts being built
in parallel; pins resolved (see `docs/ENVIRONMENT.md`).

## Benchmarks met

| WO | Met on | Evidence (run id / PR) |
|---|---|---|
| 01 | | |
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
| | | | | | |

Running total: $0.

## Blockers and decisions pending the owner

- (none)

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
