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
| 03 | | |
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

## WO-02 (branch `wo-02-sft`)

- Built: vendored OpenVLA-OFT (`src/policy/oft/`), `OpenVLAOFTPolicy` (act / SFT loss / RLPolicy
  methods, structural match to WO-03's `src/rl/policy_api.py`), LeRobot v3.0 → OFT data path,
  workspace-derived normalisation bounds, `scripts/wo02_sft.py` (+ `--probe-steps`),
  `src/eval/sim_eval.py` + `scripts/wo02_eval.py`. CPU tiny SFT runs in ~5 s; 23 tests pass.
- Blocked on the WO-01 merge (lazy imports until then): `src.data.recorder`, `src.runs`,
  `src.env` (`make_env`, `policy_obs`), `src.eval.protocol`, the SO-101 URDF, and the demo
  dataset on the Hub. Protocol freeze still to do in WO-02 once it lands (eval episode seeds are
  derived in `src/eval/sim_eval.py:episode_seeds`; consider moving them into the protocol first).
- Pending owner: approve the GPU runs (probe ≈ $0.25, SFT 20k-step cap ≈ $6–7 on A100-80GB vs
  ≈ $8.5–10.5 on RTX 4090, checkpoint sweep + full eval ≈ $2–3). Recommend the A100 (cheaper per
  run, no OOM risk). Cost line: `estimated: A100-80GB × 3.9 h = $7.0` (SFT, worst case).

Decisions taken by the agent (WO-02):

- 2026-10-26 OpenVLA-OFT on stock transformers 4.57.6 + timm 0.9.16 with a bidirectional 4-D mask
  hook, not the moojink fork (bit-identical on CPU; no uv conflicts). See ADR-0001.
- 2026-10-26 Token layout keeps OFT's `▁` (29871) before the action slots in SFT, eval and RL alike.
- 2026-10-26 SFT loss = CE on the 256-bin action slice (the distribution RL samples), not full vocab.
- 2026-10-26 Action/proprio bounds from joint limits ∩ reachable workspace (≤1°/bin), not demo min/max.
- 2026-10-26 Proprio on; proprio projector trained in full during SFT, frozen during RL (LoRA-only).
- 2026-10-26 LoRA targets every Linear incl. lm_head (peft 0.11 "all-linear", as upstream adapters).
- 2026-10-26 CI syncs the `gpu` extra too, so policy tests run (`uv sync --frozen --extra dev --extra gpu`).
