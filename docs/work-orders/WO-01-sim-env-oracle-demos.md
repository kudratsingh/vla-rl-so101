# WO-01 — Sim env, scripted oracle, few-shot demos

> **Pivot 2026-11-09 (ADR-0003):** the OpenVLA-OFT / GRPO track is stopped. Benchmark met 2026-10-26. The content below is
> kept as written for the record; the current work orders are WO-06 to WO-09.

Branch: `wo-01-sim-env`. Tier: dev (4090) after CPU smoke.

## Deliverables

1. `src/env/`: ManiSkill3 scene with the SO-101 follower (vendored from lerobot-sim2real/squint,
   header comment with source commit + license), one fixed RGB camera (128×128, pose recorded in
   `configs/sim.yaml`), three tasks `PickPlace`, `Push`, `Stack`, each with a programmatic success
   check in `src/env/success.py`.
2. Domain randomization hooks (off by default, on via config): object position/colour, lighting,
   camera pose jitter, robot colour, action latency (0–2 steps).
3. `src/data/oracle.py`: scripted motion-planner policy per task using privileged state.
4. `scripts/wo01_build_env.py --preset {tiny,sim}`: builds the env, runs the oracle, records
   10 demos per task to LeRobotDataset v2 under `data/demos/<task>/`, pushes to HF Hub when
   `--push` is set.
5. SmolVLA SFT smoke on the 10×3 demos (`scripts/wo01_smolvla_smoke.py`) — proves the dataset
   format and the train/eval plumbing; the number does not matter beyond being >0%.
6. `docs/ENVIRONMENT.md` pin table filled with resolved versions; `uv.lock` committed.

## Benchmark

Oracle ≥95% success on each task over 100 episodes; 10 demos per task on the Hub; `tiny` preset
runs on CPU in <2 min; SmolVLA smoke >0% on at least one task.

## Notes

- The sim camera pose and intrinsics chosen here are what the real camera must match in WO-06.
  Pick a pose that is physically easy to reproduce (front-left, 45° down, ~60 cm).
- "Shifted positions" cell = object spawn region 1.5× the demo region; "unseen objects" cell =
  two held-out meshes/colours never used in demos. Define both now in `src/eval/protocol.py` even
  though eval comes in WO-02.
- Reminder to owner (not agent): order the arm and cameras this week.
