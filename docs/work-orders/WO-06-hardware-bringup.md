# WO-06 — Hardware bring-up and sim/real camera match

Branch: `wo-06-hardware`. Tier: Mac + dev GPU.

## Deliverables

1. `src/robot/`: LeRobot SO-101 driver glue, calibration script wrapper, `safety.py` (joint
   limits, workspace box, per-step action clip, velocity cap, deadman on frame/serial loss).
2. `configs/real.yaml`: serial port, camera ids, intrinsics, the sim-matching camera pose, object
   grid definition.
3. `scripts/wo06_camera_match.py`: captures real frames and renders the sim camera at the same
   pose; saves side-by-side images to `docs/assets/camera-match/`.
4. Teleop recording of 5 real episodes per task (for sanity, not training) in LeRobotDataset
   format.
5. If the side-by-side shows a clear gap (colour, lighting, crop): enable the WO-01 randomization
   hooks and re-run SFT+RL once (owner approves) → "randomized" checkpoint.

## Benchmark

Arm calibrated and teleoperable; safety module blocks an out-of-bounds command in a test; real
and sim frames side by side with the same object layout; a decision recorded in STATUS.md on
whether a randomized retrain is needed.

## Notes

- Owner does the physical assembly and calibration on the Mac; the agent writes the scripts and
  the safety module first so they are ready when the arm arrives.
- STS3215 servos overheat under aggressive commands; the velocity cap is not optional.
