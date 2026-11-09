# WO-06 — Bring-up

Branch: `wo-06-hardware-values` (bring-up values, owner) and the agent's code branches. Tier: Mac
only; $0 cloud. Rewritten 2026-11-09 for the SmolVLA pivot (ADR-0003). The robot-side code
(safety, units, driver, calibration, camera match, teleop recorder) is merged (#5, #24, #29).

## Done when

Checklist steps 0–10 complete under supervision: calibration verified, per-session drift check,
teleop with the leader arm, both cameras matched and exposure/white balance locked, MPS inference
latency measured on the mock rig.

## Deliverables

1. `docs/bringup/WO-06-checklist.md` steps 0–10 run by the owner, results reported back with the
   table at its end. Powered steps 9–10 only under supervision (owner in person).
2. The checklist's **Hardware floor** in place before any demo is recorded (two cameras, P gain
   recorded, drift check, spare gripper servo, exposure / white balance locked, #29 shutdown and
   fault behaviour).
3. `configs/real.yaml` bring-up values (port, calibration-derived limits, both camera indices and
   intrinsics, locked camera controls, servo P gain) committed through a PR.
4. Both cameras matched: the front camera at the sim pose (`front_left_45`), the wrist module on
   the gripper; side-by-side frames saved under `docs/assets/camera-match/`.
5. Mac MPS inference latency for SmolVLA and ACT on the mock rig: chunk latency and the gap between
   consecutive commands, recorded (`docs/bringup/latency-measurement.md`; code in PR 2).

## Notes

- The OFT-era deliverable "randomization retrain if the camera match shows a gap" is dropped: the
  real policy is trained on real demos. The camera match now gates the two-camera recording and
  the WO-08 sim wrist camera, not a sim retrain.
- STS3215 servos overheat under aggressive commands; the velocity cap is not optional.
- No arm commands without the owner present.
