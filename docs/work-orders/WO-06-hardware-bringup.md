# WO-06 — Bring-up

Branch: `wo-06-hardware-values` (bring-up values, owner) and the agent's code branches. Tier: Mac
only; $0 cloud. Rewritten 2026-11-09 for the SmolVLA pivot (ADR-0003). The robot-side code
(safety, units, driver, calibration, camera match, teleop recorder) is merged (#5, #24, #29).
Hardware: the owner's bill of materials, [`docs/bringup/kit.md`](../bringup/kit.md) (2026-11-10).

## Done when

Checklist steps 0–10 complete under supervision: calibration verified, per-session drift check,
teleop with the leader arm, both cameras matched and exposure/white balance locked, MPS inference
latency measured on the mock rig.

## Deliverables

1. `docs/bringup/WO-06-checklist.md` steps 0–10 run by the owner, results reported back with the
   table at its end. Powered steps 9–10 only under supervision (owner in person).
2. The checklist's **Hardware floor** in place before any demo is recorded (two cameras, P gain
   recorded, drift check, spare 12 V gripper servo, lamp fixed at 5000 K, exposure / white balance
   locked, #29 shutdown and fault behaviour).
3. `configs/real.yaml` bring-up values (port, calibration-derived limits, both camera indices and
   intrinsics with the measured checkerboard square, locked camera controls, servo P gain)
   committed through a PR; both arms' bus voltages reported (checklist step 2b).
4. Both cameras matched: the C920x front camera on the magic arm at the sim pose
   (`front_left_45`), the 32 mm wrist module on the gripper mount; the white cup rim visible in
   both under the 5000 K lamp (white-on-white risk, checklist step 7); side-by-side frames saved
   under `docs/assets/camera-match/`.
5. Mac MPS inference latency for SmolVLA and ACT on the mock rig: chunk latency and the gap between
   consecutive commands, recorded (`docs/bringup/latency-measurement.md`; code in PR 2).

## Notes

- The OFT-era deliverable "randomization retrain if the camera match shows a gap" is dropped: the
  real policy is trained on real demos. The camera match now gates the two-camera recording and
  the WO-08 sim wrist camera, not a sim retrain.
- STS3215 servos overheat under aggressive commands; the velocity cap is not optional.
- No arm commands without the owner present.
