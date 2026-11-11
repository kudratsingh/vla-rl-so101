# Hardware kit (bill of materials)

The items used for training, runs and evaluation on the real arm. This is the owner's list of
2026-11-10, in his sections. Everything else in the repo follows it: protocol real-0.1
(`src/eval/protocol_real.py`), `configs/real.yaml`, the WO-06 checklist
(`docs/bringup/WO-06-checklist.md`) and the WO-07 work order. Supply voltage is read over the
servo bus (checklist step 2b). The table surface is all white.

Measured values still to come from the owner (until then they are PENDING in ONE place,
`src/eval/protocol_real.py` `MEASURED`, in mm; `freeze_problem()` refuses the freeze while any is
missing or out of range, `measurement_problems()`):

| Measurement | Field | Today |
|---|---|---|
| Red cube edge | `red_cube_edge_mm` | PENDING (25.4 mm nominal) |
| Cup outer rim diameter | `bowl_outer_rim_diameter_mm` | PENDING |
| Cup inside floor diameter | `bowl_inside_floor_diameter_mm` | PENDING |
| Table top to cup rim height | `bowl_table_to_rim_height_mm` | PENDING |
| Table top to the cup's inside floor (rim height less the inside depth; a footed cup's floor sits above the table) | `bowl_table_to_inside_floor_height_mm` | PENDING |
| Unpowered fit check of bowl B' (jaws and cube at the bowl and at C1, D1, B2, E2) | `FREEZE_STATUS` | PENDING |

Filling `MEASURED` is the one edit before the geometry rerun (`scripts/wo07_bowl_check.py`):
`src/eval/bowl_fit.py` `geometry()` reads it for the release window, the cube-to-rim clearance,
the carry-over-rim clearance, the lower-over-floor clearance and the release height. Any of those
clearances below zero, or a measurement out of range, makes the bowl check FAIL
(`bowl_fit.geometry_problems`). Until then those checks use the 25.4 mm nominal cube and the sim
bowl's geometry, labelled as a fallback in their output.

## Kit (Seeed SO-ARM101 Pro, already have)

| Item | Used for | Cited in |
|---|---|---|
| SO-101 follower arm, 12 V STS3215 servos, with 12 V supply | The arm that runs every demo and trial. 12 V STS3215: retailer listings give about 2.7 A stall at 12 V and 4-14 V operating range (Feetech datasheet not re-checked) | `configs/real.yaml` `power.follower`, `robot.servo_monitor`; checklist steps 0, 2b |
| SO-101 leader arm, 7.4 V servos, with 5 V supply | Teleop for demos (leader is read only, never driven) | `configs/real.yaml` `power.leader`, `teleop.leader`; checklist steps 0, 2b, 8 |
| Two Waveshare bus servo boards and USB cables | One board per arm, each straight into the Mac (no shared hub) | checklist step 0 item 4, steps 1, 8 |
| 32 mm USB camera module | The WRIST camera, on the gripper mount | `configs/real.yaml` `cameras.wrist`; protocol `scene_spec()`; checklist steps 6, 7 |
| Table clamps | Clamp the follower base (and the leader) to the table | checklist step 0 item 3 |

## Ordered, arriving

| Item | Used for | Cited in |
|---|---|---|
| Logitech C920x webcam | The FRONT camera at the sim pose `front_left_45` (eye [0.555, 0.350, 0.495] m, target [0.205, 0, 0] m) | `configs/real.yaml` `cameras.front`; protocol `scene_spec()`; checklist steps 6, 7 |
| NEEWER UA057 22 inch magic arm with desk clamp | Holds the C920x at the sim pose | checklist step 7 |
| 17 inch LED clamp desk lamp, 2700 K to 6500 K | The only light: set ONCE to 5000 K and never changed; position recorded at the camera match | protocol `scene_spec()`; `configs/real.yaml` camera comments; checklist step 7 |
| hand2mind 1 inch wood cubes, set of 100, six colours | All objects: RED = seen PickPlace object; BLUE = seen Stack base (Stack deferred, unused in real-0.1); PURPLE and GREEN = unseen objects. Edge 25.4 mm nominal until measured; same wood, mass and finish, only the colour differs | `src/eval/protocol_real.py` objects; WO-07 work order |
| LE TAUCI 8 oz bouillon cups, white, set of 6 | The bowl: one cup at the fixed bowl mark; the others are spares. Rim, floor and height PENDING measurement | `src/eval/protocol_real.py` `bowl_spec()`, `MEASURED` |

## Already had

| Item | Used for | Cited in |
|---|---|---|
| Inline outlet switch | The e-stop on the follower's 12 V supply: off = torque gone at once (the arm drops) | checklist step 0 item 2, steps 9-10, "Ending any session safely" |

## Print at home or Office Depot

| Item | Used for | Cited in |
|---|---|---|
| A3 placement grid from PR #29, or two tabloid sheets taped | Slot crosses, robot base outline, the fixed bowl. A3: `docs/assets/placement-grid-a3.pdf`. Tabloid: `scripts/make_placement_grid.py --preset real --layout tabloid2` (two 11 x 17 in sheets, 10 mm overlap, cut line, registration ticks R1-R3, a 100 mm scale bar on each sheet, the bowl cross on sheet 2). Both are regenerated with the real-0.1 freeze, not before | `configs/real.yaml` `eval.grid`; checklist step 10 |
| 10 by 7 checkerboard on letter paper, glued flat to cardboard, square size measured | Intrinsics of BOTH cameras: 9 x 6 inner corners; the measured square is passed as `--square-mm` (required) and recorded | `scripts/wo06_camera_intrinsics.py`; checklist step 6 |
| 2 inch matte black disc | The Push target (50.8 mm). Push is deferred until the PickPlace baseline is in | this file; checklist step 10 (sanity episodes only) |

## Still to buy, small

| Item | Used for | Cited in |
|---|---|---|
| Spare gripper servo, STS3215 12 V, about $20 | The gripper servo takes the most load; a spare of the follower's 12 V variant (never the leader's 7.4 V one) | checklist Hardware floor, step 9 |
| Wrist camera mount for the SO-101 gripper, if the kit does not include one | 3D-printed from the LeRobot SO-101 files; holds the 32 mm module | checklist step 7 |

## Deferred

| Item | Status |
|---|---|
| Green cylinder | Replaced by the green cube as an unseen object (real-0.1); the sim protocols keep their cylinder |
| Push and Stack props beyond the disc | Until the PickPlace baseline is in |

## Risks recorded with the kit

- **White on white.** The white cup on the all-white table is low contrast for both cameras. The
  camera-match step (checklist step 7) must show the cup rim visible in BOTH cameras under the
  lamp at 5000 K with exposure and white balance locked. The table stays white.
- **Sim scene differs.** The sim (protocol 0.2, frozen) has a wood-coloured table, an off-white
  sim bowl and 3 cm cubes. The WO-08 two-camera sim protocol should use a white table, a white
  cup and 25.4 mm cubes to match this scene.
- **Cup height.** An 8 oz bouillon cup may be taller than the sim bowl (36 mm). If the measured
  table-to-rim height leaves the cube's bottom below the rim at the sim oracle's carry height,
  `bowl_fit.carry_rim_clearance_m` goes negative and the bowl check FAILS (`PROBLEM carry through
the rim`). A footed cup whose inside floor sits higher than the oracle's lower (18 mm over the
table) fails the same way (`lower into the floor`). If the cup has handles,
  their orientation is recorded at the fit check and kept.
- **Smaller cube.** 25.4 mm against the sim's 30 mm: the jaws close further before contact (rough
  estimate about +0.1 rad, from the sim's 0.15-0.2 rad on a 3 cm cube; unverified), still above
  the -5 deg gripper floor. Step 9 notes whether a cube slips.
