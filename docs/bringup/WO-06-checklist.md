# WO-06 bring-up checklist (Mac, zsh)

An ordered list of steps you can copy and paste, from a bare kit to 15 teleop sanity episodes.
Work through it top to bottom. Each step lists the command, what it writes, the expected
output, a pass/fail check, and what to send back to the agent. Every command was checked against
the scripts on branch `fix/robot-safety` (PR #29) and against lerobot 0.4.4 as installed in `.venv`, and
exercised with fakes or on the CPU sim. None of it has run on the real arm yet, so the expected
outputs are what the code prints, not observed hardware behaviour.

**Motion markers.** Each marker cites the code that justifies it. Repo code is cited by symbol
(function, class or config key), not by line number, so later edits do not break the citation.

| Marker | Meaning |
|---|---|
| 🟢 **NO MOTION** | Torque stays off and nothing writes a goal position. Safe to run with the arm unattended. |
| 🟡 **TORQUE OFF, OWNER MOVES JOINTS BY HAND** | The servos are limp and you move the joints. The arm sags or drops if you let go, so support it. |
| 🔴 **POWERED MOTION: OWNER IN PERSON** | Torque is on and the software drives the follower. Keep one hand on the servo power switch. The arm is clamped, the workspace is clear, and nobody else is within reach. |

`LR/` means `.venv/lib/python3.11/site-packages/lerobot/` (lerobot 0.4.4).

## Never use these stock CLIs on this arm

| CLI | Why |
|---|---|
| `lerobot-teleoperate` | Writes leader positions straight to the follower's `Goal_Position` with no `src/robot/safety.py` (`LR/scripts/lerobot_teleoperate.py:176`). `robot.connect()` defaults to calibrate (line 222), and `max_relative_target` defaults to `None` (`LR/robots/so_follower/config_so_follower.py:37`). |
| `lerobot-record` | Drives the follower the same way (`LR/scripts/lerobot_record.py:406,519`). It also defaults to `push_to_hub=True, private=False` (lines 171-173), so it uploads publicly. |
| `lerobot-replay` | Sends recorded actions to the follower with no safety filter (`LR/scripts/lerobot_replay.py:111,127`). |
| `lerobot-find-joint-limits` | Connects both arms and calls `robot.send_action` with no safety filter (`LR/scripts/lerobot_find_joint_limits.py:96,130`). |
| `lerobot-calibrate` | Without `--robot.id` it writes `None.json`, because the file name is `<id>.json` (`LR/robots/robot.py:53`), while the repo reads `so101_follower.json`. Its `connect(calibrate=False)` (`LR/scripts/lerobot_calibrate.py:89`) enables follower torque with no Goal := Present hook. Use `scripts/wo06_calibrate.py`. |

The only stock CLIs used below are `lerobot-setup-motors` (motor IDs), `lerobot-find-cameras`
and, optionally, `lerobot-find-port`. For anything that moves the follower, use only
`scripts/wo06_teleop_record.py`, which sends every command through `src/robot/safety.py`
(`So101Driver.step` in `src/robot/driver.py`: `safety.filter`, then `_write`).

---

## 0. Prerequisites 🟢 NO MOTION

**Physical checks.** Do these with the arms unpowered.

1. **Servo voltage vs supply.** Read the label on one follower servo and one leader servo. A 7.4 V
   STS3215 runs from the 5 V supply (`docs/ENVIRONMENT.md` Hardware). A 12 V STS3215 needs the 12 V
   supply that came with it. Never swap the leader and follower supplies. Measure the supply with a
   multimeter before the first connection, and **report the label text and the measured voltage**.
2. **E-stop.** Put an inline switch (or a switched power strip) on the follower's servo supply,
   within reach of the hand you don't use for the leader. Switching it off is the e-stop. It cuts
   torque instantly, and the arm then drops.
3. **Clamp** the follower base to the table. Clear the workspace: nothing within the arm's reach
   except the task objects.
4. **USB.** Connect each Waveshare board to the Mac directly, not through a hub carrying both
   boards. That way `find-port` sees exactly one port disappear.
5. **Camera permission.** System Settings → Privacy & Security → Camera: allow your terminal app.

**Software.** Use the main checkout, on this branch or on `main` once it merges.
`configs/real.yaml` gets edited in place during bring-up. Do bring-up on a branch of its own
(e.g. `git switch -c wo-06-hardware-values`) and commit those values through a PR. As of this
branch, the tests no longer read those values (`tests/real_fixture.py`).

```bash
brew install uv molten-vk ffmpeg@7
cd ~/Robotics/vla-rl-so101
source scripts/mac_env.sh
# uv sync is EXACT: it uninstalls every extra you leave out. This set = CI (dev, gpu) + robot.
# Add `--extra smolvla` only if you also want SmolVLA on the Mac. Never run a bare `uv sync --extra robot`.
uv sync --frozen --extra dev --extra gpu --extra robot
uv run python -c "import scservo_sdk, serial, cv2, grpc; print('robot deps ok')"
uv run pytest -q tests/test_safety.py tests/test_driver.py tests/test_units.py tests/test_camera.py tests/test_teleop.py tests/test_calibrate.py
```

- **Writes:** `.venv` only.
- **Expected:** `robot deps ok`, then all passed, 0 failed (61 at the time of writing; the count
  grows as tests are added).
- **Pass/fail:** any import error or test failure is a fail. Stop and send the agent the output.
- **Why 🟢:** nothing opens a serial port. The tests use fakes only (`tests/robot_fakes.py:1`).
- **Report back:** the servo label and the measured voltage, both arms, and the `pytest` summary line.

---

## 1. Find the follower's serial port 🟢 NO MOTION

Plug in the follower board's USB. Servo power isn't needed for this step.

```bash
uv run scripts/wo06_calibrate.py --preset real find-port --write
# when prompted: unplug the FOLLOWER board's USB, press ENTER, then plug it back in
```

- **Writes:** `robot.port` in `configs/real.yaml`, replacing `/dev/tty.usbmodemXXXX`.
- **Expected:** `The follower board is on /dev/tty.usbmodem<serial>. Plug the USB cable back in.`
  then `wrote robot.port in .../configs/real.yaml`.
- **Pass/fail:** `Expected exactly one port to disappear, got [...]` means more than one cable
  was unplugged, or the board isn't enumerating. Retry. If no `/dev/tty.usbmodem*` ever
  appears, the board needs the WCH CH34x macOS driver.
- **Why 🟢:** `find_available_ports` only lists `/dev/tty*` and opens no port
  (`LR/scripts/lerobot_find_port.py:30-39`). The wrapper only compares the two lists and edits
  the YAML (`scripts/wo06_calibrate.py:153-178`).
- **Report back:** the port name.

---

## 2. Assembly and motor IDs, one motor at a time (only if needed) 🟢 NO MOTION

Skip this step if the read-only scan below already shows `{1000000: [1, 2, 3, 4, 5, 6]}` for
an assembled, powered arm.

```bash
# read-only: pings every baud rate, writes nothing (servo power ON, full chain connected)
uv run python -c "from lerobot.motors.feetech import FeetechMotorsBus as B; print(B.scan_port('/dev/tty.usbmodem<FOLLOWER>'))"
```

If the IDs are missing or duplicated, or the arm isn't assembled yet, set them one motor at a time:

```bash
uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/tty.usbmodem<FOLLOWER>
# leader (find its port first: `uv run lerobot-find-port`, unplug/replug the LEADER board):
uv run lerobot-setup-motors --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem<LEADER>
```

At each prompt (`Connect the controller board to the '<motor>' motor only and press enter.`),
plug the board's 3-pin cable into ONLY that servo. The order is gripper (6), wrist_roll (5),
wrist_flex (4), elbow_flex (3), shoulder_lift (2), shoulder_pan (1). Keep the board powered,
press ENTER, and label the servo. Leader servos may have different gear ratios per joint, so keep
the kit's servo-to-joint labels. Assemble the arm (SO-101 assembly guide) after all IDs are set.

- **Writes:** servo EEPROM: ID and Baud_Rate (1,000,000).
- **Expected:** `'gripper' motor id set to 6` … `'shoulder_pan' motor id set to 1`. Afterwards,
  `scan_port` prints `{1000000: [1, 2, 3, 4, 5, 6]}`.
- **Pass/fail:** `Motor '<x>' ... was not found` means a cable or power problem. A model-number
  error means the wrong servo is connected. If any ID is duplicated after the scan, redo that
  motor alone.
- **Why 🟢:** `setup_motors` loops over the motors (`LR/robots/so_follower/so_follower.py:172-176`;
  leader `LR/teleoperators/so_leader/so_leader.py:134-138`). `setup_motor` disables torque
  (`LR/motors/motors_bus.py:607`) and writes only ID and Baud_Rate (lines 610-616). It never
  writes Goal_Position and never enables torque. `scan_port` only broadcast-pings
  (`LR/motors/motors_bus.py:552-574`). Never run `setup-motors` with more than one servo on the cable.
- **Report back:** the final `scan_port` output for each arm.

---

## 3. Follower calibration (always fresh) 🟡 TORQUE OFF, OWNER MOVES JOINTS BY HAND

Arm clamped, servo power on, USB in.

```bash
uv run scripts/wo06_calibrate.py --preset real calibrate --device follower
```

1. `Torque goes OFF now ... support the arm, then press ENTER`: hold the arm, press ENTER.
   If torque was on (it can be, after a crashed session), the arm goes limp now.
2. There is no "use provided calibration file" question. The wrapper always calibrates fresh,
   so there is no `c` to type. A previous `so101_follower.json` is copied to
   `so101_follower.<time>.bak.json` first.
3. `Move ... to the middle of its range of motion`: put EVERY joint at mid-travel, with the wrist
   straight, the gripper half open and wrist_roll centred (this pose becomes wrist_roll's zero).
   Press ENTER.
4. Sweep each joint EXCEPT wrist_roll slowly through its whole range, touching BOTH mechanical
   end stops. Close the gripper fully, then open it fully. Press ENTER.

- **Writes:** `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`
  (+ `.bak.json` of the old one). Servo EEPROM: Homing_Offset and Min/Max_Position_Limit.
- **Expected:** a live `NAME | MIN | POS | MAX` table during the sweep, then
  `Calibration saved to .../so101_follower.json`, `saved ...`, then the `check` table from step 4
  (printed automatically).
- **Pass/fail:** any `WARNING` line in the check table means the sweep was short, so recalibrate.
  An interrupted run (Ctrl-C) prints `... re-run calibrate`. The old file stays, but the motors no
  longer match it, and the driver will refuse to connect until you recalibrate.
- **Why 🟡 (torque never enabled):** the wrapper opens only the bus (`bus.connect()`,
  `scripts/wo06_calibrate.py:200`), disables torque (line 202), clears the cached file so
  LeRobot takes the fresh path (lines 207-212), and disconnects with torque off (line 219). It no
  longer calls `device.connect()`, whose `configure()` re-enables torque at the end of
  `torque_disabled()` (`LR/robots/so_follower/so_follower.py:105,156-158`;
  `LR/motors/motors_bus.py:663-678`). It never takes LeRobot's reuse branch, which wrote homing
  offsets with torque on (`so_follower.py:113-121`). The fresh path only reads positions during the
  sweep (`LR/motors/motors_bus.py:790-830`). Covered by `tests/test_calibrate.py`.
- `--reuse-file` writes the existing file to the motors instead, still with torque off. Use it
  only to restore a known-good calibration.
- **Report back:** the whole `check` table (step 4 output).

## 4. Calibration check against the URDF, then derive the safety limits 🟢 NO MOTION

```bash
uv run scripts/wo06_calibrate.py --preset real check                  # read the table first
uv run scripts/wo06_calibrate.py --preset real check --write-limits   # only when there are no WARNING lines
```

- **Writes:** `robot.safety.joint_limits_deg` in `configs/real.yaml` (with `--write-limits`).
- **Expected:** each recorded span within 10° of the URDF span (15° for the gripper):
  shoulder_pan ≈ 220°, shoulder_lift ≈ 200°, elbow_flex ≈ 193.7°, wrist_flex ≈ 190°, gripper ≈ 110°
  (`src/robot/units.py:68-75`). wrist_roll is fixed at 0..4095 and isn't checked. Then
  `effective safety limits (deg):`, with the gripper at `[-5.0, 95.0]`.
- **Pass/fail:** no `WARNING` lines. Recalibrate on any warning: the joint's zero would differ
  from the sim's (`src/robot/units.py:33-35`).
- **Why 🟢:** reads the JSON file only and never opens the port (`scripts/wo06_calibrate.py:231-275`).
  Limits only ever narrow the URDF limits minus the margin (`src/robot/safety.py:152-174`).
- **Rerun `--write-limits`** after any change to `robot.units` in step 5. The limits depend on
  `arm_sign`/`arm_offset_deg` (`src/robot/units.py:160-178`).
- **Report back:** the table and the effective limits.

## 5. Torque-off joint verification: signs, offsets, gripper, TCP 🟡 TORQUE OFF, OWNER MOVES JOINTS BY HAND

```bash
uv run scripts/wo06_calibrate.py --preset real verify     # Ctrl-C to stop
```

It first prints the checks below and asks `Torque goes OFF now ... support the arm, then press
ENTER`: torque can still be on after a crashed session, and the arm drops when it goes off. Hold
the arm, press ENTER. Then a line prints every 0.2 s, for example
`pan=    0.4  lift=   -1.2  elbow=    0.3  wflex=    2.0  wroll=    0.0  grip=   45.1  TCP=(0.385,-0.001,0.241) in`.
The values are canonical degrees, and the TCP is in metres in the robot base frame. Support the arm
throughout.

| Check | Do | Expected |
|---|---|---|
| zero | calibration middle pose | every arm joint within ~5° of 0 (`wroll` ~0 exactly) |
| shoulder_pan sign | from the middle pose, swing the arm toward the **robot's right** (−y; clockwise seen from above) | `pan` **increases**, TCP y decreases |
| shoulder_lift sign | tilt the upper arm so the gripper goes down (and slightly forward) | `lift` **increases**, TCP z decreases |
| elbow_flex sign | fold the forearm down | `elbow` **increases**, TCP z decreases |
| wrist_flex sign | tip the gripper down | `wflex` **increases**, TCP z decreases |
| wrist_roll | roll the wrist | only `wroll` changes. Its sign is checked visually in step 7b (`--with-arm`) |
| gripper | fully closed, then fully open | `grip` ≈ −10 closed, ≈ +100 open |
| isolation | move each joint alone ±30° | only that column changes |
| TCP | lift the gripper straight up / away from the base / to the robot's left | z / x / y increase |

The expected directions come from the sim's FK (`src/env/kinematics.py`) evaluated at the zero pose.
+20° of shoulder_pan moves the TCP −11.8 cm in y; +20° of shoulder_lift moves it −11.6 cm in z;
+20° of elbow_flex −9.9 cm in z; +20° of wrist_flex −5.3 cm in z.

- **Fixes:** a joint moving the wrong way → flip its entry in `robot.units.arm_sign`. A constant
  error → `robot.units.arm_offset_deg` (degrees). A reversed gripper → swap `gripper_closed_rad`
  and `gripper_open_rad` (`configs/real.yaml` block `robot.units`). Copy-paste edit that keeps the comments:
  `uv run python -c "from src.robot.yamlio import update_yaml_file as u; u('configs/real.yaml', ['robot','units','arm_sign'], [1,1,1,1,1])"`.
  Then rerun step 4 with `--write-limits`.
- **Also note the resting pose:** fold the arm into the pose it will rest in when teleop
  starts, and read its TCP. `in` is ideal. Torque is enabled in this pose in step 9.
- **Park pose (automatic parking is OFF by default; owner decision 2026-11-08):**
  `robot.park.enabled: false` in `configs/real.yaml`. Until you turn it on, every session
  (teleop, pilot, real eval) ends the same way: the follower holds, the script asks you to
  support the arm, ENTER turns torque off. Turn it on only after you have checked, in this step,
  both of the following:
  1. **A supported rest pose.** Fold the arm into the pose it should end in (`robot.park.qpos`;
     `null` means `deploy.home_qpos`, the sim rest pose `[0, -1.30, 0.60, 1.30, 0, 0]` rad). With
     torque off, it must rest there by itself: no sagging, no tipping, nothing on a hard stop,
     the gripper clear of the table. If the sim pose does not rest, use the verify line's
     degrees, converted to canonical radians, as `robot.park.qpos`. Only use a pose whose TCP is `in`.
  2. **A clear park path.** Move the arm by hand from a typical end-of-episode pose (low over the
     table, mid-workspace) to that rest pose. Nothing may be in the way: no table edge, cable,
     camera mount or grid. The park moves all joints at once, each towards its rest angle at
     <= 4 deg per step (a joint with less to travel arrives first): trace that motion by hand.

  Then enable it with a comment-preserving edit and report both checks (step 5 row in the
  reporting summary):
  `uv run python -c "from src.robot.yamlio import update_yaml_file as u; u('configs/real.yaml', ['robot','park','enabled'], True)"`.
  With it on, a healthy end parks through the safety filter, then turns torque off; a fault or
  an abort during motion never parks.
- **Writes:** nothing.
- **Pass/fail:** every row above as expected. `motors do not match the calibration file` → step 3.
- **Why 🟡:** asks you to support the arm first (`scripts/wo06_calibrate.py:293-294`), then port
  only, with no `configure()` (line 295). It explicitly disables torque (line 297), only does
  `sync_read('Present_Position')` (line 301), and disconnects with torque off (line 316). No
  Goal_Position write anywhere.
- **Report back:** one line at the middle pose, one line in the resting pose, and any sign or
  offset you changed.

---

## 6. Camera: index, then checkerboard intrinsics 🟢 NO MOTION

```bash
uv run lerobot-find-cameras opencv          # lists cameras, saves test frames to outputs/captured_images (git-ignored)
# set the USB webcam's index (index 0 is usually the MacBook's own camera):
uv run python -c "from src.robot.yamlio import update_yaml_file as u; u('configs/real.yaml', ['cameras','front','index'], 1)"
```

Print a 9×6-inner-corner checkerboard (10×7 squares), measure one printed square in mm, and tape
it flat to something rigid. Then:

```bash
uv run scripts/wo06_camera_intrinsics.py --preset real --capture 20 --square-mm <measured> --dry-run
#  vary distance, tilt ~30 deg each way, cover every image corner; ENTER per capture
uv run scripts/wo06_camera_intrinsics.py --preset real --images data/camera_calib/<stamp> --square-mm <measured>
```

- **Writes:** `cameras.front.index` and `cameras.front.intrinsics` in `configs/real.yaml`. Frames go to
  `data/camera_calib/<stamp>/` (git-ignored).
- **Expected:** `board found in N/20 images; reprojection rms <1 px`, then `fx=... fy=... full fov H x V deg`
  and `sim fov 40 deg -> crop window x[..] y[..] px`.
- **Pass/fail:** rms ≤ 1 px. There must be no `WARNING the sim camera's 40.0 deg view needs ...` line, because
  that line means the lens is too narrow (`src/robot/camera.py:102-109`). A 40° vertical fov needs fy ≲ 659 px at
  480 px height. A narrower webcam makes the policy pipeline raise, which blocks steps 7 and 9-10.
- **Why 🟢:** the scripts import no driver and open no bus. They only open the camera through
  `OpenCVFrameSource` (`scripts/wo06_camera_intrinsics.py:31-39,60`; `src/robot/camera.py:203-221`).
- **Report back:** the intrinsics output lines.

## 7. Camera placement and the sim/real match

**7a. Real-only overlay 🟢 NO MOTION**

```bash
source scripts/mac_env.sh                                           # MoltenVK: the sim renders on CPU
uv run scripts/wo06_camera_match.py --preset real                   # all 3 tasks, seed 0
uv run scripts/wo06_camera_match.py --preset real --task PickPlace --seed 3
```

Mount the camera at the sim pose that the script prints: `eye [0.555, 0.35, 0.495] m, looking at
[0.205, 0, 0] m`, in the robot base frame (+x forward from the base, +y the robot's left, z up,
origin at the shoulder_pan axis on the table; `configs/sim.yaml:24-30`). That is 0.555 m in front
of the base, 0.35 m to the robot's left, 0.495 m above the table, aimed at the centre of the
seen region, 0.205 m in front of the base. Place the objects where the sim panel shows them.
Adjust the camera until the table edges, the base and the objects overlap in the 50/50 blend panel.

**7b. Sim arm posed at the real joints 🟡 TORQUE OFF, OWNER MOVES JOINTS BY HAND**

```bash
uv run scripts/wo06_camera_match.py --preset real --with-arm --task PickPlace   # pose the arm by hand first
```

- **Writes:** `docs/assets/camera-match/<task>_seed<seed>_<time>.png` (git-ignored) and `.json`
  (colour stats, crop window, real joints; committed).
- **Expected:** `sim camera 'front_left_45': eye ... fov 40.0 deg`, then
  `saved docs/assets/camera-match/... {"mean_abs_diff": ...}`. In 7b the sim arm overlaps the
  real arm in the blend panel, wrist_roll included.
- **Pass/fail:** the objects, base and table edge overlap within a few pixels. In 7b the arms
  overlap. On this branch a posing failure is an error, not a silent rest pose
  (`scripts/wo06_camera_match.py:89-104`, `tests/test_camera_match.py`). Before this fix, the Mac
  CPU `SerialVecEnv` had no `.agent`, and `--with-arm` showed the rest pose.
- **Why 🟢 / 🟡:** without `--with-arm` no bus is opened. With it, `read_arm_joints` does
  `bus.connect()`, `sync_read('Torque_Enable')`, one `sync_read('Present_Position')` and
  `bus.disconnect(disable_torque=False)`, with no `configure()`, no torque change and no goal write
  (`scripts/wo06_camera_match.py:62-86`). The script does not turn torque off (that could drop an
  unsupported arm): if any servo is torqued it refuses with `follower torque is ON ...`. Then
  run step 5's `verify` (it asks you to support the arm before torque goes off) or power-cycle the
  servo supply, and retry. You pose the limp arm by hand.
- **Decision for `docs/STATUS.md`** (WO-06 benchmark): from the blend and the colour stats,
  decide whether the WO-01 randomization retrain is needed (WO-06 deliverable 5).
- **Report back:** the PNG and JSON paths, `mean_abs_diff`, a screenshot of one blend, and your
  call on the randomization retrain.

---

## 8. Leader: config, port, calibration, verify 🟡 TORQUE OFF, OWNER MOVES JOINTS BY HAND

Leader on its own supply (step 0), IDs set (step 2).

```bash
# 1. add the leader block (placeholder port; comments kept):
uv run python -c "from src.robot.yamlio import update_yaml_file as u; u('configs/real.yaml', ['teleop','leader'], {'type':'so101_leader','id':'so101_leader','port':'/dev/tty.usbmodemYYYY','calibration_dir':'~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader'})"
# 2. port (unplug only the LEADER board's USB when asked):
uv run scripts/wo06_calibrate.py --preset real find-port --device leader --write
# 3. calibrate (support prompt, middle pose ENTER, sweep all but wrist_roll stop-to-stop, trigger closed+open, ENTER):
uv run scripts/wo06_calibrate.py --preset real calibrate --device leader
# 4. verify against the follower (same pose on both arms, by eye):
uv run scripts/wo06_calibrate.py --preset real verify --device leader
```

- **Writes:** `teleop.leader` in `configs/real.yaml`, and
  `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader.json` (+ `.bak.json`).
  Leader EEPROM: homing and limits.
- **Expected:** `Calibration saved to .../so101_leader.json`, `saved ...`. In verify, put the leader
  in the same pose as the follower (verified in step 5) and every joint agrees within ~5°.
  Trigger closed ≈ −10°, open ≈ +100°.
- **Pass/fail:** leader verify follows the same direction rules as step 5. Teleop applies the
  follower's `robot.units` to the leader (`main` in `scripts/wo06_teleop_record.py`, `UnitMap.from_config`),
  so a leader-only sign or offset difference can't be fixed in the config. Recalibrate whichever
  arm disagrees.
- **Why 🟡:** the leader never receives a goal position: `get_action` is `sync_read` only
  (`LR/teleoperators/so_leader/so_leader.py:140-147`), and its `configure()` only disables torque
  (lines 128-132). The calibrate/verify paths are the same torque-off bus-only paths as steps 3 and 5
  (`scripts/wo06_calibrate.py:181-220,278-316`). `_make_device` refuses the `YYYY` placeholder port
  (`scripts/wo06_calibrate.py:128-150`). Leader `verify` skips the support prompt.
- **Report back:** one leader verify line next to the follower verify line for the same pose.

---

## 9. One discarded powered rehearsal 🔴 POWERED MOTION: OWNER IN PERSON

This is the first command that drives the arm. Prerequisites: steps 3-8 pass,
`joint_limits_deg` and the intrinsics are written, and you are present with a hand on the
follower's power switch.

```bash
source scripts/mac_env.sh
uv run scripts/wo06_teleop_record.py --preset real --task Push --rehearsal
```

What happens and what you do:

1. **At connect, torque comes ON** and the follower holds wherever it rests. It should not
   move: Goal := Present before every torque enable (`install_hold_on_torque_enable` in
   `src/robot/driver.py`). **If any joint twitches, cut power.** Before that, with torque still
   off, `check_rest_pose` refuses to enable torque if any joint rests more than 1 deg outside
   its calibrated range (calibration file and servo EEPROM Min/Max_Position_Limit, which must
   agree) or more than 6 deg outside the safety joint limits (`robot.safety.torque_enable_tol_deg`).
   It prints `TORQUE ENABLE REFUSED: torque stays OFF. <joint> rests at ...`. Then move that
   joint back by hand and rerun, or recalibrate (step 3) if the joint is really inside its
   range. It also refuses when a servo's stored `Homing_Offset` (EEPROM address 31) differs
   from the calibration file's `homing_offset`, even if the ranges match:
   `<joint>: servo Homing_Offset <n> differs from the calibration file's homing_offset <m>`.
   Every reading of that servo would be shifted, so do not work around it: redo step 3, then
   step 4 `check --write-limits`. The connect log also prints the servos' firmware
   `Max_Temperature_Limit` next to the monitor's trip temperature. Unverified on hardware.
2. `[Push 1/1] arrange the scene, hold the leader, ENTER (q = end):`. Hold the leader mid-workspace:
   gripper above the table, every joint away from its stops, trigger mid-way. Press ENTER.
   **Bounded torque-on wait (proposal, on by default):** every prompt answered while the
   follower holds under torque (this one, `keep this episode ...`, `type 'resume' ...`) reads
   the servos' temperature and current before it waits and again after you answer; the final
   support prompt reads them once before it waits (torque goes off right after it). If you do not answer within `teleop.prompt_torque_timeout_s` (60 s), it
   prints `No answer for 60 s with torque ON. SUPPORT THE ARM NOW ... Torque goes OFF in 10 s
   (ENTER = now) and the session ends.` Support the arm: after `teleop.prompt_support_grace_s`
   (10 s) torque goes off and the session ends (saved episodes are kept, an unsaved one is
   discarded). The wait is a `select()` on the terminal in the same single thread; nothing
   reads the servo bus in the background. `null` restores the old wait-forever behaviour (not
   recommended).
3. **Sync (not recorded):** the follower eases onto the leader's pose, clipped to the safety limits,
   at ≤ 4°/step and 60°/s (`configs/real.yaml` keys `robot.safety.max_joint_step_deg` and
   `max_velocity_deg_s`). A fully closed or open trigger, or a leader
   joint at its stop, does not block this, also when the follower starts outside a limit (gripper
   resting closed, arm folded on its stops): like the safety filter, the sync then accepts the
   joint where the filter holds it (`Session.sync_to_leader` in `scripts/wo06_teleop_record.py`). If the
   script prints `safety rejects the leader pose (...)`, move the leader's gripper into
   `workspace_box_m`. If it prints `not in sync after 10 s ...`, it lists the joints that are lagging.
   Ctrl-C prints `sync aborted`. In every case the follower holds, and you get the prompt again.
4. `recording (30 s max, Ctrl-C to end) ...`. Move slowly. **Bring the follower low over the table,
   then press Ctrl-C once.** The current control tick finishes, `episode ended` prints, and the
   follower holds that pose (the `arm.hold()` at the end of the recording loop in `_episodes`).
5. `keep this episode (N frames)? [Y]es / [n]o, redo / [q]uit:` → type `q`.
6. With `robot.park.enabled: false` (the default for this first rehearsal): the follower holds,
   and you get `... Put a hand under the forearm ..., then press ENTER: torque goes OFF`.
   Support the arm, then press ENTER. The arm goes limp into your hand. Only once step 5's
   rest-pose and park-path checks have passed and you set `robot.park.enabled: true`:
   `ending the session: the follower PARKS (it moves; Ctrl-C stops it), then torque off`, the
   follower drives to `robot.park.qpos` at <= 4 deg/step, then `follower parked; torque OFF`.
   Keep the path clear; Ctrl-C during the park stops it. If the park cannot finish (Ctrl-C,
   timeout, a rejected pose, a deadman stop), you get `park did not complete: the follower
   holds`, then the support prompt above. After a deadman stop without `resume`, the session
   never parks: it goes straight to the hold and support prompt.

- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop-rehearsal/` (git-ignored, replaced every
  rehearsal, never pushed). The real dataset is untouched.
- **Expected:** possibly `NOTE <task>: recording the protocol instruction ...`. No
  `WARNING camera config mismatch`. Log lines print as `<LEVEL> <logger>: <message>`, e.g.
  `WARNING src.robot.driver: holding at measured pose (deg): [...]` after each sync abort and
  episode. A `WARNING src.robot.safety: safety joint_limit gripper: -10.00 -> -5.00 deg (...)`
  every tick while the trigger is fully closed is expected: the follower stops at its safety floor.
  The -5 deg gripper floor (URDF -10 deg + the 5 deg margin) stays for this first rehearsal
  (owner decision 2026-11-08). Note in the report whether a cube still slips at -5 deg.
  The per-tick `INFO src.robot.safety: safety step_clip ...` / `velocity_cap` lines (the rate
  limiter at work during sync) print only with `--verbose`. Finally
  `saved 0 episodes to .../vla-rl-so101-real-teleop-rehearsal` (or 1 if you answered Y).
- **Pass/fail:** no joint jumps at connect or at sync. The follower tracks the leader smoothly
  in the right directions. It stops when you stop. The session ends through step 6 above. Any
  `STOPPED: ...` from the deadman (stale frame, stall > 20°) means the follower holds, and the
  prompt `type 'resume' ... else end` appears. A `serial error` means torque off, and the arm drops.
  A `SERVO FAULT (over-temperature ... / over-current ...)` also means torque is off **at once**
  (no hold, no park) and the arm drops. Support it, switch off the servo supply, let the servos
  cool for at least 10 minutes, and find the cause (a blocked joint, a squeezing gripper, a
  long hold). The monitor reads `Present_Temperature` / `Present_Current` every 4th control
  tick, so each register is read every 533 ms. It trips at 55 C, or at > 1500 mA sustained for
  1 s (`robot.servo_monitor`). These thresholds are **provisional** (owner decision 2026-11-08:
  keep them for the first sessions). Note the temperatures after a session and send them, so
  they can be confirmed or changed.
- **Torque off unconfirmed:** if a torque-off write fails (USB cable, bus, a servo), the
  script prints `TORQUE OFF UNCONFIRMED (...). The servos may still be energized ... Switch OFF
  the servo supply NOW at the inline power switch (the independent power cutoff) ...`. Do that
  at once, with a hand under the arm. From that moment the process refuses every goal-position
  and torque-enable write (nothing can move it, a typed `resume` does not clear it), and it
  ends. In the pilot and the real eval the trial record, the shutdown result and the session
  end say `torque_off_unconfirmed`. Leave the supply off until the cause is found.
- **Why 🔴:** `So101Driver.connect` enables torque through LeRobot's `configure()`
  (`self.follower.connect(calibrate=False)` in `So101Driver.connect`). Each tick, the safety
  filter runs and then the bus write follows (`So101Driver.step`: `safety.filter`, then
  `_write`). The sync and recording loops drive the follower (`Session.sync_to_leader` and the
  recording loop in `_episodes`, `scripts/wo06_teleop_record.py`).
- **Report back:** the full terminal log (or the `safety`/`STOPPED` lines), and whether
  anything moved unexpectedly.

## 10. Five teleop sanity episodes per task (PickPlace, Push, Stack) 🔴 POWERED MOTION: OWNER IN PERSON

Prerequisite: step 9 went cleanly. Objects: 3 cm red and blue cubes, a bowl, and a Push
target mark, placed on the seen-region positions from step 7 (centre 0.205 m in front of the
base, ±5 cm in x, ±7 cm in y; `src/eval/protocol.py:63`).

```bash
source scripts/mac_env.sh
uv run scripts/wo06_teleop_record.py --preset real              # 3 tasks x 5 episodes, 30 s max each
```

For a private Hub copy, decide before you start: run `uv run hf auth login` once, then add `--push`
to the command above. The upload runs after the session, only if at least one episode was saved,
and always private (`dataset.push_to_hub(private=True)` at the end of `main` in
`scripts/wo06_teleop_record.py`). Never use `lerobot-record`, which
uploads publicly by default.

For each episode: arrange the scene, then hold the leader mid-range (trigger mid-way) and press ENTER.
Do the task, **finish with the follower low**, then press Ctrl-C (or wait 30 s) and answer `Y`, `n`
(redo) or `q` (end the session). After `Y` the script prints `saving episode (encoding video, a few
seconds; Ctrl-C is ignored) ...`: wait for the next prompt. If saving fails it prints
`saving failed (...); episode discarded, redo it`, the follower keeps holding, and you redo it. Between episodes the follower holds its last pose under
torque. It does not follow the leader at the prompts, so moving the leader then does nothing.
Watch servo heat. If the servos feel hot, end the session with `q`, power off, and let them cool.
The servo monitor (step 9) cuts torque by itself at its thresholds. Do not rely on it alone:
its thresholds are unconfirmed. It does not read while a prompt waits. At every prompt where
the follower holds under torque (scene, keep, resume) it reads temperature and current once
before the wait and once after you answer, before the follower moves, and the wait is capped at
`teleop.prompt_torque_timeout_s` (60 s). The final support prompt reads them once before its
wait. A current found over the limit is re-read after 1 s at once, so a sustained over-current
trips before the prompt waits.

- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop/`, a LeRobotDataset v3.0 at 15 fps
  with 224×224 video and float32[6] canonical-radian state/action. The per-frame `task` is the
  protocol instruction (`pick up the red cube and place it in the bowl`, `push the red cube to the
  target`, `stack the red cube on the blue cube`; `src/eval/protocol.py:28-32,97-98`).
  `meta/vla_rl_so101.json` records the protocol version and hash and the `teleop.instructions`
  strings, which differ for PickPlace and Push.
- **Expected:** 15 × `[<Task> k/5] ...`, then `saved 15 episodes to .../data/real_teleop/vla-rl-so101-real-teleop`.
- **Pass/fail:** 15 saved, with no unexpected motion. There is no resume: rerunning with `--overwrite`
  deletes the saved episodes (the `--overwrite` / `shutil.rmtree(root)` branch of `main` in
  `scripts/wo06_teleop_record.py`). Copy the folder aside
  before any rerun.
- **Why 🔴:** same path as step 9. The stored `action` is the safe command actually sent
  (`ACTION_KEY: out.safety.command` in the recording loop of `_episodes`).
- **Report back:** the `saved N episodes` line, per-task notes (success by eye, any
  `STOPPED`/`REJECTED` lines), and the Hub URL if you pushed.

---

## Ending any session safely

- **Teleop (🔴):** end only through `q` or a single Ctrl-C at a prompt. The script then holds,
  asks you to support the arm, and turns torque off when you press ENTER (`end_session` in
  `scripts/wo06_teleop_record.py`; normal exit of the driver's `with` block). With
  `robot.park.enabled: true` (only after step 5's checks) it parks first (step 9, item 6); a
  park that cannot finish, or a deadman stop, falls back to the hold and support prompt. If no
  one answers a prompt for 60 s with torque on, it asks you to support the arm and turns torque
  off after 10 s (step 9, item 2). A servo fault or a serial error has already turned
  torque off: support the arm. Ctrl-C during the sync aborts it and holds. During recording it ends
  the episode, not the session. Outside a prompt Ctrl-C never interrupts a servo-bus or dataset
  call: it is a flag checked between control ticks (`ctrl_c_as_flag` in
  `scripts/wo06_teleop_record.py`), and
  it is ignored while an episode saves or the follower settles into its hold. Ctrl-C at the
  "support the arm" prompt is ignored too (`Ctrl-C ignored: the follower still holds ...`): only
  ENTER, or the bounded wait running out (60 s with torque on, then the 10 s grace), turns
  torque off, so support the arm first, or cut the servo supply. Any other error (the dataset,
  the image pipeline, a bug) also holds the follower, discards the episode and shows the same
  support prompt before the script exits with the error.
- **Pilot and real eval (`scripts/wo07_pilot.py`, `scripts/wo07_real_eval.py`):** their prompts
  (move home, place the objects, score the trial, support the arm at the end) wait **without a
  time limit and without servo reads** while the arm holds under torque (accepted gap,
  `docs/STATUS.md`). Never leave the arm at one of those prompts: stay at the arm with a hand
  on the inline switch, and switch the supply off if you must step away.
- **If anything looks wrong at any moment, or the script prints `TORQUE OFF UNCONFIRMED`:**
  switch off the follower's servo supply at the inline power switch. Torque goes off instantly
  and the arm drops, so keep it low.
- **After each session:** the command has exited, the arm rests on the table, the servo supply
  is OFF (the switch), and then the USB comes out. Never leave the follower powered and torqued
  unattended.
- **Torque-off steps (🟡):** Ctrl-C stops `verify`, and every path disconnects with torque off
  (`scripts/wo06_calibrate.py:219,316`).

## Reporting summary (send to the agent)

| After | Send |
|---|---|
| 0 | servo label + measured supply voltage (both arms), `pytest` summary |
| 1 | follower port |
| 2 | `scan_port` output per arm |
| 3-4 | `check` table + effective limits |
| 5 | verify lines (middle pose, resting pose) + any `robot.units` change; rest pose supported with torque off (y/n), park path clear (y/n), whether you set `robot.park.enabled: true` |
| 6 | intrinsics lines (fx, fy, rms, crop window, any WARNING) |
| 7 | camera-match PNG/JSON paths, `mean_abs_diff`, retrain decision |
| 8 | leader vs follower verify lines for the same pose |
| 9 | rehearsal log (incl. the `Max_Temperature_Limit` line, any `TORQUE ENABLE REFUSED` / `SERVO FAULT` / `TORQUE OFF UNCONFIRMED`), end-of-session servo temperatures, whether a cube slips at the -5 deg gripper floor, anything unexpected |
| 10 | `saved N episodes`, per-task notes, Hub URL if pushed |
