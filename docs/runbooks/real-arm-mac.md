# Real-arm runbook (Mac, zsh): rehearsal, sanity episodes, 50 PickPlace demos

Copy-paste commands for the owner's Mac, in the order they run. Each block has its gate, what it
does, what to expect and what to send back. Background and the reasons behind each step:
[`docs/bringup/WO-06-checklist.md`](../bringup/WO-06-checklist.md) (steps 9 and 10, "Ending any
session safely") and [WO-07](../work-orders/WO-07-real-demos-baseline.md).

**Status (2026-11-12).** Sections C, D and E are **ON HOLD**: powered motion waits for the owner's
in-person go, and the demos also wait for the real-0.1 freeze (PR #32: measurements, bowl check,
freeze, regenerated grids). Sections A and B open no port and no camera except where a line says
the owner runs it.

Every flag below was checked against the argparse of the scripts at commit `3f1eceb` on
`feat/smolvla-real`, and every no-device command was run there. The powered commands have never
run on the arm: their expected output is what the code prints, not observed behaviour.

**One rule for every hardware session.** Start it as `.venv/bin/python scripts/...` directly in
Terminal: in the foreground, no `| tee`, no `< file`, no `&`, no `uv run`. Under `uv run` a
Ctrl-Z also stops `uv`, and the session then ends. The scripts refuse a pipe or a background job
before anything opens. To keep a log, use Terminal's Shell > Export Text As... after the session.

---

## A. Every new Terminal window (no motion)

Gate: none. Run these in each new window before anything else in this file.

```zsh
cd ~/Robotics/vla-rl-so101
git merge-base --is-ancestor 3f1eceb HEAD && echo "checkout has the runbook's commands"
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
source scripts/mac_env.sh
brew list --versions uv molten-vk ffmpeg@7
# uv sync is EXACT: it removes every extra you leave out. These four = `make setup`.
uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla
```

- **Expected:** `checkout has the runbook's commands`, three `brew` version lines, and `uv sync`
  finishes with no error. If the first line prints nothing, the checkout is older than this
  runbook: switch to `feat/smolvla-real`, or to a branch made from it (e.g. the bring-up branch).
- `scripts/mac_env.sh` sets `VK_ICD_FILENAMES` (MoltenVK), `DYLD_FALLBACK_LIBRARY_PATH`
  (Homebrew's keg-only `ffmpeg@7`, which torchcodec 0.2.1 needs) and `MVK_CONFIG_LOG_LEVEL`. It
  keeps a value that is already set. If a wrong value is already set, the preflight below says
  so, and the fix is `export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib`.

### A1. Decoder preflight (must PASS before any recording)

```zsh
.venv/bin/python scripts/preflight_decoder.py --preset real
```

It loads torchcodec and FFmpeg, checks the PyAV `libsvtav1` encoder and the camera convention in
`configs/real.yaml`, then writes a throwaway two-camera dataset in a temp folder through the
recorder's own code. It decodes every frame of both clips with `verify_episode_video` and deletes
the folder. It opens no port and no camera. It takes about 10-40 s.

- **Expected (this Mac, 2026-11-12):**
  ```
  [torchcodec] OK: torchcodec 0.2.1 from .../.venv/lib/python3.11/site-packages/torchcodec
  FFmpeg 7.1.5 (libavcodec 61.19.101)
    loaded /opt/homebrew/Cellar/ffmpeg@7/7.1.5_3/lib/libavcodec.61.19.101.dylib
    ...
  [encoder] OK: PyAV 15.1.0 (bundled libavcodec 61.19.101) encodes libsvtav1; ...
  [config] OK: front 640x480 at 30 fps, wrist 640x480 at 30 fps; dataset 15 fps, libsvtav1, backend torchcodec
  [record + decode] OK: 2 episodes x 20 frames x 2 cameras (...) written in ... s; verify_episode_video: every frame of every clip decodes; ...
  PASS: torchcodec + FFmpeg decode the recorder's two-camera clips
  ```
  The exit code is 0. Expected noise: `objc[...]: Class AVFFrameReceiver is implemented in both
  ...` (Homebrew FFmpeg, PyAV and OpenCV each bring their own `libavdevice`), and the `Svt[info]`
  encoder banner.
- **FAIL:** the line says what failed, and the next line is `fix:`. Do what it says, then rerun.
  Without `DYLD_FALLBACK_LIBRARY_PATH` it prints `[torchcodec] FAIL: torchcodec cannot load
  FFmpeg ...` and `fix: export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib ...`
  (exit code 1). Without `ffmpeg@7` it says `brew install ffmpeg@7`.
- **Why it matters:** the recorder loads torchcodec only after the first demo has been saved
  (finalize and reopen). A missing FFmpeg would otherwise first show up with the arm under torque,
  after a recorded episode.
- **Send back:** the `PASS` / `FAIL` line, plus the `FFmpeg` line.

### A2. Mac power

Plug the Mac into power. In a **second** Terminal tab, run `caffeinate -dims` and leave it
running until the session ends (Ctrl-C there). Do not wrap the session command in
`caffeinate`: the wrapper would share the session's job, like `uv run`. `pmset -g batt` should
say `'AC Power'`.

### A3. Ports and cameras (owner runs; only after a re-plug or a reboot changed something)

```zsh
ls /dev | grep usbmodem                                                        # lists only, opens nothing
.venv/bin/python scripts/wo06_calibrate.py --preset real find-port --write                   # follower (unplug/replug when asked)
.venv/bin/python scripts/wo06_calibrate.py --preset real find-port --device leader --write   # leader (needs the step 8 leader block)
.venv/bin/lerobot-find-cameras opencv          # OPENS every camera; test frames go to outputs/captured_images
```

Set a camera index that changed (checklist step 6), e.g.
`.venv/bin/python -c "from src.robot.yamlio import update_yaml_file as u; u('configs/real.yaml', ['cameras','front','index'], 1)"`.
Every recording sitting also asks you to confirm the camera roles on a labelled frame before
torque goes on (section C).

### A4. Camera controls recorded as locked (reads the config only)

The UVC tool sets exposure, white balance and focus. You write the locked values into
`configs/real.yaml` `cameras.<name>.controls` (checklist step 7 item 4). After a replug or a
reboot, re-apply them in the tool if it reset them. Then check:

```zsh
.venv/bin/python -c "from src.config import load_config as L; from src.robot.convention import camera_controls_problems as P; print('\n'.join(P(L('real'))) or 'camera controls: locked')"
```

- **Expected:** `camera controls: locked`. Until then it lists each missing value, e.g.
  `cameras.front.controls.exposure is null: not locked / not recorded yet`. The demo schedule
  refuses while any line is listed.

---

## B. Gates before any powered motion (sections C, D, E)

Every box must be ticked. Run the verification command in the same window as section A.

| # | Gate | How to verify |
|---|---|---|
| 1 | Checklist steps 0-8 done, and their rows of the reporting summary sent to the agent | `docs/STATUS.md` records them |
| 2 | The inline switch on the follower's **12 V** supply is in reach of your free hand. Off = e-stop (the arm drops) | by hand: switch it off and on once, torque off |
| 3 | Follower clamped; workspace clear; nobody else in reach; supplies labelled FOLLOWER 12 V / LEADER 5 V | by eye; step 2b `voltage` said `voltage check: OK` |
| 4 | Calibration, limits, intrinsics, leader, camera controls in the config | the command below: every line `OK` |
| 5 | Drift check this session (torque off; support the arm) | `.venv/bin/python scripts/wo06_calibrate.py --preset real verify` at the calibration middle pose: every arm joint within ~5 deg of 0 (Ctrl-C to stop) |
| 6 | Decoder preflight `PASS` in this window | A1 |
| 7 | Hands: one on the leader, the other near the inline switch; a hand near the follower's forearm whenever a session ends | by you |
| 8 | **The owner's in-person go for this session** | by you |

```zsh
.venv/bin/python - <<'EOF'
from pathlib import Path
from src.config import load_config
from src.eval import protocol_real as PR
from src.eval.bowl_fit import bowl_box_problem
from src.robot import convention as conv
c = load_config("real")
r, cams, lead = c["robot"], c["cameras"], (c["teleop"].get("leader") or {})
cal = lambda d, i: bool(d and i) and (Path(d).expanduser() / f"{i}.json").is_file()
rows = [
    ("follower port set (step 1)", "XXXX" not in str(r["port"])),
    ("follower calibration file (step 3)", cal(r["calibration_dir"], r["id"])),
    ("safety joint limits written (step 4)", bool(r["safety"]["joint_limits_deg"])),
    ("front + wrist intrinsics (step 6)", all(cams[n].get("intrinsics") for n in conv.CAMERA_NAMES)),
    ("camera controls locked (step 7)", not conv.camera_controls_problems(c)),
    ("leader configured, port set (step 8)", bool(lead) and "YYYY" not in str(lead.get("port"))),
    ("leader calibration file (step 8)", cal(lead.get("calibration_dir"), lead.get("id"))),
    ("config in the recording convention", not conv.check_config(c)),
]
for name, ok in rows:
    print(f"{'OK     ' if ok else 'MISSING'} {name}")
for p in conv.camera_controls_problems(c) + conv.check_config(c):
    print(f"        - {p}")
print(f"INFO    robot.park.enabled: {r['park']['enabled']}")
print(f"INFO    demo schedule (section E): {'ready' if not (bowl_box_problem(c) or PR.freeze_problem()) else 'REFUSES until the real-0.1 freeze lands'}")
EOF
```

- **Expected before C:** every line `OK`. On the placeholder config shipped in the repo, every
  line except the convention line says `MISSING`, and the last line says `REFUSES until the
  real-0.1 freeze lands`. The last line must say `ready` before section E; C and D do not need it.
- The scripts refuse some of these on their own: `No leader arm configured ...` (exit 2) with no
  leader, and `leader has no calibration file ...`. Missing limits or intrinsics are not refused
  in code. The checklist requires them before step 9, and this table is the check.
- **Send back:** the output, plus the step 5 `verify` line at the middle pose.

---

## C. Step 9: one discarded powered rehearsal 🔴 POWERED: ON HOLD until the owner's in-person go

Gate: every row of section B. Table: **empty**, no objects (the freeze has not landed, and the
rehearsal records camera frames). Servo power on, both USB cables in.

```zsh
.venv/bin/python scripts/wo06_teleop_record.py --preset real --task Push --rehearsal
```

What you see, in order:

1. `NOTE Push: recording the protocol instruction 'push the red cube to the target' (teleop.instructions says 'push the cube to the target')`,
   then `recording in so101-real-obs-action 1.1 (src/robot/convention.py, sha256 c387f175...)`.
   No `WARNING: this process ... shares its job with a launcher` (if it appears, you started it
   under `uv run` or a wrapper: answer `no` at the camera check below, which ends the run before
   torque, and restart as above).
2. The leader connects (its torque stays off). Then Preview opens
   `data/real_teleop/vla-rl-so101-real-teleop-rehearsal/meta/camera_check_1.png`, the script
   prints `FRONT must show the table, the grid and the arm from the front, about 45 deg above`
   and `WRIST must show the gripper jaws close up (the module on the gripper)`, and asks
   `type 'yes' if every label matches its view:`. Type `yes` only if both match. Anything else
   ends the run before torque (`camera roles not confirmed ...`): swap `cameras.<name>.index` (A3).
3. **Torque comes ON** and the follower holds where it rests. **If any joint twitches, switch off
   the supply.** The log shows `servo firmware Max_Temperature_Limit (C): ...; monitor trips at 55 C`.
   A refusal here is `TORQUE ENABLE REFUSED: torque stays OFF. <joint> rests at ...`: move that
   joint back by hand and rerun, or recalibrate (checklist step 9 item 1).
4. `[Push 1/1] arrange the scene, hold the leader, ENTER (q = end):`. Hold the leader
   mid-workspace (gripper above the table, joints away from their stops, trigger mid-way), then
   press ENTER. The follower eases onto the leader (not recorded).
5. `recording (30 s max, Ctrl-C to end) ...`. Move slowly, bring the follower low over the
   table, then press **Ctrl-C once**: `episode ended`, and the follower holds.
6. `keep this episode (N frames)? [Y]es / [n]o, redo / [q]uit:`. Type `q`.
7. `Ending the session. The follower holds its pose under torque. Put a hand under the forearm
   ...`. Support the arm, then press ENTER. Torque goes off.
8. `saved 0 episodes to .../data/real_teleop/vla-rl-so101-real-teleop-rehearsal`.

**Stop keys.** Ctrl-C ends the leader sync or the episode (the follower holds). `q` at any prompt
ends the session. The inline switch is the hard stop: the arm drops. Ctrl-Z only prints
`Ctrl-Z ignored ...`. Never press Ctrl-\ (it kills the process and leaves the servos holding
under torque). With no answer for 60 s while torque is on, you get `No answer for 60 s with
torque ON. SUPPORT THE ARM NOW ...`, and torque goes off 10 s later. A `SERVO FAULT` or a `serial
error` turns torque off at once: support the arm. `TORQUE OFF UNCONFIRMED`: switch off the supply
now.

- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop-rehearsal/` (git-ignored; replaced
  every rehearsal; never pushed: `--push` is refused with `--rehearsal`). The real dataset is
  untouched.
- **Send back:** the terminal text (Shell > Export Text As...), the `Max_Temperature_Limit` line,
  any `TORQUE ENABLE REFUSED` / `STOPPED` / `SERVO FAULT` / `TORQUE OFF UNCONFIRMED` line, how hot
  the servos feel afterwards, and anything that moved unexpectedly. No script prints servo
  temperatures after a teleop session (see "Between sessions").

---

## D. Step 10: teleop sanity episodes 🔴 POWERED: ON HOLD until the owner's in-person go, and after the real-0.1 freeze

Gates: section C went cleanly, every row of section B, and the freeze has landed. The code does
NOT enforce the freeze for these episodes: the gate is you. Required now, per the checklist:
**5 episodes each of PickPlace, Push and Stack (15).** These are sanity episodes, never training
data. WO-07's deferral of Push and Stack applies to the demos and the evaluation, not to these
episodes. Objects: the red cube, the blue cube (Stack base), the white cup on its frozen bowl
mark, and the 2-inch black disc (Push target). The PickPlace cube goes only on the five seen
slots B2 B5 E2 E5 C4. Push and Stack objects stay inside the seen region. Never the purple or
green cube.

```zsh
.venv/bin/python scripts/wo06_teleop_record.py --preset real
```

- The camera-role check and torque on are as in C. Then 15 prompts,
  `[PickPlace 1/5] arrange the scene, hold the leader, ENTER (q = end):` ... `[Stack 5/5] ...`.
  For each: arrange the scene, press ENTER, do the task, **finish with the follower low**, press
  Ctrl-C (or wait 30 s), then `Y` (keep), `n` (redo) or `q` (end). After `Y`: `saving episode
  (encoding video, a few seconds; Ctrl-C is ignored) ...`. Wait for the next prompt. Ending:
  support prompt, ENTER, `saved 15 episodes to .../data/real_teleop/vla-rl-so101-real-teleop`.
- **No resume** for these episodes. If the folder exists, the script refuses with
  `... exists; pass --overwrite to replace it, or --resume to continue`, and `--overwrite`
  deletes it. Never pass `--push` (a Hub upload is a separate owner decision).
- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop/`, a LeRobotDataset v3.0 at 15 fps:
  `observation.images.front` and `observation.images.wrist` (480 x 640 AV1 video), state and
  action float32[6] canonical radians, `timing_s`, and `meta/vla_rl_so101.json`.

**Then, before section E: move the sanity episodes out of the demos' folder.** The demo schedule
writes to the same `data/real_teleop/vla-rl-so101-real-teleop` path. With this folder still
there, it would refuse, and `--overwrite` would delete the sanity episodes. Then check them:

```zsh
mv data/real_teleop/vla-rl-so101-real-teleop data/real_teleop/vla-rl-so101-real-teleop-sanity
.venv/bin/python scripts/preflight_decoder.py --preset real --dataset data/real_teleop/vla-rl-so101-real-teleop-sanity --episodes 15
```

- **Expected:** `[dataset] OK: 15 episodes; both cameras decode across every frame; sanity / free
  teleop recording, session <UTC stamp>-wo06-teleop-<hex>, sittings ended ['supported']`, then `PASS: ...`.
  The teleop recorder does not decode its saved episodes itself: this is the decode check.
- **Send back:** `saved N episodes`, the `[dataset]` line, per-task notes (success by eye, any
  `STOPPED` / `REJECTED` line), and whether a cube slipped at the -5 deg gripper floor.

---

## E. The 50 scheduled PickPlace demos (WO-07) 🔴 POWERED: ON HOLD until the owner's in-person go; REFUSED in code until the freeze lands

Gates, all of them:
1. The real-0.1 freeze is committed (`MEASURED` filled, `FREEZE_STATUS` frozen, the bowl inside
   the safety box). The last line of the section B command says `ready`.
2. The regenerated grid is printed at 100% (each 100 mm scale bar measures 100 mm), aligned to the
   base `+`, and the bowl centre marked on the table (A3: the bowl line, 100 mm left of its end
   tick; tabloid2: the BOWL CENTRE cross on sheet 2). Today the committed PDFs are pre-freeze:
   `.venv/bin/python scripts/make_placement_grid.py --preset real --check` exits 1 (`... is
   stale`) until the freeze regenerates them. Do not print them before then.
3. A4 prints `camera controls: locked`. Sections C and D were clean, and the sanity folder has
   been moved aside (D).
4. Every row of section B, plus a spare 12 V gripper servo on hand (hardware floor).

### E1. See the schedule (no motion; reads code only)

```zsh
.venv/bin/python - <<'EOF'
from src.eval import protocol_real as PR
print(PR.citation())
for d in PR.demo_schedule():
    print(f"{d['demo_idx'] + 1:2d}/50  {PR.placement_prompt(d)}")
EOF
```

There are 50 lines, 10 per slot (B2 B5 E2 E5 C4), in one seeded order (`DEMO_SEED`). Each line is
the exact text the recorder prompts with, e.g. at this head
` 1/50  place the red cube centred on slot E2 (turned 15 deg CW); the bowl stays on its fixed mark (x 0.195, y +0.205 m)`.
Rerun it after the freeze: the bowl coordinates in the text follow `BOWL_XY_M`, and the citation
hash changes. Nothing is drawn at run time. A redo, a crash or a resume never changes which slot
and yaw come next.

### E2. Record, in sittings

```zsh
# first sitting: demos 1-10 (a new dataset; refuses if the folder exists)
.venv/bin/python scripts/wo06_teleop_record.py --preset real --task PickPlace --demo-schedule --episodes 10
# every later sitting: continue at the next unrecorded demo (10 more, or omit --episodes for all that remain)
.venv/bin/python scripts/wo06_teleop_record.py --preset real --task PickPlace --demo-schedule --resume --episodes 10
```

- Each prompt names the slot and the yaw:
  `[PickPlace 1/10] place the red cube centred on slot E2 (turned 15 deg CW); the bowl stays on its fixed mark (x ..., y ... m), hold the leader, ENTER (q = end):`.
  Place the red cube, check the cup is on its mark, then do the demo as in D (`Y` / `n` / `q`).
  `n` (redo) and a failed save keep the same placement.
- The counter restarts in every sitting. A resume first prints
  `resuming .../vla-rl-so101-real-teleop at demo 11/50: {...}`, and its `[PickPlace 1/10]` is demo
  11.
- **After an interruption** (`q`, a servo fault, a timeout, a closed Terminal, a power loss): run
  the `--resume` line. It recovers the dataset (anything a kill left half-written goes to a
  quarantine folder), then refuses unless the protocol hash, the convention, the demo schedule,
  `robot.units` and the follower's calibration file (sha256) match the recorded demos:
  `--resume refused; nothing was opened: ...`, or `--resume refused: the follower loads another
  calibration than the recorded demos ...; torque was never enabled`. A kill costs at most the
  demo being saved. Never rerun without `--resume`: with the folder present that refuses, and
  `--overwrite` deletes the recorded demos.
- Other refusals: `--resume: all 50 demos are recorded`; `--episodes must be in [1, 50] with
  --demo-schedule`; without `--task PickPlace`, `--demo-schedule records PickPlace only: pass
  --task PickPlace`.
- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop/`, with `meta/vla_rl_so101.json` (the
  placement of every saved episode, the protocol citation, the session id, one entry per sitting
  with its camera check and how it ended) and `meta/calibration_follower.json`.

### E3. What refuses before the freeze (and with what message)

Both E2 lines refuse before anything opens: no port, no camera, no dataset folder. The run order
is: the terminal check, then `No leader arm configured ...` (exit 2) while `teleop.leader` is
null, then the PickPlace-only check, then the three demo gates below, in this order. Each gate
exits with the first problem it finds:

1. The bowl (true at this head):
   `refusing the real-0.1 demo schedule; nothing was opened: the protocol bowl (0.195, 0.205) m is 1.93 cm outside the safety filter's workspace box (0.1, 0.4) x (-0.2, 0.2) m (after the 5 mm placement tolerance and the oracle's 9.3 mm TCP offset): every carry to it would be rejected (scripts/wo07_bowl_check.py). The bowl is pending the owner's decision; the filter is never loosened`
2. The freeze (once the bowl fits, until `FREEZE_STATUS` is frozen and `MEASURED` is filled):
   `refusing the real-0.1 demo schedule; nothing was opened: protocol real-0.1 is not frozen (pending: bowl fit check): ...; protocol real-0.1 prop measurements PENDING: red_cube_edge_mm, ...`
3. The camera controls:
   `refusing the real-0.1 demo schedule; nothing was opened: camera controls not locked: cameras.front.controls.exposure is null: ...`

Sections C and D are not refused by the freeze. Their gates are B and your go.

### E4. Check the 50 (no motion)

```zsh
.venv/bin/python scripts/preflight_decoder.py --preset real --dataset data/real_teleop/vla-rl-so101-real-teleop --episodes 50
```

- **Expected:** `[dataset] OK: 50 episodes; both cameras decode across every frame; demo schedule
  recording, session <UTC stamp>-wo06-teleop-<hex>, sittings ended [...]; episodes 0-49 = demos 1-50 of the
  frozen schedule; real-0.1 hash <first 12>`, then `PASS: ...`. A FAIL names the episode and the
  camera. Do not export or train on the dataset; send the output. To re-record, move the folder
  aside first.

### E5. Export bundle (local only)

```zsh
.venv/bin/python scripts/wo07_export_bundle.py --preset real demo --dataset data/real_teleop/vla-rl-so101-real-teleop
```

- **Writes:** `data/real_teleop/vla-rl-so101-real-teleop.export/` (declaration, calibration,
  rubric, placements, bundle index), then prints the bundle index JSON and the vishwaroop seal
  command. It never overwrites an existing bundle.
- **Refuses** (`REFUSED: ...`) a rehearsal, a recording made without `--demo-schedule`, an episode
  with no recorded placement, or a recording whose real-0.1 or convention hash differs from the
  code's (e.g. demos recorded before a protocol change).
- Sealing (in vishwaroop) and any Hub push are separate owner decisions. This runbook pushes
  nothing: no `--push`, no `hf upload`.
- **Send back:** the bundle index JSON and the E4 `[dataset]` line.

### Between sittings and sessions

- After each sitting: the command has exited, the arm rests on the table, the follower supply is
  OFF, then unplug USB if you are leaving. Never leave the follower powered and torqued
  unattended.
- **Heat:** touch-check the servos (the gripper first). If they feel hot, end with `q`, power
  off, and let them cool for at least 10 minutes. After a `SERVO FAULT`, do the same and find the
  cause before resuming. The monitor's thresholds (55 C, 1500 mA for 1 s) are provisional: report
  what you felt. No command prints servo temperatures after a teleop session.
- **Start of every session:** section A (preflight `PASS`), A4 (controls still locked after any
  replug or reboot), section B row 5 (drift check), the camera-role check (built into every
  sitting). Keep the lamp at 5000 K and in its recorded position, and the room light the same.
- **Mac:** on AC power, `caffeinate -dims` running in its own tab (A2).
- **Send back after each sitting:** the `saved N episodes` line, the `resuming ... at demo N/50`
  line, and any `STOPPED` / `SERVO FAULT` / refusal line.
