# Status

Updated: 2026-11-08

## Current work order

Independent WOs run in parallel (owner decision 2026-10-26, CLAUDE.md "Operating mode"). Any WO
started before its predecessors merge is listed here, marked **out of order**, before work begins.

| WO | State | Branch / PR | Notes |
|---|---|---|---|
| 01 | done, merged | #7 (merged) | See benchmarks. Demos on the Hub (`KudratSingh22/vla-rl-so101-demos`, private, built at commit `0dffe51`). |
| 02 | closed by owner decision, **below the original target band**; PR open | `wo-02-sft` | OpenVLA-OFT LoRA SFT, 4 attempts, all below the 20-60% band on seen positions (ADR-0001). Owner (2026-10-27): keep OFT, amend rule 3, close with the full protocol table of attempt 4 `step_004000` (25 demos/task): seen PickPlace 17.0%, Push 10.0%, Stack 2.7% (`docs/results.md`). |
| 03 | done, merged after owner review | #4 (merged) | **Out of order** (started before WO-01/02). CPU only, toy env. |
| 06 | part 1 merged, kit arrived, physical bring-up pending | #5 (merged), #24 (`wo-06-bringup`, PR open, merged into #29) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. Calibration, camera match and teleop are pending verification. |
| 04 | **benchmark not met; next step waits for the owner**, **out of order** (started before WO-02 merged) | #11, #13, #14, #16 (merged) | Two-task seed 0 finished (PickPlace seen 22.0%, +5.0); seeds 1-2 stopped flat at step 38. PickPlace-only finished at 20.7% (+3.7), below the owner's 27.0% bar, so stratified multi-task is not run. `docs/results.md`, ADR-0002 amendment 2. |
| 07 | code only, **out of order** (started before WO-06 bring-up is verified): pilot client/server and real-eval harness on fakes, **unverified on hardware** | #23 (`wo-07-pilot`), #29 (`fix/robot-safety`), PRs open | Mock pilot, gRPC server, `scripts/wo07_real_eval.py` (protocol 0.2), placement grid. No real trials. |
| 05, 08, 09 | not started | | |
| pre-run hardening | PR open, **waits for owner review** (touches the RL loop) | `fix/pre-run-hardening` | Review `docs/reviews/2026-11-07-project-review.md` section 9, $0. See "Pre-run hardening" below. |
| robot safety (PR 2) | PR open against `wo-07-pilot`; merge after #23, #24 and #28 (it merges #24 and #28) | `fix/robot-safety` | Review 3.3 gaps 2, 3, 5 and 7, plus the owner's 2026-11-08 review changes (homing offsets, torque off unconfirmed, real-eval session fingerprint, parking off by default), $0, **unverified on hardware**. See "Robot safety hardening" below. |

## Benchmarks met

| WO | Met on | Evidence (run id / PR) |
|---|---|---|
| 01 | 2026-10-26 | Oracle, seen cell, 100 episodes/task, Mac CPU: PickPlace 100%, Push 99%, Stack 100% (run `20261026T194952Z-wo01-sim-8ad7`, which also recorded the 10 demos/task: 30 episodes, 1968 frames, pushed privately to `KudratSingh22/vla-rl-so101-demos`). SmolVLA smoke, 2000 steps x batch 8 on Mac MPS, 10 seen episodes/task: PickPlace 50%, Push 20%, Stack 0% (single run; run `20261026T195329Z-wo01-sim-b326`). `tiny` preset build: 6.9 s on CPU. |
| 02 | **not met** (closed by owner decision 2026-10-27) | Full protocol, attempt 4 `step_004000`, 3 cells x 3 seeds x 100 episodes, GPU sim, protocol `0.1-draft` `e9ce5bb47a88`: seen 17.0 / 10.0 / 2.7% (PickPlace / Push / Stack), **below the original 20-60% band**; meets the amended rule 3 (>= 10% on 2 of 3 tasks), Push exactly at 10.0%. Run `20261027T190836Z-wo02eval-sim-cdd3` (pod `jqtaavsbwvemin`). Eval harness emits the full table from one command (`scripts/wo02_eval.py`). |
| 03 | 2026-10-26 | tiny toy GRPO, seeds 0/1/2: greedy eval 0.301/0.102/0.191 -> 0.988/1.000/0.996 of optimum 1.0 in 100 steps (>=0.95 by step 20/20/25), bit-identical reruns, ~3.5 s CPU each; runs 20261026T193248Z-wo03-tiny-s0-be5126, -193252Z-…-s1-cf3b6d, -193255Z-…-s2-d11eb5 (reruns -fe1afa/-812773/-8c72b6); branch `wo-03-rl-loop` @ 051f1e6 |
| 04 | **not met** | Two-task seed 0, protocol seen: PickPlace 22.0%, Push 7.7%, Stack 2.3% (SFT 17.0 / 10.0 / 2.7), run `20261028T234248Z-wo02eval-sim-8de3`. PickPlace-only: PickPlace 20.7%, run `20261029T233518Z-wo02eval-sim-d102`. Needed >= +20 points on 2 of 3 tasks, 3 seeds. |
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
| pod `9d8p7vyfyv6ca4` (WO-04 two-task seed 0: probe, 150 steps, protocol eval) | 04 | A100-80GB, $1.79/h | 28.4 | 51.27 | yes, 2026-10-27 (auto-gate: <= 60% dropped, <= $60 projected) |
| pod `wzo6enivtqpfnb` (WO-04 two-task seed 1, kill switch at step 38) | 04 | A100-80GB, $1.79/h | 7.8 | 14.10 | yes, 2026-10-27 (after seed 0 passed step 38) |
| pod `d74m2ppdjvvof6` (WO-04 two-task seed 2, kill switch at step 38) | 04 | A100-80GB, $1.79/h | 7.6 | 13.69 | yes, 2026-10-27 (after seed 0 passed step 38) |
| pod `s2ljj3drwg0ov0` (WO-04 PickPlace-only seed 0: probe, 150 steps, protocol eval) | 04 | A100-80GB, $1.79/h | 30.5 | 55.00 | yes, 2026-10-28 (auto-gate: <= 60% dropped, <= $60 projected) |
| pod `82a5je0ve40rlp` (rollout videos, 5 checkpoints) | 04 | RTX 4090, $0.89/h | 0.18 | 0.17 | yes, 2026-10-28 |
| pod `7gbl8q46qlxf1z` (protocol eval of two-task step 50, PickPlace-only step 100) | 04 | RTX 4090, $0.89/h | 2.17 | ~1.93 | yes, 2026-10-31 (~$2) |

Recorded spend from the rows above is about **$156.10** ($13.56 for WO-02 and $142.54 for
WO-04), including estimated charges. The revised **$1,100 cloud compute cap** leaves about
**$944** based on that ledger. The first $700 of the cap covers the core plan, leaving about
$544 for core work. The separate $400 reserve covers the original WO-05 ablations if approved.

| Remaining work | Planning allowance | Release condition |
|---|---:|---|
| WO-06 hardware bring-up, WO-07 mock deployment, WO-09 writeup | $0 cloud | Run locally. |
| WO-07 supervised SFT pilot and real evaluation | $45 | Verify WO-06 first. Use pilot timings to reprice the full comparison (900 trials across two checkpoints). |
| Recovery SFT and matched sim evaluations | $30 | Collect expert corrections in sim from representative failure states and keep evaluation data held out. Measure the SFT gain before adding RL. |
| Higher-learning-rate GRPO diagnostic | $60 | Check the SFT baseline against the existing RL entry criteria. Compare short probes with matching rollout settings and price any continuation separately. |
| WO-04 confirmatory GRPO and sim evaluation | $210 | Up to three training seeds at $70 each if the diagnostic justifies replication. Keep the same evaluation protocol and early-stop gates. |
| WO-08 targeted retraining and real re-evaluation | $140 | Up to two cycles, based on failure evidence from WO-07. |
| Core contingency | ~$59 | Retries, storage, or higher measured runtime. |
| WO-05 demo-count and samples-per-prompt ablations | $400 | Reuse matching runs and revisit scope after the sim result. Each SFT base must satisfy the RL entry criteria before a paid RL follow-up. |

Allowances use the project's measured runs and the [RunPod Secure Cloud Pod list rates](https://www.runpod.io/pricing)
checked on 2026-11-07 ($0.89/h for RTX 4090 and $1.79/h for A100 80 GB). Reconcile actual spend
and price the selected GPU, storage and duration before approving each run. The cap does not
authorize provisioning. Each paid run still needs explicit owner approval and its applicable
probe gate.

## Pre-run hardening (PR 1, branch `fix/pre-run-hardening`, 2026-11-08, $0)

This implements the $0 items of the review's section 9 (`docs/reviews/2026-11-07-project-review.md`).
Nothing ran on a GPU. Protocol `0.1-draft` (`src/eval/protocol.py`, hash `e9ce5bb47a88`) and
`src/env/success.py` are unchanged, and no reported number changed.

What changed:
- RL gates (`src/rl/gates.py`): the 64-episode kill switch and the 32-episode peak stop are
  replaced by windows of `rollout/success_rate` per trained task plus rollout entropy (`rl.gate_*`).
  There are three gates: flat at step 38, decline from the best window, and entropy collapse. The
  decline gate fires only when two consecutive non-overlapping windows are each more than
  max(`gate_max_drop`, `gate_decline_z` x SE) below the best aligned window (z = 3: z = 2 stopped
  35/100 simulated flat runs; the window SE is group-level, ~1-2 points, not ~0.5-1). They
  decide spending only; the protocol eval decides improvement. Periodic greedy evals run on the
  trained tasks only and are sanity logs. The removed config keys raise an error naming their
  replacement.
- Fixed-batch diagnostic (`src/rl/diagnostics.py`): it scores the first 32 trajectories of each
  update batch against the PPO reference, before and after all optimizer steps. It logs
  `diag/{pre,post}/*`: |dlogp|, dlogp by advantage sign, mean(A*dlogp), k3 KL, clip fraction and
  outside-clip fraction. `train/grad_norm` and per-optimizer-step lists are also logged. The
  uninformative `train/approx_kl` / `train/clip_frac` are removed. `docs/results.md` and ADR-0002
  now say the old values were structurally ~0, not evidence of small updates.
- Resume: checkpoints hold the optimizer, scheduler, RNG, trainable tensors and trainer state.
  `scripts/wo03_rl.py --resume <ckpt>` continues the step numbering. A CPU round trip is
  bit-identical. Resume from the newest `step_N` checkpoint without a suffix: a
  `step_N_interrupted` one (weights may hold part of step N) is refused unless
  `--resume-approximate`, which re-runs step N and sets `record.resume_approximate`. Checkpoints are written every 25 steps and are about 1.2 GB each for OFT.
- `configs/rl.yaml` (ADR-0002 amendment 3, proposed): 16 prompts x K=8, lr 1e-4, `ppo_epochs` 2.
  The lr follows the published LoRA recipes (RIPT-VLA, RLinf-VLA), not the logged KL.
  `configs/rl_pickplace.yaml` inherits these settings. RL seeds >= 10 are rejected
  (`src/eval/seed_ranges.py` `check_rl_seed`: an explicit [0, 10) bound, then the overlap check;
  PR #25's copy has only the overlap check, which lets seeds 11-299 through, so keep this version
  when #25 merges).
- Rollout timing split (`src/rl/rollout.py`): policy / sim_step / render / obs_fn / host_copy /
  reset / other. It goes to `metrics.jsonl` (`rollout/time/*_s`) and to the PROBE line.
- Probe modes on `scripts/wo03_rl.py`: `--lr-sweep` (offline, on one stored batch) and
  `--probe-temperatures` (rollouts only, the same prompts at each temperature). Both can be costed
  with `--estimate`.
- Demos: the recorder can keep recording through release and retreat. With
  `configs/demos_v2.yaml` it records at least 15 frames past success, until the oracle has
  finished, with a cap of 45 frames, and rejects a demo if success no longer holds at its end. The
  SFT loader can drop chunks that would contain pad frames (`data.chunk_pad: drop`). The code
  defaults are the 0.1 settings (no tail, oracle v1, `chunk_pad: repeat`), so `sim.yaml`,
  `sft*.yaml` and `rl.yaml` keep their behaviour and config hashes; `demos_v2.yaml` (and
  `demos_v2_tiny.yaml` for the CPU dry run) opt in, and an SFT config for the v2 set must set
  `chunk_pad: drop`. `wo01_build_env.py --push` refuses the two 0.1 demo repos unless
  `--overwrite-hub`. Measured on the existing 0.1 demo set: 11 of 25 PickPlace and 13 of 25
  Stack demos end below the 0.45 rad release target.
- Oracle v2 (`src/data/oracle.py`) slows to 0.5 cm/step over the last 6 cm and the descent. Results
  on the Mac CPU sim, tiny preset, 100 episodes per cell: seen PickPlace / Push / Stack 100 / 100 /
  100%, shifted 96 / 92 / 96%. Under v1 the same cells were 100 / 98 / 100% and 99 / 91 / 96%.
  Near-object decisions per episode rose from ~1.5 to ~2.6-3.0. v1 stays the default and
  reproduces the 0.1-era oracle exactly; v2 is opt-in (`data.oracle_profile: v2`).
- Demo rebuild (`scripts/wo01r_rebuild_demos.py`, `scripts/pod/demo_rebuild_job.sh`,
  `configs/demos_v2.yaml`): it rebuilds 25 demos per task on the GPU renderer into a new Hub repo
  (`KudratSingh22/vla-rl-so101-demos-25-v2`); it does not write to the 0.1 repos. It also has a
  render-gap mode that compares stored and fresh frames and greedy tokens against a codec floor.
  Preliminary single run on the Mac CPU (run 20261108T083504Z-wo01r-sim-c025, dirty tree at
  2af16f6, first 3 of 25 episodes per task (223/163/209 frames), pixels and joints only, token
  comparison skipped): the pixel gap is within 0.03 mean-abs (<= 1.4%) of the codec floor on all 3
  tasks (e.g. PickPlace 2.070 vs 2.042) and joint drift is 0. The full 25-episode check with
  tokens on the 4090 is still pending (see Next).
- Protocol 0.2 (`src/eval/protocol_v02.py`, new; `scripts/wo02_eval.py --protocol 0.2`). Its hash
  combines the file bytes of the protocol, the eval loop (`src/eval/sim_eval.py`: episode seeds,
  clip, success latch, episode end), env, success, camera and robot model with the canonical
  run settings, the clip, the episode-seed probes and the sim-stack pins. Per-episode records go to
  `runs/<id>/episodes.jsonl` under both protocols. Under 0.2 only, a 4 deg/step joint clip from
  `configs/real.yaml` applies. Runs that deviate get `protocol_complete: false` and a list of the
  deviations. 0.2 is not frozen: its hash is pinned when its first number is reported.
- Camera jitter is now drawn from the episode RNG instead of the global torch RNG. It is off in
  every current preset.
- `make setup` = `uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla`
  (`docs/ENVIRONMENT.md`).

Next paid run (proposed, not launched; each needs the owner's explicit go): the review's diagnostics,
about $5 in total.
1. Render-gap check on a 4090: `scripts/pod/demo_rebuild_job.sh` up to its approval gate (setup,
   GPU-sim oracle check, render gap of the 0.1 set with the a4 `step_004000` adapter).
   estimated: RTX 4090 x 0.6 h = $0.53. Continuing past the gate to rebuild the demos brings the
   whole job to about 0.75 h = $0.67. This is a guess: no GPU timing exists for the job.
2. One A100 pod with the SFT adapter (`--seed 3`):
   - `--lr-sweep 5e-5,2e-4,5e-4,1e-3 --sweep-adam-steps 8`, 1.20 h;
   - then `--probe-temperatures 0.7,1.0,1.3 --probe-groups 64`, 0.60 h;
   - plus 0.5 h setup.
   estimated: A100-80GB x 2.3 h = $4.12 (`--estimate` from `cost.throughput`: rollout 86
   actions/s measured on WO-04 pod gzxhwhjhdtc1kc; update 10.3 decisions/s from the WO-02 SFT
   probe, pod u87r9sjnwig58d; forward 31 decisions/s estimated). The 2.3 h excludes the 7B load
   and GPU env build of each of the two separate `uv run` invocations; `setup_hours` 0.5 covers
   only the image, `uv sync` and the download.
3. Only after 1-2, and after the recovery-SFT base clears rule 3: a GRPO run with the measured lr and
   temperature written into `configs/rl.yaml`, starting with `--probe-steps 3`. The worst case is
   estimated: A100-80GB x 41.3 h = $74 per 150-step seed, or 10.5 h = $18.72 if the flat gate stops
   it at step 38.

Proposed, not run: a rerun of the existing SFT `step_004000` and the two RL step-150 checkpoints
under protocol 0.2. It is priced from the measured eval runtime of ~1.05 h per checkpoint on a
4090, plus ~5% for the clip. estimated: RTX 4090 x 3.4-3.8 h = $3.0-3.4. Adding two-task step 50
and PickPlace-only step 100 brings it to $5.0-5.3.

Unverified on GPU: gate behaviour and thresholds (`gate_min_gain` 0.03 now applies to sampled
rollout success and needs the owner's confirmation; `gate_max_drop` 0.04, `gate_decline_z` 3 and
`gate_entropy_min` 0.5 are not measured on GPU; z = 3 is from a CPU simulation only), diag/sweep forward cost, the throughput for the 16 x 8 / 2-epoch shape, the render
hook timing on `physx_cuda`, the CUDA RNG restore (a GPU resume is not bit-identical), v2 oracle
success on `physx_cuda`, recorder tail stepping on the GPU env, render-gap replay and 7B token
comparison, and the 0.2 clip and end-flag capture on CUDA.

Deferred: the 4 deg/step clip inside RL rollouts (review Track B item 4), so RL rollouts and 0.2
eval do not yet see the same action limits; `src.eval.sim_eval.clip_joint_step` is ready to reuse.
The robot items (servo temperature/current fault response, `park()` on healthy shutdown, camera
timestamp, placement grid, `wo07_real_eval.py`) are in PR 2 (`fix/robot-safety`, on PR #23's
branch; see "Robot safety hardening" below). The plan
revision for PR #25 is on `wo-recovery-sft`. PR #25's `configs/rl_pickplace_lr*.yaml` use the
removed `early_stop_*` / `peak_stop_step` keys and need renaming when it merges, or they fail fast.

## Robot safety hardening (PR 2, branch `fix/robot-safety`, 2026-11-08, $0)

Review 3.3 and section 9's robot item. The branch is PR #23's branch (`wo-07-pilot`) with #24
(`wo-06-bringup`) and #28 (`fix/pre-run-hardening`) merged in. Tested only with
`tests/robot_fakes.py`. **Nothing here has run on the arm or camera.**

- Healthy shutdown can park (off by default since the owner's review, below). `So101Driver.park` drives to `robot.park.qpos` (`null` =
  `deploy.home_qpos`, the sim REST_QPOS) through the full safety filter, then turns torque off.
  It is wired into `scripts/wo07_pilot.py`, `scripts/wo07_real_eval.py` (`shutdown_session`:
  session completed, `q`, or Ctrl-C at a prompt) and the healthy end of
  `scripts/wo06_teleop_record.py`. It refuses once the deadman has tripped, so a fault path or
  a Ctrl-C during motion never parks: those hold, then the support prompt, then torque off.
- Servo temperature and current monitor (`src/robot/servo_health.py`). After each tick's write,
  every 4th tick does one extra `sync_read` of `Present_Temperature` (addr 63, deg C) or
  `Present_Current` (addr 69, 6.5 mA/count, bit 15 masked). Each register is read every 533 ms
  at 15 Hz. Registers were checked in lerobot 0.4.4's `STS_SMS_SERIES_CONTROL_TABLE`. Read time
  is logged against a 10 ms budget. A fault means torque off at once, the deadman latched
  TORQUE_OFF, and a `SERVO FAULT` prompt to support the arm and cut the supply: not HOLD, not a
  park. Thresholds are 55 C (the firmware cut-out is 70 C) and 1500 mA sustained for 1 s. They
  are **placeholders that need the owner's confirmation on the arm**. It does not read while a
  prompt waits. At every teleop prompt where the follower holds under torque (scene, keep,
  resume) it reads both registers once before the wait and once after the answer, before the
  follower moves; the final support prompt reads them once before its wait. A current found
  over the limit by such an on-demand read is re-read after `current_trip_s` (1 s) before the
  call returns. Over-current counts as sustained only over continuous monitoring (no gap
  between current reads longer than max(2 × 533 ms, 1 s), at least two readings).
- Torque-enable refusal. Every torque enable first runs `check_rest_pose` with torque off. It
  refuses when a joint rests more than 1 deg outside its calibrated range (the calibration file
  and the servo EEPROM must also agree) or more than 6 deg outside the safety limits. Covers
  #24's not-done item.
- Ctrl-C port-busy backstop. scservo_sdk's `txPacket` sets `port.is_using`, and only the receive
  or TxOnly paths clear it, so a Ctrl-C mid-transaction would block the hold or torque-off that
  follows. `release_port` (`clearPort()`, `is_using = False`, then drop the input buffer, as
  lerobot's `disconnect` does) runs first in `hold`, `torque_off`, `trip` and `park`. Covers
  #24's skipped finding.
- Latency. LeRobot 0.4.4 stamps a frame after `VideoCapture.read()` returns, and OpenCV 4.11's
  AVFoundation backend exposes no capture time. The metric is renamed
  `frame_return_to_first_write_ms` and is never called end-to-end.
  `docs/bringup/latency-measurement.md` gives the one-off optical measurement (screen clock
  filmed by the robot camera). With `deploy.capture_offset_ms` set, the summary adds a labelled
  `end_to_end_estimate_ms`.
- `scripts/wo07_real_eval.py`: protocol cells × `eval.trials_per_cell` (50), placement-grid slots
  from `src/eval/real_grid.py` (seeded by `eval.slot_seed`, so SFT and SFT+RL sessions are paired
  trial by trial), and the judging rule `real-judge-2` (was `real-judge-1`; see the owner
  decisions below): judged once after the arm stops, no success latch, doubt = n. Every trial record carries the protocol 0.2 hash, the 0.1-draft reference, and
  the hashes of the rule, the grid and the PDF. The run record holds the Wilson table.
  `--resume <run id>` retries slots whose policy never started, with the same placement, and
  refuses if the evaluation changed. `--hardware` is gated like the pilot. `--mock` runs on the fakes.
- `docs/assets/placement-grid-a3.pdf` (`scripts/make_placement_grid.py`, vector PDF written
  directly, no new dependency): a 6 × 6 lattice of SHIFTED_REGION whose inner 4 × 4 is exactly
  SEEN_REGION, slot ids A1-F6, the robot base outline from the URDF meshes, axes, and a 10 cm scale
  bar. Print at 100% and measure the bar. `--check` fails if the committed PDF is stale.

Owner-requested changes (owner review of PR #29, 2026-11-08), same PR, $0, fakes only:

- Homing offsets are checked before every torque enable. `check_rest_pose` compares each servo's
  stored Homing_Offset (STS3215 EEPROM address 31, 2 bytes, sign bit 11; read and decoded by
  lerobot 0.4.4 `FeetechMotorsBus.read_calibration`, checked in `.venv`) with the calibration
  file's `homing_offset`. A mismatch refuses torque, even when the ranges match, and the message
  says to redo calibration (checklist step 3). Regression test: ranges identical, one offset
  differs, refused with no torque-enable or goal write.
- Torque off unconfirmed. A failed torque-off write latches `So101Driver.torque_off_unconfirmed`
  for the rest of the process. Nothing clears it: not a deadman reset, not a later torque-off
  that succeeds. While it is set, every goal-position write (step, hold, move_to, park) and every
  torque enable is refused. The operator message says to switch off the servo supply at the
  inline power switch. Trial records get `torque_off_unconfirmed: true` and stop reason
  `torque_off_unconfirmed` (the trial's own reason is kept as `stop_reason_before`). They are
  stamped after every other field (since `real-judge-2`: a started trial is then a failure,
  fault kind `torque_off_unconfirmed`; an unstarted attempt stays unscored). The shutdown result, `session_end` and the
  summary keep the flag (`scripts/wo07_pilot.py`, `scripts/wo07_real_eval.py`). Tested with a
  fake bus whose torque-off write raises.
- Real-eval session fingerprint. Six components are hashed separately: protocol (0.2 hash, 0.1
  reference, judging rule and grid hashes), policy (server policy_id,
  `server.policy_checkpoint`), camera (`cameras` block), calibration (the motors' loaded
  calibration incl. homing offsets, the file sha256, `robot.units`), safety (effective
  SafetyConfig, `robot.safety`, `robot.servo_monitor`) and motion (control rate, deploy
  settings incl. chunk execution and timeouts, trial steps, home/park incl. `enabled`, gripper
  floor, settling interval). The fingerprint is stored in every trial record. `--resume` refuses
  when it differs and names the component and the changed keys. The table refuses to pool
  trials whose fingerprints differ. On hardware it is computed before the port opens.
- Owner decisions applied:
  - 55 C and 1500 mA for 1 s are kept and marked provisional (real.yaml, checklist).
  - The -5 deg gripper floor is kept for the first rehearsal (documented).
  - Automatic parking is OFF: `robot.park.enabled: false`. A healthy end now holds, prompts to
    support the arm, then turns torque off. Checklist step 5 says how to enable parking after
    the rest-pose and park-path checks.
  - The placements are the same for SFT and SFT+RL. `--hardware` refuses any `--slot-seed`
    other than `eval.slot_seed`. `sample_plan` is prefix-stable in the trial count (for any n
    since the review fixes). Every
    record carries `placement_plan_sha256`.
  - Settling interval: `eval.settle_s` = 3 s. The arm holds still, the servos are read once,
    then the operator scores the trial. The interval is part of judging rule `real-judge-1` (now `real-judge-2`)
    (its text and hash) and of each trial's `settle` field.
- PR #29 review fixes (2026-11-08, $0, fakes only):
  - Torque off reaches every motor. `So101Driver.torque_off` no longer calls lerobot's
    `disable_torque`, which stopped at the first motor whose write raised (lerobot's `write`
    also raises on a status error bit such as overload/overheat, though the motor accepted the
    write), leaving the servos after it energized. It broadcasts Torque_Enable=0, then writes it
    to each motor in its own try (`bus._write(..., raise_on_error=False)`), then Lock=0. An
    error bit counts as written (logged); motors with no reply are named in the
    `torque_off_unconfirmed` latch. The disconnect path uses it too. Checked against lerobot
    0.4.4's real `FeetechMotorsBus` with the packet I/O stubbed (no port).
  - A connect that fails before the port opens no longer raises the "unconfirmed" alarm.
  - Teleop: any other exception holds the follower, discards the episode and shows the bounded
    support prompt before torque goes off.
  - Real eval: an unexpected exception is never a healthy end (no park; hold + support;
    `session_end` "error: ..."); a client bug in the move home trips the deadman.
    `record.json` is written when the session is bound and after each trial, so a killed run
    resumes itself. `--resume` refuses a run that a later run of the session continued. A
    TORQUE OFF UNCONFIRMED at an earlier run's shutdown is warned about on resume (hardware:
    type `checked`) and carried forward. The policy fingerprint includes the server's
    `adapter_sha256` (content of the adapter files) and Hub revision; `--hardware` refuses a
    server without it. The placement plan is prefix-stable for any n (independent RNG streams;
    the plan, and so `grid_sha256`, changed: no session has run yet).
  - Scoring (superseded by `real-judge-2`, below; judging rule `real-judge-1` text, so its hash, changed): a trial that ran and
    then faulted (servo / serial fault in the trial or at the settle read) skips the settle,
    the operator supports the arm first, then scores it; it is never a free redo. Ctrl-C (or
    EOF) at the score prompt records the trial as a failure (`score_interrupted`), so a resume
    cannot re-roll it. **Owner check:** this applies "doubt counts as fail" to both cases; the
    alternative (abort and redo) was rejected because it turns policy-caused faults into redos.
  - `scripts/make_placement_grid.py` takes `--config/--preset` (only `real`).
- Owner decisions of 2026-11-08 (second review round, same PR, $0, fakes only). **Powered
  bring-up steps 9-10 stay on hold until this revision of #29 lands.**
  - Bounded torque-on waits in the pilot and the real eval (the former accepted gap is closed).
    Teleop's helpers moved into `src/robot/prompts.py` (`TorqueWait`, `TorqueGuard`,
    `timed_input`; mypy-strict) and teleop imports them. Every pilot / real-eval prompt answered
    while the arm holds under torque (home, placement, score, `resume` after a HOLD stop, the
    final support prompt) reads servo health before the wait and after the answer and waits at
    most `deploy.prompt_torque_timeout_s` (60 s). No answer: "SUPPORT THE ARM NOW",
    `deploy.prompt_support_grace_s` (10 s, ENTER = now), deadman TORQUE_OFF, torque off, the
    session ends (`prompt_timeout: ...`, no park). `--hardware` refuses a null or non-positive
    timeout before anything opens; mock runs keep the instant `ScriptedOperator`. Prompts with
    torque off (before connect, `type 'go'`) are not bounded.
  - Shutdown honesty. `safe_shutdown` returns a `Shutdown` (end, `support_confirmed`, cause,
    ack). `supported` only after an explicit ENTER at the support prompt. Ctrl-C there is
    ignored with a message (the arm keeps holding, the bounded wait keeps running, its deadline
    not reset); a closed stdin prints a loud warning, waits the grace without stdin, then torque
    off; a timeout releases as above. Those ends are `released_unconfirmed` with the cause
    (`timeout` / `eof`) and `support_confirmed: false`. The acknowledgement waits after a fault
    (`_support_after_fault`, `_torque_already_off`, `_torque_off_unconfirmed`) record whether
    it was given and how (`enter` / `ctrl_c` / `eof` / `timeout`; bounded by the same timeout so
    the record is always written). Teleop's `end_session` now returns its end and writes
    `session_end` and `support_confirmed` into `meta/vla_rl_so101.json`; it never claimed
    support before either (it printed only), but a closed stdin used to drop torque at once
    without the grace.
  - Trial scoring (judging rule bumped to `real-judge-2`, so a `real-judge-1` session cannot be
    resumed or pooled). `policy_started` (+ UTC time) = the trial's first action from the
    policy server was handed to `So101Driver.step` (the safety filter). A fault after that is
    a failure in the primary metric, recorded separately as `fault: {kind, stop_reason, reason,
    where}` (kinds: servo, serial, safety = joint/workspace stop, lost frame, stall, stale
    chunk; server; operator_abort = Ctrl-C during motion; client_error; torque_off_unconfirmed,
    also when torque off is unconfirmed). The operator is not asked to judge it; the support
    prompt comes with the shutdown that follows (a torque-off fault always ends the session). A
    later `resume` does not change it. A score prompt ending without a judgement (Ctrl-C, EOF,
    timeout) is a failure with `outcome_source: unjudged` (doubt = n; **owner check: flagged
    choice**). Attempts whose policy never started are `not_started`: kept in the log, never
    scored, retried with the same placement on a resume; a slot with a started attempt is never
    re-run. Primary metric = successes / started trials (Wilson); a labelled secondary
    diagnostic gives the rate excluding faulted trials and the fault counts by kind.
    **Behaviour change to note:** a Ctrl-C during the trial after the policy started used to be
    an unscored redo; it is now a failure (fault kind `operator_abort`), per "slots with any
    started attempt are done".
  - Review fixes on that round ($0, fakes only):
    - Ctrl-C outside a prompt is a flag. The hardware pilot and real eval run under
      `operator_signals` (`src/robot/prompts.py`; teleop's `ctrl_c_as_flag` uses it too):
      SIGINT only sets a flag; `DeployLoop` checks it between ticks (trial and move home), so a
      press never lands inside a bus transfer, a grace sleep, the hold or the servo read of the
      shutdown. Prompt reads switch Ctrl-C back to KeyboardInterrupt for the read itself. As a
      fallback (no handler), a Ctrl-C inside the shutdown's hold, servo read, torque-off or the
      closed-stdin grace is ignored with a message (the port is released and the call retried),
      so the shutdown always finishes and is recorded.
    - Typed-ahead input is discarded (`termios.tcflush` on a tty before every prompt): an ENTER
      pressed during motion or a double ENTER can no longer confirm support or start a trial.
    - A closed Terminal / dropped ssh (SIGHUP, also SIGTERM) no longer kills the process with
      the arm held under torque: motion stops at the next tick (trial fault kind `hangup`),
      stdout/stderr are detached, every later prompt ends as `hangup` and takes the closed-stdin
      path (grace without stdin, deadman TORQUE_OFF, torque off, `released_unconfirmed`, cause
      `hangup`, record written). Teleop likewise.
    - Durable start: the moment a trial's policy starts, a `trial_started` row goes to
      metrics.jsonl. A resume turns one without its trial row (SIGKILL, crash, power loss) into
      a failure, fault kind `session_lost`, so a started slot is never re-rolled; an exception
      between the start and the record appends that failure before it propagates. Decision rows
      carry the cell.
    - Placement-prompt ends (q, timeout, Ctrl-C, EOF, a servo / serial fault from its health
      reads) after the arm went home are kept as `not_started` attempts (stop reason and fault
      recorded).
    - The pilot keeps its trials in the record even if the shutdown raises; both scripts record
      `shutdown: "error: ..."`, `support_confirmed: false` then.
    - Teleop with `prompt_torque_timeout_s: null` keeps the grace period on a closed stdin
      (`TorqueWait(None, grace)`); its provenance file records `support_release` (cause).
- Owner review of 2026-11-08, round 4 (#29, $0, fakes only): the owner reproduced
  `timed_input()` blocking past its deadline on an open stdin pipe holding a partial line
  (`select()` said readable, `sys.stdin.readline()` then waited for the newline; a mid-line
  Ctrl-D on a tty does the same). Both remedies are in:
  - The deadline covers the whole line: `timed_input` reads fd 0 with `os.read`, each read after
    `select()` with the time left, until the newline, the deadline or EOF (a pipe one byte at a
    time, so nothing past the newline is consumed; a canonical-mode tty a line per read, so a
    mid-line Ctrl-D is a partial line, not EOF). A partial line at the deadline is a timeout
    and its bytes are dropped; a partial line then EOF is EOF (no confirmation). Ctrl-C (also
    in mid-line), the hang-up path and typed-ahead discard are unchanged. Teleop's unbounded
    `null` wait goes through the same fd reader with no deadline (review F1: it used
    `input()`, which turned 'ab' + Ctrl-D, Ctrl-D at the support prompt into a truncated line
    recorded as "supported"; now it is EOF: grace, torque off, unconfirmed). `prompt_input`
    keeps `input()` for torque-off prompts only.
  - The hardware entry points (`wo07_pilot.py --hardware`, `wo07_real_eval.py --hardware`,
    `wo06_teleop_record.py`) refuse to start unless stdin and stdout are a tty
    (`require_terminal`), after the config checks and before the `type 'go'` prompt, the policy
    client, the leader or the follower. Mock runs need no terminal. A log needs `script -q`,
    not `| tee` (stdout must be the terminal so the SUPPORT THE ARM prompts are seen).
  - Regression tests: a real pipe with a partial line times out within the deadline + 0.5 s
    (the pre-fix code hangs: checked by running the new tests against `817a0c7`), partial then
    newline = the whole line, the next line is never consumed, partial + EOF = EOF, Ctrl-C and
    SIGHUP in mid-line; a pty with a mid-line Ctrl-D times out and never answers the next
    prompt; the unbounded (`null`) read on a pty with 'ab' + Ctrl-D, Ctrl-D raises EOF and
    `TorqueGuard.support` returns `eof`, not `enter` (fails with the old `input()` path); the
    tty check refuses a piped stdin before any driver / client / `go` prompt in the pilot, the
    real eval and teleop, with stdout on a pty so only the stdin can refuse (review T29-R1;
    the tests fail if the check looks at stdout only). The timeout / Ctrl-C / EOF tests at the
    pilot and real-eval prompts (fakes) pass unchanged.
  - Ctrl-Z (review F2) was left out of round 4 and is fixed in round 5 below.
- Owner decision of 2026-11-08, round 5 (#29, $0, fakes and a pseudo-terminal only): job
  control. Before it, Ctrl-Z (SIGTSTP) at a torque-on prompt or during motion suspended the
  process with the arm holding under torque, and the bounded wait could not expire while it
  was stopped.
  - Ctrl-Z never suspends the pilot, the real eval or teleop: `prompts.job_control` (entered by
    `operator_signals`; teleop also wraps its whole torque-on `with` block in it) handles
    SIGTSTP with a one-line notice (`Ctrl-Z ignored: the arm session is running ...; use
    Ctrl-C to stop motion`) and nothing else: motion keeps ticking, an open prompt keeps
    waiting, its deadline is not reset. Scope: the whole hardware session, from before torque
    is enabled to after the final torque-off, torque-off windows inside it included (a session
    stopped there would still hold the port with its record unwritten, and `bg` would resume it
    in the background). Before the session (`type 'go'`, the config checks) Ctrl-Z behaves as
    usual: torque is off and no port is open.
  - Foreground required: `require_terminal` now also refuses a process that is not in its
    terminal's foreground process group (`os.tcgetpgrp(stdin) == os.getpgrp()`, and stdout
    when it is the same terminal): started with `&` or resumed with `bg`. It runs before the
    `type 'go'` prompt and anything is opened; `job_control` checks again on entry, before the
    camera and the arm. Mock runs are unaffected.
  - Losing the foreground during the session (SIGTTIN / SIGTTOU, which the kernel sends only to
    a background group that touches the terminal; `tcgetpgrp` polled every motion tick, at
    every prompt read and every 0.25 s inside a prompt wait; EIO on a background read) takes
    the existing hang-up path, no new shutdown path: motion stops at the next tick (a started
    trial is a failure, fault kind `lost_foreground`), every later prompt ends as
    `lost_foreground` (grace period without stdin, deadman TORQUE_OFF, torque off,
    `released_unconfirmed` with cause `lost_foreground`, never "supported"). Unlike a hang-up,
    stdout is not detached (the terminal still exists). SIGTTIN / SIGTTOU are ignored after the
    first, so the kernel can never stop the process on a background read or write.
  - Every handler the session touches (SIGINT, SIGHUP, SIGTERM, SIGTSTP, SIGTTIN, SIGTTOU) is
    restored when it ends, also on an exception.
  - **Found while testing, owner's call:** under `uv run` (the documented command) Ctrl-Z goes
    to the whole foreground job, and `uv` itself does not handle it: probed on this Mac (uv
    0.13.5, same pty shell) the `uv` process stops while its Python child keeps running. zsh
    then reports the job "suspended" and takes the terminal back, so the session loses the
    foreground and ends through the controlled shutdown above (motion stops, SUPPORT THE ARM
    NOW, torque off after the grace, `released_unconfirmed` cause `lost_foreground`): never a
    process stopped with torque on, but the session ends instead of continuing (covered by a
    pty test with a uv-like launcher). Best guess: run the three hardware scripts with the
    project's interpreter directly (`.venv/bin/python scripts/wo07_pilot.py ...`), where
    Ctrl-Z is only a notice; alternatives: disable the terminal's suspend character for the
    session (`termios` VSUSP, restored at the end; then ^Z reaches no process, but becomes an
    input byte the prompt reader must drop), or accept the session end. Not changed here (the
    approved scope); `uv run` stays in the docs until the owner picks one.
  - Tests: `tests/test_job_control.py` runs the real `--hardware` code of the pilot and the real
    eval (fakes for the follower, camera, FK and policy; `tests/pty_arm_child.py`) under a
    minimal job-control shell on a pseudo-terminal (`tests/pty_shell.py`; the child's process
    group is not orphaned, so the kernel really stops it on an unhandled Ctrl-Z: the negative
    control shows a plain child stopped). Ctrl-Z typed during motion, at the move-home,
    placement, score and support prompts: the child is never stopped (no WUNTRACED stop, `ps`
    state), keeps writing goals, answers the next ENTER / `y`, and the support prompt's 8 s
    timeout still fires on time after a Ctrl-Z inside it (then grace, torque off,
    `released_unconfirmed` cause `timeout`); in the real eval ENTER after a Ctrl-Z at the
    support prompt still confirms (`supported`). A background child is refused before the fake
    follower connects and before `type 'go'`. The shell taking the terminal back during motion
    stops motion within a tick, and at a torque-on prompt ends the wait within 3 s instead of
    the 8 s timeout, both with `lost_foreground` records as above. All five fail against the
    round-4 code. A uv-like launcher stopped by Ctrl-Z: the pilot ignores it, loses the
    foreground and ends through the controlled shutdown. In-process: handler restoration (also on an exception, nested teleop shape),
    SIGTSTP / SIGTTOU in-process, the prompt poll, the entry re-check, the polled loss in
    `DeployLoop` through `pilot.session`, and in teleop (Ctrl-Z mid-episode, then a lost
    foreground). The existing timeout / Ctrl-C / EOF / hang-up / partial-line tests pass
    unchanged; `test_require_terminal_wants_a_tty_on_stdin_and_stdout` now fakes the
    foreground for its passing case (a bare pty is not the test's controlling terminal).
Unverified on hardware: every item above. In particular: the thresholds, the read cost on the
Mac's USB-serial link, whether the park pose rests safely with torque off, the refusal
tolerances against a real calibration, the Homing_Offset read on the real servos, the
behaviour of a real failed torque-off write, `select()` + `os.read` on the Mac terminal for
the bounded waits (teleop, pilot, real eval; a macOS pty passes in the tests), the terminal
and foreground checks under `uv run` in Terminal (is `uv`'s Python child in the foreground
process group?), Ctrl-Z in the Mac Terminal during motion and at a prompt, `tcflush` of typed-ahead input on the Mac terminal, SIGHUP
delivery to the Python child under `uv run` when the Terminal window closes, the port-busy recovery on a real interrupted packet, the capture offset, and the printed
grid's scale and alignment with the real base.

## Blockers and decisions pending the owner

- WO-04: GRPO has not produced a clear gain (PickPlace seen +3.7 to +5.7 points across four
  checkpoints, no peak). The 75-step PickPlace-only proposal at lr 5e-4 ($36–50 estimated on
  2026-10-31) remains on hold. The core plan prioritizes the hardware baseline and recovery SFT
  before approving a higher-LR diagnostic. WO-05 paid ablations remain deferred pending the
  sim results and a separate scope decision.

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

- 2026-11-08 Robot safety (PR #29 review): check homing offsets before any torque enable;
  a failed torque-off is reported as "torque off unconfirmed" with motion blocked for the rest
  of the process and the flag kept in records; real-eval resume and pooling bound to the
  effective camera, calibration, safety and motion settings plus protocol and checkpoint;
  servo thresholds 55 C / 1500 mA for 1 s kept as provisional; -5 deg gripper floor kept for
  the first rehearsal; automatic parking off (`robot.park.enabled: false`) until a supported
  rest pose and a clear park path are verified after WO-06 step 5; single owner of the serial
  bus (no background reader); fixed placement grid and doubt = fail approved; the same slot
  sequence for SFT and SFT+RL; a settling interval before scoring.
- 2026-11-08 Robot safety (second round): bound the pilot / real-eval torque-on prompts (60 s +
  10 s grace, servo reads around each wait; real arm refuses null); "supported" only after an
  explicit ENTER; faults after the policy started count as failures with the fault recorded,
  unstarted attempts retried on resume; powered steps 9-10 on hold until the revised #29 lands.

- 2026-10-28 WO-04 follow-up (ADR-0002 amendment 2): no per-task advantage normalization; kill
  switch passes if ANY trained task is +3 at step 38; PickPlace-only run, success = final protocol
  eval, seen, PickPlace, 300 episodes, >= 27.0% (missed: 20.7%); stratified multi-task only if it
  cleared that bar.

## Decisions taken by the agent (one line each, link ADR if any)

- 2026-11-08 Servo fault thresholds 55 C / 1500 mA for 1 s, read every 4th tick (firmware Max_Temperature_Limit 70 C; lerobot sets the gripper's Protection_Current to 250 counts = 1625 mA); kept by the owner as provisional, to confirm with rehearsal temperatures.
- 2026-11-08 Settling interval 3 s (`eval.settle_s`) before a real trial is scored; bounded torque-on teleop prompt wait 60 s + 10 s grace (`teleop.prompt_torque_timeout_s`, `prompt_support_grace_s`), the owner's proposal implemented on by default.
- 2026-11-08 Real judging rule `real-judge-1`: judged once after the arm stops, all success conditions true at that moment, doubt = n (stricter than the sim's latched success).
- 2026-11-08 Real judging rule `real-judge-2` (owner's trial-scoring decision): a trial counts once its policy started; a fault after the start is a failure (fault recorded); an unjudged score prompt is a failure (doubt = n, agent's reading, flagged to the owner); unstarted attempts are retried, never scored. Faulted-trial support moved from a trial-level prompt to the shutdown that always follows such a fault (one acknowledgement, recorded with how it was given).
- 2026-11-08 Real placement: a stratified 6 x 6 slot lattice of SHIFTED_REGION (seen = inner 4 x 4) instead of continuous positions, so an operator can place objects by hand; one slot seed pairs the SFT and SFT+RL sessions.
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

## WO-04 (PR #11, owner-approved 2026-10-27; seed runs from `wo-04-seeds`)

- 2026-10-26 Owner settings implemented: LoRA lr 5e-5; kill switch = 64 greedy episodes/task at step 0 and at the 25% check (step 38 of 150 since ADR-0002 amendment 1) on the same seeds, stop only if the gain is < 0.03; periodic evals stay 32/task; ppo_mini_batch 64.
- 2026-10-26 configs/rl.yaml inherits configs/sft.yaml (RL policy block = the one the SFT adapter was trained with); the adapter comes from `--adapter` / `SFT_ADAPTER` (dir or Hub `user/repo/path[@rev]`).
- 2026-10-26 Trainer aborts if the step-0 eval is 0% on every task (CLAUDE.md rule 3), before any RL step and before the approval gate.
- 2026-10-26 Group-reset check: joint state must match to 1e-4 rad; images may differ in <= 0.1% of values (GPU-sim sub-scene offsets can flip silhouette pixels; CPU sim is bit-identical, different seeds differ in >= 4.5%).
- 2026-10-26 build_rl_policy refuses any active dropout (incl. a loaded adapter's LoRA dropout): log_prob runs in train() mode.
- 2026-10-26 SIGTERM (pod soft deadline) saves a `step_N_interrupted` checkpoint and finalises the run record.
- 2026-10-27 GPU from measured probes (original settings): A100 $108 vs H100 $206 per 300-step seed -> A100 (owner's 10% rule). Amended settings, A100: $55.6 per 150-step seed at K=8; K=16 worst case $59 (`--estimate`).
