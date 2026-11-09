# Status

Updated: 2026-11-09

## Pivot to SmolVLA (owner decision 2026-11-09, ADR-0003)

The OpenVLA-OFT / SimpleVLA-RL reproduction is stopped. The project is now SmolVLA on the real
SO-101 with ACT as the baseline: leader-arm teleop demos, two cameras (wrist + front), fine-tune
on a 24 GB card, inference on the Mac (MPS). Few-shot stays a sim experiment; the real arm is
data-first (CLAUDE.md rule 3). Spec: `docs/reviews/2026-11-09-smolvla-pivot-handoff.md`, refined
by the owner's decisions D1-D6 in `docs/reviews/2026-11-09-pivot-decisions.md`.

- **Current WO: 06 (bring-up)**, new table in CLAUDE.md: 06 bring-up, 07 real demos + SmolVLA/ACT
  baseline, 08 filtered behaviour cloning in sim, 09 writeup. WO-01–05 are closed (WO-04's
  benchmark unmet; its numbers stay in `docs/results.md` as a partial result, track stopped
  2026-11-09).
- **Real data plan (D1-D5):** 50 PickPlace demos = 10 at each of five seen slots from the 16-slot
  inner block (one near each corner, one central), bowl at one fixed position, both cameras.
  Eval: 50 binary trials per cell (seen 10 x 5 slots; shifted balanced over the 20 ring slots;
  unseen 25 purple cube + 25 green cube at the seen slots, per object and pooled). Slot ids, bowl
  position and seeded schedule frozen and hashed in a new real protocol version before the first
  demo; same schedule for both policies. 100 demos only if a development-trial failure analysis
  shows more data addresses the failures.
- **Kept:** protocol 0.2, the recorder fixes, the GPU demo-rebuild tooling, the pod backstop
  (watchdog, launch gate, hard-cap timer, self-delete), `src/rl` (unused), the merged robot code.
- **Deferred:** PR #25 (recovery SFT), PR #27 (Track C doc). Not touched.
- **Cancelled spend:** the diagnostics pods (the $7.42 plan and its $8.42 revision, "Pre-run
  hardening" below) were never approved and never launched: $0 spent on them. In the OFT-era
  allocation table under "Spend", the higher-LR GRPO diagnostic ($60), confirmatory GRPO ($210),
  WO-05 ablations ($400) and WO-08 real retraining ($140) lines are void; recovery SFT ($30) is
  deferred with #25; the WO-07 line ($45) is re-priced for SmolVLA/ACT in PR 2. **Spend to date
  is unchanged: about $156.10.**
- **Next paid run:** the first SmolVLA fine-tune on the real demos, after a short probe and the
  owner's explicit go. PR 2 (`feat/smolvla-real`) carries its cost line; nothing is launched
  before then.

## Current work order

Independent WOs run in parallel (owner decision 2026-10-26, CLAUDE.md "Operating mode"). Any WO
started before its predecessors merge is listed here, marked **out of order**, before work begins.

| WO | State | Branch / PR | Notes |
|---|---|---|---|
| 01 | done, merged | #7 (merged) | See benchmarks. Demos on the Hub (`KudratSingh22/vla-rl-so101-demos`, private, built at commit `0dffe51`). |
| 02 | closed by owner decision, **below the original target band**; PR open | `wo-02-sft` | OpenVLA-OFT LoRA SFT, 4 attempts, all below the 20-60% band on seen positions (ADR-0001). Owner (2026-10-27): keep OFT, amend rule 3, close with the full protocol table of attempt 4 `step_004000` (25 demos/task): seen PickPlace 17.0%, Push 10.0%, Stack 2.7% (`docs/results.md`). |
| 03 | done, merged after owner review | #4 (merged) | **Out of order** (started before WO-01/02). CPU only, toy env. |
| 06 | **current WO** (bring-up, new table); part 1 merged, kit arrived, physical bring-up pending | #5, #24 (merged) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. Calibration, camera match and teleop are pending verification. |
| 04 | **benchmark not met; track stopped 2026-11-09 (ADR-0003)**, partial result recorded, **out of order** (started before WO-02 merged) | #11, #13, #14, #16 (merged) | Two-task seed 0 finished (PickPlace seen 22.0%, +5.0); seeds 1-2 stopped flat at step 38. PickPlace-only finished at 20.7% (+3.7), below the owner's 27.0% bar, so stratified multi-task is not run. `docs/results.md`, ADR-0002 amendment 2. |
| 07 | now "real demos + SmolVLA/ACT baseline"; code only, **out of order** (started before WO-06 bring-up is verified): pilot client/server and real-eval harness on fakes, **unverified on hardware** | #23, #29 (merged); SmolVLA/ACT code is PR 2 (`feat/smolvla-real`) | Mock pilot, gRPC server, `scripts/wo07_real_eval.py` (protocol 0.2), placement grid. No real trials. |
| SmolVLA pivot, PR 2 | PR #32 open, code only, $0 (PR 1 #31 merged) | `feat/smolvla-real` | Convention, SmolVLA / ACT trainer, Mac in-process inference, protocol real-0.1, eval capture, vishwaroop export. **Unverified on hardware and GPU.** See "PR 2: SmolVLA and ACT on the real arm" below. |
| 05 | cancelled with the OFT track (ADR-0003) | | |
| 08, 09 | not started (08 is now filtered behaviour cloning in sim; plan in `docs/work-orders/WO-08-filtered-bc-sim.md`) | | |
| pre-run hardening | merged | #28 (`fix/pre-run-hardening`) | Review `docs/reviews/2026-11-07-project-review.md` section 9, $0. See "Pre-run hardening" below. |
| robot safety (PR 2) | merged | #29 (`fix/robot-safety`) | Review 3.3 gaps 2, 3, 5 and 7, plus the owner's 2026-11-08 review changes (homing offsets, torque off unconfirmed, real-eval session fingerprint, parking off by default), $0, **unverified on hardware**. See "Robot safety hardening" below. |

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

Superseded 2026-11-09 by the pivot (ADR-0003; see "Pivot to SmolVLA" at the top): the
allocations below are the OFT-era plan, kept for the record.

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
  35/100 simulated flat runs; the SE is group-level: simulated 1.1-2.1 points for a 10-step
  window per task at p 0.08-0.20, 1.6-2.9 for a two-window difference). The flat gate (review
  L-baseline) fits a per-task least-squares trend over steps 1-38 and measures the gain from step
  1, the SFT policy; the old baseline window 1-10 had already taken 0-9 updates. Provisional
  default (owner 2026-11-08, simulated only): stop only if gain + `gate_flat_z` (0.5) x SE < 3
  points. Simulated P(stop): truly flat 0.40-0.73, one task truly +5 0.06-0.09 (plain +3 bar on
  the trend 0.17-0.23, the older rollout-window rule 0.32-0.33); `gate_flat_z: 0` is the plain
  bar. Table in `src/rl/gates.py`; it does not reproduce the original 64-episode greedy gate. They
  decide spending only; the protocol eval decides improvement. Periodic greedy evals run on the
  trained tasks only and are sanity logs. The removed config keys raise an error naming their
  replacement.
- Fixed-batch diagnostic (`src/rl/diagnostics.py`): it scores the first 32 trajectories of each
  update batch against the PPO reference, before and after all optimizer steps. It logs
  `diag/{pre,post}/*`: |dlogp|, dlogp by advantage sign, mean(A*dlogp), k3 KL, clip fraction and
  outside-clip fraction. `train/grad_norm` and per-optimizer-step lists are also logged. The
  mini-batch-averaged `train/approx_kl` / `train/clip_frac` are removed. `docs/results.md` and
  ADR-0002 now say the old values were weak evidence (approx_kl halved by the on-policy mini-batch
  1, measured after one Adam step on other prompts), not a measure of the per-update change.
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
  Its headline is t = 0 (the first frame of each episode, before any replay physics); pooled
  all-frame numbers are reported separately because replay physics drift contributes to them.
  Preliminary single run on the Mac CPU (run 20261108T083504Z-wo01r-sim-c025, dirty tree at
  2af16f6, first 3 of 25 episodes per task (223/163/209 frames), pixels and joints only, token
  comparison skipped). **This is a same-renderer pipeline check:** the stored 0.1 frames and the
  fresh frames were both rendered on the Mac CPU renderer. It is not a Mac-vs-GPU renderer gap
  measurement. Headline, t = 0 (3 first frames per task, before any replay physics): pixel gap
  mean-abs 1.997 / 1.968 / 2.001 (PickPlace / Push / Stack). This run recorded no t = 0 codec
  floor and no tokens. Pooled all frames, reported separately because replay physics drift
  contributes to them (joint drift was 0 in this run): within 0.03 mean-abs (<= 1.4%) of the
  pooled codec floor on all 3 tasks (e.g. PickPlace 2.070 vs 2.042). The Mac-vs-GPU check (full
  25 episodes, with tokens, on the 4090) is still pending (see Next).
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

**CANCELLED 2026-11-09 (ADR-0003): this diagnostics pass was never approved and will not run.
The plan below is kept for the record.**

Next paid run (proposed, not launched; each pod needs the owner's explicit go): the review's
diagnostics.

**FIRM CAP: $8.42 for the whole diagnostics pass** (was $7.42, $7.65, then $7.85; round 3,
2026-11-08: +$0.57 for the A100's paired clip on/off probe, owner 2026-11-08. Re-priced with live
rates and storage, every cap counted from the pod create call instead of first boot, and an
explicit allowance for the billed time between the create call and container boot).
Caps are rounded up to the cent. The cap is enforced by three layers: the job's own delete, the in-container hard-cap timer,
and the outside watchdog (`scripts/pod/watchdog.py` on a GitHub runner and on the Mac; see "Pod
deletion backstop" and `docs/runbooks/pod-launch.md`). Each layer fires before `DEADLINE` =
`T_req` + `SPEND_CAP_SECONDS`, where `T_req` is the epoch taken just before the create call.

| Pod | GPU $/h (live, Secure) | Storage $/h | `SPEND_CAP_SECONDS` (max billable) | **Firm cap** | Expected (planned) | +25% | + full gate wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 4090, render gap (`demo_rebuild_job.sh` to its gate) | 0.89 | 0.0139 | **5940** (1.650 h) | **$1.50** | 29 min = $0.44 | $0.55 | $1.00 |
| A100 80 GB, diagnostics (`diag_job.sh`, with the clip probe), gate wait | 1.79 | 0.0139 | **13800** (3.833 h) | **$6.92** | 134 min = $4.03 | $5.04 | $5.94 |
| A100 80 GB, diagnostics, pre-approved (no gate wait) | 1.79 | 0.0139 | 12000 (3.333 h) | $6.02 | $4.03 | $5.04 | n/a |
| **Both pods** (with the A100 gate) | | | | **$8.42** | **$4.47** | **$5.59** | $6.94 |
| Both pods (A100 pre-approved) | | | | $7.52 | $4.47 | $5.59 | |

Without the clip probe (`CLIP_GROUPS=0`) the round-2 figures still hold: A100 12660 / 10860 s,
$6.35 / $5.45, both pods $7.85.

Sources and assumptions:
- GPU rates were read at 2026-11-08T13:32:45Z and again, unchanged, at 2026-11-08T14:23:29Z and
  2026-11-08T14:43:08Z with RunPod's public, unauthenticated GraphQL `gpuTypes` query (read-only):
  - RTX 4090: Secure $0.89/h, Community $0.34/h;
  - `NVIDIA A100 80GB PCIe`: Secure $1.79/h, Community $1.19/h;
  - `NVIDIA A100-SXM4-80GB`: Secure $1.79/h, Community $1.39/h.

  The diagnostics timings come from WO-04 pods, which were A100-SXM4-80GB (ADR-0002, pods
  `5rdd45j801bk5b` and `gzxhwhjhdtc1kc`). Both A100 types cost $1.79/h on Secure Cloud. The plan
  uses Secure Cloud, as every earlier pod did.
- Storage rates are from RunPod's pricing docs (docs.runpod.io/pods/pricing, read 2026-11-08,
  re-read 14:23Z):
  "Pods are billed by the second for compute and storage". Container disk costs $0.10/GB/month
  while running. Volume disk costs $0.10/GB/month running and $0.20/GB/month stopped. A network
  volume costs $0.07/GB/month and keeps billing after the pod is gone, so this plan does not use
  one. The plan's 80 GB container disk plus a 20 GB pod volume at `/workspace` cost 100 x $0.10 /
  720 h = **$0.0139/h** (assumes a 30-day month), or $0.02-0.05 per pod. Both disks stop billing at
  termination, because the plan never stops a pod.
- The disk sizes are agent estimates: no earlier pod's disk sizes are recorded in `docs/` or
  `runs/`. The container disk holds the `nvidia/cuda:12.4.1-cudnn-devel` image layer, the Linux
  CUDA venv and uv cache (~20 GB), the 15 GB OFT base in the HF cache, the adapter (~0.3 GB) and
  `runs/` (~1 GB incl. `sweep_batch.pt`), ~45 GB in all, so 80 GB leaves headroom. `/workspace`
  holds only the first-boot epoch, `job.log`, `stage` and `APPROVED.<pod id>`. A persistent pod volume is
  needed so that a container restart does not re-arm a full cap.
- Max billable time = `SPEND_CAP_SECONDS`, assuming billing starts no earlier than the create call
  (`T_req` is taken before it) and per-second billing, as the docs say. The outside watchdog sends
  its first delete at `DEADLINE` - 400 s, and three worst-case attempt cycles (~128 s each:
  3 deletes x 15.5 s + 2 reads x 3 endpoints x 8.5 s + 3 x 10 s gap/pause) end by about
  `DEADLINE` - 16 s. The in-container timer fires at `DEADLINE` - 480 s, and the job's soft deadline
  falls 1200 s before that. The firm cap is therefore (GPU + storage) $/h x cap, as long as any one
  layer works. If none has verified the pod gone by `DEADLINE` (RunPod's API down for every caller),
  the watchdog alerts and keeps trying for 900 s more, which is +$0.23 on the 4090 and +$0.45 on the
  A100, outside the firm cap. After that, only the console can stop the pod. (A watchdog that starts
  or wakes late gets its 900 s from when it reaches the terminate phase, so a very late start could
  add more; the GitHub copy normally starts within a minute of the create call.)
- The caps count from `T_req`, so they now carry an explicit allowance for the billed time before
  the container boots: cap = planned x 1.25 + the full gate wait (30 min) + the 20 min
  soft-deadline margin + the 480 s in-pod lead + 240 s from `T_req` to boot (the old 180 s
  `CAP_PRESTART_S` image-pull allowance plus ~60 s for the create call), rounded up to the minute:
  5940 / 13800 / 12000 s (A100 planned 134 min: 167.5 + 30 + 20 + 8 + 4 = 229.5 -> 230 min;
  pre-approved 199.5 -> 200 min). Measured from creation, the job's soft deadline falls at
  71 / 202 / 172 min, which covers planned x 1.25 + gate + 4 min pre-boot = 70.25 / 201.5 /
  171.5 min. If the
  image pull and create call take longer than 240 s, the job loses that time at its end (the spend
  cap still holds); `createdAt` vs boot time in the pod log shows the real figure. The expected cost
  is unchanged apart from storage. (An earlier revision of this table, $7.65, had dropped the
  pre-boot allowance: its slack was 15-45 s.)
- The margin: seed 0's RL stage took 26.8 h against its probe's 23.1 h projection (+16%, job log
  of pod `9d8p7vyfyv6ca4`). PickPlace-only took 28.7 h against 26.8 h (+7%).
- Spend approval is pending, and nothing is launched. The GitHub watchdog only works once this PR
  is on `main`. Until then, the Mac copy is the only outside layer.
Timings below are from the WO-02/WO-04 job logs on the Hub (`KudratSingh22/vla-rl-so101-rl`
`wo04/*/job.log`, `KudratSingh22/vla-rl-so101-sft` `wo02*/*/job.log`) unless marked estimated.

1. Render-gap check on a 4090: `scripts/pod/demo_rebuild_job.sh` up to its approval gate (no
   rebuild in this pass; a rebuild needs its own go and a larger cap). Planned 29 min = $0.43:
   - pod start to job launch, and teardown: 4 min (1.6-4 min outside the job on 4090 pods
     `jqtaavsbwvemin`, `cm7d9mlccy3xjm`, `82a5je0ve40rlp`);
   - setup stage (apt, `uv sync`, CUDA check): 0.8 min (0.5-0.8 min measured on 4090 pods);
   - GPU-sim v2 oracle check, 50 seen episodes x 3 tasks: 4 min (estimated; the A100 v1 check with
     20 episodes x 3 took 1.6-1.9 min, and v2 makes ~2x the near-object decisions);
   - render-gap invocation: 15 GB base download + 7B load 1.5 min (fetch ~10 s on 4090 pods),
     replay of 75 episodes + 6 codec-floor episodes 3 min (estimated), greedy tokens on stored and
     fresh frames, ~10,800 forwards (~5,000 frames x 2 + floor; the Mac preliminary run's first
     3 episodes had 54-74 frames each) at 12.9 decisions/s = 14 min (4090 protocol eval: 2,700
     episodes in ~1.05 h, sim included, so a lower bound on the rate);
   - uploads 2 min; approval gate up to `APPROVAL_WAIT_S` 30 min (not in "planned").
2. One A100 pod, `scripts/pod/diag_job.sh` (new; seed 3, SFT adapter a4 `step_004000`; every
   probe uses `configs/rl.yaml`'s clip, 4.0 deg/step):
   `--probe-temperatures 0.7,1.0,1.3 --probe-groups 64` (rollouts only, the short first step), then
   `--probe-clip --probe-groups 32` (paired clip on/off, rollouts only), then
   an approval gate (or the pre-approval file), then `--lr-sweep 5e-5,2e-4,5e-4,1e-3
   --sweep-adam-steps 8`, upload to `KudratSingh22/vla-rl-so101-rl` `diag/<pod>/`, self-delete.
   Planned 134 min = $4.00 GPU ($4.03 with storage; round 2 without the clip probe: 119 min):
   - pod start to job launch 2.5 min (1.5-2.5 min: "pod up" minus job time, 4 WO-04 A100 pods);
   - setup stage 1.9 min (0.4-1.9 min, 5 A100 job logs); adapter download 0.8 min (0.25-0.8);
   - temperature-probe invocation: 15 GB base download + 7B load + 2 GPU env builds 1.5 min
     (1.3-1.4 min: probe run start to first step, minus its baseline eval; base fetch 45-57 s,
     env setup 5-7 s); rollouts 3 T x 16 env batches x 49.2 s = 39.4 min (49.2 s = the slowest
     measured 32-env x 120-step batch, pod `s2ljj3drwg0ov0`; `--estimate` says 44.6 s);
   - paired clip probe (round 3), 14.2 min: its `--estimate` check in the gpu-check stage 0.1 min;
     cached 7B load + 2 env builds 0.5 min (as the lr sweep's invocation); rollouts 2 arms x 8 env
     batches (32 prompts x K=8 = 256 episodes per arm, 4 groups per 32-env batch) x 49.2 s =
     13.1 min (`--estimate` 44.65 s per batch: 11.9 min); upload 0.5 min (estimated, small files).
     estimated: A100-80GB x 0.24 h = $0.43 (GPU $0.42 + storage $0.003; +25% $0.53);
   - lr-sweep invocation: cached 7B load + env builds 0.5 min (10 s load + 5-7 s env setup
     measured); collection, refill cap, 12 env batches x 49.2 s = 9.8 min; greedy + score
     forwards of the 1,920-decision batch once, 2.1 min; per lr 8 Adam steps x 64 x 15 decisions /
     10.3 per s + 2 forwards = 14.5 min, x 4 = 58.0 min (forward rate 31/s estimated; the WO-04
     update with the recompute forward measured 241 s against 248 s predicted from these rates);
   - uploads (incl. the ~0.3 GB `sweep_batch.pt`, estimated) and delete 3 min; approval gate 0
     with the pre-approval file, else up to `APPROVAL_WAIT_S` 30 min (not in "planned"). The job
     also refuses to start the sweep if the soft deadline leaves less than its estimate + 25%.
   The sweep's per-lr old log-prob recompute is now done once (review L-sweep-recompute): the
   `--estimate` fell from 1.20 h to 1.15 h.
3. Only after 1-2, and after the recovery-SFT base clears rule 3: a GRPO run with the measured lr and
   temperature written into `configs/rl.yaml` (clip 4.0, owner 2026-11-08), starting with
   `--probe-steps 3`. **No full-run proposal until the paired clip probe has posted its numbers**
   (sampled success and group-drop rate with vs without the clip set the refill cost of a clipped
   run). The worst case is
   estimated: A100-80GB x 41.3 h = $74 per 150-step seed. `rl.gate_stop` is false for this first
   run (owner 2026-11-08), so the flat gate logs at step 38 and does not stop it. Budget the full
   run. (A stop at step 38 would have been 10.5 h = $18.72.)

**Approving a gate** (CLAUDE.md rule 1, all pod jobs, `scripts/pod/approval_lib.sh`): the gate
passes only on `/workspace/APPROVED.<RUNPOD_POD_ID>` written after the pod's first boot
(`/root/pod_start_epoch`). A file from an earlier pod on a reused volume, or the old unscoped
`/workspace/APPROVED`, is ignored, and the job log lists it with the reason. On the owner's
explicit go, the agent runs this over SSH on the pod, from the repo root:

    bash -c '. /root/runpod.env; . scripts/pod/approval_lib.sh; approval_write "owner go <date>: <what>"'

For a pre-approved pass, run the same command before the job reaches its gate (for example right
after launch).

**Pod deletion backstop** (2026-11-08, $0; checked against a hang-injection simulation, not on a
pod). Every new pod uses `scripts/pod/pod_start_cmd.sh` as its start command: the whole file is
the one string after the `["/bin/bash","-c"]` entrypoint, with no inline command. Its order is:
- First it arms the hard-cap timer, before apt runs. The timer fires at first boot +
  `SPEND_CAP_SECONDS` - `CAP_PRESTART_S`. `CAP_PRESTART_S` defaults to 180 s and covers billed
  time before the container starts. That figure is an estimate.
- The first-boot epoch is stored per pod id on `/workspace`, so a container restart re-arms the
  timer with only the time left.
- A cap that is not an integer deletes the pod at once.
- At the cap, the timer runs `delete_self.sh`. It tries REST v2 DELETE, then GraphQL
  `podTerminate`. Every curl has `--max-time 30`. A GraphQL reply counts only if the body holds
  `podTerminate` and no `errors`. It reinstalls curl if it is missing and repeats every ~20-80 s
  until the container dies.
- Then apt runs under `timeout`, then a key check that writes `/root/backstop_status`
  (`KEY_OK` / `KEY_BAD` / `KEY_UNVERIFIED` / `NO_CURL`, plus whether the timer is armed), then
  sshd.

The jobs (`diag_job.sh`, `demo_rebuild_job.sh`, `wo04_job.sh`) source `scripts/pod/job_lib.sh`
(and `approval_lib.sh` for the approval gate, above):
- The EXIT trap is installed before anything can fail.
- Every stage runs under `timeout -k 120`, capped at the soft deadline (hard cap - 1200 s).
- A watchdog sends TERM to the job 130 s after the soft deadline, which runs the trap.
- The trap's upload is limited to 600 s (`uv run --no-sync`). Its delete loop is limited to
  ~180 s, after which the timer keeps trying.
- Worst case from the soft deadline: 130 + 630 + 270 s < 1200 s.

Simulation: 47 cases at caps scaled to 120 s, with stub curl, uv and apt. They cover hangs in
setup, uv sync, download, probe, gate, sweep/build and both uploads, children that ignore TERM, a
step with no timeout, hung, failing or flaky deletes on either endpoint, a GraphQL error body, boot
apt hanging, and an image without curl. Results:
- 45 cases deleted the pod. The job deleted at or before the soft deadline + ~4 s scaled, or the
  timer deleted at the cap + 0.2-4.3 s scaled. In real time, each failed delete attempt costs at
  most ~80 s before the next one.
- The 2 failures (one per job) are an API that fails every call for the whole window. The loop
  kept trying (60-62 calls), but nothing inside the pod can fix that.

`tests/test_pod_backstop.py` (CPU, ~7 s) pins the main cases: a hung REST DELETE falls back to
GraphQL inside the cap, the restart re-arm works, and the start command's embedded copy of the
delete script still matches `delete_self.sh`.

Outside watchdog (2026-11-08, owner request, $0, checked against a fake API, not against RunPod):
- **Why it exists.** The in-container timer starts only once the container boots. A Mac check at
  hard cap + 5 min could let the pod run past the cap.
- **What it does.** `scripts/pod/watchdog.py` (stdlib only) starts right after the create call
  and counts from `T_req`. When `--cap-s` is given, it tightens the deadline to the pod's
  `createdAt` + cap.
  - **Confirm at launch.** It re-reads the pod for up to 120 s until the key sees it under a
    `vla-rl-so101*` name (event `confirmed`). A pod that no source shows (wrong id, or not visible
    to the key) gives an alert and exit 3 at once; 401/403 from both REST APIs gives exit 5 at
    once. Both launchers count as armed only after `confirmed` (the Mac launcher waits for it; the
    GitHub run has a separate "Confirm pod <id>" step; the launch gate below checks both). It never deletes a pod it has not
    confirmed, because the RunPod account is shared.
  - **Gone needs two sources.** A REST 404 also means "not accessible to the caller" (v2 OpenAPI),
    so "gone" needs two of REST v2, REST v1 and GraphQL to agree (404, `TERMINATED` or a null
    pod), and a pod shown by any source counts as present.
  - **Watch.** It polls every 60 s. It exits 0 early only if the pod deleted itself: every read
    over at least 120 s shows it gone.
  - **Terminate.** At `DEADLINE` - 400 s it sends REST v2 `DELETE api.runpod.io/v2/pods/{id}`,
    then REST v1 `DELETE rest.runpod.io/v1/pods/{id}`, then GraphQL `podTerminate`. A DELETE 404
    does not count as accepted, so the next endpoint is tried. If an accepted delete leaves the pod
    in place, every retry sends all three. Both REST endpoints are in RunPod's public OpenAPI specs
    (api.runpod.io/v2/openapi.json and rest.runpod.io/v1/openapi.json, read 2026-11-08), and v2
    also returns `createdAt`. A pod counts as gone after two consecutive reads at least 10 s apart.
- **When it cannot verify the pod is gone.** If that has not happened by `DEADLINE`, it alerts
  (macOS notification and bell, or a GitHub `::error::` annotation). It keeps trying for 900 s from
  the later of `DEADLINE` and its own entry into the terminate phase, always at least one full
  attempt, and again after a clock jump (a laptop waking from sleep). Then it exits 2.
- **Safeguards.** A name without the `vla-rl-so101` prefix gives exit 4, and nothing is deleted.
  The key is read only from the environment and is redacted from the JSONL log.
- **Where it runs.**
  - The GitHub copy is `.github/workflows/pod-watchdog.yml`, on `workflow_dispatch`. The key is an
    environment secret of the `pod-watchdog` environment, limited to `main`, so no workflow on
    another branch can read it. The job has a 360 min timeout, refuses deadlines more than 5.5 h
    away, and has concurrency per pod id. It only works once this PR is on `main`. Its logs are
    public.
  - The Mac copy is `scripts/pod/watchdog_mac.sh`: it reads the key from the Keychain item
    `vla-rl-so101-runpod`, runs under `nohup caffeinate -i -s`, and exits 0 only after `confirmed`.
    On battery with the lid closed, the Mac still sleeps.
- **Start command change.** `pod_start_cmd.sh` now also takes `POD_DEADLINE_EPOCH` and fires at
  min(first boot + cap - 180, `POD_DEADLINE_EPOCH` - 480). A value that is set but is not a decimal
  epoch in seconds (text, milliseconds, or a leading zero, which bash reads as octal) deletes the
  pod at once. `/root/backstop_status` gains `cap_from=creation|first_boot`.
- **Tests and runbook.** `tests/test_pod_watchdog.py` (26 cases, ~20 s) and 4 new cases in
  `tests/test_pod_backstop.py` cover this. The launch order is in `docs/runbooks/pod-launch.md`.
- **Owner, once:** create the Keychain item, the `pod-watchdog` environment (deployment branch
  `main`) and its secret `RUNPOD_API_KEY`, ideally from a dedicated key. RunPod documents no
  pod-only key scope; Restricted keys are scoped per Serverless endpoint only.

Launch gate (2026-11-08, owner request, $0, checked against a fake GitHub API and fake `ps` /
`ssh` / Keychain, not live):
- **Why.** The old runbook took the newest `pod-watchdog` run (`gh run list -L 1`), which could be
  another pod's, and went on if only one watchdog confirmed.
- **What.** `scripts/pod/launch_gate.py` (stdlib). `dispatch` sends the workflow with a fresh nonce
  (`dispatch_id`; run-name `pod-watchdog pod=<id> dispatch=<nonce>`, step `Confirm pod <id>`) and
  `return_run_details: true`. It takes the run id from the dispatch response (`workflow_run_id`),
  or else from the one run of that workflow on `main` created after the dispatch with exactly
  that run-name; if none exists yet, the gate keeps resolving it until its deadline. It then verifies
  the run: run-name, event, branch `main`, workflow file, created after the dispatch.
  `gate` passes only when the GitHub run is in progress with `Confirm pod <id>` = success by
  `T_req` + 300 s and its `Watchdog` step running, AND the Mac watchdog's log has a `confirmed`
  line for this pod with this run's tag (`--run-tag`, new; `watchdog_mac.sh` writes
  `runs/watchdog/<id>.mac.env`), by the same deadline, from a live process that runs exactly
  that watchdog. On failure it writes no token and cleans up only the verified pod: the id needs
  this launch's record (`<id>.github.json` / `<id>.mac.env` with this deadline, written after
  `T_req` - 30 s), and the new `watchdog.py --terminate-now --created-after <T_req - 30>` (same
  name-prefix and two-source checks) refuses a pod created before it (gate exit 13, nothing
  deleted). Then the agent deletes via the connector and confirms the pod gone. A failed read
  (`gh` 5xx, `ps` error) is retried, not a verdict: it fails the gate only after 60 s unreadable
  past the deadline, or before a token write (review fix, 2026-11-08: one GitHub 502 in step 8
  had deleted a confirmed pod in the fake). On a pass, `gate --ssh-host` writes `/workspace/WATCHDOGS_OK.<id>` over SSH, and
  only if the pod's `runpod.env` names this pod. `diag_job.sh`, `demo_rebuild_job.sh` and
  `wo04_job.sh` refuse to start without a valid token (rc 8, checked before setup; the EXIT trap
  then deletes the pod). The in-container backstop (`pod_start_cmd.sh`, `delete_self.sh`,
  `job_lib.sh`) and the lr-sweep approval gate are unchanged.
- **Cost.** $0 now. At launch the gate normally passes within ~1-2 min, while the image is still
  pulling, so the caps are unchanged. A gate that fails at its deadline deletes the pod
  ~6.5 min after `T_req` (5.5 min + ~1 min of cleanup): ~$0.20 on the A100 ($1.80/h with
  storage), ~$0.10 on the 4090. A token write that fails after the full 900 s SSH wait costs up to
  ~$0.70 on the A100. Both are inside the firm cap.
- **Not verified live:** the dispatch response shape with `return_run_details`, the run-name and
  step-name expressions on GitHub, REST `createdAt` in the cleanup, and `ps` on the caffeinate pid. Checking them needs a dispatch once the workflow is on
  `main`; it was not done here.

Outside our control:
- The pod cannot delete itself in four cases:
  - RunPod's API is unreachable or refuses the pod-scoped key for the whole window;
  - the key or pod id is missing (`KEY_MISSING`/`KEY_BAD`);
  - the container restarts without a persistent `/workspace`;
  - apt cannot supply curl.

  In those cases the outside watchdog (GitHub and Mac copies) still deletes the pod, unless RunPod's
  API refuses every caller. Then only the owner in the console can stop it.
- RunPod's billing start and its teardown latency are not measured here. The estimate assumes
  billing starts no earlier than the create call and stops once the pod reads as gone.

Proposed, not run: a rerun of the existing SFT `step_004000` and the two RL step-150 checkpoints
under protocol 0.2. It is priced from the measured eval runtime of ~1.05 h per checkpoint on a
4090, plus ~5% for the clip. estimated: RTX 4090 x 3.4-3.8 h = $3.0-3.4. Adding two-task step 50
and PickPlace-only step 100 brings it to $5.0-5.3.

Unverified on GPU: gate behaviour and thresholds (`gate_min_gain` 0.03 now applies to sampled
rollout success, as a gain from the SFT policy, and needs the owner's confirmation, as does
`gate_flat_z` 0.5 (provisional default, simulated only); `gate_max_drop` 0.04, `gate_decline_z` 3 and
`gate_entropy_min` 0.5 are not measured on GPU; z = 3 is from a CPU simulation only), diag/sweep forward cost, the throughput for the 16 x 8 / 2-epoch shape, the render
hook timing on `physx_cuda`, the CUDA RNG restore (a GPU resume is not bit-identical), v2 oracle
success on `physx_cuda`, recorder tail stepping on the GPU env, render-gap replay and 7B token
comparison, and the 0.2 clip and end-flag capture on CUDA.

RL rollout action clip: see "Owner instructions 2026-11-08, round 3" below. **Owner decision
2026-11-08: ON at 4.0 deg/step in `configs/rl.yaml` and `configs/rl_pickplace.yaml`** (env action
only; PPO keeps the sampled tokens). The tiny preset and the code default stay null (off), as in
WO-04.
The robot items (servo temperature/current fault response, `park()` on healthy shutdown, camera
timestamp, placement grid, `wo07_real_eval.py`) are in PR 2 (`fix/robot-safety`, on PR #23's
branch; see "Robot safety hardening" below). The plan
revision for PR #25 is on `wo-recovery-sft`. PR #25's `configs/rl_pickplace_lr*.yaml` use the
removed `early_stop_*` / `peak_stop_step` keys and need renaming when it merges, or they fail fast.

### Owner instructions 2026-11-08, round 2 (blocking before any paid run; $0, CPU only)

On PR #28 (`fix/pre-run-hardening`). Nothing ran on a GPU. Protocol `0.1-draft` (`e9ce5bb47a88`),
`src/env/success.py` and every reported number are unchanged.

- **PPO mini-batches** (`src/rl/trainer.py` `ppo_minibatches`). Each epoch splits the kept
  groups into ceil(groups / (`ppo_mini_batch` / K)) equal whole-group mini-batches. Sizes differ
  by at most one group and no group is split. Trajectories are contiguous per group
  (e = g x K + k) and are gathered by group index. A full batch of 16 groups splits as before
  (8 + 8), so its gradients are unchanged. A refill-capped batch no longer gives a short tail its
  own full Adam step: 11 groups now split 6 + 5, where they used to split 8 + 3. Every consumer
  of the old count is unchanged, because the count is the same: ceil(trajectories /
  `ppo_mini_batch`) per epoch for the lr sweep's Adam-step count and epochs, the LambdaLR steps
  and the resume state. Only the sizes change. The new log key is `train/minibatch_groups`. The
  sweep `--estimate` uses the largest equal mini-batch, which gives the same number for a full
  batch. `tests/test_rl_minibatch.py` covers 1/7/8/9/11/16/17 groups and checks the trajectories
  each Adam step sees. ADR-0002 and `configs/rl.yaml` describe the rule.
- **Approval gates** (`scripts/pod/approval_lib.sh`, all five pod jobs): see "Approving a gate"
  above. `tests/test_pod_approval.py` checks that stale, unscoped and other-pod files are ignored
  and that a fresh scoped file passes. `job_lib.sh` is unchanged.
- **Spend gates observed, not trusted.** `rl.gate_stop: false` in `configs/rl.yaml` and
  `configs/rl_pickplace.yaml` for the first RL training run. `gate_flat_z` stays at 0.5,
  provisional. Every step logs `gate/would_stop` (0/1), `gate/would_stop_reason` (every gate that
  fired) and `gate/first_would_stop_step` (0 = none yet). `record.json` `gate_decisions` (also
  written for an interrupted run), the run summary and the console `GATE SUMMARY` line give the
  first step each gate would have stopped the run. They also give the flat gate's numbers at its
  check step and say whether it would have stopped the run there (step 38, where WO-04 seeds 1-2
  stopped under the older rule). The gate log is checkpointed, so a resumed run keeps it. A
  resume from a checkpoint without a gate log (written before this round) reports the steps
  before it as unknown (`gate_log_complete: false`, `gate_log_from_step`,
  `flat_check_unknown`), never as "not reached" or "would not have stopped".
- **Render-gap reporting** (`scripts/wo01r_rebuild_demos.py`). The headline is t = 0, the first
  frame of each episode, rendered from the reset state before any replay physics. It reports
  `pixels_t0`, `tokens_stored_vs_fresh_t0` (every episode's first frame) and
  `state_max_abs_diff_t0_rad`, next to the codec floor's t0 numbers. Pooled all-frame numbers
  (`pixels_all`, `tokens_stored_vs_fresh_all`) go in a separate table and JSON block. They are
  labelled because open-loop replay physics drift contributes to them. Each run carries a
  `comparison` label keyed on the renderer, not the sim device: macOS renders through MoltenVK,
  and a Linux CUDA box renders on the GPU even with `--device cpu`. So a pod run with CPU physics
  against the 0.1 set is still labelled the Mac-vs-GPU renderer gap. The Mac result above is a **same-renderer pipeline check**, and its run
  record now carries that note.
- **RL rollout action clip (stated explicitly).** The current RL rollouts do **not** apply the
  4 deg/step joint clip that protocol 0.2 sim eval applies. I checked `src/rl/rollout.py`
  `collect_rollouts`: it passes the policy's actions to `env.step` unchanged. The same function
  runs the in-training greedy evals and the temperature probe, so they do not clip either. The
  WO-04 runs and every 0.1 protocol eval ran unclipped too. Only `scripts/wo02_eval.py
  --protocol 0.2` clips (`src/eval/sim_eval.py` `clip_joint_step`, bound from
  `configs/real.yaml`). The new key `rl.rollout_action_clip_deg` makes this explicit. It is null
  (off) in `configs/rl.yaml`, which is the setting for the next run, and it keeps the current
  behaviour bit for bit. Setting it to 4.0 applies protocol 0.2's `clip_joint_step` to every
  executed rollout and in-training-eval action and logs `rollout/action_clip_frac`; the sampled
  tokens and log-probs are unchanged. **Owner choice:** off keeps the run comparable with WO-04
  and the 0.1 numbers; 4.0 matches protocol 0.2 and the real arm. The agent recommended off.
  **Superseded: the owner chose 4.0 on 2026-11-08 (round 3 below).**
- **Follow-ups.** `push_dataset` (every demo push, wo01r included) refuses the two 0.1 repos
  unless `allow_protected` (`wo01_build_env.py --overwrite-hub` only); wo01r `--push` to one stops
  before the build. With `data.require_tail_finished` (on in `demos_v2*.yaml`), the build drops a
  successful demo whose release/retreat did not finish (oracle still running at the tail cap, or a
  grasp task ending off the release target). It reports the demo per seed and records the next
  seed instead, so one bad episode no longer blocks pushing the valid ones. A shortfall below
  `demos_per_task` still blocks the push, before it runs, on both push paths (wo01r and
  `wo01_build_env.py --push`, which records `push_refused` and exits 1). The protected-repo
  check is case-insensitive, because Hub repo ids are. `load_sft_frames` refuses a v2 demo set (v2 oracle or a
  recorded tail) unless `data.chunk_pad: drop`.
- **Mac test note.** `.venv/bin/pytest` starts with `#!/bin/sh`, and macOS strips
  `DYLD_FALLBACK_LIBRARY_PATH` from it, so torchcodec cannot find FFmpeg 7 and the recorder tests
  fail. Run `uv run python -m pytest` on the Mac instead.
  `tests/test_pod_backstop.py::test_cap_timer_retries_and_survives_restart` is timing-sensitive.
  It failed once while a `uv sync` was running alongside and passed 3 of 3 times on its own.

### Owner instructions 2026-11-08, round 3: the RL action clip ON ($0, CPU only)

On PR #28. Nothing ran on a GPU; no RunPod resource was touched (one public read-only GraphQL price
query, 14:43:08Z: A100 PCIe / SXM Secure $1.79/h, RTX 4090 $0.89/h, unchanged). Protocol
`0.1-draft` (`e9ce5bb47a88`), `src/env/success.py`, `src/eval/sim_eval.py`, `src/eval/protocol_v02.py`
and every reported number are unchanged.

- **Decision (owner, overriding the agent's "leave it off"):** `rl.rollout_action_clip_deg: 4.0`
  explicitly in `configs/rl.yaml` and `configs/rl_pickplace.yaml` (inherited, repeated so it is
  explicit). The next run's preset is `configs/rl.yaml` (the diagnostics pod's `RL_CFG` default
  and the plan above); `rl_pickplace.yaml` carries the same setting should it be used. 4.0 =
  protocol 0.2's bound, min(4 deg, 60 deg/s / 15 Hz) from `configs/real.yaml`; a test checks the
  presets equal `protocol_v02.action_clip().bound_deg`. `configs/tiny.yaml` and the code default
  stay null.
- **One function:** the rollout already called protocol 0.2's `clip_joint_step`
  (`src/eval/sim_eval.py`); nothing moved, so the eval is byte-identical (sim_eval.py and the 0.2
  hash unchanged).
- **Env action only:** `src/rl/rollout.py` stores the sampled tokens and sampler log-probs before
  the clip; the clip changes only what `env.step` receives. Old log-probs (sampler or recomputed),
  ratios and advantages are on the sampled tokens. `tests/test_rl_action_clip.py`: stored =
  sampled decision by decision, the env got `clip_joint_step(sampled, prev)` and it differs from
  the sampled action, and an update on that batch scores only sampled token rows with an
  on-policy first-mini-batch ratio (< 1e-5) and `train/logprob_mismatch` < 1e-5.
- **Config hashes:** `rl.yaml` / `rl_pickplace.yaml` hashes change (`eb4c1be43f8b` for rl.yaml at
  this head). No reported number cites them: the WO-04 RL runs cite commits `7c896d0` / `4a76953`
  (their configs are reproducible from those commits; round 1 had already changed these files),
  and every reported eval used a `sim` / `sft` config, which this round does not touch.
- **Probes read the same preset:** `--probe-temperatures`, `--lr-sweep` (its batch) and
  `--probe-steps` roll out with `rl.rollout_action_clip_deg`; no separate default. Their lines
  print it (`TEMP-PROBE ... action clip 4 deg/step`, `LR-SWEEP batch ... rollouts with action clip
  4 deg/step`, `--estimate` prints `rollout_action_clip_deg`).
- **Intervention logging:** per RL step in `metrics.jsonl` (`rollout/action_clip_frac` env steps,
  `_joint_frac`, `_frac_by_joint` (6), `_mean_abs_delta_deg` / `_max_abs_delta_deg` = |a - ref| -
  bound over the changed entries, `_env_steps`); per temperature (`probe/action_clip_*`), in the
  lr sweep's batch stats (record.json and a `lr_sweep_batch` row in metrics.jsonl), and in both
  clip-probe arms. Counted over running episodes only, from the bound (`|a - ref| > bound`), not
  from `clip_joint_step(a) != a`, which is not bit-exact when the clamp does not bind.
- **Paired probe** (`scripts/wo03_rl.py --probe-clip [--probe-groups 32]`,
  `GRPOTrainer.probe_clip`): rollouts only, no update. Both arms play the same prompts (reset
  seeds, tasks round-robin) with the same sampling seeds (torch RNG reseeded per env batch),
  K = the preset's 8, T = the preset's 1.0. Arm on applies the clip; arm off sends the sampled
  action and counts how often the clip WOULD have changed it. Per arm: sampled success +- SE,
  group-drop rate (all-equal groups; all-0 / all-1), entropy, mean episode length, intervention.
  Paired (`src/rl/paired.py`; `src/eval/paired.py` is not on this branch): per-group success
  difference +- SE over groups (the unit: a group's K samples share a scene), the episode count
  table with an exact McNemar p (labelled optimistic: episodes in a group are not independent),
  the paired dropped-group table, and a pairing check (first-decision tokens identical, 1.0 on
  CPU). 32 prompts: the paired per-group difference has SE ~2.4 points at p ~0.13 (Beta
  concentration 2, K = 8), so a 5-point change shows at 2 SE; 64 prompts would cost +13 min /
  +$0.39 for SE ~1.7. Smoke: `tests/test_rl_clip_probe.py` (CPU, tiny preset + a 6-joint test env,
  ~1 s; the toy env's state is not a joint reading, so the clip cannot apply to it).
- **diag_job stage:** `probe-clip` runs after the temperature probe and BEFORE the lr-sweep gate,
  under the launch approval: its numbers are what the owner wants before approving the sweep, and
  its own gate would idle the A100 up to 30 min ($0.90) to guard a ~14 min ($0.43) stage. It
  uploads its record, and the gate prints the last 4 `CLIP-PROBE` lines next to `TEMP-PROBE`.
  `CLIP_GROUPS=0` skips it; the gpu-check stage `--estimate`s it first, which refuses a preset
  with a null clip. Caps: above (13800 / 12000 s). The sweep keeps priority: the job runs the clip
  probe only if, at that point, the time to the soft deadline covers the clip estimate x 1.25 +
  2 min + the sweep's need (estimate x 1.25 + 2 min) + the full 30 min approval wait unless the
  pass is pre-approved; otherwise it prints "clip probe skipped" and goes to the gate (a pod
  launched at the round-2 caps 12660 / 10860 s skips it rather than losing the sweep). A clip-probe
  failure other than a timeout continues to the gate (the gate then shows no `CLIP-PROBE` lines).
- **Gates unchanged:** `rl.gate_stop: false` for the first training run in both presets, and
  the gate-decision logging (`gate/would_stop*`, `record.gate_decisions`) is in place (checked).
- **Unverified on GPU:** the clip-probe pairing on CUDA (multinomial on the CUDA generator after
  `torch.manual_seed`; GPU kernel nondeterminism may lower the first-decision token match below
  1.0), the per-step intervention tensors' overhead on `physx_cuda` (one host copy per env batch),
  and how the clip changes the episode length, success and group-drop rate of the a4 SFT policy.

#### Round 3 review fixes ($0, CPU only)

- `src/rl/rollout.py`: clip intervention counted from the bound (`|a - ref| > bound`; delta =
  `|a - ref| - bound`), not from `clip_joint_step(a) != a`. That float comparison flagged 1-ulp
  rounding as clipping (measured: 10-19% of within-bound entries near 0.01-0.1 rad on a 1e-3 rad
  target grid) and pulled the mean |delta| toward 0. The executed action is unchanged
  (`clip_joint_step`); `src/eval/` untouched. Test: within-bound steps at poses 0.01-3 rad count 0.
- `scripts/pod/diag_job.sh`: the clip-probe time check above (sweep priority) and a non-fatal
  clip-probe failure (timeout still exits 22). With today's estimates (clip 714 s, sweep 1.15 h)
  the check needs 1012 + 5295 + 1800 s = 135.1 min left (105.1 min pre-approved). At the 13800 s
  cap (soft deadline 202 min after creation) the probe runs if it is reached by 66.9 min; it is
  planned at ~48 min (~59 min with the 25% margin). At 12660 s (soft deadline 183 min) only an
  on-plan pod (<= 47.9 min) runs it; later, it is skipped and the sweep still fits.
- `src/rl/trainer.py` `lr_sweep`: records the BATCH's clip (collected: the preset's; stored: its
  `sweep_batch.json`, written next to `sweep_batch.pt`, else "unknown"), `batch_source`, and a
  `lr_sweep_batch` metrics row with the batch's clip stats. `probe_clip`: builds every env it
  visits before any reseed or timer (an RNG-using env build no longer shifts arm off's first
  draws; build time no longer lands in arm off) and reports `probe/s_per_env_batch_after_first`
  (CUDA warm-up excluded).
- No cost change: the planned stages, caps (5940 / 13800 / 12000 s) and $ above are unchanged.

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
    approved scope). Superseded by the review round below: the docs now launch the hardware
    scripts as `.venv/bin/python scripts/...` and the scripts warn under a launcher.
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
  - Review of round 5 (same approval, $0, fakes and a pty only):
    - Ctrl-Z no longer throws away a half-typed answer (F2): the terminal's line discipline
      flushed the line on Ctrl-Z, so `q`, Ctrl-Z, ENTER at `ENTER starts [ENTER / q]` started a
      trial. `job_control` now sets NOFLSH on stdin's terminal for the session (the previous
      bit restored on exit, skipped once the foreground is lost: the terminal is the shell's
      again). Every prompt read still starts with `tcflush`, so nothing typed between prompts
      (also around a Ctrl-C) answers the next one. The notice says the typed line is kept.
    - `uv run` (F1): the checklist and the three scripts' usage lines now start the hardware
      sessions as `.venv/bin/python scripts/...`, where Ctrl-Z is only the notice;
      `require_terminal` prints `WARNING: this process ... shares its job with a launcher`
      when the process is not its job's leader (e.g. `uv run`), before torque on. Not done:
      disabling the terminal's suspend character (owner's call if wanted).
    - Teleop: `job_control` now wraps `main`'s whole `try / finally`, so the TORQUE OFF
      UNCONFIRMED message and the provenance write also run under it (F3); the foreground poll
      runs before each tick (`Session.stop_requested`), so the tick that sees the loss no
      longer sends one more goal (F3, test now expects 5 frames, not 6); a lost foreground
      stops the end-of-session park before its first step, as in the pilot and real eval (F4).
      The hang-up gap noted here (teleop's park still ran after SIGHUP / SIGTERM) is closed in
      round 7 below.
    - Tests: the pty test sends its Ctrl-Z 4 s into the 8 s support wait and bounds the timeout
      at 10.5 s (a reset deadline would give about 12 s); an in-process test sends SIGTSTP
      mid-wait from a thread and checks the original deadline; a pty test types `q`, Ctrl-Z,
      ENTER at the placement prompt and gets `operator_quit` with no policy motion (fails
      without NOFLSH); NOFLSH set / restored in-process; teleop's `main` runs on fakes with
      the session handlers checked at torque on, in the session and at the record write, and
      restored after; the uv-like launcher test checks the warning; the pty harness kills its
      shell and child also when its own start fails (F5).
- Owner decision of 2026-11-08, round 7 (#29, $0, fakes only): no teleop park after a hang-up
  or a lost foreground.
  - `end_session` (`scripts/wo06_teleop_record.py`, `_park`) no longer starts the park when a
    SIGHUP / SIGTERM (`INTERRUPTS.hangup`) or a lost foreground (`INTERRUPTS.lost_foreground`,
    or the `tcgetpgrp` poll) has been recorded. The recorded signal decides, also while the
    terminal still reports the process as its foreground. Before, `end_session` cleared
    `sess.stop` and the stop check looked at the foreground and `stop` only, so a hang-up
    recorded before the park let it run to the end. The session goes straight to the hold,
    the support prompt (it ends at once as `hangup` / `lost_foreground`), the grace period
    without stdin and torque off.
  - During the park, `Session.stop_requested` (now `Interrupts.terminal_gone()` or `stop`)
    runs before every goal write (`So101Driver.move_to`: one write per tick, the check first).
    A hang-up or a lost foreground stops the park before its next write; a write already under
    way when the signal arrives completes. `So101Driver.park` also checks `stop_fn` once more
    after the pose is reached, before its torque-off, so a stop during the last write ends in
    the hold (Goal := measured, no motion) too. That check also applies to the pilot and the
    real eval, whose `stop_fn` already covers Ctrl-C, hang-up and lost foreground.
  - Records (`meta/vla_rl_so101.json`, `session_record`): `session_end: released_unconfirmed`,
    `support_confirmed: false`, `support_release.cause` `hangup` | `lost_foreground` (the
    existing names; SIGHUP and SIGTERM share `hangup`), and new `park` (`parked`,
    `skipped_after_hangup`, `skipped_after_lost_foreground`, `interrupted_by_hangup`,
    `interrupted_by_lost_foreground`, `stopped_by_ctrl_c`, `not_completed`, or null when no park
    was due), `hangup_signal` (`SIGHUP` | `SIGTERM`) and `lost_foreground` (why). A healthy end
    still parks to the end and records `parked`.
  - Tests (`tests/test_teleop.py`; parking on, a fake follower that logs every goal write as
    motion / hold and every park stop check, in order; real `os.kill` signals through the
    session's own handlers; the real `end_session`, driver park and `timed_input`; a 30 s
    SIGALRM hard timeout per test): SIGHUP and SIGTERM before the park, with no terminal and
    with the terminal still reporting the process as foreground (`tcgetpgrp == getpgrp`): no
    park goal, no park stop check, hold, `hangup` at the support prompt, the 10 s grace, torque
    off, unconfirmed. The same four during the park (the signal sent from the fake's write hook
    after the 3rd park goal): the next stop check sees it, no goal after it but the hold, then
    the same path. A SIGHUP during the last (converging) write is caught by the driver's new
    check before torque-off. Lost foreground before the park (`tcgetpgrp` names another group)
    and during it (a real SIGTTOU, or the poll): the same path, cause `lost_foreground`. A
    healthy end writes all 8 park goals, each after a stop check, records `parked`, with no
    prompt. A SIGHUP at the scene prompt runs the whole `record_session` without a park. The
    before-park and last-write tests fail against `a041e46`. The documented hardware launch
    command stays `.venv/bin/python scripts/...`.
- Round 8 (#29, $0, fakes only): review of round 7.
  - F1: the park's stop check ran before the tick's blocking camera read (`frame_time`, up to
    ~33 ms) and the servo read, so a hang-up or lost foreground recorded in that window still
    sent one more motion goal (up to 4 deg). `So101Driver.step` now takes `stop_fn` and calls it
    after the joint read and the safety filter, immediately before `_write`; True raises
    `MotionStopped` with nothing written, and `move_to` returns False (park -> hold). The check
    before each tick stays (it also skips the camera read). Only a goal whose bus write had
    already started completes. The pilot and the real eval park through the same `move_to`, so
    they get it too; their own control loops (`src/robot/client.py`) are unchanged.
  - F2: the teleop sync and episode loops had the same window. `Session.tick` passes
    `stop_requested` to `arm.step`; on `MotionStopped` the sync aborts and the episode ends (no
    goal and no frame for that tick), then the existing hold / prompt path. A Ctrl-C mid-tick
    is now also caught before the write (one goal fewer than before; otherwise unchanged).
  - Tests: a real SIGHUP from inside the park tick's camera read and from inside its servo read
    (no terminal, and terminal still foreground): no 4th goal, arm at 12 of 30 deg, hold, grace,
    torque off, unconfirmed; SIGHUP during a sync tick's and an episode tick's frame read: no
    goal that tick (the episode keeps 2 frames, discarded at the hang-up). A SIGHUP between
    `_park`'s skip check and its `sess.stop = False` (sent from the "PARKS" print) writes
    nothing, and `stop_requested` is True for a recorded SIGHUP with `stop` cleared (both fail
    with round 6's `check_foreground() or self.stop`). A real SIGTTOU recorded before the park
    while `tcgetpgrp` still names this process skips it (fails with a live-poll-only skip). The
    "every motion write follows a go-on check" assertions no longer read `log[-1]` for the first
    write. The park tests run the driver on a FakeClock shared with the camera stamps, so each
    step is exactly 4 deg on a loaded machine. The new park/sync/episode window tests fail with
    the `step` check removed. `tests/test_driver.py`'s park stop sequence counts the extra check.
Unverified on hardware: every item above. In particular: the thresholds, the read cost on the
Mac's USB-serial link, whether the park pose rests safely with torque off, the refusal
tolerances against a real calibration, the Homing_Offset read on the real servos, the
behaviour of a real failed torque-off write, `select()` + `os.read` on the Mac terminal for
the bounded waits (teleop, pilot, real eval; a macOS pty passes in the tests), the terminal
and foreground checks under `uv run` in Terminal (is `uv`'s Python child in the foreground
process group?), Ctrl-Z in the Mac Terminal during motion and at a prompt (and NOFLSH keeping a half-typed line there), `tcflush` of typed-ahead input on the Mac terminal, SIGHUP
delivery to the Python child under `uv run` when the Terminal window closes, the port-busy recovery on a real interrupted packet, the capture offset, and the printed
grid's scale and alignment with the real base.

## PR 2: SmolVLA and ACT on the real arm (branch `feat/smolvla-real`, 2026-11-09, $0)

### PR 2: train

SmolVLA and ACT fine-tune code for the real arm (SmolVLA pivot handoff, PR 2 items 1-2), branch
`feat/smolvla-real-train` on `feat/smolvla-real`, 2026-11-09. $0: nothing was launched.

- `scripts/wo07_train.py --preset {tiny,real} --policy {smolvla,act}` calls LeRobot 0.4.4's
  `lerobot_train.train` in-process, the path `wo01_smolvla_smoke.py` uses. The configs are
  `configs/wo07_train.yaml` and `configs/wo07_train_tiny.yaml`, with `policies.<p>.base` from config.
  - Before training, the script refuses any dataset whose `meta/info.json` departs from
    `src/robot/convention.py`, and any whose `meta/vla_rl_so101.json` does not cite this convention
    hash. It also refuses rehearsal recordings.
  - SmolVLA gets `convention.SMOLVLA_RENAME_MAP`. LeRobot saves that map in the checkpoint's
    preprocessor, and the reload check verifies it.
  - Each run record holds the config hash, git commit, device, the convention citation, a dataset
    hash (SHA-256 over every file) and a checkpoint content hash (`src/policy/lerobot_ckpt.py`).
- Real defaults:
  - SmolVLA: `lerobot/smolvla_base`, 20k steps, batch 64, LeRobot's dtypes (the VLM in bf16), chunk
    50, 10 flow steps, images resized to 512.
  - ACT: LeRobot defaults (100k steps, batch 8, ResNet-18 ImageNet), with chunk 50 from the
    convention.
- `--probe` runs N timed steps and writes s/step, peak VRAM and the projected h and $ to
  `probe.json`. Exit code 3 means CUDA OOM.
- `scripts/pod/wo07_train_job.sh` runs one policy per pod, in this order:
  - checks the launch-gate token (before setup);
  - runs setup;
  - downloads the dataset at `DATASET_REVISION`;
  - runs the probe at batch 64, then 32, then 16 on OOM;
  - waits at the pod-scoped approval gate;
  - runs the full training with periodic uploads;
  - uploads the final checkpoint and runs/ to the private `KudratSingh22/vla-rl-so101-real-policies`;
  - deletes the pod.
  `DRY_RUN=1` runs every stage and gate on the Mac in a sandbox, on the `tiny` preset, and makes no
  apt, Hub or RunPod call (tested with failing stubs).
- Tiny check, Mac M3 Pro CPU, fp32. These are plumbing numbers, never results. Each run trains 3
  steps at batch 2, saves, reloads, and computes the same chunk on dataset frame 0:

  | Policy | Wall time | Median step | In-memory vs reloaded chunk | Note |
  |---|---|---|---|---|
  | SmolVLA (full `smolvla_base`, no reduced model) | 28.6 s | 1.6 s | 0.0 rad (same explicit noise) | a different noise seed moves the chunk by 1.20 rad |
  | ACT (small dims, no ImageNet download) | 9.5 s | 0.49 s | 0.0 rad | |

- Defect found and fixed (`lerobot_ckpt.load_checkpoint`): LeRobot builds SmolVLA's action expert in
  bf16, so `from_pretrained` rounded fp32 weights trained on CPU to bf16. The effect was 0.156 rad on
  a chunk after 3 tiny steps. The CPU path now loads the saved weights into the fp32 model. CUDA
  and MPS checkpoints are bf16 already and are not affected. `wo01_smolvla_smoke.py load_policy` has
  the same CPU issue and is left unchanged.
- Cost, PROVISIONAL. Reprice from the probe before requesting spend. The live RTX 4090 Secure price
  is $0.89/h (public GraphQL, 2026-11-09T05:35Z; community $0.34, A100 PCIe Secure $1.79). A 50 GB
  volume costs $0.10/GB-month, about $0.17/day while it exists.
  - SmolVLA reference: the guide quotes about 4 h on one A100, about $7.16 at $1.79/h.
  - The 4090 speeds below are not measured. They assume a 4090 step takes 1.5–2.5× an A100 step at
    batch 64, if that batch fits in 24 GB.

  | Line | Estimate | Assumption |
  |---|---|---|
  | SmolVLA, batch 64 | RTX 4090 × 6.35–10.35 h = $5.65–$9.21 | includes 0.35 h of setup |
  | SmolVLA, batch 32 (OOM fallback) | RTX 4090 × 3.35–5.35 h = $2.98–$4.76 | 0.54–0.90 s/step (half the batch-64 step), steps not rescaled, so half the samples; corrected in review round 1 (was 3.6–6.4 h = $3.2–$5.7, not derivable) |
  | ACT | RTX 4090 × 1.75–3.15 h = $1.56–$2.80 | 0.05–0.10 s/step guessed |
  | Probe, per pod | RTX 4090 × 0.5–1.0 h = $0.45–$0.89 | setup, probe and up to 30 min waiting for approval |
- Open:
  - Pin `dataset.revision` once the 50 demos are pushed.
  - Owner choice: if SmolVLA drops to batch 32, keep 20k steps (the guide's count) or rescale them.
  - Size `SPEND_CAP_SECONDS` from the probe.

### PR 2: infer

Branch `feat/smolvla-real-infer` (sub-branch of `feat/smolvla-real`, no PR), 2026-11-09, $0, mock
rig only, **unverified on hardware**. Handoff PR 2 item 4: the Mac inference path.

- `src/robot/local_policy.py`: `LocalPolicyClient`, an in-process SmolVLA / ACT client for the
  existing `DeployLoop` with `PolicyClient`'s contract (`ActReply` or `ServerFault`). Deadman,
  safety clip, servo-health torque-off, prompts and trial records are the loop's own, unchanged.
  Observations go through `convention.policy_observation` and actions through
  `convention.action_targets`. In between are the checkpoint's own LeRobot processors; for
  SmolVLA that includes `SMOLVLA_RENAME_MAP` when the checkpoint saved no map.
- Checkpoint identity: `checkpoint_sha256` over config, weights, both processor JSONs and the
  processor state files (normalization stats), plus `checkpoint_revision` (Hub snapshot commit).
  `local_policy.checkpoint_sha256` refuses a mismatch before the weights load and again in
  `check_local`, before the arm is opened.
- Deadline: inference runs on a worker thread with a deadline, `local_policy.request_timeout_ms`
  (700 ms, below `deploy.max_chunk_age_ms` 800). A late or hung call is `server_timeout`, which
  holds the arm. A hung call refuses every later query.
- `DeployLoop` now takes a `CameraRig` (`pipe` gets every camera's frame). `configs/real.yaml`
  gains a `local_policy` block, with `base: null` until a fine-tune exists.

Latency on the mock rig, measured with `scripts/wo07_local_latency.py --preset real`. **Single
run, mock rig** (fake arm, two fake 640x480 cameras; real deploy loop, safety filter, sim FK and
model). **Excludes camera capture.** Mac: Apple M3 Pro (Mac15,7, 36 GB), macOS 26.5.2, torch from
`uv.lock`, commit `32f057d`. 300 bus writes per run, 15 Hz (66.7 ms period). Each run warms up
twice before the first trial.

| Policy | Device | Chunk (in-process query) p50 / p95 / max ms | Model only p50 / p95 / max ms | Gap between bus writes p50 / p95 / max ms | Actions per chunk | Write Hz | Run |
|---|---|---|---|---|---|---|---|
| SmolVLA `lerobot/smolvla_base` (bf16) | MPS | 448.4 / 463.5 / 471.8 | 441.8 / 453.3 / 468.3 | 76.5 / 526.8 / 547.9 | 5.0 | 6.2 | `20261109T055550Z-wo07-local-latency-real-e73b` |
| ACT, fresh untrained init, LeRobot defaults, chunk 50 (fp32) | MPS | 52.5 / 60.8 / 66.6 | 44.6 / 51.5 / 52.8 | 74.9 / 124.1 / 136.0 | 10.7 | 12.8 | `20261109T055653Z-wo07-local-latency-real-c3a2` |
| SmolVLA `lerobot/smolvla_base` (fp32) | CPU fallback | warm-up 914 / 869 ms; first query timed out (`server_timeout`, hold), 0 writes | | | | | `20261109T055723Z-wo07-local-latency-real-aa28` |

- No trained checkpoint exists yet. The ACT row is a freshly initialised ACT: random ResNet18,
  placeholder stats centred on `home_qpos`. The SmolVLA row is the pretrained base. Neither
  model's actions mean anything for this arm. All actions went through the safety filter
  (SmolVLA: 193 clipped, 106 rejected). To time full chunks, the script sets
  `deploy.stop_on_reject: false` for itself only. A rejected tick still writes the filter's
  hold command on schedule.
- **SmolVLA on MPS takes about 450 ms per chunk.** With `max_chunk_age_ms: 800` it executes only
  5 of its 50 actions before re-querying. Writes then arrive at about 6 Hz: about 520 ms at each
  chunk boundary, about 77 ms inside a chunk. CPU fallback (about 0.87 s per chunk) misses the
  700 ms deadline, so it is not usable at these settings.
- **Write gap inside a chunk is about 75 ms, not 66.7 ms.** On this Mac `time.sleep` overshoots
  by about 8.8 ms p50 (60 x 60 ms sleeps, standalone). This affects the existing loop pacing on
  both the gRPC and in-process paths.
- Records show `dirty: true` for the second and third runs. The cause is the previous run's
  untracked record directory; the code is commit `32f057d`.
- Owner decisions before real SmolVLA trials: (a) overlap inference with execution (async
  chunking; the loop's documented WO-08 item); (b) raise `max_chunk_age_ms`, a safety parameter
  that is the owner's call; or (c) accept about 6 Hz. No change has been made.

### PR 2: eval

Branch `feat/smolvla-real-eval` (sub-branch of `feat/smolvla-real`, no PR), 2026-11-09, $0, CPU
only, fakes and the mock rig only (no arm, camera, serial port, Hub push or RunPod action).
Handoff final2 items 5-7 and the owner's amendments D1-D6. **Unverified on hardware.**

**Protocol `real-0.1`** (`src/eval/protocol_real.py`, new; 0.1-draft and 0.2 untouched, sim keeps
its objects). Composite hash `a39c1be29ec5f49d1bd2243842077eba2a1abf6690d12987d08fcf74a763d395`,
pinned in `tests/test_protocol_real.py`; it covers the slot ids and the rules that chose them, the
bowl and the constraint values it was derived from, the objects, the eval / demo / dev schedules
in full, the grid geometry, the judging rule and rubric, the horizon and settle, and the
convention hash (`b8701c57...`). Values, not file bytes.

| Item | Frozen value | Why |
|---|---|---|
| Seen slots | `B2 B5 E2 E5` (corners of the inner 4x4) + `C4` | D1: one near each corner, one central. The block has no centre slot; `C4` is the central slot whose bowl distance (22.3 cm) is nearest the corners' mean (20.9 cm), so the middle training position is also mid-range in carry distance |
| Bowl | (0.195, +0.205) m, fixed for every demo and trial | D2: derived (`derive_bowl`) on a 5 mm lattice: every one of the 36 slot centres >= MIN_SEP 11 cm + 5 mm away, >= keep-out 15 cm + 5 mm and <= PUSH_REACH 29 cm - 5 mm from the base; nearest the grid centre; tie to +y (front camera side). 11.75 cm from the nearest slot (C1), 28.3 cm from the base. Sim IK reaches it at release / carry / retreat heights (< 0.1 mm, tilt <= 20 deg). No position ahead of the grid fits (it would be >= 37 cm out) |
| Objects | seen red cube; unseen purple cube + green cube (3 cm, the red cube's material; only colour differs) | D3: replaces the green cylinder on the real arm only |
| Eval schedule | 150 trials, one seeded order across all cells; seen 10 x 5 slots; shifted 50 over the 20 ring slots (3 at every other slot around the ring, 2 at the rest); unseen 5 per slot per object | D3/D4; interleaving keeps drift off any one cell; `schedule_sha256` `dd1345de...` |
| Demo schedule | 50 = 10 per seen slot (all seven 15 deg yaws + 3), red cube, fixed bowl | D1 |
| Dev schedule | 30 trials (10 per cell), never reported; yaws on the half-step grid so no placement equals an eval placement | D5 |
| Horizon / settle | 450 steps (30 s at 15 Hz = the teleop demo cap) / 3 s | the 0.2 horizon (150 = 10 s) fits OFT sim parity, not a teleop-paced real task |
| Scoring | binary (real-judge-2 + rubric R1-R4); partial credit only as a secondary column from the blinded review | CLAUDE.md rule 4 |

Placement sheet regenerated (`docs/assets/placement-grid-a3.pdf`): the five SEEN slots outlined,
the 11 other inner slots "not used", the protocol hash, and a "bowl line" at x = 0.195 m: the bowl
centre is 56.5 mm beyond the A3 sheet's left edge, so the sheet prints a reference tick at
y = +0.105 m and the centre is 100.0 mm further along the line.

**Real eval audit** (final2 item 5; `scripts/wo07_real_eval.py`):

| Requirement | Before this branch | Now |
|---|---|---|
| Randomized trial order, seeded, recorded, same for both policies | partly: per-cell seeded slot permutation (`real_grid.sample_plan`), cells in fixed order, `placement_plan_sha256` per record, `--hardware` refuses another `--slot-seed` | real-0.1: one frozen order across cells, `schedule_pos` + `schedule_sha256` per record |
| Initial object pose per trial from the grid | yes: `placement` per record (`client.run_trials`) | `planned_*` keys + `pose_source` |
| Named labeler and success rubric in the record | judging rule text and hash only | `labeler` (`--labeler`, required on the arm) and `rubric` per record |
| Fault after start = failure with reason | yes (`client._score`, `trial_fault`) | unchanged |
| Resume skips started, retries never-started | yes (`done_indices`, `load_prior`) | unchanged; also re-hashes the checkpoint |
| Two policies | one `--label` | `--family {smolvla,act}` (stable `family_id`), `--training-run-id`, `--checkpoint-dir` (content hash in the fingerprint); `--report A B` refuses unless same protocol hash and schedule and different checkpoints |

Protocol 0.2 stays available as `--protocol 0.2` (the existing tests run it). On the arm, real-0.1
refuses `--trials/--cells/--slot-seed`, a missing labeler or policy identity, a settle other than
3 s and any convention departure (both cameras), before anything opens.

**Evaluation capture** (6b; `src/robot/recording.py`, new): one LeRobotDataset v3.0 per session at
`runs/<session>/capture/dataset`, episode = trial, written through the same
`convention.recording_frame` and `create_dataset` as the demos (the recorder now imports them).
`DeployLoop.on_write` (`src/robot/client.py`, a 10-line hook) hands each policy write's frame and
step result to the capture; the loop reads both cameras through `RigFrameSource` (a fault on either
is a lost frame). A not-started attempt is never an episode; a capture failure never reaches the
arm or the score (recorded in `capture.error`). After the score one frame of each camera is saved as
end-state stills for the blinded review. Resume appends (LeRobot's own resume path); a dataset left
unfinalized by a SIGKILL cannot be reopened and the resume says so. Measured on the M3 Pro:
0.40 ms per control tick, 4.5 s / 7.3 s to save a 10 s / 30 s episode of both cameras (single
run, mock rig, synthetic solid-colour / still frames). Bench budget: superseded in review round 1
(below): ~4.5 h for SmolVLA, ~2.9 h for ACT per 150 trials.

**Label provenance** (6c): `scripts/wo07_blind_review.py` (new) pools the captured episodes of one
or more sessions, shuffles them (seed recorded), and shows packets named by random ids (contact
sheet of both cameras decoded with torchcodec + end-state stills), never the policy. Its
`review.json` is the only input that makes an export declare `blinded_review`; live operator scores
and automatic fault outcomes stay in the trial records with their own `outcome_source`. Label
rule (decision, see open issues): `success` = every blinded reviewer said yes (R1) AND no fault
after the start (R2, from the recorded fault).

**Export** (7; `src/eval/export_bundle.py` + `scripts/wo07_export_bundle.py`, new): `demo` (dataset
from `wo06_teleop_record.py --demo-schedule`) or `eval --session <run>` writes `declaration.json`
(every section-2 field), `calibration.json` (the follower calibration copied into the dataset at
record time, sha-checked), `rubric.json`, `placements.json` (demos) or `trials.json` (eval: every
trial and attempt), `labels.json` only with `--review`, and for trained policies every checkpoint
file + `checkpoint_files.json` (`checkpoint_id = sha256:` over all files, matched against the hash
the session recorded). `object_pose` is null: only planned poses exist and vishwaroop reads
`object_pose` as a per-frame stream. Family ids `so101-real-smolvla` / `so101-real-act`; session id
= the recording session's id (demo: the recorder's run id; eval: the session's first run id).
Refuses a missing or unknown value, a mock session without `--allow-mock`, a dev session without
`--include-dev`, an incomplete session without `--allow-incomplete`, other weights, a protocol or
convention hash other than this code's. Sealing stays in vishwaroop.

**Mock export against the real importer (D6), 2026-11-09.** vishwaroop `ba0eef6`, copied (read
only; the original was never written) and run in the copy:

```bash
S=<scratchpad>; V="$HOME/Machine Learning Projects/vishwaroop"; C=$S/vishwaroop-copy
rsync -a --exclude .git --exclude .venv --exclude '/data/' --exclude __pycache__ \
  --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache "$V/" "$C/"
cd "$C" && uv venv --python 3.13 .venv && VIRTUAL_ENV=$C/.venv uv pip install -r requirements-dev.txt
# bundles: 2 demos via the real recorder path on fakes (--demo-schedule placements), a mock
# real-0.1 session (1 trial per cell, capture on), its bundle, a blinded review and a 2nd bundle
PYTHONPATH=src .venv/bin/python -m vishwaroop.data.so101_import seal --dataset <dataset> \
  --declaration <bundle>/declaration.json --extra <bundle>/<each file in bundle.json extras> \
  --export-id mock-demo-2026-11-09a --out $S/vw-exports        # and mock-eval-2026-11-09a / b
PYTHONPATH=src .venv/bin/python -m vishwaroop.data.so101_import verify --export $S/vw-exports/<id>
PYTHONPATH=src .venv/bin/python -m vishwaroop.data.so101_import import --export $S/vw-exports/<id> \
  --split config/splits/so101-mock-v1.json --repo-root "$C"
```

Split (in the copy): demo session `train`, eval session `dev`, distinct scene groups. Results:
all three seals `{"sealed": ...}` exit 0; verify exit 0 (demo: 2 episodes, cameras front + wrist,
teleop; eval: 3 episodes, trained); import exit 0 for all three (`audit.status:
declared_group_and_lineage_checks_passed`, contract v1 validator passed). Derived episodes:
demo `teleop`, `assessment_method: unknown`, `success: null`; eval (labels null) the same with
`family_id: so101-real-smolvla`; the labelled eval export `assessment_method: blinded_review`,
`success` true/true/false, `reviewer_ids: [mock-reviewer]`; streams `state`, `rgb_front`,
`rgb_wrist`. The real seal also refused, as the mirrored validator does, a unit `degrees`, a
missing `robot_id`, a teleop policy with a `family_id`, and a labels method `operator`.

**Gates:** full pytest 520 passed (+48 new: `tests/test_protocol_real.py` 13,
`tests/test_export_bundle.py` 35); ruff, black, mypy --strict src clean; no shell files changed.
`pyproject.toml`, `uv.lock`, `src/eval/protocol.py`, `src/eval/protocol_v02.py`, `src/env/success.py`
and every reported number untouched.

**Open issues for the owner:**
- Horizon 450 steps (30 s) and the label rule (blinded R1 AND no recorded fault) are frozen in
  real-0.1: confirm or overrule before the first demo (a change is a new protocol version).
- The deploy loop still feeds the policy the front camera at the 224 crop (`PolicyImagePipeline`);
  the two-camera SmolVLA / ACT input is the inference branch's (`RigFrameSource` already carries
  both images). Its checkpoint identity should hash the same files as `checkpoint_identity`.
- Bowl at 28.3 cm is 7 mm inside PUSH_REACH: reachable in the sim IK; check the real arm's reach
  over it during the camera match before recording.
- A capture refusal on resume (unfinalized dataset) happens after the arm is connected; the trial
  records stay intact, but the session's capture cannot continue.
- vishwaroop's ET-M5-1 wording is the owner's to amend there (final2).

### PR 2: integration

Branch `feat/smolvla-real` (the PR), 2026-11-09, $0, CPU / MPS, fakes and the mock rig only
(no arm, camera, serial port, Hub push or RunPod action). `feat/smolvla-real-train`
(`ba99ab2`), `-infer` (`c0362d8`) and `-eval` (`8d2fb2a`) merged as merge commits on Stage A
(`b7faab8`); the only conflicts were the three STATUS sections, combined here under one heading.
`src/robot/client.py` auto-merged (the infer branch's `PolicyBackend` / two-camera frame plus the
eval branch's `on_write` / `object_fn` / `record_hook`).

How the pieces were made to fit (all through `src/robot/convention.py`, `so101-real-obs-action`
1.0, hash `b8701c57...c912b5`, unchanged):
- **One checkpoint identity.** There were three (trainer: sorted globs incl. `train_config.json`;
  local path: a fixed list without it; export: every file in the dir). Now
  `src/policy/lerobot_ckpt.checkpoint_identity` is the only one: `local_policy.checkpoint_identity`
  is it, `export_bundle.checkpoint_identity` hashes and bundles exactly `lerobot_ckpt.checkpoint_files`
  (and refuses if its hash ever differs), and the Hub download of the local path fetches the same
  globs. Trainer run record = deploy `checkpoint_sha256` = session `policy_identity` = export
  `checkpoint_id` (`sha256:<same>`).
- **One loader.** `local_policy.load_local_policy` now loads through the trainer's
  `lerobot_ckpt.load_checkpoint` (new keyword args `config`, `n_action_steps`,
  `preprocessor_overrides`), so the CPU fallback gets the train branch's bf16 fix (fp32 weights
  re-read past LeRobot's bf16 action expert; `strict=False` as LeRobot's own `from_pretrained`).
- **Real eval through the local path.** `scripts/wo07_real_eval.py --backend {grpc,local}`:
  `local` loads the `--checkpoint-dir` checkpoint in process, refusing it unless it hashes to the
  session's `policy_identity.checkpoint_sha256`, then `check_local` (convention, chunk 50,
  protocol), all before the arm is connected; the loop reads the `CameraRig` itself (front +
  wrist, uncropped, `local_pipe`), deadline and chunk from `local_policy`. It is the only backend
  of a real-0.1 session on the arm (`grpc` refused there; the server serves OFT) and opt-in for the
  mock (default stays the gRPC MockPolicy); protocol 0.2 refuses `local`; a resume keeps the
  session's backend (older sessions = `grpc`). The evaluation capture (`open_capture`,
  `EvalCapture.on_write`) takes the rig from either source; `DeployLoop.on_write` is typed for the
  `RigFrame` the local path delivers. This closes the Stage A / eval open issue "the deploy loop
  feeds the policy the front camera at the 224 crop" for real-0.1. `scripts/wo07_pilot.py` is
  still gRPC / front only.
- **Recorder -> trainer.** The trainer reads a dataset written by the real recorder unchanged
  (tested: `dataset.synthetic: false` on a recorder session).

New test `tests/test_pr2_pipeline.py` (7; SmolVLA cases skip without the cached base, as on CI):
two demos through the real recorder on fakes (both cameras) -> `scripts/wo07_train.py --preset
tiny` ON that dataset (ACT and SmolVLA) -> one identity across trainer / local / export -> the
saved checkpoint loaded by `LocalPolicy` -> one `run_trials` tick through the real `DeployLoop`
on the fakes (ACT: the deploy-path chunk equals the trainer loader's `predict_chunk` within 1e-5
rad) -> `wo07_real_eval --mock --backend local` (1 trial per cell, capture on: 3 episodes, both
image keys, `check_dataset_info` passes) -> eval bundle of that session + demo bundle of those
demos through the importer mirror; plus the backend refusals. 6 pipeline cases in 77 s on the M3
Pro CPU.

**Mock export against the real importer (D6), again, on the integrated pipeline.** Same
vishwaroop copy (`ba0eef6`, original never written). Scratch builder
`<scratchpad>/mockexport-tools/make_pr2_bundles.py`: 2 demos (real recorder, fakes) -> tiny ACT
and tiny SmolVLA trained on them -> one mock real-0.1 session per family through
`--backend local` with `configs/real.yaml`'s `local_policy` as is (device `mps`) -> bundles; then
`<scratchpad>/mockexport-tools/run_importer_pr2.sh` (seal, verify, import with split
`so101-mock-pr2-v1`: demo session `train`, the two eval sessions `dev`). All 3 seals, 3 verifies
and 3 imports exit 0 (`audit.status: declared_group_and_lineage_checks_passed`); verify reports
cameras front + wrist for all three; derived eval episodes carry `family_id`
`so101-real-act` / `so101-real-smolvla` and `checkpoint_id` `sha256:c5348af7...` /
`sha256:2c6d5afd...`, equal to the training runs' `checkpoint_sha256`; labels undeclared
(`assessment_method: unknown`, no blinded review in this run). Mock only, never a result.

**Gates:** full pytest 550 passed after merging main (#30) (+7 in `tests/test_pr2_pipeline.py`), `ruff check src tests scripts`, `black --check .`, `mypy --strict src`
clean; no shell file changed. `pyproject.toml`, `uv.lock`,
`src/eval/protocol.py`, `src/eval/protocol_v02.py`, `src/eval/protocol_real.py`,
`src/env/success.py`, `src/robot/convention.py` and every reported number untouched.

**First paid run awaiting approval (provisional, reprice from the probe):** SmolVLA,
`scripts/pod/wo07_train_job.sh` `POLICY=smolvla`, batch 64 (32 on OOM), 20k steps, estimated
RTX 4090 (Secure $0.89/h) x 6.35-10.35 h = $5.65-$9.21, plus the probe per pod
($0.45-$0.89). Needs the 50 demos pushed and `DATASET_REVISION` pinned first. Not launched.

### PR 2: review round 1 (fixer)

2026-11-09, $0, CPU / MPS, fakes and the mock rig only (no arm, camera, serial port, Hub push or
RunPod action). Commits `dc7d09c` (the fixes) and `02d04ce` (a loader regression found while
re-measuring). All 18 confirmed findings applied; none skipped.

**Frozen values that changed (nothing was recorded under the old ones):**
- Convention `so101-real-obs-action` **1.1**, hash `c387f175...b807b30c` (1.0 `b8701c57...` kept
  pinned in the test as history). It adds one feature, `timing_s`, float32 `[write,
  camera_frame]`: seconds since the episode's first command write, at the write and at the camera
  grab (conv-3 / F6). A `[1]`-shaped scalar does not survive LeRobot 0.4.4's `save_episode` with
  numpy 2, hence the pair. It has no `observation.` / `action` prefix, so LeRobot never makes it a
  policy input.
- real-0.1 composite hash is now **`edee44aa...83f279d749`**: it covers the convention hash. The
  schedules are unchanged (eval `dd1345de...`, dev `21fe5598...`). The placement-grid PDF was
  regenerated.

**Per finding**
- **conv-1 / F1 / F8, checkpoint provenance.** The trainer writes `vla_rl_so101.json` into every
  `pretrained_model` dir, and it is part of the checkpoint hash. It holds the training run id, the
  dataset's convention citation, `unit_map` (`robot.units`: arm sign / offset), calibration
  sha256, dataset sha256 and session, the protocol citation and, for SmolVLA, the VLM. The VLM is
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, resolved to a commit at training with the sha256
  of every non-weight file (tokenizer, processor, configs), and re-checked after training.
  - `load_local_policy` refuses another convention hash or `unit_map`.
  - SmolVLA loads its VLM, processor and tokenizer from that exact snapshot
    (`lerobot_ckpt.pinned_vlm_path`; `vlm_model_name` and the tokenizer step are pointed at the
    local snapshot in memory only).
  - The real arm requires the provenance. `--training-run-id` must equal the provenance's run id
    (mismatch refused, F8). The real arm refuses an unverified run id; so does the export, except
    for mock sessions.
  - `local_mismatches` now compares the CHECKPOINT's cited convention, not this module's own.
  - The convention docstring now says that `arm_sign` -1 mirrors a joint rather than rescaling
    it.
- **conv-2, demo-only training.** `wo07_train.py` refuses any recorded dataset that is not the
  real-0.1 demo recording: purpose demonstration, `--demo-schedule`, this protocol hash, and every
  episode placement equal to a demo of the frozen schedule at a seen slot. Only the tiny
  synthetic dataset is exempt. The protocol citation is in the run record.
- **conv-4, camera roles.** `camera.confirm_camera_roles` runs before torque, in every demo sitting
  and every real-0.1 hardware session. It writes one labelled frame of each camera (FRONT /
  WRIST) as a PNG, opens it in Preview, and requires the operator to type `yes`; anything else
  ends with nothing recorded. The record (sheet, indices, time) goes into the sitting's provenance,
  the run record and the capture provenance. macOS gives no stable camera id through OpenCV, so
  this is an operator check, not an automatic one.
- **conv-5.** The unused `convention.task_string` was removed. The task string comes only from
  `protocol.instruction`, whose templates are in the convention hash.
- **F1 (high) + F5, durable recordings.**
  - `recording.DurableDataset` finalizes (Parquet footers) and reopens through LeRobot's resume
    path after every saved episode, so each episode goes to new files.
  - A kill can cost only the episode being saved. `recover_dataset` quarantines what that kill
    leaves (a footerless data file, a video no episode references, the frame-PNG folder) and
    rebuilds `meta/info.json` and `meta/stats.json` from the finalized episodes. Leftover PNGs
    would otherwise be encoded into the next episode with the same index.
  - The evaluation capture and the demo recorder both use it, and the provenance entry is written
    before each checkpoint.
  - A resumed real-0.1 session recovers and reopens its capture in `precheck_capture` before
    anything is opened.
  - `wo06_teleop_record.py --demo-schedule --resume` continues at the next demo after recovery.
    It refuses a different protocol, convention, `robot.units`, calibration, or a broken prefix of
    the schedule. The session id is kept and each sitting is listed under `sittings`.
  - Cost: about 2.3 s per 10-frame episode on the M3 Pro, including the reopen.
- **F2, device.** On the arm, MPS unavailable is refused (`DeviceUnavailable`), never a silent CPU
  run. The in-process device and dtype are in the session fingerprint's policy component, and the
  session run record's `device` is the policy's.
- **F2 (eval), blinding.** `wo07_blind_review.py` records each session's checkpoint hash. It
  declares `blinded_review` / `policy_identity_hidden` only when the clips pool at least two
  different policies; otherwise it writes `unblinded_clip_review`. The export requires at least
  two policies, this session's among them. The tests now review the SmolVLA and the ACT session
  together, and a single-session review is refused.
- **F3, retry in place.** `_run_schedule` re-runs a never-started item before the next one, up to
  `MAX_START_ATTEMPTS` = 3 per run; after that the run ends and a resume retries the item first.
  Every record carries `attempt`.
- **F3 (latency).** `wo07_local_latency.py --capture` measures with the evaluation capture on
  textured frames (table below).
- **F4.** A resume loads the checkpoint path it just verified. `resolve_checkpoint` treats
  anything path-like as local only and refuses it if it is not a directory, so a mistyped path is
  never fetched from the Hub.
- **F4 (budget).** The bench budget is now computed from the measured write rates (below).
- **F6 / conv-3, timing.** Each trial's capture block and `trials.json` carry `write_rate_hz`,
  `write_time_span_s` and `nominal_fps`. `bundle.json` warns when a rate is more than 10% off
  15 fps and notes that LeRobot's timestamps are nominal.
- **F7, capture order.** `DeployLoop.on_settled`, called at the end of the settle and before the
  servo read and score prompt, reads the end-state stills; the record says when they were taken
  (a faulted trial has no settle). The episode is saved after the scored record is logged and
  flushed, then a `capture` row logs its index; `load_prior` merges those rows. The export fills
  an episode whose capture row a kill lost from the capture provenance.
- **F9.** Development sessions are declared `protocol_version` `real-0.1-dev`. A rehearsal, or a
  recording without `--demo-schedule`, is refused by the demo export.
- **F10.** Cost lines corrected above: batch 32 = 3.35–5.35 h = $2.98–$4.76; ACT $1.56–$2.80.
- **Found while re-measuring (`02d04ce`).** In a fresh process, `load_local_policy` failed with
  "Couldn't find a choice class for 'smolvla'". The integration's loader no longer imported
  `lerobot.policies.factory` before `PreTrainedConfig.from_pretrained`. The tests had not caught
  it because the trainer had already imported the factory in the same process. Fixed, with a
  subprocess regression test that fails without the fix.

**Latency, with and without the evaluation capture.** Single run, MOCK RIG (fake arm, two fake
640×480 cameras; real loop, safety filter, sim FK and model). Excludes camera capture. Apple M3
Pro (Mac15,7), macOS 26.5.2, commit `02d04ce` (ACT runs at `dc7d09c`, same loop code), 300 bus
writes at 15 Hz, `max_chunk_age_ms` 800. Capture on = `EvalCapture` with LeRobot's 8 PNG threads,
on a ring of 30 random-noise frames per camera (the worst case for PNG and AV1), saved and
checkpointed after each 150-write trial. Untrained ACT and the pretrained `smolvla_base`: their
actions mean nothing for this arm.

| Policy | Capture | Chunk p50 / p95 / max ms | Gap between writes p50 / p95 / max ms | Actions per chunk | Episode save + checkpoint (150 frames) | Run |
|---|---|---|---|---|---|---|
| SmolVLA `smolvla_base` (bf16, MPS) | off | 439.7 / 484.1 / 527.4 | 71.8 / 517.2 / 596.4 | 5.56 | n/a | `...075558Z-...-c7cf` |
| SmolVLA `smolvla_base` (bf16, MPS) | on | 433.9 / 476.4 / 535.7 | 71.7 / 507.3 / 607.6 | 5.66 | 4.99 / 5.10 s | `...075655Z-...-42ad` |
| ACT, untrained (fp32, MPS) | off | 53.8 / 62.1 / 74.6 | 71.6 / 121.8 / 130.8 | 10.71 | n/a | `...075342Z-...-564e` |
| ACT, untrained (fp32, MPS) | on | 45.3 / 61.5 / 70.9 | 71.5 / 114.1 / 139.5 | 10.71 | 5.28 / 5.50 s | `...075412Z-...-9df9` |

In this single run, the capture's PNG threads did not push SmolVLA's chunk p95 toward the
700 ms deadline. A save of a full 450-frame trial was not measured here; the earlier synthetic
figure was 7.3 s.

**Bench budget (estimate, from these single-run write rates).** A trial is 450 writes with no
success latch. SmolVLA writes about 6.2–6.6 per second, so about 70 s of policy; ACT writes about
12.8–13.2 per second, so about 35 s. Adding home ~5 s, place ~15 s, settle 3 s, score ~5 s and the
save ~7 s gives:
- SmolVLA: about 108 s per trial, about 4.5 h for 150 trials.
- ACT: about 70 s per trial, about 2.9 h for 150 trials.

The 450-step horizon is a fixed number of commands, so it covers different wall-clock times for
the two families (stated in `scripts/wo07_real_eval.py` and `src/eval/protocol_real.py`). The
SmolVLA session needs two bench sittings through `--resume`, or a command-rate decision first.

**Mock export through vishwaroop's own tools, re-run on round 1.** Same scratch copy
(`<scratchpad>/vishwaroop-copy`, vishwaroop `ba0eef6`; the original repo was never written).
- Commands: `uv run python <scratchpad>/mockexport-tools/make_pr2r1_bundles.py <worktree>
  <scratchpad>/mockexport-pr2r1`, then `bash <scratchpad>/mockexport-tools/run_importer_pr2r1.sh
  <scratchpad>`.
- Inputs: 2 demos through the durable recorder; tiny ACT and tiny SmolVLA trained on them (with
  provenance); one mock real-0.1 session per family through `--backend local` (MPS, capture on);
  a blinded review pooling the two sessions (6 clips, reviewer `mock-reviewer`); and one ACT dev
  session.
- Results: 4 seals, 4 verifies and 4 imports, all exit 0, `declared_group_and_lineage_checks_passed`.
  Every bundle has both cameras and streams `state`, `rgb_front`, `rgb_wrist`.
- The eval episodes import with `assessment_method: blinded_review` and `checkpoint_id`
  `sha256:60f1845c...` (ACT) / `sha256:2f5846da...` (SmolVLA), equal to the training runs'
  hashes; the bundled checkpoint files now include `vla_rl_so101.json`.
- The dev session imports with `protocol_version` `real-0.1-dev`; the demos import as teleop.
- Mock only, never a result.

**Gates.** Full pytest 564 passed + 1 skipped (the SmolVLA-only case on its ACT parameter) at
`2afaf5c`; shellcheck on `scripts/pod/wo07_train_job.sh` clean. `ruff check src tests scripts`, `black --check .` and
`mypy --strict src` are clean. No shell file changed. `pyproject.toml`, `uv.lock`,
`src/eval/protocol.py`, `src/eval/protocol_v02.py`, `src/env/success.py` and `docs/results.md`
are unchanged.

**Owner decisions still open (before the first demo):**
1. 15 Hz instead of LeRobot's 30 fps (Stage A).
2. The 450-step horizon and the label rule (blinded verdict AND no recorded fault).
3. The SmolVLA command rate: overlap inference, raise `max_chunk_age_ms` (a safety parameter),
   or accept about 6 Hz with a 4.5 h session over two sittings.
4. Whether to keep 20k steps if SmolVLA drops to batch 32.

## Blockers and decisions pending the owner

- Resolved 2026-11-09 by the pivot (ADR-0003): the OFT track is stopped, so the WO-04 item below
  is closed. Open for the owner: the seen slots, bowl position and schedule freeze (PR 2), the
  servo P gain to use (LeRobot writes 16 at connect, firmware default 32; WO-06 checklist), and
  amending vishwaroop's ET-M5-1 ("the GRPO-tuned policy") in that repo.
- (closed) WO-04: GRPO has not produced a clear gain (PickPlace seen +3.7 to +5.7 points across four
  checkpoints, no peak). The 75-step PickPlace-only proposal at lr 5e-4 ($36–50 estimated on
  2026-10-31) remains on hold. The core plan prioritizes the hardware baseline and recovery SFT
  before approving a higher-LR diagnostic. WO-05 paid ablations remain deferred pending the
  sim results and a separate scope decision.

## Owner decisions in effect

- 2026-11-09 Pivot to SmolVLA + ACT on the real SO-101 (ADR-0003); decisions D1-D6
  (`docs/reviews/2026-11-09-pivot-decisions.md`): five seen slots x 10 demos, bowl fixed, cells of
  50 binary trials, freeze before recording, development trials on a disjoint schedule, mock
  export through vishwaroop's seal and import tools from a scratch copy before the demos.

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

- 2026-11-08 RL action clip ON: `rl.rollout_action_clip_deg: 4.0` (protocol 0.2's bound) in
  `configs/rl.yaml` / `rl_pickplace.yaml`, env action only, PPO on the sampled tokens; the probes
  use the same setting; a paired clip on/off probe runs before the lr-sweep gate; no full-run
  proposal until its numbers are in (ADR-0002 amendment 3). `rl.gate_stop: false` for the first
  run.

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
