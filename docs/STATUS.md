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
| 06 | part 1 merged, kit arrived, physical bring-up pending | #5 (merged) | **Out of order**: robot-side code (safety, units, driver, calibration/camera scripts) written ahead of hardware and WO-01–05. Calibration, camera match and teleop are pending verification. |
| 04 | **benchmark not met; next step waits for the owner**, **out of order** (started before WO-02 merged) | #11, #13, #14, #16 (merged) | Two-task seed 0 finished (PickPlace seen 22.0%, +5.0); seeds 1-2 stopped flat at step 38. PickPlace-only finished at 20.7% (+3.7), below the owner's 27.0% bar, so stratified multi-task is not run. `docs/results.md`, ADR-0002 amendment 2. |
| 05, 07, 08, 09 | not started | | |
| pre-run hardening | PR open, **waits for owner review** (touches the RL loop) | `fix/pre-run-hardening` | Review `docs/reviews/2026-11-07-project-review.md` section 9, $0. See "Pre-run hardening" below. |

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

Next paid run (proposed, not launched; each pod needs the owner's explicit go): the review's
diagnostics.

**FIRM CAP: $7.65 for the whole diagnostics pass** (was $7.42; re-priced 2026-11-08 with live
rates and storage, and with every cap now counted from the pod create call instead of first
boot). The cap is enforced by three layers: the job's own delete, the in-container hard-cap timer,
and the outside watchdog (`scripts/pod/watchdog.py` on a GitHub runner and on the Mac; see "Pod
deletion backstop" and `docs/runbooks/pod-launch.md`). Each layer fires before `DEADLINE` =
`T_req` + `SPEND_CAP_SECONDS`, where `T_req` is the epoch taken just before the create call.

| Pod | GPU $/h (live, Secure) | Storage $/h | `SPEND_CAP_SECONDS` (max billable) | **Firm cap** | Expected (planned) | +25% | + full gate wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 4090, render gap (`demo_rebuild_job.sh` to its gate) | 0.89 | 0.0139 | **5700** (1.583 h) | **$1.43** | 29 min = $0.44 | $0.55 | $1.00 |
| A100 80 GB, diagnostics (`diag_job.sh`), gate wait | 1.79 | 0.0139 | **12420** (3.450 h) | **$6.22** | 119 min = $3.58 | $4.47 | $5.37 |
| A100 80 GB, diagnostics, pre-approved (no gate wait) | 1.79 | 0.0139 | 10620 (2.950 h) | $5.32 | $3.58 | $4.47 | n/a |
| **Both pods** (with the A100 gate) | | | | **$7.65** | **$4.02** | **$5.02** | $6.37 |
| Both pods (A100 pre-approved) | | | | $6.75 | $4.02 | $5.02 | |

Sources and assumptions:
- GPU rates were read at 2026-11-08T13:32:45Z with RunPod's public, unauthenticated GraphQL
  `gpuTypes` query (read-only):
  - RTX 4090: Secure $0.89/h, Community $0.34/h;
  - `NVIDIA A100 80GB PCIe`: Secure $1.79/h, Community $1.19/h;
  - `NVIDIA A100-SXM4-80GB`: Secure $1.79/h, Community $1.39/h.

  The diagnostics timings come from WO-04 pods, which were A100-SXM4-80GB (ADR-0002, pods
  `5rdd45j801bk5b` and `gzxhwhjhdtc1kc`). Both A100 types cost $1.79/h on Secure Cloud. The plan
  uses Secure Cloud, as every earlier pod did.
- Storage rates are from RunPod's pricing docs (docs.runpod.io/pods/pricing, read 2026-11-08):
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
  holds only the first-boot epoch, `job.log`, `stage` and `APPROVED`. A persistent pod volume is
  needed so that a container restart does not re-arm a full cap.
- Max billable time = `SPEND_CAP_SECONDS`, assuming billing starts no earlier than the create call
  (`T_req` is taken before it) and per-second billing, as the docs say. The outside watchdog sends
  its first delete at `DEADLINE` - 400 s, and three worst-case attempt cycles (~128 s each:
  3 deletes x 15.5 s + 2 reads x 3 endpoints x 8.5 s + 3 x 10 s gap/pause) end by about
  `DEADLINE` - 16 s. The in-container timer fires at `DEADLINE` - 480 s, and the job's soft deadline
  falls 1200 s before that. The firm cap is therefore (GPU + storage) $/h x cap, as long as any one
  layer works. If none has verified the pod gone by `DEADLINE` (RunPod's API down for every caller),
  the watchdog alerts and keeps trying for 900 s more, which is +$0.23 on the 4090 and +$0.45 on the
  A100, outside the firm cap. After that, only the console can stop the pod.
- The caps grew by 5 / 3 min because they now count from creation, and the in-pod timer and the
  watchdog fire 480 / 400 s before the deadline. Each cap is planned x 1.25 + the full gate wait +
  the 20 min soft-deadline margin + the 480 s in-pod lead. The usable time from creation (87 / 199 /
  169 min) still covers 86.25 / 198.75 / 168.75 min. The expected cost is unchanged apart from
  storage.
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
2. One A100 pod, `scripts/pod/diag_job.sh` (new; seed 3, SFT adapter a4 `step_004000`):
   `--probe-temperatures 0.7,1.0,1.3 --probe-groups 64` (rollouts only, the short first step), then
   an approval gate (or the pre-approval file), then `--lr-sweep 5e-5,2e-4,5e-4,1e-3
   --sweep-adam-steps 8`, upload to `KudratSingh22/vla-rl-so101-rl` `diag/<pod>/`, self-delete.
   Planned 119 min = $3.56:
   - pod start to job launch 2.5 min (1.5-2.5 min: "pod up" minus job time, 4 WO-04 A100 pods);
   - setup stage 1.9 min (0.4-1.9 min, 5 A100 job logs); adapter download 0.8 min (0.25-0.8);
   - temperature-probe invocation: 15 GB base download + 7B load + 2 GPU env builds 1.5 min
     (1.3-1.4 min: probe run start to first step, minus its baseline eval; base fetch 45-57 s,
     env setup 5-7 s); rollouts 3 T x 16 env batches x 49.2 s = 39.4 min (49.2 s = the slowest
     measured 32-env x 120-step batch, pod `s2ljj3drwg0ov0`; `--estimate` says 44.6 s);
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
   temperature written into `configs/rl.yaml`, starting with `--probe-steps 3`. The worst case is
   estimated: A100-80GB x 41.3 h = $74 per 150-step seed, or 10.5 h = $18.72 if the flat gate stops
   it at step 38.

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

The jobs (`diag_job.sh`, `demo_rebuild_job.sh`, `wo04_job.sh`) source `scripts/pod/job_lib.sh`:
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
  `createdAt` + cap. It polls every 60 s and exits 0 once the pod is verified gone. At `DEADLINE` -
  400 s it terminates through REST v2 `DELETE api.runpod.io/v2/pods/{id}`, then REST v1 `DELETE
  rest.runpod.io/v1/pods/{id}`, then GraphQL `podTerminate`. Both REST endpoints are in RunPod's
  public OpenAPI specs (api.runpod.io/v2/openapi.json and rest.runpod.io/v1/openapi.json, read
  2026-11-08), and v2 also returns `createdAt`. A pod counts as gone after two consecutive reads at
  least 10 s apart (404, `TERMINATED` or GraphQL null); otherwise the delete is sent again.
- **When it cannot verify the pod is gone.** If that has not happened by `DEADLINE`, it alerts
  (macOS notification and bell, or a GitHub `::error::` annotation). It keeps trying for 900 s and
  then exits 2.
- **Safeguards.** A pod this key never saw gives exit 3, because a 404 can mean "not visible to
  this key". A name without the `vla-rl-so101` prefix gives exit 4, and nothing is deleted. The key
  is read only from the environment and is redacted from the JSONL log.
- **Where it runs.**
  - The GitHub copy is `.github/workflows/pod-watchdog.yml`, on `workflow_dispatch`, with
    `secrets.RUNPOD_API_KEY`, a 360 min job timeout, refusal of deadlines more than 5.5 h away, and
    concurrency per pod id. It only works once this PR is on `main`. Its logs are public.
  - The Mac copy is `scripts/pod/watchdog_mac.sh`: it reads the key from the Keychain item
    `vla-rl-so101-runpod` and runs under `nohup caffeinate -i -s`. On battery with the lid closed,
    the Mac still sleeps.
- **Start command change.** `pod_start_cmd.sh` now also takes `POD_DEADLINE_EPOCH` and fires at
  min(first boot + cap - 180, `POD_DEADLINE_EPOCH` - 480). A value that is set but is not an epoch
  in seconds deletes the pod at once. `/root/backstop_status` gains `cap_from=creation|first_boot`.
- **Tests and runbook.** `tests/test_pod_watchdog.py` (15 cases, ~9 s) and 3 new cases in
  `tests/test_pod_backstop.py` cover this. The launch order is in `docs/runbooks/pod-launch.md`.
- **Owner, once:** create the Keychain item and the repo secret `RUNPOD_API_KEY`, ideally from a
  dedicated key. RunPod documents no pod-only key scope; Restricted keys are scoped per Serverless
  endpoint only.

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

Deferred: the 4 deg/step clip inside RL rollouts (review Track B item 4), so RL rollouts and 0.2
eval do not yet see the same action limits; `src.eval.sim_eval.clip_joint_step` is ready to reuse.
The robot items (servo temperature/current fault response, `park()` on healthy shutdown, camera
timestamp, placement grid, `wo07_real_eval.py`) belong to PR 2, on PR #23's branch. The plan
revision for PR #25 is on `wo-recovery-sft`. PR #25's `configs/rl_pickplace_lr*.yaml` use the
removed `early_stop_*` / `peak_stop_step` keys and need renaming when it merges, or they fail fast.

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

- 2026-10-28 WO-04 follow-up (ADR-0002 amendment 2): no per-task advantage normalization; kill
  switch passes if ANY trained task is +3 at step 38; PickPlace-only run, success = final protocol
  eval, seen, PickPlace, 300 episodes, >= 27.0% (missed: 20.7%); stratified multi-task only if it
  cleared that bar.

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
