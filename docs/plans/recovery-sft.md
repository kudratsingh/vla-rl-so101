# Plan: recovery SFT, then a conditional higher-LR GRPO diagnostic

Status: **plan + local code, no paid run approved.** Owner request 4 (2026-11-07). Allowances
(docs/STATUS.md): $30 for recovery SFT + matched sim evaluations, $60 for the higher-LR GRPO
diagnostic, $210 for up to three confirmatory seeds. Every paid step below still needs the
owner's explicit go on its own probe numbers (CLAUDE.md rule 1).

**Revision 2026-11-08** (project review `docs/reviews/2026-11-07-project-review.md`, sections 3.1,
8 and 9; PR 1 `fix/pre-run-hardening`, #28):
- The lr-escalation rule and every other rule keyed on the logged `train/clip_frac` /
  `train/approx_kl` are replaced by PR 1's fixed-batch pre/post-update diagnostic (section 11).
  With one PPO epoch the logged metric is structurally ~0 and uninformative; PR 1 removes it.
- The step-20 checks use PR 1's rollout-window flat gate, not the 64-episode kill switch.
- Every protocol eval must write per-episode records, so the comparisons can be paired (sections 9
  and 10).
- The matched ordinary-demo control (section 7) is unchanged.

## 0. Order (review Track B)

The review orders Track B as: $0 fixes (PR 1), then ~$5 of diagnostics, then this plan on a
rebuilt base. Its decision 3 is the owner's to make. If approved, the base is rebuilt **before**
failure mining:

1. Rebuild the 25 demos/task on the GPU renderer with PR 1's fixed recorder, which records through
   release and retreat with no pad frames in the SFT loss, and the v2 oracle. The tool is
   `scripts/pod/demo_rebuild_job.sh`, and the output goes to the new repo
   `KudratSingh22/vla-rl-so101-demos-25-v2` (~$0.7 on a 4090). The job's render-gap check runs
   first.
2. SFT from scratch on that set (A100, 8000 steps, ~$5), then the full protocol eval (~$1).
3. Then sections 3-10 below, with that new base in place of `step_004000`: mine failures, record
   corrections and the matched control on top of it, and warm-start both arms from it. Its own
   protocol table is the baseline in section 9.

Without that decision the plan runs as written, from `step_004000`. Either way the corrections and
the control are recorded on the same pod and backend.

## 1. Question

The current SFT checkpoint (WO-02 attempt 4 `step_004000`, 25 demos/task) reproduces its demos
but does not recover once it drifts off them (ADR-0001: few-shot covariate shift). GRPO on it gave
+3.7 to +5.0 points on PickPlace seen (results.md). The logged clip fraction 0.000 and
approx_kl ~3e-5 are uninformative with one PPO epoch (review 3.1), so they say nothing about the
size of the updates.

Recovery SFT asks: if the expert labels the states this policy actually fails in (DAgger-style
corrections from representative failure states), does SFT success go up, by more than the same
amount of ordinary expert data? If yes and the result meets the RL entry rule, is GRPO from that
base at a higher learning rate worth a confirmatory run?

Baseline, protocol `0.1-draft` `e9ce5bb47a88`, 3 seeds x 100 episodes, run
`20261027T190836Z-wo02eval-sim-cdd3` (single SFT run): seen PickPlace 17.0%, Push 10.0%, Stack
2.7%. It meets the amended rule 3 (>= 10% on >= 2 of 3 tasks), with Push exactly at 10.0%.

## 2. Held-out seeds and the leak this plan avoids

**Known leak.** The WO-02 sweep that picked `step_004000` ran `wo02_eval.py --cells seen --seeds 0
--episodes 50` (scripts/pod/wo02_job.sh), i.e. protocol seed-0 seen episodes 0-49. The baseline
checkpoint was chosen on part of the test set. Its protocol seen numbers carry a small optimistic
selection bias, which makes the comparison below conservative. Every comparison is also reported
on protocol seeds 1-2 only, the part of the table the selection never saw.

This plan never selects, mines or trains on protocol seeds. `src/eval/seed_ranges.py` lists every
range and `check_disjoint` / `assert_seeds_in` enforce it. The collection script calls them before
any rollout, and `scripts/wo03_rl.py` now rejects ManiSkill RL seeds whose training prompts
would land in another range (RL seed >= 10 overlaps the protocol, `check_rl_seed`).

| Range | Seeds | Used for |
|---|---|---|
| demo[0] | 1,000,000 + k, k < 100,000 | WO-01 demos (10 and 25 per task) |
| rl_train[s] | s x 10,000,000 + counter (< 1,000,000) | GRPO prompts, s in {0, 1, 2, 3} |
| protocol[0..2] | 100,000,000 + 1,000,000 x s + ... | protocol eval only |
| selection[7], [8] | 107,000,000 / 108,000,000 + ... (same formula, protocol seeds 7, 8) | checkpoint selection |
| recovery | 3,000,000,000 + 1,000,000 x task_index + k | SFT failure mining (seen cell only) |
| control | 3,100,000,000 + 1,000,000 x task_index + k | matched-control oracle demos (seen cell only) |
| rl_eval | 4,000,000,000 + i | GRPO in-training greedy evals (sanity logs) |

Failure states come only from the seen cell. Shifted and unseen-object cells stay pure
generalization tests. No episode of any protocol eval run is ever mined.

## 3. Failure-state sampling and taxonomy

`scripts/wo02r_collect_corrections.py --preset sim` rolls `step_004000` out greedily, exactly as
the protocol eval does (whole 8-action chunks, horizon 150, GPU sim). It uses 192 recovery seeds
per task (3 batches of 64). `run_batch_traced` (src/eval/sim_eval.py) is the protocol's episode
loop plus a read-only trace. It records qpos, the privileged `obs["extra"]`, every `info` flag
and the executed action at each step, and an env snapshot plus the last command at every chunk
boundary. A test checks that its outcomes match `run_batch` exactly. ManiSkill's reset runs no
physics step, so its contact query returns the previous episode's last step. The trace therefore
records `is_grasped = False` at step 0 (a fresh episode is never grasped; a test grasps, resets
and checks it). At ~10-17% baseline success, about 160 failures per task are expected for
PickPlace and Push, and more for Stack.

`src/data/recovery.py classify` names each failure and the step of its decisive event:

| Task | Mode | Rule (first that applies) | Event |
|---|---|---|---|
| PickPlace, Stack | held | still grasped at the final step, timed out without a release | final step |
| | dropped | grasped at some point, last grasp ends away from the goal | last grasped step |
| | misplaced | last grasp ends within 9 cm of the bowl / 4 cm of the base | release step |
| | knocked | object moved > 2 cm without ever being grasped | first > 1 cm move |
| | stalled | timed out, arm still (< 0.01 rad) over the last 16 steps (checked before held, and after knocked when nothing was grasped) | stall onset |
| | never_grasped | none of the above | closest TCP approach |
| Push | lifted | the `ever_lifted` latch fired (success now impossible) | first lifted step |
| | no_contact | object moved < 1 cm | closest TCP approach |
| | missed | moved, not into the target | closest object-goal approach |

The per-task counts of each mode are the failure taxonomy. They are logged in the run record
(`failure_modes`, `unrecoverable`, `reset_branch`) and reported before any training.

## 4. Branch states and oracle corrections

`choose_branch` picks the chunk boundary one boundary (8 steps) before the event (`branch_back:
1`) and moves earlier while the state is unrecoverable. It never picks the reset state itself:
an oracle run from reset is a fresh demo, i.e. the control arm's data, not a correction. A state
is unrecoverable when the object is out of reach (> 0.31 m), the bowl is tipped, the Stack base
is off the table or tipped, or the Push latch has already fired. Failures with no recoverable
boundary (`unrecoverable`) and failures recoverable only from reset (`reset_branch`, e.g. an
event in the first two chunks) are counted per mode and dropped. `interleave` takes at most 24
branches per (task, mode), the first in seed order, and orders them round-robin across modes. Any
prefix of the recorded corrections therefore keeps the mode mix, and that matters once the
load-time cap binds.

Each branch is restored (`src/env/state.py restore`): reset with the source episode's seed, which
rebuilds the task bookkeeping, then `set_state_dict`, then restore the Push latch, then a fresh
rendered obs. A round-trip check compares the restored sim state with the snapshot (tolerance
1e-4) on every batch and aborts on a mismatch, including on GPU. ManiSkill 3.0.1 caveats are
handled as documented in `src/env/state.py`:

- controller state is not restored, which is harmless for absolute `pd_joint_pos`;
- GPU `env_idx` restores need exactly the masked rows, so we always reset and restore the full batch;
- latency-DR history is not restored, so restore is refused when latency DR is on (it is off in sim.yaml);
- contact flags are stale right after a restore, so the oracle gets the `is_grasped` recorded at the boundary;
- `elapsed_steps` restarts, so each branch gets a fresh 150-step budget;
- snapshots never cross CPU and GPU sim.

The scripted oracle takes over with `reset(start_phase="infer", grasped=..., grip_cmd=...)`
(src/data/oracle.py). An object already in hand enters "lift" and is carried on. Everything else
re-plans from "hover". The oracle's first gripper command continues the policy's last one, so a
held cube is not dropped on the first step. A test restores a mid-carry state and checks the
oracle finishes without a re-grasp. A correction is kept only if the env's own success check
fires (rule 5). It is recorded from the restore point onward, never the policy's actions, in the
WO-01 frame convention (state = qpos at t, action = oracle command at t). It goes to a separate
dataset `KudratSingh22/vla-rl-so101-corrections-a4` (private) with sidecar `kind=correction`,
`failure_mode`, `event_step`, `branch_step`, `in_hand`, `source_policy` and `backend`. Per mode,
the oracle yield (successes / tried) is reported.

Caveat: corrections come from GPU PhysX, while the 25 demos were recorded on Mac CPU PhysX. The
matched control is recorded on the same pod and backend as the corrections, so this difference
does not separate the two arms.

## 5. Data mixing (fixed now)

The SFT set is the 25 demos/task (4,964 frames, about 1,655 per task) plus corrections, capped
at **60 episodes and 1,650 frames per task**. The cap is the longest prefix in recording order
(`select_budget`, src/data/oft_dataset.py). Frames are sampled uniformly (unchanged
`sft.batches`), so corrections are at most about half of each task's frames. No sampler weights
are used. `scripts/wo02_sft.py` runs the 1% normalisation-bounds clip check on the whole mix and
on each dataset separately. If displaced-arm corrections fall outside the adapter's bounds, the
run fails instead of clipping the labels.

## 6. Training: warm start, step budget (fixed now)

Both arms **warm-start from `step_004000`**, with the same lr 5e-4 (constant after 100 warmup
steps, fresh AdamW), batch 8, seed 0, the adapter's own norm bounds and **2,000 steps** (16,000
sampled frames: about 1.6 passes over about 9.9k frames when the 1,650-frame cap binds, the
expected case; up to about 3.2 passes over the 4,964 demo frames if few corrections are kept).
The budget is fixed in steps, not passes, so both arms and the cost stay unchanged whatever the
yield. A checkpoint is saved every 500 steps (`configs/sft_a5_recovery.yaml`).
Why warm start, not from scratch:

1. The corrections label this policy's own failure states. Fine-tuning it on them is the DAgger
   step being tested.
2. The GRPO diagnostic would start from the result. Keeping the lineage makes the comparison with
   the WO-04 runs (same `step_004000` origin) direct.
3. It costs a quarter of an 8,000-step retrain, so both arms plus three full evals fit in $30
   with margin.

The confound it introduces, 2,000 extra SFT steps, is shared with the control arm. Attempt 3
showed that extra steps alone did not lift closed-loop success (ADR-0001).

## 7. Matched control

`configs/sft_a5_control.yaml` is identical except for the extra dataset: fresh oracle demos from
reset on control seeds (`--mode control`, `kind=control_demo`, recorded on the same pod and
backend). Per task, it takes the longest prefix whose frames do not exceed what the recovery
config selects from the corrections (`match_frames_of`). The frame counts match to within one
demo (about 66 frames, < 5%), and both are logged in each run record. Recovery minus control
isolates "where the expert frames are" from "how many there are". The control holds about 50
demos/task, so it is a diagnostic arm only and never an RL base.

## 8. Checkpoint selection (non-protocol seeds)

For each arm, every saved checkpoint (500, 1000, 1500, 2000) is evaluated on the seen cell with
**selection seed 7** (`wo02_eval.py --cells seen --seeds 7 --episodes 100`, 3 tasks x 100). The
run is marked `protocol_complete: false`. The checkpoint with the highest 3-task mean is picked,
and a tie goes to the earlier step. This rule is fixed now. The protocol table is computed only
for the two picked checkpoints.

## 9. Evaluation on the unchanged protocol

The protocol file is unchanged (hash `e9ce5bb47a88`). On one 4090 pod and one commit, the full
protocol (3 tasks x 3 cells x 3 seeds x 100, horizon 150, GPU sim) runs for:

1. `step_004000` again: a reproduction check against run cdd3 and the paired baseline for drift.
2. The recovery pick.
3. The control pick.

All three runs play the same episode seeds (the protocol's seed formula), and each must write
per-episode records (`runs/<id>/episodes.jsonl`, PR 1: task, cell, seed, episode seed, success,
length, end reason and end-state flags). A run without them does not count for section 10.
PR 1's `wo02_eval.py` writes them under `0.1-draft` too. If the owner has frozen protocol 0.2 by
then, all three run under 0.2 instead. Either way, one protocol, one hash and one commit apply to
all three.

Every number cites the hash and its run id. Results are labelled single SFT run. Negative results
go to docs/results.md as well.

## 10. Decision rules (fixed before any data)

Primary metric: pooled seen success over the 3 tasks (900 episodes per checkpoint). The test is
**paired**: the arms play identical episode seeds, so differences are read from the per-episode
records (section 9) with a one-sided exact McNemar test on the discordant episodes. The
two-proportion test used in results.md is reported alongside. "p < 0.05" below means the paired
test.

- **Recovery improves**: recovery minus baseline (re-evaluated in step 9) is >= +3.0 points
  pooled seen over seeds 0-2 with p < 0.05 (primary; the seed-0 leak inflates the baseline, so
  this readout is conservative), **and** on protocol seeds 1-2 alone (600 episodes, the part the
  WO-02 selection never saw) recovery minus baseline is > 0 (no p requirement: 600 episodes is
  underpowered). Anything else fails the gate. Both readouts go into results.md. This precedence
  is fixed now; the owner approves spend, not the reading of the gate.
- **Due to recovery states**: recovery minus control is >= +3.0 points with p < 0.05. Otherwise
  the gain is labelled "more expert data". It still counts as an improvement for the next gate,
  but the label goes into results.md.
- **RL entry (CLAUDE.md rule 3, amended)**: the recovery pick's protocol seen success is >= 10% on
  at least 2 of 3 tasks.
- Improves **and** meets entry: propose the GRPO diagnostic (section 11), priced on the measured
  numbers below, and wait for the owner's go.
- Otherwise: stop. Record the negative result. No GRPO.

Shifted and unseen-object cells, per-task deltas, the failure taxonomy before and after (the
recovery pick re-mined on fresh recovery seeds is optional and unpriced), and the oracle yield
per mode are reported. None of them gates.

## 11. Conditional: higher-LR GRPO diagnostic (allowance $60)

Two stages, each its own owner go.

**Stage A: offline lr sweep on one fixed batch, then one matched probe (about $3-5).** From the
recovery pick, on one A100 pod:

```
uv run scripts/wo03_rl.py --preset rl --config configs/rl_pickplace.yaml --adapter <pick> \
  --seed 3 --lr-sweep 5e-5,2e-4,5e-4,1e-3 --sweep-adam-steps 8
```

This is PR 1's fixed-batch diagnostic. It collects one rollout batch and stores it as
`sweep_batch.pt`. For each lr it restores the same adapter and optimizer state, runs 8 Adam steps,
and scores the **whole** batch against the pre-update log-probs. It logs post-update k3 KL, clip
and outside-clip fraction, dlogp split by advantage sign, mean(A*dlogp), the greedy argmax flip
rate and grad_norm. Every lr sees the same batch, so lr is the only variable, and 5e-5 (the WO-04
lr) is the reference. Rule 1 picks the lr. Then one `--probe-steps 3` at that lr measures s/step,
the drop rate and the post-update `diag/post/*` medians on live batches.

**Stage B: continuation (about $9-30).** Proposed on the stage A numbers: 75 steps at the lr rule 1
picks, then the readout below.

Every setting, as `scripts/wo03_rl.py --preset rl --estimate` validates it:

- inherited unchanged from `configs/rl_pickplace.yaml` (WO-04's PickPlace-only run): K = 16, 8
  prompts per update (128 trajectories), 32 envs, temperature 1.0, horizon 120, recompute old
  log-probs, ppo_epochs 1, ppo_mini_batch 64 (two optimizer steps per update), refill cap 3,
  PickPlace trained, Push and Stack eval-only, 64-episode kill-switch evals and 32-episode periodic
  evals every 25 steps on rl_eval seeds, `early_stop_rule: any_task`, min improvement 0.03;
- changed: base = the recovery pick (`--adapter`); `seed: 3`; `lr` 5e-5 / 5e-4 / 1e-3; `steps:
  75`, `checkpoint_every: 25`;
- `early_stop_check_at_frac: 0.26`, so the kill switch is at ceil(0.26 x 75) = ceil(19.5) = step
  20 (the inherited 0.25 would give step 19; 0.2667 gives 21 in float arithmetic);
- `peak_stop_step: 0` (off): the inherited ref 50 / stop 100 fails `GRPOConfig.validate` at 75
  steps, and a peak stop at the last step stops nothing.

**After PR 1 merges**, several of these settings change:
- The removed keys raise an error that names their replacement. The configs map as follows:
  `early_stop_check_at_frac: 0.26` -> `gate_check_frac: 0.26`, plus `gate_window: 5` (a flat
  gate at step 20 comparing steps 16-20 with 1-5); `early_stop_rule` -> `gate_rule: any_task`;
  `early_stop_min_improvement` -> `gate_min_gain: 0.03`; `peak_stop_step: 0` -> the decline gate
  (`gate_max_drop`).
- `configs/rl_pickplace.yaml` then inherits PR 1's shape: 16 prompts x K=8, `ppo_epochs` 2 and
  periodic evals on PickPlace only. The three stage configs still differ only in lr.
- The stage B cost below is for the WO-04 shape. It must be repriced with `--estimate` on the
  PR 1 shape, which is about +25% per step in the worst case (16.4 vs 13.1 min/step).

The owner's diagnostic rules apply:

1. **Rule 1 (stage A, fixed-batch diagnostic).** Pick the **smallest** lr in the sweep at which
   the update moved the policy in the advantage's direction on the fixed batch. That requires all
   three of:
   - post-update mean(A*dlogp) > 0;
   - dlogp for A>0 above dlogp for A<0;
   - greedy argmax flip rate >= 1% of executed action tokens.

   An lr is too far if its post-update outside-clip fraction is > 0.2. If 5e-5 already qualifies,
   the lr stays at 5e-5. If no lr up to 1e-3 qualifies without being too far, stop and report:
   do not escalate past 1e-3. The thresholds (1% flips, 0.2 outside-clip) are the agent's
   proposal and must be confirmed by the owner before stage A runs. The logged
   `train/clip_frac` / `train/approx_kl` are not used: with one PPO epoch they are structurally
   ~0 whatever the update does (review 3.1), and PR 1 removes them.
2. **Rule 2 (stage B). Flat gate at step 20** (PR 1 `src/rl/gates.py`): stop unless at least one
   trained task's windowed `rollout/success_rate` over steps 16-20 is >= 3 points above its
   steps 1-5 window (`gate_window: 5`, `gate_check_frac: 0.26`, `gate_rule: any_task`,
   `gate_min_gain: 0.03`). The trainer implements this, so no separate job-script check is
   needed. This replaces both the 64-episode kill switch, which cannot tell +3 from 0 (review
   3.1), and the former clip-keyed rule 3. The periodic greedy evals are sanity logs only.
3. **Rule 3 (stage B, every step).** The run is also stopped if the post-update fixed-batch
   diagnostic (`diag/post/k3_kl`; argmax flips only with `diag_argmax: true`, about +31 s/step)
   collapses to the stage A 5e-5 reference level for
   10 consecutive steps, meaning the updates stopped moving the policy, or if PR 1's entropy gate
   fires (`gate_entropy_min`). The job script reads the diagnostic from `metrics.jsonl`. It must
   exist with a tiny dry run before stage B is proposed.

The readout is the final checkpoint's protocol table on a 4090. PickPlace seen (300 episodes)
versus its own base, from the same protocol run in section 9. Stage A isolates lr from the same
base. Stage B measures "GRPO gain from this base at this lr" over 75 steps.

## 12. Conditional: three confirmatory seeds (allowance $210)

These run only if the diagnostic's PickPlace seen is **>= +10 points over its base** (the owner's
+10 bar of 2026-10-28) with p < 0.05 and it passed its step-20 checks. Seeds 0, 1 and 2 use the
diagnostic's exact settings and step budget, run sequentially. Seeds 1-2 start only after seed 0
passes its step-20 flat gate, as in WO-04. At 75 steps that is the stage B config with only
`seed` changed. At 150 steps the flat gate needs `gate_check_frac: 0.133`
(ceil(19.95) = step 20; the inherited 0.25 would put it at step 38), and the decline gate
(`gate_max_drop`) replaces the old peak stop. Each seed gets a full protocol eval. Between +3 and
+10 points, the result is reported and the owner decides. Nothing runs automatically.

## 13. Cost, from measured numbers

Rates: RTX 4090 $0.89/h, A100 80 GB $1.79/h (STATUS.md, 2026-11-07). Measured inputs:

- a full protocol eval of one checkpoint (2,700 episodes) costs $0.93-0.94 on a 4090, about 1.05 h, or about 0.39 h per 1,000 episodes (results.md);
- 4090 SFT runs at 1.547-1.575 s/step (attempt 2/3 probes);
- pod setup + probe + one 30-min approval-gate timeout takes 0.58 h (the two gate-expired pods, about 0.08 h active; `APPROVAL_WAIT_S=1800`, scripts/pod/wo02_job.sh);
- WO-04 PickPlace-only took 30.5 h for probe + 150 steps + eval, so (30.5 - 0.85) / 150 = about 0.20 h per step on an A100 including its evals (pod `s2ljj3drwg0ov0`);
- the WO-04 probes took 0.80-0.86 h.

Recovery SFT + matched evaluations (one RTX 4090 pod, with gates after the probe and after collection):

| Step | Basis | h |
|---|---|---|
| Setup + probe: restore round trip on GPU, 1 rollout batch, 20 SFT steps; first gate wait | 0.58 h measured (incl. a 30-min gate timeout) | 0.6 |
| SFT rollouts, 3 x 192 recovery seeds | 576 eps x 0.39 h / 1,000 | 0.23 |
| Oracle branches (<= 3 x ~120) + control demos (3 x 60) + AV1 recording | estimate; the probe measures it | 0.5 |
| Second approval gate (after collection), up to its timeout | 30-min `APPROVAL_WAIT_S`; less if the owner answers sooner | 0.5 |
| Recovery SFT, 2,000 steps | 2,000 x 1.565 s | 0.87 |
| Control SFT, 2,000 steps | 2,000 x 1.565 s | 0.87 |
| Selection sweep, 8 checkpoints x 300 seen episodes (seed 7) | 2,400 x 0.39 h / 1,000 | 0.94 |
| Full protocol x 3 (baseline, recovery, control) | 3 x 1.05 h | 3.15 |
| Uploads and slack | estimate | 0.3 |
| **Total** | | **~8.0 h = ~$7.1** (pod cap $12, ~70% contingency) |

estimated: RTX-4090 x 8.0 h = $7.1, within the **$30** allowance.

GRPO diagnostic (A100 run + 4090 protocol eval):

| Stage | Exit | A100 h | $ |
|---|---|---|---|
| A (own go) | offline lr sweep, 4 lrs x 8 Adam steps (`--estimate`, PR 1), + 0.5 h setup | 1.7 | 3.0 |
| A | + one 3-step probe at the chosen lr | 2.55 | 4.6 |
| B (own go) | flat gate at step 20 | 0.85 + 20 x 0.20 = 4.85 | 8.7 |
| B | full 75 steps + 4090 protocol eval ($0.93 + setup) | 0.85 + 75 x 0.20 = 15.85 | 28.4 + 1.2 = 29.6 |
| A + B | worst case: sweep + probe + full run | 18.4 | ~34 |

estimated: stage A A100-80GB x 2.55 h = $4.6 (sweep + one probe); stage B A100-80GB x 15.85 h + RTX-4090 x 1.4 h =
$29.6. Together ~$34 worst case, within the **$60** allowance (pod cap $40).
`scripts/wo03_rl.py --estimate` gives 16.6 h / $29.7 for the 75-step run (worst-case refill), in
line with the measured rate.

Confirmatory seeds: 3 x $29.6 = **~$89** at 75 steps, within **$210**. If the owner wants 150 steps
per seed: 0.85 + 150 x 0.20 = 30.85 h = $55.2 + $1.2 = ~$56 per seed, **~$169** for three, still
within $210. A seed stopped at step 20 costs about $8.7.

All stages together, worst case from the caps: $12 + $40 + $169 = **~$221** of the $300 total
allowance.
Most likely, the plan stops at one of the gates well before that.

## 14. Owner decisions needed

1. **Few-shot accounting.** Do correction episodes count against the <= 25 demos/task rule
   (CLAUDE.md rule 3, amended)? Options:
   - (a) count each correction as a demo. Then the mix must shrink, e.g. 10 demos + 15 corrections/task.
   - (b) count by frames.
   - (c) treat corrections as a separate expert-labelling budget, labelled e.g. "25 demos + N oracle
     corrections (DAgger-style, M frames/task)".

   The agent's recommendation is (c). It keeps the comparison with the current base exact. Every
   table would carry the label. The control arm exceeds 25 demos/task under any reading, so it
   stays diagnostic only.
2. Go / no-go on the recovery pod (about $7, cap $12) after its probe numbers.
3. After section 10, two separate decisions: whether to run the stage A probes (about $3-5),
   then, on the probe numbers, whether to run the stage B continuation (about $9-30). Both stages
   share the $40 pod cap.
4. After section 11: whether to run the confirmatory seeds, at 75 or 150 steps. Whether WO-04's
   2-of-3-task benchmark needs a two-task (PickPlace + Push) confirmatory run instead. It would
   cost about the same, at about 0.18 h/step (A100 ledger, pod `9d8p7vyfyv6ca4`: (28.4 - 0.85) / 150).

## 15. Run sequence (after approval)

```
# pod (Linux, RTX 4090): setup as scripts/pod/wo02_job.sh, then
uv run scripts/wo02r_collect_corrections.py --preset sim --push                 # corrections
uv run scripts/wo02r_collect_corrections.py --preset sim --mode control --push  # control demos
uv run scripts/wo02_sft.py --preset sim --config configs/sft_a5_recovery.yaml
uv run scripts/wo02_sft.py --preset sim --config configs/sft_a5_control.yaml
for ck in <4 checkpoints per arm>; do
  uv run scripts/wo02_eval.py --preset sim --config configs/sft_a5_recovery.yaml \
    --adapter $ck --cells seen --seeds 7 --episodes 100
done
uv run scripts/wo02_eval.py --preset sim --adapter <step_004000>       # full protocol, baseline
uv run scripts/wo02_eval.py --preset sim --adapter <recovery pick>
uv run scripts/wo02_eval.py --preset sim --adapter <control pick>
```

The pod job script (probe stage, approval gate, upload, self-delete) follows the pattern of
`scripts/pod/wo02_job.sh`. It is written once the owner approves the plan.

## 16. Verified locally (CPU, tiny preset, $0)

- `scripts/wo02r_collect_corrections.py --preset tiny`: 2 tiny-random rollouts, both classified, 2
  branches restored with round-trip error 1.5e-10, 2/2 oracle corrections recorded (148 frames),
  about 23 s on an M3 Pro. `--mode control`: 2 control demos, about 6 s.
- `scripts/wo02_sft.py --preset tiny` on demos + corrections: 151 + 74 frames, per-dataset
  coverage 0.0 clipped, 4 steps.
- `scripts/wo03_rl.py --preset rl --estimate` on the three section 11 configs: `GRPOConfig`
  validates, RL seed 3 passes `check_rl_seed`, kill switch at step 20, evals at steps 0, 20, 25,
  50, 75; 16.6 h / $29.7 for 75 steps. `scripts/wo03_rl.py --preset tiny` (toy env, CPU): the GRPO
  loop runs end to end.
- Tests:
  - restore of swapped rows, the Push latch and a re-rendered frame;
  - traced versus untraced outcomes, on a fake env and on the CPU sim;
  - in-hand oracle takeover;
  - the taxonomy and branch rules (incl. held and the reset-state exclusion);
  - no stale `is_grasped` at a traced reset after a held grasp;
  - the seed guard;
  - capped and frame-matched loading;
  - a script smoke run.

  The GPU restore path cannot run on the Mac. The round-trip check in section 4 is its guard on
  the pod.
