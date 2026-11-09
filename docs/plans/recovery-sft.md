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
- The step-20 checks use PR 1's flat gate on rollout success, not the 64-episode kill switch
  (since the merge of main below: a trend over steps 1-20, logged only while `gate_stop` is false).
- Every protocol eval must write per-episode records, so the comparisons can be paired (sections 9
  and 10).
- The matched ordinary-demo control (section 7) is unchanged.

**Sync with PR 1, 2026-11-08** (`fix/pre-run-hardening` merged into this branch; $0, CPU only):
- `check_rl_seed` is PR 1's: RL seeds outside [0, 10) are rejected before the overlap check.
- The three `configs/rl_pickplace_lr*.yaml` use PR 1's gate keys (`gate_check_frac: 0.26`,
  `gate_window: 5`) and inherit PR 1's `rl_pickplace.yaml` shape; their costs are repriced from
  `--estimate` (section 13).
- Corrections and control demos are recorded the way `scripts/wo01_build_env.py` records the
  config's demo set: same oracle profile and same tail past success (section 4).
- Failure mining plays protocol 0.2 (the 4 deg/step clip) and writes per-episode records; the
  three protocol evals run under `--protocol 0.2`; `src/eval/paired.py` computes the paired
  readout of section 10 from the per-episode records.
- With `chunk_pad: drop` (the v2 set), the correction cap and the control match count the frames
  that enter the SFT loss, not recorded frames (section 5).

**Sync with main, 2026-11-08** (main after #28's final review rounds, #24, #23 and #29 merged into
this branch; $0, CPU only). What changed for this plan:
- Flat gate: main's `src/rl/gates.py` no longer compares two windows. At the check step it fits a
  per-task least-squares trend of rollout success over every step 1..check_step. The gain is the
  fitted change from step 1 (the base policy) to the check step. The run stops only if, for every
  trained task, gain + `gate_flat_z` x SE < `gate_min_gain` (`gate_flat_z` 0.5, provisional,
  simulated only). `gate_window` now sizes only the decline and entropy windows (sections 11, 12).
- `gate_stop: false` in `configs/rl.yaml` and `configs/rl_pickplace.yaml` (owner 2026-11-08: the
  first RL run on this recipe observes the gates before they may stop a run). The
  `configs/rl_pickplace_lr*.yaml` inherit it. Every gate still evaluates and logs
  `gate/would_stop*` and `record.gate_decisions`, but **no gate stops a stage B run**. The step-20
  exit of sections 11-13 is now an owner decision (section 14, decision 5). The worst-case cost
  lines already assume the full 75 steps, so no cap changes.
- `rollout_action_clip_deg: 4.0` (owner 2026-11-08) is inherited by the three lr configs. RL
  rollouts, in-training evals and every probe (`--lr-sweep`'s batch, `--probe-steps`) clip the env
  action as protocol 0.2 does. PPO keeps the sampled tokens. The three lr arms are matched on it,
  but they are not like-for-like with WO-04, which ran unclipped.
- PPO mini-batches are equal whole-group splits (16 groups -> 8 + 8, unchanged for a full batch).
  `--estimate` re-run on this merge: 20.6 h = $37 for 75 steps, identical for the three configs;
  `--lr-sweep` 4 lrs x 8 Adam steps 1.15 h = $2.06 (was 1.20 h); `--probe-clip --probe-groups 32`
  0.20 h = $0.36, optional here (section 13).
- `load_sft_frames` refuses a v2 set unless `chunk_pad: drop`. It checks every part of the mix:
  the demos and each extra dataset, so corrections recorded under `configs/demos_v2.yaml` are
  covered too. `configs/sft_a5_*.yaml` (0.1 set, oracle v1, `chunk_pad: repeat`) pass unchanged.
- Per-episode records carry the reset's `instruction`. `end_reason` is success | terminated |
  horizon: a truncation counts as horizon, and the raw flag stays in `flags["truncated"]`. The
  collector passes the instructions through `run_batch_traced` too.

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
   protocol table is the baseline in section 9. The two arm configs are then copies of
   `configs/sft_a5_*.yaml` on the v2 set's `data` block (`configs/demos_v2.yaml`: oracle v2,
   15/45-frame tail, final success required, `chunk_pad: drop`) with `policy.adapter` = the new
   base and the cap of section 5 recomputed; they are written once that base exists.

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
would land in another range (`check_rl_seed` rejects any seed outside [0, 10); seed 10 would
overlap the protocol).

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
the protocol 0.2 eval does (whole 8-action chunks, horizon 150, GPU sim, the 4 deg/step joint clip
from `configs/real.yaml`; `--protocol 0.2` is the default). It uses 192 recovery seeds per task
(3 batches of 64), and writes one per-episode record per rollout to `runs/<id>/episodes.jsonl`
(the eval's record format, protocol seed -1) with the protocol version and hash in the run
record. `run_batch_traced` (src/eval/sim_eval.py) is the protocol's episode
loop plus a read-only trace. It records qpos, the privileged `obs["extra"]`, every `info` flag
and the executed (clipped) action at each step, and an env snapshot plus the last command at
every chunk boundary. Tests check that its outcomes, executed actions and end flags match
`run_batch` exactly, with and without the clip. ManiSkill's reset runs no
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
`failure_mode`, `event_step`, `branch_step`, `in_hand`, `source_policy`, `backend`, `protocol`
and `oracle_profile`. Per mode, the oracle yield (successes / tried) is reported.

Corrections and control demos are recorded the way `scripts/wo01_build_env.py` records the
config's own demo set (PR 1): the same `data.oracle_profile` and the same tail past the success
step (`data.post_success_steps` / `post_success_max_steps` / `require_final_success`, with the
`success_step`, `post_success_frames` and `oracle_finished` sidecar fields). For the as-written
plan (`configs/sft_a5_*.yaml`, base `step_004000` on the 0.1 set) that is oracle v1 and no tail,
the 0.1 recorder, so the frame counts and costs below are unchanged. On the rebuilt base
(section 0) the configs take the v2 set's settings (oracle v2, 15/45-frame release + retreat
tail, final success required, `chunk_pad: drop`), and corrections then also teach the full
release.

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

With `chunk_pad: drop` (any config on the v2 set) the loader leaves each episode's last
`chunk_len - 1` = 7 frames out of the loss, in every part of the mix, and the cap and the control
match (section 7) count those trained frames: n - 7 per episode. Counting recorded frames instead
would let the arm with more, shorter episodes (the corrections) lose more frames to the drop and
end up with fewer trained frames than its cap says. `configs/sft_a5_*.yaml` keep
`chunk_pad: repeat` (the 0.1 set), where both counts are the same.

Frame counts on the rebuilt base, measured on the Mac CPU sim (single local build of
`configs/demos_v2.yaml`, not pushed, run `20261108T095637Z-wo01-sim-45cc`; the GPU-sim build
will differ slightly): the v2 set has 7,174 frames (PickPlace / Push / Stack 2,362 / 2,531 /
2,281), 6,649 trained frames under `chunk_pad: drop` (2,187 / 2,356 / 2,106), against 4,964
(about 1,655 per task) in the 0.1 set. The same "corrections <= about half of each task" rule
then gives a cap of about 2,100-2,350 trained frames per task, set from the GPU-built set's own
per-task counts, and 2,000 steps (16,000 samples) are about 1.2 passes over the full mix. Push
reached its 45-frame tail cap before the oracle finished in 14 of 25 demos, and 6 Push
candidates were rejected because success no longer held at the end of the tail (PR 1's
recorder; the oracle success rate is still 100%).

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
**selection seed 7** (`wo02_eval.py --protocol 0.2 --cells seen --seeds 7 --episodes 100`, 3
tasks x 100, the same clip as the protocol eval). The run is marked `protocol_complete: false`. The checkpoint with the highest 3-task mean is picked,
and a tie goes to the earlier step. This rule is fixed now. The protocol table is computed only
for the two picked checkpoints.

## 9. Evaluation under protocol 0.2

`src/eval/protocol.py` (`0.1-draft`, hash `e9ce5bb47a88`) is unchanged and every reported number
keeps citing it. The paired comparison runs under PR 1's protocol 0.2 (`src/eval/protocol_v02.py`:
0.1-draft's tasks, cells, seeds and episode seeds, plus the real arm's 4 deg/step joint clip and a
composite hash). On one 4090 pod and one commit, the full protocol (3 tasks x 3 cells x 3 seeds x
100, horizon 150, GPU sim) runs with `scripts/wo02_eval.py --protocol 0.2` for:

1. `step_004000` again: the paired baseline. Its difference from run cdd3 (0.1-draft, no clip) is
   the clip plus any drift, not a reproduction check. PR 1's proposed 0.2 rerun of `step_004000`
   counts as this run only if it ran at the same commit and hash.
2. The recovery pick.
3. The control pick.

All three runs play the same episode seeds (the protocol's seed formula), and each writes
per-episode records (`runs/<id>/episodes.jsonl`, PR 1: task, cell, seed, episode seed, success,
length, end reason, clipped steps and end-state flags). A run without them, with
`protocol_deviations` not empty, or with a different protocol hash does not count for section 10.
One protocol, one hash and one commit apply to all three. 0.2 is not frozen yet: the first number
reported from these runs freezes it, and its hash is then pinned in a test (PR 1).

Every number cites the hash and its run id. Results are labelled single SFT run. Negative results
go to docs/results.md as well.

## 10. Decision rules (fixed before any data)

Primary metric: pooled seen success over the 3 tasks (900 episodes per checkpoint). The test is
**paired**: the arms play identical episode seeds, so differences are read from the per-episode
records (section 9) with a one-sided exact McNemar test on the discordant episodes. The
two-proportion test used in results.md is reported alongside. "p < 0.05" below means the paired
test. `src/eval/paired.py` computes it from two eval run dirs and refuses runs whose protocol
version or hash differ or whose episodes do not match:

```
uv run python -m src.eval.paired runs/<baseline eval> runs/<recovery eval>               # seeds 0-2
uv run python -m src.eval.paired runs/<baseline eval> runs/<recovery eval> --seeds 1,2
uv run python -m src.eval.paired runs/<control eval> runs/<recovery eval>
```

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

**Stage A: offline lr sweep on one fixed batch, then one matched probe (about $5).** From the
recovery pick, on one A100 pod:

```
uv run scripts/wo03_rl.py --preset rl --config configs/rl_pickplace_lr.yaml --adapter <pick> \
  --lr-sweep 5e-5,2e-4,5e-4,1e-3 --sweep-adam-steps 8
```

This is PR 1's fixed-batch diagnostic. It collects one rollout batch and stores it as
`sweep_batch.pt`. For each lr it restores the same adapter and optimizer state, runs 8 Adam steps,
and scores the **whole** batch against the pre-update log-probs. It logs post-update k3 KL, clip
and outside-clip fraction, dlogp split by advantage sign, mean(A*dlogp), the greedy argmax flip
rate and grad_norm. Every lr sees the same batch, so lr is the only variable, and 5e-5 (the WO-04
lr) is the reference. Rule 1 picks the lr. Then one `--probe-steps 3` on the config of that lr
(`configs/rl_pickplace_lr5e-5.yaml`, `rl_pickplace_lr.yaml` = 5e-4, `rl_pickplace_lr1e-3.yaml`;
for 2e-4, a copy with only `lr` changed) measures s/step, the drop rate and the post-update
`diag/post/*` medians on live batches.

**Stage B: continuation (about $11-40).** Proposed on the stage A numbers: 75 steps at the lr rule
1 picks, then the readout below.

Every setting, as `scripts/wo03_rl.py --preset rl --estimate` validates it:

- inherited unchanged from `configs/rl_pickplace.yaml` (PR 1 shape, PickPlace-only): 16 prompts x
  K = 8 per update (128 trajectories), 32 envs, temperature 1.0, horizon 120, recompute old
  log-probs, `ppo_epochs` 2, `ppo_mini_batch` 64 (4 optimizer steps per update), refill cap 3,
  PickPlace trained, 32-episode periodic greedy evals every 25 steps on PickPlace only (sanity
  logs; Push and Stack are measured by the protocol eval), the fixed-batch diagnostic every step
  (32 trajectories), `gate_rule: any_task`, `gate_min_gain: 0.03`, `gate_flat_z: 0.5`, the
  decline gate (`gate_max_drop` 0.04, `gate_decline_z` 3), the entropy gate (`gate_entropy_min`
  0.5), `gate_stop: false` (the gates log `gate/would_stop*` and do not stop the run) and
  `rollout_action_clip_deg: 4.0` (protocol 0.2's clip on the env action);
- changed (`configs/rl_pickplace_lr*.yaml`): base = the recovery pick (`--adapter`); `seed: 3`;
  `lr` 5e-5 / 5e-4 / 1e-3; `steps: 75`, `checkpoint_every: 25`;
- `gate_check_frac: 0.26` with `gate_window: 5`: the flat gate is at ceil(0.26 x 75) = ceil(19.5)
  = step 20 (the inherited 0.25 would give step 19) and reads the trend over steps 1-20. The
  5-step window sizes the decline and entropy windows. This replaces the WO-04
  `early_stop_check_at_frac` kill switch (stop at step 20) and the `peak_stop_step` peak stop
  (off in these configs), which PR 1 removed. The trainer refuses the old keys and names the
  replacement.

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
   `train/clip_frac` / `train/approx_kl` are not used: with WO-04's one PPO epoch they were
   structurally ~0 whatever the update did, and at `ppo_epochs` 2 they average mini-batches
   whose first is exactly on-policy, so they are only weak evidence (review 3.1, configs/rl.yaml).
2. **Rule 2 (stage B). Flat gate at step 20** (`src/rl/gates.py`): flat unless, for at least one
   trained task, the trend gain of `rollout/success_rate` from step 1 to step 20, plus 0.5 x its
   SE, is >= 3 points (`gate_check_frac: 0.26`, `gate_rule: any_task`, `gate_min_gain: 0.03`,
   `gate_flat_z: 0.5`). This replaces both the 64-episode kill switch, which cannot tell +3 from
   0 (review 3.1), and the former clip-keyed rule 3. The periodic greedy evals are sanity logs
   only. **With the inherited `gate_stop: false` the trainer only logs the decision**
   (`gate/would_stop`, `gate/first_would_stop_step`, `record.gate_decisions`), so the run goes on
   to step 75. Making rule 2 stop the run needs `gate_stop: true` in the lr configs, or the job
   script stopping on the logged decision. That is the owner's call (section 14, decision 5).
3. **Rule 3 (stage B, every step).** The run is also stopped if the post-update fixed-batch
   diagnostic (`diag/post/k3_kl`; argmax flips only with `diag_argmax: true`, about +31 s/step)
   collapses to the stage A 5e-5 reference level for
   10 consecutive steps, meaning the updates stopped moving the policy, or if PR 1's entropy gate
   fires (`gate_entropy_min`; with `gate_stop: false` read from `gate/would_stop_reason` in
   `metrics.jsonl`). The job script reads both from `metrics.jsonl`. It must exist with a tiny dry
   run before stage B is proposed.

The readout is the final checkpoint's protocol table on a 4090. PickPlace seen (300 episodes)
versus its own base, from the same protocol run in section 9. Stage A isolates lr from the same
base. Stage B measures "GRPO gain from this base at this lr" over 75 steps.

## 12. Conditional: three confirmatory seeds (allowance $210)

These run only if the diagnostic's PickPlace seen is **>= +10 points over its base** (the owner's
+10 bar of 2026-10-28) with p < 0.05 and it passed its step-20 checks. Seeds 0, 1 and 2 use the
diagnostic's exact settings and step budget, run sequentially. Seeds 1-2 start only after seed 0
passes its step-20 flat gate, as in WO-04 (under `gate_stop: false`: its logged step-20
decision). At 75 steps that is the stage B config with only
`seed` changed. At 150 steps the flat gate needs `gate_check_frac: 0.133`
(ceil(19.95) = step 20; the inherited 0.25 would put it at step 38), and the decline gate
(`gate_max_drop`) replaces the old peak stop; with the inherited `gate_window` 10, step 20 is
exactly 2 windows, the minimum the trainer accepts. Each seed gets a full protocol eval. Between +3 and
+10 points, the result is reported and the owner decides. Nothing runs automatically.

## 13. Cost, from measured numbers

Rates: RTX 4090 $0.89/h, A100 80 GB $1.79/h (STATUS.md, 2026-11-07). Measured inputs:

- a full protocol eval of one checkpoint (2,700 episodes) costs $0.93-0.94 on a 4090, about 1.05 h, or about 0.39 h per 1,000 episodes (results.md);
- 4090 SFT runs at 1.547-1.575 s/step (attempt 2/3 probes);
- pod setup + probe + one 30-min approval-gate timeout takes 0.58 h (the two gate-expired pods, about 0.08 h active; `APPROVAL_WAIT_S=1800`, scripts/pod/wo02_job.sh);
- WO-04 PickPlace-only took 30.5 h for probe + 150 steps + eval, so (30.5 - 0.85) / 150 = about 0.20 h per step on an A100 including its evals (pod `s2ljj3drwg0ov0`);
- the WO-04 probes took 0.80-0.86 h;
- `scripts/wo03_rl.py --estimate` on `configs/rl_pickplace_lr.yaml` (PR 1 shape: 16 x 8,
  `ppo_epochs` 2, the per-step diagnostic; throughputs measured on WO-04 pod gzxhwhjhdtc1kc and
  WO-02 pod u87r9sjnwig58d, the scoring rate estimated): 16.43 min = 0.274 h per step worst case
  (refill cap every step), 20.6 h for 75 steps, 41.2 h for 150; `--lr-sweep` 4 x 8 Adam steps
  1.15 h (re-run on the merge of main, 2026-11-08: 1.20 h before the equal mini-batch split;
  the per-step and 75-step lines are unchanged). For the WO-04 shape the same estimator gave 0.221 h/step against the measured 0.20
  (16.6 h for 75 steps), so the worst case below is likely ~10% high.

Protocol 0.2 adds the 4 deg/step clip to every eval and to the failure-mining rollouts. PR 1
allows +5% eval time for it (the clip lengthens only successful episodes; not measured), applied
below to the rollouts, the selection sweep and the protocol evals.

Recovery SFT + matched evaluations (one RTX 4090 pod, with gates after the probe and after collection):

| Step | Basis | h |
|---|---|---|
| Setup + probe: restore round trip on GPU, 1 rollout batch, 20 SFT steps; first gate wait | 0.58 h measured (incl. a 30-min gate timeout) | 0.6 |
| SFT rollouts, 3 x 192 recovery seeds (0.2 clip) | 576 eps x 0.39 h / 1,000 x 1.05 | 0.24 |
| Oracle branches (<= 3 x ~120) + control demos (3 x 60) + AV1 recording | estimate; the probe measures it | 0.5 |
| Second approval gate (after collection), up to its timeout | 30-min `APPROVAL_WAIT_S`; less if the owner answers sooner | 0.5 |
| Recovery SFT, 2,000 steps | 2,000 x 1.565 s | 0.87 |
| Control SFT, 2,000 steps | 2,000 x 1.565 s | 0.87 |
| Selection sweep, 8 checkpoints x 300 seen episodes (seed 7, 0.2 clip) | 2,400 x 0.39 h / 1,000 x 1.05 | 0.98 |
| Full protocol 0.2 x 3 (baseline, recovery, control) | 3 x 1.05 h x 1.05 | 3.31 |
| Uploads and slack | estimate | 0.3 |
| **Total** | | **~8.2 h = ~$7.3** (pod cap $12, ~65% contingency) |

estimated: RTX-4090 x 8.2 h = $7.3, within the **$30** allowance. On the as-written path (oracle
v1, no tail) the recorded frames are those of the 0.1 recorder, so only the clip changes this
line (it was 8.0 h = $7.1). On the rebuilt base (section 0) the oracle branches and control
demos are recorded with the v2 oracle and the 15-45-frame tail, so the 0.5 h recording line
grows; the probe measures it.

GRPO diagnostic (A100 run + 4090 protocol 0.2 eval), PR 1 shape:

| Stage | Exit | A100 h | $ |
|---|---|---|---|
| A (own go) | offline lr sweep, 4 lrs x 8 Adam steps (`--estimate`), + 0.5 h setup | 1.15 + 0.5 = 1.65 | 3.0 |
| A | + one 3-step probe at the chosen lr: WO-04 probe overhead (0.85 - 3 x 0.218 = 0.19 h) + 3 x 0.274 | 1.65 + 1.0 = 2.7 | 4.8 |
| B (own go) | flat gate at step 20, **only if the owner sets `gate_stop: true` or stops on the logged decision** (inherited `gate_stop: false` runs on) | 0.85 + 20 x 0.274 = 6.3 | 11.3 |
| B | full 75 steps + 4090 protocol 0.2 eval (1.45 h = $1.3) | 0.85 + 20.6 = 21.4 | 38.4 + 1.3 = 39.7 |
| A + B | worst case: sweep + probe + full run | 24.1 | ~44.5 |

estimated: stage A A100-80GB x 2.7 h = $4.8 (sweep + one probe); stage B A100-80GB x 21.4 h +
RTX-4090 x 1.45 h = $39.7. Together ~$44.5 worst case, within the **$60** allowance but above the
former $40 pod cap: the stage A + B pod cap needs to be ~$46 (owner decision 3, section 14). At
the WO-04 shape this was 18.4 h / ~$34; the PR 1 shape costs ~25% more per step (16.4 vs 13.1
min).

Confirmatory seeds: 3 x $39.7 = **~$119** at 75 steps, within **$210**. At 150 steps per seed:
0.85 + 41.2 = 42.0 h = $75.2 + $1.3 = ~$76.5 per seed, **~$230** for three, **over** the $210
allowance in the worst case (75-step seeds, two 150-step seeds, or the owner raising the
allowance). A seed stopped at its step-20 flat gate costs about $11.3 (again only with
`gate_stop: true` or a stop on the logged decision). STATUS.md's allowance line for these seeds
is "up to three at $70 each": the 75-step seed ($39.7) fits it, a 150-step seed ($76.5) does not.

All stages together, worst case from the caps: $12 + $46 + $119 = **~$177** with 75-step seeds,
or $12 + $46 + $230 = ~$288 with 150-step seeds, of the $300 these three allowances total
(STATUS.md "Spend", 2026-11-08: inside the $700 core part of the $1,100 cloud compute cap; the
allowances themselves are unchanged). The optional `--probe-clip` (0.20 h = $0.36 on the stage A
pod) fits the ~$46 stage A + B cap.
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
3. After section 10, two separate decisions: whether to run the stage A probes (about $5),
   then, on the probe numbers, whether to run the stage B continuation (about $11-40). Both
   stages share one pod cap, which needs to be ~$46 for the PR 1 shape's worst case (was $40).
4. After section 11: whether to run the confirmatory seeds, at 75 steps (~$119) or 150 steps
   (~$230, over the $210 allowance in the worst case). Whether WO-04's
   2-of-3-task benchmark needs a two-task (PickPlace + Push) confirmatory run instead. It would
   cost about the same, at about 0.18 h/step (A100 ledger, pod `9d8p7vyfyv6ca4`: (28.4 - 0.85) / 150).
5. **Gate stops in stage B and the confirmatory seeds** (new with the merge of main, 2026-11-08).
   `configs/rl_pickplace_lr*.yaml` inherit `gate_stop: false`, so the step-20 flat gate and the
   decline / entropy gates only log. Options: (a) keep false, which follows the owner's "observe
   first" rule; the run goes to step 75 at the priced worst case, $39.7. (b) Set `gate_stop: true`
   in the lr configs, which restores the plan's step-20 exit ($11.3) under the trend rule with
   `gate_flat_z` 0.5 (provisional, simulated only). (c) Keep false and have the stage B job
   script stop the pod on `gate/first_would_stop_step`. The agent has not changed the setting.

## 15. Run sequence (after approval)

```
# pod (Linux, RTX 4090): setup as scripts/pod/wo02_job.sh, then
uv run scripts/wo02r_collect_corrections.py --preset sim --push                 # corrections (0.2 clip)
uv run scripts/wo02r_collect_corrections.py --preset sim --mode control --push  # control demos
uv run scripts/wo02_sft.py --preset sim --config configs/sft_a5_recovery.yaml
uv run scripts/wo02_sft.py --preset sim --config configs/sft_a5_control.yaml
for ck in <4 checkpoints per arm>; do                                   # selection, seed 7
  uv run scripts/wo02_eval.py --preset sim --config configs/sft_a5_<arm>.yaml --protocol 0.2 \
    --adapter $ck --cells seen --seeds 7 --episodes 100
done
uv run scripts/wo02_eval.py --preset sim --protocol 0.2 --adapter <step_004000>   # baseline
uv run scripts/wo02_eval.py --preset sim --protocol 0.2 --adapter <recovery pick>
uv run scripts/wo02_eval.py --preset sim --protocol 0.2 --adapter <control pick>
# Mac or pod, $0: the section 10 readouts from the three runs' episodes.jsonl
uv run python -m src.eval.paired runs/<baseline> runs/<recovery>
uv run python -m src.eval.paired runs/<baseline> runs/<recovery> --seeds 1,2
uv run python -m src.eval.paired runs/<control> runs/<recovery>
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
  validates, RL seed 3 passes `check_rl_seed`, flat gate (trend, logged only under
  `gate_stop: false`) at step 20, 4 periodic evals (steps 0, 25, 50, 75), clip 4.0 deg/step;
  20.6 h / $37 for 75 steps (re-run on the merge of main, 2026-11-08; it was 16.6 h / $29.7 at the
  WO-04 shape). `scripts/wo03_rl.py --preset tiny` (toy env, CPU): the GRPO
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
