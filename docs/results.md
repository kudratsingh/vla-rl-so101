# Results

Every number: run id, config hash, protocol hash, seeds, date, GPU, dollars. Single-run numbers
are labelled single-run. Sim and real are never mixed in one table.

## Sim — success rate (%), 100 episodes/cell, 3 seeds, 95% Wilson interval

SFT-only = OpenVLA-OFT LoRA, WO-02 attempt 4 `step_004000` (25 demos/task, 8000 steps; Hub
`KudratSingh22/vla-rl-so101-sft/wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000`).
**Below the original target band** (WO-02 asked for 20-60% on seen positions); it meets the
owner-amended rule 3 (>= 10% on 2 of 3 tasks: PickPlace 17.0%, Push 10.0% — exactly at the
threshold). Cells: pooled success % over 3 sim seeds x 100 episodes, [95% Wilson], (per-seed %).
One SFT training run (single-run); GPU sim (physx_cuda), RTX 4090, 2026-10-27, eval config
`6bfcad257401`, horizon 150, full 8-action chunks executed; eval $0.94 (run record).

| Task | Cell | SFT-only | SFT+RL | Δ | Run ids | Protocol |
|---|---|---|---|---|---|---|
| PickPlace | seen | 17.0 [13.2, 21.7] (18, 18, 15) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| PickPlace | shifted | 1.7 [0.7, 3.8] (2, 1, 2) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| PickPlace | unseen objects | 14.7 [11.1, 19.1] (10, 28, 6) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Push | seen | 10.0 [7.1, 13.9] (14, 9, 7) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Push | shifted | 3.7 [2.1, 6.4] (5, 3, 3) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Push | unseen objects | 6.3 [4.1, 9.7] (4, 3, 12) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Stack | seen | 2.7 [1.4, 5.2] (2, 3, 3) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Stack | shifted | 0.0 [0.0, 1.3] (0, 0, 0) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |
| Stack | unseen objects | 2.0 [0.9, 4.3] (1, 3, 2) | | | `20261027T190836Z-wo02eval-sim-cdd3` | 0.1-draft `e9ce5bb47a88` |

The SFT+RL column stays empty: it needs 3 training seeds, and only one two-task seed finished.
The two RL runs that completed are in the next table.

## Sim — WO-04 RL runs, each a single training seed (single-run)

Success %, pooled over 3 sim seeds x 100 episodes, [95% Wilson]. Protocol 0.1-draft
`e9ce5bb47a88`, horizon 150, GPU sim, A100-80GB, final checkpoint (step 150). Both start from the
SFT-only checkpoint above. The two-task run trained on PickPlace + Push, the PickPlace-only run on
PickPlace. **WO-04's benchmark (>= +20 points on 2 of 3 tasks, 3 seeds) is not met.**

| Task | Cell | SFT-only | Two-task RL | Δ | PickPlace-only RL | Δ |
|---|---|---|---|---|---|---|
| PickPlace | seen | 17.0 [13.2, 21.7] | 22.0 [17.7, 27.0] | +5.0 | 20.7 [16.5, 25.6] | +3.7 |
| PickPlace | shifted | 1.7 [0.7, 3.8] | 2.3 [1.1, 4.7] | +0.7 | 2.3 [1.1, 4.7] | +0.7 |
| PickPlace | unseen objects | 14.7 [11.1, 19.1] | 14.7 [11.1, 19.1] | 0.0 | 19.7 [15.6, 24.5] | +5.0 |
| Push | seen | 10.0 [7.1, 13.9] | 7.7 [5.2, 11.2] | -2.3 | 9.7 [6.8, 13.5] | -0.3 |
| Push | shifted | 3.7 [2.1, 6.4] | 2.0 [0.9, 4.3] | -1.7 | 4.0 [2.3, 6.9] | +0.3 |
| Push | unseen objects | 6.3 [4.1, 9.7] | 6.7 [4.4, 10.1] | +0.3 | 8.3 [5.7, 12.0] | +2.0 |
| Stack | seen | 2.7 [1.4, 5.2] | 2.3 [1.1, 4.7] | -0.3 | 2.0 [0.9, 4.3] | -0.7 |
| Stack | shifted | 0.0 [0.0, 1.3] | 0.3 [0.1, 1.9] | +0.3 | 0.0 [0.0, 1.3] | 0.0 |
| Stack | unseen objects | 2.0 [0.9, 4.3] | 2.0 [0.9, 4.3] | 0.0 | 2.0 [0.9, 4.3] | 0.0 |

| Run | RL run id | Protocol eval run id | Commit | Eval config | Pod | $ |
|---|---|---|---|---|---|---|
| Two-task, seed 0 | `20261027T205508Z-wo04-rl-s0-2992e5` | `20261028T234248Z-wo02eval-sim-8de3` | `7c896d0` | `56483b601882` | `9d8p7vyfyv6ca4` | 51.27 |
| PickPlace-only, seed 0 | `20261028T185032Z-wo04-rl-s0-84bf82` | `20261029T233518Z-wo02eval-sim-d102` | `4a76953` | `d090ed40bb1b` | `s2ljj3drwg0ov0` | 55.00 |

**The PickPlace-only run missed its bar** (owner, 2026-10-28: seen, PickPlace, 300 episodes,
>= 27.0%): 20.7%, 62/300. Neither RL gain on PickPlace seen is distinguishable from SFT-only by
an unpaired two-proportion test (two-task p = 0.12, PickPlace-only p = 0.25); the two runs do not
differ from each other (p = 0.69).

Mid-run checkpoints, same protocol (owner, 2026-10-31: is there a peak?). Two-task step 50 had the
highest in-training reading of any checkpoint (37.5% at 32 episodes). RTX 4090, $0.93 each, commit
`7dc6822`, pod `7gbl8q46qlxf1z`.

| Checkpoint | PickPlace seen | Vs same run's step 150 | Push seen | Stack seen | Eval run id |
|---|---|---|---|---|---|
| Two-task, step 50 | 21.3 [17.1, 26.3] | 22.0 | 8.7 [6.0, 12.4] | 2.7 [1.4, 5.2] | `20261031T041440Z-wo02eval-sim-4be2` |
| PickPlace-only, step 100 | 22.7 [18.3, 27.7] | 20.7 | 10.0 [7.1, 13.9] | 3.0 [1.6, 5.6] | `20261031T051736Z-wo02eval-sim-a72a` |

No peak: each mid-run checkpoint is within 2 points of its run's final checkpoint, and none reaches
27.0%. Against SFT-only (unpaired): two-task step 50 p = 0.18, PickPlace-only step 100 p = 0.08,
before allowing for having looked at four checkpoints.

## Ablations (WO-05)

| Factor | Levels | Result | Run ids |
|---|---|---|---|
| Demos per task | 1 / 5 / 10 | | |
| Samples per prompt | 4 / 8 / 16 | | |

## Real — success rate (%), 50 trials/cell, one setup

| Task | Cell | SFT-only | SFT+RL | Δ | Date | Protocol |
|---|---|---|---|---|---|---|

## Negative results and surprises

- WO-02 (2026-10-26/27): OpenVLA-OFT few-shot LoRA SFT never reached the 20-60% seen band.
  50-episode seen sweeps (1 seed, best checkpoint, PickPlace / Push / Stack): attempt 1 (10
  demos/task, lr 5e-5, 2000 steps) 8/4/2%; attempt 2 (lr 5e-4) 20/6/4%; attempt 3 (8000 steps)
  22/4/4%; attempt 4 (25 demos/task) 18/16/6%. Executing only the first 4 of each 8-action chunk
  (receding horizon) scored lower than full chunks. Details: ADR-0001.
- WO-02 the 50-episode, 1-seed sweep over-stated attempt 4 on Push (16%) vs the full protocol
  (10.0% pooled, seeds 14/9/7%). Shifted positions are the weakest cell (0-3.7%); unseen objects
  sit near seen for PickPlace (14.7%) but with a large seed spread (10/28/6%).
- WO-04 probes (2026-10-27): at rollout temperature ~1.6 with K=8, 92-93% of GRPO groups were
  all-equal (dropped), and reusing sampler log-probs as old log-probs gave |log ratio| max
  0.15-0.20 (mean 0.013). Temperature 1.0 cut drops to 72%; recompute cut the mean to 0.00012.
  ADR-0002 amendment 1.
- WO-04 (2026-10-29/30): GRPO gave +5.0 / +3.7 points on PickPlace seen after 150 steps, well short
  of the published +40-50 and of the owner's +10 bar. Two of three two-task seeds stopped flat at
  step 38 (mean rule: seed 1 13.3 -> 13.3%, seed 2 13.3 -> 11.7%).
- WO-04 giving RL all the updates did not help: PickPlace-only (20.7%) did not beat two-task (22.0%)
  on PickPlace, and Push did not fall when untrained (9.7% vs 10.0%). Interference between tasks is
  not supported; the PickPlace-only result rules out the stratified multi-task follow-up.
- WO-04 in-training evals (32 episodes, step 150: 31.2% PickPlace-only, 25.0% two-task) read higher
  than the protocol (20.7%, 22.0%). Two-task PickPlace read 37.5% at step 50 and 34.4% at step 75,
  then 21.9% at step 100. The protocol eval of step 50 reads 21.3%: that peak was eval noise (about
  +/-15 points at 32 episodes). In-training evals are not a safe way to pick a checkpoint here.
- WO-04 the updates are very small: approx_kl ~3e-5, clip fraction 0.000 in every 25-step window of
  both runs. PickPlace-only sampled success still rose (7.9% -> 12.6%, entropy 1.31 -> 1.08,
  groups dropped 61% -> 51%); two-task stayed flat (6.8% -> 7.7%, entropy ~1.5).
