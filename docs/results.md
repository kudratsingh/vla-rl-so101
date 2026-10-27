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
