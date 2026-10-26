# Results

Every number: run id, config hash, protocol hash, seeds, date, GPU, dollars. Single-run numbers
are labelled single-run. Sim and real are never mixed in one table.

## Sim — success rate (%), 100 episodes/cell, 3 seeds, 95% Wilson interval

| Task | Cell | SFT-only | SFT+RL | Δ | Run ids | Protocol |
|---|---|---|---|---|---|---|
| PickPlace | seen | | | | | |
| PickPlace | shifted | | | | | |
| PickPlace | unseen objects | | | | | |
| Push | seen | | | | | |
| Push | shifted | | | | | |
| Push | unseen objects | | | | | |
| Stack | seen | | | | | |
| Stack | shifted | | | | | |
| Stack | unseen objects | | | | | |

## Ablations (WO-05)

| Factor | Levels | Result | Run ids |
|---|---|---|---|
| Demos per task | 1 / 5 / 10 | | |
| Samples per prompt | 4 / 8 / 16 | | |

## Real — success rate (%), 50 trials/cell, one setup

| Task | Cell | SFT-only | SFT+RL | Δ | Date | Protocol |
|---|---|---|---|---|---|---|

## Negative results and surprises

- (record here, not in Slack)
