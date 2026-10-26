# WO-08 — Close the sim-to-real gap

Branch: `wo-08-gap`. Tier: rl for retrains.

## Deliverables

1. Failure taxonomy from WO-07 videos and logs: perception (lighting/colour/crop), control
   (latency, servo lag, backlash), task (success-check mismatch), other. Counts per task.
2. One targeted fix per dominant failure class, in order of count:
   - perception → stronger randomization on that axis; retrain
   - control → measure real actuator step response, add matching latency/lag to the sim actuator
     model; retrain
   - task → align the real success criterion with the sim check
3. Retrained checkpoint(s) re-evaluated with the same real protocol (50 trials per cell).

## Benchmark

Real success improves on at least 2 of 3 tasks versus WO-07, same protocol. Before/after in
results.md with the failure counts that motivated each fix.

## Notes

- One fix per retrain so the delta is attributable. Owner approves each retrain with the cost line.
- Stop after two retrain cycles regardless; remaining gap goes in the writeup as a finding.
