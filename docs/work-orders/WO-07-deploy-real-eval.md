# WO-07 — Deploy and real evaluation

Branch: `wo-07-deploy`. Tier: dev GPU (policy server) + Mac (robot client).

## Deliverables

1. `src/robot/serve.py`: gRPC policy server on the GPU box (loads base + LoRA, returns action
   chunks); `src/robot/client.py` on the Mac: camera → request → safety → bus, 10–30 Hz.
2. `scripts/wo07_real_eval.py`: drives the real protocol — 50 trials per cell per task, object
   placements from the printed grid, operator confirms success/failure per trial, writes
   `runs/<id>/real_eval.json`.
3. `docs/results.md` real table: SFT-only vs SFT+RL (and the randomized checkpoint if WO-06
   produced one).
4. Ten consecutive unscripted trials recorded on video (owner records; agent provides the
   overlay script that stamps task, trial number and outcome).

## Benchmark

Real tables filled for all three tasks; latency from frame capture to bus write measured and
logged (target <150 ms); video produced.

## Notes

- Expect real numbers well below sim. Published zero-real-demo sim-to-real landed at 17.5% → 38.5%.
  Report whatever happens; WO-08 is for fixing it.
