# WO-04 — RL in sim, the headline result (owner reviews this PR)

> **Pivot 2026-11-09 (ADR-0003):** the OpenVLA-OFT / GRPO track is stopped. **Benchmark unmet.** Recorded partial result: GRPO +5.0 (two-task) and +3.7 (PickPlace-only) points on PickPlace seen, single training seeds (`docs/results.md`). The content below is
> kept as written for the record; the current work orders are WO-06 to WO-09.

Branch: `wo-04-rl-sim`. Tier: rl (A100 80 GB).

## Deliverables

1. `scripts/wo03_rl.py --preset rl` running end to end from the WO-02 SFT checkpoint on all three
   tasks; one run = one `runs/<id>/` with record, metrics, final LoRA weights (Hub).
2. Owner-launched runs: 3 seeds. Before each, the agent posts: config hash, estimated hours,
   GPU, dollars.
3. `scripts/wo02_eval.py` on each RL checkpoint → `docs/results.md` sim table filled, SFT-only vs
   SFT+RL, all three cells.
4. Training curves (success rate vs step, reward std per group, rollout throughput episodes/h)
   saved to `runs/<id>/plots/`.

## Benchmark

SFT+RL beats SFT-only by ≥20 points on seen positions on at least 2 of 3 tasks (3-seed mean).
Shifted-position delta reported either way. $/run and episodes/hour logged in STATUS.md.

## Notes

- If the first run shows no improvement in the first 25% of steps, stop it (owner decision) and
  check: reward all-zero or all-one groups (dynamic sampling should drop them), temperature,
  LoRA lr. Do not let a flat run finish.
- Re-price here: if rollouts dominate wall-clock, present options (fewer envs + more steps,
  smaller image, H100, RLinf pipelined mode) with numbers before run 2.
