# WO-05 — Ablations and writeup v1 (sim milestone)

> **Pivot 2026-11-09 (ADR-0003):** the OpenVLA-OFT / GRPO track is stopped. Not started; cancelled with the OFT track. The content below is
> kept as written for the record; the current work orders are WO-06 to WO-09.

Branch: `wo-05-ablations`. Tier: rl.

## Deliverables

1. Demo-count ablation: SFT on 1 / 5 / 10 demos per task, then RL, one seed each → does RL close
   the gap from fewer demos?
2. Samples-per-prompt ablation: K = 4 / 8 / 16, one seed, same step budget → quality vs cost.
3. Unseen-objects cell evaluated for SFT vs SFT+RL (already in the protocol; report it).
4. Public README v1: one-paragraph claim, the results table, reproduce commands, cost table.
5. `docs/blog-draft.md`: "Single-GPU RL post-training of a VLA" — what was reproduced, what
   differed from the 8-GPU papers, what it cost.

## Benchmark

Both ablation tables filled with run ids; README reproduces every headline number with one
command each; total project spend to date in STATUS.md.

## Notes

- This is the shippable sim-only milestone. If hardware is late, publish here.
- Owner launches runs; batch the approvals (one message listing all planned runs with costs).
