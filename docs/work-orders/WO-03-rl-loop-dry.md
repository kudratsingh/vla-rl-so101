# WO-03 — GRPO loop, verified dry (owner reviews this PR)

Branch: `wo-03-rl-loop`. Tier: CPU only.

## Deliverables

1. `src/rl/`: GRPO per ADR-0002, vendored/adapted from SimpleVLA-RL (verl) with source headers.
   Components: rollout worker (N parallel ManiSkill envs, K samples per prompt), reward from
   `src/env/success.py`, group advantage, clip-higher policy loss, LoRA update, checkpoint +
   run record.
2. Synthetic verification env `src/rl/toy_env.py`: a tiny discrete task with a known optimal
   policy. The loop must drive a small policy to the optimum from random init.
3. `scripts/wo03_rl.py --preset tiny`: runs the full loop on CPU in <2 min using the toy env and
   a 2-layer policy; `--preset sim` wires the real VLA + ManiSkill (not run in this WO).
4. VRAM budget table in `configs/rl.yaml` comments: base model bf16, LoRA, activations at the
   chosen rollout batch, N envs at 128×128 RGB, headroom.

## Benchmark

Toy env reaches ≥95% of optimal return within the tiny preset's step budget, deterministically
across 3 seeds. Advantage normalization, clipping and dynamic-sampling drop are each covered by a
smoke test with a hand-computed expected value.

## Notes

- This is one of two WOs the owner reviews. A silent bug here burns money in WO-04.
- No paid run in this WO under any circumstances.
