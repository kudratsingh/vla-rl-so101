# WO-02 — Few-shot SFT of the base VLA + eval harness

Branch: `wo-02-sft`. Tier: dev (4090); rl (A100) only if VRAM forces it.

## Deliverables

1. `src/policy/`: wrapper for OpenVLA-OFT (LoRA, output head per ADR-0001) and π0.5 (openpi LoRA)
   behind one interface: `act(obs, instruction) -> action_chunk`.
2. `scripts/wo02_sft.py --config configs/sft.yaml`: few-shot SFT on ≤10 demos/task, all three
   tasks in one checkpoint. Log loss, steps, wall-clock, cost estimate.
3. `src/eval/protocol.py` (frozen after this WO; hash recorded) and `scripts/wo02_eval.py`: runs
   100 episodes × 3 cells × 3 tasks × N seeds and writes a Markdown table + JSON to
   `runs/<id>/eval.json`.
4. ADR-0001 updated to *accepted* with the measured SFT numbers.

## Benchmark

SFT checkpoint at 20–60% success on seen positions for at least 2 of 3 tasks (3 seeds). Eval
command produces the full table from one invocation. If <20% after 3 SFT attempts, switch to π0.5
per ADR-0001 and record why.

## Notes

- 20–60% is deliberate: that is the band where RL has signal (0% cannot be rescued; >90% leaves
  little headroom). Tune demo count, LoRA rank, steps and chunk size to land there — not higher.
- Owner launches all GPU runs. Provide a `tiny` dry run and a cost line in the PR.
