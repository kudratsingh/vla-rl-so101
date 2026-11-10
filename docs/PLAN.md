# Plan

Rewritten 2026-11-09 for the pivot to SmolVLA on the real SO-101 (ADR-0003). The OpenVLA-OFT /
GRPO plan that came before it is in git history and in ADR-0001 / ADR-0002. Owner clarification
on 2026-11-10: further OpenVLA-OFT / GRPO work is deferred until after full evaluation of the
real arm and the SmolVLA/ACT baseline. Its implementation and measured results are retained.

## What we are building

SmolVLA, with ACT as the baseline, fine-tuned the way LeRobot trains this arm: leader-arm teleop
demos with two cameras (wrist + front), fine-tuned on a rented 24 GB GPU, inference on the Mac
(MPS) inside the existing safety / deadman deploy loop. Both policies are evaluated on the real
arm under a frozen, hashed real protocol. A separate sim experiment tests whether reward-filtered
behaviour cloning improves a SmolVLA checkpoint, and only a measured sim gain is tried on the arm.

Few-shot stays a sim experiment. The real arm is data-first.

## Why the pivot

OFT PickPlace reached 17% on seen positions at 25 demos and GRPO missed the WO-04 benchmark
(`docs/results.md`, partial result). OFT has no SO-101 evidence; SmolVLA was pretrained on SO-100
community data (the related predecessor of this arm) and has published SO-101 fine-tunes. Details
and citations: ADR-0003.

## Phases

| Phase | WOs | Output |
|---|---|---|
| Retained OFT sim foundation | 01–05 | Sim env, oracle, demos, OFT SFT, GRPO loop and runs; recorded partial result, follow-up deferred |
| A. Bring-up | 06 | Calibrated arm, drift check, leader teleop, both cameras matched with locked exposure / white balance, Mac MPS latency on the mock rig |
| B. Real baseline | 07 | 50 PickPlace demos, SmolVLA and ACT fine-tunes, 50 binary trials per cell for each, immutable export for `vishwaroop` |
| C. Sim post-training | 08 | Filtered behaviour cloning on SmolVLA in a two-camera sim; transfer test on the arm only after a sim gain |
| D. Release | 09 | README table, model cards, reproduce commands, ADR index |

## Tasks

1. `PickPlace` — pick the named object, place it in the bowl. The real arm starts here.
2. `Push` — push the cube to the marked target.
3. `Stack` — stack cube A on cube B.

Push and Stack wait until the PickPlace baseline is in. Each task has a programmatic success check
in `src/env/success.py` (sim); the real arm is judged by the real rubric (`real-judge-2`).

## Real data and evaluation (ADR-0003, decisions D1-D5)

- Demos: 50 PickPlace = 10 at each of five seen slots chosen from the 16-slot inner block of the
  placement grid (one near each corner, one central), bowl at one fixed position, both cameras.
  100 only if a failure analysis of development trials shows more data addresses the failures.
- Cells, 50 binary trials each: seen (10 per seen slot), shifted (balanced over the 20 ring slots),
  unseen objects (25 purple cube + 25 green cube at the seen slots; per object and pooled, pooled
  is the headline). Same seeded schedule for both policies.
- The slot ids, bowl position and trial schedule are frozen and hashed in a new real protocol
  version before the first demo. Development trials use a disjoint schedule and are never reported.

## Sim evaluation

Protocols 0.1-draft (`src/eval/protocol.py`) and 0.2 (`src/eval/protocol_v02.py`) stay frozen for
the recorded OFT numbers. WO-08 adds a wrist camera to the sim scene and a new two-camera protocol
version that carries over 0.2's success rules, cells and 4°/step action limit. Sim: 100 episodes
per cell, 3 seeds, 95% Wilson interval; every number cites its protocol hash.

## Non-goals

- No learned reward models, no shaped rewards, no human-in-the-loop RL in this project.
- WO-08 is filtered behaviour cloning. OpenVLA / GRPO follow-up is scoped separately after the
  real-arm baseline evaluation.
- No serving beyond the deploy loop; no web UI.
- No new simulator; ManiSkill3 only. No Isaac Sim.
- No dexterous, deformable or bimanual tasks.

## Budget

Cloud compute keeps the **$1,100 lifetime planning cap**; recorded spend is about $156.10
(`docs/STATUS.md`). The OFT-era allocations for GRPO diagnostics, confirmatory seeds and WO-05
ablations are void. The next paid run is the first SmolVLA fine-tune, after a short probe; PR 2
carries its cost line. Each paid run still requires its own estimate and explicit owner approval
under [CLAUDE.md](../CLAUDE.md#hard-rules).

Hardware is budgeted separately (SO-101 follower + leader, wrist camera module, Logitech C920x,
spare gripper servo).

## Follow-ons (not in scope now)

- OpenVLA-OFT and GRPO after full real-arm and SmolVLA/ACT evaluation, using the retained
  simulation, LoRA, rollout and evaluation infrastructure and the baseline's failure analysis.
- Residual RL on a frozen VLA using object poses (Microsoft CoRL 2026 recipe, no code released).
- The same brain-over-skills loop on the Microduck's command block.
