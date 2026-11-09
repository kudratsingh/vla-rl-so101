# WO-07 — Real demos + SmolVLA/ACT baseline

Branch: `feat/smolvla-real` (code, PR 2 of the pivot handoff), then the owner's recording and
evaluation sessions. Tier: dev GPU (24 GB) for fine-tuning; Mac (MPS) for inference and the arm.
Rewritten 2026-11-09 for the SmolVLA pivot (ADR-0003; handoff
`docs/reviews/2026-11-09-smolvla-pivot-handoff.md`, owner decisions D1-D6 in
`docs/reviews/2026-11-09-pivot-decisions.md`, which win where they differ).

## Done when

50 PickPlace demos (10 per seen position, both cameras); SmolVLA and ACT fine-tuned on the same
demos; 50 binary trials per cell (seen, shifted, unseen) for each; immutable export.

## Seen cell, bowl and demos (D1, D2)

- **Five seen slots** from the existing 16-slot inner block of `src/eval/real_grid.py` (rows B-E,
  columns 2-5; the grid itself is unchanged). Spread across the block: one near each corner and
  one in the middle, so the 20-slot shifted ring is an extrapolation in every direction. The
  4 x 4 block has no single centre slot, so one of the four central slots (C3, C4, D3, D4) is
  chosen and the reason recorded with the freeze.
- **Demos:** 50 PickPlace = 10 at each of the five slots, leader-arm teleop, front + wrist
  cameras in every episode. The other 11 inner slots and the 20 ring slots are never used by
  demos or camera-match frames.
- **Bowl fixed** at one position for every demo and every eval trial. It must satisfy the scene's
  placement constraints for every object slot used (5 seen + 20 ring): MIN_SEP (11 cm for
  PickPlace), the base keep-out (bowl centre >= 15 cm from the base axis) and reach.
  Geometry note (agent, computed from `src/eval/real_grid.py` and `src/env/tasks.py` constants):
  no point inside the 6 x 6 grid is 11 cm from all 20 ring slot centres (the best is about
  6.5 cm), so the fixed bowl sits beside the printed grid, not on it. PR 2 picks the position,
  checks every constraint in code, checks that the front camera sees it, and freezes it.
- **Unseen objects:** the purple cube and the green cube, never in a demo or a camera-match frame.
  They replace the green cylinder for the real arm.

## Eval cells, 50 binary trials each (D3)

| Cell | Trials |
|---|---|
| Seen | 10 at each of the five seen slots |
| Shifted | 50 balanced across the 20 ring slots (2 or 3 per slot, fixed in the schedule) |
| Unseen objects | 50 at the five seen slots: 25 purple cube + 25 green cube, five per slot per object |

Unseen objects are reported per object and pooled; the pooled 50 is the headline. Scoring is
binary success under `real-judge-2` (CLAUDE.md rule 4). The paper's partial-credit score
(SmolVLA §4.1 subtasks) may be logged as a secondary column, never as the headline.

## Freeze before recording (D4)

A new real protocol version, with its own hash, freezes before the first demo:
- the five slot ids,
- the bowl position,
- the seeded trial schedule: cell membership and trial order for all 150 trials.

Protocols 0.1-draft and 0.2 stay byte-identical. Both policies run the same schedule. Grid
coordinates in every record are labelled planned poses (the operator places by hand).

## Development trials and the 100-demo decision (D5)

Start at 50 demos. Development trials run on their own small schedule, disjoint from the frozen
evaluation schedule, and are never reported as results. Going to 100 demos needs a failure
analysis of those trials showing that more data addresses the observed failures (the Cowling 2026
SmolVLA result fell at 100, as cited by the owner). The 50-per-cell evaluation is reserved for
selected checkpoints.

## Deliverables

1. **SmolVLA fine-tune** (`smolvla_base` from config `policy.base`, LeRobot 0.4.4): defaults from
   the LeRobot SmolVLA guide and the paper (§4.3): 20k steps, batch 64, bf16, action chunk 50,
   10 flow steps, images resized to 512. Camera keys `observation.images.front` and
   `observation.images.wrist`, passed through `--rename_map` to `camera1` / `camera2` as the guide
   does. `tiny` on CPU < 2 min: train a few steps, save, reload, compare outputs on one observation
   in eval mode with the policy state reset and the same explicit flow-matching noise tensor.
   Cost line provisional; reprice from a short probe before requesting spend.
2. **ACT fine-tune** on the same dataset, same script shape, same cost line.
3. **Recorder:** two image streams (front + wrist), state and action, in one LeRobotDataset
   (v3.0). The observation / action convention (camera mapping, joint units, action convention)
   frozen in one place and cited by the recorder, the trainer and the deploy loop.
4. **Mac inference** over MPS for both policies inside the existing deploy loop (safety clip,
   deadman, torque-off fault response, trial records unchanged); chunk latency and the gap between
   consecutive commands measured on the mock rig; checkpoint identity checked by content hash.
5. **Real eval** (`scripts/wo07_real_eval.py`): randomized trial order, planned object pose per
   trial from the grid, named labeler and rubric in the record, the frozen schedule above, and an
   eval LeRobotDataset with both streams. 50 trials per cell, three cells, two policies: budget
   2–3 h of bench time per policy.
6. **Export for `vishwaroop`** (`scripts/wo07_export_bundle.py`, one command, refuses on any
   missing field): per session, the dataset (`codebase_version: "v3.0"`), `declaration.json`
   with every field of vishwaroop's `docs/SO101_EXPORT.md` §2, `calibration.json`, `rubric.json`,
   `labels.json` for evaluation exports, and for trained-policy exports the weights and config
   with a content hash, a stable `family_id` (SmolVLA and ACT are two families) and a
   `training_run_id`. Session ids and run ids stay stable and distinct. Label provenance:
   `blinded_review` only when a reviewer scored the recorded clips with the policy identity
   hidden; otherwise omit the labels declaration. Before the 50 demos, a mock demo export and a
   mock evaluation export go through vishwaroop's own seal and import tools from a scratch copy of
   that repo (the original stays read-only) (D6).
7. `docs/results.md` real table filled for SmolVLA and ACT, every number with its run id and the
   real protocol hash.

## Notes

- Push and Stack wait until the PickPlace baseline is in.
- Latency (2026-11-08): the client logs `frame_return_to_first_write_ms`, which starts at
  LeRobot's frame stamp (taken after the driver returns the frame) and so excludes capture. A
  capture-to-write number needs the one-off optical offset from
  `docs/bringup/latency-measurement.md` (`deploy.capture_offset_ms`). Until that is measured,
  no latency is called end-to-end.
- Scoring (owner 2026-11-08, judging rule `real-judge-2`, `scripts/wo07_real_eval.py`): the
  primary real metric is successes / trials whose policy started (the bus write of the first
  policy goal was begun; a first goal blocked before its write, e.g. by the pre-write stop check
  after a Ctrl-C or a closed Terminal, never started and is retried). A servo, serial or safety
  fault after the start (also with torque off unconfirmed) is a failure, recorded separately
  (`fault`); the operator judges only clean trials; an unjudged score prompt (Ctrl-C, EOF, timeout) is a failure. Attempts that never
  started the policy are retried with the same placement, never scored. The success rate
  excluding faulted trials is reported only as a labelled secondary diagnostic.
- Prompts with torque on are bounded (`deploy.prompt_torque_timeout_s`, 60 s, then support,
  grace, torque off); "supported" is recorded only after an explicit ENTER
  (`src/robot/prompts.py`); `--hardware` refuses a stdin / stdout that is not a terminal before
  anything opens. Closing the Terminal (SIGHUP) takes the closed-stdin path (grace, torque off,
  `released_unconfirmed`).
- A started trial is durable: a `trial_started` row is written the moment the policy starts,
  and a resume scores a started trial without a record as a failure (fault kind
  `session_lost`), never a redo.
- The gRPC GPU policy server (`src/robot/serve.py`, merged in #23) stays available but is not
  the default path: inference runs on the Mac.
- No arm commands without the owner present. No paid run without the owner's per-run approval.
