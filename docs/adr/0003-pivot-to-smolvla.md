# ADR-0003: Pivot to SmolVLA on the real SO-101

Status: accepted (owner, 2026-11-09)
Date: 2026-11-09
Work order: WO-06 onward (new table in `CLAUDE.md`)

## Amendment on 2026-11-10

The owner clarified the sequencing: **OpenVLA-OFT and GRPO are deferred until after full
evaluation of the real arm and the SmolVLA/ACT baseline.** The simulation environments, LoRA
training, GRPO implementation, checkpoint identities and measured results remain the foundation
for that follow-up. Its scope will be informed by the real-arm baseline and failure analysis.

The current implementation priority remains real-arm bring-up and matched SmolVLA/ACT evaluation.
The original 2026-11-09 rationale below is retained; this amendment supersedes its permanent-stop
wording. Recorded outcomes and the cancelled diagnostics approvals remain historical facts.

Sources: `docs/reviews/2026-11-09-smolvla-pivot-handoff.md` (the spec) and
`docs/reviews/2026-11-09-pivot-decisions.md` (owner decisions D1-D6, which refine it and win where
they differ).

## Decision

Stop the OpenVLA-OFT / SimpleVLA-RL reproduction. The project is now SmolVLA on the real SO-101,
with ACT as the baseline, trained the way LeRobot trains this arm: leader-arm teleop demos, two
cameras (wrist + front), fine-tune on a 24 GB card, inference on the Mac (MPS).

Few-shot stays a sim experiment. The real arm is data-first: 50 PickPlace demos to start
(10 at each of five seen positions), 100 only if a development-trial failure analysis shows that
more data addresses the observed failures. Push and Stack wait until the PickPlace baseline is
in. Filtered behaviour cloning in sim (WO-08) is the only post-training experiment left; it is
never called RL or GRPO.

## Why

- **The OFT track missed its benchmarks.** OFT PickPlace reached 17.0% on seen positions at
  25 demos/task (WO-02 attempt 4, protocol 0.1-draft `e9ce5bb47a88`, run
  `20261027T190836Z-wo02eval-sim-cdd3`), below the original 20-60% band. GRPO then gave +5.0
  (two-task, 22.0%) and +3.7 (PickPlace-only, 20.7%) points on PickPlace seen, single training
  seeds, against WO-04's >= +20 points on 2 of 3 tasks over 3 seeds. Those numbers and their
  protocol hashes stay in `docs/results.md`, marked "partial result, track stopped 2026-11-09".
- **OFT has no SO-101 evidence.** Its action head was retrained from scratch for six joint
  positions (ADR-0001), which discards the pretrained end-effector prior. The SimpleVLA-RL recipe
  behind it ran on 8 A800s (`docs/reviews/2026-11-07-project-review.md`, recipe table).
- **Memory is not the reason.** No VRAM floor is claimed for OFT: this repo ran OFT LoRA SFT on
  24 GB RTX 4090s (WO-02 attempts 1-3, pods `iqrmjox6ws0e66`, `efe7plxr06gaqc`,
  `l3485bxmdj0fce`, `docs/STATUS.md` spend ledger; bf16 + gradient checkpointing, micro-batch
  4 x 2, about 19-21 GB, 1.6-2.0 s/step, ADR-0001).
- **SmolVLA's pretraining is on the related predecessor of this arm.** The SmolVLA paper
  (Shukor et al., arXiv 2506.01844) pretrained on about 23k episodes of community data (§5.1:
  "The dataset used for training contains approximately 23k trajectories"; Table 1: 481 datasets,
  22.9K episodes). §5.1 states "Our pretraining currently uses datasets collected from a single
  robot type (SO100)", and §4.1 that "SmolVLA is not pretrained on any datasets recorded for the
  SO101". §4.2 describes how the SO-101 differs: "The SO101 has better arm design for faster
  assembly and different motors". So SO-100 is the related predecessor, not the same arm.
- **Published SO-101 fine-tunes are the evidence for the choice.** Cowling 2026 (arXiv
  2610.37170) reports SmolVLA on the SO-101 at 4/40 with 25 demos and 21/40 with 50, and a lower
  result at 100 (numbers as cited by the owner, not re-checked here). The LeRobot SmolVLA guide
  reports that 25 demos were "not enough" and that 50 worked (as cited in the handoff). The
  SmolVLA paper's own SO-101 recipe (§4.1) is "10 trajectories for each of 5 distinct starting
  positions, resulting in a total of 50 demonstrations per dataset".
- **It fits the existing stack.** SmolVLA fits a 24 GB card, and the repo already has a SmolVLA
  smoke run on Mac MPS with LeRobot 0.4.4 pinned (run `20261026T195329Z-wo01-sim-b326`, single
  run, `docs/STATUS.md` benchmarks).

## What is stopped, kept, cancelled, deferred

| | Item |
|---|---|
| Stopped | The OpenVLA-OFT / SimpleVLA-RL reproduction: no further OFT SFT, GRPO runs or WO-05 ablations. WO-04's benchmark is unmet; its result is recorded as partial. |
| Kept | Protocol 0.2 (`src/eval/protocol_v02.py`, frozen for historical comparison); the recorder fixes (`configs/demos_v2.yaml`, oracle v2); the GPU demo-rebuild tooling (`scripts/wo01r_rebuild_demos.py`, `scripts/pod/demo_rebuild_job.sh`); the pod backstop (job self-delete, in-container hard-cap timer, `scripts/pod/watchdog.py` + `watchdog_mac.sh`, `scripts/pod/launch_gate.py`, `scripts/pod/pod_start_cmd.sh` with the `/workspace` volume). These are policy-agnostic. `src/rl` stays in the tree, unused. The robot code merged in #23, #24 and #29 (safety, deadman, torque-off fault response, real-eval scoring rule `real-judge-2`, renamed `real-judge-3` with the start rule of #33 in PR #32, the placement grid). |
| Cancelled | The diagnostics pods (the $7.42 plan and its $8.42 revision, `docs/STATUS.md` "Pre-run hardening"). They were proposed, never approved and never launched; no money was spent on them. |
| Deferred | PR #25 (recovery SFT) and PR #27 (Track C doc). Neither is touched by this pivot. |

## Seen cell, bowl and eval cells (D1-D4)

- **Seen cell (D1).** Five slots chosen from the existing 16-slot inner block of
  `src/eval/real_grid.py` (the grid is unchanged): one near each corner of the block and one in
  the middle, so the 20-slot shifted ring is an extrapolation in every direction. The 4 x 4 block
  has no single centre slot, so one of the four central slots is chosen and the reason recorded.
  50 PickPlace demos = 10 at each of the five slots, both cameras. The other 11 inner slots and the
  ring are never used by demos or camera-match frames.
- **Bowl fixed (D2)** at one position for every demo and every eval cell, so shifted tests object
  position and unseen tests object identity. It must satisfy MIN_SEP, the base keep-out and reach
  for every object slot used (5 seen + 20 ring), and is frozen with the slots.
- **Cells, 50 binary trials each (D3).** Seen: 10 at each of the five slots. Shifted: 50 balanced
  over the 20 ring slots (2 or 3 per slot, fixed in the schedule). Unseen objects: 50 at the five
  seen slots, 25 purple cube + 25 green cube, five per slot per object; reported per object and
  pooled, the pooled 50 is the headline. The green and purple cubes replace the green cylinder for
  the real arm, so the real evaluation is a NEW protocol version with its own hash; sim protocols
  0.1-draft and 0.2 stay byte-identical.
- **Freeze before recording (D4).** The five slot ids, the bowl position and the seeded trial
  schedule (cell membership and order) are frozen and hashed into that version before the first
  demo. Both policies run the same schedule. Grid coordinates in every record are planned poses.
  Scoring is binary success (CLAUDE.md rule 4); the paper's partial-credit score may be a
  secondary column, never the headline.
- **Development trials (D5)** run on a small schedule disjoint from the frozen one, are never
  reported as results, and decide whether 100 demos are needed. The 50-per-cell evaluation is
  reserved for selected checkpoints.

## Interface notes with `vishwaroop`

`vishwaroop` (`~/Machine Learning Projects/vishwaroop`) is a separate evaluator project, read-only
to this repo. Its `docs/SO101_EXPORT.md` (WO-023) defines what it imports; the owner seals exports
with that project's tool.

- Its SO-101 throughput scene uses the same `so101_new_calib.urdf` this repo vendors, with a fixed
  front camera plus a wrist camera at 480 x 640. The two-camera decision matches it.
- Its M5-early draft plans best-of-N on ACT first; SmolVLA plus ACT gives it two policy families,
  which its held-out protocol groups by training run and family. Keep `family_id` and
  `training_run_id` stable and distinct across exports.
- Its ET-M5-1 still names "the GRPO-tuned policy". That is the owner's to amend in that repo;
  recorded here so the two programs do not drift.
- Label provenance (D6): `blinded_review` is declared only when a reviewer scored recorded clips
  with the policy identity hidden; otherwise the labels declaration is omitted, never
  misdeclared. A mock demo export and a mock evaluation export go through vishwaroop's own seal
  and import tools, from a scratch copy of that repo, before the 50 demos.

## Alternatives considered

- Keep OFT and run the higher-LR GRPO diagnostic ($60 allowance) and confirmatory seeds ($210):
  four GRPO checkpoints showed +3.7 to +5.7 points with no peak; more spend on the same base had
  no evidence of reaching the WO-04 bar, and OFT still has no SO-101 evidence.
- π0.5 via openpi LoRA (ADR-0001 fallback): needs more than 22.5 GB for LoRA (ADR-0001) and is
  heavier to serve on the Mac; the SO-101 fine-tuning evidence cited above is for SmolVLA.
- ACT alone: kept as the baseline. It is the LeRobot reference policy for this arm, but has no
  language conditioning and no pretraining prior; SmolVLA is the subject.

## What would reopen this

- SmolVLA on 50 (or 100) real PickPlace demos stays near 0% on seen positions while ACT on the same
  demos does clearly better, or both stay near 0% after the bring-up checks pass.
- SmolVLA cannot meet the deploy loop's latency on the Mac (PR 2 measures chunk latency and the
  gap between consecutive commands on the mock rig).
- A pinned-dependency change (LeRobot beyond 0.4.4) becomes necessary for SmolVLA or ACT; that is
  its own work order.
