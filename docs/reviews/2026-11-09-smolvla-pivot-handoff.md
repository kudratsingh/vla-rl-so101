# Handoff: pivot to SmolVLA on the real SO-101

Owner decision, 2026-11-09. Supersedes `docs/reviews/2026-11-08-implementation-prompt.md` where they
conflict. Main is at c221114 with #28, #24, #23 and #29 merged.

## Decision

Stop the OpenVLA-OFT / SimpleVLA-RL reproduction. The project is now SmolVLA on the real SO-101,
with ACT as the baseline, trained the way LeRobot trains this arm: leader-arm teleop demos, two
cameras, fine-tune on a 24 GB card, inference on the Mac.

Reasons, for ADR-0003:
- OFT PickPlace reached 17% on seen positions at 25 demos; the GRPO runs missed the WO-04 benchmark.
  Those numbers and their protocol hashes stay in `docs/results.md` as a recorded partial result.
- OFT has no SO-101 evidence. Its action head was retrained from scratch for six joint positions,
  discarding the pretrained end-effector prior. The SimpleVLA-RL recipe behind it ran on 8 A800s.
- SmolVLA was pretrained on about 23k episodes of SO-100 community data, the related predecessor
  of this arm (the paper notes the SO-101 differs in arm design and motors). Published SO-101
  fine-tunes exist (Cowling 2026, arXiv 2610.37170: 4/40 at 25 demos, 21/40 at 50). It fits a
  24 GB card and the repo already has a SmolVLA smoke run on Mac MPS with LeRobot 0.4.4 pinned.
- Few-shot stays a sim experiment. The real arm is data-first. Say this plainly in the ADR and in
  CLAUDE.md rule 3.

Cancelled: the diagnostics pods (the $7.42 plan and its $8.42 revision). No approval exists for them.
Deferred: #25 (recovery SFT) and #27 (Track C doc). Keep protocol 0.2, the recorder fixes, the GPU
demo rebuild tooling and the pod backstop; they are policy-agnostic. Keep `src/rl` in the tree,
unused.

## PR 1: `docs/pivot-smolvla` from main

1. `docs/adr/0003-pivot-to-smolvla.md` with the reasons above, what is kept, what is cancelled.
2. Rewrite the CLAUDE.md goal sentence and the WO table (below). Amend rule 3 (few-shot = sim only).
   Trim `docs/PLAN.md` to match. Update `docs/STATUS.md`: current WO, cancelled spend, the pivot.
3. Mark the OFT sim rows in `docs/results.md` "partial result, track stopped 2026-11-09"; change no
   numbers.

### New work-order table

| WO | Name | Done when |
|---|---|---|
| 06 | Bring-up | Checklist steps 0–10 complete under supervision: calibration verified, per-session drift check, teleop with the leader arm, both cameras matched and exposure/white balance locked, MPS inference latency measured on the mock rig |
| 07 | Real demos + SmolVLA/ACT baseline | 50 PickPlace demos (10 per seen position, both cameras); SmolVLA and ACT fine-tuned on the same demos; 50 binary trials per cell (seen, shifted, unseen) for each; immutable export |
| 08 | Filtered BC in sim | SmolVLA fine-tuned on the 25 sim demos per task; reward-filtered BC against the sim success check; both arms from the same checkpoint, matched budget, held-out seeds; sim gain measured; transfer tested on the arm only after the sim gain is in |
| 09 | Writeup | README table, model cards, reproduce commands, ADR index, results.md with every number and protocol hash |

Push and Stack wait until the PickPlace baseline is in. The LeRobot guide reports 25 demos were
"not enough" and 50 worked; start at 50. Going to 100 is conditional on a failure analysis from
development trials showing that more data addresses the observed failures (the Cowling 2026 SmolVLA
result fell at 100). Reserve the 50-per-cell evaluation for selected checkpoints.

ADR rationale, corrected: do not claim a 25 GB LoRA floor for OFT (this repo ran OFT LoRA on 24 GB
4090s). Describe SO-100 as the related predecessor; the SmolVLA paper (§4.2, §5.1) says its
pretraining used SO-100 data only and that the SO-101 differs in arm design and motors. Cite the
published SO-101 fine-tuning results as the evidence for the choice.

## PR 2: `feat/smolvla-real` from main (after PR 1)

Training and inference code. No paid runs, no arm commands.

1. **SmolVLA fine-tune** config and script on the pinned LeRobot 0.4.4, reusing the smoke path in
   `scripts/wo01_smolvla_smoke.py`. Takes `--config` and `--preset`. `tiny` runs on CPU in under
   2 min: train a few steps, save, reload, compare outputs on the same observation in eval mode
   with the policy state reset and the same explicit flow-matching noise tensor passed to both
   calls, within a numerical tolerance (SmolVLA inference samples noise, so two valid calls differ
   otherwise). Inputs: a LeRobotDataset of real
   demos with two cameras. Defaults follow the LeRobot SmolVLA guide and the paper (arXiv
   2506.01844 §4.3): `smolvla_base`, 20k steps, batch 64, bf16, action chunk 50, 10 flow steps,
   images resized to 512. The guide quotes about 4 h on one A100 for 20k steps, about $7 at the
   secure rate of $1.79/h; on a 24 GB 4090 at $0.89/h the batch may need to drop and the run will
   take longer. Treat the cost line as provisional and reprice from a short probe before
   requesting spend. Camera keys: record as `observation.images.front` and
   `observation.images.wrist` and pass the `--rename_map` to `camera1`/`camera2` exactly as the
   guide does.
2. **ACT fine-tune** on the same dataset, same script shape, same cost line.
3. **Recorder check**: verify camera mapping, joint units and action convention through the same
   preprocessing used at inference. Freeze the observation and action convention in one place
   before the first real demo and cite it from the recorder, the trainer and the deploy loop.
4. **Mac inference path** over MPS for SmolVLA and ACT, inside the existing deploy loop:
   safety clip, deadman, torque-off fault response, trial records unchanged. Measure chunk latency
   and the gap between consecutive commands on the mock rig; record both. Verify checkpoint identity
   (content hash) the way the serve path does.
5. **Real eval**: confirm which of these the merged real-eval script already does and add the rest:
   randomized trial order, initial object pose logged per trial from the placement grid, named
   labeler and success rubric in the record, fault after policy start scored as failure with the
   reason recorded, resume skips faulted attempts and retries only slots where the policy never
   started. 50 trials per cell, three cells, two policies: budget 2–3 h of bench time per policy.
6. **Eval cells frozen before training**: seen = five demo positions chosen from the grid's 16
   inner slots (the paper's recipe: 50 demos = 10 per position, bowl fixed), 10 demos each;
   shifted = the 20-slot ring outside the seen block; unseen = purple cube and green cube, never in
   a demo or a camera-match frame. This replaces the protocol's green cylinder, so it is a new real
   protocol version with its own hash. Use the same seeded placement schedule for both policies
   and label grid coordinates as planned poses. Scoring stays binary success per CLAUDE.md rule 4;
   the paper's partial-credit score may be logged as a secondary column, never as the headline.
7. **Export for the evaluator project** (`vishwaroop`, a separate repo at
   `~/Machine Learning Projects/vishwaroop`; read-only to you, and its agents never touch this
   repo). Its `docs/SO101_EXPORT.md` (WO-023) defines what it imports. The owner seals the export
   with that project's tool; this repo only has to produce the inputs. Per recording session, write
   next to the LeRobot dataset:
   - the dataset itself in LeRobot `codebase_version: "v3.0"` (what LeRobot 0.4.4 writes);
   - `declaration.json` with every field of its §2: `format: "lerobot-v3.0"`, `dataset_root:
     "dataset"`, `units` per joint name in `observation.state` (one of `deg`, `rad`,
     `normalized_m100_100`, `normalized_0_100`; declare what was actually recorded), `calibration`
     and `rubric` file names, `task_id`, `task_version`, `protocol_version`, `recorded_at`,
     `robot_id`, `purpose` (`demonstration` or `evaluation`), `sessions` mapping every episode
     index to a session id, `policy` (`{"origin": "teleop"}` for demos; for policy rollouts
     `{"origin": "trained", "checkpoint_id", "training_run_id", "family_id", "weights", "config"}`
     with the weights and config as files), optional `labels` (`{"file", "method":
     "blinded_review" | "physical_sensor_and_review"}`) and optional `object_pose` (feature name,
     frame id, units);
   - `calibration.json` (the follower's LeRobot calibration file), `rubric.json` (the task's
     success rubric, versioned), `labels.json` mapping each episode index to `{"success",
     "reviewer_ids", "terminated", "truncated", "reason"}` for evaluation exports;
   - for trained-policy exports, the checkpoint weights and config files with a content hash,
     a stable `family_id` per training run family (SmolVLA and ACT are two families) and a
     `training_run_id`.
   Label provenance: declare `blinded_review` only when a reviewer scored the recorded clips with
   the policy identity hidden (randomized trial order makes this possible). Live operator scores
   and automatic fault outcomes keep their actual provenance in the trial records and are not
   relabelled as blinded. The importer allows the `labels` declaration to be omitted until an
   accepted method exists; omit it rather than misdeclare.
   Nothing may be guessed: the importer refuses a missing or unknown value. A session, scene group
   or training run is never split across their train/dev/regression partitions, so keep session
   ids and run ids stable and distinct. Provide a one-command `scripts/wo07_export_bundle.py` that
   assembles these files from a finished recording and refuses on any missing field.

### Interface notes with `vishwaroop` (for the agent's awareness, no action in that repo)

- Its SO-101 throughput scene uses the same `so101_new_calib.urdf` this repo vendors, with a fixed
  front camera plus a wrist camera at 480 × 640. The two-camera decision above matches it.
- Its M5-early draft plans best-of-N on ACT first; the SmolVLA plus ACT pair gives it two policy
  families, which its held-out protocol groups by training run and family. Keep `family_id` and
  `training_run_id` stable across exports.
- Its ET-M5-1 still names "the GRPO-tuned policy". That is the owner's to amend there; mention it in
  ADR-0003 so the two programs do not drift.

## Hardware floor (add to the WO-06 checklist)

- Two cameras from day one (owner decision 2026-11-09): the kit's 32 mm module on the wrist, the
  Logitech C920x as the front camera at the sim pose, both 640 × 480 at 30 fps as in the guide's
  rollout command. Record both in every demo and every eval trial. The paper used a top plus
  wrist pair on the SO-100; our elevated 45° front view is the stand-in for "top", and this
  difference is recorded, not hidden.
- Recording and eval both produce a LeRobotDataset with both image streams, state and action
  (the merged recorder is front-only and the real-eval path writes trial records only; both are
  deliverables, and a mock export is tested against the vishwaroop importer before the 50 demos).
- Servo position P gain recorded (default 32); calibration drift check at the start of every
  session; spare gripper servo on hand.
- Exposure and white balance locked on both cameras before camera match; never auto.
- Shutdown and fault behaviour as merged in #29. Powered steps 9–10 only under supervision.

## WO-08 plan (write it, do not run it)

The real policy uses front plus wrist, so the sim must too before any transfer test: add a wrist
camera to the ManiSkill scene on the gripper link, matched to the real wrist module's field of view
and pose, and rebuild the sim demos with both streams. Cost this before WO-08 starts. The
two-camera sim evaluation is a new protocol version with its own hash: protocol 0.2 hashes the
camera code and fixes a single front view, so it stays frozen for historical comparison, and the
new version carries over its success rules and the 4°/step action limit unchanged.

Reward-filtered behaviour cloning on SmolVLA in sim: sample K rollouts per prompt from the SFT
checkpoint, keep successes by the project's own success check (`src/env/success.py`, not any
simulator reward), fine-tune on them, repeat. Both comparison arms start from the same checkpoint
with matched gradient steps; evaluation on held-out seeds under protocol 0.2. Call it filtered
behaviour cloning everywhere, never RL or GRPO. Report which of `src/rl/rollout.py`, the success
checks and `src/eval` can be reused and which parts depend on OFT tokens or PPO.

## Rules (unchanged from CLAUDE.md)

No paid runs without per-run approval; probe first, approval gate before the full run; every pod
uses `scripts/pod/pod_start_cmd.sh` with a persistent volume at `/workspace`. No arm commands
without the owner present. Reward is the environment's success check only. Pins frozen. Secrets and
weights out of git. Honest reporting: sim numbers are never real numbers; single runs labelled.

## Done when

PR 1 merged with ADR-0003 and the docs aligned. PR 2 open with CI green, the tiny SmolVLA and ACT
train/save/reload checks passing on CPU, the convention frozen and cited, the Mac inference path
tested against the mock rig with latency numbers posted, the eval cells hashed, and a cost line for
the first paid SmolVLA run awaiting approval.

See also: `docs/reviews/2026-11-09-pivot-decisions.md` (owner decisions D1-D6, same day) refines this handoff and wins where they differ.
