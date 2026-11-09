# Prompt for the implementation agent (2026-11-08)

Send with `docs/reviews/2026-11-07-project-review.md` (corrected 2026-11-08; see its Corrections log).

```text
You are the implementation agent for vla-rl-so101 (/Users/kudratsingh/Robotics/vla-rl-so101).
Read CLAUDE.md, then the attached project review (docs/reviews/2026-11-07-project-review.md,
including its Corrections log at the end), then docs/STATUS.md and docs/results.md. The review
is the spec. Do not redo the review.

GOAL
Implement every item in review section 9 ("Fix list before the next paid run") that does not
require hardware or a paid run, so the next GPU run cannot be wasted on the problems the review
found. Spend $0. Do not create pods, do not touch RunPod, do not launch any training, do not
send any command to the physical arm. Get explicit approval from the owner for each paid run
and for any command to the arm.

PR 1 — branch fix/pre-run-hardening, from main
Sim/RL/data/eval items from section 9, in this order:
1. Gates: replace the 64-episode kill switch and the 32-episode peak stop in src/rl/trainer.py
   with gates on windowed rollout/success_rate and rollout entropy (window and thresholds in
   rl.yaml). These gates monitor training and decide whether to keep spending; only the fixed
   held-out protocol eval decides whether the policy improved. Add a fixed-batch pre/post-update
   diagnostic: on a held batch, log the per-token log-prob movement, k3 KL and clip fraction
   measured before and after the whole update (all optimizer steps), plus grad_norm. The current
   aggregates average in a mini-batch taken before any optimizer step and are uninformative
   (review 3.1).
2. Resume: load optimizer.pt, scheduler, step counter and RNG state from a checkpoint;
   --resume <ckpt> on the training script; round-trip smoke test on the tiny preset.
3. Demo recorder (scripts/wo01_build_env.py): keep recording through release and retreat
   (N steps after success, configurable); no pad frames in the SFT loss. Review 3.2 (11 of 25
   PickPlace demos end below the intended gripper opening).
4. Oracle (src/data/oracle.py): slower approach and descend near the grasp; keep >= 95% success
   on all 3 tasks on the CPU sim at the tiny preset; report decisions per episode before/after.
5. Protocol 0.2 (src/eval/protocol.py): new version; do NOT change 0.1-draft semantics or any
   reported number. Composite hash over protocol + env + success-check + camera files;
   per-episode records written to the run dir; 4 deg/step action clip in sim eval matching
   configs/real.yaml. Prepare locally only. In the PR body, PROPOSE (do not run) a separately
   approved rerun of the existing SFT and RL checkpoints under identical 0.2 settings; reprice
   it from the selected checkpoints and measured eval runtime (~$3 on a 4090 is provisional).
6. Rollout timing split: model forward vs sim step vs render vs host transfer, in the probe
   output and metrics.jsonl.
7. configs/rl.yaml: 16 prompts x 8 samples, lr 1e-4, ppo_epochs 2, periodic evals only on the
   trained tasks. Keep configs/rl_pickplace.yaml consistent. The lr change rests on the
   published LoRA recipes (RIPT-VLA, RLinf-VLA), not on the logged KL; say so in the PR.
8. Camera jitter seeded from the episode seed (check PR #25 first and reuse if present);
   reject RL seeds >= 10.
9. Makefile setup target includes --extra robot and --extra smolvla, matching ENVIRONMENT.md.
10. Demo-rebuild script that regenerates the 25 demos/task on the GPU renderer in one command
    (review 3.2, Mac vs GPU renderer). Dry-runnable on CPU tiny.
11. Review and PR #25 plan revisions (docs only, same PR): use the matched K=16 sampled-success
    and group-drop figures; treat the scene-mixture fit as illustrative; keep the learning
    ceiling and the "$28 saved" claim out of STATUS/ADR text. In PR #25's
    docs/plans/recovery-sft.md (coordinate: PR #25 is a draft branch wo-recovery-sft), replace
    the "escalate to lr 1e-3 when clip fraction rounds to 0" rule with the fixed-batch
    diagnostic from item 1; keep the matched ordinary-demo control; log per-episode outcomes
    so the comparison can be paired. Correct docs/results.md and docs/adr/0002-rl-recipe.md
    where they cite "updates too small (approx_kl ~3e-5, clip 0)" as evidence: the metric is
    uninformative, not the conclusion proven. Never alter reported numbers.

OUT OF SCOPE for PR 1: src/robot/*, the branches of PRs #23 and #24, anything that changes
0.1-draft protocol semantics or already-reported numbers.

PR 2 — branch fix/robot-safety, based on PR #23's branch (wo-07-pilot), only after PR 1 is open
src/robot belongs to #23. Distinguish healthy shutdown from faults:
- Healthy shutdown (normal exit, Ctrl-C with the arm idle): park() to home_qpos through
  src/robot/safety.py, then torque off.
- Thermal or overcurrent fault (periodic Present_Temperature / Present_Current read above
  thresholds from configs/real.yaml): a separately validated stop response. HOLD keeps the
  servos energized and a powered park adds motion during the fault, so the response is
  torque-off with the operator prompt to support the arm. Test it on its own with the fakes.
- Torque-enable refusal when any joint rests outside its calibrated limits.
- Latency: the client starts its clock after the camera driver returns the frame
  (src/robot/client.py:45). Verify whether the camera's timestamp represents capture time and
  can be aligned with the host clock; if yes, use it; if not, add a documented
  optical-event-to-command measurement procedure and label the logged metric as excluding
  capture. Do not call the metric end-to-end until that is verified.
- scripts/wo07_real_eval.py (50 trials/cell, protocol 0.2 hash and per-trial records).
- Generate the A3 placement-grid PDF for the seen/shifted spawn regions from the sim constants.
Everything testable with tests/robot_fakes.py.

TRACK C PREP (docs + a probe plan, $0 now)
Add pi0.5 action-expert fine-tuning as an option for the separate real-data track (review
section 8, Track C). Basis: Cowling 2026 (arxiv 2610.37170) reports 28/40 on its own
pick-and-place task at 25 real SO-101 demos with the action expert fine-tuned; the LoRA variant
scored 4/110; 10.6 A100-hours per training run, about $19 at the current secure A100 list rate
before setup and evaluation. Verify compatibility with the pinned LeRobot 0.4.4 pi0.5 code and
the Mac-side client in PR #23, write the probe config and cost line, and stop there. Label the
SmolVLA paper's 90/50 scores as partial-credit results wherever they appear.

RULES (from CLAUDE.md, non-negotiable)
- Pins frozen. No dependency changes.
- Reward stays the env success check. No shaped rewards.
- Minimal tests: one CPU smoke test per changed module, tiny preset, each < 2 min. `make smoke`
  must pass after `uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla`.
- Conventional Commits. PR body: what / benchmark result / cost ($0) / what's next.
- docs/STATUS.md updated: what changed, what the next paid run needs, what is unverified.
- ruff, black, mypy --strict on src/ clean.
- If an item needs a GPU or money to verify, implement it, mark it "unverified on GPU" in the
  PR body and STATUS.md, and move on. Do not ask for approval to spend; propose with a cost line
  (estimated: <GPU> x <h> = $<x>) and wait.
- Ask the owner only for decisions that cannot be undone. Otherwise decide and note it in the PR.

DONE WHEN
PR 1 is open against main with CI green, every section-9 item either checked off or explicitly
deferred with a reason, and a one-screen summary at the top of the PR body describing the next
paid run (config, probe, projected A100 hours and $) WITHOUT launching it. Then PR 2 the same
way, then the Track C probe plan as a short doc.
```
