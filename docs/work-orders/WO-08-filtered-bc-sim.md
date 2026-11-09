# WO-08 — Filtered behaviour cloning in sim

Branch: `wo-08-filtered-bc`. Tier: dev GPU (24 GB) for SmolVLA fine-tunes and GPU-sim rollouts;
CPU `tiny` first. Written 2026-11-09 for the SmolVLA pivot (ADR-0003; plan from the handoff's
"WO-08 plan" section). This is a plan: nothing here runs until WO-07's baseline is in and the
owner approves each paid run with its cost line.

This work order is **filtered behaviour cloning**. It is not RL and not GRPO, and nothing in its
code, configs, logs or results is called either.

## Done when

SmolVLA fine-tuned on the 25 sim demos per task; reward-filtered BC against the sim success check;
both arms from the same checkpoint, matched budget, held-out seeds; sim gain measured; transfer
tested on the arm only after the sim gain is in.

## Plan

1. **Wrist camera in the sim scene.** The real policy uses front plus wrist, so the sim must too
   before any transfer test. Add a camera mounted on the gripper link (`gripper_link`,
   `src/env/robot.py`), matched to the real wrist module's field of view and mount pose as
   measured in WO-06. The front camera stays `front_left_45`.
2. **Rebuild the sim demos with both streams** (25 per task, oracle v2 recorder settings of
   `configs/demos_v2.yaml`) with the GPU demo-rebuild tooling (`scripts/wo01r_rebuild_demos.py`,
   `scripts/pod/demo_rebuild_job.sh`). Cost this before WO-08 starts; for reference, the
   front-only render-gap pod was planned at about $0.44 expected, $1.50 firm cap (RTX 4090,
   `docs/STATUS.md`, never launched).
3. **A new two-camera sim protocol version** with its own hash. Protocol 0.2 hashes the camera
   code and fixes a single front view, so it stays frozen for historical comparison; the new
   version carries over 0.2's tasks, cells, seeds, success rules and the 4°/step action limit
   unchanged. Constraint found in the code: 0.2's hash covers the bytes of `src/env/__init__.py`,
   `tasks.py`, `cameras.py`, `success.py`, `robot.py`, `randomization.py`, `vector.py` and
   `src/eval/sim_eval.py` (`FILES` in `src/eval/protocol_v02.py`). Editing any of them, even a
   comment, changes 0.2's hash. So pin 0.2's current hash in a test first, then put the wrist
   camera, the two-camera observation function and the new protocol in new files that import
   the frozen ones.
4. **SFT base.** SmolVLA (`policy.base` from config) fine-tuned on the rebuilt 25 demos per task,
   both camera streams, with the same script as WO-07. Entry rule (CLAUDE.md rule 3): a base at
   0% on a task is fixed first, not filtered.
5. **Reward-filtered behaviour cloning.** Sample K rollouts per prompt (task + reset seed) from
   the SFT checkpoint; keep the episodes that succeed by the project's own success check
   (`info["success"]`, computed by `src/env/success.py` in `src/env/tasks.py`; never any
   simulator dense reward); fine-tune on the kept episodes; repeat for a fixed number of rounds.
   Train on the executed (clipped) actions, since those produced the success.
6. **Comparison.** Both arms start from the same SFT checkpoint with matched gradient steps: the
   control arm continues SFT on the original demos only; the filtered-BC arm trains on the kept
   rollouts (the mix with the original demos is fixed in the config before the run). Rollout
   prompt seeds come from a training range registered in `src/eval/seed_ranges.py`; evaluation
   is on held-out protocol seeds under the new two-camera version (0.2's successor: same rules,
   seeds and clip). 100 episodes per cell, 3 seeds, Wilson intervals.
7. **Transfer.** Only after a sim gain is in is transfer tested on the arm, under WO-07's frozen
   real protocol and schedule. The transfer design is written and costed at that point.

## What can be reused (from reading the code, 2026-11-09)

| Code | Reuse | Depends on OFT tokens or PPO |
|---|---|---|
| `src/env/success.py` | Unchanged; it is the filter. Pure tensor checks, policy-agnostic. | No |
| `src/rl/rollout.py` group layout and episode loop | The rule "a group = one reset seed repeated K times in consecutive envs", chunk executed step by step, episode end at success / termination / truncation / step limit, binary reward = success before the end. | The loop calls `RLPolicy.sample()` and reads `step.tokens.shape[1]` for the chunk length. |
| `src/rl/rollout.py` `check_group_reset`, `RolloutTimer`, `instrument_render`, `ClipStats`, the clip via `clip_joint_step` | As is. `check_group_reset` loops over every observation key, so it already covers two image streams. | No |
| `src/rl/rollout.py` `RolloutBatch`, `collect_rollouts` storage | Not as is. It stores `tokens` and sampler `log_probs` (E, S, C, A) for the PPO ratio, and observations only at decision boundaries. Filtered BC needs the observation (both images) and executed action at every env step, written as a LeRobotDataset of the kept episodes. | Yes: tokens, log-probs |
| `src/rl/policy_api.py` (`RLPolicy`, `RolloutStep`, `TokenLogProbs`) | Not reused. Its contract is OFT's 256-bin action tokens, tempered `log_prob`, LoRA-only grads. SmolVLA is flow matching: no tokens, no temperature; rollout diversity comes from the flow noise, whose seed is recorded per episode. | Yes: tokens |
| `src/rl/grpo.py`, `src/rl/trainer.py`, `src/rl/diagnostics.py`, `src/rl/gates.py` | Not reused: group advantages, clip-higher loss, PPO epochs, recomputed old log-probs, log-prob movement, entropy gates. Only the run-record / checkpoint / `metrics.jsonl` patterns carry over. | Yes: PPO |
| `src/rl/paired.py` | Reusable for paired counts of two arms on the same (G, K) prompts. | No |
| `src/rl/toy_env.py`, `toy_policy.py` | Not reused (token toy policy). | Yes: tokens |
| `src/eval/sim_eval.py` (`run_batch`, `evaluate`, `summarize`, `to_markdown`, `episode_seeds`, `clip_joint_step`) | Reused: the policy enters through `ActFn(obs, instructions) -> (N, C, A)`. Two edits needed in new code, not in this hashed file: a two-camera observation function (`src.env.policy_obs` returns only the front `image`), and the video `frames` hook, which reads `obs["image"]`. `exec_steps` already allows executing part of SmolVLA's 50-step chunk. | No |
| `src/eval/protocol.py`, `protocol_v02.py` | Imported for tasks, cells, seeds and regions by the new version; never edited. 0.2's `SETTINGS` fix a single front camera and image size 224, so they do not describe the two-camera runs. | No |
| `src/eval/seed_ranges.py` | Reused to register the rollout seed range and keep it disjoint from protocol seeds. It imports `EVAL_SEED_BASE` / `TRAIN_SEED_STRIDE` from `src/rl/trainer.py` (constants only). | Import only |

## Notes

- One SmolVLA fine-tune and one GPU-sim rollout pass are the cost drivers; price both from a
  short probe before requesting spend.
- If the base is at 0% on a task, there is nothing to filter: fix the base.
- Sim numbers are never presented as real numbers.
