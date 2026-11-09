# Project review, 2026-11-07

Scope: the whole project at `main` @ 1781561 plus the three draft PRs opened today (#23 WO-07
pilot, #24 WO-06 bring-up, #25 recovery SFT). Method: full read of the docs and configs; code
audits of `src/rl`, `src/policy` + `src/data` + `src/env` + `src/eval`, and `src/robot` +
`scripts/wo06_*`; a read-only RunPod billing reconciliation; a survey of the published VLA-RL
recipes and of what others have achieved on the SO-100/SO-101 arm; the test suite run locally.
Nothing was changed in the repo except this file. No pods were started.

Confidence markers: **verified** = reproduced by me or read directly in code/logs/API;
**audit** = from one of the code audits, with file references; **inference** = reasoning from
the numbers; **external** = published source, URL given.

## 1. Verdict in six points

1. **The code is sound.** No bug was found in the GRPO math or wiring, in the SFT/eval token
   path, or in the robot driver (it matches LeRobot 0.4.4 call for call). The test suite passes
   (121 tests, verified) once the venv has all extras installed. The RunPod ledger matches the
   API to the cent ($156.08 vs $156.10, verified) and the account has no idle resources.
2. **The RL null result is explained by scale and signal, not by a bug.** At the same number of
   optimizer steps, the published SimpleVLA-RL curve sits near +40 points where this project is
   at +4 (external, inference). The whole 150-step run carried about 19 upstream steps' worth
   of prompts, with 4 prompts per optimizer step. The logged `approx_kl` and `clip_frac` are
   structurally near zero with one PPO epoch and cannot measure step size; the "updates too
   small" hypothesis was built on numbers that carry no information (audit, verified by the
   logs). The PickPlace-only run plateaued at a constant learning rate after step ~90, so step
   size alone is not the limiter either.
3. **The 17% SFT ceiling has concrete, cheap-to-check causes that were not on the list.**
   Demos stop recording at the exact step the success check fires, often mid-release, with no
   retreat (verified in `scripts/wo01_build_env.py`). All SFT demos were rendered on the Mac
   CPU renderer while every eval runs on the GPU renderer (verified in the run record). The
   oracle is fast (TCP ~22 cm/s, about 9 policy decisions per episode), so one chunk sets the
   grasp from several cm away. 11 of the 25 PickPlace demos end below the intended gripper
   opening (verified on the dataset, 2026-11-08). In the K=16 runs, 61% of groups were all-equal
   at 7.9% sampled success, against ~27% expected if every spawn state were equally hard, so
   success is concentrated in a subset of spawn states (inference; the aggregates cannot say
   what fraction). GRPO cannot lift the states it never succeeds on; that is what the
   recovery-SFT plan is aimed at.
4. **The GPU plan is right on the card and wrong on the gates.** The A100 at $1.79/h is the
   correct choice: the run is rollout-bound (86 env actions/s), and the H100 was only 15% faster
   at 2.2x the price. But the kill switch (64 episodes, +3 points) and the peak stop (32 vs 32
   episodes) decide on noise: a null run passes about half the time and a real +5 run is killed
   about a third of the time. Seeds 1-2 ($28) were stopped on noise. There is no resume path,
   so cheaper interruptible pods cannot be used and a stopped run is lost. Rollout time has no
   model-vs-sim split, so the dominant cost is unprofiled.
5. **High-accuracy real pick-and-place is not on track via the current path alone.** Sim seen
   success is 17-22%, shifted positions 2%, and no few-shot 7B VLA has a published real result
   on this arm. What has worked on SO-101 (external): pi0.5 with its action expert fine-tuned on
   25 real demos (28/40 on one pick-and-place task; its LoRA variant scored ~4%), SmolVLA or ACT
   fine-tuned on ~50 real teleop demos (70-90% in-distribution, SmolVLA scored with partial
   credit), and small-image RL trained in ManiSkill3 with heavy randomization (Squint: 73/80 real
   trials zero-shot). The research thesis (few-shot SFT + RL) can continue, but a practical real
   baseline should run in parallel once the arm is up, and it costs little.
6. **Plan: three tracks, about $25 of cloud before any run over $10.** Hardware bring-up ($0),
   a set of cheap diagnostics and $0 fixes before the next RL run, and a real-data
   pick-and-place baseline (pi0.5 action-expert fine-tune or SmolVLA). Details in section 8.

## 2. Where the project stands

| Item | State |
|---|---|
| WO-01 env/oracle/demos | Done. Oracle 99-100%. |
| WO-02 SFT | Closed below band: seen 17.0 / 10.0 / 2.7% (PickPlace / Push / Stack), shifted 1.7 / 3.7 / 0.0%. |
| WO-03 RL dry | Done, toy env reaches optimum. |
| WO-04 RL sim | Not met. Two-task +5.0, PickPlace-only +3.7 on PickPlace seen; neither distinguishable from SFT (p 0.12 / 0.25). |
| WO-06 hardware | Code done; physical bring-up pending. PR #24 adds the checklist and fixes the calibrate/teleop flows. |
| WO-07 deploy | PR #23 adds the gRPC server, client, deadman handling, trial records and a mock rig. `wo07_real_eval.py` and the placement grid do not exist yet. |
| Recovery SFT | PR #25: plan + local code, no paid run. |
| Spend | $156.08 of the $1,100 cap (API, verified). October account total was $188.03; the extra $31.95 is 10 pods not in this project's ledger (other projects on the shared account). November: $0 so far. |
| Open PRs | #15 (rollout videos, since 2026-10-28), #23, #24, #25 (drafts, today). |
| Local env | `make smoke` failed with 4 import errors until `uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla`; then 121 passed. The Makefile's `setup` target omits `robot`; PR #24's note about `uv sync` uninstalling missing extras is correct. |

## 3. Code readiness

### 3.1 RL loop (`src/rl`) — ready, with reporting and gating fixes needed before the next run

Correct (audit, high confidence): group advantage `(r - mean)/(unbiased std + 1e-6)` in g*K+k
order; PPO ratio against recomputed old log-probs; clip-higher 0.2/0.28; masks cover only the 48
executed action tokens; seq-mean-token-mean weights; gradient reaches every LoRA module
including `lm_head`; no dropout; same temperature for sampling and scoring; same image/proprio
preprocessing in rollout, update and eval; token-to-bin encode/decode consistent with the SFT
dataset; reward = success at any step ends the episode with 1, horizon gives 0.

Definite flaws (audit, verified against the run logs):

- `train/approx_kl` and `train/clip_frac` are averaged over both mini-batches, and mini-batch 1
  is exactly on-policy (ratio 1) with one PPO epoch. The steps that trained on a single
  mini-batch log exactly 0.0. The logged value is at most half the KL on mini-batch 2, which
  was measured after one Adam step on different prompts. RMS per-token |delta log p| is about
  0.013, versus a clip edge of 0.22, so clip fraction is ~0 whether or not learning happens.
  `src/rl/grpo.py:141`, `src/rl/trainer.py:443-476`.
- Kill switch: 64 greedy episodes per task at p ~ 0.13 gives a paired-difference SD of 4-6
  points against a +3 bar (`configs/rl.yaml:113-117`). Peak stop compares two 32-episode evals
  (`configs/rl_pickplace.yaml:11-12`). Neither can distinguish +3 from 0. A higher-power signal
  already exists for free: `rollout/success_rate` over ~290 episodes per step (SE ~0.5 points per
  10-step window) plus entropy. Use it to monitor and to decide whether to keep spending; its
  episodes share scenes and come from a changing policy, so only the fixed held-out protocol
  eval measures improvement.
- No resume: `save_checkpoint` writes `optimizer.pt`, nothing loads it (`src/rl/trainer.py:498`).
- The STATUS claim that "every step hit the refill cap" was true of the K=8 probes only. In the
  K=16 runs 16-26% of steps hit it; trained trajectories averaged 122-126 of 128.
- Grad clip 1.0 is not a lever under Adam (rescaling g cancels); lr and epochs are.
- Minor waste: Stack (and Push in the PickPlace-only run) evaluated at every eval (~$0.5/run);
  redundant mini-batch-1 recompute when `ppo_epochs: 1` (~$2/run); the probe's 3 steps are
  discarded and the 64-episode baseline re-run (~$1.2/pod).

Why the gain was small, ranked (audit + external, inference):

1. Signal-starved updates: 4 prompts per optimizer step, 2 steps per update, 1,200 groups and
   300 Adam steps per run. Upstream: 64 prompts x 8, ~4 Adam steps per update, 8 GPUs for 1-2
   days (192-384 GPU-hours vs our 31).
2. Success is concentrated in a subset of spawn states. In the K=16 PickPlace-only run the
   all-equal-group rate was 61% falling to 51% while sampled success rose 7.9% to 12.6%; a policy
   equally likely to succeed on every spawn state would drop about 27% falling to 12%
   (independent-sampling model). A two-population fit puts ~40-50% of states in a solvable pool
   at ~20-26% sampled success, but that is one simplified model: the aggregates cannot identify
   the share or set a ceiling on what GRPO can reach. (An earlier draft combined a K=8 probe's 72%
   drop rate with a 13% greedy figure and derived a +17 ceiling; both withdrawn 2026-11-08.)
3. Sampled-vs-greedy gap: at T=1.0 sampled success (7.9%) is half the greedy rate (17%) with
   entropy 1.3 nats/token. The run spent its updates shedding sampling noise (entropy 1.32 to
   1.07), which barely moves the argmax that the protocol evaluates.
4. Step size: LoRA r32 with alpha 16 (scale 0.5) at lr 5e-5 is about 2x below the upstream-
   equivalent (RLinf, RL4VLA and RIPT's default use 1e-4 for LoRA). This rests on the published
   recipes, not on the logged KL/clip figures, which are uninformative (above). Plausibly
   explains the slow first 90 steps, not the plateau. Measure policy movement on a fixed batch
   before and after one update before changing the learning rate further.
5. K=16 x 8 prompts is probably the wrong shape. 16 prompts x K=8 at the same cost doubles the
   prompts per update; the cost is more all-equal groups on hard states, which dynamic sampling
   already discards.

### 3.2 SFT, data and eval (`src/policy`, `src/data`, `src/eval`) — ready, with four real defects

Definite (verified or audit with file references):

1. **Demos are truncated at the first success step.** `scripts/wo01_build_env.py:106-137`
   records frames `[:ep_len]` where `ep_len` stops the step success fires. In 5/10 PickPlace and
   6/10 Stack demos the last gripper command is 0.22 rad, not the 0.45 release target, and there
   is at most 1 retreat frame. LeRobot pads the final chunk by repeating that action 7 times and
   the loader trains on the pad (`src/data/oft_dataset.py:60-76`). The policy learns "stop
   opening and freeze"; an episode with the cube in the bowl but still touching a jaw cannot
   finish. Fix: record ~15 frames past success through full open and retreat.
2. **Renderer mismatch between training and eval.** Both demo sets were recorded on the Mac
   (`runs/20261027T053512Z-wo01-sim-4ebf/record.json`: device cpu, Apple M3 Pro, MoltenVK), every
   eval and every RL rollout runs on the GPU sim with the NVIDIA renderer. ADR-0001's
   "identical tokens on demo frames" check ran on Mac frames only. Unmeasured gap. Check: render
   the demo seeds on a pod and diff against the dataset frames, compare predicted tokens (~$0.1).
   Fix: record demos on the pod.
3. **The eval protocol was never frozen.** `PROTOCOL_VERSION = "0.1-draft"`, and the hash is the
   SHA-256 of `protocol.py`'s bytes only (`src/eval/protocol.py:17,101-103`). It does not cover
   success thresholds (`src/env/success.py`), spawn sampling (`src/env/tasks.py`), the horizon,
   control/physics rates, camera, image size, the stop-at-success rule or the sim backend. Rule
   4 asks for every number to cite a protocol hash that pins the evaluation; this one does not.
4. **The real arm's 4 deg/step safety clip is absent from sim eval and RL** (`configs/real.yaml:28`
   vs `src/eval/sim_eval.py`). Chunk-boundary jumps that sim accepts will be clipped on the arm.

Also (audit): eval stores only success counts per cell, so failures cannot be classified
(`src/eval/sim_eval.py:122-133`); the WO-02 checkpoint sweep used protocol seed-0 episodes 0-49
(small upward bias, already disclosed in PR #25); camera jitter uses the global torch RNG
(`src/env/tasks.py:337`); default `num_envs: 64` simulates 128 episodes for 100 when run by hand;
`configs/real.yaml` teleop instructions drop "red" (PR #24 already switches recording to the
protocol strings).

Why SFT plateaus, ranked, with the cheapest check (audit, medium confidence):

1. Too few, too coarse decisions. About one 8-step chunk per phase; TCP moves a median 44 mm per
   chunk (90th pct ~80 mm); the cube is ~19 px wide. The pre-grasp pose comes from one absolute
   prediction made 5-10 cm away. Receding horizon made things worse because the chunks were
   trained open-loop. Check on the Mac: TCP error of the predicted descend endpoint on
   fresh-seed oracle frames. Fix: a slower oracle approach (~1 cm/step) and/or chunk 4 trained
   as such.
2. Truncated release (above).
3. Render gap (above).
4. Overfitting to joint-position fingerprints: 13 epochs at constant lr 5e-4, a fully trained
   proprio MLP, 0.02 rad start-pose noise making each demo unique; unseen objects score the same
   as seen while shifted positions collapse to 2%. No held-out metric exists in the SFT loop.
   Check on the Mac: held-out L1 on fresh seeds, image/proprio swap and blank ablations.
5. Contradictory labels: cube yaw sampled in +/-45 deg with a grasp-face rule that flips the
   wrist-roll target by ~90 deg at the edge (~20% of episodes within 10 deg of it); 15 of 25 Push
   demos contain an oracle restart.

Success checks (`src/env/success.py`, audit): evaluated every step, latched on the first passing
step; PickPlace needs the cube inside the bowl footprint and height band, no jaw contact, at
rest; Push needs within 2.5 cm, never lifted, not grasped, at rest (slow push-through counts);
Stack has no hold time. A real judge scoring after the arm stops will be stricter; define the real
judging rule before WO-07.

### 3.3 Robot side (`src/robot`, `scripts/wo06_*`, PRs #23/#24) — ready for bring-up, with safety gaps

Verified against the installed LeRobot 0.4.4 (audit): class names, config fields, calibration
path, observation/action keys (`<motor>.pos`, gripper 0-100), bus calls, the Goal := Present
hook before every torque enable. Safety filter: URDF limits minus 5 deg narrowed by calibration,
TCP workspace box via the same URDF FK as sim, 4 deg/step and 60 deg/s caps (above the oracle's
3.7 deg/step), latched deadman on stale frame / serial error / stall. PR #23's client reads a
fresh frame every tick (needed: a chunk lasts 533 ms against a 300 ms deadman), treats RPC
timeouts as HOLD, logs latency and records trials with the protocol hash. PR #24 fixes real
hazards in the stock LeRobot flows (calibrate re-enabling torque, exception exits dropping the
arm, infinite redo loop).

Gaps, ranked:

1. No hardware stop path. If the Mac process dies the servos hold their last goal with torque
   on. An inline power switch on the follower supply is mandatory (PR #24's checklist says so).
2. No servo temperature or current monitoring. STS3215s overheat; the gripper is exempt from the
   stall check and squeezes at 0%. Add a periodic `Present_Temperature` / `Present_Current` read
   with a fault response validated on its own: HOLD keeps the servos energized and a powered
   park adds motion during the fault, so the response is torque-off (owner supporting the arm,
   as PR #24's prompt already does), not HOLD and not park. ~2-3 h of work.
3. Exit, Ctrl-C or an exception cuts torque and the arm drops. Add `park()` to the rest pose
   before torque-off on healthy shutdown only (PR #24 adds the "support the arm" prompt, which
   is the manual version); fault paths keep the torque-off response in gap 2.
4. Camera exposure, white balance and focus are not locked anywhere; a C920 auto-adjusts by
   default. Lock them with a vendor tool before the camera match. Camera pose alignment is
   visual overlay only (no PnP); the sim's camera jitter (+/-2 cm, +/-1.5 deg fov) sets the
   tolerance, and `randomize` is off in `sim.yaml`.
5. Camera latency is invisible: the client starts its clock after the camera driver returns the
   frame (`src/robot/client.py:45` on PR #23), so exposure and buffering are excluded. Use the
   driver's capture timestamp only after verifying it represents capture time and aligns with the
   host clock; otherwise measure optical-event-to-command latency once with an LED or clock test.
   The <150 ms WO-07 target otherwise excludes the largest term.
6. The gripper floor: sim "closed" is -0.17 rad (-10 deg), the real safety floor is -5 deg, so
   every real grip is weaker than in sim (PR #23 flags this). Decide before WO-07.
7. Missing for WO-07: `wo07_real_eval.py` (grid sampler, 50-trial loop, operator prompt),
   `docs/assets/placement-grid-a3.pdf`, the video overlay script. PR #23 covers the rest.

## 4. Comparison with the published recipes (external; URLs in section 10)

| Recipe | Model / FT | lr | prompts x samples per update | Adam steps per update | GPUs, time | Result |
|---|---|---|---|---|---|---|
| This project | OFT-7B, LoRA r32 | 5e-5 | 8 x 16 = 128 | 2 (1 epoch) | 1 x A100, 31 h | 17 -> 21-22 |
| SimpleVLA-RL | OFT-7B, full | 5e-6 | 64 x 8 = 512 | ~4 (1 epoch) | 8 x A800, 1-2 days | LIBERO 1-demo avg 48.9 -> 96.9; Long 17.3 -> 91.7 |
| RIPT-VLA (OFT) | OFT-7B, LoRA r32 | 1e-4 (5e-5 on Long) | 24 x 8 = 192 | 1 epoch | 4 x A5000 | Long 92.9 -> 93.8 |
| RIPT-VLA (QueST) | 20M, full | 1e-6 | 10 x 16 | 20 epochs | 1-3 GPUs | 1-shot ~4 -> 97 |
| RLinf-VLA GRPO | OFT-7B, LoRA | 1e-4 | 32 groups x 8 | batch 640 | 8 x H100 | ManiSkill 28.1 -> 94.1 |
| RL4VLA (PPO) | OpenVLA-7B, LoRA r32 | 1e-4 | 64 envs x 80 steps | 32 | 1 x A100, ~42 h | +42.6 OOD |
| piRL | pi0/pi0.5 expert, full | 5e-6 | batch 2048-5120 | 4-5 epochs | 8 x H100 | 57.6 -> 97.6 |

Read off SimpleVLA-RL's LIBERO-Long one-demo curve (about +/-3 points): +19 at step 25, +32 at
step 50, +42 at step 75, +63 at step 150. Matching this project on used trajectories lands at
about +24; on Adam steps at about +42; on updates at about +63. The project is at +4-5. The
numbers the project cites check out: 7.3 -> 25.4 is RoboTwin 2.0 with a 100-trajectory base
(5-task average); 17.5 -> 38.5 is real-robot deployment of a sim-trained policy with no real
demos. CLAUDE.md's "primary table not re-checked" note can be removed.

What they do that this project does not: more trajectories and 4-32 Adam steps per update;
LoRA lr 1e-4; temperature 1.5-1.7 on a healthier base (not applicable at ours: 1.6 dropped 93% of
groups); 6-12x the GPU-hours. The group-dropping itself is standard; the project's waste rate is
the symptom of the weak, unevenly-successful base, not of the method.

## 5. Compute and money

Reconciliation (verified via the RunPod API): 17 project pods sum to $156.08. No pods, volumes,
endpoints or templates exist, so the account costs $0/h now. Remaining under the $1,100 cap:
about $944; under the $700 core cap: about $544.

Current list prices ($/h, secure / community): RTX 4090 0.89 / 0.34; A100 80 GB 1.79 / 1.19-1.39;
H100 PCIe 2.89 / 1.99; H100 SXM 3.99 / 2.69; RTX PRO 6000 96 GB 2.49 / 1.69; H200 5.29 / 3.59.
Community pods are 22-33% cheaper on the A100 but can be interrupted; without a resume path
they are unusable for a 30 h run. With resume, a 25-step checkpoint cadence loses at most ~3 h.

Where the RL dollar goes (inference from measured timings): per 150-step run, rollout ~9 x
44.5 s = ~6.7 min and update ~3.4 min of a ~11.5 min step. About 58% of rollout episodes land in
dropped groups (~$18 of $55). The forward pass at batch 32 should take ~1/3 of the 44.5 s; the
rest is sim stepping, rendering, preprocessing and host transfers, none of it instrumented.
Levers, in order of expected $ saved per run: raise base success (fewer dropped groups, the
recovery-SFT purpose); profile and fix the rollout overhead; 64 envs per batch (38 GB VRAM
headroom); 16 x 8 instead of 8 x 16; drop eval-only tasks from periodic evals.

Gates: replace the 64-episode kill switch and the 32-episode peak stop with windows of
`rollout/success_rate` and entropy (free, SE ~0.5 points per 10 steps), and keep the greedy
evals as sanity checks only. Seeds 1-2 cost about $28 and produced only early-stop readings,
with no completed comparison; what another gate would have saved is unknown. The 32-episode
peak stop did produce one wrong conclusion (the step-50 "peak").

Hidden costs: pod setup ~0.5 h per run ($0.5-0.9); a 30 GB network volume with the 15 GB base
checkpoint and the uv cache would cost ~$2/month and cut setup to ~0.1 h, roughly break-even at
the planned run count. Hub storage is free at this scale.

The staged budget in STATUS.md is sensible in shape. Two adjustments: run the ~$5 of
diagnostics below before the $60 GRPO diagnostic, and fold the demo-recording fixes into the
recovery-SFT pod so the new base is built on corrected, GPU-rendered demos.

## 6. Is high-accuracy pick-and-place on track?

Not on the current path alone. The best sim number is 22% on seen positions and 2% on shifted
positions, so real placement noise alone would push most trials into the near-zero regime even
with perfect sim-to-real transfer. Add the known transfer gaps (stiff, strong sim servos with
no lag or backlash; no camera latency in sim; a -5 deg gripper floor; a single 224 px view with a
19 px cube) and the WO-07 SFT-vs-SFT+RL table will most likely read single digits in both
columns. That is still a publishable negative result under rule 9, but it is not a working
pick-and-place.

What others reached on this arm (external):

| Project | Method | Demos | Compute | Real result | URL |
|---|---|---|---|---|---|
| Squint (UCSD) | visual SAC, 16x16 wrist image, ManiSkill3 SO-101, heavy DR | 0 | ~15 min, 1 x RTX 3090 | 73/80 zero-shot across 8 tasks | arxiv.org/abs/2602.21203 |
| lerobot-sim2real | RGB PPO from scratch, 128 px, camera/fov DR, greenscreen | 0 | ~1 h, RTX 4090 | grasp cube zero-shot (video only, no rate) | github.com/StoneT2000/lerobot-sim2real |
| Cowling 2026, SO-101 | pi0.5 action-expert fine-tune | 12 / 25 / 50-200 | 10.6 A100-h per run (paper) | 16/40 at 12 demos, 28/40 at 25, ~70% plateau; pi0.5 LoRA 4/110; ACT, DP, SmolVLA scaling curves | arxiv.org/html/2610.37170 |
| SmolVLA paper, SO-101 | 0.45B VLA fine-tune | 50 (10 at 5 positions) | 1 GPU, hours | pick-place-lego 90% ID, 50% OOD; ACT 70 / 40 (partial-credit scoring) | arxiv.org/html/2506.01844 |
| SmolVLA paper, SO-100 | same | 50 per task | same | pick-place 75, stack 90, sort 70; pi0 100 / 40 / 45 (partial-credit scoring) | same |
| ACT case study | ACT | 30-50 | RTX 5080 | 5/5 fixed start; 3/7 varied starts | openelab.io (SO-ARM101 ACT case study) |
| LeRobot HIL-SERL | real-world SAC + interventions | ~15-30 | 1 GPU, 1-4 h on hardware | near-100% on narrow pick-lift tasks (docs; no table) | huggingface.co/docs/lerobot/hilserl |
| pi0.5 community | LoRA few-shot | 8-60 | n/a | barely moves (8-30 demos); 1.7% in sim with 60; consistent with the LoRA row above | ozyphus.github.io/yerobot.html |
| OpenVLA / OFT on SO-10x | none found | | | | |

Takeaways: on this arm, 65-90% real success comes from 25-50 real demos per task (pi0.5
action-expert fine-tune at 25; SmolVLA and ACT at 50, partial-credit scored) or from small-image
RL with heavy randomization; for pi0.5 the fine-tuning recipe mattered more than demo count
(full action expert vs LoRA). Few-shot 7B VLAs have no real result here.
Practical lessons that transfer: align the real camera to the sim camera with a live overlay and
a multi-axis mount; composite a photo of the real empty table behind the render; randomize
colour, lighting, camera pose and fov, gripper stiffness and joint offsets (colour jitter alone
was worth 15 points of real success in Squint); slow the real arm down at deployment; grip pads
on PLA jaws; report success as fixed-count trials over varied positions.

## 7. Risks not yet accounted for

- WO-04's benchmark needs +20 on 2 of 3 tasks, but Stack is at 2.7% with 0/64 at RL step 0. No
  RL recipe lifts a 0% base (the project's own rule 3). Either Stack's SFT is fixed first or the
  benchmark is stated on PickPlace + Push.
- The published recipes' "few-shot" bases were at 17-49% from demos recorded by humans at slow
  speed in a mature benchmark; this project's base comes from a fast scripted oracle in a new
  scene. The gap in base quality is the largest uncontrolled variable.
- Protocol version "0.1-draft" with a file-bytes hash means a fix to `success.py` or `tasks.py`
  would silently invalidate every number in results.md. Freeze a 0.2 with a composite hash before
  the next eval.
- The deploy stack serves a 7B model from a pod over an SSH tunnel; every real trial session
  costs pod time (~$0.9/h on a 4090, model load ~5 min) and depends on the tunnel. A SmolVLA or
  ACT policy runs on the Mac with no pod.
- Servo heat under sustained teleop and eval sessions has no instrumentation.

## 8. Recommended plan

### Track A: hardware bring-up (this week, $0 cloud)

Merge PR #24 after a read, then follow `docs/bringup/WO-06-checklist.md` steps 0-10. Non-
negotiable additions: an inline power switch on the follower supply; lock camera exposure,
white balance and focus; measure camera latency once. Collect the 5 sanity episodes per task,
then keep going on PickPlace to 50 teleop demos (10 at each of 5 positions covering the seen
region, as in the SmolVLA paper; the first 25 already unlock the pi0.5 option). That is about an
hour of teleop and it unlocks Track C.

### Track B: fix, diagnose, then retrain (next 1-2 weeks, ~$25 before any run over $10)

$0 code fixes first (one PR, CPU-tested):
1. Gate RL on `rollout/success_rate` and entropy windows; log per-(epoch, mini-batch) clip and
   k3-KL with mini-batch 1 excluded; surface `grad_norm`. Keep greedy evals as sanity only.
2. Resume from checkpoint (load optimizer state and step counter).
3. Record demos through release and retreat (~15 frames past success); drop pad frames in the
   loader; slow the oracle approach and descend to ~1 cm/step.
4. Protocol 0.2: composite hash over `protocol.py`, `success.py`, the `tasks.py` sampling
   constants, horizon, image size, chunk execution rule and backend; store per-episode outcomes
   and end-state flags; apply the real 4 deg/step clip in sim eval and RL rollouts.
5. Rollout timing split (model / sim step / render / preprocess) in the probe output.
6. `configs/rl.yaml`: 16 prompts x K=8, lr 1e-4 to 5e-4 for LoRA, `ppo_epochs` 2 (or mini-batch
   16 trajectories), drop eval-only tasks from periodic evals.

Diagnostics (~$5 total, one 4090 pod plus a short A100 probe, each needs an explicit go):
- Render-gap check: render demo seeds on the GPU sim, diff against the dataset frames, compare
  predicted tokens on both (~$0.1).
- Offline replay lr sweep: one stored 128-trajectory batch, lr in {5e-5, 2e-4, 5e-4, 1e-3} x 8
  Adam steps; log on-batch k3-KL, clip fraction, delta log p split by advantage sign, greedy
  argmax flip rate (~$2.5). Decides lr without a rollout run.
- Temperature probe, rollouts only: drop rate and sampled success at T in {0.7, 1.0, 1.3}
  (~$1-2). The published recipes push T up; the sampled-vs-greedy gap says push it down; measure.

Then the amended recovery plan (PR #25), ~$10-15: re-record the 25 demos/task on the pod with
the fixed recorder (~$0.3), SFT from scratch (A100, 8000 steps, ~$5), full protocol eval (~$1);
then corrections + matched control on top of that base as PR #25 specifies (~$7). Only if the
new base clears >= 10% on 2 of 3 tasks and the diagnostics show healthy on-batch KL: the GRPO
diagnostic (20-75 steps, $11-29), then confirmatory seeds ($85-160). Expected outcome if all of
this goes right: PickPlace seen in the 30-45% range after RL; +20 on two tasks remains unlikely
while Stack is at 2.7%.

### Track C: a real-data pick-and-place baseline (as soon as 25-50 demos exist, ~$0-25)

Two options, same deploy path. (a) pi0.5 with the action expert fine-tuned on 25-50 real
PickPlace demos: the strongest evidenced few-shot result on this arm (28/40 at 25 demos, Cowling
2026). The paper reports 10.6 A100-hours per training run, about $19 at the current secure
A100 rate before setup and evaluation; the LoRA variant is not an option (4/110). Verify that the
pinned LeRobot 0.4.4 pi0.5 code and the Mac client can serve it, and price a probe first.
(b) SmolVLA (already pinned as the WO-01 smoke model) on 50 demos, ACT as a cheaper second arm;
train overnight on the Mac for free or on a 4090 for ~$3-10. Deploy through PR #23's client (the
policy runs on the Mac, no pod) and score 50 trials with the WO-07 protocol, binary success, so
the SmolVLA paper's partial-credit 70-90% is a reference, not a forecast. This delivers a working
pick-and-place, validates the whole real-eval loop, camera, safety and operator protocol before
the VLA checkpoints are deployed, and gives a real-data reference column for the SFT vs SFT+RL
table. It is post-training of a pre-trained VLA, so it sits inside the project's thesis; it is
not few-shot, so label it as such.

### Decisions needed from the owner

1. Approve the $0 fix PR scope above (gates, resume, recorder, protocol 0.2, config shape).
2. Approve ~$5 of diagnostics (render gap, offline lr sweep, temperature probe).
3. Amend the recovery plan to rebuild the base on corrected GPU-rendered demos first (~$6 more).
4. Add Track C (pi0.5 action-expert or SmolVLA/ACT on 25-50 real demos) to the plan as the
   practical pick-and-place deliverable, and decide how it is labelled next to the few-shot
   results.
5. Restate the WO-04 benchmark on PickPlace + Push, or fund a Stack SFT fix.
6. Few-shot accounting for correction episodes (PR #25's question; recommendation: separate
   labelled budget).
7. Review and merge or close PRs #15, #23, #24, #25.

## 9. Fix list before the next paid run (all $0)

- [ ] Gates on rollout success/entropy windows (monitoring and spend decisions only; protocol eval decides improvement); fixed-batch pre/post-update policy-movement diagnostic; grad_norm in dashboard
- [ ] PR #25: replace the "escalate to lr 1e-3 when clip fraction rounds to 0" rule with the fixed-batch diagnostic; keep the matched ordinary-demo control; log per-episode outcomes for paired comparison
- [ ] Resume path
- [ ] Demo recorder: frames through release + retreat; no pad frames in training
- [ ] Slower oracle approach/descend
- [ ] Protocol 0.2 with composite hash, per-episode records, 4 deg/step clip; keep 0.1-draft numbers; propose (not run) a separately approved rerun of the existing SFT and RL checkpoints under 0.2, repriced from measured runtime (~$3 on a 4090 is provisional)
- [ ] Rollout timing split in the probe
- [ ] rl.yaml shape: 16 x 8, lr 1e-4+, ppo_epochs 2, no eval-only tasks in periodic evals
- [ ] Camera jitter seeded from the episode seed; RL seed >= 10 rejected (PR #25 has this)
- [ ] Makefile `setup` includes `--extra robot`
- [ ] Servo temperature/current monitor with a torque-off fault response (not HOLD, not a powered park); `park()` before torque-off on healthy shutdown only; camera capture timestamp verified before latency is called end-to-end; placement grid PDF; `wo07_real_eval.py`

## 10. Sources

SimpleVLA-RL: arxiv.org/abs/2509.09674, github.com/PRIME-RL/SimpleVLA-RL (examples/run_openvla_oft_rl_libero.sh);
RIPT-VLA: arxiv.org/abs/2505.17016, github.com/Ariostgx/ript-vla; RLinf-VLA: arxiv.org/abs/2510.06710,
github.com/RLinf/RLinf (examples/embodiment/config/maniskill_grpo_openvlaoft.yaml); RL4VLA: arxiv.org/abs/2505.19789;
piRL: arxiv.org/abs/2510.25889; VLA-RL: arxiv.org/abs/2505.18719; TGRPO: arxiv.org/abs/2506.08440;
OpenVLA-OFT: github.com/moojink/openvla-oft (LIBERO.md); LoRA lr guidance: thinkingmachines.ai/blog/lora;
Squint: arxiv.org/abs/2602.21203, github.com/aalmuzairee/squint; lerobot-sim2real: github.com/StoneT2000/lerobot-sim2real
(docs/zero_shot_rgb_sim2real.md); SmolVLA: arxiv.org/html/2506.01844 (real-world scoring: section 4.1, partial credit);
Cowling 2026, "From 12 to 200 Demonstrations" (SO-101, pi0.5/SmolVLA/ACT/DP): arxiv.org/html/2610.37170;
HIL-SERL: huggingface.co/docs/lerobot/hilserl;
GR00T N1.5 SO-101: huggingface.co/blog/nvidia/gr00t-n1-5-so101-tuning; Microsoft residual RL: arxiv.org/pdf/2606.18953;
RunPod prices: runpod.io/pricing and the RunPod API, 2026-11-07.

## Corrections log

- 2026-11-08, after an independent read-only critique: withdrew the 70/30 bimodal split, the +17
  ceiling and the "$28 saved" claim; replaced the pi0.5 row with Cowling 2026 (action-expert
  fine-tune 28/40 at 25 demos; LoRA 4/110); marked SmolVLA scores as partial credit; changed the
  thermal/overcurrent response from HOLD to torque-off and limited park() to healthy shutdown;
  noted that the latency clock starts after frame return; noted that PR #25's lr 1e-3 escalation
  rule keys on an uninformative metric. The KL/clip finding, the spend reconciliation (RunPod API
  vs ledger) and the local test run (121 passed) were made in the 2026-11-07 session and were not
  independently reproduced by the critic.
