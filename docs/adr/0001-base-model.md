# ADR-0001: Base VLA model

Status: accepted (2026-10-27) — OpenVLA-OFT kept; no π0.5 switch (owner). SFT stayed below the original
20–60% target band; RL starts from the attempt-4 checkpoint (step 4000, seen 18/16/6%) under the amended rule 3.
Date: 2026-10-26 (WO-02 implementation findings added the same day)
Work order: WO-02

## Decision

Default to OpenVLA-OFT with LoRA, under the output head used by the vendored RL code
(SimpleVLA-RL's cross-entropy head), re-doing few-shot SFT under that head so RL and SFT share one
checkpoint format. Fall back to π0.5 (openpi LoRA, RLinf πRL path) if OpenVLA-OFT few-shot SFT is
below 20% on seen positions after a reasonable tuning pass (≤3 SFT runs).

### How it is run (WO-02)

- **No transformers fork.** `moojink/transformers-openvla-oft` is v4.40.1 plus one commit
  (`bc339d9`) that makes `LlamaSdpaAttention` bidirectional (`is_causal=False`, pad-only mask).
  We vendor SimpleVLA-RL @ `7c51662`'s OFT model code (`src/policy/oft/vendor/`, MIT) and run it on
  stock **transformers 4.57.6** (already locked for SmolVLA) + **timm 0.9.16**, with two edits:
  `_supports_sdpa` as a class attribute, and a forward pre-hook that hands SDPA a 4-D boolean
  "attend to every non-pad key" mask. Verified on CPU (py3.11, torch 2.6.0): logits and greedy
  actions **bit-identical** to the fork; without the hook the model is causal (|Δlogit| 0.19).
  A fork-saved checkpoint loads bit-identically; all 982 tensors of openvla-7b map onto the model.
  No flash-attn (the fork's flash path was causal too — SDPA is asserted at load), no
  bitsandbytes, no TensorFlow, no uv `conflicts`. The fork + peft 0.21.2 does not even import.
- **Head / tokens.** 256 uniform bin edges on [-1, 1] → 255 centres; tokens are indices into the
  256 vocab rows 31744..31999 (logit slice `[..., -320:-64]`), sampled/scored with
  `softmax(logits / T)` — identical to `generate_action_verl` / `dp_rob.py` (parity tests).
  SFT loss is cross-entropy on that same 256-wide slice (upstream SFT used full-vocab CE; the
  slice is exactly the distribution RL samples).
- **Token layout (one builder for SFT, eval, RL):** `<s> In: What action should the robot take to
  {instruction}?\nOut: ▁ [6×8 action slots] </s>`, right-padded. `▁` (29871) is what OFT's
  training tokenisation produces and what its `predict_action` re-inserts; SimpleVLA-RL's rollout
  drops it — we keep it everywhere.
- **Embodiment changes.** ACTION_DIM 6 (absolute joint targets, rad), chunk K = 8 (0.53 s at
  15 Hz), proprio on (new 6→4096→4096 GELU projector, trained in full in SFT, frozen in RL),
  one 224 px front image (90%-area crop: random + colour jitter in SFT, centred at eval/RL).
- **Normalisation bounds are not demo min/max** (that would cap the reachable action space at
  the demo range and make the shifted cell unreachable): joint limits ∩ FK-reachable task
  workspace (protocol shifted region + 5 cm, TCP tilt ≤ 75°, 0.5/99.5% FK quantiles), ∪ the rest
  pose, a π-wide wrist-roll window, the commanded gripper range, + 0.05 rad, ≤ 1°/bin enforced.
  On the SO-101 URDF: pan [-1.29, 1.28], lift [-1.75, 0.93], elbow [-1.11, 1.69], wrist_flex
  [-1.56, 1.66], roll ±1.62, gripper [-0.18, 0.65] rad; worst resolution 0.73°/bin.
- **LoRA.** r 32, α 16, dropout 0, every nn.Linear incl. `lm_head` (peft 0.11's "all-linear", as
  in OFT/SimpleVLA-RL adapters; peft 0.21 would drop lm_head), proprio projector excluded.
- **Base checkpoint.** `openvla/openvla-7b` @ `47a0ec7f` (MIT; Llama 2 licence on the LLM
  weights; 15.1 GB bf16; 7.558B params with the proprio projector; 126M trainable). Attempt-2 init
  if SFT < 20%: `Haozhan72/Openvla-oft-SFT-libero10-trajall` (MIT), already adapted to parallel
  decoding + chunking + the CE head.
- **SFT recipe** (SimpleVLA-RL issue #10 = OFT `finetune.py --use_l1_regression False`): AdamW
  lr 5e-4, batch 8, MultiStepLR ×0.1, image aug, no warmup; 20k-step cap, checkpoint every 2.5k,
  pick the checkpoint that lands in the 20–60% band on seen positions. *Superseded by the owner's
  budget: LoRA only, 2000 steps (sim.yaml), then 8000 — see "Measured SFT results".*

## Why

- OpenVLA-OFT is the model the open GRPO recipe targets; its action chunking gives ~108 Hz
  inference on LIBERO, which is what makes rollouts affordable on one GPU.
- π0.5 has the stronger prior (few-shot 77% on LIBERO before RL in πRL) but needs >22.5 GB for
  LoRA and the flow-matching RL path (Flow-SDE/Flow-Noise) is more complex to run single-GPU.
  openpi @ `215abfb` also pins torch 2.7.1, jax 0.5.3, transformers 4.53.2 (overwriting files in
  site-packages), numpy<2, tensorflow-cpu 2.15 and a git lerobot — every one conflicts with this
  repo's pins — and its PyTorch port does not support LoRA.
- SmolVLA is used only to smoke-test the data path in WO-01; its ceiling is too low for the
  headline result.

## Compute (estimates — replaced by `scripts/wo02_sft.py --probe-steps 20`)

Forward ≈ 4.8 TFLOP/sample (LLM 2 × 6.61e9 × 326 tokens + ViTs 0.36); LoRA training ≈ 2× forward
(3× with gradient checkpointing). A100-80GB: batch 8, no checkpointing, ~41 GB, 0.5–0.62 s/step.
RTX 4090: bf16 + checkpointing (Llama layers and ViT blocks), micro-batch 4 × 2, ~19–21 GB,
1.6–2.0 s/step (OpenVLA paper anchor: LoRA r32 batch 16 = 59.7 GB without checkpointing; int8 lost
13 points of success, so no 8-bit base). 20k steps: A100 ≈ 3.3–3.9 h incl. setup ≈ $6–7; 4090 ≈
9.4–11.6 h ≈ $8.4–10.3 — the A100 is cheaper per run. Rollouts: ~0.07 s per chunk at batch 1
(OFT Table II), ~230–285 actions/s batched on an A100.

## Alternatives considered

- GR00T N1.7 — 40 GB+ fine-tune, no open RL recipe targets it.
- Train an ACT/Diffusion Policy from scratch — not post-training of a pre-trained model; out of scope.
- The transformers fork as a conflicting uv extra — resolves, but splits transformers, tokenizers
  and peft into two versions each and cannot coexist with the `gpu` extra (peft 0.21 vs 4.40.1).

## What would reopen this

OpenVLA-OFT SFT <20% on seen after 3 runs; or VRAM for OFT + sim + rollouts exceeding 80 GB at
≥8 parallel envs.

## Measured SFT results (WO-02, RTX 4090, seen cell, 50 episodes/task, GPU sim, protocol `e9ce5bb4`)

In band = 20–60% on ≥ 2 of 3 tasks. Best checkpoint per attempt (all checkpoints in STATUS/results):

| Attempt | Change | Steps | Best seen (PickPlace / Push / Stack) | Train fit at end (token acc / L1) | $ |
|---|---|---|---|---|---|
| 1 | lr 5e-5 (sim.yaml) | 2000 | 8 / 4 / 2% (step 1000) | 0.73 / 0.016 | 0.94 |
| 2 | lr 5e-4 (OFT LoRA rate) | 2000 | 20 / 6 / 4% (step 2000, still rising) | 0.69 / 0.024 | 1.17 |
| 3 | lr 5e-4, 8000 steps | 8000 | 22 / 4 / 4% (step 3000); plateau ~15 / ≤8 / ≤8% | 0.84 / 0.010 | 4.01 |
| 4 | 25 demos/task (A100) | 8000 | 18 / 16 / 6% (step 4000) | — | ~5.0 |

Ruled out on the way (free, Mac): the 30 demos replay to success with zero joint-state drift
(data ↔ env aligned); the SFT forward and the eval path give identical action tokens on demo
frames (no preprocessing mismatch). Attempt 3 fits its demos far better than 1–2 yet closed-loop
success plateaus: the policy reproduces 10 demos/task but does not recover once it drifts off
them — few-shot covariate shift, which more steps alone do not fix.

## Owner decision path (2026-10-27) and why

1. **Receding-horizon recheck, eval only (≤ $0.5).** Re-score existing checkpoints executing 4
   of the 8 predicted actions before re-querying. Not an SFT attempt: an inference setting that
   directly targets compounding open-loop error. If ≥ 2 tasks land in the band, OpenVLA-OFT stays.
2. **If not: 25 demos/task, one SFT run (8000 steps, lr 5e-4, ~$4) before π0.5.** CLAUDE.md rule 3
   is amended from ≤ 10 to ≤ 25 demos/task — still few-shot. Why before π0.5: the failure mode is
   data coverage, which more demos address directly and cheaply; π0.5 would replace the whole
   stack — a separate openpi environment (conflicting pins), A100-only JAX LoRA, and a
   flow-matching RL path (RLinf Flow-SDE) that replaces WO-03/04's token-level GRPO.
3. **π0.5 only if step 2 also fails**, with a plan and cost put to the owner first.

### Outcome of the decision path (2026-10-27)

- Step 1, receding horizon (execute 4 of 8): worse on every rechecked checkpoint — a3 step 3000
  22/4/4% → 6/4/4%, a3 step 6000 18/4/8% → 0/4/0%, a2 step 2000 20/6/4% → 2/2/2% ($0.31).
  Re-querying mid-chunk breaks the action continuity the chunked policy was trained on.
- Step 2, 25 demos/task, 8000 steps (A100, lr 5e-4): best 18/16/6% (step 4000). More data helped
  Push (≤ 8% → 16%) but no task reached 20%.
- Step 3, π0.5: next per the owner's path; plan + cost go to the owner before any spend.

## Owner decision (2026-10-27): keep OpenVLA-OFT, amend the RL start rule

- **Do not switch to π0.5.**
- **CLAUDE.md rule 3 amended:** RL may start from a checkpoint with ≥ 10% success on seen positions
  on at least 2 of 3 tasks; the 20–60% band stays the target. Basis: SimpleVLA-RL (arXiv 2509.09674) failure-mode analysis: RL lifted a 100-trajectory SFT base from 7.3% to 25.4% (numbers as cited by the owner; they match a secondary summary of the paper, the primary table not re-checked) — RL
  still adds signal from a low but nonzero base, while a 0% base gets no RL gain.
- **Selected SFT checkpoint:** attempt 4 (25 demos/task, lr 5e-4, A100), step 4000 — sweep (seen,
  50 episodes/task, seed 0) 18 / 16 / 6%, two tasks ≥ 10%. Hub:
  `KudratSingh22/vla-rl-so101-sft` `wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000`.
- **WO-02 closes** with the full protocol table for that checkpoint, labelled *below the original
  target band* in results.md.
- **What would reopen this:** WO-04's RL fails the kill switch from this base (< 3-point gain at the
  25% check) — then revisit the base model (π0.5) with the owner.
