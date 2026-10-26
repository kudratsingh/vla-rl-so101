# ADR-0001: Base VLA model

Status: proposed (decide at end of WO-02)
Date: 2026-10-26
Work order: WO-02

## Decision

Default to OpenVLA-OFT with LoRA, under the output head used by the vendored RL code
(SimpleVLA-RL's cross-entropy head), re-doing few-shot SFT under that head so RL and SFT share one
checkpoint format. Fall back to π0.5 (openpi LoRA, RLinf πRL path) if OpenVLA-OFT few-shot SFT is
below 20% on seen positions after a reasonable tuning pass (≤3 SFT runs).

## Why

- OpenVLA-OFT is the model the open GRPO recipe targets; its action chunking gives ~108 Hz
  inference on LIBERO, which is what makes rollouts affordable on one GPU.
- π0.5 has the stronger prior (few-shot 77% on LIBERO before RL in πRL) but needs >22.5 GB for
  LoRA and the flow-matching RL path (Flow-SDE/Flow-Noise) is more complex to run single-GPU.
- SmolVLA is used only to smoke-test the data path in WO-01; its ceiling is too low for the
  headline result.

## Alternatives considered

- GR00T N1.7 — 40 GB+ fine-tune, no open RL recipe targets it.
- Train an ACT/Diffusion Policy from scratch — not post-training of a pre-trained model; out of scope.

## What would reopen this

OpenVLA-OFT SFT <20% on seen after 3 runs; or VRAM for OFT + sim + rollouts exceeding 80 GB at
≥8 parallel envs.
