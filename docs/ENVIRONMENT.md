# Environment spec

All versions below are pinned. Changing any of them is a work order of its own.

## Pin table (fill exact values in WO-01; the agent records the resolved commit/version here)

| Component | Pin | Why |
|---|---|---|
| Python | 3.11.x | ManiSkill3 + LeRobot + verl all support it; 3.12/3.14 untested with SAPIEN wheels |
| CUDA image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | Matches current torch wheels |
| PyTorch | 2.5.x + cu124 | OpenVLA-OFT and verl tested here |
| ManiSkill3 | `mani_skill==3.0.0bX` (record exact) | GPU-parallel sim + render; SO-101 assets |
| SAPIEN | as required by ManiSkill3 pin | — |
| LeRobot | git commit (record SHA) | APIs move between releases; sim2real repos pin commits |
| lerobot-sim2real / squint | git commit (record SHA) | Source of SO-101 ManiSkill assets and system-ID tools; vendored into `src/env/` |
| OpenVLA-OFT | git commit (record SHA) + HF checkpoint id | Base model |
| openpi (fallback) | git commit (record SHA) | π0.5 LoRA path |
| SimpleVLA-RL / verl | git commit (record SHA) | GRPO recipe; vendored into `src/rl/` |
| RLinf (optional) | git commit (record SHA) | Only if π0.5 path is chosen |
| transformers / peft | match OpenVLA-OFT requirements | LoRA |
| torchcodec | latest compatible with torch pin | Video decode (never PyAV) |
| feetech-servo-sdk (via LeRobot) | from LeRobot pin | SO-101 bus |
| grpcio | 1.6x | Policy serving |
| uv | latest | Lock file committed |

Record resolved pins in `uv.lock` and in this table.

## GPU tiers (RunPod, Oct 2026 list prices)

| Tier | Card | $/h | Used for |
|---|---|---|---|
| dev | RTX 4090 24 GB | 0.89 | WO-01 sim/oracle/demos, SmolVLA smoke, OpenVLA-OFT LoRA SFT (fits with 8-bit base + LoRA; verify), eval |
| rl | A100 80 GB | 1.79 | WO-04/05 GRPO (VLA + sim + rollouts on one card) |
| fast | H100 PCIe 80 GB | 2.89 | Only if A100 throughput makes a run >12 h |

Presets map to tiers: `tiny` (CPU, any machine, <2 min), `sim` (dev), `rl` (rl), `real` (dev or Mac).

## Install — GPU box (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> && cd vla-rl-so101
uv sync --extra gpu            # torch cu124, mani_skill, lerobot, peft, torchcodec, verl deps
uv run python -c "import mani_skill, torch; print(torch.cuda.is_available())"
uv run scripts/wo01_build_env.py --preset tiny   # CPU smoke must pass before any GPU work
```

Vulkan is required by SAPIEN for rendering; RunPod's CUDA images ship it. If `vulkaninfo` fails,
install `libvulkan1 mesa-vulkan-drivers` and set `VK_ICD_FILENAMES` to the NVIDIA ICD.

## Install — Mac (control, calibration, deploy)

```bash
brew install uv
uv sync --extra robot          # lerobot robot extras, feetech sdk, grpcio, opencv; no torch-cuda
uv run lerobot-find-port       # find the Waveshare board
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX
```

Policy inference for OpenVLA-OFT runs on the GPU box and is served to the Mac over gRPC;
SmolVLA can run on the Mac CPU/MPS for smoke tests only.

## Hardware

- SO-101 follower (leader optional). Feetech STS3215 servos, Waveshare bus board, 5 V 4 A supply.
- 2 USB webcams at 640×480 @ 30 fps: one fixed overhead/front (matches the sim camera pose),
  one optional wrist cam. Record intrinsics in `configs/real.yaml`.
- Table clamps; printed placement grid (A3) for randomized object positions; cubes 3 cm, bowl.

## Data locations

| What | Where |
|---|---|
| Demos / real datasets | HF Hub `KudratSingh22/vla-rl-so101-*` (LeRobotDataset v2 format) |
| Checkpoints | HF Hub (private) or RunPod network volume; never git |
| Run records | `runs/<id>/record.json` committed; metrics/logs ignored |

## Known time sinks (avoid)

- PyAV video decode → 3–30% GPU utilization. Use torchcodec.
- Unpinned LeRobot → import-path breaks. Pin the commit.
- SimpleVLA-RL changed OpenVLA-OFT's output head; official OFT checkpoints need re-SFT under its
  head, or use the vendored head consistently from WO-02 onward. Decide once in ADR-0001.
- VLA + sim + replay on one card: budget VRAM explicitly in `configs/rl.yaml`
  (base model bf16 + LoRA + activations + N parallel envs with 128×128 RGB).
- Isaac Sim is out of scope: RTX/Vulkan setup on rented GPUs is the largest setup sink.
