# Environment spec

All versions below are pinned. Changing any of them is a work order of its own.

## Pin table (resolved in WO-01 from `uv.lock`; upstream SHAs recorded here)

| Component | Pin | Why |
|---|---|---|
| Python | 3.11.x | ManiSkill3 + LeRobot support it; LeRobot 0.4.4 is the LAST release supporting 3.11 (0.5.0+ requires >=3.12) |
| CUDA image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | Matches current torch wheels |
| PyTorch | 2.6.0 (+cu124 on Linux from the PyTorch cu124 index; PyPI CPU/MPS build on macOS); torchvision 0.21.0 | Every LeRobot release since 0.3 requires torchcodec>=0.2.1, and torchcodec 0.2.x pairs with torch 2.6, so the kit's 2.5.x pin was not satisfiable |
| ManiSkill3 | `mani-skill==3.0.1` (stable, 2026-04-21) | Replaces the kit's 3.0.0bX beta. GPU-parallel sim + render. mplib (motion planner) is Linux-only, so the oracle uses IK |
| SAPIEN | 3.0.3 | Required by ManiSkill3 pin; first macOS wheels in 3.0.2 |
| LeRobot | 0.4.4, `8fff0fde7c79f23a93d845d1a50e985de01f8b8a` (tag v0.4.4, 2026-02-27) | APIs move between releases; dataset format v3.0 (its CODEBASE_VERSION) |
| SO-101 robot model | TheRobotStudio/SO-ARM100 (Apache-2.0), `Simulation/SO101/so101_new_calib.urdf` @ `a758567c3978dfeefe282ede0500085a48fe8f78` (edits listed in `src/env/assets/so101/README.md`) | Vendored into `src/env/assets/so101/` |
| lerobot-sim2real / squint | not vendored | lerobot-sim2real has NO license on GitHub; "squint" could not be located on GitHub |
| OpenVLA-OFT | `e4287e94541f459edc4feabc4e181f537cd569a8` (moojink/openvla-oft, MIT, 2025-09-09); model code vendored from SimpleVLA-RL `7c51662` into `src/policy/oft/vendor/`; checkpoint `openvla/openvla-7b` @ `47a0ec7fc4ec123775a391911046cf33cf9ed83f` | Base model (ADR-0001). Runs on stock transformers 4.57.6 + a bidirectional-mask hook (bit-identical to the moojink transformers fork, which is NOT used); no flash-attn, bitsandbytes or TensorFlow |
| openpi (fallback) | `215abfb217dbac7d5f1273282331b9b1866c0479` (Physical-Intelligence/openpi, 2026-08-24); recorded in WO-01, installed in WO-02/03 | π0.5 LoRA path |
| SimpleVLA-RL / verl | `7c51662df27b586f9e8a1ab35fcf849f2b8852f9` (PRIME-RL/SimpleVLA-RL, MIT, 2026-01-06); recorded in WO-01, installed in WO-02/03 | GRPO recipe; vendored into `src/rl/` |
| RLinf (optional) | `fab7f407fbc611085eb163962e3c89c0ab424c1a` (RLinf/RLinf, 2026-10-26); recorded in WO-01, installed in WO-02/03 | Only if π0.5 path is chosen |
| transformers / peft / timm | transformers 4.57.6 / peft 0.21.2 / timm 0.9.16 (`gpu` extra; transformers shared with `smolvla`) | LoRA; timm 0.9.x is required by the Prismatic vision code |
| huggingface-hub / datasets | 0.35.3 / 4.8.5 | From `uv.lock` |
| torchcodec | 0.2.1 | Video decode (never PyAV). Needs FFmpeg 4-7 shared libs. No CUDA torchcodec wheel exists for 0.2.x on the cu124 index; CPU decode is used |
| feetech-servo-sdk (via LeRobot) | from LeRobot pin | SO-101 bus |
| grpcio | 1.85.0 | Policy serving |
| uv | 0.13.5 | Lock file committed |

Record resolved pins in `uv.lock` and in this table.

## GPU tiers (RunPod, Oct 2026 list prices)

| Tier | Card | $/h | Used for |
|---|---|---|---|
| dev | RTX 4090 24 GB | 0.89 | WO-01 sim/oracle/demos, SmolVLA smoke, OpenVLA-OFT LoRA SFT in bf16 with gradient checkpointing (~19–21 GB est.; no 8-bit base), eval |
| rl | A100 80 GB | 1.79 | WO-04/05 GRPO (VLA + sim + rollouts on one card) |
| fast | H100 PCIe 80 GB | 2.89 | Only if A100 throughput makes a run >12 h |

Presets map to tiers: `tiny` (CPU, any machine, <2 min), `sim` (dev), `rl` (rl), `real` (dev or Mac).

## Install — GPU box (Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get install -y ffmpeg      # torchcodec (Ubuntu 22.04 ships FFmpeg 4.4)
git clone <repo> && cd vla-rl-so101
uv sync --extra gpu            # torch cu124, mani_skill, lerobot, peft, torchcodec, verl deps
uv run python -c "import mani_skill, torch; print(torch.cuda.is_available())"
uv run scripts/wo01_build_env.py --preset tiny   # CPU smoke must pass before any GPU work
```

Vulkan is required by SAPIEN for rendering. On RunPod (driver 595.91.07, `nvidia/cuda:12.4.1` image,
pod env `NVIDIA_DRIVER_CAPABILITIES=all`) the runtime mounts `/etc/vulkan/icd.d/nvidia_icd.json`
pointing at `libGLX_nvidia.so.0`, which cannot create an instance headless
(`ERROR_INCOMPATIBLE_DRIVER`). Point the ICD at NVIDIA's EGL library instead, at the path SAPIEN
checks: `/usr/share/vulkan/icd.d/nvidia_icd.json` with `"library_path": "libEGL_nvidia.so.0"`
(`scripts/pod/wo02_job.sh` does this). Verified: GPU sim (`physx_cuda`) and CPU sim both render.

## Install — Mac (control, calibration, deploy)

```bash
brew install uv molten-vk ffmpeg@7   # ffmpeg@7 is keg-only; FFmpeg 9 is too new for torchcodec 0.2.1
source scripts/mac_env.sh      # VK_ICD_FILENAMES (MoltenVK, so SAPIEN can render) + DYLD_FALLBACK_LIBRARY_PATH (ffmpeg@7)
uv sync --extra robot          # lerobot robot extras, feetech sdk, grpcio, opencv; no torch-cuda
uv run lerobot-find-port       # find the Waveshare board
uv run lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX
```

`make` targets export these env vars automatically. ManiSkill CPU sim + rendering works on
Apple Silicon this way (used for the `tiny` preset and oracle runs).

Policy inference for OpenVLA-OFT runs on the GPU box and is served to the Mac over gRPC;
SmolVLA can run on the Mac CPU/MPS for smoke tests only.

## Install — development (all extras, any machine)

`make setup` runs the full sync from the frozen lock; `make smoke` must pass after it.

```bash
uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla   # = make setup
make check && make smoke   # ruff, black, mypy --strict on src/; CPU smoke tests (tiny preset)
```

On a Mac, `make` exports the MoltenVK / FFmpeg 7 variables above. CI (`.github/workflows/ci.yml`)
syncs `dev` + `gpu`.

## Hardware

- SO-101 follower (leader optional). Feetech STS3215 servos, Waveshare bus board, 5 V 4 A supply.
- 2 USB webcams at 640×480 @ 30 fps: one fixed overhead/front (matches the sim camera pose),
  one optional wrist cam. Record intrinsics in `configs/real.yaml`.
- Table clamps; printed placement grid (A3) for randomized object positions; cubes 3 cm, bowl.

## Data locations

| What | Where |
|---|---|
| Demos / real datasets | HF Hub `KudratSingh22/vla-rl-so101-*` (LeRobotDataset v3.0 format, one multi-task dataset) |
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
