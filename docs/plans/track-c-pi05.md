# Plan: Track C option (a), a pi0.5 action-expert fine-tune on real SO-101 demos

Status: **plan and probe config only. No paid run is approved, and nothing here has been run on a
GPU.** Source: review section 8, Track C (owner, 2026-11-08). Track C is a real-data baseline,
not the project's few-shot thesis. At 25-50 demos it is above the rule-3 limit of 10 demos per
task, so every number it produces is labelled "real-data baseline, N demos, not few-shot" (owner
decision 4 in the review). Every paid step below needs the owner's explicit go (CLAUDE.md rule 1).

## 1. Verdict

- **The pinned LeRobot 0.4.4 includes pi0.5** (`lerobot/policies/pi05`, `--policy.type=pi05`).
  It has the switch the paper's winning recipe uses (`--policy.train_expert_only=true`) and LoRA
  through `--peft.*`, which is the recipe the paper found fails (4/110).
- **pi0.5 cannot run in this project's frozen environment** (verified). The 0.4.4 pi0.5 model
  needs Hugging Face's patched transformers branch `fix/lerobot_openpi`, which reports version
  4.53.3. The project pins stock transformers 4.57.6, which OFT and SmolVLA share. In our `.venv`,
  `from transformers.models.siglip import check` raises `ImportError`, so `PI05Pytorch.__init__`
  raises "An incorrect transformer version is used". The stock Gemma also lacks the action
  expert's adaRMS conditioning layers (section 3). The PyPI wheel of 0.4.4 has no `pi` extra,
  because PyPI does not allow git dependencies. Track C therefore needs a **separate environment**
  (section 5), and that is an owner decision under rule 2.
- **The data format already matches.** `scripts/wo06_teleop_record.py` writes LeRobotDataset v3.0
  with `observation.images.front` (224 x 224 RGB), `observation.state` and `action` (float32[6]
  canonical radians) at 15 fps. LeRobot 0.4.4 writes the q01/q99 stats that pi0.5's QUANTILES
  normalization needs (verified on `data/demos/vla-rl-so101-demos-tiny/meta/stats.json`). If the
  model trains on these files, no unit conversion is needed anywhere.
- **PR #23's client can serve pi0.5 without code changes.** The server needs a `Pi05Served`
  adapter and must run in the pi0.5 environment. Run the first eval session on the GPU server
  through the existing SSH tunnel. Running on the Mac is possible in principle but unmeasured
  (section 6).
- **Cost**, using the paper's 10.6 A100-hours per run (I checked this figure against the paper):
  probe `estimated: A100 80GB x 0.9 h = $1.61`; one full training run
  `estimated: A100 80GB x 11.1 h = $19.87`, which covers training plus 0.5 h of setup and excludes
  evaluation.

## 2. Evidence base: Cowling 2026 (arXiv 2610.37170)

The owner and the review supplied these figures: 28/40 at 25 real SO-101 demos with the action
expert fine-tuned, LoRA at 4/110, and 10.6 A100-hours per run (about $19 at $1.79/h secure, before
setup and evaluation). On 2026-11-08 I read the paper's HTML (arxiv.org/html/2610.37170) through a
summarising web fetch, not line by line. The following matched the paper:

| Item | Paper | Matches owner/review figure |
|---|---|---|
| pi0.5 expert-only, binary success | 16/40 at 12 demos, **28/40 at 25**, 27/40 at 50, 30/40 at 100, 29/40 at 200 (130/200 total) | yes |
| pi0.5 LoRA (r32 on the expert's q/v projections plus 4 projections, 2.7M weights) | 1/30, 2/30, 1/30, 0/10, 0/10 = **4/110** | yes |
| Training time per run (Table 19, median of completed runs) | **10.6 h** on an "A100 large" (Hugging Face Jobs) for both recipes | yes |
| Recipe | `lerobot/pi05_base`; VLM frozen, whole action expert trained (693,422,112 weights, 16.7%); 6,000 steps x batch 32; lr 2.5e-5, 1,000 warm-up steps, cosine decay to 2.5e-6 at step 6,000; AdamW wd 0.01, clip 1.0; fp32 training with gradient checkpointing; ~6.3 s/step | n/a |
| Inputs | front + wrist USB cams, 640 x 480 at 30 fps, fed as 224 x 224 in 2 of 3 camera slots; 6 joint targets in LeRobot units (degrees, gripper 0-100); 30 Hz control (27.3 Hz achieved); 50-action chunks, all 50 executed | n/a |
| Inference | RTX 5070 Ti 16 GB workstation, bf16, 10 flow steps; median 0.17 s arm pause per chunk | n/a |
| Software | LeRobot **0.6.2 dev** (commit 8352ca62) plus the peft extra, **not 0.4.4** | n/a |
| Scoring | binary (green ball in the basket within 30 s), 10 fixed placements, dim light | n/a |
| SmolVLA in the same paper (VLM frozen, binary) | 2/40, **4/40 at 25**, 21/40 at 50, 14/40, 15/40 | n/a |

How these figures transfer to us: the paper's task is the author's own (ball into basket), its
software is a later LeRobot, and it uses two cameras at 30 Hz. We would use one front camera at
15 Hz (the wrist camera is `enabled: false` in `configs/real.yaml`) and the WO-07 cube-in-bowl
protocol. Treat 28/40 as a reference, not a forecast.

**SmolVLA scores.** The SmolVLA paper's 90% (in distribution) and 50% (out of distribution) on
SO-101 pick-place-lego, with ACT at 70/40, are **partial-credit results**. The paper scores
pick-and-place at 0.5 for the grasp plus 0.5 for the place (arxiv.org/html/2506.01844, section
4.1; the rule is stated for the SO-100 tasks and is not restated for the Lego task). They are not
comparable to our binary WO-07 success. Cowling's binary SmolVLA numbers above are much lower.

## 3. Compatibility with the pinned LeRobot 0.4.4 (read from `.venv` code)

| Question | Answer from code | Where |
|---|---|---|
| Is pi0.5 shipped? | yes: `PI05Config` registered as `"pi05"`, `PI05Policy`, `make_pi05_pre_post_processors` | `lerobot/policies/pi05/` |
| Base checkpoint | `lerobot/pi05_base`, 14,467,165,872-byte fp32 `model.safetensors`. **Pin revision `9e55186ad36e66b95cda57bc47818d9e6237ae30`** (2026-01-22): same weights blob and `config.json` as `main`, but its processor JSON uses only steps that 0.4.4 registers. `main` (since 2026-06-03) adds `relative_actions_processor`, which 0.4.4 does not know. | HF API, read 2026-11-08 |
| Does the 0.4.4 config parser accept the base config? | yes, `PreTrainedConfig.from_pretrained` gives a `PI05Config` (chunk 50, fp32) | checked locally |
| Action-expert switch | `train_expert_only: bool = False`. When true, `_set_requires_grad` freezes the whole `paligemma` (SigLIP + Gemma 2B) and the expert plus projections train, which is the paper's "expert-only". `freeze_vision_encoder` freezes only SigLIP. | `configuration_pi05.py:81-82`, `modeling_pi05.py:412-420` |
| LoRA switch (not to be used) | `--peft.method_type=LORA --peft.r=<r>`; default targets are the expert's q/v projections plus action/time projections (`_get_default_peft_targets`), the placement the paper reports at 4/110. `peft` 0.21.2 is installed. | `configs/default.py:73`, `modeling_pi05.py:1274` |
| Trainable size | On a meta device with stock transformers, expert-only gives 576,954,400 trainable of 3,500,289,808. The missing adaRMS layers (37 norms x Linear(1024, 3072), each replacing a 1024-weight RMSNorm) add 116,467,712, for **693,422,112**, exactly the paper's count. With the fork, `lerobot-train`'s logged learnable-parameter count must equal this number, which makes a one-line check that the fork is active. | counted locally |
| Action chunk | `chunk_size = 50`, `n_action_steps = 50`, `max_action_dim = 32` (our 6 dims are zero-padded, and losses and outputs are truncated to the dataset's action dim) | `configuration_pi05.py:37-42` |
| Units | No unit assumption anywhere: state and action are quantile-normalized per dimension (q01/q99 to [-1, 1]), and the state is then tokenized into 256 bins in the prompt. Radians and degrees both work, as long as training data and deployment use the same units. LeRobot's SO follower defaults to `use_degrees=True` with the gripper at 0-100 (`so_follower.py:51,60`). This project records canonical radians (`src/robot/units.py`). | `processor_pi05.py:72-84`, `normalize_processor.py:362` |
| Images | Any `observation.images.*` key from the dataset. With `--policy.type=pi05`, input features come from the dataset, so our single `observation.images.front` maps through and no `rename_map` is needed. Images are float [0, 1] CHW, resized with padding to 224 (a no-op for our 224 crop) and scaled to [-1, 1]. Camera slots the batch does not provide are filled with masked -1 images. | `factory.py:458-472`, `modeling_pi05.py:1131-1195` |
| Normalization stats | dataset `q01`/`q99` are required. LeRobot 0.4.4 writes them by default (`DEFAULT_QUANTILES`). On a pretrained path, `lerobot-train` overrides the base checkpoint's stats with the dataset's. | `compute_stats.py:20`, `lerobot_train.py:254-280` |
| Tokenizer | `google/paligemma-3b-pt-224`. **Access is gated (manual approval)** on the Hub: the owner's HF account must accept it before the pod can download it. | `processor_pi05.py:145`, HF API |
| Video decode (rule 6) | `DatasetConfig.video_backend` falls back to PyAV silently when torchcodec is missing, so the command passes `--dataset.video_backend=torchcodec` explicitly | `configs/default.py:36`, `video_utils.py:117` |
| Resume | `--resume=true --config_path=<ckpt>/pretrained_model/train_config.json` | `configs/train.py:45-90` |

## 4. Probe config (exact flags; parsed with 0.4.4's `draccus` into `TrainPipelineConfig`)

Every flag below parses into `TrainPipelineConfig` with the pinned 0.4.4 and gives `PI05Config`
with the values shown (checked locally; no training run). The probe can run **before any real
demo exists**, on the sim demo set `KudratSingh22/vla-rl-so101-demos` (same schema: one 224 px
front image and 6-D radian state and action). It measures only what a paid run needs to know:
whether the environment works, VRAM use, seconds per step and inference latency.

Pod setup (Linux, in its own venv, not the project's lock; see decision D1):

```bash
apt-get update && apt-get install -y ffmpeg   # torchcodec needs FFmpeg 4-7 shared libs
uv venv /workspace/pi05-env --python 3.11 && source /workspace/pi05-env/bin/activate
uv pip install "torch==2.6.0" "torchcodec==0.2.1" \
  "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@8fff0fde7c79f23a93d845d1a50e985de01f8b8a" \
  "transformers @ git+https://github.com/huggingface/transformers.git@dcddb970176382c0fcf4521b0c0e6fc15894dfe0" \
  "grpcio==1.73.1"                      # tag v0.4.4; fix/lerobot_openpi head (4.53.3); serve.py needs grpc
python -c "from transformers.models.siglip import check; assert check.check_whether_transformers_replace_is_installed_correctly()"
hf auth login                           # token whose account accepted google/paligemma-3b-pt-224
hf download lerobot/pi05_base --revision 9e55186ad36e66b95cda57bc47818d9e6237ae30 \
  --local-dir /workspace/models/pi05_base@9e55186ad36e
```

`torchcodec==0.2.1` is the project pin (`docs/ENVIRONMENT.md`). It is listed explicitly because
torchcodec wheels do not declare their torch version: left unpinned, the resolver picks
torchcodec 0.10.0 (resolved with `uv pip compile` for linux x86_64, 2026-11-08), which is built
for a newer torch than 2.6.0 and fails to load.

Probe (200 steps, nothing saved):

```bash
lerobot-train \
  --dataset.repo_id=KudratSingh22/vla-rl-so101-demos \
  --dataset.video_backend=torchcodec \
  --policy.type=pi05 \
  --policy.pretrained_path=/workspace/models/pi05_base@9e55186ad36e \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true \
  --policy.dtype=float32 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.optimizer_lr=2.5e-5 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=6000 --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --num_workers=8 --seed=1000 --wandb.enable=false \
  --steps=200 --save_checkpoint=false --log_freq=10 \
  --output_dir=checkpoints/track-c/pi05-expert-probe --job_name=pi05_expert_probe
```

The probe passes when: the logged learnable-parameter count is 693,422,112; there is no OOM at
batch 32; the loss falls over 200 steps; and the median s/step is recorded (the paper reports
~6.3 s with two cameras, so one camera may be faster). The probe then times
`predict_action_chunk` on the base weights in bf16 on the same GPU (10 flow steps, 20 calls,
median ms). That timing decides `deploy.request_timeout_ms` (section 6).

Full run (25 real PickPlace demos from `wo06_teleop_record.py`, not yet recorded). It uses the
probe's flags except for the dataset, the step and log counts, the output directory, and
checkpoint saving, which the probe turns off and the full run turns on:

```bash
lerobot-train \
  --dataset.repo_id=KudratSingh22/vla-rl-so101-real-pickplace \
  --dataset.video_backend=torchcodec \
  --policy.type=pi05 \
  --policy.pretrained_path=/workspace/models/pi05_base@9e55186ad36e \
  --policy.train_expert_only=true --policy.freeze_vision_encoder=true \
  --policy.dtype=float32 --policy.gradient_checkpointing=true --policy.compile_model=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --policy.optimizer_lr=2.5e-5 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=6000 --policy.scheduler_decay_lr=2.5e-6 \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --num_workers=8 --seed=1000 --wandb.enable=false \
  --steps=6000 --save_checkpoint=true --save_freq=3000 --log_freq=100 \
  --output_dir=/workspace/checkpoints/track-c/pi05-expert-25demo --job_name=pi05_expert_25demo
```

`--output_dir` is absolute and sits on the `/workspace` volume, so checkpoints survive a pod stop
and the paths below do not depend on the shell's working directory. It must not exist before the
run: without `--resume`, 0.4.4 raises `FileExistsError`. With `steps=6000` and `save_freq=3000`,
0.4.4 saves after step 3,000 and after the last step (`step % save_freq == 0 or step == steps`) to
`<output_dir>/checkpoints/<step, 6 digits>/`, and points the `checkpoints/last` symlink at the
newest one:

```text
/workspace/checkpoints/track-c/pi05-expert-25demo/checkpoints/
  003000/{pretrained_model,training_state}/
  006000/{pretrained_model,training_state}/
  last -> 006000
```

Parsing these flags with the pinned 0.4.4 (`TrainPipelineConfig.validate()`, locally) gives
exactly those two checkpoint directories. Each `pretrained_model/` holds the full fp32
`model.safetensors` (about 14.5 GB) plus `config.json`, `train_config.json` and the
pre/post-processor files. Each `training_state/` adds the AdamW state for the 693M trained
weights (about 5.5 GB, estimated). That makes about 20 GB per checkpoint and about 40 GB for
both, on top of the 14.5 GB base, so the volume needs at least about 60 GB free.

If the pod is interrupted, resume from the newest checkpoint. 0.4.4 reloads the saved train
config, so no other flags are needed:

```bash
lerobot-train --resume=true \
  --config_path=/workspace/checkpoints/track-c/pi05-expert-25demo/checkpoints/last/pretrained_model/train_config.json
```

Afterwards, upload the step-6,000 `pretrained_model/`, not `training_state/`, to a private Hub
repo (`hf upload` creates the repo as private if it does not exist):

```bash
hf upload KudratSingh22/vla-rl-so101-pi05-expert-25demo \
  /workspace/checkpoints/track-c/pi05-expert-25demo/checkpoints/006000/pretrained_model . \
  --repo-type=model --private
```

These are the paper's settings, kept unchanged so the result can be compared with its 28/40.
6,000 x 32 samples is about 26 passes over ~7,500 frames (25 demos x ~20 s x 15 Hz), consistent
with the paper's fixed step count.

## 5. Environment decision (rule 2)

The pi0.5 environment differs from the frozen lock in transformers (4.53.3 from the fork vs
4.57.6), in lerobot's `pi` extra (scipy < 1.15), and in being installed from git. It cannot be
merged into `uv.lock` without breaking OFT/SmolVLA, so the choices are:

- **D1-a (recommended):** a separate, pinned pod-only environment, using exactly the commands in
  section 4 (all commits pinned), recorded in `docs/ENVIRONMENT.md` as a Track C environment and
  opened as its own small work order. The project lock stays untouched.
- D1-b: a second `uv` project under `tracks/pi05/` with its own lock. This is more reproducible,
  but it is more process than one baseline needs.
- D1-c: wait for a LeRobot release whose pi0.5 runs on stock transformers (the paper used 0.6.2
  dev). That means upgrading the lerobot pin, which is its own WO and touches the robot driver
  that the review verified against 0.4.4.

## 6. Serving through PR #23 (`wo-07-pilot`), OFT vs pi0.5

What stays the same: the wire format (`serve.pack`/`unpack`, uint8 (224, 224, 3) image plus
float32 (6,) radian state in, float32 (C, 6) radian actions out), `check_server`, every safety
path (`So101Driver.step`, deadman, stale-chunk drop), trial records and the protocol hash.
**`src/robot/client.py` needs no code change.**

What must change (server side, about 60 lines plus a smoke test, not written in this PR):

1. `serve.py`: add a `Pi05Served(ServedPolicy)`.
   - Load `PI05Policy.from_pretrained(ckpt)` (with `dtype="bfloat16"` for inference) and the
     checkpoint's saved pre- and post-processors (`make_pre_post_processors(cfg,
     pretrained_path=ckpt)`).
   - `act()` builds `{"observation.images.front": image, "observation.state": state}`, runs
     `prepare_observation_for_inference` (uint8 HWC to float [0, 1] CHW, batch dimension) and
     adds `task=instruction`. It then runs the preprocessor, `predict_action_chunk`, the
     postprocessor, and returns `actions[0, :chunk_len]`.
   - Call `policy.reset()` per trial (nothing is queued, because we use `predict_action_chunk`,
     not `select_action`).
   - Add `--policy-kind {oft,pi05}` to `main()`. The `policy_id` is the pi0.5 checkpoint id.
2. Run the server in the pi0.5 environment (section 4). Its top-level imports (`grpc`, `numpy`,
   `src.config`, `src.eval.protocol`, `src.robot.units`, `src.runs`) are pure Python and work
   there. Only `load_oft` imports the OFT stack, and it does so lazily.
3. `configs/real.yaml` `deploy`:
   - The server's `chunk_len` (the slice of the 50-action chunk it returns) must equal
     `deploy.chunk_len`, which the client checks in `Info`.
   - Keep 8 (533 ms open loop at 15 Hz, within `max_chunk_age_ms: 800`).
   - Executing more of the chunk, like the paper's 50 (3.3 s at our 15 Hz), would mean raising
     `max_chunk_age_ms` above the chunk's duration plus the query time. That is a safety-config
     change for the owner, and it is not proposed here.
   - `request_timeout_ms: 400` must exceed pi0.5's inference time plus the tunnel RTT. The
     paper's 0.17 s pause on a 16 GB 5070 Ti suggests it fits on an A100/4090. Unverified until
     the probe's timing.
4. Units: none, if the policy trains on `wo06_teleop_record.py` data (canonical radians in and
   out). If it ever trains on stock `lerobot-record` data (degrees, gripper 0-100), the server
   must convert both ways with `src/robot/units.UnitMap`, because the client and safety filter
   speak radians only.

Where inference runs:

- **GPU server (default for the first session).** In bf16 the weights are about 7.2 GB (3.62B
  parameters; 14.5 GB fp32 file / 4 bytes), so a 24 GB RTX 4090 at $0.89/h secure ($0.34
  community) is enough. Use the same SSH-tunnel path as OFT.
- **Mac (possible, unmeasured).** The project `.venv` cannot load pi0.5 (verified, section 1), so
  it needs the pi0.5 environment on macOS too. LeRobot keeps bf16 on MPS (`get_safe_dtype` only
  maps float64 to float32 there). The 7.2 GB bf16 weights fit in unified memory, but the latency
  of 10 flow-matching steps on an M3 Pro is unknown. Measuring it costs $0: time
  `predict_action_chunk` locally on the trained checkpoint. If it beats `request_timeout_ms`
  minus a margin, run `serve.py --device mps` on the Mac bound to 127.0.0.1. The client code is
  unchanged and no pod is needed for eval sessions.

## 7. Cost lines (provisional; the probe reprices the full run)

- Probe: 0.5 h setup (env plus 14.5 GB weights) + 200 x 6.3 s (0.35 h) + 5 min latency check:
  `estimated: A100 80GB (secure $1.79/h) x 0.9 h = $1.61`.
- Full 25-demo training run: 10.6 h (paper, Table 19) + 0.5 h setup:
  `estimated: A100 80GB (secure $1.79/h) x 11.1 h = $19.87`. If the probe shows a different
  s/step, the new figure is 6,000 x s/step / 3600 + 0.5 h. A community A100 ($1.19-1.39/h) is
  usable because `--resume` exists in 0.4.4; with a 3,000-step save cadence it loses at most
  about 5.3 h.
- Real eval serving (3 cells x 50 trials, about 2 h of tunnel time):
  `estimated: RTX 4090 (secure $0.89/h) x 2 h = $1.78`. This is $0 if the Mac serves it.

Rule-1 run records apply (`runs/<id>/record.json`). `lerobot-train` does not write them, so wrap
the command or write the record by hand.

## 8. Decisions for the owner

1. D1: the environment choice (section 5). The recommendation is D1-a.
2. Accept the `google/paligemma-3b-pt-224` license on the owner's HF account (gated, manual).
3. Go/no-go on the $1.61 probe. It can run now, on the sim demos.
4. After 25 real PickPlace demos exist (Track A): go/no-go on the ~$20 training run.
5. Optional: enable the wrist camera for the Track C recordings (the paper used front + wrist).
   That would change `configs/real.yaml` and the recorder schema, and it would break schema
   parity with the sim demos.

## 9. Not verified

- Anything on a GPU: VRAM, s/step on our one-camera data, bf16 inference latency on A100/4090/MPS.
- That LeRobot 0.4.4's pi0.5 plus the `fix/lerobot_openpi` fork reproduces the paper's result. The
  paper used LeRobot 0.6.2 dev.
- That the 0.4.4 pre/post-processor JSON at revision `9e55186ad36e` loads end to end. All its
  step names are registered in 0.4.4, but the tokenizer download needs gated access, so I did not
  load it.
- The paper figures were read through a summarising fetch of the HTML. Spot-check Tables 3 and
  19 before citing them in `docs/results.md`.
