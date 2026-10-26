#!/usr/bin/env bash
# WO-02 on one RunPod GPU pod: setup -> probe -> LoRA SFT (sim.yaml budget) -> checkpoint sweep
# (seen cell) -> full protocol eval of the checkpoint in the 20-60% band -> upload to the Hub ->
# terminate this pod. Launched over SSH from the repo root on the pod:
#   nohup bash scripts/pod/wo02_job.sh > /workspace/job.log 2>&1 &
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS; written by
# the pod's start command) and /root/.hf_token. The pod also has a hard self-delete timer at the cap.
set -uo pipefail
source /root/runpod.env
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN="$(cat /root/.hf_token)"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-sft}"
CFG=configs/sft.yaml
STAGE_FILE=/workspace/stage
# soft deadline 20 min before the hard cap, so results are uploaded before the pod disappears
DEADLINE=$(( $(cat /root/pod_start_epoch) + SPEND_CAP_SECONDS - 1200 ))

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
run_until_deadline() {
  local s; s=$(left)
  if [ "$s" -le 60 ]; then echo "[job] soft deadline reached before: $*"; return 124; fi
  timeout "$s" "$@"
}

upload() {
  uv run python - "$1" <<'EOF' || echo "[job] upload failed"
import os, sys
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
api.create_repo(repo, private=True, exist_ok=True)
if os.path.isdir("runs"):
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"wo02/{pod}/runs",
                      commit_message=f"WO-02 pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"wo02/{pod}/job.log",
                 repo_id=repo, commit_message=f"WO-02 pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/wo02/{pod}")
EOF
}

finish() {
  local rc=$?
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null)"
  upload "final (rc=$rc)"
  echo "[job] terminating pod $RUNPOD_POD_ID"
  curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    || curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY"
}
trap finish EXIT

stage setup
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 git >/dev/null || exit 10
# NVIDIA Vulkan ICD for SAPIEN rendering if the container runtime did not mount one
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  mkdir -p /usr/share/vulkan/icd.d
  echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
    > /usr/share/vulkan/icd.d/nvidia_icd.json
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11
uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name())" || exit 12
uv run hf download KudratSingh22/vla-rl-so101-demos --repo-type dataset \
  --local-dir data/demos/vla-rl-so101-demos >/dev/null || exit 13

stage gpu-sim-check
SIM_DEVICE=cpu; EVAL_ENVS=8
if uv run python - <<'EOF'
import torch
from src.config import load_config
from src.env import make_env, policy_obs
env = make_env(load_config("sim"), "PickPlace", num_envs=4)
obs, _ = env.reset(seed=[0, 1, 2, 3])
for _ in range(5):
    obs, _, _, _, info = env.step(policy_obs(obs)["state"])
img = policy_obs(obs)["image"]
assert tuple(img.shape) == (4, 224, 224, 3) and img.float().std() > 1, img.shape
assert info["success"].shape == (4,)
print("[job] GPU sim + render ok")
EOF
then SIM_DEVICE=cuda; EVAL_ENVS=50; fi
echo "[job] eval sim device: $SIM_DEVICE ($EVAL_ENVS envs)"

stage probe
run_until_deadline uv run scripts/wo02_sft.py --preset sim --config "$CFG" --probe-steps 20 || exit 20

stage sft
run_until_deadline uv run scripts/wo02_sft.py --preset sim --config "$CFG" || exit 30
RUN=$(ls -d runs/*-wo02-sim-* | sort | tail -1)
echo "[job] SFT run: $RUN"
upload "SFT checkpoints"

stage sweep
for CK in "$RUN"/checkpoints/step_*; do
  run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$CK" \
    --cells seen --seeds 0 --episodes 50 --num-envs "$EVAL_ENVS" --sim-device "$SIM_DEVICE" || exit 40
  echo "$CK $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" >> "$RUN/sweep.txt"
done

stage select
# owner's target: 20-60% on seen positions (not max success) -> most tasks in band, then mean nearest 40%
BEST=$(uv run python - "$RUN/sweep.txt" <<'EOF'
import json, sys
best = None
for line in open(sys.argv[1]):
    ck, ev = line.split()
    rows = json.load(open(f"{ev}/eval.json"))["rows"]
    rates = [r["pooled"] for r in rows if r["cell"] == "seen"]
    in_band = sum(0.2 <= x <= 0.6 for x in rates)
    key = (in_band, -abs(sum(rates) / len(rates) - 0.4))
    print(f"[job] {ck}: seen {[round(x, 3) for x in rates]} in-band {in_band}", file=sys.stderr)
    if best is None or key > best[0]:
        best = (key, ck)
print(best[1])
EOF
) || exit 50
echo "$BEST" > "$RUN/selected_checkpoint.txt"
echo "[job] selected: $BEST"

stage full-eval
run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$BEST" \
  --num-envs "$EVAL_ENVS" --sim-device "$SIM_DEVICE" || exit 60

stage done
