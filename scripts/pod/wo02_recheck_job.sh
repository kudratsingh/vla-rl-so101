#!/usr/bin/env bash
# WO-02 eval-only recheck on one RunPod GPU pod (owner, 2026-10-27: approved for $0.5): re-score
# existing SFT checkpoints on the seen cell with a receding horizon (execute EXEC_STEPS of each
# 8-action chunk, then re-query), upload, self-delete. No training, so no probe/approval gate:
# the owner approved the dollar amount directly. Launched over SSH from the repo root on the pod:
#   CKPTS="<hub sub-path> ..." EXEC_STEPS=4 nohup bash scripts/pod/wo02_recheck_job.sh > /workspace/job.log 2>&1 &
set -uo pipefail
source /root/runpod.env
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN="$(cat /root/.hf_token)"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-sft}"
CFG="${SFT_CFG:-configs/sft.yaml}"
EXEC_STEPS="${EXEC_STEPS:-4}"
STAGE_FILE=/workspace/stage
# short job: keep only a 3-minute margin before the pod's hard cap for the (small) upload
DEADLINE=$(( $(cat /root/pod_start_epoch) + SPEND_CAP_SECONDS - ${SOFT_MARGIN_S:-180} ))

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }
left() { echo $(( DEADLINE - $(date +%s) )); }

finish() {
  local rc=$?
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null)"
  uv run python - "rc=$rc" <<'EOF' || echo "[job] upload failed"
import os, sys
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
if os.path.isdir("runs"):
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"wo02-recheck/{pod}/runs",
                      allow_patterns=["*/record.json", "*/eval.json", "*/eval.md", "*/metrics.jsonl"],
                      commit_message=f"WO-02 recheck pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"wo02-recheck/{pod}/job.log",
                 repo_id=repo, commit_message=f"WO-02 recheck pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/wo02-recheck/{pod}")
EOF
  echo "[job] terminating pod $RUNPOD_POD_ID"
  curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    || curl -fsS https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
}
trap finish EXIT

stage setup
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 >/dev/null || exit 10
mkdir -p /usr/share/vulkan/icd.d  # EGL ICD: see scripts/pod/wo02_job.sh / docs/ENVIRONMENT.md
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11

stage gpu-render-check
uv run python - <<'EOF' || exit 12
from src.config import load_config
from src.env import make_env, policy_obs
env = make_env(load_config("sim"), "PickPlace", num_envs=2)
obs, _ = env.reset(seed=[0, 1])
img = policy_obs(obs)["image"]
assert tuple(img.shape) == (2, 224, 224, 3) and img.float().std() > 1
print("[job] GPU sim + render ok")
EOF

for CK in ${CKPTS}; do
  stage "eval ${CK##*/runs/}"
  uv run hf download "$HUB_REPO" --include "$CK/*" --local-dir /workspace/ckpts >/dev/null || exit 20
  s=$(left); [ "$s" -le 60 ] && { echo "[job] soft deadline before $CK"; exit 124; }
  timeout "$s" uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "/workspace/ckpts/$CK" \
    --cells seen --seeds 0 --episodes 50 --num-envs 50 --sim-device cuda --exec-steps "$EXEC_STEPS" \
    || exit 30
  echo "[job] $CK -> $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)"
done
stage done
