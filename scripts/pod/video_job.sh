#!/usr/bin/env bash
# Rollout videos on one RunPod GPU pod: for each checkpoint, scripts/record_videos.py (8 greedy
# seen-cell episodes per task, tiled MP4), upload to the Hub, self-delete. Eval only, no training.
# Launched over SSH from the repo root on the pod:
#   CKPTS="sft-a4=KudratSingh22/vla-rl-so101-sft/wo02/.../step_004000 rl-s1-38=KudratSingh22/vla-rl-so101-rl/.../adapter" \
#     nohup bash scripts/pod/video_job.sh > /workspace/job.log 2>&1 &
set -uo pipefail
source /root/runpod.env
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max \
    || { q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); }
  if [ -n "${q:-}" ] && [ "$q" != max ] && [ "$q" -gt 0 ] 2>/dev/null; then
    export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
  fi
fi
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN="$(cat /root/.hf_token)"
export HF_TOKEN RUNPOD_POD_ID
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-rl}"
STAGE_FILE=/workspace/stage
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
if os.path.isdir("runs/videos"):
    api.upload_folder(repo_id=repo, folder_path="runs/videos", path_in_repo=f"videos/{pod}",
                      commit_message=f"rollout videos pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"videos/{pod}/job.log",
                repo_id=repo, commit_message=f"rollout videos pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/videos/{pod}")
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
mkdir -p /usr/share/vulkan/icd.d  # EGL ICD: see docs/ENVIRONMENT.md
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11

for spec in ${CKPTS}; do
  label=${spec%%=*}; hub=${spec#*=}
  stage "video $label"
  s=$(left); [ "$s" -le 60 ] && { echo "[job] soft deadline before $label"; exit 124; }
  timeout "$s" uv run scripts/record_videos.py --preset sim --adapter "$hub" --label "$label" \
    --sim-device cuda || exit 20
done
stage done
