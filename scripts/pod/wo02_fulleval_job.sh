#!/usr/bin/env bash
# WO-02 close-out: full protocol eval (3 tasks x 3 cells x 3 seeds x 100 episodes, GPU sim) of one
# SFT checkpoint. setup -> render check -> PROBE (50 episodes, timed, projected h and $) -> owner
# APPROVAL GATE (/workspace/APPROVED) -> full eval -> upload -> self-delete. Launched over SSH:
#   CKPT=<hub sub-path> RATE_USD_H=0.89 nohup bash scripts/pod/wo02_fulleval_job.sh > /workspace/job.log 2>&1 &
set -uo pipefail
source /root/runpod.env
if [ -z "${OMP_NUM_THREADS:-}" ] && read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then
  export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
fi
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN="$(cat /root/.hf_token)"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-sft}"
CFG="${SFT_CFG:-configs/sft.yaml}"
STAGE_FILE=/workspace/stage
DEADLINE=$(( $(cat /root/pod_start_epoch) + SPEND_CAP_SECONDS - ${SOFT_MARGIN_S:-300} ))

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
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"wo02-fulleval/{pod}/runs",
                      allow_patterns=["*/record.json", "*/eval.json", "*/eval.md", "*/metrics.jsonl"],
                      commit_message=f"WO-02 full eval pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"wo02-fulleval/{pod}/job.log",
                 repo_id=repo, commit_message=f"WO-02 full eval pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/wo02-fulleval/{pod}")
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
uv run hf download "$HUB_REPO" --include "$CKPT/*" --local-dir /workspace/ckpts >/dev/null || exit 12
ADAPTER="/workspace/ckpts/$CKPT"

stage probe
t0=$(date +%s)
uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$ADAPTER" --tasks PickPlace \
  --cells seen --seeds 0 --episodes 50 --num-envs 50 --sim-device cuda || exit 20
dt=$(( $(date +%s) - t0 ))
uv run python - "$dt" "${RATE_USD_H:-0}" <<'EOF'
import sys
dt, rate = float(sys.argv[1]), float(sys.argv[2])
h = dt * 2700 / 50 / 3600  # includes one model load per 50 episodes: an upper bound for the full run
print(f"[job] PROBE 50 episodes in {dt:.0f} s -> full eval (2700 episodes) <= {h:.2f} h, <= ${h * rate:.2f} at ${rate}/h")
EOF

stage await-approval
APPROVAL_WAIT_S="${APPROVAL_WAIT_S:-1800}"
t1=$(date +%s)
until [ -f /workspace/APPROVED ]; do
  if [ $(( $(date +%s) - t1 )) -ge "$APPROVAL_WAIT_S" ]; then
    echo "[job] no owner approval within $((APPROVAL_WAIT_S / 60)) min; stopping"; exit 21
  fi
  sleep 15
done
echo "[job] approved: $(cat /workspace/APPROVED)"

stage full-eval
s=$(left); [ "$s" -le 60 ] && { echo "[job] soft deadline before the full eval"; exit 124; }
timeout "$s" uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$ADAPTER" \
  --num-envs 50 --sim-device cuda || exit 30
echo "[job] full eval -> $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)"
stage done
