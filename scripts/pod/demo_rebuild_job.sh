#!/usr/bin/env bash
# WO-01r demo rebuild on one RunPod GPU pod (RTX 4090 is enough): setup -> GPU-sim oracle check
# (v2 oracle >= 95% seen per task) -> PROBE = render-gap check of the 0.1 demo set (Mac CPU
# renders vs this GPU renderer, pixels + SFT-adapter tokens, ~$0.1) -> owner approval gate ->
# rebuild 25 demos/task with the fixed recorder + v2 oracle on the GPU renderer, tail check, push
# to a NEW private dataset repo -> render-gap of the new set on the same backend (sanity: should
# sit at the codec floor) -> upload runs/ + job log to the Hub -> terminate this pod.
# Launched over SSH from the repo root on the pod:
#   nohup bash scripts/pod/demo_rebuild_job.sh > /workspace/job.log 2>&1 &
# Optional: DEMO_CFG (configs/demos_v2.yaml), HUB_REPO (results repo, KudratSingh22/vla-rl-so101-sft),
#   GAP_ADAPTER (override render_gap.adapter), APPROVAL_WAIT_S (1800), VERIFY_GAP (1).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS,
# POD_HARD_DEADLINE) and /root/pod_start_epoch, both written by scripts/pod/pod_start_cmd.sh (the
# pod's start command, which also arms the hard self-delete timer), and /root/.hf_token. Every
# stage is time-limited and a watchdog stops the job at the soft deadline (scripts/pod/job_lib.sh).
# NOT RUN YET: needs the owner's explicit go (CLAUDE.md rule 1).
set -uo pipefail
# shellcheck source=scripts/pod/job_lib.sh
. "$(dirname "$0")/job_lib.sh"
STAGE_FILE=/workspace/stage
export PATH="$HOME/.local/bin:$PATH"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-sft}"

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }

upload() {
  timeout -k 30 "$UPLOAD_TIMEOUT_S" uv run --no-sync python - "$1" <<'EOF' || echo "[job] upload failed or timed out"
import os, sys
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
api.create_repo(repo, private=True, exist_ok=True)
if os.path.isdir("runs"):
    # record.json, metrics.jsonl, render_gap.json; not the codec-floor video scratch datasets
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"wo01r/{pod}/runs",
                      ignore_patterns=["**/codec_floor/**"],
                      commit_message=f"WO-01r pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"wo01r/{pod}/job.log",
                 repo_id=repo, commit_message=f"WO-01r pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/wo01r/{pod}")
EOF
}

# installed before anything can fail: every exit uploads (time-limited), then self-deletes
finish() {
  local rc=$?
  stop_watchdog
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null)"
  upload "final (rc=$rc)"
  delete_pod
}
trap finish EXIT
pod_env || exit 9
start_watchdog

if [ -z "${OMP_NUM_THREADS:-}" ]; then
  read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max \
    || { q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); }
  if [ -n "${q:-}" ] && [ "$q" != max ] && [ "$q" -gt 0 ] 2>/dev/null; then
    export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
  fi
fi
HF_TOKEN="$(cat /root/.hf_token)"
export HF_TOKEN
CFG="${DEMO_CFG:-configs/demos_v2.yaml}"
GAP_ARGS=()
[ -n "${GAP_ADAPTER:-}" ] && GAP_ARGS=(--adapter "$GAP_ADAPTER")

stage setup
run_capped 900 apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 git >/dev/null || exit 10
echo "[job] WO-01r demo rebuild, config $CFG, commit $(git rev-parse --short HEAD)"
# Vulkan for SAPIEN rendering: same EGL-library ICD as scripts/pod/wo02_job.sh / wo04_job.sh
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || run_capped 300 sh -c 'curl -LsSf --max-time 120 https://astral.sh/uv/install.sh | sh'
run_capped 1200 uv sync --frozen --extra gpu --extra dev || exit 11
run_capped 300 uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name())" || exit 12

stage gpu-sim-check
# the rebuild records on the GPU sim: the v2 oracle (tuned on the CPU sim) must still succeed there
run_until_deadline uv run scripts/wo01_oracle_eval.py --preset sim --config "$CFG" --device cuda \
  --num-envs 50 --episodes 50 --cells seen --out /workspace/oracle_gpu.json || exit 13
run_capped 120 uv run python - <<'EOF' || { echo "[job] v2 oracle < 95% on the GPU sim: no rebuild here"; exit 14; }
import json
res = json.load(open("/workspace/oracle_gpu.json"))["results"]
rates = {t: c["seen"]["rate"] for t, c in res.items()}
print("[job] v2 oracle on GPU sim:", rates,
      "decisions:", {t: c["seen"]["decisions_success"] for t, c in res.items()})
raise SystemExit(0 if min(rates.values()) >= 0.95 else 1)
EOF

stage probe-render-gap
# review 3.2 diagnostic: the 0.1 set's Mac renders vs this GPU renderer, SFT adapter tokens on both
run_until_deadline uv run scripts/wo01r_rebuild_demos.py --preset sim --config "$CFG" \
  --mode render-gap "${GAP_ARGS[@]}" || exit 20
upload "render-gap probe"

# Probe -> owner approval gate: no rebuild without an explicit go given AFTER the probe numbers
# are posted. The agent posts them and, on the owner's go, creates /workspace/APPROVED over SSH.
stage await-approval
APPROVAL_WAIT_S="${APPROVAL_WAIT_S:-1800}"
t0=$(date +%s)
until [ -f /workspace/APPROVED ]; do
  if [ $(( $(date +%s) - t0 )) -ge "$APPROVAL_WAIT_S" ]; then
    echo "[job] no owner approval within $((APPROVAL_WAIT_S / 60)) min; stopping"
    exit 21
  fi
  sleep 15
done
echo "[job] approved: $(cat /workspace/APPROVED)"

stage build
run_until_deadline uv run scripts/wo01r_rebuild_demos.py --preset sim --config "$CFG" \
  --mode build --push || exit 30
upload "rebuilt demos"

if [ "${VERIFY_GAP:-1}" = 1 ]; then
  stage verify-render-gap
  # the new set re-rendered on the backend that recorded it: expected at the codec floor
  NEW_REPO=$(run_capped 120 uv run python -c "from src.config import load_config; print(load_config('sim', '$CFG')['data']['hub_repo'])")
  run_until_deadline uv run scripts/wo01r_rebuild_demos.py --preset sim --config "$CFG" \
    --mode render-gap --gap-dataset "$NEW_REPO" "${GAP_ARGS[@]}" || exit 40
fi

stage "done"
