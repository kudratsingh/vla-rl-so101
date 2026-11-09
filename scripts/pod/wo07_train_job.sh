#!/usr/bin/env bash
# WO-07 real-arm fine-tune on one RunPod RTX 4090 (24 GB) pod, ONE POLICY PER POD (SmolVLA or ACT,
# the same demos): setup -> dataset download at a pinned Hub commit -> PROBE (scripts/wo07_train.py
# --probe: s/step, peak VRAM, projected h and $ of the full run; batch 64 -> 32 -> 16 on CUDA OOM)
# -> owner approval gate -> full run at the probed batch (+ the save/reload check) -> upload runs/
# and the final checkpoint to a private Hub repo -> terminate this pod.
# Launched over SSH from the repo root on the pod:
#   POLICY=smolvla DATASET_REVISION=<commit> nohup bash scripts/pod/wo07_train_job.sh > /workspace/job.log 2>&1 &
# Required: POLICY (smolvla | act), DATASET_REVISION (Hub commit of the 50-demo dataset).
# Optional: TRAIN_CFG (configs/wo07_train.yaml), DATASET_REPO (its dataset.repo_id), HUB_REPO
#   (KudratSingh22/vla-rl-so101-real-policies), PROBE_BATCHES (smolvla "64 32 16", act "8"),
#   APPROVAL_WAIT_S (1800), UPLOAD_EVERY_S (7200).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS,
# POD_HARD_DEADLINE) and /root/pod_start_epoch, both written by scripts/pod/pod_start_cmd.sh (the
# pod's start command, which also arms the hard self-delete timer; persistent volume at
# /workspace), and /root/.hf_token. The launch gate (scripts/pod/launch_gate.py,
# docs/runbooks/pod-launch.md) must have written /workspace/WATCHDOGS_OK.<pod id> before the job
# starts. Every stage is time-limited and a watchdog stops the job at the soft deadline
# (scripts/pod/job_lib.sh).
#
# DRY RUN (the Mac, from the repo root, $0): DRY_RUN=1 POLICY=act bash scripts/pod/wo07_train_job.sh
#   The same stages and gates in a throw-away sandbox (mktemp) standing in for /workspace and
#   /root: a simulated launch-gate token and owner approval for a fake pod id "dryrun-<pid>", the
#   probe and the full run on the `tiny` preset (CPU), and NO apt, uv sync, Hub download, Hub upload,
#   RunPod call or pod delete: each is printed as "[dry-run] would ...". Refuses to run on a pod.
# NOT RUN YET on a pod: needs the owner's explicit go (CLAUDE.md rule 1).
set -uo pipefail
DRY_RUN=${DRY_RUN:-0}
if [ "$DRY_RUN" = 1 ]; then
  if [ -e /root/runpod.env ]; then
    echo "[job] DRY_RUN=1 refused: /root/runpod.env exists (this is a pod)"
    exit 2
  fi
  SANDBOX=$(mktemp -d "${TMPDIR:-/tmp}/wo07-dryrun.XXXXXX")
  export APPROVAL_DIR="$SANDBOX/workspace" POD_START_EPOCH_FILE="$SANDBOX/pod_start_epoch"
  export WATCHDOG_TOKEN_DIR="$SANDBOX/workspace" APPROVAL_POLL_S=1
  export RUNPOD_POD_ID="dryrun-$$"
  mkdir -p "$APPROVAL_DIR"
  # macOS strips DYLD_* when it starts /bin/bash (SIP): re-export torchcodec's FFmpeg path
  if [ "$(uname -s)" = Darwin ]; then
    # shellcheck source=scripts/mac_env.sh
    . "$(dirname "$0")/../mac_env.sh"
  fi
fi
# shellcheck source=scripts/pod/job_lib.sh
. "$(dirname "$0")/job_lib.sh"
# shellcheck source=scripts/pod/approval_lib.sh
. "$(dirname "$0")/approval_lib.sh"
WS=${APPROVAL_DIR}
STAGE_FILE="$WS/stage"
JOB_LOG="$WS/job.log"
UPLOADER=""
export PATH="$HOME/.local/bin:$PATH"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-real-policies}"
POLICY=${POLICY:-}
CFG="${TRAIN_CFG:-configs/wo07_train.yaml}"
PRESET=real

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }

if [ "$DRY_RUN" = 1 ]; then
  PRESET=tiny
  CFG=configs/wo07_train_tiny.yaml
  # GNU timeout is not on macOS: the dry run is a plumbing check, run without the time limit
  if ! command -v timeout >/dev/null; then
    timeout() {
      [ "$1" = -k ] && shift 2
      shift
      "$@"
    }
  fi
  delete_pod() { echo "[dry-run] would terminate pod $RUNPOD_POD_ID (scripts/pod/delete_self.sh)"; }
fi

# the newest run dir of this job's policy ("" if none): runs/<stamp>-wo07-<preset>-<hex>
latest_run() {
  local r best=""
  for r in runs/*-wo07-"$PRESET"-*; do
    [ -f "$r/record.json" ] || continue
    grep -q "\"policy\": \"$POLICY\"" "$r/record.json" || continue
    best=$r
  done
  echo "$best"
}

upload() {
  local why=$1 run ck=""
  run=$(latest_run)
  [ -n "$run" ] && [ -d "$run/train/checkpoints/last/pretrained_model" ] \
    && ck="$run/train/checkpoints/last/pretrained_model"
  if [ "$DRY_RUN" = 1 ]; then
    echo "[dry-run] would upload to $HUB_REPO wo07/$POLICY/$RUNPOD_POD_ID: runs/ (record, metrics, probe;" \
      "no checkpoints), the job log, and checkpoint '${ck:-none}' ($why)"
    return 0
  fi
  CK="$ck" timeout -k 30 "$UPLOAD_TIMEOUT_S" uv run --no-sync python - "$why" <<'EOF' || echo "[job] upload failed or timed out"
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
dest = f"wo07/{os.environ['POLICY']}/{pod}"
api.create_repo(repo, private=True, exist_ok=True)
if os.path.isdir("runs"):
    # records, metrics and probe files; checkpoints separately (the last one only), no dataset copy
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"{dest}/runs",
                      ignore_patterns=["**/train/checkpoints/**", "**/dataset/**"],
                      commit_message=f"WO-07 {os.environ['POLICY']} pod {pod}: {sys.argv[1]}")
ck = os.environ.get("CK")
if ck:
    run_id = Path(ck).parts[1]
    api.upload_folder(repo_id=repo, folder_path=ck, path_in_repo=f"{dest}/{run_id}/pretrained_model",
                      commit_message=f"WO-07 {os.environ['POLICY']} pod {pod}: checkpoint ({sys.argv[1]})")
api.upload_file(path_or_fileobj=os.environ["JOB_LOG"], path_in_repo=f"{dest}/job.log",
                repo_id=repo, commit_message=f"WO-07 pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/{dest}")
EOF
}
export JOB_LOG POLICY

# installed before anything can fail: every exit uploads (time-limited), then self-deletes
finish() {
  local rc=$?
  stop_watchdog
  [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null)"
  upload "final (rc=$rc)"
  delete_pod
  if [ "$DRY_RUN" = 1 ]; then rm -rf "$SANDBOX"; fi
}
trap finish EXIT

if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] sandbox $SANDBOX stands in for /workspace + /root; pod id $RUNPOD_POD_ID"
  echo $(($(date +%s) - 5)) > "$POD_START_EPOCH_FILE"
  # shellcheck disable=SC2034  # read by job_lib.sh (left, run_capped, start_watchdog)
  DEADLINE=$(($(date +%s) + 3600))
else
  pod_env || exit 9
fi
# both outside watchdogs confirmed THIS pod (launch-gate token), before any paid setup: rc 8
if [ "$DRY_RUN" = 1 ]; then
  echo "pod=$RUNPOD_POD_ID github_run=1 dispatch=dryrun mac_pid=$$ mac_tag=dryrun written=$(date -u +%FT%TZ)" \
    > "$(watchdog_token_file)"
  echo "[dry-run] simulated launch-gate token $(watchdog_token_file)"
fi
require_watchdogs || exit 8
start_watchdog

case "$POLICY" in
  smolvla) PROBE_BATCHES=${PROBE_BATCHES:-"64 32 16"} ;;
  act) PROBE_BATCHES=${PROBE_BATCHES:-"8"} ;;
  *) echo "[job] POLICY must be smolvla or act, got '$POLICY'"; exit 7 ;;
esac
REV_ARGS=()
if [ "$DRY_RUN" != 1 ]; then
  [ -n "${DATASET_REVISION:-}" ] || { echo "[job] DATASET_REVISION (Hub commit of the demo set) is required"; exit 7; }
  REV_ARGS=(--dataset-revision "$DATASET_REVISION")
fi
if [ -z "${OMP_NUM_THREADS:-}" ] && [ "$DRY_RUN" != 1 ]; then
  read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max \
    || { q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); }
  if [ -n "${q:-}" ] && [ "$q" != max ] && [ "$q" -gt 0 ] 2>/dev/null; then
    export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
  fi
fi
if [ "$DRY_RUN" != 1 ]; then
  HF_TOKEN="$(cat /root/.hf_token)"
  export HF_TOKEN
fi
TRAIN=(uv run --no-sync python scripts/wo07_train.py --preset "$PRESET" --config "$CFG" --policy "$POLICY")

stage setup
echo "[job] WO-07 $POLICY fine-tune, config $CFG, commit $(git rev-parse --short HEAD 2>/dev/null)"
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] would: apt-get install ffmpeg git; uv sync --frozen --extra gpu --extra smolvla; CUDA check"
else
  run_capped 900 apt-get install -y -qq --no-install-recommends ffmpeg git >/dev/null || exit 10
  command -v uv >/dev/null || run_capped 300 sh -c 'curl -LsSf --max-time 120 https://astral.sh/uv/install.sh | sh'
  run_capped 1200 uv sync --frozen --extra gpu --extra smolvla || exit 11
  run_capped 300 uv run --no-sync python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name())" || exit 12
fi

stage dataset
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] would: hf download <dataset.repo_id> --repo-type dataset --revision \$DATASET_REVISION (tiny builds a synthetic set)"
else
  DATASET_REPO=${DATASET_REPO:-$(uv run --no-sync python -c "from src.config import load_config; print(load_config('real', '$CFG')['dataset']['repo_id'])")}
  DS_ROOT=$(uv run --no-sync python -c "from src.config import load_config; print(load_config('real', '$CFG')['dataset']['root'])")
  run_capped 1800 uv run --no-sync hf download "$DATASET_REPO" --repo-type dataset \
    --revision "$DATASET_REVISION" --local-dir "$DS_ROOT/${DATASET_REPO##*/}" >/dev/null || exit 13
  echo "[job] dataset $DATASET_REPO @ $DATASET_REVISION -> $DS_ROOT/${DATASET_REPO##*/}"
fi

stage probe
# PROBE: a short timed run, no checkpoint (wo07_train.py --probe). The largest batch in
# PROBE_BATCHES that fits wins (rc 3 = CUDA OOM -> next); prints the `[probe]` lines.
BATCH=""
for B in $PROBE_BATCHES; do
  run_capped 3600 "${TRAIN[@]}" --probe --batch-size "$B" ${REV_ARGS[@]+"${REV_ARGS[@]}"}
  rc=$?
  if [ "$rc" = 0 ]; then BATCH=$B; break; fi
  if [ "$rc" != 3 ]; then echo "[job] probe failed (rc=$rc) at batch $B"; exit 20; fi
  echo "[job] batch $B does not fit in VRAM; trying the next"
done
[ -n "$BATCH" ] || { echo "[job] no batch in '$PROBE_BATCHES' fits"; exit 22; }
upload "probe (batch $BATCH)"

# Probe -> owner approval gate (CLAUDE.md rule 1): no full run without an explicit go given AFTER
# the probe numbers (step time, peak VRAM, batch, projected h and $) are posted. The agent posts
# them; on the owner's go it writes this pod's approval file over SSH (`approval_write`,
# scripts/pod/approval_lib.sh: /workspace/APPROVED.<pod id>, newer than the pod's first boot). No
# approval within APPROVAL_WAIT_S -> exit (upload + self-delete).
stage await-approval
[ -f "$JOB_LOG" ] && grep "\[probe\]" "$JOB_LOG" | tail -4
echo "[job] pod up $(( ($(date +%s) - $(cat "$POD_START_EPOCH_FILE")) / 60 )) min so far; soft deadline in $(( $(left) / 60 )) min"
if [ "$DRY_RUN" = 1 ]; then
  approval_write "DRY RUN: simulated owner go" || exit 21
  echo "[dry-run] simulated owner approval $(approval_file)"
fi
await_approval "${APPROVAL_WAIT_S:-1800}" || exit 21

stage train
# periodic checkpoint upload; its sleep is killed with it (an orphaned sleep would hold the log open)
(
  sp=""
  trap '[ -n "$sp" ] && kill "$sp" 2>/dev/null; exit 0' TERM
  while :; do
    sleep "${UPLOAD_EVERY_S:-7200}" &
    sp=$!
    wait "$sp"
    upload "periodic (training)"
  done
) &
UPLOADER=$!
run_until_deadline "${TRAIN[@]}" --batch-size "$BATCH" ${REV_ARGS[@]+"${REV_ARGS[@]}"}
RC=$?
kill "$UPLOADER" 2>/dev/null; UPLOADER=""
RUN=$(latest_run)
[ -n "$RUN" ] || { echo "[job] no WO-07 run directory"; exit 30; }
echo "[job] run $RUN (rc=$RC)"
[ "$RC" = 0 ] || exit 31

stage upload
upload "trained checkpoint"
stage "done"
