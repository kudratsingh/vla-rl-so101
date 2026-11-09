#!/usr/bin/env bash
# Review Track B diagnostics (docs/reviews/2026-11-07-project-review.md section 8) on one RunPod
# A100-80GB pod: setup -> GPU check -> SFT adapter download -> temperature probe (rollouts only,
# no update: the short first step, ~0.7 h) -> paired clip on/off probe (rollouts only, ~0.24 h;
# owner 2026-11-08) -> owner approval gate -> offline lr sweep on one stored rollout batch
# (~1.2 h) -> upload runs/ + job log to the private Hub repo (KudratSingh22/vla-rl-so101-rl,
# diag/<pod>/) -> terminate this pod. Trains nothing that is kept.
# Every probe reads the action clip from the preset (RL_CFG's rl.rollout_action_clip_deg;
# configs/rl.yaml: 4.0 deg/step, owner 2026-11-08) and prints it; there is no separate default.
# Launched over SSH from the repo root on the pod:
#   nohup bash scripts/pod/diag_job.sh > /workspace/job.log 2>&1 &
# Optional: SEED (3; RL seeds 0-2 are the WO-04 runs, check_rl_seed allows 0-9), SFT_ADAPTER (WO-02
#   a4 step_004000), RL_CFG (configs/rl.yaml), TEMPS (0.7,1.0,1.3), PROBE_GROUPS (64),
#   CLIP_GROUPS (32 prompts x the preset's K per arm; 0 skips the clip probe; it is also skipped
#   when the pod's cap leaves no room for it AND the lr sweep, and a clip-probe failure other than
#   a timeout continues to the gate: the lr sweep keeps priority),
#   LRS (5e-5,2e-4,5e-4,1e-3), SWEEP_ADAM_STEPS (8), APPROVAL_WAIT_S (1800), HUB_REPO.
# Approval gate (CLAUDE.md rule 1): the lr sweep starts only once THIS pod's approval file
#   (/workspace/APPROVED.<RUNPOD_POD_ID>, written after the pod's first boot: scripts/pod/
#   approval_lib.sh; a stale or unscoped /workspace/APPROVED never counts) exists. The agent writes
#   it over SSH with `approval_write` on the owner's explicit go after posting the TEMP-PROBE and
#   CLIP-PROBE lines, OR before launch when the owner pre-approved the whole pass (a dollar cap); a valid
#   file that already exists when the gate is reached counts as that pre-approval. No valid file
#   within APPROVAL_WAIT_S -> exit (upload + self-delete).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS,
# POD_HARD_DEADLINE) and /root/pod_start_epoch, both written by scripts/pod/pod_start_cmd.sh (the
# pod's start command, which also arms the hard self-delete timer: docs/STATUS.md "Next paid run",
# SPEND_CAP_SECONDS 13800 = 3.83 h with the clip probe; 12000 pre-approved), and /root/.hf_token. Every stage is time-limited and a watchdog
# stops the job at the soft deadline (scripts/pod/job_lib.sh).
# NOT RUN YET: needs the owner's explicit go (CLAUDE.md rule 1).
set -uo pipefail
# shellcheck source=scripts/pod/job_lib.sh
. "$(dirname "$0")/job_lib.sh"
# shellcheck source=scripts/pod/approval_lib.sh
. "$(dirname "$0")/approval_lib.sh"
STAGE_FILE=/workspace/stage
export PATH="$HOME/.local/bin:$PATH"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-rl}"

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }
pod_minutes() { echo $(( ($(date +%s) - $(cat /root/pod_start_epoch 2>/dev/null || date +%s)) / 60 )); }

upload() {
  timeout -k 30 "$UPLOAD_TIMEOUT_S" uv run --no-sync python - "$1" <<'EOF' || echo "[job] upload failed or timed out"
import os, sys
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
dest = f"diag/{pod}"
api.create_repo(repo, private=True, exist_ok=True)
if os.path.isdir("runs"):
    # record.json, metrics.jsonl and the lr sweep's sweep_batch.pt (replayable with --sweep-batch)
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"{dest}/runs",
                      commit_message=f"diagnostics pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"{dest}/job.log",
                 repo_id=repo, commit_message=f"diagnostics pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/{dest}")
EOF
}

# installed before anything can fail: every exit uploads (time-limited), then self-deletes
finish() {
  local rc=$?
  stop_watchdog
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null); pod up $(pod_minutes) min"
  upload "final (rc=$rc)"
  delete_pod
}
trap finish EXIT
pod_env || exit 9
# both outside watchdogs confirmed THIS pod (launch-gate token), before any paid setup: rc 8
require_watchdogs || exit 8
start_watchdog

# cgroup CPU quota, not the host core count (scripts/pod/wo04_job.sh)
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max \
    || { q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); }
  if [ -n "${q:-}" ] && [ "$q" != max ] && [ "$q" -gt 0 ] 2>/dev/null; then
    export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
  fi
fi
echo "[job] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} (nproc $(nproc))"
HF_TOKEN="$(cat /root/.hf_token)"
export HF_TOKEN
SEED="${SEED:-3}"
SFT_ADAPTER="${SFT_ADAPTER:-KudratSingh22/vla-rl-so101-sft/wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000}"
export SEED SFT_ADAPTER
CFG="${RL_CFG:-configs/rl.yaml}"
TEMPS="${TEMPS:-0.7,1.0,1.3}"
PROBE_GROUPS="${PROBE_GROUPS:-64}"
CLIP_GROUPS="${CLIP_GROUPS:-32}"
LRS="${LRS:-5e-5,2e-4,5e-4,1e-3}"
SWEEP_ADAM_STEPS="${SWEEP_ADAM_STEPS:-8}"

stage setup
run_capped 900 apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 git >/dev/null || exit 10
echo "[job] diagnostics seed $SEED, SFT adapter $SFT_ADAPTER, config $CFG, commit $(git rev-parse --short HEAD)"
# Vulkan for SAPIEN rendering: same EGL-library ICD as scripts/pod/wo04_job.sh
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || run_capped 300 sh -c 'curl -LsSf --max-time 120 https://astral.sh/uv/install.sh | sh'
run_capped 1200 uv sync --frozen --extra gpu --extra dev || exit 11

stage gpu-check
run_capped 300 uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name(), round(torch.cuda.mem_get_info()[1] / 1e9), 'GB')" || exit 12
# both probe commands must parse (seed range, config) before any GPU minute is spent on them
# (`set -o pipefail`: a failing --estimate fails the pipe; it prints the preset's clip setting)
run_capped 300 uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" --estimate \
  --probe-temperatures "$TEMPS" --probe-groups "$PROBE_GROUPS" | tail -2 || exit 13
CLIP_S=0
if [ "$CLIP_GROUPS" != 0 ]; then
  # refuses a preset whose rl.rollout_action_clip_deg is null: nothing to compare
  # (the estimate prints thousands separators: stripped before awk reads the number)
  CLIP_S=$(run_capped 300 uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" --estimate \
    --probe-clip --probe-groups "$CLIP_GROUPS" | awk '/clip_probe_s:/ {gsub(",", "", $2); printf "%d", $2}')
  is_int "$CLIP_S" && [ "$CLIP_S" -gt 0 ] || exit 13
  echo "[job] clip probe estimate ${CLIP_S} s (cost.throughput)"
fi
SWEEP_H=$(run_capped 300 uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" --estimate \
  --lr-sweep "$LRS" --sweep-adam-steps "$SWEEP_ADAM_STEPS" | awk '/lr_sweep_h:/ {gsub(",", "", $2); print $2}')
[ -n "$SWEEP_H" ] || exit 13
echo "[job] lr sweep estimate ${SWEEP_H} h (cost.throughput)"
# the sweep must finish before the soft deadline, with the 25% estimate-error margin
NEED_S=$(awk -v h="$SWEEP_H" 'BEGIN { printf "%d", h * 3600 * 1.25 + 120 }')

stage download
# the SFT adapter (fails fast on a wrong path); the 15 GB base downloads in the first probe
run_capped 900 uv run python - <<'EOF' || exit 16
import os
from src.policy.openvla_oft import resolve_adapter
path = resolve_adapter(os.environ["SFT_ADAPTER"])
assert path is not None and (path / "norm_stats.json").is_file(), path
print(f"[job] SFT adapter: {path}: {sorted(p.name for p in path.iterdir())}")
EOF

stage probe-temperatures
# rollouts only, the same prompts at each temperature: sampled success, group-drop rate, entropy
run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
  --adapter "$SFT_ADAPTER" --probe-temperatures "$TEMPS" --probe-groups "$PROBE_GROUPS" || exit 20
upload "temperature probe"

# the clip probe runs only if the lr sweep still fits after it: clip x 1.25 + 2 min, the sweep's
# NEED_S, and the full approval wait unless the pass is pre-approved (a pod launched at the
# round-2 caps, 12660 / 10860 s, skips it here instead of losing the sweep at the gate)
CLIP_NEED_S=$(( CLIP_S * 5 / 4 + 120 + NEED_S ))
approval_valid || CLIP_NEED_S=$(( CLIP_NEED_S + ${APPROVAL_WAIT_S:-1800} ))
if [ "$CLIP_GROUPS" != 0 ] && [ "$(left)" -lt "$CLIP_NEED_S" ]; then
  echo "[job] clip probe skipped: $(( $(left) / 60 )) min to the soft deadline < $(( CLIP_NEED_S / 60 ))" \
    "min for it + the lr sweep (+ approval wait); the lr sweep keeps priority"
elif [ "$CLIP_GROUPS" != 0 ]; then
  stage probe-clip
  # paired, rollouts only: the same prompts and sampling seeds with the preset's clip and without;
  # per arm sampled success, group-drop rate, clip intervention (arm off: would have intervened).
  # Before the gate: its numbers inform the owner's go on the lr sweep, and a gate of its own
  # would idle the A100 for up to APPROVAL_WAIT_S (30 min = $0.90) to save a ~14 min ($0.43) stage
  # Optional: a timeout (no time left) ends the job; any other failure continues to the gate,
  # whose NEED_S check still guards the cap; the gate output then shows no CLIP-PROBE lines
  run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
    --adapter "$SFT_ADAPTER" --probe-clip --probe-groups "$CLIP_GROUPS"
  rc=$?
  [ "$rc" = 124 ] && exit 22
  [ "$rc" = 0 ] || echo "[job] clip probe failed (rc=$rc); continuing to the lr-sweep gate"
  upload "clip probe (rc=$rc)"
fi

stage await-approval
grep "\[wo03_rl\] TEMP-PROBE" /workspace/job.log | tail -4
grep "\[wo03_rl\] CLIP-PROBE" /workspace/job.log | tail -4 || echo "[job] no CLIP-PROBE lines (skipped or failed)"
echo "[job] pod up $(pod_minutes) min so far (setup + checks + temperature + clip probes); next: lr sweep" \
  "$LRS x $SWEEP_ADAM_STEPS Adam steps, est. ${SWEEP_H} h (+25% = $(( NEED_S / 60 )) min);" \
  "soft deadline in $(( $(left) / 60 )) min"
if [ "$(left)" -lt "$NEED_S" ]; then
  echo "[job] not enough time before the soft deadline for the lr sweep; stopping"
  exit 23
fi
if approval_valid; then
  echo "[job] pre-approved at launch: $(cat "$(approval_file)")"
else
  await_approval "${APPROVAL_WAIT_S:-1800}" || exit 21
  if [ "$(left)" -lt "$NEED_S" ]; then
    echo "[job] the approval wait used the time the lr sweep needs; stopping"
    exit 23
  fi
fi

stage lr-sweep
# one rollout batch (saved as runs/<id>/sweep_batch.pt), then per lr: restore adapter + Adam
# state, SWEEP_ADAM_STEPS optimizer steps, on-batch movement (k3 KL, clip, dlogp, argmax flips)
run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
  --adapter "$SFT_ADAPTER" --lr-sweep "$LRS" --sweep-adam-steps "$SWEEP_ADAM_STEPS" || exit 30
grep "\[wo03_rl\] LR-SWEEP" /workspace/job.log | tail -5

stage "done"
