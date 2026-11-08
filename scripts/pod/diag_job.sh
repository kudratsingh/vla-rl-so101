#!/usr/bin/env bash
# Review Track B diagnostics (docs/reviews/2026-11-07-project-review.md section 8) on one RunPod
# A100-80GB pod: setup -> GPU check -> SFT adapter download -> temperature probe (rollouts only,
# no update: the short first step, ~0.7 h) -> owner approval gate -> offline lr sweep on one
# stored rollout batch (~1.2 h) -> upload runs/ + job log to the private Hub repo
# (KudratSingh22/vla-rl-so101-rl, diag/<pod>/) -> terminate this pod. Trains nothing that is kept.
# Launched over SSH from the repo root on the pod:
#   nohup bash scripts/pod/diag_job.sh > /workspace/job.log 2>&1 &
# Optional: SEED (3; RL seeds 0-2 are the WO-04 runs, check_rl_seed allows 0-9), SFT_ADAPTER (WO-02
#   a4 step_004000), RL_CFG (configs/rl.yaml), TEMPS (0.7,1.0,1.3), PROBE_GROUPS (64),
#   LRS (5e-5,2e-4,5e-4,1e-3), SWEEP_ADAM_STEPS (8), APPROVAL_WAIT_S (1800), HUB_REPO.
# Approval gate (CLAUDE.md rule 1): the lr sweep starts only once /workspace/APPROVED exists. The
#   agent creates it over SSH on the owner's explicit go after posting the TEMP-PROBE lines, OR
#   before launch when the owner pre-approved the whole pass (a dollar cap); a file that already
#   exists when the gate is reached counts as that pre-approval. No file within APPROVAL_WAIT_S ->
#   exit (upload + self-delete).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS; written by
# the pod's start command), /root/pod_start_epoch and /root/.hf_token. The pod also has a hard
# self-delete timer at the cap (docs/STATUS.md "Next paid run": SPEND_CAP_SECONDS 12240 = 3.4 h).
# NOT RUN YET: needs the owner's explicit go (CLAUDE.md rule 1).
set -uo pipefail
source /root/runpod.env
# cgroup CPU quota, not the host core count (scripts/pod/wo04_job.sh)
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  read -r q p 2>/dev/null < /sys/fs/cgroup/cpu.max \
    || { q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null); p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null); }
  if [ -n "${q:-}" ] && [ "$q" != max ] && [ "$q" -gt 0 ] 2>/dev/null; then
    export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
  fi
fi
echo "[job] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} (nproc $(nproc))"
export RUNPOD_POD_ID
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN="$(cat /root/.hf_token)"
export HF_TOKEN
SEED="${SEED:-3}"
SFT_ADAPTER="${SFT_ADAPTER:-KudratSingh22/vla-rl-so101-sft/wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000}"
export SEED SFT_ADAPTER
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-rl}"
CFG="${RL_CFG:-configs/rl.yaml}"
TEMPS="${TEMPS:-0.7,1.0,1.3}"
PROBE_GROUPS="${PROBE_GROUPS:-64}"
LRS="${LRS:-5e-5,2e-4,5e-4,1e-3}"
SWEEP_ADAM_STEPS="${SWEEP_ADAM_STEPS:-8}"
STAGE_FILE=/workspace/stage
# soft deadline 20 min before the hard cap, so results are uploaded before the pod disappears
DEADLINE=$(( $(cat /root/pod_start_epoch) + SPEND_CAP_SECONDS - 1200 ))

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
run_until_deadline() {
  local s; s=$(left)
  if [ "$s" -le 60 ]; then echo "[job] soft deadline reached before: $*"; return 124; fi
  timeout "$s" "$@"  # SIGTERM: wo03_rl.py finishes its run record (status interrupted)
}
pod_minutes() { echo $(( ($(date +%s) - $(cat /root/pod_start_epoch)) / 60 )); }

upload() {
  uv run python - "$1" <<'EOF' || echo "[job] upload failed"
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

finish() {
  local rc=$?
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null); pod up $(pod_minutes) min"
  upload "final (rc=$rc)"
  echo "[job] terminating pod $RUNPOD_POD_ID"
  # the pod-scoped key is accepted by REST v2 and GraphQL (REST v1 answers 403)
  curl -fsS -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    || curl -fsS https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}"
}
trap finish EXIT

stage setup
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 git >/dev/null || exit 10
echo "[job] diagnostics seed $SEED, SFT adapter $SFT_ADAPTER, config $CFG, commit $(git rev-parse --short HEAD)"
# Vulkan for SAPIEN rendering: same EGL-library ICD as scripts/pod/wo04_job.sh
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11

stage gpu-check
uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name(), round(torch.cuda.mem_get_info()[1] / 1e9), 'GB')" || exit 12
# both probe commands must parse (seed range, config) before any GPU minute is spent on them
uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" --estimate \
  --probe-temperatures "$TEMPS" --probe-groups "$PROBE_GROUPS" | tail -1 || exit 13
SWEEP_H=$(uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" --estimate \
  --lr-sweep "$LRS" --sweep-adam-steps "$SWEEP_ADAM_STEPS" | awk '/lr_sweep_h:/ {print $2}')
[ -n "$SWEEP_H" ] || exit 13
echo "[job] lr sweep estimate ${SWEEP_H} h (cost.throughput)"

stage download
# the SFT adapter (fails fast on a wrong path); the 15 GB base downloads in the first probe
uv run python - <<'EOF' || exit 16
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

stage await-approval
grep "\[wo03_rl\] TEMP-PROBE" /workspace/job.log | tail -4
# the sweep must finish before the soft deadline, with the 25% estimate-error margin
NEED_S=$(awk -v h="$SWEEP_H" 'BEGIN { printf "%d", h * 3600 * 1.25 + 120 }')
echo "[job] pod up $(pod_minutes) min so far (setup + checks + temperature probe); next: lr sweep" \
  "$LRS x $SWEEP_ADAM_STEPS Adam steps, est. ${SWEEP_H} h (+25% = $(( NEED_S / 60 )) min);" \
  "soft deadline in $(( $(left) / 60 )) min"
if [ "$(left)" -lt "$NEED_S" ]; then
  echo "[job] not enough time before the soft deadline for the lr sweep; stopping"
  exit 23
fi
if [ -f /workspace/APPROVED ]; then
  echo "[job] pre-approved at launch: $(cat /workspace/APPROVED)"
else
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
