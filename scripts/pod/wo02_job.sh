#!/usr/bin/env bash
# WO-02 on one RunPod GPU pod: setup -> probe -> owner approval -> LoRA SFT (sim.yaml budget) -> checkpoint sweep
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
CFG="${SFT_CFG:-configs/sft.yaml}"
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
# Vulkan for SAPIEN rendering. RunPod mounts an ICD at /etc/vulkan/icd.d pointing at
# libGLX_nvidia.so.0, which cannot create an instance headless (driver 595.91.07, pod
# iqrmjox6ws0e66); NVIDIA's EGL library exposes the same Vulkan driver and works. SAPIEN only
# checks /usr/share/vulkan/icd.d/nvidia_icd.json, and otherwise substitutes its own GLX-based file.
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11
uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name())" || exit 12
DEMOS_REPO="${DEMOS_REPO:-KudratSingh22/vla-rl-so101-demos}"  # must match the config's data.hub_repo
uv run hf download "$DEMOS_REPO" --repo-type dataset \
  --local-dir "data/demos/${DEMOS_REPO##*/}" >/dev/null || exit 13

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
then
  # the oracle was tuned on the CPU sim; evaluate policies on the GPU sim only if the oracle
  # still succeeds there (>= 90% per task, seen cell)
  if uv run scripts/wo01_oracle_eval.py --preset sim --device cuda --num-envs 20 --episodes 20 \
      --cells seen --out /workspace/oracle_gpu.json && uv run python - <<'EOF'
import json
res = json.load(open("/workspace/oracle_gpu.json"))["results"]
rates = {t: c["seen"]["rate"] for t, c in res.items()}
print("[job] oracle on GPU sim:", rates)
raise SystemExit(0 if min(rates.values()) >= 0.9 else 1)
EOF
  then SIM_DEVICE=cuda; EVAL_ENVS=50; fi
fi
echo "[job] eval sim device: $SIM_DEVICE ($EVAL_ENVS envs)"

# START_AT=sweep SFT_RUN=runs/<id>: evaluate an existing (or still running) SFT run; skips the
# probe, the approval gate and training (that run was already approved)
if [ "${START_AT:-setup}" = sweep ]; then
  RUN="$SFT_RUN"
  while pgrep -f "scripts/wo02_sft.py" >/dev/null; do sleep 30; done
  [ -f "$RUN/record.json" ] && [ -d "$RUN/checkpoints" ] || { echo "[job] no SFT run at $RUN"; exit 31; }
  echo "[job] resuming at sweep for SFT run: $RUN"
  upload "SFT checkpoints"
else

stage probe
run_until_deadline uv run scripts/wo02_sft.py --preset sim --config "$CFG" --probe-steps 20 || exit 20

# Probe -> owner approval gate (owner rule, 2026-10-26): no full run without an explicit go given
# AFTER the probe numbers are posted. The agent posts them and, on the owner's go, creates
# /workspace/APPROVED over SSH. No approval within APPROVAL_WAIT_S -> exit (upload + self-delete).
stage await-approval
grep "full run .* extrapolated" /workspace/job.log | tail -1
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

stage sft
run_until_deadline uv run scripts/wo02_sft.py --preset sim --config "$CFG" || exit 30
RUN=$(ls -d runs/*-wo02-sim-* | sort | tail -1)
echo "[job] SFT run: $RUN"
upload "SFT checkpoints"
fi

stage sweep
for CK in "$RUN"/checkpoints/step_*; do
  run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$CK" \
    --cells seen --seeds 0 --episodes 50 --num-envs "$EVAL_ENVS" --sim-device "$SIM_DEVICE" || exit 40
  echo "$CK $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" >> "$RUN/sweep.txt"
done

stage select
# Owner rules (2026-10-26). In band = 20-60% seen success on >= 2 of 3 tasks (WO-02 benchmark).
# Final checkpoint in band -> use it. Otherwise the EARLIEST swept checkpoint in band (the owner's
# rule when the final one lands above 60%), before any retrain. None in band -> no full eval; the
# report decides (below: up to two more SFT attempts, then pi0.5; above: cut steps or demos).
BEST=$(uv run python - "$RUN/sweep.txt" <<'EOF'
import json, sys
rows = []
for line in open(sys.argv[1]):
    ck, ev = line.split()
    rates = [r["pooled"] for r in json.load(open(f"{ev}/eval.json"))["rows"] if r["cell"] == "seen"]
    in_band = sum(0.2 <= x <= 0.6 for x in rates) >= 2
    side = "in band" if in_band else ("above" if sum(x > 0.6 for x in rates) >= 2 else "below")
    rows.append((ck, in_band, side))
    print(f"[job] {ck}: seen {[round(x, 3) for x in rates]} -> {side}", file=sys.stderr)
rows.sort()  # step_000500 < step_001000 < ... (zero-padded)
final = rows[-1]
if final[1]:
    pick, why = final[0], "final checkpoint in band"
elif any(r[1] for r in rows):
    pick, why = next(r[0] for r in rows if r[1]), f"final {final[2]} -> earliest checkpoint in band"
else:
    pick, why = "", f"no checkpoint in band (final {final[2]})"
print(f"[job] selection: {why}", file=sys.stderr)
print(pick)
EOF
) || exit 50
if [ -z "$BEST" ]; then
  echo "none" > "$RUN/selected_checkpoint.txt"
  echo "[job] no checkpoint in the 20-60% band; skipping the full eval"
  stage done
  exit 0
fi
echo "$BEST" > "$RUN/selected_checkpoint.txt"
echo "[job] selected: $BEST"

stage full-eval
run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$BEST" \
  --num-envs "$EVAL_ENVS" --sim-device "$SIM_DEVICE" || exit 60

stage done
