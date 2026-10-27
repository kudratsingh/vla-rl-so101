#!/usr/bin/env bash
# WO-04 on one RunPod A100-80GB pod, ONE SEED PER POD: setup -> GPU-sim check (render, group
# reset, oracle; RL needs the GPU sim) -> SFT adapter download -> PROBE (SFT baseline eval + a few
# GRPO steps: s/step, episodes/h, VRAM, projected h and $ per seed) -> owner approval gate -> full
# GRPO run (kill switch at 25% of the steps) -> full-protocol eval of the final RL checkpoint ->
# upload runs/ to the private Hub repo -> terminate this pod.
# Launched over SSH from the repo root on the pod (one seed per pod):
#   SEED=0 SFT_ADAPTER=KudratSingh22/vla-rl-so101-sft/wo02/<pod>/runs/<run>/checkpoints/step_00N000 \
#     nohup bash scripts/pod/wo04_job.sh > /workspace/job.log 2>&1 &
# SFT_ADAPTER: the WO-02-selected checkpoint (its selected_checkpoint.txt), as a Hub path
#   user/repo/path/in/repo[@rev] or a local dir (src/policy/openvla_oft.py resolve_adapter).
# Optional: HUB_REPO (results repo), RL_CFG (configs/rl.yaml), PROBE_STEPS (3), APPROVAL_WAIT_S
#   (1800), UPLOAD_EVERY_S (7200), EVAL_SFT=1 (also run the full protocol on the SFT adapter here,
#   for an SFT row on the same GPU sim; needed if the WO-02 full eval ran on the CPU sim).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS; written by
# the pod's start command), /root/pod_start_epoch and /root/.hf_token. The pod also has a hard
# self-delete timer at the cap; size the cap from the probe's projection (+ ~1 h).
set -uo pipefail
source /root/runpod.env
# Containers see every host core (128 on some A100 hosts) but get a cgroup CPU quota (~14);
# size torch/OpenMP thread pools to the quota or CPU-side work (IK, data loading) crawls.
if [ -z "${OMP_NUM_THREADS:-}" ] && read -r q p < /sys/fs/cgroup/cpu.max 2>/dev/null && [ "$q" != max ]; then
  export OMP_NUM_THREADS=$(( (q + p - 1) / p )) MKL_NUM_THREADS=$(( (q + p - 1) / p ))
fi
export RUNPOD_POD_ID
export PATH="$HOME/.local/bin:$PATH"
HF_TOKEN="$(cat /root/.hf_token)"
export HF_TOKEN
: "${SEED:?set SEED (one pod per seed: 0, 1 or 2)}"
SFT_ADAPTER="${SFT_ADAPTER:-KudratSingh22/vla-rl-so101-sft/wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000}"  # WO-02 base (owner 2026-10-27: a4 step 4000, seen 18/16/6%)
export SEED SFT_ADAPTER
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-rl}"
CFG="${RL_CFG:-configs/rl.yaml}"
PROBE_STEPS="${PROBE_STEPS:-3}"
STAGE_FILE=/workspace/stage
UPLOADER=""
# soft deadline 20 min before the hard cap, so results are uploaded before the pod disappears
DEADLINE=$(( $(cat /root/pod_start_epoch) + SPEND_CAP_SECONDS - 1200 ))

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }
left() { echo $(( DEADLINE - $(date +%s) )); }
run_until_deadline() {
  local s; s=$(left)
  if [ "$s" -le 60 ]; then echo "[job] soft deadline reached before: $*"; return 124; fi
  timeout "$s" "$@"  # SIGTERM: wo03_rl.py saves a step_*_interrupted checkpoint and its record
}

upload() {
  uv run python - "$1" <<'EOF' || echo "[job] upload failed"
import os, sys
from huggingface_hub import HfApi
api, repo, pod = HfApi(), os.environ["HUB_REPO"], os.environ["RUNPOD_POD_ID"]
dest = f"wo04/seed{os.environ['SEED']}/{pod}"
api.create_repo(repo, private=True, exist_ok=True)
if os.path.isdir("runs"):
    api.upload_folder(repo_id=repo, folder_path="runs", path_in_repo=f"{dest}/runs",
                      commit_message=f"WO-04 seed {os.environ['SEED']} pod {pod}: {sys.argv[1]}")
api.upload_file(path_or_fileobj="/workspace/job.log", path_in_repo=f"{dest}/job.log",
                 repo_id=repo, commit_message=f"WO-04 pod {pod}: job log")
print(f"[job] uploaded to https://huggingface.co/{repo}/tree/main/{dest}")
EOF
}

finish() {
  local rc=$?
  [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null
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
echo "[job] WO-04 seed $SEED, SFT adapter $SFT_ADAPTER, config $CFG, commit $(git rev-parse --short HEAD)"
# Vulkan for SAPIEN rendering. RunPod mounts an ICD at /etc/vulkan/icd.d pointing at
# libGLX_nvidia.so.0, which cannot create an instance headless (driver 595.91.07, pod
# iqrmjox6ws0e66); NVIDIA's EGL library exposes the same Vulkan driver and works. SAPIEN only
# checks /usr/share/vulkan/icd.d/nvidia_icd.json, and otherwise substitutes its own GLX-based file.
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --frozen --extra gpu --extra dev || exit 11
uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name(), round(torch.cuda.mem_get_info()[1] / 1e9), 'GB')" || exit 12

stage gpu-sim-check
# RL rolls out on the GPU sim (configs/rl.yaml: one 32-env physx_cuda env per task, all three
# alive at once, 224 px). Check render + stepping, that identical seeds give identical scenes
# within src/rl/rollout.py's tolerances (exact on CPU; the GPU sim offsets sub-scenes), and that
# the CPU-tuned oracle still succeeds there. Any failure: no RL on this pod.
uv run python - "$CFG" <<'EOF' || exit 13
import sys
import torch
from src.config import load_config
from src.env import make_env, policy_obs
from src.rl.rollout import check_group_reset
cfg = load_config("rl", sys.argv[1])
n, k = int(cfg["rl"]["rollout_envs"]), int(cfg["rl"]["samples_per_prompt"])
envs = {t: make_env(cfg, t, num_envs=n) for t in cfg["env"]["tasks"]}
for task, env in envs.items():
    seeds = [1000 + s for s in range(n // k) for _ in range(k)]  # n // k groups of k
    obs, _ = env.reset(seed=seeds)
    p = policy_obs(obs)
    img, state = p["image"], p["state"]
    size = int(cfg["env"]["image_size"])
    assert tuple(img.shape) == (n, size, size, 3) and img.float().std() > 1, img.shape
    g = img.reshape(n // k, k, -1)
    px = float((g != g[:, :1]).float().mean(-1).max())
    sd = float((state.reshape(n // k, k, -1) - state.reshape(n // k, k, -1)[:, :1]).abs().max())
    frac = check_group_reset(p, k, "warn")
    print(f"[job] {task}: group reset: image values differing <= {px:.5f}, "
          f"state max |diff| {sd:.2e}, check frac {frac:.5f}")
    check_group_reset(p, k, "error")
    for _ in range(5):
        obs, _, _, _, info = env.step(policy_obs(obs)["state"])
    assert info["success"].shape == (n,)
for env in envs.values():
    env.close()
print("[job] GPU sim + render + group reset ok (3 envs alive at once)")
EOF
uv run scripts/wo01_oracle_eval.py --preset sim --device cuda --num-envs 20 --episodes 20 \
  --cells seen --out /workspace/oracle_gpu.json || exit 14
uv run python - <<'EOF' || { echo "[job] oracle < 90% on the GPU sim: no RL here"; exit 15; }
import json
res = json.load(open("/workspace/oracle_gpu.json"))["results"]
rates = {t: c["seen"]["rate"] for t, c in res.items()}
print("[job] oracle on GPU sim:", rates)
raise SystemExit(0 if min(rates.values()) >= 0.9 else 1)
EOF

stage download
# the selected SFT adapter (fails fast on a wrong path); the 15 GB base downloads in the probe
uv run python - <<'EOF' || exit 16
import os
from src.policy.openvla_oft import resolve_adapter
path = resolve_adapter(os.environ["SFT_ADAPTER"])
assert path is not None and (path / "norm_stats.json").is_file(), path
print(f"[job] SFT adapter: {path}: {sorted(p.name for p in path.iterdir())}")
EOF

stage probe
# Probe -> owner approval gate (owner rule, 2026-10-26): the 64-episode/task SFT baseline (the
# same eval as the run's step 0; exits non-zero if it is 0 on every task, CLAUDE.md rule 3) and
# PROBE_STEPS GRPO steps; prints the `[wo03_rl] PROBE` lines. Saves no checkpoint.
run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
  --adapter "$SFT_ADAPTER" --probe-steps "$PROBE_STEPS" || exit 20

stage await-approval
grep "\[wo03_rl\] PROBE" /workspace/job.log | tail -6
echo "[job] pod up $(( ($(date +%s) - $(cat /root/pod_start_epoch)) / 60 )) min so far (setup + checks + probe); soft deadline in $(( $(left) / 3600 )) h"
# The agent posts the PROBE lines; on the owner's explicit go it creates /workspace/APPROVED over
# SSH. No approval within APPROVAL_WAIT_S -> exit (upload + self-delete).
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

stage rl
( while sleep "${UPLOAD_EVERY_S:-7200}"; do upload "periodic (rl running)"; done ) &
UPLOADER=$!
run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
  --adapter "$SFT_ADAPTER"
RC=$?
kill "$UPLOADER" 2>/dev/null; UPLOADER=""
RUN=$(ls -d runs/*-wo04-rl-s"$SEED"-* 2>/dev/null | sort | tail -1)
[ -n "$RUN" ] || { echo "[job] no RL run directory"; exit 30; }
STATUS=$(uv run python -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "$RUN/record.json")
echo "[job] RL run $RUN: status $STATUS (rc=$RC)"
[ -f "$RUN/metrics.jsonl" ] && uv run python - "$RUN" <<'EOF'
import json, sys
for line in open(f"{sys.argv[1]}/metrics.jsonl"):
    r = json.loads(line)
    if "eval/success" in r:
        tasks = {k.split("/")[-1]: round(v, 3) for k, v in r.items() if k.startswith("eval/success/")}
        print(f"[job] eval step {r['step']}: {r['eval/success']:.3f} {tasks} "
              f"({int(r.get('eval/episodes_per_task', 0))} ep/task)")
EOF
[ "$RC" -eq 0 ] || { echo "[job] RL run did not finish (rc=$RC); see $RUN"; exit 31; }
if [ "$STATUS" = stopped_flat ]; then
  # owner decision: a flat run stops at 25% and is diagnosed (frac_groups_dropped, temperature,
  # LoRA lr), not evaluated
  stage stopped-flat
  echo "[job] KILL SWITCH: eval gain at the check step < rl.early_stop_min_improvement; no protocol eval"
  exit 0
fi
[ "$STATUS" = finished ] || { echo "[job] unexpected RL status $STATUS"; exit 32; }
upload "RL run"

stage full-eval
CK=$(ls -d "$RUN"/checkpoints/step_[0-9][0-9][0-9][0-9][0-9][0-9] | sort | tail -1)
echo "[job] final RL checkpoint: $CK"
run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$CK/adapter" \
  --num-envs 50 --sim-device cuda || exit 40
echo "$CK $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" > "$RUN/protocol_eval.txt"
if [ "${EVAL_SFT:-0}" = 1 ]; then
  stage sft-eval
  run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$SFT_ADAPTER" \
    --num-envs 50 --sim-device cuda || exit 41
  echo "$SFT_ADAPTER $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" >> "$RUN/protocol_eval.txt"
fi

stage "done"
