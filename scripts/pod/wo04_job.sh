#!/usr/bin/env bash
# WO-04 on one RunPod A100-80GB pod, ONE SEED PER POD: setup -> GPU-sim check (render, group
# reset, oracle; RL needs the GPU sim) -> SFT adapter download -> PROBE (SFT baseline eval + a few
# GRPO steps: s/step, episodes/h, VRAM, projected h and $ per seed) -> owner approval gate -> full
# GRPO run (rollout-window spend gates, rl.gate_*) -> full-protocol eval of the final RL checkpoint ->
# upload runs/ to the private Hub repo -> terminate this pod.
# Launched over SSH from the repo root on the pod (one seed per pod):
#   SEED=0 SFT_ADAPTER=KudratSingh22/vla-rl-so101-sft/wo02/<pod>/runs/<run>/checkpoints/step_00N000 \
#     nohup bash scripts/pod/wo04_job.sh > /workspace/job.log 2>&1 &
# SFT_ADAPTER: the WO-02-selected checkpoint (its selected_checkpoint.txt), as a Hub path
#   user/repo/path/in/repo[@rev] or a local dir (src/policy/openvla_oft.py resolve_adapter).
# Optional: AUTO_MAX_DROP + AUTO_MAX_USD (owner-pre-approved automatic gate, see stage
#   await-approval), HUB_REPO (results repo), RL_CFG (configs/rl.yaml), PROBE_STEPS (3), APPROVAL_WAIT_S
#   (1800), UPLOAD_EVERY_S (7200), EVAL_SFT=1 (also run the full protocol on the SFT adapter here,
#   for an SFT row on the same GPU sim; needed if the WO-02 full eval ran on the CPU sim).
# Needs /root/runpod.env (RUNPOD_POD_ID, pod-scoped RUNPOD_API_KEY, SPEND_CAP_SECONDS,
# POD_HARD_DEADLINE) and /root/pod_start_epoch, both written by scripts/pod/pod_start_cmd.sh (the
# pod's start command, which also arms the hard self-delete timer; size the cap from the probe's
# projection (+ ~1 h)), and /root/.hf_token. Stages are time-limited and a watchdog stops the job
# at the soft deadline (scripts/pod/job_lib.sh).
set -uo pipefail
# shellcheck source=scripts/pod/job_lib.sh
. "$(dirname "$0")/job_lib.sh"
STAGE_FILE=/workspace/stage
UPLOADER=""
export PATH="$HOME/.local/bin:$PATH"
export HUB_REPO="${HUB_REPO:-KudratSingh22/vla-rl-so101-rl}"

stage() { echo "$1" > "$STAGE_FILE"; echo "[job] $(date -u +%H:%M:%S) stage: $1"; }

upload() {
  timeout -k 30 "$UPLOAD_TIMEOUT_S" uv run --no-sync python - "$1" <<'EOF' || echo "[job] upload failed or timed out"
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

# installed before anything can fail: every exit uploads (time-limited), then self-deletes
finish() {
  local rc=$?
  stop_watchdog
  [ -n "$UPLOADER" ] && kill "$UPLOADER" 2>/dev/null
  echo "[job] exit rc=$rc at stage $(cat "$STAGE_FILE" 2>/dev/null)"
  upload "final (rc=$rc)"
  delete_pod
}
trap finish EXIT
pod_env || exit 9
start_watchdog
# Containers see every host core (128-256 on some A100 hosts) but get a cgroup CPU quota
# (~14-27); size torch/OpenMP thread pools to the quota or CPU-side work (IK, data loading)
# crawls. cgroup v2 exposes cpu.max ("max" = none); v1 hosts (pod gzxhwhjhdtc1kc) cfs_quota_us.
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
: "${SEED:?set SEED (one pod per seed: 0, 1 or 2)}"
SFT_ADAPTER="${SFT_ADAPTER:-KudratSingh22/vla-rl-so101-sft/wo02/u87r9sjnwig58d/runs/20261027T073615Z-wo02-sim-4b4b/checkpoints/step_004000}"  # WO-02 base (owner 2026-10-27: a4 step 4000, seen 18/16/6%)
export SEED SFT_ADAPTER
CFG="${RL_CFG:-configs/rl.yaml}"
PROBE_STEPS="${PROBE_STEPS:-3}"

stage setup
run_capped 900 apt-get install -y -qq --no-install-recommends \
  ffmpeg libvulkan1 git >/dev/null || exit 10
echo "[job] WO-04 seed $SEED, SFT adapter $SFT_ADAPTER, config $CFG, commit $(git rev-parse --short HEAD)"
# Vulkan for SAPIEN rendering. RunPod mounts an ICD at /etc/vulkan/icd.d pointing at
# libGLX_nvidia.so.0, which cannot create an instance headless (driver 595.91.07, pod
# iqrmjox6ws0e66); NVIDIA's EGL library exposes the same Vulkan driver and works. SAPIEN only
# checks /usr/share/vulkan/icd.d/nvidia_icd.json, and otherwise substitutes its own GLX-based file.
mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version":"1.0.1","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.4.329"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
command -v uv >/dev/null || run_capped 300 sh -c 'curl -LsSf --max-time 120 https://astral.sh/uv/install.sh | sh'
run_capped 1200 uv sync --frozen --extra gpu --extra dev || exit 11
run_capped 300 uv run python -c "import torch; assert torch.cuda.is_available(); print('[job] gpu', torch.cuda.get_device_name(), round(torch.cuda.mem_get_info()[1] / 1e9), 'GB')" || exit 12

stage gpu-sim-check
# RL rolls out on the GPU sim (configs/rl.yaml: one 32-env physx_cuda env per task, all three
# alive at once, 224 px). Check render + stepping, that identical seeds give identical scenes
# within src/rl/rollout.py's tolerances (exact on CPU; the GPU sim offsets sub-scenes), and that
# the CPU-tuned oracle still succeeds there. Any failure: no RL on this pod.
run_capped 1200 uv run python - "$CFG" <<'EOF' || exit 13
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
run_until_deadline uv run scripts/wo01_oracle_eval.py --preset sim --device cuda --num-envs 20 --episodes 20 \
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
run_capped 900 uv run python - <<'EOF' || exit 16
import os
from src.policy.openvla_oft import resolve_adapter
path = resolve_adapter(os.environ["SFT_ADAPTER"])
assert path is not None and (path / "norm_stats.json").is_file(), path
print(f"[job] SFT adapter: {path}: {sorted(p.name for p in path.iterdir())}")
EOF

stage probe
# Probe -> owner approval gate (owner rule, 2026-10-26): the SFT baseline (rl.eval_episodes greedy
# episodes per trained task, a sanity check; the same eval as the run's step 0; exits non-zero if it is 0 on every task, CLAUDE.md rule 3) and
# PROBE_STEPS GRPO steps; prints the `[wo03_rl] PROBE` lines. Saves no checkpoint.
run_until_deadline uv run scripts/wo03_rl.py --preset rl --config "$CFG" --seed "$SEED" \
  --adapter "$SFT_ADAPTER" --probe-steps "$PROBE_STEPS" || exit 20

stage await-approval
grep "\[wo03_rl\] PROBE" /workspace/job.log | tail -6
echo "[job] pod up $(( ($(date +%s) - $(cat /root/pod_start_epoch)) / 60 )) min so far (setup + checks + probe); soft deadline in $(( $(left) / 3600 )) h"
# The agent posts the PROBE lines; on the owner's explicit go it creates /workspace/APPROVED over
# SSH. No approval within APPROVAL_WAIT_S -> exit (upload + self-delete).
# Owner pre-approval (2026-10-27, WO-04 seeds): pass the gate automatically when the probe drops
# <= AUTO_MAX_DROP of groups AND projects <= AUTO_MAX_USD for the full run; otherwise stop here
# (upload + self-delete) and the agent reports. Unset -> the manual gate below.
if [ -n "${AUTO_MAX_DROP:-}" ] && [ -n "${AUTO_MAX_USD:-}" ]; then
  PROBE_RUN=$(ls -d runs/*-wo04probe-rl-s"$SEED"-* 2>/dev/null | sort | tail -1)
  if [ -n "$PROBE_RUN" ] && uv run python - "$PROBE_RUN/record.json" "$AUTO_MAX_DROP" "$AUTO_MAX_USD" <<'EOF'; then
import json, sys
r = json.load(open(sys.argv[1]))
drop, usd = float(r["probe"]["frac_groups_dropped"]), float(r["projection"]["run_usd"])
max_drop, max_usd = float(sys.argv[2]), float(sys.argv[3])
ok = drop <= max_drop and usd <= max_usd
print(f"[job] auto-gate: groups dropped {drop:.1%} (max {max_drop:.0%}), projected full run "
      f"${usd:.2f} per seed (max ${max_usd:.2f}) -> {'PASS' if ok else 'STOP'}")
raise SystemExit(0 if ok else 3)
EOF
    echo "owner pre-approved 2026-10-27: auto-gate passed (drop <= $AUTO_MAX_DROP, projected <= \$$AUTO_MAX_USD)" \
      > /workspace/APPROVED
  else
    echo "[job] auto-gate: conditions not met (or no probe record); stopping for the owner"
    exit 22
  fi
fi
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
  echo "[job] FLAT GATE: windowed rollout success gain at the check step < rl.gate_min_gain; no protocol eval"
  exit 0
fi
# stopped_peak (decline or entropy gate, rl.gate_max_drop / rl.gate_entropy_min; record.stop_reason):
# evaluated like a finished run
[ "$STATUS" = finished ] || [ "$STATUS" = stopped_peak ] || { echo "[job] unexpected RL status $STATUS"; exit 32; }
upload "RL run"

stage full-eval
CK=$(ls -d "$RUN"/checkpoints/step_[0-9][0-9][0-9][0-9][0-9][0-9] | sort | tail -1)
echo "[job] final RL checkpoint: $CK"
run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$CK/adapter" \
  --num-envs 50 --sim-device cuda || exit 40
echo "$CK $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" > "$RUN/protocol_eval.txt"
# owner 2026-10-28: also score the best mid-run checkpoint (highest in-training trained-task eval;
# earliest on a tie) when it is not the final one
BEST=$(uv run python - "$RUN" <<'EOF'
import json, os, sys
run = sys.argv[1]
evals = [json.loads(l) for l in open(f"{run}/metrics.jsonl")]
evals = [(r["step"], r["eval/success_train"]) for r in evals
         if "eval/success_train" in r and r["step"] > 0
         and os.path.isdir(f"{run}/checkpoints/step_{r['step']:06d}")]
if evals:
    step, val = max(evals, key=lambda e: (e[1], -e[0]))
    print(f"{run}/checkpoints/step_{step:06d}")
EOF
)
if [ -n "$BEST" ] && [ "$BEST" != "$CK" ]; then
  stage best-eval
  echo "[job] best mid-run checkpoint: $BEST"
  run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$BEST/adapter" \
    --num-envs 50 --sim-device cuda || exit 42
  echo "$BEST $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" >> "$RUN/protocol_eval.txt"
fi
if [ "${EVAL_SFT:-0}" = 1 ]; then
  stage sft-eval
  run_until_deadline uv run scripts/wo02_eval.py --preset sim --config "$CFG" --adapter "$SFT_ADAPTER" \
    --num-envs 50 --sim-device cuda || exit 41
  echo "$SFT_ADAPTER $(ls -d runs/*-wo02eval-sim-* | sort | tail -1)" >> "$RUN/protocol_eval.txt"
fi

stage "done"
