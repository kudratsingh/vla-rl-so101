# Sourced by the pod job scripts (diag_job.sh, demo_rebuild_job.sh, wo04_job.sh): time budget,
# time-limited stages, soft-deadline watchdog and self-delete. Not executable on its own.
#
# Budget. H = the pod's hard cap: POD_HARD_DEADLINE written by scripts/pod/pod_start_cmd.sh (first
# boot + SPEND_CAP_SECONDS - CAP_PRESTART_S); pods started by an older command fall back to
# pod_start_epoch + SPEND_CAP_SECONDS. Soft deadline D = H - SOFT_MARGIN_S (1200 s).
#   - every stage is time-limited to min(its own limit, D) with `timeout -k KILL_AFTER_S` (120 s):
#     a child that ignores SIGTERM is SIGKILLed instead of blocking the job;
#   - a watchdog sends SIGTERM to the job's bash at D + KILL_AFTER_S + 10, which runs the EXIT
#     trap at once (bash runs the EXIT trap on SIGTERM, even with a foreground child running),
#     covering anything that is not time-limited or a `timeout` that itself hung;
#   - the trap's upload is limited to UPLOAD_TIMEOUT_S (600 s) + 30 and its delete loop to 180 s
#     (+ one in-flight attempt, <= ~90 s).
#   130 + 630 + 270 = 1030 s < 1200 s, so the job's own delete normally lands before H; the hard-cap
#   timer (pod_start_cmd.sh) is the backstop and SIGKILL / OOM of the job's bash leaves only it.
# shellcheck shell=bash
POD_LIB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOFT_MARGIN_S=${SOFT_MARGIN_S:-1200}
UPLOAD_TIMEOUT_S=${UPLOAD_TIMEOUT_S:-600}
KILL_AFTER_S=${KILL_AFTER_S:-120}
WATCHDOG=""
export DEBIAN_FRONTEND=noninteractive

is_int() { case "${1:-}" in '' | *[!0-9]*) return 1 ;; esac; }

# source /root/runpod.env and set DEADLINE (the soft deadline); non-zero = do not start
pod_env() {
  # shellcheck disable=SC1091
  . /root/runpod.env || { echo "[job] /root/runpod.env missing"; return 9; }
  is_int "${SPEND_CAP_SECONDS:-}" || { echo "[job] SPEND_CAP_SECONDS '${SPEND_CAP_SECONDS:-}' is not an integer"; return 9; }
  local start hard
  start=$(cat /root/pod_start_epoch 2>/dev/null)
  is_int "$start" || { echo "[job] /root/pod_start_epoch missing or not an integer"; return 9; }
  hard=${POD_HARD_DEADLINE:-$((start + SPEND_CAP_SECONDS))}
  is_int "$hard" || { echo "[job] POD_HARD_DEADLINE '$hard' is not an integer"; return 9; }
  DEADLINE=$((hard - SOFT_MARGIN_S))
  export RUNPOD_POD_ID
  echo "[job] hard cap in $((hard - $(date +%s))) s, soft deadline in $((DEADLINE - $(date +%s))) s" \
    "(boot ${POD_BOOT_N:-?}; start command $([ -n "${POD_HARD_DEADLINE:-}" ] && echo pod_start_cmd.sh || echo legacy))"
}

left() { echo $((DEADLINE - $(date +%s))); }

# run_capped LIMIT_S CMD...: CMD time-limited to min(LIMIT_S, time to the soft deadline); 0 = no
# own limit. rc 124 = timed out (or no time left), 137 = killed after ignoring SIGTERM.
run_capped() {
  local lim=$1 s
  shift
  s=$(left)
  if [ "$s" -le 60 ]; then
    echo "[job] soft deadline reached before: $*"
    return 124
  fi
  if [ "$lim" -gt 0 ] && [ "$lim" -lt "$s" ]; then s=$lim; fi
  timeout -k "$KILL_AFTER_S" "$s" "$@"
}
# SIGTERM: wo03_rl.py finishes its run record (status interrupted) / saves an interrupted checkpoint
run_until_deadline() { run_capped 0 "$@"; }

start_watchdog() {
  local job=$$ s
  s=$(($(left) + KILL_AFTER_S + 10))
  [ "$s" -gt 0 ] || s=0
  (
    sp=""
    trap '[ -n "$sp" ] && kill "$sp" 2>/dev/null; exit 0' TERM
    sleep "$s" &
    sp=$!
    wait "$sp"
    echo "[job] $(date -u +%H:%M:%S) soft deadline passed: watchdog stops the job (EXIT trap: upload + delete)"
    kill -TERM "$job"
  ) &
  WATCHDOG=$!
}
stop_watchdog() {
  if [ -n "$WATCHDOG" ]; then kill "$WATCHDOG" 2>/dev/null; fi
  WATCHDOG=""
}

# the job's own self-delete: at most ~180 s of attempts, then the hard-cap timer keeps trying
delete_pod() {
  echo "[job] terminating pod ${RUNPOD_POD_ID:-?}"
  DELETE_RECHECK_S=0 bash "$POD_LIB_DIR/delete_self.sh" 180
}
