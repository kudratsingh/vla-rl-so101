#!/usr/bin/env bash
# Start the outside-the-pod watchdog (scripts/pod/watchdog.py) on the owner's Mac, detached, under
# caffeinate. Run it right after the pod create call (docs/runbooks/pod-launch.md):
#   bash scripts/pod/watchdog_mac.sh <pod_id> <deadline_epoch> [cap_s] [lead_s]
# deadline_epoch = the epoch taken just BEFORE the create call + SPEND_CAP_SECONDS (absolute).
#
# Key: read from the macOS Keychain, item "vla-rl-so101-runpod", and exported only into the
# watchdog's environment (never argv, never a file, never printed). One-time setup by the owner
# (prompts for the key, so it is not typed on the command line or kept in shell history):
#   security add-generic-password -a "$USER" -s vla-rl-so101-runpod -w
# Check it exists (prints no key):  security find-generic-password -s vla-rl-so101-runpod >/dev/null && echo ok
# Optional env: WATCHDOG_DIR (log dir, default runs/watchdog), WATCHDOG_NAME_PREFIX (vla-rl-so101),
# WATCHDOG_CONFIRM_WAIT_S (150). Exit 0 only once the watchdog logged "confirmed" (it sees the pod
# under the expected name); 3 no Keychain item; 4 watchdog exited (wrong id, refused key, name
# mismatch); 5 still unconfirmed after the wait (it keeps running, but the layer is not armed).
#
# Launch gate: each run gets a fresh tag (--run-tag, 16 hex chars) that the watchdog writes into
# every log line, and this script writes <dir>/<pod>.mac.env (pod, pid, tag, start, deadline, log)
# BEFORE it waits. scripts/pod/launch_gate.py reads it and counts only a "confirmed" line with this
# pod and this tag, written after this start, while that pid is alive and still runs
# `watchdog.py --pod-id <pod> ... --run-tag <tag>`: a line from an earlier run or another pod never
# passes the gate.
#
# Limits, honestly: `caffeinate -i -s` keeps the Mac from idle/system sleep while the watchdog
# runs, but -s only holds on AC power, and closing the lid on battery sleeps the Mac anyway; a
# sleeping, dead or offline laptop runs no watchdog. That is why the GitHub Actions watchdog
# (.github/workflows/pod-watchdog.yml) runs as well; this one is the second, local copy.
set -euo pipefail
pod=${1:-}
deadline=${2:-}
cap=${3:-}
lead=${4:-400}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
case "$pod" in '' | *[!A-Za-z0-9_-]*) echo "usage: $0 <pod_id> <deadline_epoch> [cap_s] [lead_s]" >&2; exit 2 ;; esac
case "$deadline" in '' | *[!0-9]* | 0*) echo "deadline_epoch must be an integer epoch (s)" >&2; exit 2 ;; esac
case "$cap" in *[!0-9]* | 0*) echo "cap_s must be an integer" >&2; exit 2 ;; esac
case "$lead" in '' | *[!0-9]* | 0?*) echo "lead_s must be an integer" >&2; exit 2 ;; esac
now=$(date +%s)
if [ "$deadline" -le "$now" ]; then echo "warning: deadline is already past; the watchdog terminates at once" >&2; fi
if ! key=$(security find-generic-password -s vla-rl-so101-runpod -w 2>/dev/null) || [ -z "$key" ]; then
  echo "no Keychain item vla-rl-so101-runpod; one-time: security add-generic-password -a \"\$USER\" -s vla-rl-so101-runpod -w" >&2
  exit 3
fi
py="$repo/.venv/bin/python"
[ -x "$py" ] || py=$(command -v python3)
dir=${WATCHDOG_DIR:-$repo/runs/watchdog}   # runs/ is git-ignored
mkdir -p "$dir"
log="$dir/$pod.jsonl"
out="$dir/$pod.out"
state="$dir/$pod.mac.env"
skip=0
if [ -f "$log" ]; then skip=$(awk 'END { print NR }' "$log"); fi
tag=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')
case "$tag" in ???????????????? ) ;; *) echo "could not read a run tag from /dev/urandom" >&2; exit 2 ;; esac
# pods are named vla-rl-so101-<purpose> (runbook); a different name means a wrong pod id: no delete
args=(--pod-id "$pod" --deadline-epoch "$deadline" --lead-s "$lead" --log "$log"
  --expect-name-prefix "${WATCHDOG_NAME_PREFIX:-vla-rl-so101}" --run-tag "$tag")
[ -n "$cap" ] && args+=(--cap-s "$cap")
# export inside a subshell: the key is in the child's environment only, not in any argv (ps)
(
  export RUNPOD_API_KEY="$key"
  exec nohup caffeinate -i -s "$py" "$repo/scripts/pod/watchdog.py" "${args[@]}" > "$out" 2>&1
) &
pid=$!
unset key
# the launch gate's record of THIS run (no key in it); written atomically
printf 'pod=%s\npid=%s\ntag=%s\nstart=%s\ndeadline=%s\nlog=%s\n' \
  "$pod" "$pid" "$tag" "$now" "$deadline" "$log" > "$state.tmp.$$" && mv -f "$state.tmp.$$" "$state"
# Armed only once the watchdog has CONFIRMED this key sees the pod under the expected name (a wrong
# id or a refused key would otherwise watch nothing until the fire time). Only lines written by this
# run count: the log is appended per pod id.
wait_s=${WATCHDOG_CONFIRM_WAIT_S:-150}
case "$wait_s" in '' | *[!0-9]* | 0?*) wait_s=150 ;; esac
t0=$(date +%s)
while :; do
  if awk -v s="$skip" -v t="\"tag\": \"$tag\"" 'NR > s && /"event": "confirmed"/ && index($0, t) { f = 1 } END { exit !f }' "$log" 2>/dev/null; then
    echo "watchdog CONFIRMED the pod; pid $pid (caffeinate -i -s), pod $pod, deadline $(date -r "$deadline" -u +%FT%TZ) (in $((deadline - now)) s), lead $lead s"
    echo "run tag: $tag; gate state: $state"
    echo "log: $log"
    echo "out: $out"
    echo "stop by hand (only once the pod is verified gone, or on the owner's word): pkill -f 'watchdog.py --pod-id $pod'"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "watchdog exited before confirming the pod (wrong id, refused key, name mismatch?); see $out" >&2
    tail -5 "$out" >&2 || true
    exit 4
  fi
  if [ $(($(date +%s) - t0)) -ge "$wait_s" ]; then
    echo "watchdog (pid $pid) has NOT confirmed the pod after $wait_s s; it keeps running but will not" \
      "delete an unconfirmed pod. Runbook: treat the Mac layer as down; check $log" >&2
    exit 5
  fi
  sleep 2
done
