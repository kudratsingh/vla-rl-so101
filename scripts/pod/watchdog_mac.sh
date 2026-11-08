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
# Optional env: WATCHDOG_DIR (log dir, default runs/watchdog), WATCHDOG_NAME_PREFIX (vla-rl-so101).
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
case "$deadline" in '' | *[!0-9]*) echo "deadline_epoch must be an integer epoch (s)" >&2; exit 2 ;; esac
case "$cap" in *[!0-9]*) echo "cap_s must be an integer" >&2; exit 2 ;; esac
case "$lead" in '' | *[!0-9]*) echo "lead_s must be an integer" >&2; exit 2 ;; esac
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
# pods are named vla-rl-so101-<purpose> (runbook); a different name means a wrong pod id: no delete
args=(--pod-id "$pod" --deadline-epoch "$deadline" --lead-s "$lead" --log "$log"
  --expect-name-prefix "${WATCHDOG_NAME_PREFIX:-vla-rl-so101}")
[ -n "$cap" ] && args+=(--cap-s "$cap")
# export inside a subshell: the key is in the child's environment only, not in any argv (ps)
(
  export RUNPOD_API_KEY="$key"
  exec nohup caffeinate -i -s "$py" "$repo/scripts/pod/watchdog.py" "${args[@]}" > "$out" 2>&1
) &
pid=$!
unset key
sleep 1
if kill -0 "$pid" 2>/dev/null; then
  echo "watchdog pid $pid (caffeinate -i -s), pod $pod, deadline $(date -r "$deadline" -u +%FT%TZ) (in $((deadline - now)) s), lead $lead s"
  echo "log: $log"
  echo "out: $out"
  echo "stop by hand (only once the pod is verified gone, or on the owner's word): pkill -f 'watchdog.py --pod-id $pod'"
else
  echo "watchdog exited at once; see $out" >&2
  tail -5 "$out" >&2 || true
  exit 4
fi
