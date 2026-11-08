# Sourced by the pod job scripts: the owner-approval gate (CLAUDE.md rule 1). Not executable.
#
# A job passes its gate only on an approval file that is
#   - SCOPED TO THIS POD: $APPROVAL_DIR/APPROVED.<RUNPOD_POD_ID> (APPROVAL_DIR = /workspace), and
#   - NEWER THAN THE POD: its mtime is after /root/pod_start_epoch (the pod's first boot, written
#     by scripts/pod/pod_start_cmd.sh),
# so a file left on a reused network volume by an earlier pod, or the old unscoped
# /workspace/APPROVED, never counts (owner 2026-11-08). Ignored files are listed in the job log.
# A container restart of the SAME pod keeps its first-boot epoch, so an approval given on this pod
# before the restart still counts (the spend cap is per pod too).
#
# The agent approves on the owner's explicit go (or before launch, for a pre-approved pass), over
# SSH on the pod, from the repo root:
#   bash -c '. /root/runpod.env; . scripts/pod/approval_lib.sh; approval_write "owner go <date>: <what>"'
# shellcheck shell=bash
APPROVAL_DIR=${APPROVAL_DIR:-/workspace}
POD_START_EPOCH_FILE=${POD_START_EPOCH_FILE:-/root/pod_start_epoch}
APPROVAL_POLL_S=${APPROVAL_POLL_S:-15}

approval_file() { echo "$APPROVAL_DIR/APPROVED.${RUNPOD_POD_ID:-}"; }

# mtime in epoch seconds: GNU stat on the pod, BSD stat on a Mac (tests)
_approval_mtime() { stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null; }

_approval_start() {
  local s
  s=$(cat "$POD_START_EPOCH_FILE" 2>/dev/null)
  case "$s" in '' | *[!0-9]*) return 1 ;; esac
  echo "$s"
}

# approval_why FILE: "" if FILE is a valid approval for this pod, else the reason it is not
approval_why() {
  local f=$1 start m
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "RUNPOD_POD_ID unset"; return; }
  [ "$f" = "$(approval_file)" ] || { echo "not this pod's file ($(approval_file))"; return; }
  [ -f "$f" ] || { echo "missing"; return; }
  start=$(_approval_start) || { echo "$POD_START_EPOCH_FILE missing or not an integer"; return; }
  m=$(_approval_mtime "$f")
  case "$m" in '' | *[!0-9]*) echo "mtime unreadable"; return ;; esac
  [ "$m" -gt "$start" ] || echo "stale: written $((start - m)) s before this pod first booted"
}

# 0 iff this pod's approval file exists and was written after the pod's first boot
approval_valid() { [ -z "$(approval_why "$(approval_file)")" ]; }

# log every APPROVED* file in APPROVAL_DIR that does not count, and why
approval_report_ignored() {
  local f why
  for f in "$APPROVAL_DIR"/APPROVED "$APPROVAL_DIR"/APPROVED.*; do
    [ -e "$f" ] || continue
    why=$(approval_why "$f")
    [ -n "$why" ] && echo "[job] approval gate IGNORES $f: $why"
  done
  return 0
}

# approval_write MSG: write this pod's approval file (the agent, on the owner's go)
approval_write() {
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "[approval] RUNPOD_POD_ID unset: source /root/runpod.env" >&2; return 1; }
  local f tmp
  f=$(approval_file)
  tmp="$f.tmp.$$"
  printf '%s (pod %s, %s)\n' "$1" "$RUNPOD_POD_ID" "$(date -u +%FT%TZ)" > "$tmp" && mv -f "$tmp" "$f"
}

# await_approval WAIT_S: poll until approval_valid; rc 21 after WAIT_S seconds without one
await_approval() {
  local wait=$1 t0
  approval_report_ignored
  t0=$(date +%s)
  until approval_valid; do
    if [ $(($(date +%s) - t0)) -ge "$wait" ]; then
      echo "[job] no owner approval ($(approval_file), newer than the pod start) within $((wait / 60)) min; stopping"
      return 21
    fi
    sleep "$APPROVAL_POLL_S"
  done
  echo "[job] approved: $(cat "$(approval_file)")"
}
