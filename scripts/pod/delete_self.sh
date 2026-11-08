#!/usr/bin/env bash
# Delete THIS RunPod pod. Shared by the pod's hard-cap timer (scripts/pod/pod_start_cmd.sh embeds
# a byte-identical copy; tests/test_pod_backstop.py checks) and the jobs' EXIT traps.
# Usage: delete_self.sh [budget_s]   budget 0 (default) = keep trying until the container dies.
# Each attempt: REST v2 DELETE, then GraphQL podTerminate (the pod-scoped key works on both, not
# on REST v1), every curl time-limited. GraphQL answers HTTP 200 even on failure, so only a body
# with podTerminate and no "errors" counts. An accepted delete is re-sent if the container is still
# alive after DELETE_RECHECK_S (the pod should be gone by then). Never logs the key.
set -u
[ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ] || . /root/runpod.env 2>/dev/null
budget=${1:-0}
mt=${DELETE_MAX_TIME_S:-30}   # per curl call
pause=${DELETE_PAUSE_S:-20}   # between failed attempts
recheck=${DELETE_RECHECK_S:-90}
t0=$(date +%s)
n=0
log() { echo "[delete] $(date -u +%FT%TZ) $*"; }
bounded() { if command -v timeout >/dev/null; then timeout -k 5 $((mt + 15)) "$@"; else "$@"; fi; }
have_curl() {
  command -v curl >/dev/null && return 0
  # nvidia/cuda base images purge curl; the boot apt step normally installs it
  log "curl missing; installing it"
  DEBIAN_FRONTEND=noninteractive bounded apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null 2>&1 \
    || { bounded apt-get update -qq >/dev/null 2>&1; DEBIAN_FRONTEND=noninteractive bounded apt-get install -y -qq --no-install-recommends curl ca-certificates >/dev/null 2>&1; }
  command -v curl >/dev/null
}
attempt() {
  local code body rc
  code=$(bounded curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time "$mt" \
    -X DELETE "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  rc=$?
  log "attempt $n: REST v2 DELETE rc=$rc http=${code:-none}"
  case "$rc:$code" in 0:2??) return 0 ;; esac
  body=$(bounded curl -sS --connect-timeout 10 --max-time "$mt" https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}")
  rc=$?
  log "attempt $n: GraphQL podTerminate rc=$rc body=$(printf %s "$body" | tr -d '\n' | cut -c1-200)"
  [ "$rc" = 0 ] && case "$body" in *'"errors"'*) false ;; *'"podTerminate"'*) true ;; *) false ;; esac
}
if [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
  log "RUNPOD_POD_ID or RUNPOD_API_KEY empty: this pod cannot delete itself; only the Mac-side backstop can"
  exit 2
fi
while :; do
  n=$((n + 1))
  if have_curl && attempt; then
    log "attempt $n: delete accepted"
    [ "$budget" -gt 0 ] && exit 0
    sleep "$recheck"
    log "still alive ${recheck} s after an accepted delete; sending it again"
    continue
  fi
  if [ "$budget" -gt 0 ] && [ $(($(date +%s) - t0)) -ge "$budget" ]; then
    log "gave up after $n attempts in ${budget} s; the pod's hard-cap timer keeps trying"
    exit 1
  fi
  sleep "$pause"
done
