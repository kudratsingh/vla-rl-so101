# Container start command for every project pod (entrypoint ["/bin/bash","-c"]): pass this WHOLE
# file as the one start-command string, not a path (the repo is not on the pod at boot). Pod env:
# SPEND_CAP_SECONDS (integer s), PUBLIC_KEY, POD_DEADLINE_EPOCH (integer epoch s: the launcher's
# epoch taken just before the create call + SPEND_CAP_SECONDS), optional CAP_PRESTART_S (180) and
# POD_DEADLINE_LEAD_S (480); RunPod injects RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY.
# docs/STATUS.md "Pod deletion backstop", docs/runbooks/pod-launch.md.
# 1. Hard-cap timer FIRST, before apt: it fires at min(first boot + SPEND_CAP_SECONDS -
#    CAP_PRESTART_S, POD_DEADLINE_EPOCH - POD_DEADLINE_LEAD_S). CAP_PRESTART_S is the allowance for
#    billed time before the container starts (image pull; estimated); POD_DEADLINE_EPOCH counts from
#    creation, so it needs no estimate. The lead (480 s) is longer than the outside watchdog's
#    (scripts/pod/watchdog.py, 400 s), so the pod normally deletes itself first.
#    A POD_DEADLINE_EPOCH that is set but not an integer epoch in seconds deletes the pod at once.
#    The first-boot epoch lives on /workspace, so a container restart re-arms the timer with only
#    the time left instead of a fresh cap. A non-integer cap deletes the pod at once. Numbers with
#    a leading zero are refused too: bash arithmetic reads them as octal, and "08"/"09" would abort
#    this whole script before the timer is armed (cap/deadline: delete now; the others: default).
# 2. The timer runs /root/delete_self.sh (a copy of scripts/pod/delete_self.sh): REST v2 DELETE
#    then GraphQL podTerminate, every curl time-limited, repeated until the container dies.
# 3. apt (sshd, curl) under timeout; then a key check -> /root/backstop_status; then sshd.
# Logs: /root/boot.log and the container log. The job scripts read /root/runpod.env.
L=/root/boot.log
mkdir -p /root /workspace
log() { echo "[boot] $* ($(date -u +%FT%TZ))" | tee -a "$L"; }
cap=${SPEND_CAP_SECONDS:-}
pre=${CAP_PRESTART_S:-180}
case "$pre" in '' | *[!0-9]* | 0?*) pre=180 ;; esac
case "$cap" in '' | *[!0-9]* | 0?*) log "SPEND_CAP_SECONDS '$cap' is not a decimal integer: deleting the pod now"; cap=0; pre=0 ;; esac
F=/workspace/.first_boot_epoch.${RUNPOD_POD_ID:-nopod}  # per pod: a reused network volume must not carry an old epoch
[ -s "$F" ] || date +%s > "$F"
first=$(cat "$F")
case "$first" in '' | *[!0-9]* | 0*) first=$(date +%s) ;; esac
boots=$(($(cat "$F.boots" 2>/dev/null || echo 0) + 1))
echo "$boots" > "$F.boots"
hard=$((first + cap - pre))
dl=${POD_DEADLINE_EPOCH:-}
dlead=${POD_DEADLINE_LEAD_S:-480}
case "$dlead" in '' | *[!0-9]* | 0?*) dlead=480 ;; esac
src="first boot $first + cap $cap - prestart $pre"
from=first_boot
case "$dl" in
  '') ;;
  *[!0-9]* | 0* | ????????????*)  # not an integer, leading 0 (octal), > 11 digits (ms): fail safe
    log "POD_DEADLINE_EPOCH '$dl' is not an epoch in seconds: deleting the pod now"
    hard=0; src="bad POD_DEADLINE_EPOCH"; from=bad ;;
  *) if [ $((dl - dlead)) -lt "$hard" ]; then hard=$((dl - dlead)); src="creation deadline $dl - lead $dlead"; from=creation; fi ;;
esac
umask 077
{
  printf 'export RUNPOD_POD_ID=%q\n' "${RUNPOD_POD_ID:-}"
  printf 'export RUNPOD_API_KEY=%q\n' "${RUNPOD_API_KEY:-}"
  printf 'export SPEND_CAP_SECONDS=%q\n' "${SPEND_CAP_SECONDS:-}"
  printf 'export POD_HARD_DEADLINE=%s\nexport POD_BOOT_N=%s\n' "$hard" "$boots"
  printf 'export POD_DEADLINE_EPOCH=%q\n' "$dl"
} > /root/runpod.env
echo "$first" > /root/pod_start_epoch
cat > /root/delete_self.sh <<'DELETE_SELF_EOF'
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
  log "RUNPOD_POD_ID or RUNPOD_API_KEY empty: this pod cannot delete itself; only the outside watchdog (scripts/pod/watchdog.py) can"
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
DELETE_SELF_EOF
umask 022
rem=$((hard - $(date +%s)))
[ "$rem" -gt 0 ] || rem=0
log "boot $boots; hard cap: delete in $rem s (epoch $hard = $src)"
(
  sleep "$rem"
  log "spend cap reached; deleting pod"
  bash /root/delete_self.sh 0 2>&1 | tee -a "$L"
) &
echo $! > /root/cap_timer.pid
log "installing sshd, curl"
export DEBIAN_FRONTEND=noninteractive
if timeout -k 30 600 apt-get update -qq \
  && timeout -k 30 900 apt-get install -y -qq --no-install-recommends openssh-server curl ca-certificates git > /dev/null; then
  log "apt ok"
else
  log "apt FAILED or timed out; the cap timer installs curl itself if it has to"
fi
if ! command -v curl > /dev/null; then
  st=NO_CURL
elif [ -z "${RUNPOD_POD_ID:-}" ] || [ -z "${RUNPOD_API_KEY:-}" ]; then
  st=KEY_MISSING
else
  # timeout(1) so a curl that ignores --max-time cannot block sshd (and the job) for the whole cap
  code=$(timeout -k 5 30 curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time 20 \
    "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY")
  case "$code" in 2??) st=KEY_OK ;; 401) st=KEY_BAD ;; *) st=KEY_UNVERIFIED ;; esac
  st="$st http=$code"
fi
kill -0 "$(cat /root/cap_timer.pid)" 2> /dev/null && st="$st timer=armed" || st="$st timer=DEAD"
st="$st cap_from=$from"
echo "$st" > /root/backstop_status
log "backstop: $st"
mkdir -p /root/.ssh /run/sshd
echo "${PUBLIC_KEY:-}" > /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
if /usr/sbin/sshd; then log "sshd up"; else log "sshd FAILED"; fi
sleep infinity
