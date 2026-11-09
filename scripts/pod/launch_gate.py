#!/usr/bin/env python3
"""Launch gate on the Mac: a pod job starts only after BOTH outside watchdogs confirmed THIS pod.

Owner request 2026-11-08: capture the GitHub watchdog's run id at dispatch and verify that it
belongs to the pod just created; start the job only once the GitHub AND the Mac watchdog confirmed
that exact pod; if either has not confirmed by the deadline, abort and clean up the verified pod.
Runbook: docs/runbooks/pod-launch.md. Python stdlib only; every external command (gh, ps, ssh,
security, the cleanup watchdog) goes through one injectable runner, so the tests use fakes.

  dispatch  Dispatch .github/workflows/pod-watchdog.yml on main with a fresh nonce (dispatch_id).
            The run id comes from the dispatch response (`workflow_run_id`, GitHub REST "Create a
            workflow dispatch event", 200); if the response has none, it is resolved by listing
            this workflow's workflow_dispatch runs on main created at/after the dispatch whose
            run-name is exactly "pod-watchdog pod=<pod> dispatch=<nonce>": exactly one match, else
            exit 4. Either way the run is then verified (title, event, branch main, workflow file,
            created at/after the dispatch - SKEW_S) and <dir>/<pod>.github.json is written.
  gate      Poll until both layers confirmed this pod, or fail:
              GitHub: the run with the captured id, re-verified as above, is in progress, its step
                "Confirm pod <pod>" concluded success no later than --confirm-by (+ SKEW_S), and its
                "Watchdog" step is running;
              Mac: <dir>/<pod>.mac.env (written by scripts/pod/watchdog_mac.sh) names this pod and
                deadline; its log holds a "confirmed" line for this pod with this run's tag, written
                after this run's start and no later than --confirm-by; that pid is alive and runs
                `watchdog.py --pod-id <pod> ... --run-tag <tag>`.
            Confirmation must happen by --confirm-by (default T_req + 300 s); a gate started later
            still passes if both confirmed in time and are still running. Fails at once if a layer
            reports failure (step failed/skipped, run finished, process gone, run not this pod's).
            With --ssh-host it then writes the job's token /workspace/WATCHDOGS_OK.<pod> over SSH
            (only after re-checking both layers, and only if the pod's /root/runpod.env names this
            pod); the job scripts refuse to start without it (scripts/pod/approval_lib.sh,
            require_watchdogs, rc 8). Run it once without --ssh-host right after the dispatch, and
            again with --ssh-host once the pod's SSH address is known.
            On failure: abort (no token) and clean up: `watchdog.py --terminate-now` with the
            Keychain key (same name-prefix and two-source "gone" checks as the watchdog), and say
            what to check by hand.

Exit codes: 0 dispatched and verified / gate passed (token written with --ssh-host); 2 bad input;
3 dispatch failed (no run); 4 run id not resolved or not this pod's run; 10 gate failed and the
cleanup verified the pod gone; 11 gate failed and the cleanup did NOT verify it gone (delete it
through the connector now and confirm it is gone); 12 gate failed, --no-cleanup given.
Every check is logged as JSONL to <dir>/<pod>.gate.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_REPO = "kudratsingh/vla-rl-so101"
WORKFLOW = "pod-watchdog.yml"
WORKFLOW_PATH = ".github/workflows/pod-watchdog.yml"
BRANCH = "main"
NAME_PREFIX = "vla-rl-so101"
KEYCHAIN_ITEM = "vla-rl-so101-runpod"
POD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")
NONCE_RE = re.compile(r"^[0-9a-f]{16,64}$")
TAG_RE = re.compile(r"^[0-9a-f]{16}$")
SKEW_S = 30.0  # Mac clock vs GitHub timestamps

# (argv, stdin, extra env, timeout_s) -> (rc, stdout, stderr)
Runner = Callable[[list[str], "str | None", "dict[str, str] | None", float], tuple[int, str, str]]


def run_cmd(
    argv: list[str],
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60,
) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            argv,
            input=stdin,
            env={**os.environ, **env} if env else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, "", f"{type(e).__name__}: {e}"
    return p.returncode, p.stdout, p.stderr


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(s: Any) -> float | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def expected_title(pod: str, nonce: str) -> str:
    return f"pod-watchdog pod={pod} dispatch={nonce}"


@dataclass
class Ctx:
    pod: str
    dir: Path
    repo: str = GH_REPO
    run: Runner = run_cmd
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    out: Callable[[str], None] = field(default=lambda s: print(s, flush=True))

    @property
    def gh_state(self) -> Path:
        return self.dir / f"{self.pod}.github.json"

    @property
    def mac_state(self) -> Path:
        return self.dir / f"{self.pod}.mac.env"

    def log(self, event: str, **kw: Any) -> None:
        t = self.now()
        rec = {"t": iso(t), "epoch": round(t, 3), "pod": self.pod, "event": event, **kw}
        line = json.dumps(rec, default=str)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / f"{self.pod}.gate.jsonl", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            self.out(f"[gate] log write failed: {e}")
        self.out(f"[gate] {line}")

    def gh(self, *args: str, stdin: str | None = None) -> tuple[int, Any, str]:
        rc, so, se = self.run(["gh", "api", *args], stdin, None, 30.0)
        try:
            j = json.loads(so) if so.strip() else None
        except ValueError:
            j = None
        return rc, j, (se or so)[:300]


# ---- GitHub run ----------------------------------------------------------------------------------
def verify_run(ctx: Ctx, run: Any, nonce: str, dispatched: float) -> str:
    """ "" if `run` (a REST workflow-run object) is this pod's watchdog run, else why not."""
    if not isinstance(run, dict):
        return "no run object"
    want = expected_title(ctx.pod, nonce)
    if run.get("display_title") != want:
        return f"run-name {run.get('display_title')!r} != {want!r}"
    if run.get("event") != "workflow_dispatch":
        return f"event {run.get('event')!r}"
    if run.get("head_branch") != BRANCH:
        return f"branch {run.get('head_branch')!r} (needs {BRANCH})"
    path = str(run.get("path") or "")
    if path != WORKFLOW_PATH and not path.startswith(WORKFLOW_PATH + "@"):
        return f"workflow {path!r}"
    created = parse_ts(run.get("created_at"))
    if created is None or created < dispatched - SKEW_S:
        return f"created {run.get('created_at')!r}, before the dispatch at {iso(dispatched)}"
    return ""


def resolve_run(ctx: Ctx, nonce: str, dispatched: float, wait_s: float) -> tuple[int | None, str]:
    """The ONE run of this workflow on main, created after the dispatch, titled with this nonce."""
    since = quote(">=" + iso(dispatched - SKEW_S), safe="")
    path = (
        f"repos/{ctx.repo}/actions/workflows/{WORKFLOW}/runs"
        f"?event=workflow_dispatch&branch={BRANCH}&per_page=100&created={since}"
    )
    end = ctx.now() + wait_s
    while True:
        rc, j, err = ctx.gh(path)
        runs = j.get("workflow_runs") if isinstance(j, dict) else None
        if rc == 0 and isinstance(runs, list):
            want = expected_title(ctx.pod, nonce)
            hits = [r for r in runs if isinstance(r, dict) and r.get("display_title") == want]
            if len(hits) > 1:
                return None, f"{len(hits)} runs carry this nonce: ambiguous"
            if hits:
                rid = hits[0].get("id")
                return (rid, "") if isinstance(rid, int) else (None, "run without an id")
        if ctx.now() >= end:
            return None, f"no run titled with this nonce after {wait_s:.0f} s ({err or 'listed'})"
        ctx.sleep(5)


def cmd_dispatch(ctx: Ctx, a: argparse.Namespace) -> int:
    nonce = secrets.token_hex(8)
    inputs = {
        "pod_id": ctx.pod,
        "deadline_epoch": str(a.deadline_epoch),
        "cap_s": str(a.cap_s) if a.cap_s else "",
        "lead_s": str(a.lead_s),
        "dispatch_id": nonce,
    }
    body = json.dumps({"ref": BRANCH, "inputs": inputs})
    dispatched = ctx.now()
    rc, j, err = ctx.gh(
        "-X", "POST", f"repos/{ctx.repo}/actions/workflows/{WORKFLOW}/dispatches", "--input", "-",
        stdin=body,
    )  # fmt: skip
    state: dict[str, Any] = {
        "pod": ctx.pod,
        "nonce": nonce,
        "dispatched": round(dispatched, 3),
        "deadline_epoch": a.deadline_epoch,
        "repo": ctx.repo,
        "run_id": None,
    }
    ctx.log("dispatch", rc=rc, nonce=nonce, response=j if isinstance(j, dict) else None, err=err)
    if rc != 0:
        _write_json(ctx.gh_state, state)
        ctx.out(f"[gate] dispatch FAILED (rc {rc}): {err}. The GitHub layer is down: run the gate,")
        ctx.out("[gate] which aborts and cleans up.")
        return 3
    rid = j.get("workflow_run_id") if isinstance(j, dict) else None
    how = "dispatch response"
    if not isinstance(rid, int):
        rid, why = resolve_run(ctx, nonce, dispatched, a.resolve_wait_s)
        how = "run list"
        if rid is None:
            _write_json(ctx.gh_state, state)
            ctx.log("run_unresolved", why=why)
            ctx.out(f"[gate] run id NOT resolved: {why}")
            return 4
    rc, run, err = ctx.gh(f"repos/{ctx.repo}/actions/runs/{rid}")
    why = verify_run(ctx, run, nonce, dispatched) if rc == 0 else f"read failed: {err}"
    ctx.log("run_verify", run_id=rid, source=how, ok=not why, why=why)
    if why:
        _write_json(ctx.gh_state, state)
        ctx.out(f"[gate] run {rid} is NOT this pod's watchdog run: {why}")
        return 4
    state.update(run_id=rid, html_url=run.get("html_url"), source=how)
    _write_json(ctx.gh_state, state)
    ctx.out(
        f"[gate] GitHub watchdog run {rid} ({how}) belongs to pod {ctx.pod}: {run.get('html_url')}"
    )
    return 0


def github_status(ctx: Ctx, confirm_by: float, deadline: float) -> tuple[str, str, dict[str, Any]]:
    """("ok" | "wait" | "fail", why, info) for the GitHub layer."""
    try:
        st = json.loads(ctx.gh_state.read_text())
    except (OSError, ValueError):
        return "fail", f"no dispatch state {ctx.gh_state} (run `launch_gate.py dispatch` first)", {}
    rid, nonce = st.get("run_id"), str(st.get("nonce") or "")
    if st.get("pod") != ctx.pod or not NONCE_RE.match(nonce):
        return "fail", "dispatch state is for another pod or has no nonce", {}
    if st.get("deadline_epoch") != deadline:
        return "fail", f"dispatched deadline {st.get('deadline_epoch')} != {deadline}", {}
    if not isinstance(rid, int):
        return "fail", "dispatch captured no verified run id", {}
    info = {"github_run": rid, "dispatch": nonce}
    rc, run, err = ctx.gh(f"repos/{ctx.repo}/actions/runs/{rid}")
    if rc != 0:
        return "wait", f"run read failed: {err}", info
    why = verify_run(ctx, run, nonce, float(st.get("dispatched") or 0))
    if why:
        return "fail", f"run {rid} is not this pod's: {why}", info
    if run.get("status") == "completed":
        return "fail", f"run {rid} already completed ({run.get('conclusion')}): not watching", info
    rc, jobs, err = ctx.gh(f"repos/{ctx.repo}/actions/runs/{rid}/jobs")
    if rc != 0 or not isinstance(jobs, dict):
        return "wait", f"jobs read failed: {err}", info
    steps = [
        s
        for jb in jobs.get("jobs") or []
        if isinstance(jb, dict) and jb.get("name") == "watchdog"
        for s in jb.get("steps") or []
        if isinstance(s, dict)
    ]
    by = {s.get("name"): s for s in steps}
    conf = by.get(f"Confirm pod {ctx.pod}")
    if conf is None:
        others = [n for n in by if str(n).startswith("Confirm pod")]
        if others:
            return "fail", f"run confirms {others}, not 'Confirm pod {ctx.pod}'", info
        return "wait", "confirm step not started", info
    if conf.get("status") != "completed":
        return "wait", f"confirm step {conf.get('status')}", info
    if conf.get("conclusion") != "success":
        return "fail", f"confirm step concluded {conf.get('conclusion')}", info
    done = parse_ts(conf.get("completed_at"))
    if done is None or done > confirm_by + SKEW_S:
        return "fail", f"confirmed at {conf.get('completed_at')}, after {iso(confirm_by)}", info
    wd = by.get("Watchdog")
    if wd is None or wd.get("status") != "in_progress":
        if wd is not None and wd.get("status") == "completed":
            return "fail", f"Watchdog step ended ({wd.get('conclusion')})", info
        return "wait", "Watchdog step not running yet", info
    return "ok", f"run {rid}: Confirm pod {ctx.pod} success, Watchdog running", info


# ---- Mac watchdog --------------------------------------------------------------------------------
def read_env_file(p: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def mac_status(ctx: Ctx, confirm_by: float, deadline: float) -> tuple[str, str, dict[str, Any]]:
    try:
        st = read_env_file(ctx.mac_state)
    except OSError:
        return "wait", f"no Mac watchdog state {ctx.mac_state} yet", {}
    pid_s, tag, log = st.get("pid", ""), st.get("tag", ""), st.get("log", "")
    if st.get("pod") != ctx.pod or not pid_s.isdigit() or not TAG_RE.match(tag) or not log:
        return "fail", "Mac state is for another pod or incomplete", {}
    if st.get("deadline") != str(deadline):
        return "fail", f"Mac watchdog deadline {st.get('deadline')} != {deadline}", {}
    try:
        start = float(st.get("start", ""))
    except ValueError:
        return "fail", "Mac state has no start epoch", {}
    pid = int(pid_s)
    info = {"mac_pid": pid, "mac_tag": tag}
    rc, cmdline, _ = ctx.run(["ps", "-p", str(pid), "-o", "command="], None, None, 10.0)
    cl = f" {cmdline.strip()} "
    if rc != 0 or not cmdline.strip():
        return "fail", f"Mac watchdog pid {pid} is not running", info
    if (
        "watchdog.py" not in cl
        or f" --pod-id {ctx.pod} " not in cl
        or f" --run-tag {tag} " not in cl
    ):
        return "fail", f"pid {pid} is not this run's watchdog: {cmdline.strip()[:200]!r}", info
    confirmed: float | None = None
    exited = None
    try:
        lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not isinstance(r, dict) or r.get("pod") != ctx.pod or r.get("tag") != tag:
            continue
        ep = r.get("epoch")
        if not isinstance(ep, (int, float)) or ep < start - 1:
            continue
        if r.get("event") == "confirmed" and confirmed is None:
            confirmed = float(ep)
        if r.get("event") == "exit":
            exited = r.get("rc")
    if exited is not None:
        return "fail", f"Mac watchdog exited (rc {exited})", info
    if confirmed is None:
        return "wait", "no 'confirmed' from this Mac watchdog run yet", info
    if confirmed > confirm_by:
        return "fail", f"Mac confirmed at {iso(confirmed)}, after {iso(confirm_by)}", info
    return "ok", f"pid {pid} tag {tag} confirmed at {iso(confirmed)}", info


# ---- token, cleanup ------------------------------------------------------------------------------
TOKEN_SCRIPT = r"""set -eu
env_file=${RUNPOD_ENV_FILE:-/root/runpod.env}
dir=${WATCHDOG_TOKEN_DIR:-/workspace}
. "$env_file"
[ "${RUNPOD_POD_ID:-}" = "$1" ] || { echo "POD_MISMATCH ${RUNPOD_POD_ID:-unset}"; exit 7; }
f="$dir/WATCHDOGS_OK.$1"
umask 077
printf '%s\n' "$2" > "$f.tmp.$$"
mv -f "$f.tmp.$$" "$f"
echo "TOKEN $(head -n 1 "$f")"
"""


def write_token(ctx: Ctx, a: argparse.Namespace, info: dict[str, Any]) -> tuple[bool, str]:
    line = (
        f"pod={ctx.pod} github_run={info['github_run']} dispatch={info['dispatch']}"
        f" mac_pid={info['mac_pid']} mac_tag={info['mac_tag']} written={iso(ctx.now())}"
    )
    argv = [
        a.ssh_bin, "-p", str(a.ssh_port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new", f"{a.ssh_user}@{a.ssh_host}",
        "bash", "-s", "--", ctx.pod, f"'{line}'",
    ]  # fmt: skip
    rc, so, se = ctx.run(argv, TOKEN_SCRIPT, None, 60.0)
    ok = rc == 0 and f"TOKEN {line}" in so
    return ok, (so + se).strip()[-300:]


def cleanup(ctx: Ctx, a: argparse.Namespace) -> int:
    """watchdog.py --terminate-now with the Keychain key in the child's env only."""
    rc, key, _ = ctx.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_ITEM, "-w"], None, None, 15.0
    )
    key = key.strip()
    if rc != 0 or not key:
        ctx.log("cleanup", rc=None, why="no Keychain item")
        return 5
    argv = [
        sys.executable, str(REPO_ROOT / "scripts/pod/watchdog.py"), "--terminate-now",
        "--pod-id", ctx.pod, "--deadline-epoch", str(a.deadline_epoch),
        "--log", str(ctx.dir / f"{ctx.pod}.cleanup.jsonl"), "--expect-name-prefix", NAME_PREFIX,
    ]  # fmt: skip
    rc, so, _ = ctx.run(argv, None, {"RUNPOD_API_KEY": key}, 1500.0)
    del key
    ctx.log("cleanup", rc=rc, tail=so.strip().splitlines()[-1:] if so.strip() else [])
    return rc


def fail(ctx: Ctx, a: argparse.Namespace, reason: str, gh: str, mac: str) -> int:
    ctx.log("gate_failed", reason=reason, github=gh, mac=mac)
    ctx.out(f"[gate] FAILED ({reason}): GitHub: {gh}; Mac: {mac}. NOT launching the job.")
    if a.no_cleanup:
        ctx.out("[gate] --no-cleanup: the pod is NOT deleted. Delete it through the connector now.")
        return 12
    ctx.out(f"[gate] cleanup: watchdog.py --terminate-now --pod-id {ctx.pod}")
    rc = cleanup(ctx, a)
    if rc == 0:
        ctx.out(f"[gate] pod {ctx.pod} verified gone (two sources). Confirm via the connector too.")
        return 10
    ctx.out(
        f"[gate] cleanup did NOT verify pod {ctx.pod} gone (rc {rc}; see"
        f" {ctx.dir / (ctx.pod + '.cleanup.jsonl')}). Delete it through the connector NOW and"
        " confirm it is gone."
    )
    return 11


def cmd_gate(ctx: Ctx, a: argparse.Namespace) -> int:
    confirm_by = a.confirm_by if a.confirm_by else a.t_req + a.gate_s
    ctx.log("gate_start", confirm_by=iso(confirm_by), ssh=bool(a.ssh_host))
    gh = mac = "not checked"
    info: dict[str, Any] = {}
    while True:
        gs, gh, gi = github_status(ctx, confirm_by, a.deadline_epoch)
        ms, mac, mi = mac_status(ctx, confirm_by, a.deadline_epoch)
        ctx.log("check", github=gs, github_why=gh, mac=ms, mac_why=mac)
        if "fail" in (gs, ms):
            return fail(ctx, a, "watchdog not confirmed for this pod", gh, mac)
        if gs == ms == "ok":
            info = {**gi, **mi}
            break
        if ctx.now() > confirm_by + SKEW_S:
            return fail(ctx, a, f"not both confirmed by {iso(confirm_by)}", gh, mac)
        ctx.sleep(a.poll_s)
    ctx.log("gate_passed", **info)
    ctx.out(f"[gate] PASSED: both watchdogs confirmed pod {ctx.pod}. GitHub: {gh}; Mac: {mac}")
    if not a.ssh_host:
        ctx.out("[gate] no --ssh-host: token not written; rerun with --ssh-host before the job.")
        return 0
    end = ctx.now() + a.ssh_wait_s
    while True:
        # re-check both layers right before the write: a watchdog that died since does not count
        gs, gh, _ = github_status(ctx, confirm_by, a.deadline_epoch)
        ms, mac, _ = mac_status(ctx, confirm_by, a.deadline_epoch)
        if gs != "ok" or ms != "ok":
            return fail(ctx, a, "a watchdog stopped before the token was written", gh, mac)
        ok, detail = write_token(ctx, a, info)
        ctx.log("token", ok=ok, detail=detail)
        if ok:
            ctx.out(f"[gate] token written: /workspace/WATCHDOGS_OK.{ctx.pod}. Launch the job now.")
            return 0
        if "POD_MISMATCH" in detail:
            return fail(ctx, a, f"SSH target is another pod ({detail})", gh, mac)
        if ctx.now() >= end:
            return fail(ctx, a, f"token not written after {a.ssh_wait_s:.0f} s: {detail}", gh, mac)
        ctx.sleep(a.poll_s)


def _write_json(p: Path, d: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(p)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("dispatch", "gate"):
        s = sub.add_parser(name)
        s.add_argument("--pod-id", required=True)
        s.add_argument("--deadline-epoch", required=True, type=int)
        s.add_argument("--dir", default=str(REPO_ROOT / "runs/watchdog"))
        s.add_argument("--repo", default=GH_REPO)
        s.add_argument("--poll-s", type=float, default=10.0)
    d = sub.choices["dispatch"]
    d.add_argument("--cap-s", type=int, default=None)
    d.add_argument("--lead-s", type=int, default=400)
    d.add_argument("--resolve-wait-s", type=float, default=60.0)
    g = sub.choices["gate"]
    g.add_argument("--t-req", type=int, default=None, help="epoch taken just before the create")
    g.add_argument("--gate-s", type=float, default=300.0, help="confirm by T_req + this")
    g.add_argument("--confirm-by", type=float, default=None, help="absolute; overrides --gate-s")
    g.add_argument("--ssh-host", default="")
    g.add_argument("--ssh-port", type=int, default=22)
    g.add_argument("--ssh-user", default="root")
    g.add_argument("--ssh-bin", default="ssh")
    g.add_argument("--ssh-wait-s", type=float, default=900.0)
    g.add_argument("--no-cleanup", action="store_true", help="never delete (tests / owner's word)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, **inject: Any) -> int:
    a = parse_args(argv)
    if not POD_ID_RE.match(a.pod_id) or not 1.6e9 < a.deadline_epoch < 4.1e9:
        print("[gate] bad --pod-id or --deadline-epoch", file=sys.stderr)
        return 2
    if a.cmd == "gate" and a.confirm_by is None and a.t_req is None:
        print("[gate] gate needs --t-req (or --confirm-by)", file=sys.stderr)
        return 2
    if a.cmd == "gate" and a.ssh_host and not re.match(r"^[A-Za-z0-9.:-]{1,255}$", a.ssh_host):
        print("[gate] bad --ssh-host", file=sys.stderr)
        return 2
    ctx = Ctx(pod=a.pod_id, dir=Path(a.dir), repo=a.repo, **inject)
    return cmd_dispatch(ctx, a) if a.cmd == "dispatch" else cmd_gate(ctx, a)


if __name__ == "__main__":
    sys.exit(main())
