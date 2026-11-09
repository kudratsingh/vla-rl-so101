"""Launch gate (scripts/pod/launch_gate.py) and the job-side token check (approval_lib.sh), on CPU.

Every external command goes through a fake runner: a fake GitHub REST API behind `gh api`, a fake
`ps` table, a fake Keychain, a recorded cleanup call, and an `ssh` that runs the real token script
locally under bash (same remote-string concatenation as sshd). Nothing here dispatches a workflow,
reaches GitHub or RunPod, or opens SSH; the clock is virtual.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/pod/launch_gate.py"
LIB = REPO / "scripts/pod/approval_lib.sh"
POD = "podabc123"
OTHER = "podzzz999"
WF = ".github/workflows/pod-watchdog.yml"
KEY = "rpa_TESTKEY_gate_0123456789"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("launch_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["launch_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


lg = _load()


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class World:
    """Fake GitHub + ps + Keychain + ssh + cleanup, with a virtual clock."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.t = 1_800_000_000.0
        self.runs: dict[int, dict[str, Any]] = {}
        self.steps: dict[int, list[dict[str, Any]]] = {}
        self.next_id = 9000
        self.dispatch_mode = "id"  # id | empty | fail | other_id
        self.dispatched: list[dict[str, Any]] = []
        self.procs: dict[int, str] = {}
        self.cleanups: list[tuple[list[str], dict[str, str] | None]] = []
        self.cleanup_rc = 0
        self.keychain = True
        self.ws = tmp / "workspace"
        self.ws.mkdir()
        self.env_file = tmp / "runpod.env"
        self.env_file.write_text(f"export RUNPOD_POD_ID={POD}\nexport RUNPOD_API_KEY=podkey\n")
        self.ssh_calls = 0
        self.ssh_fail_first = 0
        self.on_poll: Any = None  # callable(world) run on each sleep
        self.gh_fail: list[str] = []  # path substrings: the next matching `gh api` read is a 502
        self.out: list[str] = []

    # clock
    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s
        if self.on_poll is not None:
            self.on_poll(self)

    # GitHub objects
    def add_run(
        self,
        pod: str,
        nonce: str,
        *,
        created: float | None = None,
        branch: str = "main",
        path: str = WF,
        event: str = "workflow_dispatch",
        status: str = "in_progress",
    ) -> int:
        rid = self.next_id
        self.next_id += 1
        self.runs[rid] = {
            "id": rid,
            "name": "pod-watchdog",
            "display_title": f"pod-watchdog pod={pod} dispatch={nonce}",
            "event": event,
            "head_branch": branch,
            "path": path,
            "status": status,
            "conclusion": None,
            "created_at": _iso(self.t if created is None else created),
            "html_url": f"https://github.com/x/y/actions/runs/{rid}",
        }
        self.steps[rid] = [
            {"name": "Set up job", "status": "completed", "conclusion": "success"},
            {"name": "Validate inputs", "status": "completed", "conclusion": "success"},
            {"name": f"Confirm pod {pod}", "status": "in_progress", "conclusion": None},
            {"name": "Watchdog", "status": "pending", "conclusion": None},
        ]
        return rid

    def confirm_github(self, rid: int, ok: bool = True) -> None:
        conf, wd = self.steps[rid][2], self.steps[rid][3]
        conf.update(status="completed", conclusion="success" if ok else "failure")
        conf["completed_at"] = _iso(self.t)
        wd.update(status="in_progress")

    def our_run(self) -> int:
        st = json.loads((self.tmp / "wd" / f"{POD}.github.json").read_text())
        rid = st["run_id"]
        assert isinstance(rid, int)
        return rid

    # Mac watchdog
    def start_mac(self, pod: str = POD, deadline: int = 0, tag: str = "0123456789abcdef") -> int:
        pid = 4242 + len(self.procs)
        log = self.tmp / "wd" / f"{pod}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        (self.tmp / "wd" / f"{pod}.mac.env").write_text(
            f"pod={pod}\npid={pid}\ntag={tag}\nstart={int(self.t)}\ndeadline={deadline}\nlog={log}\n"
        )
        self.procs[pid] = (
            f"caffeinate -i -s /x/python /r/scripts/pod/watchdog.py --pod-id {pod}"
            f" --deadline-epoch {deadline} --expect-name-prefix vla-rl-so101 --run-tag {tag}"
        )
        return pid

    def mac_log(self, event: str, pod: str = POD, tag: str = "0123456789abcdef", **kw: Any) -> None:
        log = self.tmp / "wd" / f"{POD}.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        rec = {"epoch": self.t, "pod": pod, "tag": tag, "event": event, **kw}
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # the runner
    def run(
        self, argv: list[str], stdin: str | None, env: dict[str, str] | None, timeout: float
    ) -> tuple[int, str, str]:
        if argv[:2] == ["gh", "api"]:
            return self._gh(argv[2:], stdin)
        if argv[0] == "ps":
            cl = self.procs.get(int(argv[2]))
            return (0, cl + "\n", "") if cl else (1, "", "")
        if argv[0] == "security":
            return (0, KEY + "\n", "") if self.keychain else (44, "", "not found")
        if argv[0] == "ssh":
            return self._ssh(argv, stdin)
        if argv[0] == sys.executable and "--terminate-now" in argv:
            self.cleanups.append((argv, env))
            return self.cleanup_rc, '{"event": "exit"}\n', ""
        raise AssertionError(f"unexpected command {argv}")

    def _gh(self, args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if args[:2] == ["-X", "POST"]:
            body = json.loads(stdin or "{}")
            self.dispatched.append(body)
            if self.dispatch_mode == "fail":
                return 1, "", "HTTP 404: workflow not found on main"
            inp = body["inputs"]
            rid = self.add_run(inp["pod_id"], inp["dispatch_id"])
            # like GitHub: 200 + workflow_run_id only with return_run_details true, else 204
            if self.dispatch_mode == "empty" or body.get("return_run_details") is not True:
                return 0, "", ""
            if self.dispatch_mode == "other_id":
                rid = self.add_run(OTHER, "f" * 16)
            return 0, json.dumps({"workflow_run_id": rid, "html_url": "u", "run_url": "u"}), ""
        path = args[0]
        for k, sub in enumerate(self.gh_fail):
            if sub in path:
                del self.gh_fail[k]
                return 1, "", "gh: HTTP 502: Bad Gateway"
        if "/actions/workflows/" in path and path.split("?")[0].endswith("/runs"):
            # the fake ignores the server-side filters: the gate must filter on its own
            runs = sorted(self.runs.values(), key=lambda r: -r["id"])
            return 0, json.dumps({"total_count": len(runs), "workflow_runs": runs}), ""
        if "/actions/runs/" in path:
            tail = path.split("/actions/runs/")[1]
            rid = int(tail.split("/")[0])
            if rid not in self.runs:
                return 1, "", "HTTP 404"
            if tail.endswith("/jobs"):
                job = {"name": "watchdog", "status": "in_progress", "steps": self.steps[rid]}
                return 0, json.dumps({"total_count": 1, "jobs": [job]}), ""
            return 0, json.dumps(self.runs[rid]), ""
        raise AssertionError(f"unexpected gh api {args}")

    def _ssh(self, argv: list[str], stdin: str | None) -> tuple[int, str, str]:
        self.ssh_calls += 1
        if self.ssh_fail_first > 0:
            self.ssh_fail_first -= 1
            return 255, "", "ssh: connect to host: Connection refused"
        i = next(k for k, x in enumerate(argv) if "@" in x)
        remote = " ".join(argv[i + 1 :])  # sshd hands the joined string to the login shell
        env = {
            "PATH": "/usr/bin:/bin",
            "RUNPOD_ENV_FILE": str(self.env_file),
            "WATCHDOG_TOKEN_DIR": str(self.ws),
        }
        p = subprocess.run(
            ["bash", "-c", remote], input=stdin, env=env, capture_output=True, text=True, timeout=20,
            check=False,
        )  # fmt: skip
        return p.returncode, p.stdout, p.stderr

    def main(self, *argv: str) -> int:
        return int(
            lg.main(
                [argv[0], "--pod-id", POD, "--dir", str(self.tmp / "wd"), *argv[1:]],
                run=self.run,
                now=self.now,
                sleep=self.sleep,
                out=self.out.append,
            )
        )


@pytest.fixture
def w(tmp_path: Path) -> World:
    return World(tmp_path)


def _dispatch(w: World, deadline: int) -> int:
    return w.main("dispatch", "--deadline-epoch", str(deadline), "--cap-s", "13800")


def _gate(w: World, deadline: int, t_req: float, *extra: str) -> int:
    return w.main(
        "gate", "--deadline-epoch", str(deadline), "--t-req", str(int(t_req)), "--poll-s", "10",
        *extra,
    )  # fmt: skip


def _ssh_args() -> tuple[str, ...]:
    return ("--ssh-host", "203.0.113.7", "--ssh-port", "22022", "--ssh-wait-s", "120")


def _gate_events(w: World) -> list[dict[str, Any]]:
    p = w.tmp / "wd" / f"{POD}.gate.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()]


# ---- dispatch: the captured run must be THIS pod's run --------------------------------------------
def test_dispatch_captures_and_verifies_the_run_id(w: World) -> None:
    deadline = int(w.t) + 13800
    w.add_run(OTHER, "a" * 16)  # an unrelated, earlier watchdog run
    assert _dispatch(w, deadline) == 0
    body = w.dispatched[0]
    assert body["ref"] == "main" and body["inputs"]["pod_id"] == POD
    assert body["return_run_details"] is True
    nonce = body["inputs"]["dispatch_id"]
    assert lg.NONCE_RE.match(nonce)
    st = json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())
    assert (
        st["run_id"] == w.our_run() and st["nonce"] == nonce and st["source"] == "dispatch response"
    )
    assert w.runs[st["run_id"]]["display_title"] == f"pod-watchdog pod={POD} dispatch={nonce}"


def test_dispatch_response_naming_an_unrelated_run_is_refused(w: World) -> None:
    w.dispatch_mode = "other_id"
    assert _dispatch(w, int(w.t) + 3600) == 4
    st = json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())
    assert st["run_id"] is None
    assert any("NOT this pod's watchdog run" in x for x in w.out)


def test_dispatch_without_run_id_resolves_by_exact_nonce(w: World) -> None:
    w.dispatch_mode = "empty"
    w.add_run(POD, "b" * 16)  # same pod, another dispatch's nonce: never picked
    w.add_run(OTHER, "c" * 16)
    assert _dispatch(w, int(w.t) + 3600) == 0
    st = json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())
    assert st["source"] == "run list"
    assert w.runs[st["run_id"]]["display_title"].endswith(st["nonce"])


@pytest.mark.parametrize("bad", ["created_before", "branch", "workflow", "event", "two_matches"])
def test_unrelated_runs_with_the_nonce_cannot_be_captured(
    w: World, bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dispatch response carries no id; a run with this very nonce exists but is created before
    the dispatch / on another branch / of another workflow / by another event, or twice."""
    w.dispatch_mode = "empty"
    nonce = "d" * 16
    monkeypatch.setattr(lg.secrets, "token_hex", lambda n: nonce)
    real = w._gh

    def gh(args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if args[:2] == ["-X", "POST"]:
            w.dispatched.append(json.loads(stdin or "{}"))
            kw: dict[str, Any] = {
                "created_before": {"created": w.t - 600},
                "branch": {"branch": "fix/pre-run-hardening"},
                "workflow": {"path": ".github/workflows/ci.yml"},
                "event": {"event": "push"},
                "two_matches": {},
            }[bad]
            w.add_run(POD, nonce, **kw)
            if bad == "two_matches":
                w.add_run(POD, nonce)
            return 0, "", ""
        return real(args, stdin)

    w._gh = gh  # type: ignore[method-assign]
    assert _dispatch(w, int(w.t) + 3600) == 4
    assert json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())["run_id"] is None


def test_dispatch_failure_and_no_matching_run(w: World) -> None:
    w.dispatch_mode = "fail"
    assert _dispatch(w, int(w.t) + 3600) == 3
    w.dispatch_mode = "empty"
    real = w._gh

    def gh(args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if args[:2] == ["-X", "POST"]:
            w.add_run(OTHER, json.loads(stdin or "{}")["inputs"]["dispatch_id"])  # other pod id
            return 0, "", ""
        return real(args, stdin)

    w._gh = gh  # type: ignore[method-assign]
    t0 = w.t
    assert _dispatch(w, int(w.t) + 3600) == 4
    assert 60 <= w.t - t0 <= 70  # waited --resolve-wait-s for a matching run, then gave up


# ---- gate: both layers, this pod, by the deadline -------------------------------------------------
def _launch(w: World, mac: bool = True, github: bool = True) -> tuple[int, float, int]:
    t_req = w.t
    deadline = int(t_req) + 13800
    w.t += 20  # create call
    if mac:
        w.start_mac(deadline=deadline)
        w.t += 3
        w.mac_log("start")
        w.mac_log("confirmed", name="vla-rl-so101-diag")
    assert _dispatch(w, deadline) == 0
    rid = w.our_run()
    if github:
        w.t += 40
        w.confirm_github(rid)
    return deadline, t_req, rid


def test_both_confirmed_passes_and_writes_a_token_the_job_accepts(w: World) -> None:
    deadline, t_req, rid = _launch(w)
    assert _gate(w, deadline, t_req) == 0  # no --ssh-host: pass, no token
    assert not list(w.ws.iterdir()) and w.ssh_calls == 0
    w.ssh_fail_first = 2  # sshd not up yet: retried
    assert _gate(w, deadline, t_req, *_ssh_args()) == 0
    tok = (w.ws / f"WATCHDOGS_OK.{POD}").read_text().strip()
    assert tok.startswith(f"pod={POD} github_run={rid} dispatch=")
    assert " mac_pid=4242 mac_tag=0123456789abcdef written=" in tok
    assert w.cleanups == [] and w.ssh_calls == 3
    # the job-side check accepts exactly this file (pod booted before it was written)
    res = _job_check(w.tmp, w.ws, POD, booted_ago_s=60)
    assert res.returncode == 0 and "launch gate passed" in res.stdout, res.stdout


def test_github_only_cannot_pass(w: World) -> None:
    deadline, t_req, _ = _launch(w, mac=False)
    rc = _gate(w, deadline, t_req)
    assert rc == 10  # timed out, aborted, cleanup verified the pod gone
    assert len(w.cleanups) == 1
    argv, env = w.cleanups[0]
    assert argv[argv.index("--pod-id") + 1] == POD and "--expect-name-prefix" in argv
    assert argv[argv.index("--created-after") + 1] == str(int(t_req) - 30)
    assert env == {"RUNPOD_API_KEY": KEY} and KEY not in " ".join(argv)
    assert not list(w.ws.iterdir())
    ev = _gate_events(w)[-2]
    assert ev["event"] == "gate_failed" and "not both confirmed" in ev["reason"]
    assert w.t <= t_req + 300 + 30 + 10 + 1  # gave up at the deadline (+ skew + one poll)


def test_mac_only_cannot_pass(w: World) -> None:
    deadline, t_req, _ = _launch(w, github=False)  # GitHub run never got past "Confirm pod"
    assert _gate(w, deadline, t_req, *_ssh_args()) == 10
    assert len(w.cleanups) == 1 and w.ssh_calls == 0 and not list(w.ws.iterdir())


def test_github_confirm_failure_aborts_at_once(w: World) -> None:
    deadline, t_req, rid = _launch(w, github=False)
    w.confirm_github(rid, ok=False)
    t0 = w.t
    assert _gate(w, deadline, t_req) == 10
    assert w.t == t0  # no waiting for the deadline
    assert "confirm step concluded failure" in _gate_events(w)[-2]["github"]


@pytest.mark.parametrize(
    "how", ["other_tag", "other_pod", "before_start", "dead_pid", "other_proc"]
)
def test_stale_or_foreign_mac_confirmation_cannot_pass(w: World, how: str) -> None:
    """A 'confirmed' line from an earlier run (other tag, or before this run's start), from another
    pod, or a pid that is gone / is not this run's watchdog: never counts."""
    t_req = w.t
    deadline = int(t_req) + 13800
    if how == "before_start":
        w.mac_log("confirmed")  # same tag text, but written before this run started
        w.t += 5
    pid = w.start_mac(deadline=deadline)
    w.t += 3
    if how == "other_tag":
        w.mac_log("confirmed", tag="ffffffffffffffff")
    elif how == "other_pod":
        w.mac_log("confirmed", pod=OTHER)
    elif how != "before_start":
        w.mac_log("confirmed")
    if how == "dead_pid":
        del w.procs[pid]
    if how == "other_proc":
        w.procs[pid] = (
            f"caffeinate -i -s python watchdog.py --pod-id {OTHER} --run-tag 0123456789abcdef"
        )
    assert _dispatch(w, deadline) == 0
    w.confirm_github(w.our_run())
    rc = _gate(w, deadline, t_req, *_ssh_args())
    assert rc == 10 and len(w.cleanups) == 1 and not list(w.ws.iterdir())


@pytest.mark.parametrize("how", ["other_pod_run", "completed", "old_step_name", "late"])
def test_unrelated_or_dead_github_run_cannot_pass(w: World, how: str) -> None:
    deadline, t_req, rid = _launch(w, github=False)
    if how == "other_pod_run":  # the captured id is swapped for another pod's confirmed run
        other = w.add_run(OTHER, "e" * 16)
        w.confirm_github(other)
        p = w.tmp / "wd" / f"{POD}.github.json"
        st = json.loads(p.read_text())
        st["run_id"] = other
        p.write_text(json.dumps(st))
    elif how == "completed":  # confirmed, then the run ended: nothing is watching
        w.confirm_github(rid)
        w.runs[rid].update(status="completed", conclusion="failure")
    elif how == "old_step_name":  # an older workflow without the pod id in the step name
        w.steps[rid][2]["name"] = "Confirm pod visible"
        w.confirm_github(rid)
    else:  # confirmed only after the gate deadline
        w.t = t_req + 400
        w.confirm_github(rid)
    assert _gate(w, deadline, t_req) == 10
    assert len(w.cleanups) == 1


def test_confirmation_arriving_while_waiting_passes(w: World) -> None:
    deadline, t_req, rid = _launch(w, github=False)

    def tick(world: World) -> None:
        if world.t - t_req > 120 and world.steps[rid][2]["status"] != "completed":
            world.confirm_github(rid)

    w.on_poll = tick
    assert _gate(w, deadline, t_req) == 0 and w.cleanups == []


def test_watchdog_dying_before_the_token_aborts(w: World) -> None:
    deadline, t_req, _ = _launch(w)
    assert _gate(w, deadline, t_req) == 0
    w.mac_log("exit", rc=130)  # Mac watchdog stopped by hand after the first gate pass
    assert _gate(w, deadline, t_req, *_ssh_args()) == 10
    assert not list(w.ws.iterdir())


def test_ssh_target_of_another_pod_is_refused(w: World) -> None:
    deadline, t_req, _ = _launch(w)
    w.env_file.write_text(f"export RUNPOD_POD_ID={OTHER}\n")
    assert _gate(w, deadline, t_req, *_ssh_args()) == 10
    assert not list(w.ws.iterdir()) and w.ssh_calls == 1  # no retry against another pod


@pytest.mark.parametrize(("cleanup_rc", "keychain", "want"), [(2, True, 11), (0, False, 11)])
def test_cleanup_not_verified_says_so(w: World, cleanup_rc: int, keychain: bool, want: int) -> None:
    deadline, t_req, _ = _launch(w, mac=False)
    w.cleanup_rc, w.keychain = cleanup_rc, keychain
    assert _gate(w, deadline, t_req) == want
    assert any("Delete it through the connector NOW" in x for x in w.out)


def test_no_cleanup_flag_and_bad_input(w: World) -> None:
    deadline, t_req, _ = _launch(w, mac=False)
    assert _gate(w, deadline, t_req, "--no-cleanup") == 12 and w.cleanups == []
    assert w.main("gate", "--deadline-epoch", str(deadline)) == 2  # no --t-req
    assert _gate(w, deadline, deadline + 1) == 2  # T_req after the deadline
    assert _gate(w, deadline, t_req, "--ssh-host", "x;rm -rf /") == 2


# ---- transient read errors are not verdicts; cleanup touches only this launch's pod ----------------
def test_dispatch_204_is_resolved_later_by_the_gate(w: World) -> None:
    """GitHub creates the run after dispatch stopped listing (exit 4, pending): the gate keeps
    resolving it by the exact nonce, with the same checks, and passes."""
    t_req = w.t
    deadline = int(t_req) + 13800
    w.t += 20
    w.start_mac(deadline=deadline)
    w.t += 3
    w.mac_log("confirmed")
    real = w._gh
    held: list[dict[str, Any]] = []

    def gh(args: list[str], stdin: str | None) -> tuple[int, str, str]:
        if args[:2] == ["-X", "POST"]:
            held.append(json.loads(stdin or "{}"))
            return 0, "", ""  # 204, and the run does not exist yet
        return real(args, stdin)

    w._gh = gh  # type: ignore[method-assign]
    assert _dispatch(w, deadline) == 4
    st = json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())
    assert st["run_id"] is None and st["resolve"] == "pending"
    rid = w.add_run(POD, held[0]["inputs"]["dispatch_id"])
    w.add_run(POD, "9" * 16)  # same pod, another nonce: never picked
    w.t += 30
    w.confirm_github(rid)
    assert _gate(w, deadline, t_req) == 0 and w.cleanups == []
    st = json.loads((w.tmp / "wd" / f"{POD}.github.json").read_text())
    assert st["run_id"] == rid and st["source"] == "run list (gate)"


def test_dispatch_pending_that_never_resolves_fails_at_the_deadline(w: World) -> None:
    deadline, t_req, _ = _launch(w, github=False)
    p = w.tmp / "wd" / f"{POD}.github.json"
    st = json.loads(p.read_text())
    st.update(run_id=None, resolve="pending", nonce="7" * 16)  # no run carries this nonce
    p.write_text(json.dumps(st))
    assert _gate(w, deadline, t_req) == 10 and len(w.cleanups) == 1
    assert w.t <= t_req + 300 + 30 + 10 + 1


@pytest.mark.parametrize("where", ["run", "jobs"])
def test_transient_github_502_in_step_8_does_not_delete_a_confirmed_pod(
    w: World, where: str
) -> None:
    """Step 5 passed; step 8 runs ~15 min later (past the deadline). One 502 on the run or jobs
    read is retried, not a failure."""
    deadline, t_req, rid = _launch(w)
    assert _gate(w, deadline, t_req) == 0
    w.t += 900
    w.gh_fail = [f"/actions/runs/{rid}/jobs" if where == "jobs" else f"/actions/runs/{rid}"]
    assert _gate(w, deadline, t_req, *_ssh_args()) == 0
    assert w.cleanups == [] and (w.ws / f"WATCHDOGS_OK.{POD}").exists()


def test_transient_502_during_the_ssh_wait_is_retried(w: World) -> None:
    deadline, t_req, rid = _launch(w)
    w.t += 900
    w.ssh_fail_first = 3  # sshd still coming up
    w.gh_fail = [f"/actions/runs/{rid}/jobs", "/jobs"]  # first two polls (past the deadline)
    fired: list[bool] = []

    def tick(world: World) -> None:  # one more 502 between the 2nd and 3rd SSH attempt
        if world.ssh_calls == 2 and not fired:
            fired.append(True)
            world.gh_fail = [f"/actions/runs/{rid}"]

    w.on_poll = tick
    assert _gate(w, deadline, t_req, *_ssh_args()) == 0
    assert fired and w.gh_fail == [] and w.ssh_calls == 4
    assert w.cleanups == [] and (w.ws / f"WATCHDOGS_OK.{POD}").exists()


def test_github_unreadable_past_the_grace_fails(w: World) -> None:
    """A layer that cannot be read at all past the deadline is not confirmed: after
    --read-grace-s the gate fails and cleans up."""
    deadline, t_req, rid = _launch(w)
    w.t += 900
    w.gh_fail = [f"/actions/runs/{rid}"] * 100
    t0 = w.t
    assert _gate(w, deadline, t_req) == 10 and len(w.cleanups) == 1
    assert 60 <= w.t - t0 <= 70
    assert "unreadable for 60 s" in _gate_events(w)[-2]["reason"]


def test_unreadable_after_the_pass_fails_within_the_grace_not_the_ssh_wait(w: World) -> None:
    """After the pass, the token is never written without a fresh "ok": a layer that stays
    unreadable fails the gate after --read-grace-s (60 s), long before --ssh-wait-s (120 s)."""
    deadline, t_req, rid = _launch(w)
    w.t += 900
    w.ssh_fail_first = 1

    def tick(world: World) -> None:
        if not world.gh_fail:
            world.gh_fail = [f"/actions/runs/{rid}"] * 100

    w.on_poll = tick
    t0 = w.t
    assert _gate(w, deadline, t_req, *_ssh_args()) == 10 and w.ssh_calls == 1
    assert 60 <= w.t - t0 <= 75 and not list(w.ws.iterdir())
    assert "not readable as confirmed for 60 s" in _gate_events(w)[-2]["reason"]


def test_mac_ps_failure_is_retried_not_a_verdict(w: World) -> None:
    deadline, t_req, _ = _launch(w)
    real = w.run
    n = {"ps": 0}

    def run(argv: list[str], stdin: str | None, env: Any, timeout: float) -> tuple[int, str, str]:
        if argv[0] == "ps" and n["ps"] < 2:
            n["ps"] += 1
            return 127, "", "OSError: fork failed"
        return real(argv, stdin, env, timeout)

    w.run = run  # type: ignore[method-assign]
    w.t += 900
    assert _gate(w, deadline, t_req) == 0 and w.cleanups == []


def test_gate_on_a_foreign_pod_id_deletes_nothing(w: World) -> None:
    """A mistyped / older project pod id with no launch evidence: the gate fails at once and
    never runs the cleanup on it."""
    t_req = w.t
    deadline = int(t_req) + 13800
    rc = w.main(
        "gate", "--deadline-epoch", str(deadline), "--t-req", str(int(t_req)), "--poll-s", "10"
    )
    assert rc == 13 and w.cleanups == []
    assert any("NOTHING deleted" in x for x in w.out)
    assert _gate_events(w)[-1]["event"] == "cleanup_refused"


@pytest.mark.parametrize("how", ["older_launch", "other_deadline"])
def test_gate_with_another_launchs_state_deletes_nothing(w: World, how: str) -> None:
    """State files for this id exist, but from an earlier launch (dispatched / started before this
    T_req) or with another deadline: not this launch's pod, nothing deleted."""
    deadline, t_req, _ = _launch(w, github=False)
    if how == "older_launch":
        w.t += 3600
        new_t_req = w.t
        assert _gate(w, deadline, new_t_req) == 13
    else:
        assert _gate(w, deadline + 60, t_req) == 13
    assert w.cleanups == []


def test_cleanup_refusing_an_older_pod_maps_to_13(w: World) -> None:
    deadline, t_req, _ = _launch(w, mac=False)
    w.cleanup_rc = 7  # watchdog.py: createdAt before --created-after
    assert _gate(w, deadline, t_req) == 13 and len(w.cleanups) == 1
    assert any("NOTHING deleted" in x for x in w.out)


def test_pod_mismatch_names_both_pods(w: World) -> None:
    deadline, t_req, _ = _launch(w)
    w.env_file.write_text(f"export RUNPOD_POD_ID={OTHER}\n")
    assert _gate(w, deadline, t_req, *_ssh_args()) == 10
    assert any(f"is pod {OTHER}, not {POD}" in x and "NOT touched" in x for x in w.out)
    argv, _ = w.cleanups[0]
    assert argv[argv.index("--pod-id") + 1] == POD  # only the verified pod, never the SSH target


# ---- job side: the token -------------------------------------------------------------------------
def _job_check(
    tmp: Path, ws: Path, pod: str, booted_ago_s: int = 600
) -> subprocess.CompletedProcess[str]:
    start = tmp / "pod_start_epoch"
    start.write_text(f"{int(time.time()) - booted_ago_s}\n")
    env = {
        **os.environ,
        "APPROVAL_DIR": str(ws),
        "POD_START_EPOCH_FILE": str(start),
        "RUNPOD_POD_ID": pod,
    }
    return subprocess.run(
        ["bash", "-c", f". {LIB}\nrequire_watchdogs"],
        env=env, check=False, capture_output=True, text=True, timeout=30,
    )  # fmt: skip


def _tok(ws: Path, name: str, line: str, mtime: int) -> None:
    p = ws / name
    p.write_text(line + "\n")
    os.utime(p, (mtime, mtime))


GOOD = f"pod={POD} github_run=9001 dispatch={'a' * 16} mac_pid=4242 mac_tag={'b' * 16} written=x"


@pytest.mark.parametrize(
    ("name", "line", "age", "why"),
    [
        (None, None, 0, "missing"),
        (f"WATCHDOGS_OK.{POD}", GOOD, 900, "stale"),  # written before this pod first booted
        (f"WATCHDOGS_OK.{POD}", GOOD.replace(f"pod={POD}", f"pod={OTHER}"), 10, "another pod"),
        (f"WATCHDOGS_OK.{POD}", GOOD.replace("github_run=9001 ", ""), 10, "no GitHub"),
        (f"WATCHDOGS_OK.{POD}", GOOD.replace("mac_pid=4242 ", ""), 10, "no Mac"),
        (f"WATCHDOGS_OK.{OTHER}", GOOD.replace(f"pod={POD}", f"pod={OTHER}"), 10, "missing"),
        ("WATCHDOGS_OK", GOOD, 10, "missing"),
    ],
)
def test_job_refuses_missing_stale_or_foreign_token(
    tmp_path: Path, name: str | None, line: str | None, age: int, why: str
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    if name and line:
        _tok(ws, name, line, int(time.time()) - age)
    res = _job_check(tmp_path, ws, POD, booted_ago_s=600)
    assert res.returncode == 8, res.stdout
    assert "REFUSING to start" in res.stdout and why in res.stdout, res.stdout
    if name and not name.endswith(POD):
        assert f"IGNORES {ws / name}" in res.stdout


def test_job_scripts_check_the_token_before_setup() -> None:
    for name in ("diag_job.sh", "demo_rebuild_job.sh", "wo04_job.sh"):
        text = (REPO / "scripts/pod" / name).read_text()
        i = text.index("require_watchdogs || exit 8")
        assert text.index("pod_env || exit 9") < i < text.index("stage setup"), name
        assert text.index("trap finish EXIT") < i, name  # a refusal still deletes the pod
    # the lr-sweep approval gate is separate and unchanged in diag_job.sh
    diag = (REPO / "scripts/pod/diag_job.sh").read_text()
    assert 'await_approval "${APPROVAL_WAIT_S:-1800}" || exit 21' in diag


def test_cli_with_stub_gh_on_path(tmp_path: Path) -> None:
    """The real subprocess path: a stub `gh` that answers the dispatch with a run id and the run
    read with this pod's run (nothing reaches GitHub)."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    rec = tmp_path / "gh.log"
    now = _iso(time.time())
    (stubs / "gh").write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> "{rec}"\n'
        'case "$*" in\n'
        '  *dispatches*) body=$(cat); nonce=$(echo "$body" | sed -E \'s/.*"dispatch_id": "([0-9a-f]+)".*/\\1/\')\n'
        f'    echo "$nonce" > "{tmp_path}/nonce"\n'
        '    case "$body" in *\'"return_run_details": true\'*) echo \'{"workflow_run_id": 77}\' ;; esac ;;\n'
        f'  *actions/runs/77) n=$(cat "{tmp_path}/nonce")\n'
        f'    printf \'{{"id":77,"display_title":"pod-watchdog pod={POD} dispatch=%s","event":"workflow_dispatch","head_branch":"main","path":"{WF}","status":"queued","created_at":"{now}"}}\' "$n" ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    (stubs / "gh").chmod(0o755)
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "dispatch", "--pod-id", POD, "--deadline-epoch",
         str(int(time.time()) + 3600), "--dir", str(tmp_path / "wd")],
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, timeout=30, check=False,
    )  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr
    st = json.loads((tmp_path / "wd" / f"{POD}.github.json").read_text())
    assert st["run_id"] == 77 and st["source"] == "dispatch response"
    assert "actions/workflows/pod-watchdog.yml/dispatches --input -" in rec.read_text()
