"""Outside-the-pod watchdog (scripts/pod/watchdog.py) against a fake RunPod API on 127.0.0.1.

The fake serves REST v2 (/v2/pods/{id}), REST v1 (/v1/pods/{id}) and GraphQL (/graphql) and
counts every call. A virtual clock makes sleep() instant, so poll intervals and deadlines of
minutes run in milliseconds; HTTP timeouts stay real (0.3 s here) so hangs are bounded for real.
Nothing here can reach RunPod: every URL points at the loopback fake.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/pod/watchdog.py"
KEY = "rpa_TESTKEY_0123456789abcdef"
POD = "fakepod123"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pod_watchdog", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pod_watchdog"] = mod
    spec.loader.exec_module(mod)
    return mod


wd = _load()


class FakeApi:
    """State + per-route behaviour. mode values: ok | fail (500) | hang | errors (GraphQL 200 with
    an errors body) | down (close the socket without an answer)."""

    def __init__(self) -> None:
        self.present = True
        self.name = "vla-rl-so101-diag"
        self.created: str | None = None
        self.modes: dict[str, str] = {}
        self.gone_after_gets: int | None = None  # pod deletes itself after this many GETs
        self.noop_deletes = 0  # accepted deletes that do not remove the pod
        self.gql_errors_first = 0  # GraphQL podTerminate answers with errors this many times
        self.echo_auth = False  # error bodies echo the Authorization header (redaction test)
        self.hang_s = 3.0
        self.calls: list[tuple[str, str]] = []
        self.gets = 0
        self.lock = threading.Lock()

    def mode(self, route: str) -> str:
        return self.modes.get(route, "ok")


def _handler(api: FakeApi) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def _send(self, code: int, body: Any = None) -> None:
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _route(self, method: str) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode() if n else ""
            if self.path == "/graphql":
                q = json.loads(raw).get("query", "")
                route = "gql_delete" if "podTerminate" in q else "gql_get"
            elif self.path.startswith("/v2/pods/"):
                route = f"v2_{method.lower()}"
            elif self.path.startswith("/v1/pods/"):
                route = f"v1_{method.lower()}"
            else:
                self._send(400)
                return
            with api.lock:
                api.calls.append((method, route))
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                self._send(401, {"error": "unauthorized"})
                return
            m = api.mode(route)
            if m == "hang":
                time.sleep(api.hang_s)
                return
            if m == "down":
                self.close_connection = True
                return
            if m == "fail":
                body = {"error": "boom"}
                if api.echo_auth:
                    body["auth"] = self.headers.get("Authorization", "")
                self._send(500, body)
                return
            pod_id = self.path.rsplit("/", 1)[-1]
            with api.lock:
                if route.endswith("get"):
                    api.gets += 1
                    if api.gone_after_gets is not None and api.gets > api.gone_after_gets:
                        api.present = False
                present = api.present and (route.startswith("gql") or pod_id == POD)
                if route.endswith("delete") and route != "gql_delete":
                    if api.noop_deletes > 0:
                        api.noop_deletes -= 1
                    else:
                        api.present = False
                    self._send(204 if present else 404)
                    return
                if route == "gql_delete":
                    if m == "errors" or api.gql_errors_first > 0:
                        api.gql_errors_first = max(0, api.gql_errors_first - 1)
                        err: dict[str, Any] = {"message": "Something went wrong"}
                        if api.echo_auth:
                            err["auth"] = self.headers.get("Authorization", "")
                        self._send(200, {"errors": [err], "data": {"podTerminate": None}})
                        return
                    if api.noop_deletes > 0:
                        api.noop_deletes -= 1
                    else:
                        api.present = False
                    self._send(200, {"data": {"podTerminate": None}})
                    return
                if route == "gql_get":
                    pod = {"id": POD, "name": api.name, "desiredStatus": "RUNNING"}
                    self._send(200, {"data": {"pod": pod if api.present else None}})
                    return
                if not present:
                    self._send(404, {"title": "Not Found"})
                    return
                body = {"id": POD, "name": api.name, "status": "RUNNING"}
                if route == "v2_get" and api.created:
                    body["createdAt"] = api.created
                self._send(200, body)

        def do_GET(self) -> None:
            self._route("GET")

        def do_DELETE(self) -> None:
            self._route("DELETE")

        def do_POST(self) -> None:
            self._route("POST")

    return H


@pytest.fixture
def api() -> Iterator[tuple[FakeApi, str]]:
    state = FakeApi()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    srv.daemon_threads = True
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield state, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


class VClock(wd.Clock):  # type: ignore[misc,name-defined]
    """Real time + virtual offset: sleep() advances the offset instantly."""

    def __init__(self) -> None:
        self.offset = 0.0
        self.slept = 0.0

    def now(self) -> float:
        return time.time() + self.offset

    def sleep(self, s: float) -> None:
        if s > 0:
            self.offset += s
            self.slept += s


def _run(
    tmp: Path, base: str, deadline_in: float, *extra: str, key: str = KEY, clock: Any = None
) -> tuple[int, list[dict[str, Any]], list[str], list[tuple[str, str]], Any]:
    clock = clock or VClock()
    out: list[str] = []
    alerts: list[tuple[str, str]] = []
    log = tmp / "wd.jsonl"
    argv = [
        "--pod-id", POD,
        "--deadline-epoch", str(clock.now() + deadline_in),
        "--log", str(log),
        "--api-v2", base,
        "--api-v1", f"{base}/v1",
        "--graphql", f"{base}/graphql",
        "--timeout-s", "0.3",
        "--delete-timeout-s", "0.3",
        "--connect-timeout-s", "0.3",
        *extra,
    ]  # fmt: skip
    rc = wd.main(
        argv,
        env={"RUNPOD_API_KEY": key},
        clock=clock,
        out=out.append,
        notify=lambda t, m: alerts.append((t, m)),
    )
    recs = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    return rc, recs, out, alerts, clock


def _events(recs: list[dict[str, Any]], ev: str) -> list[dict[str, Any]]:
    return [r for r in recs if r["event"] == ev]


def _deletes(api: FakeApi) -> list[str]:
    return [r for _, r in api.calls if r.endswith("delete")]


def test_self_deleted_pod_exits_early_without_delete(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.gone_after_gets = 3
    rc, recs, _, alerts, clock = _run(tmp_path, base, 3600)
    assert rc == 0
    assert _deletes(state) == []
    assert _events(recs, "verified_gone")[0]["phase"] == "watch"
    assert clock.slept < 3600 - 400  # well before the fire time
    assert alerts == []


def test_terminates_at_fire_time_and_verifies_gone(tmp_path: Path, api: Any) -> None:
    state, base = api
    rc, recs, _, _, _ = _run(tmp_path, base, 1000)
    assert rc == 0
    assert _deletes(state) == ["v2_delete"]
    fire = _events(recs, "fire")[0]
    assert 395 <= fire["deadline_in_s"] <= 401  # lead 400 s
    v = _events(recs, "verify")
    assert [x["state"] for x in v[-2:]] == ["gone", "gone"]
    assert v[-1]["epoch"] - v[-2]["epoch"] >= 10  # two reads >= 10 s apart
    assert _events(recs, "exit")[-1]["rc"] == 0
    # done before the deadline
    assert _events(recs, "exit")[-1]["epoch"] < recs[0]["deadline_epoch"]


def test_rest_500_falls_back_to_graphql(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.modes.update(v2_delete="fail", v1_delete="fail")
    rc, recs, _, _, _ = _run(tmp_path, base, 10)
    assert rc == 0
    assert _deletes(state) == ["v2_delete", "v1_delete", "gql_delete"]
    d = _events(recs, "delete")
    assert [x["http"] for x in d] == [500, 500, 200] and d[-1]["accepted"] is True


def test_graphql_errors_body_is_not_success(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.modes.update(v2_delete="fail", v1_delete="fail")
    state.gql_errors_first = 2
    rc, recs, _, _, _ = _run(tmp_path, base, 10)
    assert rc == 0
    g = [x for x in _events(recs, "delete") if x["source"] == "graphql"]
    assert [x["accepted"] for x in g] == [False, False, True]
    assert all(x["http"] == 200 for x in g)
    assert len(_events(recs, "still_present_or_unknown")) == 2


def test_accepted_delete_but_pod_still_present_is_resent(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.noop_deletes = 2
    rc, recs, _, _, _ = _run(tmp_path, base, 10)
    assert rc == 0
    assert _deletes(state) == ["v2_delete"] * 3
    assert [x["accepted"] for x in _events(recs, "still_present_or_unknown")] == [True, True]


def test_hanging_endpoints_are_bounded_by_timeouts(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.modes.update(v2_get="hang", v2_delete="hang")
    t0 = time.monotonic()
    rc, recs, _, _, _ = _run(tmp_path, base, 10)
    assert rc == 0
    assert time.monotonic() - t0 < 6  # 3 s server hangs, 0.3 s client timeouts
    hung = [r for r in recs if r.get("source") == "rest_v2" and r["event"] in ("read", "delete")]
    assert hung and all(r["http"] is None and r["s"] <= 0.3 + 0.7 for r in hung)
    assert _deletes(state) == ["v2_delete", "v1_delete"]


def test_started_after_fire_time_terminates_at_once(tmp_path: Path, api: Any) -> None:
    state, base = api
    rc, recs, _, _, clock = _run(tmp_path, base, 30)  # lead 400 s > 30 s left
    assert rc == 0
    assert _events(recs, "started_after_fire_time")
    assert clock.slept <= 10 + 10 + 1  # verify gap + the gap after an accepted delete
    assert _deletes(state) == ["v2_delete"]


def test_api_down_for_the_whole_window_alerts_and_fails(tmp_path: Path, api: Any) -> None:
    state, base = api
    for r in ("v2_get", "v1_get", "gql_get", "v2_delete", "v1_delete", "gql_delete"):
        state.modes[r] = "fail"
    rc, recs, _, alerts, _ = _run(tmp_path, base, 200, "--overrun-s", "300")
    assert rc == 2
    titles = [t for t, _ in alerts]
    assert any("cannot read" in t for t in titles)
    assert any("DEADLINE PASSED" in t for t in titles) and any("GAVE UP" in t for t in titles)
    n = max(x["attempt"] for x in _events(recs, "delete"))
    # bounded: one attempt per >= retry pause (10 s) over the 200 + 300 s window
    assert 5 <= n <= (200 + 300) // 10 + 1, n
    assert _events(recs, "exit")[-1]["rc"] == 2


def test_key_never_logged_even_when_echoed(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.echo_auth = True
    state.modes.update(v2_delete="fail", v1_delete="fail")
    state.gql_errors_first = 1
    rc, _, out, alerts, _ = _run(tmp_path, base, 10)
    assert rc == 0
    text = (tmp_path / "wd.jsonl").read_text() + "\n".join(out) + repr(alerts)
    assert KEY not in text
    assert "Bearer ***" in text  # the echoed header reached the log, redacted


def test_missing_key_refuses_and_alerts(tmp_path: Path, api: Any) -> None:
    state, base = api
    rc, _, _, alerts, _ = _run(tmp_path, base, 10, key="")
    assert rc == 5 and alerts and state.calls == []


def test_pod_never_seen_is_flagged(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.present = False
    rc, _, _, alerts, clock = _run(tmp_path, base, 600)
    assert rc == 3  # 404 from the start: gone, or invisible to this key
    assert clock.slept >= 600 - 400 - 1  # no early exit on a pod it never saw
    assert any("never seen" in t for t, _ in alerts)


def test_name_mismatch_deletes_nothing(tmp_path: Path, api: Any) -> None:
    state, base = api
    state.name = "someone-elses-pod"
    rc, _, _, alerts, _ = _run(tmp_path, base, 10, "--expect-name-prefix", "vla-rl-so101")
    assert rc == 4 and _deletes(state) == [] and alerts


def test_created_at_tightens_the_deadline_and_late_start_warns(tmp_path: Path, api: Any) -> None:
    state, base = api
    clock = VClock()
    created = clock.now() - 1000
    state.created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))
    # launcher deadline 1000 s away with a 1300 s cap => started 300 s after its T_req (warn);
    # createdAt + cap is ~300 s away => the deadline tightens and the fire time (lead 400) has passed
    rc, recs, _, _, _ = _run(tmp_path, base, 1000, "--cap-s", "1300", clock=clock)
    assert rc == 0
    assert _events(recs, "warn_started_late")[0]["source"] == "cap"
    t = _events(recs, "deadline_tightened")[0]
    assert abs(t["new"] - (int(created) + 1300)) < 2
    assert _events(recs, "started_after_fire_time")
    assert 290 < _events(recs, "fire")[0]["deadline_in_s"] < 302


def test_runs_under_plain_python3_isolated(tmp_path: Path, api: Any) -> None:
    """CLI under `python3 -I` (stdlib only, as on the GitHub runner), real clock: a pod past its
    fire time is terminated and verified gone; the key never reaches stdout or stderr."""
    state, base = api
    log = tmp_path / "cli.jsonl"
    p = subprocess.run(
        [
            sys.executable, "-I", str(SCRIPT),
            "--pod-id", POD, "--deadline-epoch", str(int(time.time()) + 5), "--log", str(log),
            "--api-v2", base, "--api-v1", f"{base}/v1", "--graphql", f"{base}/graphql",
            "--verify-gap-s", "0.2", "--timeout-s", "1", "--no-alerts",
        ],
        env={"RUNPOD_API_KEY": KEY, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )  # fmt: skip
    assert p.returncode == 0, p.stdout + p.stderr
    assert KEY not in p.stdout + p.stderr + log.read_text()
    assert _deletes(state) == ["v2_delete"]
    assert '"event": "verified_gone"' in log.read_text()


def test_mac_launcher_keeps_key_out_of_argv(tmp_path: Path) -> None:
    """scripts/pod/watchdog_mac.sh with stub `security` / `caffeinate`: the Keychain key reaches
    the child's environment only (not argv, not stdout); no Keychain item -> exit 3."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    rec = tmp_path / "rec"
    (stubs / "security").write_text(
        f'#!/bin/bash\n[ -n "${{NO_ITEM:-}}" ] && exit 44\necho "{KEY}"\n'
    )
    (stubs / "caffeinate").write_text(
        f'#!/bin/bash\necho "argv: $*" > "{rec}"\n'
        f'[ "$RUNPOD_API_KEY" = "{KEY}" ] && echo env-ok >> "{rec}"\nsleep 3\n'
    )
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "WATCHDOG_DIR": str(tmp_path / "wd"), "HOME": "/tmp"}
    cmd = ["bash", str(REPO / "scripts/pod/watchdog_mac.sh"), POD, str(int(time.time()) + 900)]
    p = subprocess.run(
        [*cmd, "900"], env=env, capture_output=True, text=True, timeout=20, check=False
    )
    assert p.returncode == 0, p.stdout + p.stderr
    time.sleep(0.5)
    r = rec.read_text()
    assert "env-ok" in r and KEY not in r and "--pod-id fakepod123" in r and "--cap-s 900" in r
    assert "--expect-name-prefix vla-rl-so101" in r
    assert KEY not in p.stdout + p.stderr
    q = subprocess.run(
        cmd, env={**env, "NO_ITEM": "1"}, capture_output=True, text=True, timeout=20, check=False
    )
    assert q.returncode == 3 and KEY not in q.stdout + q.stderr
