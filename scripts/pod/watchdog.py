#!/usr/bin/env python3
"""Pod watchdog that runs outside the pod: terminate one RunPod pod before its spend deadline.

It is independent of the container: the in-pod hard-cap timer (scripts/pod/pod_start_cmd.sh) can
only start once the container boots, and it cannot help if the container never starts, restarts
without its volume, or loses its key. This process starts right after the create call, on the Mac
(scripts/pod/watchdog_mac.sh) and on a GitHub-hosted runner (.github/workflows/pod-watchdog.yml).
Runbook: docs/runbooks/pod-launch.md.

Python stdlib only, so it runs under the repo venv and under a bare python3 on ubuntu-latest.

Clock. --deadline-epoch is absolute: the launcher takes the epoch immediately BEFORE the create call
and adds SPEND_CAP_SECONDS, so the cap counts from creation, not boot. With --cap-s, the pod's
createdAt (REST v2) tightens it to min(deadline, createdAt + cap). Fire time = deadline - lead.

Behaviour:
  - before the fire time: read the pod every --poll-s; exit 0 once it is verified gone (it deleted
    itself), provided this key has seen it exist (a 404 can also mean "not visible to this key");
  - at the fire time (at once if started after it): terminate with REST v2 DELETE
    (api.runpod.io/v2/pods/{id}), then REST v1 DELETE (rest.runpod.io/v1/pods/{id}), then GraphQL
    podTerminate (counted only with no "errors"); every request has a connect and a total timeout;
  - verified gone = two consecutive reads >= --verify-gap-s apart that show no pod (404, status
    TERMINATED, or GraphQL pod null), not an accepted delete; otherwise the delete is sent again;
  - not verified gone by the deadline: alert (macOS notification + bell; GitHub ::error::) and keep
    trying until deadline + --overrun-s, then exit 2.
Exit codes: 0 verified gone; 2 not verified by deadline + overrun; 3 gone but this key never saw
the pod (check the console); 4 pod name does not match --expect-name-prefix (nothing deleted);
5 bad input or no key.

The key comes only from the environment (RUNPOD_API_KEY), never argv, and is redacted from every
line this script writes. Every attempt is logged as one JSONL line to --log.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

API_V2 = "https://api.runpod.io"
API_V1 = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
POD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

PRESENT, GONE, UNKNOWN = "present", "gone", "unknown"


class Clock:
    """Wall clock; tests inject one whose sleep() advances virtual time."""

    def now(self) -> float:
        return time.time()

    def sleep(self, s: float) -> None:
        if s > 0:
            time.sleep(s)


@dataclass
class Resp:
    status: int | None  # None = no HTTP answer (timeout, refused, DNS, TLS)
    body: str = ""
    error: str = ""
    elapsed: float = 0.0


@dataclass
class Read:
    state: str
    source: str
    detail: str = ""
    created: float | None = None
    name: str | None = None


@dataclass
class Cfg:
    pod_id: str
    deadline: float
    lead_s: float
    log: str
    cap_s: float | None = None
    poll_s: float = 60.0
    verify_gap_s: float = 10.0
    retry_pause_s: float = 10.0
    overrun_s: float = 900.0
    connect_timeout_s: float = 5.0
    timeout_s: float = 8.0
    delete_timeout_s: float = 15.0
    late_warn_s: float = 180.0
    expect_name_prefix: str = ""
    api_v2: str = API_V2
    api_v1: str = API_V1
    graphql: str = GRAPHQL
    alerts: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _iso(t: float) -> str:
    utc = timezone.utc  # noqa: UP017 - datetime.UTC is 3.11+, this also runs on 3.9
    return datetime.fromtimestamp(t, utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_created(s: Any) -> float | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        # Python < 3.11 (macOS /usr/bin/python3 is 3.9) does not parse a trailing Z
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: FURB162
    except ValueError:
        return None


def _base_ok(url: str) -> bool:
    """https only, except a loopback fake API in tests: the key must never travel in clear text."""
    u = urlsplit(url)
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost"))


class Watchdog:
    def __init__(
        self,
        cfg: Cfg,
        key: str,
        clock: Clock | None = None,
        out: Callable[[str], None] | None = None,
        notify: Callable[[str, str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._key = key
        self.clock = clock or Clock()
        self._out = out or (lambda s: print(s, flush=True))
        self._notify = notify or _notify
        self.seen = False
        self.created: float | None = None
        self.deadline = cfg.deadline
        self.attempts = 0
        self.requests = 0

    # ---- output --------------------------------------------------------------------------
    def redact(self, s: str) -> str:
        return s.replace(self._key, "***") if self._key else s

    def log(self, event: str, **kw: Any) -> None:
        now = self.clock.now()
        rec = {"t": _iso(now), "epoch": round(now, 3), "pod": self.cfg.pod_id, "event": event, **kw}
        line = self.redact(json.dumps(rec, sort_keys=False, default=str))
        try:
            d = os.path.dirname(self.cfg.log)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.cfg.log, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:  # a full disk must not stop the watchdog
            self._out(f"[watchdog] log write failed: {e}")
        self._out(f"[watchdog] {line}")

    def alert(self, title: str, msg: str) -> None:
        self.log("alert", title=title, msg=msg)
        if self.cfg.alerts:
            self._notify(self.redact(title), self.redact(msg))

    # ---- HTTP ----------------------------------------------------------------------------
    def request(
        self, method: str, url: str, body: dict[str, Any] | None = None, total: float = 0.0
    ) -> Resp:
        """One HTTP request with a connect timeout and a total wall-clock timeout."""
        self.requests += 1
        res: list[Resp] = []
        t0 = time.monotonic()
        tot = total or self.cfg.timeout_s
        ct = min(self.cfg.connect_timeout_s, tot)

        def go() -> None:
            conn: http.client.HTTPConnection | None = None
            try:
                u = urlsplit(url)
                host = u.hostname or ""
                cls = (
                    http.client.HTTPSConnection
                    if u.scheme == "https"
                    else http.client.HTTPConnection
                )
                conn = cls(host, u.port, timeout=ct)
                conn.connect()
                if conn.sock is not None:
                    conn.sock.settimeout(max(0.05, tot - (time.monotonic() - t0)))
                path = u.path + (f"?{u.query}" if u.query else "")
                data = json.dumps(body).encode() if body is not None else None
                hdr = {"Authorization": f"Bearer {self._key}", "Accept": "application/json"}
                if data is not None:
                    hdr["Content-Type"] = "application/json"
                conn.request(method, path or "/", body=data, headers=hdr)
                r = conn.getresponse()
                txt = r.read(65536).decode("utf-8", "replace")
                res.append(Resp(r.status, txt))
            except Exception as e:  # noqa: BLE001 - every failure is "no answer"
                res.append(Resp(None, error=f"{type(e).__name__}: {e}"[:200]))
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001, S110 - closing a dead socket
                        pass

        th = threading.Thread(target=go, daemon=True)
        th.start()
        th.join(tot + 0.5)
        el = time.monotonic() - t0
        if not res:  # still blocked (e.g. DNS): abandon the daemon thread
            return Resp(None, error=f"total timeout {tot:g} s", elapsed=el)
        r = res[0]
        r.elapsed = el
        return r

    def _gql(self, query: str, total: float = 0.0) -> tuple[Resp, dict[str, Any] | None]:
        r = self.request("POST", self.cfg.graphql, {"query": query}, total)
        try:
            j = json.loads(r.body) if r.status == 200 else None
        except ValueError:
            j = None
        return r, j if isinstance(j, dict) else None

    # ---- reads ---------------------------------------------------------------------------
    def _read_rest(self, base: str, path: str, src: str) -> Read:
        r = self.request("GET", f"{base}{path}")
        self.log("read", source=src, http=r.status, err=r.error, s=round(r.elapsed, 2))
        if r.status == 404:
            return Read(GONE, src, "404")
        if r.status == 200:
            try:
                j = json.loads(r.body)
            except ValueError:
                return Read(UNKNOWN, src, "200 with non-JSON body")
            if not isinstance(j, dict):
                return Read(UNKNOWN, src, "200 with non-object body")
            st = str(j.get("status") or j.get("desiredStatus") or "")
            name = j.get("name") if isinstance(j.get("name"), str) else None
            created = parse_created(j.get("createdAt"))
            if st.upper() == "TERMINATED":
                return Read(GONE, src, "status TERMINATED", created, name)
            if not j.get("id"):
                return Read(UNKNOWN, src, "200 without an id")
            return Read(PRESENT, src, st or "?", created, name)
        return Read(UNKNOWN, src, f"http {r.status}" if r.status else r.error)

    def _read_gql(self) -> Read:
        q = 'query { pod(input: {podId: "' + self.cfg.pod_id + '"}) { id name desiredStatus } }'
        r, j = self._gql(q)
        self.log("read", source="graphql", http=r.status, err=r.error, s=round(r.elapsed, 2))
        if j is None:
            return Read(UNKNOWN, "graphql", f"http {r.status}" if r.status else r.error)
        if j.get("errors"):
            return Read(UNKNOWN, "graphql", "errors in body")
        data = j.get("data")
        if not isinstance(data, dict) or "pod" not in data:
            return Read(UNKNOWN, "graphql", "no data.pod")
        pod = data["pod"]
        if pod is None:
            return Read(GONE, "graphql", "pod null")
        if isinstance(pod, dict):
            name = pod.get("name") if isinstance(pod.get("name"), str) else None
            if str(pod.get("desiredStatus", "")).upper() == "TERMINATED":
                return Read(GONE, "graphql", "desiredStatus TERMINATED", name=name)
            return Read(PRESENT, "graphql", str(pod.get("desiredStatus", "?")), name=name)
        return Read(UNKNOWN, "graphql", "unexpected data.pod")

    def read(self) -> Read:
        """First decisive answer from REST v2, then REST v1, then GraphQL."""
        pid = self.cfg.pod_id
        rd = Read(UNKNOWN, "none")
        for fn in (
            lambda: self._read_rest(self.cfg.api_v2, f"/v2/pods/{pid}", "rest_v2"),
            lambda: self._read_rest(self.cfg.api_v1, f"/pods/{pid}", "rest_v1"),
            self._read_gql,
        ):
            rd = fn()
            if rd.state != UNKNOWN:
                break
        if rd.state == PRESENT:
            self.seen = True
        if rd.created is not None and self.created is None:
            self.created = rd.created
            self._tighten()
        return rd

    def _tighten(self) -> None:
        if self.created is None:
            return
        late = self.clock.now() - self.created
        if late > self.cfg.late_warn_s and not self.cfg.extra.get("warned_late"):
            self.cfg.extra["warned_late"] = True
            self.log("warn_started_late", seconds_after_create=round(late, 1), source="createdAt")
            _gh("warning", f"watchdog started {late:.0f} s after the pod's createdAt")
        if self.cfg.cap_s is None:
            return
        d2 = self.created + self.cfg.cap_s
        self.log("created_at", created=_iso(self.created), created_plus_cap=round(d2, 3))
        if d2 < self.deadline:
            self.log("deadline_tightened", old=round(self.deadline, 3), new=round(d2, 3))
            self.deadline = d2

    @property
    def fire(self) -> float:
        return self.deadline - self.cfg.lead_s

    def name_ok(self, rd: Read) -> bool:
        p = self.cfg.expect_name_prefix
        return not p or rd.name is None or rd.name.startswith(p)

    # ---- terminate -----------------------------------------------------------------------
    def terminate(self) -> bool:
        """One attempt over all three endpoints; True if any of them accepted."""
        self.attempts += 1
        n, pid = self.attempts, self.cfg.pod_id
        for src, base, path in (
            ("rest_v2", self.cfg.api_v2, f"/v2/pods/{pid}"),
            ("rest_v1", self.cfg.api_v1, f"/pods/{pid}"),
        ):
            r = self.request("DELETE", f"{base}{path}", total=self.cfg.delete_timeout_s)
            ok = r.status is not None and (200 <= r.status < 300 or r.status == 404)
            self.log(
                "delete", attempt=n, source=src, http=r.status, err=r.error, s=round(r.elapsed, 2)
            )
            if ok:
                return True
        r, j = self._gql(
            'mutation { podTerminate(input: {podId: "' + pid + '"}) }', self.cfg.delete_timeout_s
        )
        ok = j is not None and not j.get("errors") and "podTerminate" in (j.get("data") or {})
        self.log(
            "delete",
            attempt=n,
            source="graphql",
            http=r.status,
            err=r.error,
            accepted=ok,
            body=r.body.replace("\n", " ")[:200],
            s=round(r.elapsed, 2),
        )
        return ok

    def verify_gone(self) -> bool:
        """Two consecutive reads, >= verify_gap_s apart, that both show no pod."""
        a = self.read()
        self.log("verify", read=1, state=a.state, source=a.source, detail=a.detail)
        if a.state != GONE:
            return False
        self.clock.sleep(self.cfg.verify_gap_s)
        b = self.read()
        self.log("verify", read=2, state=b.state, source=b.source, detail=b.detail)
        return b.state == GONE

    # ---- main loop -----------------------------------------------------------------------
    def run(self) -> int:
        c = self.cfg
        now = self.clock.now()
        self.log(
            "start",
            deadline=_iso(self.deadline),
            deadline_epoch=round(self.deadline, 3),
            lead_s=c.lead_s,
            cap_s=c.cap_s,
            fire_in_s=round(self.fire - now, 1),
        )
        if c.cap_s is not None:
            late = now - (c.deadline - c.cap_s)
            if late > c.late_warn_s:
                c.extra["warned_late"] = True
                self.log("warn_started_late", seconds_after_create=round(late, 1), source="cap")
                _gh("warning", f"watchdog started {late:.0f} s after the create call")
        rd = self.read()
        self.log("first_read", state=rd.state, source=rd.source, detail=rd.detail, name=rd.name)
        if not self.name_ok(rd):
            return self._refuse_name(rd)
        if rd.state == UNKNOWN:
            self.alert("pod watchdog: cannot read the pod", f"{c.pod_id}: {rd.detail}")
        if self.clock.now() >= self.fire:
            self.log("started_after_fire_time", late_s=round(self.clock.now() - self.fire, 1))
        # phase 1: watch until the fire time; exit early if the pod deleted itself
        streak_t: float | None = None
        while self.clock.now() < self.fire:
            if rd.state == GONE and self.seen:
                if streak_t is not None and self.clock.now() - streak_t >= c.verify_gap_s:
                    self.log("verified_gone", phase="watch", requests=self.requests)
                    return 0
                if streak_t is None:
                    streak_t = self.clock.now()
                self.clock.sleep(min(c.verify_gap_s, max(0.0, self.fire - self.clock.now())))
            else:
                streak_t = None
                self.clock.sleep(min(c.poll_s, max(0.0, self.fire - self.clock.now())))
            if self.clock.now() >= self.fire:
                break
            rd = self.read()
            if not self.name_ok(rd):
                return self._refuse_name(rd)
            if rd.state != GONE:
                streak_t = None
        return self._terminate_phase()

    def _terminate_phase(self) -> int:
        c = self.cfg
        end = self.deadline + c.overrun_s
        self.log("fire", deadline_in_s=round(self.deadline - self.clock.now(), 1))
        alerted = False
        while True:
            if not alerted and self.clock.now() >= self.deadline:
                alerted = True
                self.alert(
                    "pod watchdog: DEADLINE PASSED",
                    f"pod {c.pod_id} not verified gone; still trying for {c.overrun_s:.0f} s."
                    " Delete it in the RunPod console now.",
                )
            if self.clock.now() >= end:
                self.alert(
                    "pod watchdog: GAVE UP",
                    f"pod {c.pod_id} not verified gone after {self.attempts} attempts."
                    " Delete it in the RunPod console.",
                )
                self.log("exit", rc=2, attempts=self.attempts, requests=self.requests)
                return 2
            accepted = self.terminate()
            if accepted:
                self.clock.sleep(min(c.verify_gap_s, max(0.0, end - self.clock.now())))
            if self.verify_gone():
                if not self.seen:
                    self.alert(
                        "pod watchdog: gone, but never seen",
                        f"{c.pod_id}: this key never saw the pod; a 404 can mean 'not"
                        " visible to this key'. Check the RunPod console.",
                    )
                    self.log("exit", rc=3, attempts=self.attempts, requests=self.requests)
                    return 3
                self.log("verified_gone", phase="terminate", attempts=self.attempts)
                self.log("exit", rc=0, attempts=self.attempts, requests=self.requests)
                return 0
            self.log("still_present_or_unknown", attempt=self.attempts, accepted=accepted)
            self.clock.sleep(min(c.retry_pause_s, max(0.0, end - self.clock.now())))

    def _refuse_name(self, rd: Read) -> int:
        self.alert(
            "pod watchdog: pod name mismatch, NOT deleting",
            f"{self.cfg.pod_id} is named {rd.name!r}, expected prefix"
            f" {self.cfg.expect_name_prefix!r}. Wrong pod id?",
        )
        self.log("exit", rc=4)
        return 4


def _gh(level: str, msg: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title=pod watchdog::{msg}", flush=True)


def _notify(title: str, msg: str) -> None:
    """macOS notification + terminal bell, GitHub ::error:: annotation; never raises."""
    _gh("error", f"{title}: {msg}")
    if sys.platform == "darwin" and shutil.which("osascript"):
        esc = msg.replace("\\", "\\\\").replace('"', '\\"')
        t = title.replace("\\", "\\\\").replace('"', '\\"')
        try:
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{esc}" with title "{t}" sound name "Sosumi"',
                ],
                timeout=10,
                check=False,
                capture_output=True,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        with open("/dev/tty", "w") as tty:
            tty.write("\a")
    except OSError:
        sys.stderr.write("\a")
        sys.stderr.flush()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--deadline-epoch", required=True, type=float)
    p.add_argument(
        "--lead-s",
        type=float,
        default=400.0,
        help="start terminating this many s before the deadline (default 400 >= 3 worst-case"
        " attempt cycles: 3 deletes x 15.5 s + 2 reads x 3 endpoints x 8.5 s + 3 x 10 s"
        " gap/pause = 128 s)",
    )
    p.add_argument("--log", required=True)
    p.add_argument(
        "--cap-s", type=float, default=None, help="SPEND_CAP_SECONDS (enables createdAt)"
    )
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--verify-gap-s", type=float, default=10.0)
    p.add_argument("--retry-pause-s", type=float, default=10.0)
    p.add_argument("--overrun-s", type=float, default=900.0)
    p.add_argument("--connect-timeout-s", type=float, default=5.0)
    p.add_argument("--timeout-s", type=float, default=8.0, help="total, per read request")
    p.add_argument("--delete-timeout-s", type=float, default=15.0, help="total, per delete")
    p.add_argument("--expect-name-prefix", default="")
    p.add_argument("--no-alerts", action="store_true")
    p.add_argument("--api-v2", default=API_V2, help=argparse.SUPPRESS)
    p.add_argument("--api-v1", default=API_V1, help=argparse.SUPPRESS)
    p.add_argument("--graphql", default=GRAPHQL, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
    clock: Clock | None = None,
    out: Callable[[str], None] | None = None,
    notify: Callable[[str, str], None] | None = None,
) -> int:
    a = parse_args(argv)
    env = dict(os.environ) if env is None else env
    key = env.get("RUNPOD_API_KEY", "").strip()
    say = out or (lambda s: print(s, flush=True))
    bad = []
    if not POD_ID_RE.match(a.pod_id):
        bad.append("--pod-id must be 4-64 of [A-Za-z0-9_-]")
    if a.deadline_epoch < 1.6e9 or a.deadline_epoch > 4.1e9:
        bad.append("--deadline-epoch is not a plausible epoch")
    if min(a.lead_s, a.verify_gap_s, a.overrun_s) < 0 or min(a.timeout_s, a.delete_timeout_s) <= 0:
        bad.append("negative lead / gap / overrun or non-positive timeout")
    if a.cap_s is not None and a.cap_s <= 0:
        bad.append("--cap-s must be positive")
    if not all(_base_ok(u) for u in (a.api_v2, a.api_v1, a.graphql)):
        bad.append("API URLs must be https (http only on 127.0.0.1)")
    if not key:
        bad.append("RUNPOD_API_KEY is empty")
    cfg = Cfg(
        pod_id=a.pod_id,
        deadline=a.deadline_epoch,
        lead_s=a.lead_s,
        log=a.log,
        cap_s=a.cap_s,
        poll_s=a.poll_s,
        verify_gap_s=a.verify_gap_s,
        retry_pause_s=a.retry_pause_s,
        overrun_s=a.overrun_s,
        connect_timeout_s=a.connect_timeout_s,
        timeout_s=a.timeout_s,
        delete_timeout_s=a.delete_timeout_s,
        expect_name_prefix=a.expect_name_prefix,
        api_v2=a.api_v2.rstrip("/"),
        api_v1=a.api_v1.rstrip("/"),
        graphql=a.graphql,
        alerts=not a.no_alerts,
    )
    wd = Watchdog(cfg, key, clock=clock, out=say, notify=notify)
    if bad:
        wd.alert("pod watchdog NOT running", "; ".join(bad))
        wd.log("exit", rc=5)
        return 5
    try:
        return wd.run()
    except KeyboardInterrupt:
        wd.alert("pod watchdog stopped by hand", f"pod {cfg.pod_id} is no longer watched")
        wd.log("exit", rc=130)
        return 130


def _sigterm(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt  # kill / a cancelled GitHub run: alert that the pod is unwatched


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm)
    sys.exit(main())
