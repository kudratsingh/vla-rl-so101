"""Pod deletion backstop (scripts/pod/pod_start_cmd.sh + delete_self.sh) on CPU, no network.

The start command runs against a fake root with stub curl / apt-get / sshd on PATH and a 3 s cap.
The stub curl honours --max-time, so a "hung" REST DELETE costs one --max-time before the
GraphQL fallback. Nothing here can reach RunPod: every curl on PATH is the stub.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START = REPO / "scripts/pod/pod_start_cmd.sh"
DELETE = REPO / "scripts/pod/delete_self.sh"

CURL_STUB = r"""#!/bin/bash
# logs "<t> <METHOD> <kind>"; REST_MODE / GQL_MODE = ok | hang | fail; hang = sleep --max-time, rc 28
mt=30; method=GET; url=""; data=""
while [ $# -gt 0 ]; do
  case "$1" in
    --max-time) mt="$2"; shift ;;
    -X) method="$2"; shift ;;
    -d) data="$2"; method=POST; shift ;;
    -o|-w|-H|--connect-timeout) shift ;;
    http*) url="$1" ;;
  esac; shift
done
case "$url" in
  https://api.runpod.io/v2/pods/fake-pod) kind=REST; mode=${REST_MODE:-ok} ;;
  https://api.runpod.io/graphql) kind=GQL; mode=${GQL_MODE:-ok} ;;
  *) echo "$(date +%s) REFUSED $url" >> "$STUB_LOG"; exit 6 ;;
esac
echo "$(date +%s) $method $kind $mode" >> "$STUB_LOG"
[ "$method" = GET ] && { printf 200; exit 0; }
case "$mode" in
  hang) sleep "$mt"; exit 28 ;;
  fail) printf 500; exit 0 ;;
esac
[ "$method" = DELETE ] && [ "$kind" = REST ] && { echo "$(date +%s) REST" >> "$DELETED"; printf 204; exit 0; }
case "$data" in *podTerminate*) echo "$(date +%s) GQL" >> "$DELETED" ;; esac
echo '{"data":{"podTerminate":null}}'
"""


def _stubs(d: Path) -> Path:
    d.mkdir()
    (d / "curl").write_text(CURL_STUB)
    (d / "apt-get").write_text("#!/bin/bash\nexit 0\n")
    (d / "sshd").write_text("#!/bin/bash\nexit 0\n")
    if shutil.which("timeout") is None:  # macOS: GNU timeout semantics are not under test here
        (d / "timeout").write_text(
            '#!/bin/bash\nwhile [ "${1#-}" != "$1" ]; do [ "$1" = -k ] && shift; shift; done\n'
            'shift; exec "$@"\n'
        )
    for f in d.iterdir():
        f.chmod(0o755)
    return d


def _boot(tmp: Path, cap: int, **env: str) -> tuple[subprocess.Popen[bytes], Path, Path]:
    fr = tmp / "fs"
    for p in ("root", "workspace", "run"):
        (fr / p).mkdir(parents=True, exist_ok=True)
    stubs = tmp / "stubs"
    if not stubs.exists():
        _stubs(stubs)
    text = START.read_text()
    for a, b in (
        ("/root/", f"{fr}/root/"),
        ("/root ", f"{fr}/root "),
        ("/workspace", f"{fr}/workspace"),
        ("/run/sshd", f"{fr}/run/sshd"),
        ("/usr/sbin/sshd", f"{stubs}/sshd"),
        ("sleep infinity", "sleep 60"),
    ):
        assert a in text, a
        text = text.replace(a, b)
    log, deleted = tmp / "curl.log", tmp / "deleted"
    full_env = {
        "PATH": f"{stubs}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(fr / "root"),
        "SPEND_CAP_SECONDS": str(cap),
        "CAP_PRESTART_S": "0",
        "RUNPOD_POD_ID": "fake-pod",
        "RUNPOD_API_KEY": "fake-key",
        "PUBLIC_KEY": "ssh-ed25519 AAAAFAKE test",
        "DELETE_MAX_TIME_S": "1",
        "DELETE_PAUSE_S": "1",
        "DELETE_RECHECK_S": "30",
        "STUB_LOG": str(log),
        "DELETED": str(deleted),
        "https_proxy": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        **env,
    }
    proc = subprocess.Popen(
        ["bash", "-c", text],
        env=full_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, log, deleted


def _wait_deleted(proc: subprocess.Popen[bytes], deleted: Path, limit: float) -> float | None:
    t0 = time.time()
    try:
        while time.time() - t0 < limit:
            if deleted.exists() and deleted.read_text().strip():
                return time.time() - t0
            time.sleep(0.1)
        return None
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def test_start_cmd_embeds_delete_self() -> None:
    """The start command carries a byte-identical copy of scripts/pod/delete_self.sh (the repo
    is not on the pod at boot)."""
    text = START.read_text()
    head = "cat > /root/delete_self.sh <<'DELETE_SELF_EOF'\n"
    body = text[text.index(head) + len(head) : text.index("\nDELETE_SELF_EOF\n") + 1]
    assert body == DELETE.read_text()
    # the timer is armed before apt runs
    assert text.index("bash /root/delete_self.sh 0") < text.index(
        "timeout -k 30 600 apt-get update"
    )


def test_cap_timer_falls_back_when_rest_delete_hangs(tmp_path: Path) -> None:
    """3 s cap, REST v2 DELETE hangs: the timer gives up on it after --max-time (1 s) and
    deletes through GraphQL, inside cap + max-time + slack."""
    proc, log, deleted = _boot(tmp_path, 3, REST_MODE="hang")
    t = _wait_deleted(proc, deleted, 15)
    calls = log.read_text().split("\n") if log.exists() else []
    assert t is not None, calls
    assert t <= 3 + 1 + 4, t
    assert deleted.read_text().split()[1] == "GQL"
    kinds = [c.split()[1:3] for c in calls if c]
    assert kinds[0] == ["GET", "REST"]  # boot key check
    assert ["DELETE", "REST"] in kinds and ["POST", "GQL"] in kinds
    assert "REFUSED" not in log.read_text()
    status = (tmp_path / "fs/root/backstop_status").read_text()
    assert status.startswith("KEY_OK http=200") and "timer=armed" in status


def test_cap_timer_retries_and_survives_restart(tmp_path: Path) -> None:
    """A restart re-arms the timer from the FIRST boot (no fresh cap): a pod first booted 100 s
    ago with a 3 s cap deletes at once; both endpoints fail first, the loop retries."""
    ws = tmp_path / "fs/workspace"
    ws.mkdir(parents=True)
    (ws / ".first_boot_epoch.fake-pod").write_text(f"{int(time.time()) - 100}\n")
    proc, log, deleted = _boot(tmp_path, 3, REST_MODE="fail", GQL_MODE="fail")
    time.sleep(2.5)  # API down: repeated attempts, none accepted; then the container restarts
    assert log.read_text().count("DELETE REST fail") >= 2, log.read_text()
    assert log.read_text().count("POST GQL fail") >= 2, log.read_text()
    assert not deleted.exists()
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5)
    proc, log, deleted = _boot(tmp_path, 3)  # second boot, same workspace, API up
    t = _wait_deleted(proc, deleted, 10)
    assert t is not None and t <= 4, (t, log.read_text())
    assert (tmp_path / "fs/workspace/.first_boot_epoch.fake-pod.boots").read_text().strip() == "2"


def _env_file(tmp: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (tmp / "fs/root/runpod.env").read_text().splitlines():
        k, _, v = line.removeprefix("export ").partition("=")
        out[k] = v
    return out


def test_deadline_epoch_counts_from_creation(tmp_path: Path) -> None:
    """POD_DEADLINE_EPOCH (create time + cap) fires the timer at deadline - lead even when the
    boot-relative cap is far away (60 s here)."""
    dl = int(time.time()) + 3
    proc, log, deleted = _boot(tmp_path, 60, POD_DEADLINE_EPOCH=str(dl), POD_DEADLINE_LEAD_S="1")
    t = _wait_deleted(proc, deleted, 10)
    assert t is not None and t <= 2 + 4, (t, log.read_text() if log.exists() else "")
    env = _env_file(tmp_path)
    assert env["POD_HARD_DEADLINE"] == str(dl - 1) and env["POD_DEADLINE_EPOCH"] == str(dl)
    status = (tmp_path / "fs/root/backstop_status").read_text()
    assert "timer=armed" in status and "cap_from=creation" in status, status


def test_deadline_epoch_later_than_boot_cap_keeps_boot_cap(tmp_path: Path) -> None:
    """The earlier of the two wins: a deadline beyond first boot + cap leaves the boot cap."""
    proc, _, deleted = _boot(tmp_path, 2, POD_DEADLINE_EPOCH=str(int(time.time()) + 600))
    t = _wait_deleted(proc, deleted, 10)
    assert t is not None and t <= 2 + 4, t
    assert "cap_from=first_boot" in (tmp_path / "fs/root/backstop_status").read_text()


def test_bad_deadline_epoch_deletes_at_once(tmp_path: Path) -> None:
    """A deadline that is not an epoch in seconds (text, milliseconds, or a leading zero, which
    bash arithmetic reads as octal and "09..." would abort the script before the timer is armed)
    fails safe: delete now, though the boot cap alone would allow 60 s."""
    for i, bad in enumerate(("soon", str(int(time.time() * 1000)), "0900000000")):
        d = tmp_path / str(i)
        d.mkdir()
        proc, log, deleted = _boot(d, 60, POD_DEADLINE_EPOCH=bad)
        t = _wait_deleted(proc, deleted, 10)
        assert t is not None and t <= 3, (bad, t, log.read_text() if log.exists() else "")
        assert _env_file(d)["POD_HARD_DEADLINE"] == "0"
        # the timer can fire before the key check writes backstop_status: read the boot log
        assert (
            "is not an epoch in seconds: deleting the pod now"
            in (d / "fs/root/boot.log").read_text()
        )


def test_leading_zero_values_do_not_abort_the_start_command(tmp_path: Path) -> None:
    """Leading zeros are octal to bash arithmetic ('09' is a hard error that would abort the whole
    start command before the timer is armed). SPEND_CAP_SECONDS '09700' is refused like any
    non-decimal cap (delete now); padded optional values fall back to their defaults."""
    a = tmp_path / "opt"
    a.mkdir()
    p, _, deleted = _boot(a, 900, CAP_PRESTART_S="090", POD_DEADLINE_LEAD_S="09")
    st = a / "fs/root/backstop_status"
    t0 = time.time()
    try:
        while time.time() - t0 < 10 and not st.exists():
            time.sleep(0.1)
    finally:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=5)
    assert "timer=armed" in st.read_text() and not deleted.exists()
    first = int((a / "fs/root/pod_start_epoch").read_text())
    assert _env_file(a)["POD_HARD_DEADLINE"] == str(first + 900 - 180)  # default prestart
    b = tmp_path / "cap"
    b.mkdir()
    p2, log2, deleted2 = _boot(b, 60, SPEND_CAP_SECONDS="09700")
    t2 = _wait_deleted(p2, deleted2, 10)
    assert t2 is not None and t2 <= 3, (t2, log2.read_text() if log2.exists() else "")
    assert "is not a decimal integer: deleting the pod now" in (b / "fs/root/boot.log").read_text()
