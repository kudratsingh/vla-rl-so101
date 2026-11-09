"""Pod approval gate (scripts/pod/approval_lib.sh) on CPU: only THIS pod's approval file, written
after the pod's first boot, passes; a stale file (reused volume, earlier pod) or the old unscoped
/workspace/APPROVED never does (owner 2026-11-08)."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "scripts/pod/approval_lib.sh"
POD = "pod-abc123"


def _bash(tmp: Path, script: str, pod: str = POD) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "APPROVAL_DIR": str(tmp / "workspace"),
        "POD_START_EPOCH_FILE": str(tmp / "pod_start_epoch"),
        "APPROVAL_POLL_S": "0.2",
        "RUNPOD_POD_ID": pod,
    }
    return subprocess.run(
        ["bash", "-c", f". {LIB}\n{script}"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _setup(tmp: Path, booted_ago_s: int = 600) -> tuple[Path, int]:
    ws = tmp / "workspace"
    ws.mkdir()
    start = int(time.time()) - booted_ago_s
    (tmp / "pod_start_epoch").write_text(f"{start}\n")
    return ws, start


def _touch(path: Path, mtime: int) -> None:
    path.write_text("owner go\n")
    os.utime(path, (mtime, mtime))


def test_stale_and_unscoped_files_are_ignored(tmp_path: Path) -> None:
    ws, start = _setup(tmp_path)
    _touch(ws / "APPROVED", start + 60)  # the old unscoped name, even if fresh
    _touch(ws / "APPROVED.other-pod", start + 60)  # an earlier pod on the same volume
    assert _bash(tmp_path, "approval_valid").returncode == 1
    _touch(ws / f"APPROVED.{POD}", start - 5)  # this pod's name, but written before it booted
    assert _bash(tmp_path, "approval_valid").returncode == 1
    out = _bash(tmp_path, "approval_report_ignored").stdout
    assert f"IGNORES {ws / 'APPROVED'}: not this pod's file" in out
    assert "APPROVED.other-pod: not this pod's file" in out
    assert re.search(rf"APPROVED\.{POD}: stale: written 5 s before", out)
    res = _bash(tmp_path, "await_approval 1; echo rc=$?")
    assert "rc=21" in res.stdout and "no owner approval" in res.stdout
    # no pod id, or no pod start epoch: nothing counts
    _touch(ws / f"APPROVED.{POD}", start + 60)
    assert _bash(tmp_path, "approval_valid", pod="").returncode == 1
    (tmp_path / "pod_start_epoch").unlink()
    assert _bash(tmp_path, "approval_valid").returncode == 1


def test_fresh_scoped_file_is_accepted(tmp_path: Path) -> None:
    ws, start = _setup(tmp_path)
    _touch(ws / f"APPROVED.{POD}", start + 60)
    assert _bash(tmp_path, "approval_valid").returncode == 0
    res = _bash(tmp_path, "await_approval 1")
    assert res.returncode == 0 and "[job] approved: owner go" in res.stdout
    assert "IGNORES" not in res.stdout


def test_approval_written_while_waiting_and_by_approval_write(tmp_path: Path) -> None:
    ws, _ = _setup(tmp_path)
    _touch(ws / "APPROVED", int(time.time()))  # unscoped: ignored, gate keeps waiting
    script = '( sleep 1; approval_write "owner go 2026-11-08: lr sweep" ) & await_approval 20'
    res = _bash(tmp_path, script)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"approved: owner go 2026-11-08: lr sweep (pod {POD}," in res.stdout
    assert (ws / f"APPROVED.{POD}").is_file() and not list(ws.glob("*.tmp.*"))
    assert _bash(tmp_path, 'approval_write "x"', pod="").returncode == 1


def test_job_scripts_use_the_scoped_gate() -> None:
    """No job script tests the bare /workspace/APPROVED any more; each sources the library."""
    for name in (
        "diag_job.sh",
        "demo_rebuild_job.sh",
        "wo04_job.sh",
        "wo02_job.sh",
        "wo02_fulleval_job.sh",
    ):
        text = (REPO / "scripts/pod" / name).read_text()
        assert "approval_lib.sh" in text, name
        assert "await_approval" in text, name
        assert not re.search(r"-f /workspace/APPROVED\b|> /workspace/APPROVED\b", text), name
