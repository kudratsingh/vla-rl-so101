import errno
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from src.config import config_hash, load_config
from src.runs import Run


def test_run_record_and_metrics(tmp_path: Path) -> None:
    cfg = load_config("tiny")
    with Run(work_order="WO-01", script=__file__, cfg=cfg, device="cpu", runs_dir=tmp_path) as run:
        assert json.loads(run.record_path.read_text())["status"] == "running"
        run.log_metrics(step=0, loss=1.5)
        run.log_metrics(step=1, loss=1.0)
        run.results["oracle_success"] = {"PickPlace": 1.0}

    rec = json.loads(run.record_path.read_text())
    assert rec["run_id"] == run.id and run.id.endswith(rec["run_id"][-4:])
    assert rec["work_order"] == "WO-01" and rec["preset"] == "tiny"
    assert rec["config_hash"] == config_hash(cfg) and rec["config"] == cfg
    assert rec["status"] == "ok" and rec["wall_clock_s"] >= 0
    assert rec["start_utc"].endswith("Z") and rec["end_utc"].endswith("Z")
    assert rec["actual_usd"] == 0.0 and rec["estimated_usd"] == 0.0
    assert rec["device"]["device"] == "cpu" and rec["device"]["name"]
    assert "commit" in rec["git"] and "dirty" in rec["git"]
    assert rec["results"] == {"oracle_success": {"PickPlace": 1.0}}
    rows = [json.loads(x) for x in run.metrics_path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1] and rows[1]["loss"] == 1.0


def test_failed_run_is_recorded(tmp_path: Path) -> None:
    cfg = load_config("tiny")
    with (
        pytest.raises(RuntimeError),
        Run(work_order="WO-01", script=__file__, cfg=cfg, device="cpu", runs_dir=tmp_path) as run,
    ):
        raise RuntimeError("boom")
    rec = json.loads(run.record_path.read_text())
    assert rec["status"] == "failed" and "boom" in rec["results"]["error"]


def _synced_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch ``src.runs.sync_fd``: every sync appends the synced path's name (the runs dir is
    "<runs>") and then runs for real."""
    from src import runs

    names: list[str] = []
    real_sync = runs.sync_fd

    def sync(fd: int) -> str:
        st = os.fstat(fd)
        for p in [tmp_path, *tmp_path.rglob("*")]:
            q = p.stat()
            if (q.st_dev, q.st_ino) == (st.st_dev, st.st_ino):
                names.append("<runs>" if p == tmp_path else p.name)
                break
        else:
            names.append("?")
        return real_sync(fd)

    monkeypatch.setattr(runs, "sync_fd", sync)
    return names


def test_record_writes_and_durable_rows_are_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner 2026-11-09 (power loss): every record.json write is tmp + sync + rename + directory
    sync (and the runs dir once, for the new run directory's entry); ``log_metrics`` never syncs;
    ``log_durable`` syncs metrics.jsonl before it returns (and the run dir on its first row)."""
    synced = _synced_names(tmp_path, monkeypatch)
    run = Run(work_order="WO-01", script=__file__, cfg=load_config("tiny"), device="cpu",
              runs_dir=tmp_path)  # fmt: skip
    want = "F_FULLFSYNC" if sys.platform == "darwin" else "fsync"
    assert run.sync_method == want
    assert synced == ["record.json.tmp", run.id, "<runs>"]  # before the rename, then the dirs
    synced.clear()
    run.log_metrics(kind="decision", x=1)
    assert synced == []  # an ordinary row: a plain append
    assert run.log_durable(kind="trial_started", trial_idx=0) == want
    assert synced == ["metrics.jsonl", run.id]  # the file, then (first durable row) its entry
    synced.clear()
    run.log_durable(kind="trial_start_retracted", trial_idx=0)
    assert synced == ["metrics.jsonl"]
    synced.clear()
    run.flush()
    assert synced == ["record.json.tmp", run.id]
    assert not list(run.dir.glob("*.tmp"))
    rows = [json.loads(x)["kind"] for x in run.metrics_path.read_text().splitlines()]
    assert rows == ["decision", "trial_started", "trial_start_retracted"]


def test_sync_uses_f_fullfsync_falls_back_only_when_unsupported_and_raises_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F_FULLFSYNC where the OS has it; ``os.fsync`` without it (Linux) or when the filesystem
    rejects it as unsupported; any other error propagates, also out of ``log_durable`` (the row
    is appended but not known durable: the marker path blocks the first goal on it)."""
    from src import runs

    p = tmp_path / "f"
    p.write_text("x")
    monkeypatch.setattr(runs, "_F_FULLFSYNC", None)
    assert runs.sync_path(p) == "fsync"
    run = Run(work_order="WO-01", script=__file__, cfg=load_config("tiny"), device="cpu",
              runs_dir=tmp_path / "r")  # fmt: skip
    cmds: list[int] = []

    def failing(code: int) -> Any:
        def f(fd: int, cmd: int, *a: Any) -> int:
            cmds.append(cmd)
            raise OSError(code, os.strerror(code))

        return f

    monkeypatch.setattr(runs, "_F_FULLFSYNC", 51)  # F_FULLFSYNC's value on macOS
    monkeypatch.setattr(fcntl, "fcntl", failing(errno.ENOTSUP))
    assert runs.sync_path(p).startswith("fsync (F_FULLFSYNC unsupported here: E")
    assert cmds == [51]
    monkeypatch.setattr(fcntl, "fcntl", failing(errno.EIO))
    with pytest.raises(OSError, match="Input/output error"):
        runs.sync_path(p)
    with pytest.raises(OSError, match="Input/output error"):
        run.log_durable(kind="trial_started")
    assert json.loads(run.metrics_path.read_text())["kind"] == "trial_started"
