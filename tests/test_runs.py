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

    def sync(fd: int, *a: Any) -> str:
        st = os.fstat(fd)
        for p in [tmp_path, *tmp_path.rglob("*")]:
            q = p.stat()
            if (q.st_dev, q.st_ino) == (st.st_dev, st.st_ino):
                names.append("<runs>" if p == tmp_path else p.name)
                break
        else:
            names.append("?")
        return real_sync(fd, *a)

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
    monkeypatch.delattr(fcntl, "F_FULLFSYNC", raising=False)
    assert runs.sync_path(p) == "fsync"
    run = Run(work_order="WO-01", script=__file__, cfg=load_config("tiny"), device="cpu",
              runs_dir=tmp_path / "r")  # fmt: skip
    cmds: list[int] = []

    def failing(code: int) -> Any:
        def f(fd: int, cmd: int, *a: Any) -> int:
            cmds.append(cmd)
            raise OSError(code, os.strerror(code))

        return f

    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)  # its value on macOS
    monkeypatch.setattr(fcntl, "fcntl", failing(errno.ENOTSUP))
    assert runs.sync_path(p).startswith("fsync (F_FULLFSYNC unsupported here: E")
    assert cmds == [51]
    monkeypatch.setattr(fcntl, "fcntl", failing(errno.EIO))
    with pytest.raises(OSError, match="Input/output error"):
        runs.sync_path(p)
    with pytest.raises(OSError, match="Input/output error"):
        run.log_durable(kind="trial_started")
    assert json.loads(run.metrics_path.read_text())["kind"] == "trial_started"


# -- owner 2026-11-10: F_FULLFSYNC required for real-arm journal writes on macOS -----------------
class _Syncs:
    """``fcntl.fcntl`` / ``os.fsync`` spy on a simulated macOS: ``F_FULLFSYNC`` (51) present and
    succeeding (logged, no real flush) unless ``missing`` (the attribute removed) or ``errno``
    (fcntl raises it); every call logged as (method, "file" | "dir")."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import stat

        self.log: list[tuple[str, str]] = []
        self.errno: int | None = None
        self.dir_errno: int | None = None  # fcntl(F_FULLFSYNC) raises it on directories only
        self._mp = monkeypatch
        real_fcntl, real_fsync = fcntl.fcntl, os.fsync

        def kind(fd: int) -> str:
            return "dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file"

        def fake_fcntl(fd: int, cmd: int, *a: Any) -> Any:
            if cmd != 51:
                return real_fcntl(fd, cmd, *a)
            self.log.append(("F_FULLFSYNC", kind(fd)))
            if self.errno is not None:
                raise OSError(self.errno, os.strerror(self.errno))
            if self.dir_errno is not None and kind(fd) == "dir":
                raise OSError(self.dir_errno, os.strerror(self.dir_errno))
            return 0

        def fsync(fd: int) -> None:
            self.log.append(("fsync", kind(fd)))
            real_fsync(fd)

        monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
        monkeypatch.setattr(fcntl, "fcntl", fake_fcntl)
        monkeypatch.setattr(os, "fsync", fsync)

    def missing(self) -> None:
        self._mp.delattr(fcntl, "F_FULLFSYNC")


def test_journal_sync_policy_requires_f_fullfsync_only_on_a_real_arm_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.runs import SyncPolicy, journal_sync_policy

    req, best = SyncPolicy.FULL_FSYNC_REQUIRED, SyncPolicy.BEST_EFFORT
    assert journal_sync_policy(real_arm=True, platform="darwin") is req
    assert journal_sync_policy(real_arm=True, platform="linux") is best  # Linux: os.fsync
    assert journal_sync_policy(real_arm=False, platform="darwin") is best  # mock / scripted
    monkeypatch.setattr(sys, "platform", "darwin")
    assert journal_sync_policy(real_arm=True) is req  # sys.platform, read per call
    from test_real_eval import ev

    assert ev.journal_policy("hardware") is req and ev.journal_policy("mock") is best
    monkeypatch.setattr(sys, "platform", "linux")
    assert ev.journal_policy("hardware") is best


@pytest.mark.parametrize("fail", ["missing", "ENOTSUP", "EINVAL", "ENOTTY", "EIO"])
def test_required_f_fullfsync_never_falls_back_to_fsync_on_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    """Under ``full_fsync_required`` a FILE sync is a successful F_FULLFSYNC or it raises
    ``FullFsyncRequired`` with the errno (ENOTSUP when fcntl has no F_FULLFSYNC): ``os.fsync`` is
    never run on the file in its place. The best-effort policy falls back on the same failures
    (except EIO), and a directory keeps the best-effort sync under either policy."""
    from src import runs
    from src.runs import FullFsyncRequired, SyncPolicy

    req = SyncPolicy.FULL_FSYNC_REQUIRED
    p = tmp_path / "metrics.jsonl"
    p.write_text("{}\n")
    s = _Syncs(monkeypatch)
    assert runs.sync_path(p, req) == "F_FULLFSYNC" and s.log == [("F_FULLFSYNC", "file")]
    s.log.clear()
    if fail == "missing":
        s.missing()
    else:
        s.errno = getattr(errno, fail)
    with pytest.raises(FullFsyncRequired) as e:
        runs.sync_path(p, req)
    want = errno.ENOTSUP if fail == "missing" else getattr(errno, fail)
    assert e.value.errno == want and "no fsync fallback" in str(e.value)
    assert ("fsync", "file") not in s.log  # the mutation "fall back to fsync" fails here
    with pytest.raises(FullFsyncRequired):
        runs.write_durable(tmp_path / "record.json", "{}\n", req)
    assert not (tmp_path / "record.json").exists()  # never renamed into place unsynced
    assert ("fsync", "file") not in s.log
    s.log.clear()
    if fail == "EIO":  # a failing disk: best effort raises too, so does the directory
        with pytest.raises(OSError):
            runs.sync_path(p)
        with pytest.raises(OSError):
            runs.sync_path(tmp_path, req)
        return
    assert runs.sync_path(p).startswith("fsync")  # best effort (mock): the recorded fallback
    assert runs.sync_path(tmp_path, req).startswith("fsync")  # a directory: best effort
    assert s.log[-1] == ("fsync", "dir")


def test_required_f_fullfsync_directories_and_linux(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directories: F_FULLFSYNC where the filesystem takes it (APFS does), else os.fsync, under
    either policy. Linux (no F_FULLFSYNC, best effort): os.fsync on files and directories."""
    from src import runs
    from src.runs import SyncPolicy

    s = _Syncs(monkeypatch)
    assert runs.sync_path(tmp_path, SyncPolicy.FULL_FSYNC_REQUIRED) == "F_FULLFSYNC"
    assert s.log == [("F_FULLFSYNC", "dir")]
    s.log.clear()
    s.missing()  # Linux
    p = tmp_path / "f"
    p.write_text("x")
    linux = runs.journal_sync_policy(real_arm=True, platform="linux")
    assert runs.sync_path(p, linux) == "fsync" and runs.sync_path(tmp_path, linux) == "fsync"
    assert runs.write_durable(tmp_path / "r.json", "{}", linux) == "fsync"
    assert s.log == [("fsync", "file"), ("fsync", "dir"), ("fsync", "file"), ("fsync", "dir")]


def test_run_records_its_sync_policy_and_a_required_run_never_uses_fsync_on_its_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.runs import FullFsyncRequired, SyncPolicy

    s = _Syncs(monkeypatch)
    run = Run(work_order="WO-07", script=__file__, cfg=load_config("tiny"), device="cpu",
              runs_dir=tmp_path, sync_policy=SyncPolicy.FULL_FSYNC_REQUIRED)  # fmt: skip
    assert run.log_durable(kind="session") == "F_FULLFSYNC"
    run.flush()
    rec = json.loads(run.record_path.read_text())
    assert rec["sync"] == {
        "policy": "full_fsync_required",
        "methods": ["F_FULLFSYNC"],
        "dir_methods": ["F_FULLFSYNC"],
    }
    assert ("fsync", "file") not in s.log
    s.missing()
    with pytest.raises(FullFsyncRequired, match="fcntl has no F_FULLFSYNC"):
        run.log_durable(kind="trial_started", trial_idx=0)
    with pytest.raises(FullFsyncRequired):
        run.flush()
    assert ("fsync", "file") not in s.log
    # a mock session's run: best effort; without F_FULLFSYNC it falls back, and says so
    mock = Run(work_order="WO-07", script=__file__, cfg=load_config("tiny"), device="cpu",
               runs_dir=tmp_path)  # fmt: skip
    assert mock.log_durable(kind="session") == "fsync"
    mock.flush()
    assert json.loads(mock.record_path.read_text())["sync"] == {
        "policy": "best_effort",
        "methods": ["fsync"],
        "dir_methods": ["fsync"],
    }


@pytest.mark.parametrize("fail", ["missing", "ENOTSUP"])
def test_probe_full_fsync_refuses_an_unsupported_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    from src import runs
    from src.runs import FullFsyncRequired, SyncPolicy

    req = SyncPolicy.FULL_FSYNC_REQUIRED
    s = _Syncs(monkeypatch)
    d = tmp_path / "runs"
    ok = runs.probe_full_fsync(d, req)
    assert (ok["file"], ok["directory"], ok["policy"]) == ("F_FULLFSYNC",) * 2 + (str(req),)
    if fail == "missing":
        s.missing()
    else:
        s.errno = errno.ENOTSUP
    with pytest.raises(FullFsyncRequired):
        runs.probe_full_fsync(d, req)
    assert list(d.iterdir()) == []  # the probe file is gone either way
    assert runs.probe_full_fsync(d, SyncPolicy.BEST_EFFORT)["file"].startswith("fsync")


def test_a_directory_that_rejects_f_fullfsync_is_recorded_and_noted_never_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F4: a volume that takes F_FULLFSYNC on files but not on directories. The rename of
    ``write_durable`` is made durable by the directory sync, so its fallback is recorded
    (record.json ``sync.dir_methods``) and the preflight says so (``directory_note``); file syncs
    stay F_FULLFSYNC and the session is not refused for it (directories: best effort)."""
    from src import runs
    from src.runs import SyncPolicy

    req = SyncPolicy.FULL_FSYNC_REQUIRED
    s = _Syncs(monkeypatch)
    s.dir_errno = errno.ENOTSUP
    probe = runs.probe_full_fsync(tmp_path / "runs", req)
    assert probe["file"] == "F_FULLFSYNC" and probe["directory"].startswith("fsync")
    assert "not F_FULLFSYNC" in probe["directory_note"]
    assert "directory_note" not in runs.probe_full_fsync(tmp_path / "runs", SyncPolicy.BEST_EFFORT)
    assert runs.replace_durable(tmp_path / "r.json", "{}", req) == (
        "F_FULLFSYNC",
        "fsync (F_FULLFSYNC unsupported here: ENOTSUP)",
    )
    run = Run(work_order="WO-07", script=__file__, cfg=load_config("tiny"), device="cpu",
              runs_dir=tmp_path / "runs", sync_policy=req)  # fmt: skip
    run.log_durable(kind="session")
    run.flush()
    sync = json.loads(run.record_path.read_text())["sync"]
    assert sync["methods"] == ["F_FULLFSYNC"]
    assert sync["dir_methods"] == ["fsync (F_FULLFSYNC unsupported here: ENOTSUP)"]
    assert ("fsync", "file") not in s.log


@pytest.mark.parametrize("fail", ["missing", "ENOTSUP"])
def test_sync_tree_under_the_required_policy_raises_on_a_file_never_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    """Review F1/F2: ``sync_tree`` (a saved capture episode, a recovered dataset) passes the
    policy to every file: F_FULLFSYNC on each, or ``FullFsyncRequired`` with no fsync fallback
    (a mutation that drops ``policy`` from its file syncs fails here)."""
    from src import runs
    from src.runs import FullFsyncRequired, SyncPolicy

    req = SyncPolicy.FULL_FSYNC_REQUIRED
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")
    (tmp_path / "data.parquet").write_bytes(b"x")
    s = _Syncs(monkeypatch)
    assert runs.sync_tree(tmp_path, policy=req) == 2
    assert s.log.count(("F_FULLFSYNC", "file")) == 2 and ("fsync", "file") not in s.log
    s.log.clear()
    if fail == "missing":
        s.missing()
    else:
        s.errno = errno.ENOTSUP
    with pytest.raises(FullFsyncRequired):
        runs.sync_tree(tmp_path, policy=req)
    assert ("fsync", "file") not in s.log
    assert runs.sync_tree(tmp_path) == 2  # best effort (mock): the fallback
    assert ("fsync", "file") in s.log
