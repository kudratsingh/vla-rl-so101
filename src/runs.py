"""Run records (CLAUDE.md rule 1): `runs/<id>/record.json` plus a `metrics.jsonl` appender.

Usage:

    with Run(work_order="WO-01", script=__file__, cfg=cfg, device="cpu") as run:
        run.log_metrics(step=0, loss=1.0)
        run.results["oracle_success"] = 0.97

`record.json` is written at start (status "running"), and rewritten on exit with the end time,
wall-clock, dollars, results and status ("ok" or "failed"). Only `record.json` is committed;
everything else under `runs/` is ignored by git.

Durability (owner 2026-11-09, power-loss recovery of the real-eval session). Every record.json
write is atomic and synced: tmp file, sync, rename, sync of the run directory (and of the runs
directory on the run's first write, so the new directory's entry survives too). A record write
is rare (start, an explicit ``flush``, the end), so this costs nothing per step. ``log_metrics``
rows are plain appends (they survive a killed process, not a power loss); ``log_durable`` appends
a row AND syncs metrics.jsonl (and its directory on the first durable row) before it returns,
for the few rows a resume depends on (scripts/wo07_real_eval.py ``DURABLE_ROWS``).

Sync policy (``SyncPolicy``, owner 2026-11-10), fixed per ``Run`` at construction and recorded
in its record.json (``sync``: the policy and every method used):

* ``best_effort`` (mock / scripted sessions, Linux, every non-arm run): ``fcntl(F_FULLFSYNC)``
  where the OS has it (macOS: ``fsync`` alone does not flush the drive's write cache), else
  ``os.fsync`` (Linux); a filesystem that rejects F_FULLFSYNC as unsupported (ENOTSUP / EINVAL /
  ENOTTY) falls back to ``os.fsync``, and the method string says so.
* ``full_fsync_required`` (``journal_sync_policy``: a session that drives the real arm on macOS,
  ``sys.platform == "darwin"``): every FILE sync of the run's journal (record.json's tmp file,
  the ``log_durable`` rows of metrics.jsonl, the evaluation capture's journals, provenance and
  episode files) must be a successful ``fcntl(F_FULLFSYNC)``. fcntl without F_FULLFSYNC, or
  fcntl raising (ENOTSUP, EINVAL, EIO, ...), raises ``FullFsyncRequired`` (an ``OSError``
  carrying the errno): NO ``os.fsync`` fallback, so the caller treats the write as not durable
  (the first policy goal is blocked: ``DeployLoop._first_goal_check``).

Directories keep the platform's best-effort sync under both policies (``sync_path`` on a
directory fd): F_FULLFSYNC where the filesystem accepts it on a directory (it does on APFS:
checked 2026-11-10 on macOS 26, ``fcntl(dirfd, F_FULLFSYNC)`` returns 0), else ``os.fsync``
(Linux, or a filesystem that rejects it on directories). A directory sync never blocks motion by
itself, but it is what makes a create or a rename durable: in ``write_durable`` the rename comes
AFTER the file's F_FULLFSYNC, so on a volume that rejects F_FULLFSYNC on directories the rename
is on stable storage only once the drive cache is next flushed. On the arm that is the next
journal F_FULLFSYNC (F_FULLFSYNC flushes the whole drive cache), at the latest the next
``trial_started`` row, before any motion. The record says which directory method ran (record.json
``sync.dir_methods``) and ``probe_full_fsync`` says so up front (``directory_note``).
``probe_full_fsync`` checks the policy once on a temp file in the runs directory before a
real-arm session opens anything. Any other ``OSError`` of a sync propagates: the caller must
treat the row as not durable.

Journal writes under the session policy (the resume logic reads them): record.json (every
write), the ``log_durable`` rows of metrics.jsonl (``session``, ``trial_started``,
``trial_start_retracted``), and the evaluation capture (src/robot/recording.py): its provenance,
staged episode journals, saved episode files and end-state stills, the dataset a new session or
an empty restart creates (meta, calibration copy) and the files a capture recovery rewrites
(``meta/info.json``, ``meta/stats.json``, the quarantine moves).
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import platform
import secrets
import socket
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from src.config import REPO_ROOT, Config, config_hash

RUNS_DIR = REPO_ROOT / "runs"

# RunPod list prices, Oct 2026 (docs/PLAN.md). Used only when running on a RunPod pod and no
# explicit hourly rate is given. Matched by substring of torch.cuda.get_device_name().
GPU_HOURLY_USD: dict[str, float] = {
    "RTX 4090": 0.89,
    "A100": 1.79,
    "H100": 2.89,
}


_FULLFSYNC_UNSUPPORTED = {errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.ENOTTY}


class SyncPolicy(StrEnum):
    """How a file sync may be done (module doc)."""

    BEST_EFFORT = "best_effort"
    FULL_FSYNC_REQUIRED = "full_fsync_required"


class FullFsyncRequired(OSError):
    """A file sync under ``SyncPolicy.FULL_FSYNC_REQUIRED`` that was not a successful
    ``fcntl(F_FULLFSYNC)``. ``errno`` is the failure's (ENOTSUP when fcntl has no F_FULLFSYNC)."""


def journal_sync_policy(*, real_arm: bool, platform: str | None = None) -> SyncPolicy:
    """The policy of a session's journal: F_FULLFSYNC required on a real-arm session on macOS,
    best effort otherwise (Linux keeps ``os.fsync``; mock / scripted sessions may fall back)."""
    plat = sys.platform if platform is None else platform
    if real_arm and plat == "darwin":
        return SyncPolicy.FULL_FSYNC_REQUIRED
    return SyncPolicy.BEST_EFFORT


def _err(code: int | None) -> str:
    return str(errno.errorcode.get(code, code)) if code is not None else "?"


def sync_fd(fd: int, policy: SyncPolicy = SyncPolicy.BEST_EFFORT) -> str:
    """Flush ``fd``'s data to stable storage under ``policy`` (module doc); the method used.
    A directory fd always gets the best-effort sync. Raises ``OSError`` when the sync fails
    (``FullFsyncRequired`` when a required F_FULLFSYNC was unavailable or failed)."""
    full: int | None = getattr(fcntl, "F_FULLFSYNC", None)  # macOS only; read per call
    if policy is SyncPolicy.FULL_FSYNC_REQUIRED and not stat.S_ISDIR(os.fstat(fd).st_mode):
        if full is None:
            raise FullFsyncRequired(
                errno.ENOTSUP,
                "F_FULLFSYNC required (real arm, macOS) but fcntl has no F_FULLFSYNC; "
                "no fsync fallback",
            )
        try:
            fcntl.fcntl(fd, full)
        except OSError as e:
            raise FullFsyncRequired(
                e.errno,
                f"F_FULLFSYNC required (real arm, macOS) and it failed: {_err(e.errno)} "
                f"({e.strerror}); no fsync fallback",
            ) from e
        return "F_FULLFSYNC"
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
            return "F_FULLFSYNC"
        except OSError as e:
            if e.errno not in _FULLFSYNC_UNSUPPORTED:
                raise
            os.fsync(fd)
            return f"fsync (F_FULLFSYNC unsupported here: {_err(e.errno)})"
    os.fsync(fd)
    return "fsync"


def sync_path(path: Path, policy: SyncPolicy = SyncPolicy.BEST_EFFORT) -> str:
    """``sync_fd`` on a file or a directory (a directory sync makes a create / rename durable;
    it is best effort under every policy, module doc)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return sync_fd(fd, policy)
    finally:
        os.close(fd)


def probe_full_fsync(directory: Path, policy: SyncPolicy) -> dict[str, Any]:
    """Preflight of a session's journal (module doc): write, sync under ``policy``, and remove a
    temp file in ``directory`` (created if missing), then sync the directory. Returns what ran;
    raises ``OSError`` (``FullFsyncRequired``) when the filesystem cannot do what ``policy``
    requires, so a real-arm session refuses before anything is opened."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".sync-probe-{os.getpid()}-{secrets.token_hex(4)}"
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, b"sync probe\n")
        method = sync_fd(fd, policy)
    finally:
        os.close(fd)
        probe.unlink(missing_ok=True)
    out: dict[str, Any] = {
        "dir": str(directory),
        "policy": str(policy),
        "file": method,
        "directory": sync_path(directory, policy),
        "platform": sys.platform,
    }
    if policy is SyncPolicy.FULL_FSYNC_REQUIRED and out["directory"] != "F_FULLFSYNC":
        out["directory_note"] = (
            f"directory sync is {out['directory']!r}, not F_FULLFSYNC: a rename is durable at "
            "the next journal F_FULLFSYNC (before the next motion), not when it returns"
        )
    return out


def write_durable(path: Path, text: str, policy: SyncPolicy = SyncPolicy.BEST_EFFORT) -> str:
    """Atomic, synced replace of ``path``: tmp + sync + rename + sync of the directory. Returns
    the file's sync method. A failed file sync raises before the rename: ``path`` keeps its last
    synced content."""
    return replace_durable(path, text, policy)[0]


def replace_durable(
    path: Path, text: str, policy: SyncPolicy = SyncPolicy.BEST_EFFORT
) -> tuple[str, str]:
    """``write_durable``; (file sync method, directory sync method)."""
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with os.fdopen(fd, "w", closefd=False) as f:
            f.write(text)
            f.flush()
        method = sync_fd(fd, policy)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    return method, sync_path(path.parent)


def sync_tree(
    root: Path,
    skip: frozenset[str] = frozenset(),
    since: float | None = None,
    policy: SyncPolicy = SyncPolicy.BEST_EFFORT,
) -> int:
    """``sync_path`` on every regular file under ``root`` (with ``since``: only those modified
    at or after that ``time.time()``, less 2 s of mtime slack), then on every directory (deepest
    first, ``root`` last): a tree written through the page cache (e.g. a LeRobot episode save) is
    on stable storage when this returns. Top-level entries named in ``skip`` are left out.
    Returns the number of files synced. Raises ``OSError`` when a sync fails."""
    root = Path(root)
    files = 0
    dirs: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in skip]
            filenames = [f for f in filenames if f not in skip]
        dirs.append(here)
        for name in filenames:
            f = here / name
            if not f.is_file() or f.is_symlink():
                continue
            if since is None or f.stat().st_mtime >= since - 2.0:
                sync_path(f, policy)
                files += 1
    for d in reversed(dirs):
        sync_path(d)
    return files


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(ts: datetime) -> str:
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit and whether the work tree has uncommitted changes."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def device_info(device: str) -> dict[str, Any]:
    """Device actually used, with the accelerator's name (GPU model, Apple chip, or CPU)."""
    info: dict[str, Any] = {
        "device": device,
        "host": socket.gethostname(),
        "os": platform.platform(),
    }
    name: str | None = None
    try:
        import torch

        info["torch"] = torch.__version__
        if device.startswith("cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass
    if name is None and sys.platform == "darwin":
        try:
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            name = None
    info["name"] = name or platform.processor() or platform.machine()
    info["runpod_pod_id"] = os.environ.get("RUNPOD_POD_ID")
    return info


def hourly_usd(dev: dict[str, Any]) -> float:
    """$/h for this run: explicit VLA_RL_HOURLY_USD, else RunPod list price, else 0 (local)."""
    explicit = os.environ.get("VLA_RL_HOURLY_USD")
    if explicit:
        return float(explicit)
    if not dev.get("runpod_pod_id"):
        return 0.0
    name = str(dev.get("name", ""))
    for key, price in GPU_HOURLY_USD.items():
        if key in name:
            return price
    return 0.0


def new_run_id(work_order: str, preset: str) -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    wo = work_order.lower().replace("-", "")
    return f"{stamp}-{wo}-{preset}-{secrets.token_hex(2)}"


class Run:
    """One run: owns `runs/<id>/`, writes `record.json`, appends to `metrics.jsonl`."""

    def __init__(
        self,
        *,
        work_order: str,
        script: str | Path,
        cfg: Config,
        device: str,
        estimated_usd: float = 0.0,
        runs_dir: Path = RUNS_DIR,
        run_id: str | None = None,
        sync_policy: SyncPolicy = SyncPolicy.BEST_EFFORT,
    ) -> None:
        preset = str(cfg.get("preset", "unknown"))
        self.id = run_id or new_run_id(work_order, preset)
        self.dir = Path(runs_dir) / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.results: dict[str, Any] = {}
        self._t0 = time.monotonic()
        self.sync_policy = SyncPolicy(sync_policy)  # fixed for the run's life (module doc)
        self.sync_method: str | None = None  # of the last sync (module doc)
        self._sync_methods: list[str] = []  # every distinct file-sync method used, in order
        self._dir_methods: list[str] = []  # every distinct directory-sync method used (module doc)
        self._dir_entry_synced = False  # runs_dir holds this run's directory durably
        self._metrics_entry_synced = False  # the run directory holds metrics.jsonl durably
        dev = device_info(device)
        script_path = Path(script).resolve()
        try:
            script_rel = str(script_path.relative_to(REPO_ROOT))
        except ValueError:
            script_rel = str(script_path)
        self.record: dict[str, Any] = {
            "run_id": self.id,
            "work_order": work_order,
            "script": script_rel,
            "argv": sys.argv[1:],
            "preset": preset,
            "config_hash": config_hash(cfg),
            "config": cfg,
            "git": git_state(),
            "device": dev,
            "start_utc": _iso(_utc_now()),
            "end_utc": None,
            "wall_clock_s": None,
            "hourly_usd": hourly_usd(dev),
            "estimated_usd": float(estimated_usd),
            "actual_usd": None,
            "status": "running",
            "sync": {
                "policy": str(self.sync_policy),
                "methods": self._sync_methods,
                "dir_methods": self._dir_methods,
            },
            "results": self.results,
        }
        self._write()

    @property
    def record_path(self) -> Path:
        return self.dir / "record.json"

    @property
    def metrics_path(self) -> Path:
        return self.dir / "metrics.jsonl"

    def log_metrics(self, step: int | None = None, **metrics: Any) -> None:
        row: dict[str, Any] = {"t": round(time.monotonic() - self._t0, 3)}
        if step is not None:
            row["step"] = step
        row.update(metrics)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def log_durable(self, step: int | None = None, **metrics: Any) -> str:
        """``log_metrics``, then metrics.jsonl synced to stable storage before returning (and the
        run directory, on the first durable row: the file's entry). Returns the sync method.
        Raises ``OSError`` when the append or the sync fails: the row may or may not be on disk,
        and the caller treats it as not saved."""
        self.log_metrics(step, **metrics)
        method = sync_path(self.metrics_path, self.sync_policy)  # no fallback when required
        self._synced(method)
        if not self._metrics_entry_synced:
            self._dir_synced(sync_path(self.dir))
            self._metrics_entry_synced = True
        return method

    def finish(self, status: str = "ok") -> Path:
        wall = time.monotonic() - self._t0
        self.record["end_utc"] = _iso(_utc_now())
        self.record["wall_clock_s"] = round(wall, 2)
        self.record["actual_usd"] = round(self.record["hourly_usd"] * wall / 3600.0, 4)
        self.record["status"] = status
        self._write()
        return self.record_path

    def flush(self) -> None:
        """Write record.json now (results so far), so a run killed before ``finish`` keeps them."""
        self._write()

    def _write(self) -> None:
        """record.json, atomic and synced (module doc)."""
        method, dir_method = replace_durable(
            self.record_path,
            json.dumps(self.record, indent=2, default=str) + "\n",
            self.sync_policy,
        )
        self._synced(method)
        self._dir_synced(dir_method)
        if not self._dir_entry_synced:  # the run directory's own entry (mkdir) in runs_dir
            self._dir_synced(sync_path(self.dir.parent))
            self._dir_entry_synced = True

    def _synced(self, method: str) -> None:
        self.sync_method = method
        if method not in self._sync_methods:  # record.json's "sync": in the next write
            self._sync_methods.append(method)

    def _dir_synced(self, method: str) -> None:
        if method not in self._dir_methods:  # record.json's "sync.dir_methods" (module doc)
            self._dir_methods.append(method)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.results.setdefault("error", f"{type(exc).__name__}: {exc}")
        self.finish("ok" if exc_type is None else "failed")
