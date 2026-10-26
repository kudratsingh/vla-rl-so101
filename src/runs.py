"""Run records (CLAUDE.md rule 1): `runs/<id>/record.json` plus a `metrics.jsonl` appender.

Usage:

    with Run(work_order="WO-01", script=__file__, cfg=cfg, device="cpu") as run:
        run.log_metrics(step=0, loss=1.0)
        run.results["oracle_success"] = 0.97

`record.json` is written at start (status "running"), and rewritten on exit with the end time,
wall-clock, dollars, results and status ("ok" or "failed"). Only `record.json` is committed;
everything else under `runs/` is ignored by git.
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
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
    ) -> None:
        preset = str(cfg.get("preset", "unknown"))
        self.id = run_id or new_run_id(work_order, preset)
        self.dir = Path(runs_dir) / self.id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.results: dict[str, Any] = {}
        self._t0 = time.monotonic()
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

    def finish(self, status: str = "ok") -> Path:
        wall = time.monotonic() - self._t0
        self.record["end_utc"] = _iso(_utc_now())
        self.record["wall_clock_s"] = round(wall, 2)
        self.record["actual_usd"] = round(self.record["hourly_usd"] * wall / 3600.0, 4)
        self.record["status"] = status
        self._write()
        return self.record_path

    def _write(self) -> None:
        tmp = self.record_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.record, indent=2, default=str) + "\n")
        tmp.replace(self.record_path)

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
