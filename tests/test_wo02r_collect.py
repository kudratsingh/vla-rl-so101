"""scripts/wo02r_collect_corrections.py end to end on the tiny preset (CPU sim, tiny random OFT):
rollout -> classify -> restore -> oracle -> record, and the control mode. Never pushes."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Self

import pytest

from src.data.recorder import read_sidecar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
pytestmark = pytest.mark.needs_render


class _Run:
    def __init__(self, **kw: Any) -> None:
        self.id, self.results, self.dir = "collect-test", {}, Path(".")
        self.metrics: list[dict[str, Any]] = []

    def log_metrics(self, step: int | None = None, **m: Any) -> None:
        self.metrics.append(m)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_collect_corrections_and_control_tiny(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wo02r_collect_corrections as script

    runs = types.ModuleType("src.runs")
    runs.Run = _Run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.runs", runs)
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(f"inherit: configs/sft_tiny.yaml\ndata:\n  dataset_root: {tmp_path}\n")

    out = script.main(["--preset", "tiny", "--config", str(cfg)])
    stats = out["tasks"]["PickPlace"]
    assert stats["rollouts"] == 2 and stats["sft_success_rate"] == 0.0
    assert sum(stats["failure_modes"].values()) == 2 and stats["tried"] >= 1
    assert stats["restore_max_err"] < 1e-4 and out["dataset"]["episodes"] == stats["tried"]
    rows = read_sidecar(tmp_path / "vla-rl-so101-correction-tiny")
    assert rows and all(r["kind"] == "correction" and r["cell"] == "seen" for r in rows)
    assert all(r["seed"] >= 3_000_000_000 and r["failure_mode"] for r in rows)
    assert all(r["branch_step"] <= r["event_step"] for r in rows)

    out = script.main(["--preset", "tiny", "--config", str(cfg), "--mode", "control"])
    assert out["tasks"]["PickPlace"]["control_demos"] == 2 and out["source_policy"] is None
    rows = read_sidecar(tmp_path / "vla-rl-so101-control-tiny")
    assert [r["kind"] for r in rows] == ["control_demo"] * 2
    assert all(3_100_000_000 <= r["seed"] < 3_101_000_000 for r in rows)
