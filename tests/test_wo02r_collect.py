"""scripts/wo02r_collect_corrections.py end to end on the tiny preset (CPU sim, tiny random OFT):
rollout -> classify -> restore -> oracle -> record, and the control mode. Never pushes."""

from __future__ import annotations

import json
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
    dir_root = Path(".")

    def __init__(self, **kw: Any) -> None:
        self.id, self.results, self.dir = "collect-test", {}, self.dir_root
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
    monkeypatch.setattr(_Run, "dir_root", tmp_path)
    cfg = tmp_path / "tiny.yaml"
    # record like configs/demos_v2_tiny.yaml: oracle v2 and the release/retreat tail
    cfg.write_text(
        f"inherit: configs/sft_tiny.yaml\ndata:\n  dataset_root: {tmp_path}\n"
        "  oracle_profile: v2\n  post_success_steps: 15\n  post_success_max_steps: 45\n"
    )

    out = script.main(["--preset", "tiny", "--config", str(cfg)])
    stats = out["tasks"]["PickPlace"]
    assert stats["rollouts"] == 2 and stats["sft_success_rate"] == 0.0
    assert sum(stats["failure_modes"].values()) == 2 and stats["tried"] >= 1
    assert stats["restore_max_err"] < 1e-4 and out["dataset"]["episodes"] == stats["tried"]
    rows = read_sidecar(tmp_path / "vla-rl-so101-correction-tiny")
    assert rows and all(r["kind"] == "correction" and r["cell"] == "seen" for r in rows)
    assert all(r["seed"] >= 3_000_000_000 and r["failure_mode"] for r in rows)
    assert all(r["branch_step"] <= r["event_step"] for r in rows)
    assert all(r["oracle_profile"] == "v2" and r["protocol"] == "0.2" for r in rows)
    assert out["protocol_version"] == "0.2" and out["action_clip_rad"] > 0
    eps = [json.loads(x) for x in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 2 and all(e["seed"] == -1 and e["cell"] == "seen" for e in eps)
    assert all(3_000_000_000 <= e["episode_seed"] < 3_001_000_000 for e in eps)

    out = script.main(["--preset", "tiny", "--config", str(cfg), "--mode", "control"])
    assert out["tasks"]["PickPlace"]["control_demos"] == 2 and out["source_policy"] is None
    bad = tmp_path / "bad.yaml"  # PR 1's 0.1 Hub denylist holds for the collector too
    bad.write_text(cfg.read_text() + "recovery:\n  hub_repo: KudratSingh22/vla-rl-so101-demos-25\n")
    with pytest.raises(ValueError, match="0.1-era demo set"):
        script.main(["--preset", "tiny", "--config", str(bad), "--push"])
    rows = read_sidecar(tmp_path / "vla-rl-so101-control-tiny")
    assert [r["kind"] for r in rows] == ["control_demo"] * 2
    ok = [r for r in rows if r["oracle_success"]]
    assert ok and all(r["post_success_frames"] >= 15 for r in ok)  # release + retreat recorded
    assert all(r["num_frames"] == r["success_step"] + r["post_success_frames"] for r in ok)
    assert all(3_100_000_000 <= r["seed"] < 3_101_000_000 for r in rows)
