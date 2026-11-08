"""Smoke test: the WO-01r CPU dry run (tiny preset) builds demos with the release/retreat tail and
re-renders them; CPU against CPU, the render gap must equal the codec floor (no physics drift)."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.config import REPO_ROOT
from src.data.recorder import read_sidecar
from src.runs import Run

pytestmark = pytest.mark.needs_render


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tiny_dry_run_builds_full_tails_and_zero_render_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "tiny_rebuild.yaml"
    cfg.write_text(f"inherit: configs/tiny.yaml\ndata:\n  dataset_root: {tmp_path / 'demos'}\n")
    runs = tmp_path / "runs"
    build = _load("wo01_build_env")
    rebuild = _load("wo01r_rebuild_demos")
    monkeypatch.setattr(build, "Run", functools.partial(Run, runs_dir=runs))
    monkeypatch.setattr(rebuild, "Run", functools.partial(Run, runs_dir=runs))
    monkeypatch.setattr(rebuild, "_build_env_module", lambda: build)

    assert rebuild.main(["--preset", "tiny", "--config", str(cfg)]) == 0

    (record,) = [
        json.loads((d / "record.json").read_text()) for d in runs.iterdir() if "wo01r" in d.name
    ]
    res = record["results"]
    tail = res["tail"]["PickPlace"]
    assert tail["demos"] == 2 and tail["end_at_release"] == 2 and tail["oracle_finished"] == 2
    assert tail["mean_post_success_frames"] >= 15 and tail["min_tcp_rise_after_success_cm"] > 2
    rows = read_sidecar(tmp_path / "demos" / "vla-rl-so101-demos-tiny")
    assert all(r["num_frames"] == r["success_step"] + r["post_success_frames"] for r in rows)
    assert {r["oracle_profile"] for r in rows} == {"v2"}

    gap = res["render_gap"]["tasks"]["PickPlace"]
    assert gap["state_max_abs_diff_t0_rad"] == 0.0 and gap["state_mean_max_abs_diff_rad"] == 0.0
    floor = gap["codec_floor"]
    assert gap["pixels_all"]["mean_abs"] == pytest.approx(floor["pixels"]["mean_abs"], abs=0.05)
    tokens = gap["tokens_stored_vs_fresh"]
    assert tokens["frames"] > 0
    assert tokens["token_agreement"] == pytest.approx(
        floor["tokens_raw_vs_decoded"]["token_agreement"], abs=0.02
    )
