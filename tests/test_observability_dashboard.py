"""SO-101 dashboard tests: no model, simulator, network or hardware imports."""

import json
import re
from pathlib import Path

import pytest

from observability.dashboard import _wilson, load_runs, write_dashboard


def make_run(
    root: Path,
    name: str,
    rows: list[dict],
    *,
    script: str = "scripts/wo02_eval.py",
    **meta: object,
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    record = {"script": script, "status": "ok", **meta}
    (folder / "record.json").write_text(json.dumps(record))
    (folder / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return folder


def test_reads_grpo_metrics_and_shows_sim_aggregates(tmp_path: Path) -> None:
    make_run(
        tmp_path,
        "test0",
        [
            {
                "kind": "train",
                "step": 3,
                "rollout/frac_groups_dropped": 0.6,
                "train/approx_kl": 0.02,
            },
            {"task": "PickPlace", "cell": "seen", "seed": 0, "episodes": 100, "successes": 20},
            {"task": "PickPlace", "cell": "seen", "seed": 1, "episodes": 100, "successes": 30},
            {"task": "Push", "cell": "shifted", "seed": 2, "episodes": 100, "successes": 2},
        ],
        actual_usd=5.25,
        git={"commit": "b" * 40},
    )
    run = load_runs(tmp_path)[0]
    assert run["series"]["rollout/frac_groups_dropped"] == [[3.0, 0.6]]
    assert run["cost_usd"] == 5.25
    assert run["commit"] == "bbbbbbbb"
    seen = next(r for r in run["benchmarks"] if r["task"] == "PickPlace")
    assert (seen["successes"], seen["episodes"], seen["seeds"], seen["rate"]) == (
        50,
        200,
        2,
        0.25,
    )
    assert seen["ci"][0] < 0.25 < seen["ci"][1]


def test_deduplicates_repeated_sim_results_and_excludes_real_runs(tmp_path: Path) -> None:
    make_run(
        tmp_path,
        "sim",
        [
            {"task": "PickPlace", "cell": "seen", "seed": 0, "episodes": 100, "successes": 1},
            {"task": "PickPlace", "cell": "seen", "seed": 0, "episodes": 100, "successes": 17},
        ],
    )
    make_run(
        tmp_path,
        "real",
        [{"task": "PickPlace", "cell": "seen", "seed": 0, "episodes": 50, "successes": 40}],
        script="scripts/wo07_real_eval.py",
    )
    runs = {r["id"]: r for r in load_runs(tmp_path)}
    assert runs["sim"]["benchmarks"][0]["successes"] == 17
    assert runs["real"]["benchmarks"] == []


def test_filters_sensitive_text_and_nonfinite_values(tmp_path: Path) -> None:
    run = make_run(
        tmp_path,
        "safe",
        [
            {
                "kind": "train",
                "step": 1,
                "loss": 1.1,
                "train/entropy": 0.3,
                "prompt": "MY PRIVATE INSTRUCTION",
                "secret_token": 25.0,
                "train/unsafe<script>": 5.0,
                "train/large": float("inf"),
            },
            {"kind": "error", "traceback": "MY PRIVATE TRACE"},
        ],
        config={"password": "MY PRIVATE CONFIG"},
    )
    with (run / "metrics.jsonl").open("a") as f:
        f.write("{not_finished\n")
    target = tmp_path / "observability" / "index.html"
    assert write_dashboard(tmp_path, target) == 1
    result = target.read_text()
    for forbidden in (
        "MY PRIVATE INSTRUCTION",
        "MY PRIVATE TRACE",
        "MY PRIVATE CONFIG",
        "secret_token",
        "train/unsafe<script>",
    ):
        assert forbidden not in result
    match = re.search(r'<script id="data" type="application/json">(.*?)</script>', result)
    assert match is not None
    assert '"loss"' in match.group(1)
    assert "Infinity" not in match.group(1)


def test_refuses_writes_inside_any_run_directory(tmp_path: Path) -> None:
    run = make_run(tmp_path, "training_run", [{"step": 1, "loss": 0.6}])
    original = (run / "record.json").read_bytes()
    with pytest.raises(ValueError, match="runs/observability"):
        write_dashboard(tmp_path, run / "record.json")
    assert (run / "record.json").read_bytes() == original


def test_does_not_follow_symlinks_or_depend_on_cloud(tmp_path: Path) -> None:
    real = tmp_path / "actual"
    make_run(real, "training", [{"step": 1, "train/pg_loss": 1.1}])
    root = tmp_path / "runs"
    root.mkdir()
    (root / "linked").symlink_to(real / "training", target_is_directory=True)
    assert load_runs(root) == []
    target = tmp_path / "output.html"
    assert write_dashboard(root, target) == 0
    assert "SO-101 Observatory" in target.read_text()
    assert "https://" not in target.read_text()


def test_wilson_interval_handles_zero_and_all_success() -> None:
    lo, hi = _wilson(0, 100)
    assert lo == 0 and 0 < hi < 0.1
    lo, hi = _wilson(100, 100)
    assert hi == 1 and 0.9 < lo < 1
