import json
from pathlib import Path

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
