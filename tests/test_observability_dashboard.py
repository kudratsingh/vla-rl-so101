"""SO-101 dashboard tests: no model, simulator, network or hardware imports."""

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from observability.dashboard import _wilson, load_runs, write_dashboard


def make_run(
    root: Path,
    name: str,
    rows: list[dict[str, Any]],
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
        results={"protocol_hash": "a" * 64},
    )
    run = load_runs(tmp_path)[0]
    assert run["series"]["rollout/frac_groups_dropped"] == [[3.0, 0.6]]
    assert run["cost_usd"] == 5.25
    assert run["commit"] == "bbbbbbbb"
    assert run["protocol_hash"] == "a" * 64
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
        results={"protocol_hash": "a" * 64},
    )
    runs = {r["id"]: r for r in load_runs(tmp_path)}
    assert runs["sim"]["benchmarks"][0]["successes"] == 17
    assert runs["real"]["benchmarks"] == []
    assert runs["real"]["protocol_hash"] is None


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


@pytest.mark.parametrize("protocol_hash", [None, "bad<script>", "a" * 63, {"hash": "a" * 64}])
def test_rejects_invalid_protocol_hashes(tmp_path: Path, protocol_hash: object) -> None:
    make_run(
        tmp_path,
        "sim",
        [],
        results={"protocol_hash": protocol_hash, "private": "PRIVATE RESULT"},
    )
    target = tmp_path / "observability/index.html"
    write_dashboard(tmp_path, target)
    assert load_runs(tmp_path)[0]["protocol_hash"] is None
    assert "PRIVATE RESULT" not in target.read_text()
    assert "bad<script>" not in target.read_text()


@pytest.mark.parametrize("candidate_hash", ["a" * 64, "b" * 64, None])
def test_comparison_requires_matching_protocol_hashes(
    tmp_path: Path, candidate_hash: str | None
) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to exercise dashboard interactions")
    for name, protocol_hash, successes in [
        ("baseline", "a" * 64, 20),
        ("candidate", candidate_hash, 90),
    ]:
        make_run(
            tmp_path,
            name,
            [
                {
                    "task": "PickPlace",
                    "cell": "seen",
                    "seed": 0,
                    "episodes": 100,
                    "successes": successes,
                }
            ],
            results={"protocol_hash": protocol_hash},
        )
    target = tmp_path / "observability/index.html"
    write_dashboard(tmp_path, target)
    html = target.read_text()
    payload = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.DOTALL
    )
    scripts = re.findall(r"<script>(.*?)</script>", html, re.DOTALL)
    assert payload is not None and scripts
    # Execute the actual report in a small DOM, then inspect the visible comparison cells.
    harness = """
class Element {
    constructor(){this.children=[];this.textContent="";this.value="";this.style={};}
    appendChild(child){this.children.push(child);return child;}
    append(...children){this.children.push(...children);}
    replaceChildren(){this.children=[];}
    setAttribute(){}
    addEventListener(){}
}
const elements=new Map();
globalThis.document={
    getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},
    createElement(){return new Element();},
    createElementNS(){return new Element();}
};
document.getElementById("data").textContent=__DATA__;
""".replace("__DATA__", json.dumps(payload.group(1)))
    inspect = """
baseBox.value="baseline";candBox.value="candidate";renderBench();
console.log(JSON.stringify({
    status:document.getElementById("protocolStatus").textContent,
    row:document.getElementById("benchmarkBody").children[0].children.map(c=>c.textContent)
}));
"""
    script = tmp_path / "dashboard-check.js"
    script.write_text(harness + "\n".join(scripts) + inspect)
    result = subprocess.run(
        [node, str(script)], check=True, capture_output=True, text=True, timeout=10
    )
    displayed = json.loads(result.stdout)
    assert "Baseline baseline protocol: " + "a" * 64 in displayed["status"]
    assert displayed["row"][2].startswith("20.0%")
    if candidate_hash is not None:
        assert "Candidate candidate protocol: " + candidate_hash in displayed["status"]
        assert displayed["row"][3].startswith("90.0%")
    else:
        assert "success rates hidden" in displayed["status"]
        assert displayed["row"][3] == "Unverified protocol"
    if candidate_hash == "a" * 64:
        assert displayed["row"][4] == "+70.0"
    else:
        assert displayed["row"][4] == "—"
        assert "delta withheld" in displayed["status"]
        if candidate_hash is not None:
            assert "Protocol mismatch" in displayed["status"]
