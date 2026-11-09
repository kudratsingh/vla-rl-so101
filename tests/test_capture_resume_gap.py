"""Owner 2026-11-09, the capture / resume gap: a real-0.1 mock session (capture on) is SIGKILLed
after its 3rd trial's SCORE was logged and flushed and before (or while) its episode was saved.
After ``--resume``: three scored trials, the slot never re-run, and the 3rd recording recovered
from its staged journal and attached (or, with the journal gone, flagged ``recording_missing`` and
the export refused as complete / built declared INCOMPLETE). Fakes only: fake arm, fake rig,
MockPolicy behind gRPC on 127.0.0.1, fake clock; the kill is a real SIGKILL of a child process
(tests/capture_kill_child.py)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.config import REPO_ROOT
from src.eval import export_bundle as X
from src.robot import convention as conv
from src.robot import recording
from src.robot.client import ScriptedOperator

HARD_TIMEOUT_S = 300.0
CHILD_TIMEOUT_S = 240.0


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_gap", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_gap"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")
xb = _load("wo07_export_bundle")


class _HardTimeout(BaseException):
    """BaseException: no ``except Exception`` in the session code can swallow it."""


@pytest.fixture
def hard_timeout() -> Iterator[None]:
    """Fail, never hang: SIGALRM after HARD_TIMEOUT_S, re-armed every second after it fires
    (a handler that swallowed the first one meets the next)."""

    def fire(_signum: int, _frame: Any) -> None:
        signal.setitimer(signal.ITIMER_REAL, 1.0)
        raise _HardTimeout(f"test exceeded its {HARD_TIMEOUT_S:.0f} s hard timeout")

    prev = signal.signal(signal.SIGALRM, fire)
    signal.setitimer(signal.ITIMER_REAL, HARD_TIMEOUT_S)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)


def _killed_session(tmp_path: Path, mode: str) -> tuple[Path, Path, Path]:
    """(runs dir, the killed run's dir, checkpoint dir) after the child SIGKILLed itself."""
    runs = tmp_path / "runs"
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"w")
    (ckpt / "config.json").write_text('{"type": "act"}\n')
    child = REPO_ROOT / "tests" / "capture_kill_child.py"
    proc = subprocess.run(
        [sys.executable, str(child), str(runs), str(ckpt), mode],
        check=False,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        cwd=REPO_ROOT,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )
    assert proc.returncode == -signal.SIGKILL, (
        proc.returncode,
        proc.stdout[-2000:],
        proc.stderr[-4000:],
    )
    run_dirs = [p.parent for p in runs.glob("*/record.json")]
    assert len(run_dirs) == 1, run_dirs
    return runs, run_dirs[0], ckpt


def _resume(runs: Path, run_dir: Path) -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec = ev.main(
            ["--preset", "real", "--mock", "--fake-clock", "--resume", run_dir.name,
             "--max-steps", "5", "--runs-dir", str(runs)]
        )  # fmt: skip
    return {"path": Path(rec), **json.loads(Path(rec).read_text())}


def _export(runs: Path, run_id: str, out: Path, *extra: str) -> dict[str, Any]:
    argv = ["--preset", "real", "eval", "--session", run_id, "--runs-dir", str(runs)]
    return dict(xb.main([*argv, "--out", str(out), "--allow-mock", *extra]))


def _check_no_rerun(rec2: dict[str, Any]) -> list[dict[str, Any]]:
    """Three scored trials, one record per slot, none of them run again by the resume."""
    trials = rec2["results"]["trials"]
    scored = [t for t in trials if t.get("policy_started") and t["outcome"] in ev.SCORED]
    assert len(scored) == 3 and len(trials) == 3
    assert len({t["trial_id"] for t in scored}) == 3
    assert all(t["attempt"] == 1 for t in scored)
    metrics = rec2["path"].parent / "metrics.jsonl"
    lines = metrics.read_text().splitlines() if metrics.is_file() else []
    rows = [json.loads(x) for x in lines]
    assert not [r for r in rows if r.get("kind") in ("trial", "trial_started")]  # nothing re-run
    return scored


@pytest.mark.parametrize(
    ("mode", "how"),
    [
        ("after_record", "recovered"),  # the owner's interruption: score saved, episode not
        ("mid_save", "recovered"),  # inside LeRobot's save: data written, nothing finalized
        ("after_save", "filled_from"),  # saved + provenance, its capture row lost
    ],
)
def test_kill_between_score_and_episode_save_is_recovered(
    tmp_path: Path, hard_timeout: None, mode: str, how: str
) -> None:
    runs, run_dir, _ = _killed_session(tmp_path, mode)
    settings, before = ev.load_prior(run_dir)
    assert len(before) == 3 and all(t["outcome"] in ev.SCORED for t in before)
    third = before[-1]
    assert third["capture"]["episode_index"] is None  # the gap: scored, no recording
    capture_root = runs / settings["session_id"] / "capture"
    if mode != "after_save":
        assert recording.has_staged(capture_root, third["trial_id"])  # staged before the score
        assert third["capture"]["stage"]["staged"]

    rec2 = _resume(runs, run_dir)
    scored = _check_no_rerun(rec2)
    idx = {t["trial_id"]: t["capture"]["episode_index"] for t in scored}
    assert sorted(idx.values()) == [0, 1, 2]
    got = next(t for t in scored if t["trial_id"] == third["trial_id"])["capture"]
    assert how in got
    cap = rec2["results"]["capture"]
    assert cap["recording_missing"] == [] and cap["recoverable_on_resume"] == []
    assert cap["reconciled"]["attached" if how == "filled_from" else "recovered"] == [
        third["trial_id"]
    ]
    assert not list((capture_root / recording.STAGED_DIR).glob("*"))  # journal deleted once saved
    prov = json.loads((capture_root / "dataset" / recording.PROVENANCE).read_text())
    for tid, i in idx.items():  # every episode index maps to ITS trial (never re-mapped)
        assert prov["episodes"][str(i)]["trial_id"] == tid
    eps = X.check_dataset(capture_root / "dataset", {"units": conv.DECLARED_UNITS})
    lengths = {int(e["episode_index"]): int(e["length"]) for e in eps}
    assert lengths[idx[third["trial_id"]]] == got["frames"] == third["steps"]

    index = _export(runs, rec2["path"].parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3
    assert not index["protocol_version"].endswith(X.INCOMPLETE_SUFFIX)


def test_unrecoverable_recording_is_flagged_and_the_export_is_incomplete(
    tmp_path: Path, hard_timeout: None
) -> None:
    runs, run_dir, _ = _killed_session(tmp_path, "after_record")
    settings, before = ev.load_prior(run_dir)
    third = before[-1]
    capture_root = runs / settings["session_id"] / "capture"
    shutil.rmtree(recording.staged_path(capture_root, third["trial_id"]))  # nothing to recover

    rec2 = _resume(runs, run_dir)
    scored = _check_no_rerun(rec2)  # the slot keeps its score and is never re-run
    lost = next(t for t in scored if t["trial_id"] == third["trial_id"])
    assert lost["capture"]["recording_missing"] is True
    assert lost["capture"]["episode_index"] is None
    assert "no staged journal" in lost["capture"]["reason"]
    assert lost["outcome"] == third["outcome"]
    assert rec2["results"]["capture"]["recording_missing"] == [third["trial_id"]]
    others = sorted(t["capture"]["episode_index"] for t in scored if t is not lost)
    assert others == [0, 1]

    with pytest.raises(SystemExit, match="REFUSED: INCOMPLETE"):
        _export(runs, rec2["path"].parent.name, tmp_path / "export")
    index = _export(
        runs, rec2["path"].parent.name, tmp_path / "export2", "--allow-missing-recordings"
    )
    assert index["status"] == "INCOMPLETE" and index["episodes"] == 2
    assert index["protocol_version"].endswith(X.INCOMPLETE_SUFFIX)
    assert [m["trial_id"] for m in index["recording_missing"]] == [third["trial_id"]]
    decl = json.loads((tmp_path / "export2" / X.DECLARATION_FILE).read_text())
    assert decl["protocol_version"] == index["protocol_version"]
    trials_json = json.loads((tmp_path / "export2" / X.TRIALS_FILE).read_text())
    assert trials_json["export_status"] == "INCOMPLETE"

    # a second resume changes nothing: the flag stands, no slot is re-run
    rec3 = _resume(runs, rec2["path"].parent)
    _check_no_rerun(rec3)
    assert rec3["results"]["capture"]["recording_missing"] == [third["trial_id"]]
