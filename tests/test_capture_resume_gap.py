"""Owner 2026-11-09, the capture / resume gap: a real-0.1 mock session (capture on) is SIGKILLed
after its 3rd trial's SCORE was logged and flushed and before (or while) its episode was saved.
After ``--resume``: three scored trials, the slot never re-run, and the 3rd recording recovered
from its staged journal and attached (or, with the journal gone, flagged ``recording_missing`` and
the export refused as complete / built declared INCOMPLETE). Fakes only: fake arm, fake rig,
MockPolicy behind gRPC on 127.0.0.1, fake clock; the kill is a real SIGKILL of a child process
(tests/capture_kill_child.py).

Review round 2: the recovered episode is compared with the episode as captured before the kill
(state, action, timing exactly; the first and last frame of each camera by their per-read stamp),
a kill at trial 2 of 3 recovers episode 1 before the resume records trial 3 as episode 2 (F5);
a client error after the start stages the open episode under its trial id (F1); a trial whose
first write faulted needs no recording (F2); a transient save failure during the recovery is
retried by the next resume (F3); a not-started attempt and its retry in the same second keep
their own capture blocks (F4)."""

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

import numpy as np
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


def _ckpt(tmp_path: Path) -> Path:
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"w")
    (ckpt / "config.json").write_text('{"type": "act"}\n')
    return ckpt


def _killed_session(tmp_path: Path, mode: str, kill_at: int = 3) -> tuple[Path, Path, Path]:
    """(runs dir, the killed run's dir, checkpoint dir) after the child SIGKILLed itself."""
    runs = tmp_path / "runs"
    ckpt = _ckpt(tmp_path)
    child = REPO_ROOT / "tests" / "capture_kill_child.py"
    proc = subprocess.run(
        [sys.executable, str(child), str(runs), str(ckpt), mode, str(kill_at)],
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


def _new_session(runs: Path, ckpt: Path) -> Path:
    """A real-0.1 mock session in process (3 trials, capture on); its record path."""
    argv = ["--preset", "real", "--mock", "--fake-clock", "--max-steps", "5", "--task",
            "PickPlace", "--trials", "1", "--labeler", "mock-labeler", "--family", "act",
            "--training-run-id", "run-act", "--checkpoint-dir", str(ckpt), "--runs-dir",
            str(runs)]  # fmt: skip
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        return Path(ev.main(argv))


def _only_run(runs: Path) -> Path:
    (rec,) = runs.glob("*/record.json")
    return rec.parent


def _assert_episode_is_the_captured_one(
    runs: Path, session_id: str, idx: int, expected_tid: str
) -> None:
    """Review F5: the saved episode ``idx`` IS the trial's capture as it stood before the kill
    (tests/capture_kill_child.py ``expected.npz``): state, action and timing exactly; the first
    and last frame of each camera by the per-read stamp (AV1 is lossy; the stamps of two reads
    differ by >= 7 grey levels, of two episodes by many more)."""
    exp = np.load(runs / "expected.npz")
    assert json.loads((runs / "expected.json").read_text())["trial_id"] == expected_tid
    root = runs / session_id / "capture" / "dataset"
    rows = sorted(
        (r for r in X._read_parquet_dir(root / "data") if int(r["episode_index"]) == idx),
        key=lambda r: int(r["frame_index"]),
    )
    assert len(rows) == len(exp["state"]) > 0
    for key, name in ((conv.STATE_KEY, "state"), (conv.ACTION_KEY, "action")):
        got = np.array([r[key] for r in rows], dtype=np.float32)
        np.testing.assert_array_equal(got, exp[name])
    np.testing.assert_array_equal(
        np.array([r[conv.TIMING_KEY] for r in rows], dtype=np.float32), exp["timing"]
    )
    ds = recording.open_dataset(ev.capture_repo_id(session_id), root)
    inner = (slice(16, 80), slice(16, 80))  # inside the stamped 96 x 96 block
    for name, key in conv.IMAGE_KEYS.items():
        for tag, row in (("first", rows[0]), ("last", rows[-1])):
            img = ds[int(row["index"])][key]  # float CHW in [0, 1], decoded with torchcodec
            got = float(img.permute(1, 2, 0).numpy()[inner].mean()) * 255.0
            want = float(exp[f"{name}_{tag}"][inner].mean())
            assert abs(got - want) < 3.0, (name, tag, got, want)
    assert float(exp["front_first"][inner].mean()) != float(exp["front_last"][inner].mean())


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
    _assert_episode_is_the_captured_one(
        runs, settings["session_id"], idx[third["trial_id"]], third["trial_id"]
    )

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


def test_kill_at_trial_2_recovers_episode_1_then_records_trial_3(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F5: the recovery runs BEFORE any new trial, so the recovered trial 2 is episode 1
    and the resume's own trial 3 is episode 2; each index maps to its own trial."""
    runs, run_dir, _ = _killed_session(tmp_path, "after_record", kill_at=2)
    settings, before = ev.load_prior(run_dir)
    assert len(before) == 2 and before[-1]["capture"]["episode_index"] is None
    second = before[-1]["trial_id"]

    rec2 = _resume(runs, run_dir)
    trials = rec2["results"]["trials"]
    scored = [t for t in trials if t.get("policy_started") and t["outcome"] in ev.SCORED]
    assert len(scored) == 3 == len(trials) and all(t["attempt"] == 1 for t in scored)
    idx = {t["trial_id"]: t["capture"]["episode_index"] for t in scored}
    third = next(
        t["trial_id"] for t in scored if t["trial_id"] not in (before[0]["trial_id"], second)
    )
    assert (idx[before[0]["trial_id"]], idx[second], idx[third]) == (0, 1, 2)
    assert rec2["results"]["capture"]["reconciled"]["recovered"] == [second]
    rows = [json.loads(x) for x in (rec2["path"].parent / "metrics.jsonl").read_text().splitlines()]
    assert [r["trial_id"] for r in rows if r.get("kind") == "trial"] == [third]  # only trial 3 ran
    prov = json.loads(
        (runs / settings["session_id"] / "capture" / "dataset" / recording.PROVENANCE).read_text()
    )
    assert {i: prov["episodes"][str(i)]["trial_id"] for i in range(3)} == {
        0: before[0]["trial_id"],
        1: second,
        2: third,
    }
    _assert_episode_is_the_captured_one(runs, settings["session_id"], 1, second)
    index = _export(runs, rec2["path"].parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def test_client_error_after_the_start_stages_the_open_episode_by_its_trial_id(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F1: a client error after trial 2's policy started (the process lives) appends its
    ``session_lost`` record WITH its trial id, so the session end stages the open episode and the
    resume recovers it (it was flagged ``recording_missing`` under trial id None before)."""
    import src.robot.client as C

    runs, ckpt = tmp_path / "runs", _ckpt(tmp_path)
    real_score = C._score
    calls = {"n": 0}

    def flaky_score(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("client bug while scoring")
        return real_score(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(C, "_score", flaky_score)
        with pytest.raises(RuntimeError, match="client bug while scoring"):
            _new_session(runs, ckpt)
    run_dir = _only_run(runs)
    res = json.loads((run_dir / "record.json").read_text())["results"]
    lost = res["trials"][-1]
    assert lost["fault"]["kind"] == "session_lost" and lost["policy_started"]
    tid = lost["trial_id"]
    assert tid is not None and tid.endswith(f"/{lost['cell']}/{int(lost['trial_idx']):03d}")
    assert lost["capture"]["stage"]["staged"] and lost["capture"]["stage"]["frames"] == 5
    capture_root = runs / res["session"]["session_id"] / "capture"
    assert recording.has_staged(capture_root, tid)
    assert res["capture"]["recoverable_on_resume"] == [tid]
    assert res["capture"]["recording_missing"] == []

    rec2 = _resume(runs, run_dir)
    assert rec2["results"]["capture"]["reconciled"]["recovered"] == [tid]
    trials = rec2["results"]["trials"]
    got = next(t for t in trials if t["trial_id"] == tid)
    assert got["capture"]["episode_index"] == 1 and got["capture"]["frames"] == 5
    assert len([t for t in trials if t.get("policy_started")]) == 3
    index = _export(runs, rec2["path"].parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def test_a_trial_whose_first_write_faulted_needs_no_recording(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F2: trial 3's first policy write fails on the bus (the trial started: the action
    went to the filter) before any frame was captured. It is a scored failure with nothing to
    record: no RECORDING MISSING, and the export is COMPLETE with the trial listed under
    ``uncaptured_started_trials``."""
    from robot_fakes import FakeFollower

    from src.robot.client import DeployLoop

    runs, ckpt = tmp_path / "runs", _ckpt(tmp_path)
    real_run, real_send = DeployLoop.run_trial, FakeFollower.send_action
    armed = {"trial": 0, "fail": False}

    def run_trial(self: Any, *a: Any, **k: Any) -> Any:
        armed["trial"] += 1
        armed["fail"] = armed["trial"] == 3
        return real_run(self, *a, **k)

    def send_action(self: Any, action: dict[str, Any]) -> dict[str, Any]:
        if armed["fail"]:
            armed["fail"] = False
            raise ConnectionError("Failed to sync write 'Goal_Position' (test)")
        return dict(real_send(self, action))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DeployLoop, "run_trial", run_trial)
        mp.setattr(FakeFollower, "send_action", send_action)
        rec = _new_session(runs, ckpt)
    res = json.loads(rec.read_text())["results"]
    faulted = res["trials"][-1]
    assert faulted["policy_started"] and faulted["outcome"] == "failure"
    assert faulted["steps"] == 0
    assert faulted["capture"]["frames"] == 0 and faulted["capture"]["episode_index"] is None
    assert faulted["capture"]["reason"] == recording.NO_POLICY_WRITE
    assert not recording.recording_required(faulted)
    assert res["capture"]["recording_missing"] == []
    assert res["capture"]["recoverable_on_resume"] == []

    index = _export(runs, rec.parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 2
    assert index["uncaptured_started_trials"] == 1 and index["recording_missing"] == []
    trials_json = json.loads((tmp_path / "export" / X.TRIALS_FILE).read_text())
    assert [u["trial_id"] for u in trials_json["uncaptured_started_trials"]] == [
        faulted["trial_id"]
    ]


def test_a_transient_save_failure_during_recovery_is_retried_by_the_next_resume(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F3: the first resume's recovery hits a transient save error: the trial is NOT
    flagged ``recording_missing`` (the journal is whole and kept), the session and the export
    agree it is recoverable on resume, and the next resume recovers it: export COMPLETE."""
    runs, run_dir, _ = _killed_session(tmp_path, "after_record")
    settings, before = ev.load_prior(run_dir)
    tid = before[-1]["trial_id"]
    capture_root = runs / settings["session_id"] / "capture"
    real_save = recording.DurableDataset.save_episode
    once = {"n": 0}

    def save_once_fails(self: Any, before_checkpoint: Any = None) -> int:
        once["n"] += 1
        if once["n"] == 1:
            raise OSError("disk full (transient)")
        return int(real_save(self, before_checkpoint))

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(recording.DurableDataset, "save_episode", save_once_fails)
        rec2 = _resume(runs, run_dir)
    got = next(t for t in rec2["results"]["trials"] if t["trial_id"] == tid)["capture"]
    assert got["episode_index"] is None and got["recovery_failed"] is True
    assert not got.get("recording_missing")
    cap = rec2["results"]["capture"]
    assert cap["reconciled"]["recovery_failed"] == [tid]
    assert cap["recoverable_on_resume"] == [tid] and cap["recording_missing"] == []
    assert recording.has_staged(capture_root, tid)
    with pytest.raises(SystemExit, match="staged: --resume"):
        _export(runs, rec2["path"].parent.name, tmp_path / "export")

    rec3 = _resume(runs, rec2["path"].parent)
    _check_no_rerun(rec3)
    assert rec3["results"]["capture"]["reconciled"]["recovered"] == [tid]
    got = next(t for t in rec3["results"]["trials"] if t["trial_id"] == tid)["capture"]
    assert got["episode_index"] == 2
    _assert_episode_is_the_captured_one(runs, settings["session_id"], 2, tid)
    index = _export(runs, rec3["path"].parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def test_a_same_second_retry_keeps_each_attempt_its_own_capture(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F4: trial 1's first attempt never starts (the move home fails) and is retried in
    the same second (``start_utc`` has one-second resolution; pinned here). ``load_prior`` keys
    rows by attempt too and merges a capture row only into a started record: the export holds
    each episode once."""
    import src.robot.client as C
    from src.robot.client import DeployLoop
    from src.robot.driver import SerialFault

    runs, ckpt = tmp_path / "runs", _ckpt(tmp_path)
    real_home = DeployLoop.go_home
    homes = {"n": 0}

    def go_home(self: Any) -> Any:
        homes["n"] += 1
        if homes["n"] == 1:
            raise SerialFault("home write failed (test)")
        return real_home(self)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DeployLoop, "go_home", go_home)
        mp.setattr(C, "utc_now", lambda: "2026-11-09T12:00:00Z")
        rec = _new_session(runs, ckpt)
    _, trials = ev.load_prior(rec.parent)
    first = [t for t in trials if t["trial_id"] == trials[0]["trial_id"]]
    assert [(t["attempt"], bool(t["policy_started"])) for t in first] == [(1, False), (2, True)]
    assert first[0]["start_utc"] == first[1]["start_utc"]  # the same second
    assert first[0]["capture"].get("episode_index") is None
    assert first[1]["capture"]["episode_index"] == 0
    index = _export(runs, rec.parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


# -- PR #33 x the capture: a retracted start never has (or lends) a recording ---------------------
def test_a_retracted_start_has_no_recording_and_its_retry_records_its_own(
    tmp_path: Path, hard_timeout: None
) -> None:
    """The first trial's ``trial_started`` row takes 400 ms on the fake clock: the frame goes
    stale, the first goal is blocked and the marker retracted (``not_started``, attempt 1). The
    in-place retry (attempt 2) starts and is captured as its own episode. Attempt 1 has no
    recording and nothing staged; its retraction row carries its attempt and trial id; resume
    keys it apart from attempt 2; the export is COMPLETE with one episode per started trial."""
    from src.robot.client import DeployLoop
    from src.runs import Run

    loops: list[DeployLoop] = []

    class SlowMarkerLoop(DeployLoop):
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            loops.append(self)

    real_log = Run.log_metrics
    slowed = [0]

    def log_metrics(run: Run, step: int | None = None, **row: Any) -> None:
        if row.get("kind") == "trial_started" and not slowed[0]:
            slowed[0] = 1
            loops[-1].clock.advance(0.4)  # type: ignore[attr-defined]
        real_log(run, step, **row)

    runs, ckpt = tmp_path / "runs", _ckpt(tmp_path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "DeployLoop", SlowMarkerLoop)
        mp.setattr(Run, "log_metrics", log_metrics)
        rec = _new_session(runs, ckpt)
    r1 = json.loads(rec.read_text())["results"]
    tid = r1["trials"][0]["trial_id"]
    first = [t for t in r1["trials"] if t["trial_id"] == tid]
    assert [(t["attempt"], bool(t["policy_started"])) for t in first] == [(1, False), (2, True)]
    assert first[0]["stop_reason"] == "deadman" and "start marker took" in first[0]["stop_detail"]
    assert first[0]["capture"]["episode_index"] is None
    assert "not started" in first[0]["capture"]["reason"] and not first[0]["capture"]["frames"]
    assert not recording.recording_required(first[0])
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    (retracted,) = [r for r in rows if r["kind"] == "trial_start_retracted"]
    assert (retracted["trial_id"], retracted["attempt"]) == (tid, 1)
    capture_root = runs / r1["session"]["session_id"] / "capture"
    assert not list((capture_root / recording.STAGED_DIR).glob("*"))  # nothing staged, ever
    _, trials = ev.load_prior(rec.parent)
    again = [t for t in trials if t["trial_id"] == tid]
    assert [(t["attempt"], t["capture"].get("episode_index")) for t in again] == [(1, None), (2, 0)]
    eps = X.check_dataset(capture_root / "dataset", {"units": conv.DECLARED_UNITS})
    lengths = {int(e["episode_index"]): int(e["length"]) for e in eps}
    assert lengths[0] == again[1]["capture"]["frames"] == again[1]["steps"]
    prov = json.loads((capture_root / "dataset" / recording.PROVENANCE).read_text())
    assert prov["episodes"]["0"]["trial_id"] == tid and len(prov["episodes"]) == 3
    index = _export(runs, rec.parent.name, tmp_path / "export")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def test_a_journal_of_another_attempt_or_a_not_started_slot_is_discarded(tmp_path: Path) -> None:
    """A staged journal is never attached to an attempt it was not written for: the resume's
    reconcile discards the journal of a slot none of whose attempts started (e.g. a retracted
    start), and ``recover_staged`` discards one whose recorded attempt differs from the record it
    would be recovered for (``journal_mismatch``; journals without ``attempt`` match on the rest).
    """
    tid = "sess/seen/000"
    d = recording.staged_path(tmp_path, tid)
    d.mkdir(parents=True)
    rec1 = {"cell": "seen", "trial_idx": 0, "start_utc": "2026-11-09T12:00:00Z", "attempt": 1}
    (d / recording.STAGED_INDEX).write_text(json.dumps({"trial_id": tid, "record": rec1}))

    class Cap:
        def __init__(self) -> None:
            self.root = tmp_path
            self.provenance: dict[str, Any] = {"episodes": {}}
            self.dataset = None  # no episode count: nothing claims one here

        def drop_staged(self, trial_id: str) -> None:
            shutil.rmtree(recording.staged_path(tmp_path, trial_id), ignore_errors=True)

        def recover_staged(self, trial_id: str, expect: Any = None) -> dict[str, Any]:
            raise AssertionError("a not-started slot is never recovered")

    logged: list[dict[str, Any]] = []
    trials = [{**rec1, "trial_id": tid, "policy_started": False, "outcome": "not_started"}]
    out = ev.reconcile_captures(Cap(), trials, lambda **row: logged.append(row))  # type: ignore[arg-type]
    assert out["discarded"] == [{"trial_id": tid, "reason": recording.STAGED_NOT_STARTED}]
    assert not recording.has_staged(tmp_path, tid) and logged == []

    assert recording.journal_mismatch(rec1, {**rec1, "attempt": 2}) == "attempt 1 != 2"
    assert recording.journal_mismatch(rec1, rec1) == ""
    no_attempt = {k: v for k, v in rec1.items() if k != "attempt"}
    assert recording.journal_mismatch(no_attempt, {**rec1, "attempt": 2}) == ""
    assert recording.journal_mismatch(rec1, None) == ""


def test_a_staged_journal_of_another_attempt_is_not_recovered_into_this_one(
    tmp_path: Path, hard_timeout: None
) -> None:
    """End to end: the run was killed after trial 3's score, its journal staged. The journal is
    made to name another attempt of the slot; the resume does NOT attach it: it is discarded
    and trial 3 is flagged ``recording_missing`` (the export refuses a complete bundle)."""
    runs, run_dir, _ = _killed_session(tmp_path, "after_record")
    settings, before = ev.load_prior(run_dir)
    third = before[-1]
    capture_root = runs / settings["session_id"] / "capture"
    index = recording.staged_path(capture_root, third["trial_id"]) / recording.STAGED_INDEX
    meta = json.loads(index.read_text())
    assert meta["record"] == {k: third.get(k) for k in recording.JOURNAL_KEYS}
    meta["record"]["attempt"] = int(third["attempt"]) + 1
    index.write_text(json.dumps(meta))

    rec2 = _resume(runs, run_dir)
    got = next(t for t in rec2["results"]["trials"] if t["trial_id"] == third["trial_id"])
    assert got["capture"]["episode_index"] is None and got["capture"]["recording_missing"]
    assert recording.STAGED_OTHER_ATTEMPT in got["capture"]["reason"]
    assert not recording.has_staged(capture_root, third["trial_id"])
    assert rec2["results"]["capture"]["reconciled"]["recording_missing"] == [third["trial_id"]]


def _lose_episode(ds_root: Path, idx: int) -> None:
    """Power loss before episode ``idx`` (the newest) reached the disk: its data / video /
    episode-metadata files and its provenance entry are gone (the page cache held them)."""
    import pyarrow.parquet as pq

    info = json.loads((ds_root / "meta" / "info.json").read_text())
    (meta,) = [
        f
        for f in sorted((ds_root / "meta" / "episodes").rglob("*.parquet"))
        if idx in [int(r["episode_index"]) for r in pq.read_table(f).to_pylist()]
    ]
    (ep,) = pq.read_table(meta).to_pylist()  # its own file (one file per saved episode)
    assert int(ep["episode_index"]) == idx
    gone = [meta, ds_root / str(info["data_path"]).format(
        chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])]  # fmt: skip
    for k, v in info["features"].items():
        if v.get("dtype") == "video":
            gone.append(ds_root / str(info["video_path"]).format(
                video_key=k, chunk_index=ep[f"videos/{k}/chunk_index"],
                file_index=ep[f"videos/{k}/file_index"]))  # fmt: skip
    for f in gone:
        f.unlink()
    prov_path = ds_root / recording.PROVENANCE
    prov = json.loads(prov_path.read_text())
    del prov["episodes"][str(idx)]
    prov_path.write_text(json.dumps(prov))


def test_a_recorded_episode_lost_to_a_power_cut_is_flagged_never_claimed_twice(
    tmp_path: Path, hard_timeout: None
) -> None:
    """S32-D2: trial 2's record (synced record.json) names episode 1, but a power cut took the
    episode (the page cache held it). The resume's reconcile sees the recovered dataset holds 1
    episode: trial 2 has NO episode (``lost``; no journal: ``recording_missing``), and trial 3's
    staged journal is recovered as episode 1, so no two records claim one index; the export
    refuses a complete bundle (never 'claimed by two trial records') and an INCOMPLETE one
    names trial 2. The session's saves are synced (``sync_s``, ``synced_files``)."""
    runs, run_dir, _ = _killed_session(tmp_path, "after_record")
    settings, before = ev.load_prior(run_dir)
    second, third = before[1], before[2]
    assert second["capture"]["episode_index"] == 1 and second["capture"]["synced_files"] >= 1
    assert "sync_error" not in second["capture"] and second["capture"]["sync_s"] >= 0
    ds_root = runs / settings["session_id"] / "capture" / "dataset"
    _lose_episode(ds_root, 1)

    rec2 = _resume(runs, run_dir)
    scored = _check_no_rerun(rec2)
    by_tid = {t["trial_id"]: t["capture"] for t in scored}
    assert by_tid[second["trial_id"]]["episode_index"] is None
    assert by_tid[second["trial_id"]]["recording_missing"] is True
    assert by_tid[second["trial_id"]]["lost_episode"]["episode_index"] == 1
    assert (
        by_tid[third["trial_id"]]["episode_index"] == 1 and by_tid[third["trial_id"]]["recovered"]
    )
    rc = rec2["results"]["capture"]["reconciled"]
    assert [x["trial_id"] for x in rc["lost"]] == [second["trial_id"]]
    assert rc["recording_missing"] == [second["trial_id"]] and rc["recovered"] == [
        third["trial_id"]
    ]
    claims = [c["episode_index"] for c in by_tid.values() if c["episode_index"] is not None]
    assert sorted(claims) == [0, 1]
    prov = json.loads((ds_root / recording.PROVENANCE).read_text())
    assert prov["episodes"]["1"]["trial_id"] == third["trial_id"]

    with pytest.raises(SystemExit, match="REFUSED: INCOMPLETE"):
        _export(runs, rec2["path"].parent.name, tmp_path / "export")
    index = _export(
        runs, rec2["path"].parent.name, tmp_path / "export2", "--allow-missing-recordings"
    )
    assert index["status"] == "INCOMPLETE" and index["episodes"] == 2
    assert [m["trial_id"] for m in index["recording_missing"]] == [second["trial_id"]]


def test_a_saved_episode_is_synced_before_its_journal_is_dropped(
    tmp_path: Path, hard_timeout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S32-D2: ``EvalCapture.finish`` syncs the files the save wrote (and every directory) BEFORE
    it deletes the staged journal; a failed sync keeps the journal and says so."""
    order: list[str] = []
    real_tree, real_drop = recording.sync_tree, recording.EvalCapture.drop_staged
    fail = {"on": False}

    def sync_tree(
        root: Path, skip: frozenset[str] = frozenset(), since: Any = None, policy: Any = None
    ) -> int:
        if fail["on"]:
            raise OSError(5, "Input/output error")
        order.append(f"sync:{Path(root).name}")
        return int(real_tree(root, skip, since, *([] if policy is None else [policy])))

    def drop(self: Any, trial_id: str) -> None:
        order.append("drop")
        real_drop(self, trial_id)

    monkeypatch.setattr(recording, "sync_tree", sync_tree)
    monkeypatch.setattr(recording.EvalCapture, "drop_staged", drop)
    rec = _new_session(tmp_path / "runs", _ckpt(tmp_path))
    r = json.loads(rec.read_text())["results"]
    saves = [i for i, e in enumerate(order) if e == "sync:dataset"]
    assert len(saves) == 3 and all(order[i + 1] == "drop" for i in saves)
    assert all(t["capture"]["episode_index"] is not None for t in r["trials"])

    fail["on"] = True
    order.clear()
    rec2 = _new_session(tmp_path / "runs2", tmp_path / "pretrained_model")
    r2 = json.loads(rec2.read_text())["results"]
    for t in r2["trials"]:
        assert t["capture"]["episode_index"] is not None  # saved: the score never changes
        assert "Input/output error" in t["capture"]["sync_error"]
        assert recording.has_staged(rec2.parent.parent / r2["session"]["session_id"] / "capture",
                                    t["trial_id"])  # fmt: skip
    assert "drop" not in order
