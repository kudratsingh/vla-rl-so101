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
    else:  # review F2: the summary / --report rule fills it from the provenance, as the export
        assert not any(ev.recording_gate(before, settings, capture_root).values())

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
        if since is None:  # the new dataset's sync at open (review F3), not an episode save
            order.append(f"open:{Path(root).name}")
            return int(real_tree(root, skip, since, *([] if policy is None else [policy])))
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
    assert order[0] == "open:dataset"  # the new dataset is synced before the first trial
    assert all(t["capture"]["episode_index"] is not None for t in r["trials"])

    fail["on"] = True
    order.clear()
    rec2 = _new_session(tmp_path / "runs2", tmp_path / "pretrained_model")
    r2 = json.loads(rec2.read_text())["results"]
    for t in r2["trials"]:
        assert t["capture"]["episode_index"] is None  # saved, not durable: PENDING (owner)
        assert t["capture"]["pending_episode_index"] is not None
        assert "Input/output error" in t["capture"]["sync_error"]
        assert recording.has_staged(rec2.parent.parent / r2["session"]["session_id"] / "capture",
                                    t["trial_id"])  # fmt: skip
    assert "drop" not in order


def test_an_episode_whose_save_sync_failed_stays_pending_until_a_resume_syncs_it(
    tmp_path: Path, hard_timeout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner hold on PR #32 (P1): every episode save's sync fails (EIO), so the saved episodes
    are not known to be durable. They stay PENDING: no record claims the index, the session is
    not protocol-complete and the export refuses (allow-missing too), naming the trials. A
    resume whose confirmation sync fails leaves them pending; one with a healthy sync verifies
    and syncs them (journals dropped, indices attached), then the export is COMPLETE."""
    real_tree = recording.sync_tree
    fail = {"on": True, "confirm": False}

    def sync_tree(
        root: Path, skip: frozenset[str] = frozenset(), since: Any = None, policy: Any = None
    ) -> int:
        if since is not None and fail["on"]:  # an episode save's sync, not a dataset open
            raise OSError(5, "Input/output error")
        if since is None and skip == frozenset({"images"}) and fail["confirm"]:  # confirm_saved
            raise OSError(5, "Input/output error (confirm)")
        return int(real_tree(root, skip, since, *([] if policy is None else [policy])))

    monkeypatch.setattr(recording, "sync_tree", sync_tree)
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    r = json.loads(rec.read_text())["results"]
    capture_root = runs / r["session"]["session_id"] / "capture"
    tids = [t["trial_id"] for t in r["trials"]]
    assert len(tids) == 3
    for out, extra in (("export", ()), ("export-allow", ("--allow-missing-recordings",))):
        with pytest.raises(SystemExit, match="PENDING") as refused:
            _export(runs, rec.parent.name, tmp_path / out, *extra)
        assert all(tid in str(refused.value) for tid in tids)
    for i, t in enumerate(r["trials"]):
        cap = t["capture"]
        assert t["outcome"] in ev.SCORED  # the score stands
        assert cap["episode_index"] is None and cap["pending_episode_index"] == i, cap
        assert "Input/output error" in cap["sync_error"]
        assert recording.has_staged(capture_root, t["trial_id"])
    assert r["summary"]["recordings_pending"] == tids
    prov_path = capture_root / "dataset" / recording.PROVENANCE
    prov = json.loads(prov_path.read_text())
    assert all("Input/output error" in prov["episodes"][str(i)]["sync_pending"] for i in range(3))

    # a resume: episode 0's provenance says one frame more than it holds (the verification
    # refuses it before any sync), the confirmation sync of the others fails: all stay pending
    prov["episodes"]["0"]["frames"] += 1
    prov_path.write_text(json.dumps(prov))
    fail.update(on=False, confirm=True)
    rec1 = _resume(runs, rec.parent)
    prov = json.loads(prov_path.read_text())  # restored: its own count again
    prov["episodes"]["0"]["frames"] -= 1
    prov_path.write_text(json.dumps(prov))
    _check_no_rerun(rec1)
    assert rec1["results"]["capture"]["reconciled"]["pending_sync"] == tids
    assert rec1["results"]["capture"]["recoverable_on_resume"] == tids
    assert rec1["results"]["summary"]["recordings_pending"] == tids
    assert not rec1["results"]["summary"]["protocol_complete"]
    for i, t in enumerate(rec1["results"]["trials"]):
        why = "frames" if i == 0 else "confirm"
        assert recording.capture_pending(t["capture"]) and why in t["capture"]["confirm_error"]
        assert recording.has_staged(capture_root, t["trial_id"])
    with pytest.raises(SystemExit, match="PENDING"):
        _export(runs, rec1["path"].parent.name, tmp_path / "export1")

    fail["confirm"] = False
    rec2 = _resume(runs, rec1["path"].parent)
    scored = _check_no_rerun(rec2)
    assert sorted(t["capture"]["episode_index"] for t in scored) == [0, 1, 2]
    assert all(not recording.capture_pending(t["capture"]) for t in scored)
    assert rec2["results"]["capture"]["reconciled"]["attached"] == tids
    assert rec2["results"]["summary"]["recordings_pending"] == []
    assert not list((capture_root / recording.STAGED_DIR).glob("*"))
    prov = json.loads((capture_root / "dataset" / recording.PROVENANCE).read_text())
    assert not [e for e in prov["episodes"].values() if "sync_pending" in e]
    index = _export(runs, rec2["path"].parent.name, tmp_path / "export2")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3
    ds_root = capture_root / "dataset"  # the confirmation's verification of a saved episode
    frames = int(prov["episodes"]["0"]["frames"])
    assert recording.episode_problem(ds_root, 0, frames) == ""
    assert "frames" in recording.episode_problem(ds_root, 0, frames + 1)
    assert "0 metadata rows" in recording.episode_problem(ds_root, 3)


def test_an_episode_whose_checkpoint_failed_after_the_save_stays_pending(
    tmp_path: Path, hard_timeout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review DR-1 / F1 (PR #32 hold round): episode 2 is saved, then its checkpoint fails
    (EIO): its block is PENDING like a failed sync (no index claimed, the journal kept, the
    provenance marked), the summary lists it, the session prints the resume hint and the export
    refuses. A record that still claims the index with its journal staged (written before the
    pending block existed) is unconfirmed by the same rule (the summary with the capture root,
    the export), and a healthy resume confirms it (verified, synced), then the export is
    COMPLETE."""
    real_checkpoint = recording.DurableDataset.checkpoint
    calls = {"n": 0}

    def checkpoint(self: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(5, "Input/output error (checkpoint)")
        real_checkpoint(self)

    monkeypatch.setattr(recording.DurableDataset, "checkpoint", checkpoint)
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    data = json.loads(rec.read_text())
    r = data["results"]
    capture_root = runs / r["session"]["session_id"] / "capture"
    tids = [t["trial_id"] for t in r["trials"]]
    cap = r["trials"][2]["capture"]
    assert cap["episode_index"] is None and cap["pending_episode_index"] == 2, cap
    assert "checkpoint failed" in cap["error"] and recording.capture_pending(cap)
    assert recording.has_staged(capture_root, tids[2])
    assert r["summary"]["recordings_pending"] == [tids[2]]
    assert r["capture"]["recoverable_on_resume"] == [tids[2]]  # the --resume hint
    prov_path = capture_root / "dataset" / recording.PROVENANCE
    prov = json.loads(prov_path.read_text())
    assert "checkpoint failed" in prov["episodes"]["2"]["sync_pending"]
    with pytest.raises(SystemExit, match="PENDING") as refused:
        _export(runs, rec.parent.name, tmp_path / "export0")
    assert tids[2] in str(refused.value)

    # the same episode as a record that claims it (no error beside it), the journal staged
    r["trials"][2]["capture"] = {
        k: v for k, v in cap.items() if k not in ("pending_episode_index", "reason", "error")
    } | {"episode_index": 2}
    rec.write_text(json.dumps(data))
    del prov["episodes"]["2"]["sync_pending"]
    prov_path.write_text(json.dumps(prov))
    settings, trials = ev.load_prior(rec.parent)
    assert ev.table(trials, settings, xb._dcfg_stub())["recordings_pending"] == []
    assert ev.table(trials, settings, xb._dcfg_stub(), capture_root)["recordings_pending"] == [
        tids[2]
    ]
    assert ev.capture_gaps(capture_root, trials)["recoverable_on_resume"] == [tids[2]]
    with pytest.raises(SystemExit, match="PENDING") as refused:
        _export(runs, rec.parent.name, tmp_path / "export1")
    assert tids[2] in str(refused.value)

    monkeypatch.setattr(recording.DurableDataset, "checkpoint", real_checkpoint)
    rec1 = _resume(runs, rec.parent)
    scored = _check_no_rerun(rec1)
    assert rec1["results"]["capture"]["reconciled"]["confirmed"] == [tids[2]]
    assert sorted(t["capture"]["episode_index"] for t in scored) == [0, 1, 2]
    assert rec1["results"]["summary"]["recordings_pending"] == []
    assert not list((capture_root / recording.STAGED_DIR).glob("*"))
    index = _export(runs, rec1["path"].parent.name, tmp_path / "export2")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def _clip(ds_root: Path, idx: int, key: str) -> Path:
    """The video file holding episode ``idx`` of camera ``key`` (its episode-metadata row)."""
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    (ep,) = [r for r in X._read_parquet_dir(ds_root / "meta" / "episodes")
             if int(r["episode_index"]) == idx]  # fmt: skip
    return ds_root / str(info["video_path"]).format(
        video_key=key,
        chunk_index=ep[f"videos/{key}/chunk_index"],
        file_index=ep[f"videos/{key}/file_index"],
    )


def _last_sample(data: bytes) -> tuple[int, int]:
    """(byte offset, size) of the last video packet of a one-track MP4 (its ``stsz`` / ``stco``
    / ``stsc`` sample tables): the episode's final frame (review REC-1)."""
    import struct

    def box(path: tuple[str, ...], s: int = 0, e: int = len(data)) -> tuple[int, int]:
        while s + 8 <= e:
            size, typ = struct.unpack(">I4s", data[s : s + 8])
            head = 8
            if size == 1:
                size, head = struct.unpack(">Q", data[s + 8 : s + 16])[0], 16
            if typ.decode() == path[0]:
                return (s + head, s + size) if len(path) == 1 else box(path[1:], s + head, s + size)
            s += size
        raise AssertionError(f"no {path[0]} box")

    stbl = ("moov", "trak", "mdia", "minf", "stbl")
    b, _ = box((*stbl, "stsz"))
    fixed, count = struct.unpack(">4xII", data[b : b + 12])
    sizes = (
        [fixed] * count
        if fixed
        else list(struct.unpack(f">{count}I", data[b + 12 : b + 12 + 4 * count]))
    )
    b, _ = box((*stbl, "stco"))
    (n,) = struct.unpack(">I", data[b + 4 : b + 8])
    chunks = struct.unpack(f">{n}I", data[b + 8 : b + 8 + 4 * n])
    b, _ = box((*stbl, "stsc"))
    (m,) = struct.unpack(">I", data[b + 4 : b + 8])
    runs = [struct.unpack(">III", data[b + 8 + 12 * k : b + 20 + 12 * k]) for k in range(m)]
    per_chunk = [
        next(spc for first, spc, _ in reversed(runs) if first <= c + 1) for c in range(len(chunks))
    ]
    first_of_last = sum(per_chunk[:-1])  # the last chunk holds the last sample
    return chunks[-1] + sum(sizes[first_of_last:-1]), sizes[-1]


def _corrupt(path: Path, how: str) -> None:
    data = path.read_bytes()
    if how == "middle":  # bytes overwritten inside the frame data
        mid = len(data) // 2
        path.write_bytes(data[:mid] + b"\x00" * 4096 + data[mid + 4096 :])
    elif how == "last":  # bytes overwritten inside the final frame's packet only
        off, size = _last_sample(data)
        assert size >= 64 and off + size <= len(data), (off, size)
        mid = off + size // 2
        path.write_bytes(data[:mid] + b"\x00" * 16 + data[mid + 16 :])
    else:  # "tail": the file cut short (the index at its end gone)
        path.write_bytes(data[: len(data) - 4000])


def _tree_bytes(d: Path) -> dict[str, bytes]:
    return {
        p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize(
    "corruption",
    [  # episode 0: one camera; episode 1: both cameras; episode 2 healthy (still confirmed)
        {0: {"front": "middle"}, 1: {"front": "tail", "wrist": "tail"}},
        {0: {"wrist": "tail"}, 1: {"front": "middle", "wrist": "middle"}},
        # review REC-1: only the final frame's packet damaged (a decode that stops short of the
        # last frame would confirm these)
        {0: {"front": "last"}, 1: {"front": "last", "wrist": "last"}},
    ],
    ids=["front-middle_both-tail", "wrist-tail_both-middle", "front-last_both-last"],
)
def test_a_pending_episode_whose_video_does_not_decode_is_never_confirmed(
    tmp_path: Path,
    hard_timeout: None,
    monkeypatch: pytest.MonkeyPatch,
    corruption: dict[int, dict[str, str]],
) -> None:
    """Owner 2026-11-11 (PR #32 hold, issue 1): a pending real-0.1 mock capture (every save's
    sync failed) whose MP4s were corrupted before the resume. The resume decodes both cameras'
    clips (torchcodec) across each pending episode's full frame range BEFORE it clears pending
    or drops the journal: a clip that does not decode keeps the episode pending, its staged
    journal byte-for-byte, the error recorded (camera, frame range), the score unchanged, the
    export refused and the summary not complete. The healthy pending episode is confirmed;
    once the good MP4s are restored, the next resume confirms the others."""
    real_tree = recording.sync_tree
    fail = {"on": True}

    def sync_tree(
        root: Path, skip: frozenset[str] = frozenset(), since: Any = None, policy: Any = None
    ) -> int:
        if since is not None and fail["on"]:  # an episode save's sync, not a dataset open
            raise OSError(5, "Input/output error")
        return int(real_tree(root, skip, since, *([] if policy is None else [policy])))

    monkeypatch.setattr(recording, "sync_tree", sync_tree)
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    r = json.loads(rec.read_text())["results"]
    capture_root = runs / r["session"]["session_id"] / "capture"
    ds_root = capture_root / "dataset"
    tids = [t["trial_id"] for t in r["trials"]]
    assert all(recording.capture_pending(t["capture"]) for t in r["trials"])
    good: dict[Path, bytes] = {}
    for idx, cams in corruption.items():
        for name, how in cams.items():
            clip = _clip(ds_root, idx, conv.IMAGE_KEYS[name])
            good[clip] = clip.read_bytes()
            _corrupt(clip, how)
    journals = {tid: _tree_bytes(recording.staged_path(capture_root, tid)) for tid in tids}

    fail["on"] = False  # a healthy sync: only the decode can refuse
    rec1 = _resume(runs, rec.parent)
    _check_no_rerun(rec1)
    bad = [tids[i] for i in sorted(corruption)]
    res = rec1["results"]
    assert res["capture"]["reconciled"]["pending_sync"] == bad
    assert res["capture"]["reconciled"]["attached"] == [tids[2]]  # decoded, synced, confirmed
    assert res["summary"]["recordings_pending"] == bad
    assert not res["summary"]["protocol_complete"]
    for i, (t0, t1) in enumerate(zip(r["trials"], res["trials"], strict=True)):
        assert t1["outcome"] == t0["outcome"] and t1.get("success") == t0.get("success")
        if i not in corruption:
            assert t1["capture"]["episode_index"] == 2
            assert not recording.has_staged(capture_root, t1["trial_id"])
            continue
        cap = t1["capture"]
        assert recording.capture_pending(cap) and cap["pending_episode_index"] == i, cap
        err = cap["confirm_error"]
        frames = int(cap["frames"])
        for name in conv.CAMERA_NAMES:  # names the camera that failed, and only it
            assert (conv.IMAGE_KEYS[name] in err) == (name in corruption[i]), err
        assert f"frames 0-{frames - 1}" in err, err
        assert (
            _tree_bytes(recording.staged_path(capture_root, t1["trial_id"]))
            == journals[t1["trial_id"]]
        )  # the journal is kept byte-for-byte
    prov = json.loads((ds_root / recording.PROVENANCE).read_text())
    for i in corruption:
        assert prov["episodes"][str(i)]["sync_pending"]
        assert (
            conv.IMAGE_KEYS[next(iter(corruption[i]))] in prov["episodes"][str(i)]["confirm_error"]
        )
    got = X.recording_blockers(res["trials"], capture_root)  # the --report / export rule
    assert [(p["trial_id"], "frames 0-" in p["why"]) for p in got["pending"]] == [
        (tid, True) for tid in bad
    ]  # pending, the decode error the reason
    for i in range(3):
        problems = recording.verify_episode_video(ds_root, i)
        assert bool(problems) == (i in corruption), problems
    with pytest.raises(SystemExit, match="PENDING") as refused:
        _export(runs, rec1["path"].parent.name, tmp_path / "export1")
    assert all(tid in str(refused.value) for tid in bad)

    for clip, data in good.items():  # the good MP4s back: the next resume confirms them
        clip.write_bytes(data)
    rec2 = _resume(runs, rec1["path"].parent)
    scored = _check_no_rerun(rec2)
    assert rec2["results"]["capture"]["reconciled"]["attached"] == bad
    assert sorted(t["capture"]["episode_index"] for t in scored) == [0, 1, 2]
    assert rec2["results"]["summary"]["recordings_pending"] == []
    assert all(t["outcome"] == t0["outcome"] for t, t0 in zip(scored, r["trials"], strict=True))
    assert not list((capture_root / recording.STAGED_DIR).glob("*"))
    prov = json.loads((ds_root / recording.PROVENANCE).read_text())
    assert not [e for e in prov["episodes"].values() if "sync_pending" in e or "confirm_error" in e]
    index = _export(runs, rec2["path"].parent.name, tmp_path / "export2")
    assert index["status"] == "COMPLETE" and index["episodes"] == 3


def _edit_episodes(ds_root: Path, index: int, **cols: Any) -> None:
    """Overwrite columns of episode ``index``'s metadata row (in place)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for f in sorted((ds_root / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f)
        rows = t.to_pylist()
        if not any(int(r["episode_index"]) == index for r in rows):
            continue
        for r in rows:
            if int(r["episode_index"]) == index:
                r.update(cols)
        pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), f)


def test_the_video_check_refuses_every_mismatch_with_the_episode(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review REC-1: each guard of ``verify_episode_video`` on its own, against a healthy mock
    episode whose metadata no longer matches its clips: a camera of the convention not declared
    as a video feature, a clip span that is not ``length / fps``, a decoded frame of another
    shape, and frames whose pts are not the episode's own (twice the fps and twice the frames
    over the same span: the span check passes, every other frame decodes its neighbour)."""
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    sid = json.loads(rec.read_text())["results"]["session"]["session_id"]
    good = runs / sid / "capture" / "dataset"
    assert recording.verify_episode_video(good, 2) == []
    info = json.loads((good / "meta" / "info.json").read_text())
    (ep,) = [r for r in X._read_parquet_dir(good / "meta" / "episodes")
             if int(r["episode_index"]) == 2]  # fmt: skip
    front, wrist = conv.IMAGE_KEYS["front"], conv.IMAGE_KEYS["wrist"]
    n, fps = int(ep["length"]), float(info["fps"])

    def no_wrist(ds: Path) -> str:
        i = json.loads((ds / "meta" / "info.json").read_text())
        del i["features"][wrist]
        (ds / "meta" / "info.json").write_text(json.dumps(i))
        return f"{wrist}: not a video feature"

    def span(ds: Path) -> str:
        _edit_episodes(
            ds, 2, **{f"videos/{front}/to_timestamp": ep[f"videos/{front}/to_timestamp"] + 2 / fps}
        )
        return f"{front}: frames 0-{n - 1} of episode 2"

    def shape(ds: Path) -> str:
        i = json.loads((ds / "meta" / "info.json").read_text())
        h, w, c = i["features"][front]["shape"]
        i["features"][front]["shape"] = [h // 2, w // 2, c]
        (ds / "meta" / "info.json").write_text(json.dumps(i))
        return "frames, the feature is"

    def pts(ds: Path) -> str:
        i = json.loads((ds / "meta" / "info.json").read_text())
        i["fps"] = 2 * fps
        (ds / "meta" / "info.json").write_text(json.dumps(i))
        _edit_episodes(ds, 2, length=2 * n)
        return "frame 1 decodes the frame at"

    for case in (no_wrist, span, shape, pts):
        ds = tmp_path / case.__name__
        shutil.copytree(good, ds)
        want = case(ds)
        problems = recording.verify_episode_video(ds, 2)
        assert any(want in p for p in problems), (case.__name__, problems)


def test_an_episode_claimed_against_its_provenance_is_refused_and_lost(
    tmp_path: Path, hard_timeout: None
) -> None:
    """Review F1: a record whose claimed episode is another trial's in the capture provenance
    (an older record of a session a later resume re-mapped) is never exported (episodes keep
    the trial id their provenance gives them) and the summary / --report rule counts the
    recording lost. Review F2: a record killed before its capture row (the placeholder block)
    gets its episode back from the provenance in that rule, as the export fills it."""
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    data = json.loads(rec.read_text())
    res = data["results"]
    settings, trials = res["session"], res["trials"]
    capture_root = runs / settings["session_id"] / "capture"
    assert not any(ev.recording_gate(trials, settings, capture_root).values())
    t0, t1 = trials[0], trials[1]
    t0["capture"] = {"episode_index": None, "reason": ev.CAPTURE_PENDING}  # F2
    assert not any(ev.recording_gate(trials, settings, capture_root).values())
    t0["capture"], t1["capture"] = {"episode_index": 1, "frames": 5}, {
        "episode_index": 0,
        "frames": 5,
    }
    gate = ev.recording_gate(trials, settings, capture_root)
    assert gate["recording_missing"] == [t0["trial_id"], t1["trial_id"]], gate
    rec.write_text(json.dumps(data))
    metrics = rec.parent / "metrics.jsonl"  # the record alone holds the claims
    rows = [json.loads(x) for x in metrics.read_text().splitlines()]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows if r.get("kind") != "capture"))
    with pytest.raises(SystemExit, match="belong to other trials"):
        _export(runs, rec.parent.name, tmp_path / "export")
    with pytest.raises(SystemExit, match="belong to other trials"):
        _export(runs, rec.parent.name, tmp_path / "export_incomplete", "--allow-missing-recordings")


def test_a_save_whose_video_does_not_decode_keeps_its_journal_and_stays_pending(
    tmp_path: Path, hard_timeout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner 2026-11-11 (issue 1), the save path (``EvalCapture.finish``, a recovery's save too):
    episode 1 is saved and synced, its clip does not decode: PENDING (no index claimed, the
    journal kept, ``confirm_error`` in the block and the provenance), the score stands."""
    real = recording.verify_episode_video

    def verify(ds_root: Path, index: int) -> list[str]:
        if index == 1:
            return ["observation.images.wrist: frames 0-4 of episode 1: RuntimeError: corrupt"]
        return list(real(ds_root, index))

    monkeypatch.setattr(recording, "verify_episode_video", verify)
    runs = tmp_path / "runs"
    rec = _new_session(runs, _ckpt(tmp_path))
    r = json.loads(rec.read_text())["results"]
    capture_root = runs / r["session"]["session_id"] / "capture"
    caps = [t["capture"] for t in r["trials"]]
    assert [c["episode_index"] for c in caps] == [0, None, 2]
    assert recording.capture_pending(caps[1]) and caps[1]["pending_episode_index"] == 1
    assert "frames 0-4" in caps[1]["confirm_error"] and "sync_error" not in caps[1]
    assert r["trials"][1]["outcome"] in ev.SCORED
    assert [recording.has_staged(capture_root, t["trial_id"]) for t in r["trials"]] == [
        False,
        True,
        False,
    ]
    assert r["summary"]["recordings_pending"] == [r["trials"][1]["trial_id"]]
    prov = json.loads((capture_root / "dataset" / recording.PROVENANCE).read_text())
    assert "corrupt" in prov["episodes"]["1"]["confirm_error"]
    assert prov["episodes"]["1"]["sync_pending"]


# ------------------------------------------------------------------------------------------------
# Owner 2026-11-11 (PR #32 hold): the session summary / --report and the export reach the SAME
# verdict on the same capture on disk (``export_bundle.recording_blockers``, one reading:
# ``_capture_on_disk``). A damaged capture never reads complete in one and refused in the other.
# ------------------------------------------------------------------------------------------------
GATE_KEYS = (
    "recordings_pending",
    "recordings_unsaved",
    "recording_missing",
    "recordings_unverified",
)


@pytest.fixture(scope="module")
def healthy_capture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    """One finished real-0.1 mock session with capture on (3 trials, 3 episodes with real
    clips): (its base dir, run id). Each case copies it before damaging it."""
    base = tmp_path_factory.mktemp("parity")
    rec = _new_session(base / "runs", _ckpt(base))
    return base, rec.parent.name


def _meta_file(ds: Path, idx: int) -> Path:
    import pyarrow.parquet as pq

    (f,) = [
        f
        for f in sorted((ds / "meta" / "episodes").rglob("*.parquet"))
        if idx in [int(r["episode_index"]) for r in pq.read_table(f).to_pylist()]
    ]
    return f


def _edit_prov(ds: Path, edit: Any) -> None:
    p = ds / recording.PROVENANCE
    prov = json.loads(p.read_text())
    edit(prov)
    p.write_text(json.dumps(prov))


def _damage(case: str, runs: Path, sid: str, trials: list[dict[str, Any]]) -> None:
    cap = runs / sid / "capture"
    ds = cap / "dataset"
    prov = ds / recording.PROVENANCE
    if case == "provenance_missing":
        prov.unlink()
    elif case == "provenance_unreadable":
        prov.write_text(prov.read_text()[:40])
    elif case == "provenance_wrong_shape":
        prov.write_text(
            json.dumps({"session_id": sid, "episodes": [t["trial_id"] for t in trials]})
        )
    elif case == "provenance_other_session":
        _edit_prov(ds, lambda p: p.update(session_id="another-session"))
    elif case == "provenance_extra_episode":  # an entry for an episode the dataset lacks
        _edit_prov(ds, lambda p: p["episodes"].update({"7": {"trial_id": "x/PickPlace/seen/9"}}))
    elif case == "provenance_gives_episode_to_another_trial":
        _edit_prov(ds, lambda p: p["episodes"]["0"].update(trial_id=trials[1]["trial_id"]))
    elif case == "info_missing":
        (ds / "meta" / "info.json").unlink()
    elif case == "info_unreadable":
        (ds / "meta" / "info.json").write_bytes(b"\x00\xff{not json")
    elif case == "episodes_meta_missing":
        shutil.rmtree(ds / "meta" / "episodes")
    elif case == "episodes_meta_unreadable":
        for f in (ds / "meta" / "episodes").rglob("*.parquet"):
            f.write_bytes(b"PAR1 garbage, not a footer")
    elif case == "episodes_meta_lacks_one_episode":
        _meta_file(ds, 2).unlink()
    elif case == "video_file_missing":
        _clip(ds, 1, conv.IMAGE_KEYS["wrist"]).unlink()
    elif case == "data_file_missing":
        info = json.loads((ds / "meta" / "info.json").read_text())
        import pyarrow.parquet as pq

        (ep,) = pq.read_table(_meta_file(ds, 1)).to_pylist()
        (ds / str(info["data_path"]).format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])).unlink()  # fmt: skip
    elif case == "capture_dir_missing":
        shutil.rmtree(cap)
    elif case == "staged_journal_present":
        staged = recording.staged_path(cap, trials[0]["trial_id"])
        staged.mkdir(parents=True)
        (staged / recording.STAGED_INDEX).write_text("{}")
    else:
        assert case == "healthy", case


def _export_category(msg: str) -> str:
    """The blocker kind an export refusal names (the summary's gate key)."""
    if msg.startswith("REFUSED: PENDING"):
        return "recordings_pending"
    if msg.startswith("REFUSED: UNVERIFIED") or X.PROVENANCE_UNREADABLE in msg:
        return "recordings_unverified"
    if msg.startswith("REFUSED: capture provenance"):  # readable, refused (another session's)
        return "recordings_unverified"
    return "recording_missing"


PARITY_CASES = {  # case -> the one blocker kind both name (None: complete in both)
    "healthy": None,
    "provenance_missing": "recordings_unverified",
    "provenance_unreadable": "recordings_unverified",
    "provenance_wrong_shape": "recordings_unverified",
    "provenance_other_session": "recordings_unverified",
    "provenance_extra_episode": None,
    "provenance_gives_episode_to_another_trial": "recording_missing",
    "info_missing": "recordings_unverified",
    "info_unreadable": "recordings_unverified",
    "episodes_meta_missing": "recording_missing",
    "episodes_meta_unreadable": "recording_missing",
    "episodes_meta_lacks_one_episode": "recording_missing",
    "video_file_missing": "recordings_unverified",
    "data_file_missing": "recordings_unverified",
    "capture_dir_missing": "recording_missing",
    "staged_journal_present": "recordings_pending",
}


def parity_verdicts(base: Path, run_id: str, case: str, tmp: Path) -> dict[str, Any]:
    """(report, export) verdicts on a copy of ``base`` damaged by ``case``. The report side is
    ``table`` with the session's capture on disk (what ``--report`` recomputes per session;
    its recording gate is the part of ``protocol_complete`` the capture decides), the export
    side the CLI (``scripts/wo07_export_bundle.py eval``, its complete bundle)."""
    shutil.copytree(base, tmp, dirs_exist_ok=True)
    runs = tmp / "runs"
    res = json.loads((runs / run_id / "record.json").read_text())["results"]
    trials, settings = res["trials"], res["session"]
    sid = str(settings["session_id"])
    _damage(case, runs, sid, trials)
    summ = ev.table(trials, settings, ev.SavedTargets(150.0), runs / sid / "capture")
    gate = [k for k in GATE_KEYS if summ.get(k)]
    reasons = [r.rsplit(" (", 1)[0] for r in summ.get("recording_reasons") or []]
    try:
        index = _export(runs, run_id, tmp / "export")
        export, msg = index["status"] == "COMPLETE", index["status"]
    except SystemExit as e:
        export, msg = False, str(e)
    except Exception as e:  # noqa: BLE001 - a crash is a refusal without a reason
        export, msg = False, f"CRASH {type(e).__name__}: {e}"
    return {
        "report_ok": summ["complete"] and not gate,
        "report_kind": gate[0] if gate else None,
        "reasons": reasons,
        "export_ok": export,
        "export_kind": None if export else _export_category(msg),
        "export_msg": msg,
        "scores": [(t["trial_id"], t["outcome"]) for t in trials],
        "cells": summ["cells"],
    }


@pytest.mark.parametrize("case", list(PARITY_CASES))
def test_report_and_export_reach_the_same_verdict_on_a_damaged_capture(
    healthy_capture: tuple[Path, str], tmp_path: Path, hard_timeout: None, case: str
) -> None:
    base, run_id = healthy_capture
    v = parity_verdicts(base, run_id, case, tmp_path)
    want = PARITY_CASES[case]
    assert v["report_ok"] == v["export_ok"] == (want is None), (case, v)
    assert v["report_kind"] == v["export_kind"] == want, (case, v)
    if want is not None:  # the export names the summary's / --report's reason (or vice versa)
        msg = v["export_msg"].removeprefix("REFUSED: ")
        assert any(r in msg or msg in r for r in v["reasons"]), (case, v)
        assert not v["export_msg"].startswith("CRASH"), (case, v)
    healthy = parity_verdicts(base, run_id, "healthy", tmp_path / "healthy")
    assert v["scores"] == healthy["scores"] and v["cells"] == healthy["cells"]  # scores kept
