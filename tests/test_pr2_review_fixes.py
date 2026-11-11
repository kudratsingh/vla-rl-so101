"""PR 2 review fixes (round 1), CPU and fakes only: durable recordings (F1), the demo resume
(F5), camera roles (conv-4), the trainer's demo-only refusal (conv-2), never-started items retried
in place (F3), the capture saved after the record with settled stills and real timing (F6/F7,
conv-3), dev exports declared as such and rehearsals refused (F9)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.eval import export_bundle as X
from src.eval import protocol_real as PR
from src.robot import convention as conv
from src.robot import recording
from src.robot.camera import CameraRig, CameraRoleError, confirm_camera_roles
from src.robot.client import ScriptedOperator
from src.robot.units import UnitMap


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_fix", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_fix"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")
xb = _load("wo07_export_bundle")
train = _load("wo07_train")
teleop = _load("wo06_teleop_record")


def _frame(i: int) -> dict[str, Any]:
    img = np.full(conv.IMAGE_SHAPE_HWC, (40 * i) % 255, np.uint8)
    q = np.full(6, 0.01 * i)
    return conv.recording_frame(
        {"front": img, "wrist": 255 - img},
        q,
        q,
        conv.DEMO_TASK_STRING,
        write_time_s=i / 15,
        frame_time_s=i / 15 - 0.01,
    )


def _simulate_kill_mid_save(root: Path, next_index: int) -> None:
    """What SIGKILL during a save leaves: a footerless data file, its video, frame PNGs, and
    meta/info.json already counting the episode."""
    (root / "data/chunk-000" / f"file-{next_index:03d}.parquet").write_bytes(b"PAR1 cut short")
    for key in conv.IMAGE_KEYS.values():
        vd = root / "videos" / key / "chunk-000"
        (vd / f"file-{next_index:03d}.mp4").write_bytes(b"partial")
        (root / "images" / key / f"episode-{next_index:06d}").mkdir(parents=True)
    info = json.loads((root / "meta/info.json").read_text())
    info["total_episodes"] += 1
    info["total_frames"] += 5
    (root / "meta/info.json").write_text(json.dumps(info))


def test_durable_dataset_loses_at_most_the_episode_being_saved(tmp_path: Path) -> None:
    """F1: every saved episode is finalized (Parquet footers) and reopened; a kill mid-save is
    recovered to the finalized episodes, which stay readable, and recording continues."""
    root = tmp_path / "ds"
    ds = recording.DurableDataset(recording.create_dataset("local/durable", root), "local/durable")
    for ep in range(2):
        for i in range(5):
            ds.add_frame(_frame(i + ep))
        seen: list[int] = []
        assert ds.save_episode(before_checkpoint=seen.append) == ep and seen == [ep]
    ds.inner.stop_image_writer()
    _simulate_kill_mid_save(root, 2)
    report = recording.recover_dataset(root, tmp_path / "quarantine")
    assert report["episodes"] == 2 and report["frames"] == 10
    assert "data/chunk-000/file-002.parquet" in report["moved"] and "images/" in report["moved"]
    assert recording.recover_dataset(root, tmp_path / "q2")["moved"] == []  # idempotent
    again = recording.DurableDataset(recording.open_dataset("local/durable", root), "local/durable")
    for i in range(3):
        again.add_frame(_frame(i))
    assert again.save_episode() == 2
    again.close()
    eps = X.check_dataset(root, {"units": conv.DECLARED_UNITS})  # the importer mirror reads it
    assert [int(e["length"]) for e in eps] == [5, 5, 3]
    from src.data.recorder import load_demo_dataset

    loaded = load_demo_dataset("local/durable", root)
    assert loaded.num_episodes == 3 and len(loaded) == 13
    assert loaded[12][conv.TIMING_KEY].tolist() == pytest.approx([2 / 15, 2 / 15 - 0.01])
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats[conv.STATE_KEY]["count"] == [13]  # rebuilt without the lost episode


@pytest.fixture(scope="module")
def mock_session(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """A real-0.1 mock session (gRPC MockPolicy, fake rig, capture on), one trial per cell."""
    base = tmp_path_factory.mktemp("sess")
    runs = base / "runs"
    ckpt = base / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"w")
    (ckpt / "config.json").write_text('{"type": "act"}\n')
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec = ev.main(
            [
                "--preset", "real", "--mock", "--fake-clock", "--max-steps", "5",
                "--task", "PickPlace", "--trials", "1", "--labeler", "mock-labeler",
                "--family", "act", "--training-run-id", "run-act", "--checkpoint-dir", str(ckpt),
                "--runs-dir", str(runs),
            ]
        )  # fmt: skip
    return runs, Path(rec), ckpt


def test_capture_after_the_record_with_settled_stills_and_timing(
    mock_session: tuple[Path, Path, Path],
) -> None:
    """F7: the end-state stills are read at the end of the settle (before the score) and the
    episode is saved after the record was logged (a capture row); conv-3 / F6: real timing."""
    runs, rec, _ = mock_session
    res = json.loads(rec.read_text())["results"]
    rows = [json.loads(x) for x in (rec.parent / "metrics.jsonl").read_text().splitlines()]
    kinds = [r.get("kind") for r in rows if r.get("kind") in ("trial", "capture")]
    assert kinds == ["trial", "capture"] * 3  # each record is logged before its capture
    for t in res["trials"]:
        cap = t["capture"]
        assert cap["episode_index"] is not None and cap["nominal_fps"] == conv.FPS
        assert "settling interval" in cap["end_state_when"] and cap["write_rate_hz"] > 0
        assert t["attempt"] == 1
    # load_prior merges the capture rows into records that lack them (a killed run)
    rec_json = json.loads(rec.read_text())
    for t in rec_json["results"]["trials"]:
        t["capture"] = {"episode_index": None, "reason": ev.CAPTURE_PENDING}
    copy = runs / "copy-run"
    copy.mkdir()
    (copy / "record.json").write_text(json.dumps(rec_json))
    shutil.copy(rec.parent / "metrics.jsonl", copy / "metrics.jsonl")
    _, trials = ev.load_prior(copy)
    assert sorted(t["capture"]["episode_index"] for t in trials) == [0, 1, 2]
    shutil.rmtree(copy)


def test_a_killed_capture_is_recovered_before_the_resume_connects_the_arm(
    mock_session: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """F1: the session's capture after a kill mid-save: precheck_capture (before any device)
    recovers it; the resume reopens and appends; the export still bundles every episode."""
    runs0, rec0, _ckpt = mock_session
    runs = tmp_path / "runs"
    shutil.copytree(runs0, runs)
    rec = runs / rec0.parent.name / "record.json"
    settings, _ = ev.load_prior(rec.parent)
    ds_root = runs / settings["session_id"] / "capture" / "dataset"
    _simulate_kill_mid_save(ds_root, 3)
    report = ev.precheck_capture(runs, settings, ev.recorded_mode(rec.parent))
    assert report is not None and report["episodes"] == 3 and report["moved"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec2 = ev.main(
            ["--preset", "real", "--mock", "--fake-clock", "--resume", rec.parent.name,
             "--max-steps", "5", "--runs-dir", str(runs)]
        )  # fmt: skip
    assert json.loads(Path(rec2).read_text())["results"]["capture"]["episodes"] == 3
    out = tmp_path / "export"
    index = xb.main(
        ["--preset", "real", "eval", "--session", Path(rec2).parent.name, "--runs-dir",
         str(runs), "--out", str(out), "--allow-mock"]
    )  # fmt: skip
    assert index["episodes"] == 3 and index["timing"]["nominal_fps"] == conv.FPS


def test_never_started_item_is_retried_in_its_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """F3: a not-started attempt (e.g. a first-query timeout the operator resumed) is retried
    before the next schedule item; after MAX_START_ATTEMPTS the run ends (resume retries it)."""
    settings = {
        "task": PR.TASK, "trials_per_cell": 50, "schedule": "eval", "session_id": "s",
        "cells": list(PR.CELLS), "settle_s": 3.0, "labeler": "x", "policy_identity": None,
    }  # fmt: skip
    order: list[int] = []
    starts = iter([False, True, True, True])  # item 0: one failed start, then started

    def fake_run_trials(_loop: Any, _op: Any, **kw: Any) -> tuple[list[Any], str]:
        rec = {
            "cell": kw["cell"], "trial_idx": kw["indices"][0], "policy_started": next(starts),
            "start_utc": str(len(order)), **kw["record_extra"],
        }  # fmt: skip
        kw["record_hook"](rec)
        kw["trials"].append(rec)
        order.append(rec["schedule_pos"])
        return kw["trials"], "completed" if len(order) < 4 else "operator_quit"

    monkeypatch.setattr(ev, "run_trials", fake_run_trials)
    trials: list[dict[str, Any]] = []
    why = ev._run_schedule(
        None, None, settings, {}, trials, {}, None, log_row=lambda **_k: None,
        max_steps=None, camera_calibrated=False, on_trial_start=None,
    )  # fmt: skip
    assert order == [0, 0, 1, 2] and why == "operator_quit"
    assert [t["attempt"] for t in trials] == [1, 2, 1, 1]
    # never starting: the run ends after MAX_START_ATTEMPTS, the item stays first for a resume
    starts = iter([False] * 10)
    order.clear()
    why = ev._run_schedule(
        None, None, settings, {}, [], {}, None, log_row=lambda **_k: None,
        max_steps=None, camera_calibrated=False, on_trial_start=None,
    )  # fmt: skip
    assert order == [0] * ev.MAX_START_ATTEMPTS and why.startswith("not_started")


def test_the_trainer_takes_only_the_real_0_1_demo_recording() -> None:
    """conv-2: an eval capture, a sanity recording, another protocol or a placement outside the
    frozen demo schedule is refused; the demo schedule itself passes."""
    sched = PR.demo_schedule()
    good = {
        "purpose": "demonstration",
        "demo_schedule": True,
        "protocol": PR.citation(),
        "episodes": {str(i): sched[i] for i in range(3)},
    }
    assert train.demo_problems(good, 3) == []
    assert "eval capture" in " ".join(train.demo_problems({**good, "purpose": "evaluation"}, 3))
    assert "--demo-schedule" in " ".join(train.demo_problems({**good, "demo_schedule": False}, 3))
    assert "protocol" in " ".join(train.demo_problems({**good, "protocol": None}, 3))
    ring = dict(sched[0], object_slot="A1")
    bad = {**good, "episodes": {**good["episodes"], "0": ring}}
    assert "not a demo of the frozen schedule" in " ".join(train.demo_problems(bad, 3))
    assert "no recorded placement" in " ".join(train.demo_problems(good, 4))


def test_demo_resume_checks() -> None:
    """F5: a demo recording is continued only under the same protocol, convention, units and
    an unbroken prefix of the frozen schedule."""
    sched = PR.demo_schedule()
    units = UnitMap()
    prov = {
        "demo_schedule": True,
        "rehearsal": False,
        "protocol": PR.citation(),
        "convention": conv.citation(),
        "unit_map": json.loads(json.dumps(dataclasses.asdict(units))),
        "episodes": {str(i): sched[i] for i in range(3)},
    }
    assert teleop.resume_problems(prov, units, 3, sched) == []
    mirrored = dataclasses.replace(units, arm_sign=(1.0, 1.0, 1.0, -1.0, 1.0))
    assert "robot.units" in " ".join(teleop.resume_problems(prov, mirrored, 3, sched))
    gap = {**prov, "episodes": {"0": sched[0], "2": sched[2]}}
    assert teleop.resume_problems(gap, units, 3, sched)
    swapped = {**prov, "episodes": {"0": sched[1], "1": sched[0], "2": sched[2]}}
    assert teleop.resume_problems(swapped, units, 3, sched)
    assert teleop.resume_problems({**prov, "rehearsal": True}, units, 3, sched)


def test_camera_roles_are_confirmed_on_a_labelled_sheet(tmp_path: Path) -> None:
    """conv-4: one frame of every camera, labelled; only 'yes' lets recording start."""
    from robot_fakes import FakeCam

    rig = CameraRig({"front": FakeCam(), "wrist": FakeCam()})
    shown: list[Path] = []
    got = confirm_camera_roles(
        rig, tmp_path / "c.png", lambda _p: "yes", indices={"front": 0}, show=shown.append
    )
    assert got["indices"] == {"front": 0} and Path(got["sheet"]).is_file() and shown
    for answer in ("no", "", "y"):
        with pytest.raises(CameraRoleError, match="not confirmed"):
            confirm_camera_roles(rig, tmp_path / "d.png", lambda _p, a=answer: a)


def test_dev_sessions_and_rehearsals_are_never_exported_as_results(
    tmp_path: Path, mock_session: tuple[Path, Path, Path]
) -> None:
    """F9: a development session is declared real-0.1-dev; a rehearsal demo is refused."""
    _, _, ckpt = mock_session
    runs = tmp_path / "runs"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec = ev.main(
            [
                "--preset", "real", "--mock", "--fake-clock", "--max-steps", "4",
                "--task", "PickPlace", "--schedule", "dev", "--trials", "1",
                "--labeler", "mock", "--family", "act", "--training-run-id", "run-act",
                "--checkpoint-dir", str(ckpt), "--runs-dir", str(runs),
            ]
        )  # fmt: skip
    out = tmp_path / "dev.export"
    argv = ["--preset", "real", "eval", "--session", Path(rec).parent.name, "--runs-dir"]
    with pytest.raises(SystemExit, match="development session"):
        xb.main([*argv, str(runs), "--out", str(out), "--allow-mock"])
    index = xb.main([*argv, str(runs), "--out", str(out), "--allow-mock", "--include-dev"])
    decl = json.loads((out / X.DECLARATION_FILE).read_text())
    assert decl["protocol_version"] == "real-0.1-mock-dev" == index["protocol_version"]
    # a rehearsal demo recording
    demo = tmp_path / "demo" / "dataset"
    (demo / "meta").mkdir(parents=True)
    recording.write_json(
        demo / recording.PROVENANCE,
        {"protocol": PR.citation(), "convention": conv.citation(), "rehearsal": True},
    )
    with pytest.raises(X.ExportRefused, match="rehearsal"):
        X.build_demo_bundle(demo, tmp_path / "demo.export")
