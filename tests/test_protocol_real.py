"""Real-arm protocol real-0.1 (src/eval/protocol_real.py), its placement sheet, and the real-0.1
path of scripts/wo07_real_eval.py with the evaluation capture (src/robot/recording.py), in mock
mode: fake arm, two fake cameras, mock policy. No device is opened."""

from __future__ import annotations

import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.eval import protocol as P1
from src.eval import protocol_real as PR
from src.eval import real_grid as G
from src.robot import convention as conv
from src.robot.client import ScriptedOperator

# Pinned: frozen before the first demo (owner D4). A change fails here on purpose: add a new real
# protocol version instead (CLAUDE.md rule 4). PENDING (PR.FREEZE_STATUS): the current computed
# value after real-judge-3 (2026-11-09); the bowl fit check, the rubric bump and the PDF follow.
FROZEN = {"real-0.1": "dca199e6da6275bddc047812e1ea83379a70f21778f564315c5483811c0954f2"}
# the hash prefix the committed placement-grid PDF prints: real-0.1 before real-judge-3. The PDF
# is regenerated with the freeze, not before (owner 2026-11-09)
PDF_HASH12_PENDING = "edee44aa957d"
SCHEDULES = {
    "eval": "dd1345de09b4595048d6f23f2e38057a65d6e06d39f79d7a554e189fc48d0e71",
    "dev": "21fe5598a9784ae119960b444a715a3bf36b9510491749c41e965889fe26af59",
}


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")


def test_real_protocol_is_frozen_and_sim_protocols_untouched() -> None:
    assert PR.protocol_hash() == FROZEN[PR.PROTOCOL_VERSION]
    assert {k: PR.schedule_sha256(k) for k in SCHEDULES} == SCHEDULES
    assert P1.protocol_hash().startswith("e9ce5bb47a88")  # 0.1-draft, cited by every sim number
    comp = PR.hash_components()
    assert comp["convention"] == conv.citation()  # the observation/action convention is pinned
    assert comp["bowl"]["planned_xy_m"] == [0.195, 0.205]


def test_seen_slots_are_the_inner_corners_and_the_derived_central_slot() -> None:
    inner = {s.id for s in G.cell_slots("seen")}
    assert set(PR.SEEN_SLOTS) <= inner and len(PR.SEEN_SLOTS) == 5
    rows = sorted({s[0] for s in inner})
    cols = sorted({s[1] for s in inner})
    corners = {f"{r}{c}" for r in (rows[0], rows[-1]) for c in (cols[0], cols[-1])}
    assert set(PR.CORNER_SLOTS) == corners  # one near each corner
    assert PR.central_slot() == "C4" and PR.SEEN_SLOTS[-1] == PR.central_slot()
    assert set(PR.ring_order()) == {s.id for s in G.cell_slots("shifted")}


def test_bowl_is_derived_and_valid_for_every_slot() -> None:
    from src.env import tasks as T

    assert PR.derive_bowl() == PR.BOWL_XY_M
    b = np.array(PR.BOWL_XY_M)
    for s in G.all_slots():  # all 36: also valid if a seen slot were re-chosen
        assert float(np.linalg.norm(b - [s.x, s.y])) >= T.MIN_SEP["PickPlace"] + PR.BOWL_MARGIN_M
    r = float(np.linalg.norm(b))
    assert T.BASE_KEEPOUT + 0.065 <= r <= T.PUSH_REACH - PR.BOWL_MARGIN_M
    rep = PR.bowl_report()
    assert rep["valid"] and rep["min_slot_distance_m"] == pytest.approx(0.1175, abs=1e-4)


def test_eval_schedule_cells_balance_and_order() -> None:
    sched = PR.eval_schedule()
    assert sched == PR.eval_schedule() and len(sched) == 150
    assert [t["schedule_pos"] for t in sched] == list(range(150))
    by_cell = Counter(t["cell"] for t in sched)
    assert by_cell == {"seen": 50, "shifted": 50, "unseen_objects": 50}
    for cell in PR.CELLS:  # trial_idx is a stable index inside the cell
        assert sorted(t["trial_idx"] for t in sched if t["cell"] == cell) == list(range(50))
    seen = Counter(t["object_slot"] for t in sched if t["cell"] == "seen")
    assert seen == {s: 10 for s in PR.SEEN_SLOTS}
    shifted = Counter(t["object_slot"] for t in sched if t["cell"] == "shifted")
    assert set(shifted) == set(PR.ring_order()) and set(shifted.values()) == {2, 3}
    assert shifted == PR.shifted_counts()
    unseen = Counter(
        (t["object_slot"], t["object"]) for t in sched if t["cell"] == "unseen_objects"
    )
    assert unseen == {(s, o.label): 5 for s in PR.SEEN_SLOTS for o in PR.UNSEEN_OBJECTS}
    assert {o.label for o in PR.UNSEEN_OBJECTS} == {"purple cube", "green cube"}
    for t in sched:
        assert t["planned_bowl_xy_m"] == list(PR.BOWL_XY_M)  # the bowl never moves
        assert t["planned_object_yaw_deg"] in PR.YAWS_DEG and "planned" in t["pose_source"]
        assert t["instruction"] == P1.instruction("PickPlace", PR.object_by_label(t["object"]))
        if t["cell"] != "unseen_objects":
            assert t["object"] == "red cube"
    # the order interleaves the cells (drift never lines up with a cell)
    first = [t["cell"] for t in sched[:30]]
    assert len(set(first)) == 3


def test_demo_and_dev_schedules() -> None:
    demos = PR.demo_schedule()
    assert len(demos) == 50 and [d["demo_idx"] for d in demos] == list(range(50))
    assert Counter(d["object_slot"] for d in demos) == {s: 10 for s in PR.SEEN_SLOTS}
    assert {d["object"] for d in demos} == {"red cube"}  # never the unseen objects
    for s in PR.SEEN_SLOTS:  # every yaw of the grid at every demo position
        assert {d["planned_object_yaw_deg"] for d in demos if d["object_slot"] == s} == set(
            map(float, PR.YAWS_DEG)
        )
    dev = PR.dev_schedule()
    assert len(dev) == 30 and Counter(t["cell"] for t in dev) == PR.DEV_TRIALS
    pose = {
        (t["object"], t["object_slot"], t["planned_object_yaw_deg"]) for t in PR.eval_schedule()
    }
    assert not pose & {(t["object"], t["object_slot"], t["planned_object_yaw_deg"]) for t in dev}
    assert all(abs(t["planned_object_yaw_deg"]) <= 45 for t in dev)


def test_rubric_is_binary_with_partial_credit_secondary() -> None:
    r = PR.rubric()
    assert r["scoring"] == "binary" and r["protocol_hash"] == PR.protocol_hash()
    assert r["judging_rule"] == ev.JUDGING_RULE_VERSION == "real-judge-3"
    assert "bus write of its first action" in r["when"] and "safety filter" not in r["when"]
    assert "never the headline" in r["secondary"]["partial_credit"]
    assert ev.JUDGING_RULE is PR.JUDGING_RULE  # one text, one hash
    rows = PR.cell_tables(
        [
            {
                "cell": "unseen_objects",
                "outcome": "success",
                "placement": {"object": "purple cube"},
            },
            {"cell": "unseen_objects", "outcome": "failure", "placement": {"object": "green cube"}},
            {"cell": "seen", "outcome": "not_started"},
        ]
    )
    assert rows["unseen_objects (pooled, headline)"]["n"] == 2
    assert rows["unseen_objects/purple cube"]["successes"] == 1 and rows["seen"]["n"] == 0


def test_placement_sheet_marks_the_seen_slots_and_the_bowl() -> None:
    grid_pdf = _load("make_placement_grid")
    pending = PR.FREEZE_STATUS != PR.FROZEN  # stale by the hash prefix only (test below)
    assert grid_pdf.main(["--preset", "real", "--check"]) == (1 if pending else 0)
    ops = grid_pdf.draw()
    assert ops.count("(SEEN)") == 5 and ops.count("(not used)") == 11
    assert "BOWL LINE" in ops and "centre x = 0.195 m, y = +0.205 m" in ops
    assert PR.protocol_hash()[:12] in ops


def test_the_freeze_is_pending_and_every_real_session_and_demo_refuses() -> None:
    """Owner 2026-11-09: the bowl fit check of B' is pending. The protocol says so
    (``FREEZE_STATUS``), the bowl and the rubric version are unchanged, the hardware session and
    the demo schedule refuse on the freeze status (and on the bowl box), and the committed
    placement-grid PDF differs from what the script draws ONLY in the printed hash prefix (it is
    regenerated with the freeze)."""
    assert PR.FREEZE_STATUS == "pending: bowl fit check" and PR.FREEZE_STATUS != PR.FROZEN
    assert PR.BOWL_XY_M == (0.195, 0.205) and PR.RUBRIC_VERSION == "1"
    problem = PR.freeze_problem()
    assert problem is not None and "pending: bowl fit check" in problem
    from src.config import load_config

    cfg = load_config("real")
    s = {"schedule": "eval", "trials_per_cell": 50, "cells": list(PR.CELLS), "capture": True}
    assert problem in ev.real_hardware_problems(cfg, s)
    grid_pdf = _load("make_placement_grid")
    assert grid_pdf.main(["--preset", "real", "--check"]) == 1  # stale by the hash, on purpose
    committed = (REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf").read_bytes()
    ops = grid_pdf.draw()
    assert PR.protocol_hash()[:12] in ops and PDF_HASH12_PENDING not in ops
    assert grid_pdf.pdf_bytes(ops.replace(PR.protocol_hash()[:12], PDF_HASH12_PENDING)) == committed


# ------------------------------------------------------------------------------------------------
# scripts/wo07_real_eval.py, protocol real-0.1, mock rig with two cameras and the capture
# ------------------------------------------------------------------------------------------------
class QuitAfter(ScriptedOperator):
    def __init__(self, confirms: int) -> None:
        self.left = confirms

    def confirm(self, msg: str) -> bool:
        self.left -= 1
        return self.left >= 0


def ckpt_dir(root: Path, payload: bytes = b"weights-a") -> Path:
    d = root / f"ckpt-{payload.hex()[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(payload)
    (d / "config.json").write_text('{"type": "smolvla"}\n')
    return d


def real_args(tmp: Path, *extra: str, family: str = "smolvla", payload: bytes = b"w") -> list[str]:
    return [
        "--preset", "real", "--mock", "--fake-clock", "--max-steps", "6", "--runs-dir", str(tmp),
        "--labeler", "mock-labeler", "--family", family, "--training-run-id", f"train-{family}",
        "--checkpoint-dir", str(ckpt_dir(tmp.parent, payload)), *extra,
    ]  # fmt: skip


def _results(rec: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(rec.read_text())["results"]
    return out


def test_mock_session_runs_the_schedule_and_captures_every_started_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data.recorder import load_demo_dataset

    runs = tmp_path / "runs"
    # quit at the 4th confirm: trial 1 done (2 confirms), trial 2's placement prompt quits -> a
    # not-started attempt, never an episode
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(3))
    rec1 = ev.main(real_args(runs, "--task", "PickPlace", "--trials", "1"))
    r1 = _results(rec1)
    assert r1["session_end"] == "operator_quit"
    started = [t for t in r1["trials"] if t["policy_started"]]
    attempts = [t for t in r1["trials"] if not t["policy_started"]]
    assert len(started) == 1 and len(attempts) == 1
    assert started[0]["capture"]["episode_index"] == 0 and started[0]["capture"]["frames"] == 6
    assert attempts[0]["capture"]["episode_index"] is None
    assert attempts[0]["outcome"] == "not_started"

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(real_args(runs, "--resume", rec1.parent.name))
    r2 = _results(rec2)
    assert r2["session_end"] == "completed" and r2["summary"]["complete"]
    sid = r2["session"]["session_id"]
    assert sid == rec1.parent.name
    scored = [t for t in r2["trials"] if t["policy_started"]]
    want = ev.session_items(r2["session"])  # the first item of each cell, in schedule order
    assert [(t["cell"], t["trial_idx"]) for t in scored] == [
        (w["cell"], w["trial_idx"]) for w in want
    ]  # the frozen order, across cells; the started slot was never re-run
    for t in scored:
        assert t["protocol_version"] == "real-0.1" and t["protocol_hash"] == PR.protocol_hash()
        assert t["schedule_sha256"] == SCHEDULES["eval"] and t["reportable"] is True
        assert t["labeler"] == "mock-labeler" and t["rubric"]["id"] == PR.RUBRIC_ID
        assert t["placement"]["planned_bowl_xy_m"] == list(PR.BOWL_XY_M)
        assert t["instruction"] == t["placement"]["instruction"]
        assert t["trial_id"] == f"{sid}/{t['cell']}/{t['trial_idx']:03d}"
        assert t["policy_identity"]["family_id"] == "so101-real-smolvla"
    rows = r2["summary"]["real_rows"]
    assert rows["unseen_objects (pooled, headline)"]["n"] == 1
    # one dataset for the session; the resume appended; episode <-> trial both ways
    capture = runs / sid / "capture"
    ds_root = capture / "dataset"
    prov = json.loads((ds_root / "meta" / "vla_rl_so101.json").read_text())
    assert prov["runs"] == [rec1.parent.name, rec2.parent.name]
    assert {int(k): v["trial_id"] for k, v in prov["episodes"].items()} == {
        t["capture"]["episode_index"]: t["trial_id"] for t in scored
    }
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert conv.check_dataset_info(info) == [] and info["total_episodes"] == 3
    ds = load_demo_dataset("local/x", ds_root)
    for t in scored:
        ep = ds.meta.episodes[t["capture"]["episode_index"]]
        assert ep["tasks"] == [t["instruction"]] and ep["length"] == 6
        s = ds[int(ep["dataset_from_index"])]
        front = s[conv.IMAGE_KEYS["front"]]
        wrist = s[conv.IMAGE_KEYS["wrist"]]
        assert float((front + wrist - 1.0).abs().mean()) < 0.05  # wrist = 255 - front: no swap
        for cam in conv.CAMERA_NAMES:
            assert (capture / t["capture"]["end_state"][cam]).is_file()
    assert (ds_root / "meta" / "calibration_follower.json").is_file()


def test_dev_schedule_is_never_reportable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    argv = real_args(tmp_path / "runs", "--task", "PickPlace", "--trials", "1", "--no-capture")
    rec = ev.main([*argv, "--schedule", "dev", "--cells", "seen"])
    r = _results(rec)
    t = r["trials"][0]
    assert t["schedule"] == "dev" and t["reportable"] is False
    assert t["schedule_sha256"] == SCHEDULES["dev"] and not r["summary"]["reportable"]
    assert t["placement"]["planned_object_yaw_deg"] % 15 != 0  # the disjoint half-step yaws


def test_real_hardware_refusals_before_anything_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ev.load_config("real")
    s = {
        "schedule": "eval",
        "trials_per_cell": 2,
        "cells": ["seen"],
        "labeler": None,
        "policy_identity": None,
        "capture": True,
    }
    out = "\n".join(ev.real_hardware_problems(cfg, s))
    assert "--labeler is required" in out and "policy identity" in out
    assert "whole frozen schedule" in out
    cfg["eval"]["settle_s"] = 1.0
    assert any("settle_s" in p for p in ev.real_hardware_problems(cfg, s))
    with pytest.raises(SystemExit):  # parser: real-0.1's schedule is frozen
        ev.main(["--preset", "real", "--hardware", "--task", "PickPlace", "--slot-seed", "3"])
    with pytest.raises(SystemExit, match="PickPlace only"):
        ev.main(["--preset", "real", "--mock", "--task", "Push", "--no-capture"])


def test_report_pairs_two_policies_on_one_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    runs = tmp_path / "runs"
    common = ("--task", "PickPlace", "--trials", "1", "--no-capture")
    a = ev.main(real_args(runs, *common, family="smolvla", payload=b"a"))
    b = ev.main(real_args(runs, *common, family="act", payload=b"b"))
    out = ev.report([a.parent, b.parent])
    assert set(out["rows"]) == set(PR.cell_tables([]))
    assert "| row | smolvla | act |" in capsys.readouterr().out
    assert not out["protocol_complete"]  # 1 of 50 per cell
    same = ev.main(real_args(runs, *common, family="act", payload=b"a"))
    with pytest.raises(SystemExit, match="same checkpoint"):
        ev.report([a.parent, same.parent])
    data = json.loads(b.read_text())
    data["results"]["session"]["protocol"]["real_eval"]["schedule_sha256"] = "0" * 64
    b.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="different protocol hashes or schedules"):
        ev.report([a.parent, b.parent])


def test_resume_refuses_a_changed_policy_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))
    runs = tmp_path / "runs"
    rec = ev.main(real_args(runs, "--task", "PickPlace", "--trials", "1", "--no-capture"))
    data = json.loads(rec.read_text())
    st = data["results"]["session"]
    st["policy_identity"]["checkpoint_sha256"] = "f" * 64  # the session ran other weights
    rec.write_text(json.dumps(data))
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    with pytest.raises(SystemExit, match="resume refused: the checkpoint files changed"):
        ev.main(real_args(runs, "--resume", rec.parent.name))


def test_planned_pose_math_matches_the_grid() -> None:
    for t in PR.eval_schedule()[:20]:
        s = {x.id: x for x in G.all_slots()}[t["object_slot"]]
        assert t["planned_object_xy_m"] == [s.x, s.y]
        assert math.dist(t["planned_object_xy_m"], t["planned_bowl_xy_m"]) >= 0.11
