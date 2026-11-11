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
# value after the owner's kit list (2026-11-10: hand2mind 1-inch cubes, LE TAUCI cup with its
# measurements pending, white table, 5000 K lamp, cameras; then + the inside-floor height,
# 1d5674420b6d); before it dca199e6da62 (real-judge-3).
# The measurements, the bowl fit check, the rubric bump and the PDF follow.
FROZEN = {"real-0.1": "da110112b5d06c9b4a1f17cea62547defc8cad7f63de3ec98d1f9caa9511d4ab"}
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


def test_two_tabloid_placement_sheets(tmp_path: Path) -> None:
    """Owner kit list 2026-11-10: the grid on two taped tabloid (11 x 17 in) sheets. Two pages,
    deterministic bytes, a 100 mm scale bar and a label on each, registration ticks at the same
    heights on both sides of the join, the bowl cross ON sheet 2, the A3 file untouched."""
    grid_pdf = _load("make_placement_grid")
    pages = grid_pdf.draw_tabloid()
    assert len(pages) == 2 and pages == grid_pdf.draw_tabloid()
    data = grid_pdf.layout_bytes("tabloid2")
    assert data == grid_pdf.layout_bytes("tabloid2")  # deterministic
    assert data.count(b"/Type /Page ") == 2 and b"/MediaBox [0 0 792.00 1224.00]" in data
    s1, s2 = pages
    assert "(SHEET 1 of 2: grid + robot base" in s1 and "(SHEET 2 of 2: bowl side" in s2
    for sheet in pages:
        assert sheet.count("10 cm scale bar: must measure 100 mm on THIS sheet") == 1
        assert PR.protocol_hash()[:12] in sheet
        for i in (1, 2, 3):
            assert f"(R{i}) Tj" in sheet
    assert "CUT sheet 1 along this dashed line" in s1 and "JOIN line" in s2
    # the fixed bowl's centre lies on sheet 2, left of the join, with the cross-join ruler check
    bx, _ = grid_pdf.page(*PR.BOWL_XY_M)
    assert grid_pdf.SHEET2_X0_MM + 10.0 < bx < grid_pdf.JOIN_X_MM - 10.0
    assert "(BOWL CENTRE)" in s2 and "mm from slot C1's centre cross" in s2
    assert "PENDING the measured outer rim diameter" in s2
    # every slot cross is on sheet 1, right of its cut line
    for s in G.all_slots():
        assert grid_pdf.page(s.x, s.y)[0] - 3 > grid_pdf.JOIN_X_MM
    out = tmp_path / "tab.pdf"
    assert grid_pdf.main(["--preset", "real", "--layout", "tabloid2", "--out", str(out)]) == 0
    assert out.read_bytes() == data
    assert (
        grid_pdf.main(["--preset", "real", "--layout", "tabloid2", "--out", str(out), "--check"])
        == 0
    )
    assert grid_pdf.layout_bytes("a3") == grid_pdf.pdf_bytes(grid_pdf.draw())
    assert grid_pdf.OUTS["a3"] == grid_pdf.OUT  # the A3 default path is unchanged


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


def test_kit_objects_bowl_and_scene_are_the_owners_items() -> None:
    """Owner kit list 2026-11-10: hand2mind 1-inch wood cubes (red seen, purple + green unseen,
    blue the deferred Stack base), the LE TAUCI cup with its measurements PENDING in one place,
    and the scene facts (white table, 5000 K lamp, locked controls, both cameras) in the hash."""
    comp = PR.hash_components()
    seen, unseen = comp["objects"]["seen"], comp["objects"]["unseen"]
    assert seen["label"] == "red cube" and [o["label"] for o in unseen] == [
        "purple cube",
        "green cube",
    ]
    for o in (seen, *unseen, comp["objects"]["stack_base"]):
        assert o["edge_mm"] == 25.4 and "NOMINAL" in o["edge_source"]
        assert "hand2mind" in o["set"] and o["material"].startswith("wood")
        assert "as supplied in the set" in o["colour"]
    assert comp["objects"]["stack_base"]["label"] == "blue cube"
    assert "unused in real-0.1" in comp["objects"]["stack_base"]["use"]
    for o in (PR.SEEN_OBJECT, *PR.UNSEEN_OBJECTS, PR.STACK_BASE_OBJECT):
        assert o.size == pytest.approx(0.0254)  # never the sim's 3 cm on the real arm
    bowl = comp["bowl"]
    assert "LE TAUCI 8 oz bouillon cup, white" in bowl["product"] and bowl["colour"] == "white"
    for k in (
        "outer_rim_diameter_mm",
        "inside_floor_diameter_mm",
        "table_to_rim_height_mm",
        "table_to_inside_floor_height_mm",
    ):
        assert k in bowl and bowl[k] is None  # PENDING
    assert (
        comp["props"]["pending"] == PR.pending_measurements()
        and len(PR.pending_measurements()) == 5
    )
    # the deferred tasks' text carries the real props: the 2-inch disc, half the 25.4 mm edge
    assert "2-inch (50.8 mm) matte black" in PR.JUDGING_RULE["Push"]
    assert "~12.7 mm" in PR.JUDGING_RULE["Stack"] and "1.5 cm" not in PR.JUDGING_RULE["Stack"]
    assert all("deferred" in PR.JUDGING_RULE[t] for t in ("Push", "Stack"))
    scene = comp["scene"]
    assert scene["table_surface"] == "all white" and "5000 K" in scene["lighting"]
    assert "locked" in scene["camera_controls"] and "low contrast" in scene["risk"]
    from src.config import load_config

    sim_cam = load_config("sim")["env"]["camera"]
    assert scene["front_camera"]["eye_m"] == sim_cam["eye"]
    assert scene["front_camera"]["target_m"] == sim_cam["target"]
    assert scene["front_camera"]["device"] == "Logitech C920x"
    assert "32 mm" in scene["wrist_camera"]["device"]


def test_pending_measurements_refuse_the_freeze_and_move_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PR, "FREEZE_STATUS", PR.FROZEN)
    problem = PR.freeze_problem()  # frozen status alone is not enough
    assert problem is not None and "prop measurements PENDING" in problem
    assert "not frozen" not in problem and "bowl_table_to_rim_height_mm" in problem
    before = PR.protocol_hash()
    m = PR.PropMeasurements(25.2, 110.0, 70.0, 58.0, 8.0)
    monkeypatch.setattr(PR, "MEASURED", m)
    assert PR.pending_measurements() == [] and PR.freeze_problem() is None
    assert PR.cube_edge_mm() == (25.2, "measured (red cube)")
    assert PR.hash_components()["bowl"]["table_to_rim_height_mm"] == 58.0
    assert PR.protocol_hash() != before  # the measurements are hash inputs
    from src.eval import bowl_fit as BF

    assert BF.geometry().measured and BF.geometry().rim_height_m == pytest.approx(0.058)
    assert BF.geometry().floor_height_m == pytest.approx(0.008)


def test_measurements_in_the_wrong_unit_or_impossible_refuse_the_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F5: a value in inches / cm, or a cup the cube cannot sit in, never freezes."""
    monkeypatch.setattr(PR, "FREEZE_STATUS", PR.FROZEN)
    assert PR.measurement_problems(PR.PropMeasurements(25.2, 110.0, 70.0, 58.0, 8.0)) == []
    for m, what in (
        (PR.PropMeasurements(1.0, 110.0, 70.0, 58.0, 8.0), "red_cube_edge_mm"),  # inches
        (PR.PropMeasurements(25.2, 4.1, 2.6, 2.4, 0.3), "bowl_outer_rim_diameter_mm"),  # cm
        (PR.PropMeasurements(25.2, 110.0, 30.0, 58.0, 8.0), "bowl_inside_floor_diameter_mm"),
        (PR.PropMeasurements(25.2, 60.0, 70.0, 58.0, 8.0), "<= the inside floor"),
        (PR.PropMeasurements(25.2, 110.0, 70.0, 58.0, 60.0), "rim height"),
    ):
        monkeypatch.setattr(PR, "MEASURED", m)
        problem = PR.freeze_problem()
        assert problem is not None and "out of range" in problem and what in problem, m


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


def test_report_on_two_full_mock_sessions_is_labelled_mock_never_protocol_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Owner, PR #32 hold: two complete 150-trial mock sessions reported protocol_complete=True.
    Only hardware sessions make a protocol table; mock is labelled; mixed or unknown refuses."""
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    runs = tmp_path / "runs"
    full = ("--task", "PickPlace", "--trials", str(P1.REAL_TRIALS_PER_CELL), "--no-capture")
    full = (*full, "--max-steps", "1")
    a = ev.main(real_args(runs, *full, family="smolvla", payload=b"a"))
    b = ev.main(real_args(runs, *full, family="act", payload=b"b"))
    assert all(_results(r)["summary"]["complete"] for r in (a, b))  # 150 scored trials each
    capsys.readouterr()
    out = ev.report([a.parent, b.parent])
    printed = capsys.readouterr().out
    assert not out["protocol_complete"], "a mock report must never be protocol_complete"
    assert out["mode"] == "mock" and out["mock_complete"] and out["label"] == ev.MOCK_REPORT_LABEL
    assert printed.count(ev.MOCK_REPORT_LABEL) == 2  # the first and the last line

    def set_mode(rec: Path, mode: str | None) -> None:
        data = json.loads(rec.read_text())
        for blob in (data["results"], *data["results"]["trials"]):
            blob.pop("mode", None)
            if mode is not None:
                blob["mode"] = mode
        rec.write_text(json.dumps(data))

    for rec in (a, b):  # the same two records as if run on the arm
        set_mode(rec, "hardware")
    out = ev.report([a.parent, b.parent])
    assert out["protocol_complete"] and out["mode"] == "hardware"
    assert "MOCK" not in capsys.readouterr().out

    class Dcfg:
        latency_target_ms, capture_offset_ms = 150.0, None

    # review DR-3: a full 150-trial session with ONE episode not confirmed durable is not a
    # protocol table (a pending block, or a 5126daa record: its index beside the sync error)
    res = _results(a)
    trials, settings = res["trials"], res["session"]
    assert ev.table(trials, settings, Dcfg())["protocol_complete"]
    t0 = next(t for t in trials if t.get("policy_started") and t["outcome"] in ev.SCORED)
    for cap in (
        {"episode_index": None, "pending_episode_index": 0},
        {"episode_index": 0, "sync_error": "OSError: [Errno 5] (the staged journal is kept)"},
    ):
        t0["capture"] = cap
        summ = ev.table(trials, settings, Dcfg())
        assert summ["complete"] and summ["recordings_pending"] == [t0["trial_id"]]
        assert not summ["protocol_complete"]
    set_mode(b, "mock")
    with pytest.raises(SystemExit, match="mixes a hardware and a mock session"):
        ev.report([a.parent, b.parent])
    set_mode(b, None)  # no recorded mode: not hardware, refused
    with pytest.raises(SystemExit, match="no recorded session mode"):
        ev.report([a.parent, b.parent])


@pytest.fixture(scope="module")
def full_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str, str]:
    """Two finished 150-trial real-0.1 mock sessions (no capture, one command per trial):
    (runs dir, run id a, run id b). Tests copy the tree before changing it."""
    runs = tmp_path_factory.mktemp("full_pair") / "runs"
    full = ("--task", "PickPlace", "--trials", str(P1.REAL_TRIALS_PER_CELL), "--no-capture")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        a = ev.main(real_args(runs, *full, "--max-steps", "1", family="smolvla", payload=b"a"))
        b = ev.main(real_args(runs, *full, "--max-steps", "1", family="act", payload=b"b"))
    return runs, a.parent.name, b.parent.name


def _write_capture(capture_root: Path, sid: str, owners: dict[int, str]) -> None:
    """A stand-in capture dataset with what the recording rule reads (``recording_blockers``):
    its provenance (episode -> trial id) and its episode index table. No frames or video."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = capture_root / "dataset" / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    episodes = {str(k): {"trial_id": tid} for k, tid in sorted(owners.items())}
    (meta / "vla_rl_so101.json").write_text(json.dumps({"session_id": sid, "episodes": episodes}))
    pq.write_table(
        pa.table({"episode_index": sorted(owners)}),
        meta / "episodes" / "chunk-000" / "file-000.parquet",
    )


def _captured_hardware_copy(full_pair: tuple[Path, str, str], tmp: Path) -> tuple[Path, Path]:
    """A copy of ``full_pair`` relabelled as two captured hardware sessions: mode hardware,
    capture on, every scored trial claiming its own episode, which the session's capture
    dataset holds under that trial's id (``_write_capture``). Their SAVED summaries still say
    ``protocol_complete`` True. Returns the two record paths."""
    import shutil

    runs, ra, rb = full_pair
    shutil.copytree(runs.parent, tmp, dirs_exist_ok=True)
    recs = []
    for rid in (ra, rb):
        rec = tmp / "runs" / rid / "record.json"
        data = json.loads(rec.read_text())
        res = data["results"]
        res["mode"] = "hardware"
        res["session"]["capture"] = True
        owners: dict[int, str] = {}
        for t in res["trials"]:
            t["mode"] = "hardware"
            if t.get("policy_started") and t["outcome"] in ev.SCORED:
                t["capture"] = {"episode_index": len(owners), "frames": 1}
                owners[len(owners)] = t["trial_id"]
        assert res["summary"]["protocol_complete"]  # the saved summary, as 5126daa wrote it
        rec.write_text(json.dumps(data))
        sid = str(res["session"]["session_id"])
        _write_capture(tmp / "runs" / sid / "capture", sid, owners)
        recs.append(rec)
    return recs[0], recs[1]


def _capture_owners(rec: Path) -> tuple[Path, str, dict[int, str]]:
    """(capture root, session id, episode -> trial id) of a ``_captured_hardware_copy`` record."""
    res = json.loads(rec.read_text())["results"]
    sid = str(res["session"]["session_id"])
    owners = {
        int(t["capture"]["episode_index"]): t["trial_id"]
        for t in res["trials"]
        if (t.get("capture") or {}).get("episode_index") is not None
    }
    return rec.parent.parent / sid / "capture", sid, owners


def _first_scored(rec: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads(rec.read_text())
    t = next(t for t in data["results"]["trials"] if t.get("policy_started"))
    return data, t


def test_report_recomputes_completeness_and_never_trusts_a_stale_saved_summary(
    full_pair: tuple[Path, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Owner, PR #32 hold (issue 2): a legacy record (5126daa: an ``episode_index`` beside the
    save's ``sync_error``, or an index claimed while its journal is still staged) whose SAVED
    summary says ``protocol_complete`` True made ``--report`` print a complete table. The report
    recomputes completeness from the trial blocks and the capture on disk with the export's
    pending-recording rule."""
    from src.robot import recording

    a, b = _captured_hardware_copy(full_pair, tmp_path)
    assert ev.report([a.parent, b.parent])["protocol_complete"]  # every recording confirmed
    data, t0 = _first_scored(a)
    t0["capture"] = {"episode_index": 0, "sync_error": "OSError: [Errno 5] (journal kept)"}
    a.write_text(json.dumps(data))
    assert data["results"]["summary"]["protocol_complete"]  # stale: still says complete
    capsys.readouterr()
    out = ev.report([a.parent, b.parent])
    assert not out["protocol_complete"], "--report trusted the saved summary of a pending episode"
    assert t0["trial_id"] in capsys.readouterr().out

    t0["capture"] = {"episode_index": 0, "frames": 1}  # no error; its journal still staged
    a.write_text(json.dumps(data))
    sid = data["results"]["session"]["session_id"]
    staged = recording.staged_path(a.parent.parent / sid / "capture", t0["trial_id"])
    staged.mkdir(parents=True)
    (staged / recording.STAGED_INDEX).write_text("{}")
    out = ev.report([a.parent, b.parent])
    assert not out["protocol_complete"], "--report ignored a claimed episode still staged"
    assert staged.is_dir()  # the report never deletes staged data


def test_required_recordings_gate_completeness_and_scores_stand(
    full_pair: tuple[Path, str, str], tmp_path: Path
) -> None:
    """Owner, PR #32 hold: every scored trial whose policy run produced a frame (capture on)
    requires its recording; a scored trial whose FIRST write faulted (zero frames) requires
    none. Either way the trial stays scored in the running rates with its original score. A
    required recording pending, staged-unsaved or lost (``recording_missing``) blocks
    ``protocol_complete`` in the session summary and in ``--report``."""
    from src.eval import export_bundle as X
    from src.robot import recording

    class Dcfg:
        latency_target_ms, capture_offset_ms = 150.0, None

    a, b = _captured_hardware_copy(full_pair, tmp_path)
    data, t0 = _first_scored(a)
    res = data["results"]
    trials, settings = res["trials"], res["session"]
    root, sid, owners = _capture_owners(a)
    base = ev.table(trials, settings, Dcfg(), root)
    assert base["protocol_complete"]
    tid = t0["trial_id"]
    # t0's episode 0 is not in the dataset from here on (never saved, or lost)
    _write_capture(root, sid, {k: v for k, v in owners.items() if v != tid})
    t0["capture"] = {"episode_index": None, "frames": 0, "reason": recording.NO_POLICY_WRITE}
    summ = ev.table(trials, settings, Dcfg(), root)
    assert summ["protocol_complete"] and summ["real_rows"] == base["real_rows"]
    assert summ["cells"] == base["cells"]  # scored, in the rates, its score unchanged
    staged = recording.staged_path(root, tid)
    for cap, key in (
        ({"episode_index": None, "recording_missing": True, "reason": "no journal"}, "missing"),
        ({"episode_index": None, "reason": "ended before the episode was saved"}, "unsaved"),
        ({"episode_index": None, "pending_episode_index": 0, "sync_error": "OSError"}, "pending"),
    ):
        t0["capture"] = cap
        if key == "unsaved":  # a whole staged journal: a resume would save it
            staged.mkdir(parents=True)
            (staged / recording.STAGED_INDEX).write_text("{}")
        summ = ev.table(trials, settings, Dcfg(), root)
        assert not summ["protocol_complete"], cap
        gate = {
            k: summ[k] for k in ("recordings_pending", "recordings_unsaved", "recording_missing")
        }
        assert {k: v for k, v in gate.items() if v} == {
            {"missing": "recording_missing", "unsaved": "recordings_unsaved"}.get(
                key, "recordings_pending"
            ): [tid]
        }, (cap, gate)
        assert summ["real_rows"] == base["real_rows"] and summ["cells"] == base["cells"]
        got = X.recording_blockers(trials, root)
        assert [g["trial_id"] for g in got["pending"] + got["missing"]][:1] == [tid]
        a.write_text(json.dumps(data))  # the saved summary still says complete
        assert not ev.report([a.parent, b.parent])["protocol_complete"], cap
        if key == "unsaved":  # the report never deletes staged data
            assert (staged / recording.STAGED_INDEX).is_file()
    assert summ["recordings_pending"] == [tid]


def test_a_lost_capture_or_a_claim_it_does_not_hold_blocks_the_report(
    full_pair: tuple[Path, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review F1 (PR #32 hold): a REQUIRED recording lost on disk blocks a complete report and
    summary as it refuses the export: the capture dataset missing or unreadable, a claimed
    episode it does not hold, or one its provenance gives another trial (an older record of a
    session a later resume re-mapped). The scores stand. Review F2: an episode saved before its
    trial's capture row (the record keeps the placeholder block) is the trial's, as the export
    fills it: no blocker."""
    import shutil

    a, b = _captured_hardware_copy(full_pair, tmp_path)
    data, t0 = _first_scored(a)
    res = data["results"]
    trials, settings = res["trials"], res["session"]
    root, sid, owners = _capture_owners(a)
    tid = t0["trial_id"]
    targets = ev.SavedTargets(150.0)
    base = ev.table(trials, settings, targets, root)
    assert base["protocol_complete"] and ev.report([a.parent, b.parent])["protocol_complete"]

    t0["capture"] = {"episode_index": None, "reason": ev.CAPTURE_PENDING}  # F2: killed before row
    assert ev.table(trials, settings, targets, root)["protocol_complete"]

    other = next(v for v in owners.values() if v != tid)
    for owners_now in (
        {k: v for k, v in owners.items() if v != tid},  # episode 0 not in the dataset
        {**owners, 0: other},  # episode 0 is another trial's in the provenance
    ):
        t0["capture"] = {"episode_index": 0, "frames": 1}
        _write_capture(root, sid, owners_now)
        summ = ev.table(trials, settings, targets, root)
        assert summ["recording_missing"] == [tid] and not summ["protocol_complete"], owners_now
        assert summ["real_rows"] == base["real_rows"] and summ["cells"] == base["cells"]
        a.write_text(json.dumps(data))
        capsys.readouterr()
        assert not ev.report([a.parent, b.parent])["protocol_complete"]
        assert "recordings LOST" in capsys.readouterr().out
    _write_capture(root, sid, owners)
    assert ev.table(trials, settings, targets, root)["protocol_complete"]

    shutil.rmtree(root / "dataset")  # every required recording gone
    summ = ev.table(trials, settings, targets, root)
    assert len(summ["recording_missing"]) == len(owners) and not summ["protocol_complete"]
    assert summ["real_rows"] == base["real_rows"]
    lost = ev.recording_blockers(trials, root)["missing"]
    assert "missing or unreadable" in lost[0]["reason"], lost[0]
    assert not ev.report([a.parent, b.parent])["protocol_complete"]


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


def test_camera_controls_and_the_printed_grid_are_checked_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F3 / F5: a session refuses while the scene's locked controls are not recorded, and
    records the printed grid actually used (eval.grid_layout), not always the A3 file."""
    cfg = ev.load_config("real")
    problems = conv.camera_controls_problems(cfg)
    for k in ("front.controls.exposure", "front.controls.focus", "wrist.controls.set_with"):
        assert any(k in p for p in problems), k
    s = {"schedule": "eval", "trials_per_cell": 50, "cells": list(PR.CELLS), "capture": True}
    assert any(p.startswith("camera controls:") for p in ev.real_hardware_problems(cfg, s))
    for name in ("front", "wrist"):
        cfg["cameras"][name]["controls"].update(exposure=150, set_with="uvc tool 1.0", focus=30)
    assert conv.camera_controls_problems(cfg) == []
    cfg["cameras"]["wrist"]["controls"]["auto_white_balance"] = True
    assert conv.camera_controls_problems(cfg) == [
        "cameras.wrist.controls.auto_white_balance is True, must be false"
    ]
    assert ev.grid_layout_from_config(cfg) == "a3"
    a3 = ev.protocol_fields(protocol=PR.PROTOCOL_VERSION)["real_eval"]
    assert a3["grid_layout"] == "a3" and a3["grid_pdf_sha256"] is not None
    tab = ev.protocol_fields(protocol=PR.PROTOCOL_VERSION, grid_layout="tabloid2")["real_eval"]
    assert tab["grid_pdf"].endswith("placement-grid-tabloid2.pdf")
    assert tab["grid_pdf_sha256"] != a3["grid_pdf_sha256"]
    grid_pdf = _load("make_placement_grid")
    assert {k: str(v.relative_to(REPO_ROOT)) for k, v in grid_pdf.OUTS.items()} == ev.GRID_PDFS
    cfg["eval"]["grid_layout"] = "tabloid2"
    if not (REPO_ROOT / ev.GRID_PDFS["tabloid2"]).is_file():  # generated with the freeze
        assert any("not generated" in p for p in ev.real_hardware_problems(cfg, s))
    cfg["eval"]["grid_layout"] = "letter"
    with pytest.raises(ValueError, match="grid_layout"):
        ev.grid_layout_from_config(cfg)
