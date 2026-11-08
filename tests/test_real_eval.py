"""WO-07 real eval (scripts/wo07_real_eval.py), its placement grid (src/eval/real_grid.py) and the
printed grid (scripts/make_placement_grid.py), in mock mode: fake arm, fake camera, mock policy."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.eval import protocol as P1
from src.eval import protocol_v02 as P2
from src.eval import real_grid as G
from src.robot.client import ScriptedOperator


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")
grid_pdf = _load("make_placement_grid")


def test_grid_lattice_matches_the_protocol_regions() -> None:
    slots = G.all_slots()
    assert len(slots) == 36 and slots[0].id == "A1" and slots[-1].id == "F6"
    seen, shifted = G.cell_slots("seen"), G.cell_slots("shifted")
    assert len(seen) == 16 and len(shifted) == 20 and G.cell_slots("unseen_objects") == seen
    assert all(P1.SEEN_REGION.contains(s.x, s.y) for s in seen)
    assert all(
        P1.SHIFTED_REGION.contains(s.x, s.y) and not P1.SEEN_REGION.contains(s.x, s.y)
        for s in shifted
    )
    a1 = slots[0]  # A = nearest the base (smallest x), 1 = robot's left (largest y)
    assert a1.x == min(s.x for s in slots) and a1.y == max(s.y for s in slots)


@pytest.mark.parametrize("task", P1.TASKS)
def test_sample_plan_is_seeded_balanced_and_feasible(task: str) -> None:
    from src.env import tasks as T

    plan = G.sample_plan(task, "seen", 50, seed=0)
    assert plan == G.sample_plan(task, "seen", 50, seed=0)  # SFT and SFT+RL sessions pair up
    assert plan != G.sample_plan(task, "seen", 50, seed=1)
    seen_ids = {s.id for s in G.cell_slots("seen")}
    counts: dict[str, int] = {}
    for p in plan:
        assert p["object_slot"] in seen_ids and p["goal_slot"] in seen_ids
        assert abs(p["object_yaw_deg"]) <= 45 and p["object_yaw_deg"] % 15 == 0
        d = math.dist(p["object_xy_m"], p["goal_xy_m"])
        if task == "Push":
            assert T.PUSH_DIST[0] <= d <= T.PUSH_DIST[1]
        else:
            assert d >= T.MIN_SEP[task]
        counts[p["object_slot"]] = counts.get(p["object_slot"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1  # every valid slot used ~equally
    assert "slot " + plan[0]["object_slot"] in G.placement_prompt("red cube", plan[0])


def test_placement_pdf_is_committed_a3_and_current() -> None:
    assert grid_pdf.main(["--check"]) == 0  # the committed PDF is what the script draws now
    data = (REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf").read_bytes()
    assert data.startswith(b"%PDF-1.4") and b"/MediaBox [0 0 841.89 1190.55]" in data  # A3 pt
    assert b"10 cm scale bar" in data
    assert all(f"({s.id})".encode() in data for s in G.all_slots())  # ids = the sampler's
    ops = grid_pdf.draw()
    mm = grid_pdf.PT_PER_MM
    bar = f"{14.0 * mm:.2f} {360.0 * mm:.2f} m {114.0 * mm:.2f} {360.0 * mm:.2f} l S"
    assert bar in ops  # a 100 mm line at 1:1 (72 pt per inch)
    x, y = grid_pdf.page(0.1, 0.0)  # 10 cm along +x from the base origin is 100 mm up the page
    assert (x, y) == (grid_pdf.BASE_X_MM, grid_pdf.BASE_Y_MM + 100.0)


class QuitAfter(ScriptedOperator):
    def __init__(self, confirms: int) -> None:
        self.left = confirms

    def confirm(self, msg: str) -> bool:
        self.left -= 1
        return self.left >= 0


def _args(tmp: Path, *extra: str) -> list[str]:
    return ["--preset", "real", "--mock", "--fake-clock", "--max-steps", "16"] + [
        "--runs-dir",
        str(tmp),
        *extra,
    ]


def test_mock_session_records_protocol_02_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 trials per cell; the operator quits at the 5th confirm (2 confirms per trial): the seen
    # cell is done, shifted has not started
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(4))
    rec1 = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--label", "sft"))
    r1 = json.loads(rec1.read_text())["results"]
    assert r1["session_end"] == "operator_quit" and r1["shutdown"] == "parked"
    assert [t["cell"] for t in r1["trials"]] == ["seen", "seen"]
    assert not r1["summary"]["complete"] and not r1["summary"]["protocol_complete"]
    t0 = r1["trials"][0]
    assert t0["protocol_hash"] == P2.protocol_hash() and t0["protocol_version"] == "0.2"
    assert t0["protocol_ref"] == {"version": P1.PROTOCOL_VERSION, "hash": P1.protocol_hash()}
    assert t0["real_eval"]["judging_rule"] == ev.JUDGING_RULE_VERSION
    assert t0["placement"] == G.sample_plan("PickPlace", "seen", 2, 0)[0]
    assert "frame_return_to_first_write_ms" in t0["latency"]

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(_args(tmp_path, "--resume", rec1.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["resumed_from"] == rec1.parent.name and r2["session_end"] == "completed"
    assert r2["session"]["session_id"] == rec1.parent.name and r2["session"]["label"] == "sft"
    by_cell: dict[str, list[Any]] = {}
    for t in r2["trials"]:
        by_cell.setdefault(t["cell"], []).append(t["trial_idx"])
    assert by_cell == {c: [0, 1] for c in P1.CELLS}  # seen kept, the rest run once
    s = r2["summary"]
    assert s["complete"] and not s["protocol_complete"]  # 2, not 50, per cell
    for cell in P1.CELLS:
        c = s["cells"][f"PickPlace/{cell}"]
        assert c["n"] == 2 and c["target_n"] == 2
        assert c["wilson_95"] == pytest.approx(P1.wilson_interval(c["successes"], 2), abs=1e-4)
    shifted = [t for t in r2["trials"] if t["cell"] == "shifted"]
    ring = {sl.id for sl in G.cell_slots("shifted")}
    assert shifted and all(t["placement"]["object_slot"] in ring for t in shifted)
    assert np.isfinite(s["frame_return_to_first_write_ms"]["median"])

    # a changed evaluation (here: the session's recorded hash) refuses to resume
    rec = json.loads(rec2.read_text())
    rec["results"]["session"]["protocol"]["protocol_hash"] = "0" * 64
    rec2.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="resume refused"):
        ev.main(_args(tmp_path, "--resume", rec2.parent.name))


def test_hardware_mode_is_gated_like_the_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        ev.main(["--preset", "real", "--hardware", "--task", "PickPlace"])
    assert e.value.code == 2
    monkeypatch.setattr("builtins.input", lambda _msg="": "no")
    with pytest.raises(SystemExit, match="nothing was opened"):
        ev.main(["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"])
