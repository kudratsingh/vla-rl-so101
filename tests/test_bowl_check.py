"""scripts/wo07_bowl_check.py: the PickPlace oracle run kinematically through the REAL safety filter
(configs/real.yaml) at a bowl position; CPU, offline, no device."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.config import REPO_ROOT, load_config
from src.eval import bowl_fit as BF
from src.eval import protocol_real as PR
from src.robot.safety import load_sim_fk


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wo07_bowl_check_t", REPO_ROOT / "scripts" / "wo07_bowl_check.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wo07_bowl_check_t"] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_bowl_t", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_bowl_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_frozen_bowl_is_rejected_at_the_bowl_and_a_box_inside_one_passes() -> None:
    """The frozen (0.195, 0.205) bowl puts the TCP outside workspace_box_m (y <= 0.20) from the
    carry on: rejected. A bowl 15 mm inside the box face passes every bowl phase (only the
    gripper's intended -5 deg floor clip). The pick never depends on the bowl."""
    cfg = load_config("real")
    fk = load_sim_fk()
    frozen = bc.verify(cfg, PR.BOWL_XY_M, ["C4"], [0.0], fk, "safety")
    assert frozen["table"]["C4"]["carry"] == "reject"
    assert all(frozen["table"]["C4"][p] == "reject" for p in bc.BOWL_PHASES)
    assert frozen["table"]["C4"]["hover"] == "pass"
    assert not frozen["passes_carry_and_bowl_phases"]
    inside = bc.verify(cfg, (0.205, 0.185), ["C4"], [0.0], fk, "safety")
    assert inside["passes_carry_and_bowl_phases"]
    assert inside["cube_released_over_bowl_floor"] == "1/1"


def test_constraint_helpers_and_tiny_cli(tmp_path: Path) -> None:
    assert bc.central_slot_for(PR.BOWL_XY_M)["central_slot"] == PR.central_slot() == "C4"
    a = bc.analytic(PR.BOWL_XY_M)
    assert a["sep_margin_m"] >= 0 and a["reach_margin_m"] >= 0  # the frozen rule holds ...
    cfg = load_config("real")
    assert bc.box_margin(PR.BOWL_XY_M, cfg, 0.0) < 0  # ... the safety box does not
    out = bc.main(["--preset", "tiny", "--runs-dir", str(tmp_path), "--ik-limits", "safety"])
    (res,) = out["results"]
    assert set(res["table"]) == {"C4", "A1"} and set(res["table"]["C4"]) == set(bc.PHASES)
    assert set(res) >= {"release_window", "bowl_phase_clips", "joint_margin_bowl_phases_deg"}
    assert res["box_margin_search_offset_m"] == bc.box_margin(
        (res["bowl_xy_m"][0], res["bowl_xy_m"][1]), cfg, BF.TCP_TO_BOWL_OFFSET_M
    )


def test_release_window_and_the_pending_bowl_preflights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release window (the TCP disc over the bowl floor inside the box after the 5 mm
    tolerance), and the hardware preflights: the eval and the demo schedule refuse the frozen
    bowl (outside the box) before anything opens; a bowl inside it passes."""
    cfg = load_config("real")
    box = BF.workspace_box(cfg)
    r = bc.RELEASE_RADIUS_M
    # sim floor radius 45 mm (the cup's floor is PENDING) less the nominal 1-inch cube's 12.7 mm
    assert r == pytest.approx(0.045 - 0.0127) and r == BF.release_radius_m()
    assert BF.release_window((0.25, 0.0), box, r) == {
        "radius_m": 0.0323,
        "inside_fraction": 1.0,
        "inscribed_radius_m": 0.0323,
    }
    b = BF.release_window((0.115, 0.185), box, r)  # two faces 15 mm away, 10 mm after the 5 mm
    assert b["inscribed_radius_m"] == 0.01 and 0.4 < b["inside_fraction"] < 0.6
    assert BF.release_window(PR.BOWL_XY_M, box, r)["inscribed_radius_m"] == 0.0

    problem = BF.bowl_box_problem(cfg)
    assert problem is not None and "1.93 cm outside" in problem
    assert BF.bowl_box_problem(cfg, (0.205, 0.185)) is None
    ev = _load_script("wo07_real_eval")
    s = {"schedule": "eval", "trials_per_cell": 2, "cells": ["seen"], "capture": True}
    assert problem in ev.real_hardware_problems(cfg, s)

    teleop = _load_script("wo06_teleop_record")
    cfg_leader = load_config("real")
    cfg_leader["teleop"]["leader"] = {"type": "so101_leader", "port": "/dev/null"}
    monkeypatch.setattr(teleop, "require_terminal", lambda _what: None)
    monkeypatch.setattr(teleop, "load_config", lambda *_a, **_k: cfg_leader)
    monkeypatch.setattr(
        sys, "argv", ["wo06_teleop_record.py", "--preset", "real", "--task", "PickPlace",
                      "--demo-schedule"]
    )  # fmt: skip
    with pytest.raises(SystemExit, match="refusing the real-0.1 demo schedule; nothing was opened"):
        teleop.main()
    # the freeze status refuses on its own too (the bowl fit check is pending, PR.FREEZE_STATUS):
    # with a bowl that fits the box, the demo schedule and the hardware session still refuse
    monkeypatch.setattr(teleop, "bowl_box_problem", lambda *_a, **_k: None)
    with pytest.raises(SystemExit, match="not frozen \\(pending: bowl fit check\\)"):
        teleop.main()
    monkeypatch.setattr(ev, "bowl_box_problem", lambda *_a, **_k: None)
    assert PR.freeze_problem() in ev.real_hardware_problems(cfg, s)


def test_prop_geometry_reads_the_one_measurement_place() -> None:
    """Owner kit list 2026-11-10: the cube and cup geometry of every bowl check comes from
    protocol_real.MEASURED; pending values fall back to the sim bowl, labelled, and the nominal
    1-inch cube; filling MEASURED is the one edit that moves the release window and clearances."""
    g = BF.geometry()
    assert g.cube_half_m == pytest.approx(0.0127) and not g.measured
    assert "NOMINAL" in g.sources["cube_half_m"]
    for k in ("floor_radius_m", "rim_radius_m", "rim_height_m", "floor_height_m"):
        assert g.sources[k].startswith("PENDING")
    assert g.floor_height_m == pytest.approx(0.006)  # the sim floor stands in
    assert g.rim_height_m == pytest.approx(0.036)  # sim wall 30 + floor 6 mm stands in
    assert bc.GEOMETRY == g and bc.CUBE_HALF_M == g.cube_half_m
    # the oracle's carry: TCP 72 mm, cube centre 3 mm below it, half-edge 12.7 mm, sim rim 36 mm
    assert BF.carry_rim_clearance_m(g) == pytest.approx(0.072 - 0.003 - 0.0127 - 0.036)
    m = PR.PropMeasurements(
        red_cube_edge_mm=25.0,
        bowl_outer_rim_diameter_mm=110.0,
        bowl_inside_floor_diameter_mm=70.0,
        bowl_table_to_rim_height_mm=60.0,
        bowl_table_to_inside_floor_height_mm=8.0,
    )
    gm = BF.geometry(m)
    assert gm.measured and gm.to_dict()["all_measured"]
    assert (gm.cube_half_m, gm.floor_radius_m, gm.rim_radius_m, gm.rim_height_m) == (
        pytest.approx(0.0125),
        pytest.approx(0.035),
        pytest.approx(0.055),
        pytest.approx(0.060),
    )
    assert BF.release_radius_m(gm) == pytest.approx(0.0225)
    assert BF.carry_rim_clearance_m(gm) < 0  # a 60 mm cup: the oracle's carry hits the rim
    d = 0.0983  # B' to its nearest slot
    assert BF.cube_rim_clearance_m(d, gm) == pytest.approx(d - 0.055 - 0.0125 * 2**0.5 - 0.005)
    assert any("carry through the rim" in p for p in BF.geometry_problems(gm))
    assert BF.lower_floor_clearance_m(gm) == pytest.approx(0.006 + 0.012 - 0.008)


def test_a_tall_or_footed_cup_fails_the_bowl_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review F1 / F4: the kinematic scene models neither the rim nor the floor, so a measured cup
    whose rim the oracle's carry passes through, or whose inside floor sits above the oracle's
    lower, FAILS the check at a bowl whose phases all pass."""
    cfg = load_config("real")
    fk = load_sim_fk()
    assert BF.geometry_problems() == []  # the pending fallback (sim bowl, 25.4 mm cube) is clear
    ok = bc.check_bowl(cfg, (0.205, 0.185), ["C4"], [0.0], fk, "safety")
    assert ok["passes_carry_and_bowl_phases"] and bc._verdict(ok) == "pass"
    tall = BF.geometry(PR.PropMeasurements(25.4, 110.0, 70.0, 60.0, 6.0))
    footed = BF.geometry(PR.PropMeasurements(25.4, 110.0, 70.0, 45.0, 25.0))
    for g, what in ((tall, "carry through the rim"), (footed, "lower into the floor")):
        monkeypatch.setattr(bc, "GEOMETRY", g)
        v = bc.check_bowl(cfg, (0.205, 0.185), ["C4"], [0.0], fk, "safety")
        assert v["kinematic_phases_ok"] and not v["passes_carry_and_bowl_phases"]
        assert any(what in p for p in v["geometry_problems"])
        assert bc._verdict(v).startswith("FAIL (") and what in bc._verdict(v)
    # the release lands on the measured floor, not the sim's
    assert footed.floor_height_m == pytest.approx(0.025)
