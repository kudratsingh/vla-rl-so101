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


def test_sim_rule_bowl_is_rejected_at_the_bowl_and_the_frozen_b_prime_passes() -> None:
    """The sim rule's (0.195, 0.205) bowl puts the TCP outside workspace_box_m (y <= 0.20) from
    the carry on: rejected. The frozen B' (0.205, 0.185), 15 mm inside the box face, passes every
    bowl phase (only the gripper's intended -5 deg floor clip). The pick never depends on the
    bowl."""
    cfg = load_config("real")
    fk = load_sim_fk()
    sim_rule = bc.verify(cfg, PR.BOWL_SIM_RULE_XY_M, ["C4"], [0.0], fk, "safety")
    assert sim_rule["table"]["C4"]["carry"] == "reject"
    assert all(sim_rule["table"]["C4"][p] == "reject" for p in bc.BOWL_PHASES)
    assert sim_rule["table"]["C4"]["hover"] == "pass"
    assert not sim_rule["passes_carry_and_bowl_phases"]
    inside = bc.verify(cfg, PR.BOWL_XY_M, ["C4"], [0.0], fk, "safety")
    assert PR.BOWL_XY_M == (0.205, 0.185)
    assert inside["passes_carry_and_bowl_phases"]
    assert inside["cube_released_over_bowl_floor"] == "1/1"


def test_constraint_helpers_and_tiny_cli(tmp_path: Path) -> None:
    assert bc.central_slot_for(PR.BOWL_XY_M)["central_slot"] == PR.central_slot() == "C4"
    assert bc.central_slot_for(PR.BOWL_SIM_RULE_XY_M)["central_slot"] == "C4"
    a = bc.analytic(PR.BOWL_SIM_RULE_XY_M)
    assert a["sep_margin_m"] >= 0 and a["reach_margin_m"] >= 0  # the sim rule holds ...
    cfg = load_config("real")
    assert bc.box_margin(PR.BOWL_SIM_RULE_XY_M, cfg, 0.0) < 0  # ... the safety box does not
    b = bc.analytic(PR.BOWL_XY_M)  # B': the box holds, the separation is the real-only exception
    assert b["sep_margin_m"] == pytest.approx(-0.0167, abs=1e-4) and b["reach_margin_m"] >= 0
    assert b["min_slot_distance_m"] >= PR.BOWL_EXCEPTION["real_min_sep_m"]
    assert b["cube_rim_clearance_m"] > 0
    assert bc.box_margin(PR.BOWL_XY_M, cfg, BF.TCP_TO_BOWL_OFFSET_M) >= 0
    out = bc.main(["--preset", "tiny", "--runs-dir", str(tmp_path), "--ik-limits", "safety"])
    (res,) = out["results"]
    assert set(res["table"]) == {"C4", "A1"} and set(res["table"]["C4"]) == set(bc.PHASES)
    assert set(res) >= {"release_window", "bowl_phase_clips", "joint_margin_bowl_phases_deg"}
    assert res["box_margin_search_offset_m"] == bc.box_margin(
        (res["bowl_xy_m"][0], res["bowl_xy_m"][1]), cfg, BF.TCP_TO_BOWL_OFFSET_M
    )


def test_release_window_and_the_bowl_preflights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release window (the TCP disc over the bowl floor inside the box after the 5 mm
    tolerance), and the hardware preflights: only the approved B' passes; the sim-rule bowl
    (outside the box) and any other bowl (even one inside the box) are refused by the eval and the
    demo schedule before anything opens, and so is a protocol that is not frozen."""
    cfg = load_config("real")
    box = BF.workspace_box(cfg)
    r = bc.RELEASE_RADIUS_M
    assert BF.release_window((0.25, 0.0), box, r) == {
        "radius_m": 0.03,
        "inside_fraction": 1.0,
        "inscribed_radius_m": 0.03,
    }
    b = BF.release_window((0.115, 0.185), box, r)  # two faces 15 mm away, 10 mm after the 5 mm
    assert b["inscribed_radius_m"] == 0.01 and 0.4 < b["inside_fraction"] < 0.6
    assert BF.release_window(PR.BOWL_SIM_RULE_XY_M, box, r)["inscribed_radius_m"] == 0.0
    bp = BF.release_window(PR.BOWL_XY_M, box, r)
    assert bp["inscribed_radius_m"] == 0.01 and bp["inside_fraction"] == pytest.approx(
        0.71, abs=0.01
    )

    assert BF.bowl_box_problem(cfg) is None and BF.bowl_box_problem(cfg, (0.205, 0.185)) is None
    for other in ((0.205, 0.180), (0.115, 0.185), (0.205, -0.185)):  # inside the box, not approved
        assert BF.box_margin(other, box, BF.TCP_TO_BOWL_OFFSET_M) >= 0
        other_problem = BF.bowl_box_problem(cfg, other)
        assert other_problem is not None and "not the approved real-0.1 bowl" in other_problem
    ev = _load_script("wo07_real_eval")
    s = {"schedule": "eval", "trials_per_cell": 2, "cells": ["seen"], "capture": True}
    assert not any("bowl" in p for p in ev.real_hardware_problems(cfg, s))
    # the protocol bowl moved off the approved one (an edit without re-approval): refused
    monkeypatch.setattr(PR, "BOWL_XY_M", PR.BOWL_SIM_RULE_XY_M)
    problem = BF.bowl_box_problem(cfg)
    assert problem is not None and "not the approved real-0.1 bowl" in problem
    assert problem in ev.real_hardware_problems(cfg, s)
    monkeypatch.setattr(BF, "APPROVED_BOWL_XY_M", PR.BOWL_SIM_RULE_XY_M)  # approved, outside box
    problem = BF.bowl_box_problem(cfg)
    assert problem is not None and "1.93 cm outside" in problem
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
    # the freeze status refuses on its own too (a protocol that is not frozen): with the bowl
    # guard passing, the demo schedule and the hardware session still refuse
    monkeypatch.setattr(teleop, "bowl_box_problem", lambda *_a, **_k: None)
    monkeypatch.setattr(PR, "FREEZE_STATUS", "pending: bowl fit check")
    with pytest.raises(SystemExit, match="not frozen \\(pending: bowl fit check\\)"):
        teleop.main()
    monkeypatch.setattr(ev, "bowl_box_problem", lambda *_a, **_k: None)
    assert PR.freeze_problem() in ev.real_hardware_problems(cfg, s)
