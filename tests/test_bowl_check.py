"""scripts/wo07_bowl_check.py: the PickPlace oracle run kinematically through the REAL safety filter
(configs/real.yaml) at a bowl position; CPU, offline, no device."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from src.config import REPO_ROOT, load_config
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
