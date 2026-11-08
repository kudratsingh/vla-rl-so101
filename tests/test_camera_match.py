"""scripts/wo06_camera_match.py: the sim side poses the arm at given joints on the CPU sim (Mac)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.config import REPO_ROOT, load_config

_spec = importlib.util.spec_from_file_location(
    "wo06_camera_match", REPO_ROOT / "scripts" / "wo06_camera_match.py"
)
assert _spec is not None and _spec.loader is not None
match = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(match)


@pytest.mark.needs_render
def test_with_arm_poses_the_cpu_sim_arm() -> None:
    sim_cfg = load_config("sim")
    sim_cfg["device"] = "cpu"  # what the script forces: make_env returns a SerialVecEnv
    rest = match.sim_frame(sim_cfg, "PickPlace", 0, "seen", None)
    posed = match.sim_frame(
        sim_cfg, "PickPlace", 0, "seen", np.array([0.5, 0.3, -0.3, 0.2, 0.4, 1.0])
    )
    assert rest is not None and posed is not None and rest.shape == posed.shape
    assert np.abs(rest.astype(int) - posed.astype(int)).mean() > 1.0  # the arm moved in the image
