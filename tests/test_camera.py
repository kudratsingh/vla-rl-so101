"""Policy image pipeline, checkerboard intrinsics, config writing. Synthetic images only."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
import yaml
from robot_fakes import real_cfg

from src.config import REPO_ROOT, load_config
from src.robot.camera import (
    Intrinsics,
    PolicyImagePipeline,
    calibrate_intrinsics,
    fov_crop_window,
    sim_mismatches,
)
from src.robot.yamlio import set_yaml_value


def test_real_camera_config_matches_sim() -> None:
    real, sim = real_cfg(), load_config("sim")
    assert real["cameras"]["front"]["crop_to"] == 224  # OpenVLA-OFT input (lead, 2026-10-26)
    # this branch's sim.yaml still says image_size 128; WO-01 moves it to 224
    assert [m for m in sim_mismatches(real, sim) if "crop_to" not in m] == []
    bad = {"env": {"image_size": 128, "camera": {"fov_deg": 50, "pose": "front_left_45"}}}
    assert len(sim_mismatches(real, bad)) == 2
    assert real["robot"]["control_hz"] == 15


def test_uncalibrated_pipeline_centre_square() -> None:
    cfg = real_cfg()
    pipe = PolicyImagePipeline.from_config(cfg)  # intrinsics: null in the repo config
    assert not pipe.calibrated and pipe.window == (80, 0, 560, 480)
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, :80] = 255  # left margin outside the crop
    out = pipe(frame)
    side = cfg["cameras"]["front"]["crop_to"]  # size comes from the config
    assert out.shape == (side, side, 3) and out.dtype == np.uint8 and out.max() == 0
    with pytest.raises(ValueError):
        pipe(np.zeros((480, 640), np.uint8))


def test_calibrated_crop_matches_sim_fov() -> None:
    intr = Intrinsics(fx=400.0, fy=400.0, cx=330.0, cy=235.0, dist=(), width=640, height=480)
    x0, y0, x1, y1 = fov_crop_window(intr, 640, 480, 60.0)
    half = 400 * math.tan(math.radians(30))
    assert (x1 - x0) == pytest.approx(2 * half, abs=1) and (y1 - y0) == pytest.approx(
        2 * half, abs=1
    )
    assert (x0 + x1) / 2 == pytest.approx(330, abs=1) and (y0 + y1) / 2 == pytest.approx(235, abs=1)
    pipe = PolicyImagePipeline(640, 480, 224, 60.0, intr)
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(frame, (330, 235), 6, (255, 255, 255), -1)  # principal point -> image centre
    out = pipe(frame)
    ys, xs = np.nonzero(out[..., 0] > 60)
    assert xs.mean() == pytest.approx(111.5, abs=2) and ys.mean() == pytest.approx(111.5, abs=2)
    # a lens narrower than the sim fov cannot be matched by cropping
    with pytest.raises(ValueError, match="too narrow"):
        PolicyImagePipeline(640, 480, 224, 60.0, Intrinsics(600, 600, 320, 240, (), 640, 480))
    # undistortion path runs and keeps the shape
    dist = Intrinsics(400, 400, 320, 240, (-0.2, 0.05, 0.0, 0.0, 0.0), 640, 480)
    assert PolicyImagePipeline(640, 480, 224, 60.0, dist)(frame).shape == (224, 224, 3)
    assert Intrinsics.from_dict(dist.to_dict()) == dist


def _board(pattern: tuple[int, int], sq_px: int, margin: int) -> np.ndarray:
    cols, rows = pattern[0] + 1, pattern[1] + 1
    img = np.full((rows * sq_px + 2 * margin, cols * sq_px + 2 * margin), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * sq_px, margin + c * sq_px
                img[y : y + sq_px, x : x + sq_px] = 0
    return img


def _view(board: np.ndarray, k: np.ndarray, rvec: np.ndarray, t: np.ndarray, s: float, m: int):
    r, _ = cv2.Rodrigues(rvec)
    a = np.array([[s, 0, -m * s], [0, s, -m * s], [0, 0, 1.0]])  # board px -> metres
    h = k @ np.column_stack([r[:, 0], r[:, 1], t]) @ a
    gray = cv2.warpPerspective(board, h, (640, 480), flags=cv2.INTER_AREA, borderValue=180)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_checkerboard_intrinsics_recovers_known_camera() -> None:
    pattern, sq_px, margin, square_m = (9, 6), 40, 40, 0.025
    board = _board(pattern, sq_px, margin)
    k = np.array([[600.0, 0, 322.0], [0, 600.0, 238.0], [0, 0, 1]])
    s = square_m / sq_px
    w_m, h_m = board.shape[1] * s, board.shape[0] * s
    rng = np.random.default_rng(0)
    views = []
    for _ in range(10):
        rvec = np.array([*rng.uniform(-0.45, 0.45, 2), rng.uniform(-0.3, 0.3)])
        r, _ = cv2.Rodrigues(rvec)
        centre = np.array(
            [rng.uniform(-0.03, 0.03), rng.uniform(-0.02, 0.02), rng.uniform(0.5, 0.6)]
        )
        t = centre - r @ np.array([w_m / 2, h_m / 2, 0.0])
        views.append(_view(board, k, rvec, t, s, margin))
    views.append(np.full((480, 640, 3), 127, np.uint8))  # a frame without the board
    intr, found = calibrate_intrinsics(views, pattern, square_m)
    assert found[-1] is False and sum(found) == 10
    assert intr.fx == pytest.approx(600, rel=0.02) and intr.fy == pytest.approx(600, rel=0.02)
    assert intr.cx == pytest.approx(322, abs=8) and intr.cy == pytest.approx(238, abs=8)
    assert (intr.width, intr.height) == (640, 480) and intr.rms_px is not None
    assert intr.rms_px < 1.0
    with pytest.raises(ValueError, match="need"):
        calibrate_intrinsics(views[:3], pattern, square_m)


def test_yaml_value_written_in_place() -> None:
    text = (REPO_ROOT / "configs" / "real.yaml").read_text()
    intr = {"fx": 600.5, "fy": 601.0, "cx": 320.0, "cy": 240.0, "dist": [0.1, -0.2, 0, 0, 0]}
    out = set_yaml_value(text, ["cameras", "front", "intrinsics"], intr)
    before, after = yaml.safe_load(text), yaml.safe_load(out)
    assert after["cameras"]["front"]["intrinsics"] == intr
    after["cameras"]["front"]["intrinsics"] = None
    assert after == before  # nothing else changed
    assert out.count("#") == text.count("#")  # every comment kept
    assert "# filled in WO-06" in out
    lim = {"shoulder_pan": [-100.0, 100.0]}
    out2 = set_yaml_value(out, ["robot", "safety", "joint_limits_deg"], lim)
    assert yaml.safe_load(out2)["robot"]["safety"]["joint_limits_deg"] == lim
    assert out2.count("#") == text.count("#")
    # a block value is replaced as a whole
    out3 = set_yaml_value(out2, ["robot", "units"], {"arm_sign": [1, 1, 1, 1, -1]})
    assert yaml.safe_load(out3)["robot"]["units"] == {"arm_sign": [1, 1, 1, 1, -1]}
    assert yaml.safe_load(out3)["robot"]["safety"]["joint_limits_deg"] == lim
    with pytest.raises(KeyError):
        set_yaml_value(text, ["cameras", "side", "intrinsics"], 1)
