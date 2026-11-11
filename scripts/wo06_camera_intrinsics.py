#!/usr/bin/env python
"""WO-06: checkerboard intrinsics for ``cameras.front`` AND ``cameras.wrist``, into configs/real.yaml.

The board (owner kit list 2026-11-10, ``docs/bringup/kit.md``): a 10 x 7-square checkerboard
(9 x 6 INNER corners, the default ``--pattern 9x6``) printed on letter paper at 100%, glued flat to
cardboard. Measure one printed square with a ruler over several squares (e.g. 9 squares / 9) and
pass it: ``--square-mm`` is REQUIRED (no default; a wrong square size scales every distance), and
it is recorded with the result (``intrinsics.board``). Run it once per camera, on the Mac (zsh):

    uv run scripts/wo06_camera_intrinsics.py --preset real --camera front --square-mm <measured> --capture 20
    uv run scripts/wo06_camera_intrinsics.py --preset real --camera wrist --square-mm <measured> --capture 20
    uv run scripts/wo06_camera_intrinsics.py --preset real --camera front --square-mm <measured> --images DIR

Captures must be at the streaming resolution (cameras.<camera> width x height), under the 5000 K
lamp with the camera's exposure, white balance and (C920x) focus already locked (WO-06 checklist
step 7 items 1-4 come BEFORE step 6): focus changes fx / fy and the distortion, so the front camera
refuses to write intrinsics while ``cameras.front.controls.autofocus`` is not false or ``focus`` is
null, and the controls in effect are recorded (``intrinsics.board.controls_at_capture``). Redo the
intrinsics if the focus ever changes. Vary the
board pose between captures: near/far, tilted ~30 deg each way, and covering every image corner
(the wrist module: hold the board in front of the gripper). Writes
``cameras.<camera>.intrinsics`` (fx, fy, cx, cy in px; OpenCV dist k1 k2 p1 p2 k3; rms_px; the board
pattern and the measured square size) unless --dry-run. For the front camera it also reports
whether the lens covers the sim camera's fov (``cameras.front.fov_deg``); the wrist stream is
recorded uncropped (``src/robot/convention.py``), so it has no crop check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG_DIR, add_config_args, load_config
from src.robot.camera import (
    CameraFault,
    OpenCVFrameSource,
    calibrate_intrinsics,
    find_checkerboard,
    fov_crop_window,
)
from src.robot.yamlio import update_yaml_file

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_images(folder: Path) -> list[np.ndarray[Any, Any]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip unreadable {p}")
            continue
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"loaded {len(images)} images from {folder}")
    return images


def capture(cam_cfg: dict[str, Any], n: int, pattern: tuple[int, int], save_dir: Path) -> list[Any]:
    save_dir.mkdir(parents=True, exist_ok=True)
    images: list[Any] = []
    with OpenCVFrameSource(cam_cfg) as cam:
        while len(images) < n:
            input(f"[{len(images) + 1}/{n}] move the board to a new pose, press ENTER ")
            try:
                img = cam.read().image
            except CameraFault as e:
                print(f"camera: {e}; try again")
                continue
            ok = find_checkerboard(img, pattern) is not None
            print("  board found" if ok else "  board NOT found - reposition and retry")
            if ok:
                path = save_dir / f"calib_{len(images):02d}.png"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                images.append(img)
    print(f"saved {len(images)} frames to {save_dir}")
    return images


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", type=int, metavar="N", help="grab N frames from the camera")
    src.add_argument("--images", type=Path, metavar="DIR", help="folder of board photos")
    parser.add_argument("--camera", choices=("front", "wrist"), default="front")
    parser.add_argument(
        "--pattern", default="9x6", help="inner corners COLSxROWS (default 9x6 = 10 x 7 squares)"
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        required=True,
        help="the MEASURED printed square size, mm (required: no default)",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="where --capture saves frames")
    parser.add_argument("--dry-run", action="store_true", help="do not write the config")
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    if not 5.0 <= args.square_mm <= 60.0:
        parser.error(f"--square-mm {args.square_mm}: measure the printed square, in mm")
    cfg = load_config(args.preset, args.config)
    cam_cfg = cfg["cameras"][args.camera]
    controls = dict(cam_cfg.get("controls") or {})
    if args.camera == "front" and (
        controls.get("autofocus") is not False or controls.get("focus") is None
    ):
        msg = (
            "cameras.front.controls: autofocus must be false and the locked focus recorded "
            "(focus) BEFORE the intrinsics: focus changes fx / fy and the distortion (WO-06 step 7 "
            "item 4 comes before step 6)"
        )
        if not args.dry_run:
            sys.exit(f"refusing to write intrinsics: {msg}")
        print(f"WARNING {msg}")
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern = (cols, rows)

    if args.capture:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = args.save_dir or REPO_ROOT / "data" / "camera_calib" / f"{args.camera}-{stamp}"
        images = capture(cam_cfg, args.capture, pattern, save_dir)
    else:
        images = load_images(args.images)
    want = (int(cam_cfg["width"]), int(cam_cfg["height"]))
    sizes = {(im.shape[1], im.shape[0]) for im in images}
    if sizes != {want}:
        sys.exit(f"images are {sorted(sizes)}; cameras.{args.camera} streams {want[0]}x{want[1]}")

    intr, found = calibrate_intrinsics(images, pattern, args.square_mm / 1000.0)
    print(f"board found in {sum(found)}/{len(found)} images; reprojection rms {intr.rms_px:.3f} px")
    hfov, vfov = intr.fov_deg()
    print(
        f"cameras.{args.camera}: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} "
        f"cy={intr.cy:.1f} px; dist={[round(d, 4) for d in intr.dist]}; full fov {hfov:.1f} x "
        f"{vfov:.1f} deg"
    )
    if intr.rms_px is not None and intr.rms_px > 1.0:
        print("WARNING rms > 1 px: blurry or bent board, or too few varied poses; consider redoing")
    if args.camera == "front":
        fov = float(cam_cfg.get("fov_deg", 60.0))
        try:
            win = fov_crop_window(intr, want[0], want[1], fov)
            print(
                f"sim fov {fov:.0f} deg -> crop window x[{win[0]}:{win[2]}] y[{win[1]}:{win[3]}] px"
            )
        except ValueError as e:
            print(f"WARNING {e}")
    out = intr.to_dict()
    out["board"] = {
        "inner_corners": f"{cols}x{rows}",
        "squares": f"{cols + 1}x{rows + 1}",
        "square_mm": round(float(args.square_mm), 3),
        "square_source": "measured by the owner (--square-mm)",
        "images_used": int(sum(found)),
        # the camera controls (focus, exposure, white balance) these intrinsics were taken under
        "controls_at_capture": controls,
    }
    print(f"board {out['board']}")
    if args.dry_run:
        print("dry run: config not written")
        return out
    path = Path(args.config) if args.config else CONFIG_DIR / f"{args.preset}.yaml"
    update_yaml_file(path, ["cameras", args.camera, "intrinsics"], out)
    print(f"wrote cameras.{args.camera}.intrinsics in {path}")
    return out


if __name__ == "__main__":
    main()
