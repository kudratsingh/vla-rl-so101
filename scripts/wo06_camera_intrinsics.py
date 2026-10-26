#!/usr/bin/env python
"""WO-06: checkerboard intrinsics for ``cameras.front``, written into configs/real.yaml.

Print a checkerboard (default 9x6 INNER corners = 10x7 squares, 25 mm squares; measure the printed
square and pass --square-mm), tape it flat to something rigid, then on the Mac (zsh):

    uv run scripts/wo06_camera_intrinsics.py --preset real --capture 20        # live from cameras.front
    uv run scripts/wo06_camera_intrinsics.py --preset real --images DIR        # photos already taken

Captures must be at the streaming resolution (cameras.front width x height). Vary the board pose
between captures: near/far, tilted ~30 deg each way, and covering every image corner. Writes
``cameras.front.intrinsics`` (fx, fy, cx, cy in px; OpenCV dist k1 k2 p1 p2 k3; rms_px) unless
--dry-run, and reports whether the lens covers the sim camera's fov (``cameras.front.fov_deg``).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(parser)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", type=int, metavar="N", help="grab N frames from cameras.front")
    src.add_argument("--images", type=Path, metavar="DIR", help="folder of board photos")
    parser.add_argument("--camera", default="front")
    parser.add_argument("--pattern", default="9x6", help="inner corners COLSxROWS (default 9x6)")
    parser.add_argument("--square-mm", type=float, default=25.0, help="printed square size, mm")
    parser.add_argument("--save-dir", type=Path, default=None, help="where --capture saves frames")
    parser.add_argument("--dry-run", action="store_true", help="do not write the config")
    args = parser.parse_args()
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    cfg = load_config(args.preset, args.config)
    cam_cfg = cfg["cameras"][args.camera]
    cols, rows = (int(v) for v in args.pattern.lower().split("x"))
    pattern = (cols, rows)

    if args.capture:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        save_dir = args.save_dir or REPO_ROOT / "data" / "camera_calib" / stamp
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
        f"fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.cx:.1f} cy={intr.cy:.1f} px; "
        f"dist={[round(d, 4) for d in intr.dist]}; full fov {hfov:.1f} x {vfov:.1f} deg"
    )
    if intr.rms_px is not None and intr.rms_px > 1.0:
        print("WARNING rms > 1 px: blurry or bent board, or too few varied poses; consider redoing")
    fov = float(cam_cfg.get("fov_deg", 60.0))
    try:
        win = fov_crop_window(intr, want[0], want[1], fov)
        print(f"sim fov {fov:.0f} deg -> crop window x[{win[0]}:{win[2]}] y[{win[1]}:{win[3]}] px")
    except ValueError as e:
        print(f"WARNING {e}")
    if args.dry_run:
        print("dry run: config not written")
        return
    path = Path(args.config) if args.config else CONFIG_DIR / f"{args.preset}.yaml"
    update_yaml_file(path, ["cameras", args.camera, "intrinsics"], intr.to_dict())
    print(f"wrote cameras.{args.camera}.intrinsics in {path}")


if __name__ == "__main__":
    main()
