#!/usr/bin/env python
"""WO-06: real front-camera frame vs sim front-camera render, side by side, same policy pipeline.

Real side: one frame from ``cameras.front`` (or --real-image), through ``src.robot.camera``'s
PolicyImagePipeline - the exact crop/resize the policy sees in WO-07. Sim side: the sim ``front``
camera via ``src.env.make_env`` (lands from WO-01; imported lazily, skipped with a message until
then). Saves ``docs/assets/camera-match/<task>_seed<seed>_<time>.png`` (+ ``.json`` colour stats):
[real full frame with the crop window | real policy view | sim policy view | 50/50 blend].

On the Mac (zsh; the sim renders on CPU through MoltenVK):

    source scripts/mac_env.sh
    uv run scripts/wo06_camera_match.py --preset real                          # all tasks, seed 0
    uv run scripts/wo06_camera_match.py --preset real --task PickPlace --seed 3
    uv run scripts/wo06_camera_match.py --preset real --real-image photo.png   # no camera needed
    uv run scripts/wo06_camera_match.py --preset real --with-arm               # sim arm at real joints

Place the real objects where the sim panel shows them; in the blend panel table edges, robot base
and objects must overlap. ``--with-arm`` reads the follower's joints (read-only; refuses if torque
is on) and poses the sim arm identically (an error, not a silent rest pose, if that fails):
overlapping arms validate the unit mapping and camera pose.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args, load_config
from src.robot.camera import (
    OpenCVFrameSource,
    PolicyImagePipeline,
    sim_mismatches,
)
from src.robot.units import UnitMap

PANEL = 320
Img = np.ndarray[Any, Any]


def real_frame(cfg: dict[str, Any], image: Path | None) -> Img:
    if image is not None:
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            sys.exit(f"cannot read {image}")
        return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    with OpenCVFrameSource(cfg["cameras"]["front"]) as cam:
        return cam.read().image


def read_arm_joints(cfg: dict[str, Any], bus: Any = None) -> Img:
    """Follower joint angles (canonical rad). Opens the port only: no configure, torque untouched.

    Refuses if any servo is torqued (e.g. after a crashed session): the operator poses the arm by
    hand, and turning torque off here would drop an unsupported arm.
    """
    if bus is None:
        from src.robot.driver import follower_from_config

        bus = follower_from_config(cfg).bus
    bus.connect()
    try:
        if not bus.is_calibrated:
            sys.exit("follower motors do not match the calibration file; run wo06_calibrate.py")
        torqued = [m for m, v in bus.sync_read("Torque_Enable").items() if int(v)]
        if torqued:
            sys.exit(
                f"follower torque is ON ({', '.join(torqued)}; crashed session?): support the arm "
                "and run `wo06_calibrate.py --preset real verify` (or power-cycle the servo "
                "supply), then pose the limp arm and retry"
            )
        raw = bus.sync_read("Present_Position")
    finally:
        bus.disconnect(disable_torque=False)
    return UnitMap.from_config(cfg).to_canonical({f"{m}.pos": v for m, v in raw.items()})


def sim_base(env: Any) -> Any:
    """The ManiSkill env holding the robot: on CPU ``make_env`` returns a ``SerialVecEnv`` (no
    ``.agent``; its ``.unwrapped`` is itself), so take its single sub-env."""
    return env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped


def pose_sim_arm(env: Any, qpos: Img) -> None:
    """Set the sim arm to ``qpos`` (canonical rad). Fails loudly: --with-arm without a posed sim
    arm would show the rest pose and the overlay check would be meaningless."""
    import torch

    robot = sim_base(env).agent.robot
    robot.set_qpos(torch.as_tensor(np.asarray(qpos)[None], dtype=torch.float32))
    got = np.asarray(robot.get_qpos().cpu().numpy()).reshape(-1)
    if not np.allclose(got, qpos, atol=1e-4):
        raise RuntimeError(f"sim arm not posed: asked {np.round(qpos, 4)}, got {np.round(got, 4)}")


def sim_frame(
    sim_cfg: dict[str, Any], task: str, seed: int, cell: str, qpos: Img | None
) -> Img | None:
    """Sim front-camera policy image, or None (with the reason printed) before WO-01 lands."""
    try:
        from src.env import make_env, policy_obs

        env = make_env(sim_cfg, task, num_envs=1, cell=cell, render=True)
    except (ImportError, NotImplementedError, AttributeError) as e:
        print(
            f"sim side unavailable ({type(e).__name__}: {e}). It needs src.env.make_env from the "
            "WO-01 branch (wo-01-sim-env); saving the real side only."
        )
        return None
    try:
        obs, _ = env.reset(seed=seed)
        if qpos is not None:
            pose_sim_arm(env, qpos)
            obs = sim_base(env).get_obs()
        img = policy_obs(obs)["image"][0]
        return np.asarray(img.cpu().numpy() if hasattr(img, "cpu") else img, dtype=np.uint8)
    finally:
        env.close()


def _panel(img: Img, label: str, width: int = PANEL) -> Img:
    h = PANEL
    interp = cv2.INTER_NEAREST if img.shape[0] < h else cv2.INTER_AREA
    body = cv2.resize(img, (width, h), interpolation=interp)
    bar = np.full((26, width, 3), 255, np.uint8)
    cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def compose(raw: Img, window: tuple[int, int, int, int], real: Img, sim: Img | None) -> Img:
    framed = raw.copy()
    x0, y0, x1, y1 = window
    cv2.rectangle(framed, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 2)
    w_raw = round(PANEL * raw.shape[1] / raw.shape[0])
    panels = [_panel(framed, "real frame + policy crop", w_raw), _panel(real, "real (policy view)")]
    if sim is None:
        blank = np.full_like(real, 200)
        panels += [_panel(blank, "sim: needs WO-01 make_env"), _panel(blank, "blend: n/a")]
    else:
        sim_r = cv2.resize(sim, real.shape[1::-1], interpolation=cv2.INTER_AREA)
        blend = cv2.addWeighted(real, 0.5, sim_r, 0.5, 0.0)
        panels += [_panel(sim_r, "sim (policy view)"), _panel(blend, "50/50 blend")]
    return np.hstack(panels)


def colour_stats(real: Img, sim: Img | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "real_mean_rgb": real.reshape(-1, 3).mean(0).round(1).tolist(),
        "real_std_rgb": real.reshape(-1, 3).std(0).round(1).tolist(),
    }
    if sim is not None:
        sim_r = cv2.resize(sim, real.shape[1::-1], interpolation=cv2.INTER_AREA)
        out["sim_mean_rgb"] = sim_r.reshape(-1, 3).mean(0).round(1).tolist()
        out["sim_std_rgb"] = sim_r.reshape(-1, 3).std(0).round(1).tolist()
        diff = np.abs(real.astype(np.float32) - sim_r.astype(np.float32))
        out["mean_abs_diff"] = round(float(diff.mean()), 2)
    return out


def print_camera_placement(sim_cfg: dict[str, Any]) -> None:
    try:
        from src.env.cameras import resolve_front_camera

        spec = resolve_front_camera(sim_cfg["env"].get("camera"))
        print(
            f"sim camera '{spec.pose_name}': eye {spec.eye} m, looking at {spec.target} m "
            f"(robot base frame: +x forward, +y robot's left, z up), fov {spec.fov_deg} deg"
        )
    except (ImportError, AttributeError, KeyError) as e:
        print(f"(sim camera placement unavailable until WO-01 lands: {type(e).__name__})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--sim-config", type=Path, default=None, help="default configs/sim.yaml")
    parser.add_argument("--task", action="append", default=None, help="repeatable; default all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell", default="seen")
    parser.add_argument("--real-image", type=Path, default=None, help="use a saved photo")
    parser.add_argument("--with-arm", action="store_true", help="pose the sim arm at real joints")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "assets" / "camera-match")
    args = parser.parse_args()
    if args.preset != "real":
        parser.error("WO-06 hardware scripts run with --preset real")
    cfg = load_config(args.preset, args.config)
    sim_cfg = load_config("sim", args.sim_config)
    sim_cfg["device"] = "cpu"  # the Mac renders the single sim env on CPU
    for m in sim_mismatches(cfg, sim_cfg):
        print(f"WARNING camera config mismatch, {m}")
    print_camera_placement(sim_cfg)

    pipe = PolicyImagePipeline.from_config(cfg)
    raw = real_frame(cfg, args.real_image)
    real = pipe(raw)
    qpos = read_arm_joints(cfg) if args.with_arm else None
    tasks = args.task or list(sim_cfg["env"]["tasks"])
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for task in tasks:
        sim = sim_frame(sim_cfg, task, args.seed, args.cell, qpos)
        name = f"{task}_seed{args.seed}_{stamp}"
        img = compose(raw, pipe.window, real, sim)
        cv2.imwrite(str(args.out / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        meta = {
            "task": task,
            "seed": args.seed,
            "cell": args.cell,
            "calibrated_intrinsics": pipe.calibrated,
            "crop_window_px": list(pipe.window),
            "policy_image_px": pipe.out_size,
            "camera_pose": cfg["cameras"]["front"].get("pose"),
            "real_joints_rad": None if qpos is None else [round(float(v), 4) for v in qpos],
            **colour_stats(real, sim),
        }
        (args.out / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"saved {args.out / name}.png  {json.dumps(colour_stats(real, sim))}")
        if sim is None:
            break  # same real-only image for every task


if __name__ == "__main__":
    main()
