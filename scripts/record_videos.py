"""Rollout videos of a checkpoint: one batch of greedy episodes per task, tiled into one MP4.
    uv run scripts/record_videos.py --preset sim --adapter <dir | hub path> --label sft-a4
    uv run scripts/record_videos.py --preset tiny --out /tmp/vids          # CPU smoke
Shows the policy's own camera (224 px). Tiles are the protocol's episode seeds for (task, cell,
seed), so a tile is the same scene in every checkpoint's video. Each tile's border turns green
(success) or red (failure) when its episode ends; it then holds its last frame. Writes
<out>/<label>/<task>-<cell>.mp4 (H.264, 15 fps) and summary.json. Not a protocol eval.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio.v2 as imageio
import numpy as np
import torch

from src.config import REPO_ROOT, add_config_args, load_config
from src.eval.sim_eval import episode_seeds, run_batch
from src.policy.factory import make_policy

GREEN, RED, GREY = (46, 160, 67), (207, 34, 46), (110, 118, 129)


def tile(frames: list[torch.Tensor], success: torch.Tensor, length: torch.Tensor) -> np.ndarray:
    """(T, N, H, W, 3) frames -> (T, rows*H', cols*W', 3) uint8 grid with outcome borders."""
    video = torch.stack(frames).to(torch.uint8).numpy()  # (T, N, H, W, 3)
    t_len, n, h, w, _ = video.shape
    cols = min(4, n)
    rows = math.ceil(n / cols)
    b = 4
    out = np.zeros((t_len, rows * (h + 2 * b), cols * (w + 2 * b), 3), dtype=np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y0, x0 = r * (h + 2 * b), c * (w + 2 * b)
        end = int(length[i])  # frames[0] is the reset frame, frames[end] the last step's
        for t in range(t_len):
            src = video[min(t, end), i]
            color = GREY if t < end else (GREEN if bool(success[i]) else RED)
            out[t, y0 : y0 + h + 2 * b, x0 : x0 + w + 2 * b] = color
            out[t, y0 + b : y0 + b + h, x0 + b : x0 + b + w] = src
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--adapter", default=None, help="SFT/RL checkpoint dir or Hub path")
    parser.add_argument("--label", default="policy", help="output subfolder name")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--cell", default="seen")
    parser.add_argument("--seed", type=int, default=0, help="protocol seed for episode seeds")
    parser.add_argument("--episodes", type=int, default=8, help="tiles per video")
    parser.add_argument("--max-steps", type=int, default=None, help="default env.max_episode_steps")
    parser.add_argument("--sim-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "videos"))
    args = parser.parse_args(argv)
    default = "sft_tiny.yaml" if args.preset == "tiny" else "sft.yaml"
    cfg = load_config(args.preset, args.config or REPO_ROOT / "configs" / default)
    device = str(cfg.get("device", "cpu"))
    protocol: Any = importlib.import_module("src.eval.protocol")
    env_mod: Any = importlib.import_module("src.env")
    tasks = args.tasks or list(cfg["env"]["tasks"])
    max_steps = int(args.max_steps or cfg["env"].get("max_episode_steps", 150))
    sim_cfg = {**cfg, "device": args.sim_device} if args.sim_device else cfg
    policy = make_policy(cfg, device, adapter=args.adapter or cfg["policy"].get("adapter"))
    policy.eval()
    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"adapter": str(args.adapter), "cell": args.cell, "seed": args.seed}
    for task in tasks:
        seeds = episode_seeds(
            args.seed,
            list(protocol.TASKS).index(task),
            list(protocol.CELLS).index(args.cell),
            args.episodes,
        )
        env = env_mod.make_env(sim_cfg, task, num_envs=args.episodes, cell=args.cell)
        try:
            frames: list[torch.Tensor] = []
            success, length = run_batch(
                policy.act_batch, env, env_mod.policy_obs, seeds, max_steps, frames=frames
            )
        finally:
            env.close()
        path = out_dir / f"{task}-{args.cell}.mp4"
        imageio.mimwrite(
            path,
            list(tile(frames, success, length)),
            fps=15,
            codec="libx264",
            quality=8,
            macro_block_size=8,
        )
        summary[task] = {
            "successes": int(success.sum()),
            "episodes": args.episodes,
            "episode_seeds": seeds,
            "video": path.name,
        }
        print(
            f"[record_videos] {args.label} {task}: {int(success.sum())}/{args.episodes} -> {path}"
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
