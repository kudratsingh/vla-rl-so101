"""WO-01 benchmark: scripted-oracle success per task and cell.

    source scripts/mac_env.sh
    uv run python scripts/wo01_oracle_eval.py --preset sim --device cpu --episodes 100 \
        --cells seen,shifted --out runs/wo01-oracle/oracle_eval.json

Episodes run in batches of `--num-envs` sub-envs; batch b is reset with seed `--seed + b`
(per-env seeds derived as ManiSkill does). An episode succeeds if the env's success check fires
before the time limit (ManiSkill terminates on success).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import add_config_args, config_hash, load_config
from src.data.oracle import make_oracle
from src.env import CELLS, TASKS, make_env
from src.eval.protocol import PROTOCOL_VERSION, protocol_hash, wilson_interval


def run_cell(
    cfg: dict[str, Any], task: str, cell: str, episodes: int, num_envs: int, seed: int
) -> dict[str, Any]:
    env = make_env(cfg, task, num_envs=num_envs, cell=cell, render=False)
    oracle = make_oracle(task, env)
    max_steps = int(cfg["env"]["max_episode_steps"])
    successes, lengths = [], []
    t0 = time.time()
    for b in range(math.ceil(episodes / num_envs)):
        obs, _ = env.reset(seed=seed + b)
        oracle.reset(obs)
        done = torch.zeros(num_envs, dtype=torch.bool)
        success = torch.zeros(num_envs, dtype=torch.bool)
        length = torch.full((num_envs,), max_steps)
        for t in range(max_steps):
            obs, _, term, trunc, info = env.step(oracle.act(obs))
            term, trunc = term.cpu().bool(), trunc.cpu().bool()
            newly = (term | trunc) & ~done
            success |= newly & info["success"].cpu().bool()
            length[newly] = t + 1
            done |= newly
            if bool(done.all()):
                break
        take = min(num_envs, episodes - b * num_envs)
        successes += success[:take].tolist()
        lengths += length[:take].tolist()
    env.close()
    k, n = int(sum(successes)), len(successes)
    lo, hi = wilson_interval(k, n)
    ok_len = [ln for ln, s in zip(lengths, successes, strict=True) if s]
    return {
        "successes": k,
        "episodes": n,
        "rate": k / n,
        "ci95": [round(lo, 4), round(hi, 4)],
        "mean_steps_success": round(sum(ok_len) / len(ok_len), 1) if ok_len else None,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default=None, help="override cfg device (Mac: cpu)")
    parser.add_argument(
        "--num-envs", type=int, default=None, help="default: cfg env.num_envs on cuda, 1 on cpu"
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--cells", default="seen,shifted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.preset, args.config)
    if args.device:
        cfg["device"] = args.device
    num_envs = args.num_envs or (int(cfg["env"]["num_envs"]) if cfg["device"] != "cpu" else 1)
    tasks = args.tasks.split(",")
    cells = args.cells.split(",")
    for c in cells:
        if c not in CELLS:
            raise SystemExit(f"unknown cell {c}")

    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        results[task] = {}
        for cell in cells:
            r = run_cell(cfg, task, cell, args.episodes, num_envs, args.seed)
            results[task][cell] = r
            print(
                f"{task:10s} {cell:15s} {r['successes']:4d}/{r['episodes']:<4d} "
                f"{100 * r['rate']:5.1f}%  CI95 [{100 * r['ci95'][0]:.1f}, {100 * r['ci95'][1]:.1f}]  "
                f"steps {r['mean_steps_success']}  ({r['seconds']}s)",
                flush=True,
            )

    print("\n| task | " + " | ".join(cells) + " |")
    print("|---|" + "---|" * len(cells))
    for task in tasks:
        row = [
            f"{100 * results[task][c]['rate']:.0f}% ({results[task][c]['successes']}/{results[task][c]['episodes']})"
            for c in cells
        ]
        print(f"| {task} | " + " | ".join(row) + " |")

    if args.out:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
        except OSError:
            commit = ""
        record = {
            "benchmark": "wo01_oracle",
            "preset": args.preset,
            "device": cfg["device"],
            "num_envs": num_envs,
            "seed": args.seed,
            "config_hash": config_hash(cfg),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_hash": protocol_hash(),
            "git_commit": commit,
            "command": " ".join(sys.argv),
            "results": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
