"""WO-02 sim evaluation: greedy policy, tasks x cells x seeds, one table from one command.

    uv run scripts/wo02_eval.py --preset sim --adapter runs/<sft-id>/checkpoints/step_010000
    uv run scripts/wo02_eval.py --preset sim --adapter <dir> --cells seen --seeds 0 --episodes 50
    uv run scripts/wo02_eval.py --preset tiny                 # CPU smoke (tiny random OFT)

Defaults for sim/rl presets are the frozen protocol's (src/eval/protocol.py: all tasks, all cells,
SIM_SEEDS, SIM_EPISODES_PER_CELL); anything narrower is marked `protocol_complete: false` (e.g. a
checkpoint sweep). Writes runs/<id>/eval.json + eval.md (+ record.json).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import REPO_ROOT, Config, add_config_args, config_hash, load_config
from src.eval.sim_eval import CellSeedResult, evaluate, summarize, to_markdown
from src.policy.factory import make_policy


def default_config(preset: str) -> Path:
    return REPO_ROOT / "configs" / ("sft_tiny.yaml" if preset == "tiny" else "sft.yaml")


def plan(cfg: Config, args: argparse.Namespace, protocol: Any) -> dict[str, Any]:
    """Tasks/cells/seeds/episodes: CLI > tiny preset's cfg.eval > the protocol."""
    tiny = cfg["preset"] == "tiny"
    ev = cfg.get("eval", {})
    tasks = args.tasks or (list(cfg["env"]["tasks"]) if tiny else list(protocol.TASKS))
    cells = args.cells or (list(ev.get("cells", ["seen"])) if tiny else list(protocol.CELLS))
    seeds = args.seeds or (list(ev.get("seeds", [0])) if tiny else list(protocol.SIM_SEEDS))
    episodes = args.episodes or (
        int(ev.get("episodes_per_cell", 4)) if tiny else int(protocol.SIM_EPISODES_PER_CELL)
    )
    complete = (
        list(tasks) == list(protocol.TASKS)
        and list(cells) == list(protocol.CELLS)
        and list(seeds) == list(protocol.SIM_SEEDS)
        and episodes == int(protocol.SIM_EPISODES_PER_CELL)
    )
    return {
        "tasks": tasks,
        "cells": cells,
        "seeds": seeds,
        "episodes": episodes,
        "complete": complete,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--adapter", default=None, help="SFT/RL checkpoint dir or Hub id")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--cells", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="per (task, cell, seed)")
    parser.add_argument("--num-envs", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.preset, args.config or default_config(args.preset))
    device = str(cfg.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"preset {args.preset!r} needs a CUDA GPU (use --preset tiny on CPU)")
    adapter = args.adapter or cfg["policy"].get("adapter")
    if adapter is None and not cfg["policy"].get("tiny_random", False):
        raise ValueError("--adapter (an SFT/RL checkpoint) is required outside the tiny preset")

    protocol: Any = importlib.import_module("src.eval.protocol")  # WO-01 (frozen in WO-02)
    env_mod: Any = importlib.import_module("src.env")  # WO-01
    runs: Any = importlib.import_module("src.runs")  # WO-01
    p = plan(cfg, args, protocol)
    policy = make_policy(cfg, device, adapter=adapter)
    policy.eval()
    num_envs = int(args.num_envs or cfg["env"].get("num_envs", 16))
    max_steps = int(cfg["env"].get("max_episode_steps", 150))

    def make_env(task: str, cell: str, n: int) -> Any:
        return env_mod.make_env(cfg, task, num_envs=n, cell=cell)

    with runs.Run(work_order="WO-02-eval", script=__file__, cfg=cfg, device=device) as run:

        def log(r: CellSeedResult) -> None:
            run.log_metrics(**r.__dict__, rate=r.successes / r.episodes)
            print(f"[wo02_eval] {r.task:9s} {r.cell:15s} seed {r.seed}: {r.successes}/{r.episodes}")

        results = evaluate(
            policy.act_batch,
            make_env,
            env_mod.policy_obs,
            tasks=p["tasks"],
            cells=p["cells"],
            seeds=p["seeds"],
            episodes=p["episodes"],
            num_envs=num_envs,
            max_steps=max_steps,
            all_tasks=list(protocol.TASKS),
            all_cells=list(protocol.CELLS),
            log=log,
        )
        meta = {
            "policy": str(adapter or "tiny-random"),
            "config_hash": config_hash(cfg),
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_hash": protocol.protocol_hash(),
            "protocol_complete": p["complete"],
            "plan": p,
            "run_id": run.id,
        }
        summary = summarize(results, protocol.wilson_interval, meta)
        (run.dir / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
        md = to_markdown(summary)
        (run.dir / "eval.md").write_text(md)
        run.results.update(
            protocol_hash=meta["protocol_hash"],
            protocol_complete=p["complete"],
            table={f"{r['task']}/{r['cell']}": r["pooled"] for r in summary["rows"]},
        )
        print(md)
        return dict(summary, run_dir=str(run.dir))


if __name__ == "__main__":
    main()
