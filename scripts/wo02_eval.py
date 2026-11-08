"""WO-02 sim evaluation: greedy policy, tasks x cells x seeds, one table from one command.

    uv run scripts/wo02_eval.py --preset sim --adapter runs/<sft-id>/checkpoints/step_010000
    uv run scripts/wo02_eval.py --preset sim --adapter <dir> --cells seen --seeds 0 --episodes 50
    uv run scripts/wo02_eval.py --preset tiny                 # CPU smoke (tiny random OFT)

    uv run scripts/wo02_eval.py --preset sim --adapter <dir> --protocol 0.2   # 4 deg/step clip

Defaults for sim/rl presets are the frozen protocol's (src/eval/protocol.py: all tasks, all cells,
SIM_SEEDS, SIM_EPISODES_PER_CELL); anything narrower is marked `protocol_complete: false` (e.g. a
checkpoint sweep). Writes runs/<id>/eval.json + eval.md + episodes.jsonl (one record per episode)
(+ record.json).

`--protocol` picks the protocol version (default 0.1-draft, the version every reported number
cites). 0.2 (src/eval/protocol_v02.py) adds the real arm's per-step joint clip from
configs/real.yaml, a composite hash, and marks a run whose config deviates from its settings
(horizon, image size, backend, camera, chunk rule) `protocol_complete: false`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.config import REPO_ROOT, Config, add_config_args, config_hash, load_config
from src.eval.sim_eval import CellSeedResult, EpisodeRecord, evaluate, summarize, to_markdown
from src.policy.factory import make_policy

PROTOCOL_MODULES = {"0.1-draft": "src.eval.protocol", "0.2": "src.eval.protocol_v02"}


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
    parser.add_argument(
        "--exec-steps",
        type=int,
        default=None,
        help="execute the first N actions of each chunk, then re-query (default: whole chunk)",
    )
    parser.add_argument(
        "--sim-device",
        choices=["cuda", "cpu"],
        default=None,
        help="sim backend, independent of the policy device (default: cfg device; cpu is serial)",
    )
    parser.add_argument(
        "--protocol",
        choices=sorted(PROTOCOL_MODULES),
        default="0.1-draft",
        help="eval protocol version (default 0.1-draft; 0.2 adds the 4 deg/step clip)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.preset, args.config or default_config(args.preset))
    device = str(cfg.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"preset {args.preset!r} needs a CUDA GPU (use --preset tiny on CPU)")
    adapter = args.adapter or cfg["policy"].get("adapter")
    if adapter is None and not cfg["policy"].get("tiny_random", False):
        raise ValueError("--adapter (an SFT/RL checkpoint) is required outside the tiny preset")

    # 0.1-draft: WO-01 (frozen in WO-02); 0.2: composite hash + action clip
    protocol: Any = importlib.import_module(PROTOCOL_MODULES[args.protocol])
    env_mod: Any = importlib.import_module("src.env")  # WO-01
    runs: Any = importlib.import_module("src.runs")  # WO-01
    p = plan(cfg, args, protocol)
    policy = make_policy(cfg, device, adapter=adapter)
    policy.eval()
    num_envs = int(args.num_envs or cfg["env"].get("num_envs", 16))
    max_steps = int(cfg["env"].get("max_episode_steps", 150))

    sim_cfg = {**cfg, "device": args.sim_device} if args.sim_device else cfg
    sim_device = str(sim_cfg.get("device", "cpu"))
    v02: dict[str, Any] = {}
    action_clip: float | None = None
    if args.protocol == "0.2":
        action_clip = float(protocol.action_clip().bound_rad)
        deviations = protocol.deviations(cfg, sim_device=sim_device, exec_steps=args.exec_steps)
        v02 = {  # hash + components read once, before any episode runs
            "protocol_hash": protocol.protocol_hash(),
            "action_clip_rad": action_clip,
            "protocol_deviations": deviations,
            "protocol_components": protocol.hash_components(),
        }
        p["complete"] = p["complete"] and not deviations
        for d in deviations:
            print(f"[wo02_eval] protocol 0.2 deviation (protocol_complete: false): {d}")

    def make_env(task: str, cell: str, n: int) -> Any:
        return env_mod.make_env(sim_cfg, task, num_envs=n, cell=cell)

    with runs.Run(work_order="WO-02-eval", script=__file__, cfg=cfg, device=device) as run:

        def log(r: CellSeedResult) -> None:
            run.log_metrics(**r.__dict__, rate=r.successes / r.episodes)
            print(f"[wo02_eval] {r.task:9s} {r.cell:15s} seed {r.seed}: {r.successes}/{r.episodes}")

        with (run.dir / "episodes.jsonl").open("w") as episodes_file:

            def log_episode(rec: EpisodeRecord) -> None:
                episodes_file.write(json.dumps(asdict(rec)) + "\n")

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
                exec_steps=args.exec_steps,
                action_clip=action_clip,
                log_episode=log_episode,
            )
        meta = {
            "policy": str(adapter or "tiny-random"),
            "config_hash": config_hash(cfg),
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_hash": protocol.protocol_hash(),
            "protocol_complete": p["complete"],
            "exec_steps": args.exec_steps,
            "sim_device": sim_device,
            "plan": p,
            "run_id": run.id,
            "episode_records": "episodes.jsonl",
            **v02,
        }
        summary = summarize(results, protocol.wilson_interval, meta)
        (run.dir / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
        md = to_markdown(summary)
        (run.dir / "eval.md").write_text(md)
        run.results.update(
            protocol_version=meta["protocol_version"],
            protocol_hash=meta["protocol_hash"],
            protocol_complete=p["complete"],
            table={f"{r['task']}/{r['cell']}": r["pooled"] for r in summary["rows"]},
        )
        print(md)
        return dict(summary, run_dir=str(run.dir))


if __name__ == "__main__":
    main()
