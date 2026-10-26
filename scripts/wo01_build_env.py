#!/usr/bin/env python
"""WO-01 deliverable 4: build each task's env, run the scripted oracle, record demos.

For every task in `env.tasks`: run seeded oracle episodes on the `seen` cell until the first
`data.demos_per_task` successful episodes are recorded AND at least `--eval-episodes` episodes
have run; report oracle success over all episodes run. All tasks go into ONE LeRobotDataset
(v3.0) at `<data.dataset_root>/<repo name>`, rebuilt from scratch each time.

    uv run scripts/wo01_build_env.py --preset tiny             # CPU, < 2 min
    uv run scripts/wo01_build_env.py --preset sim --push       # GPU box; pushes privately to the Hub
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args, load_config
from src.data.recorder import DemoRecorder
from src.runs import Run

# Demo episodes use seeds DEMO_SEED_BASE + cfg.seed * 100_000 + k (k = 0, 1, ...), disjoint from
# small eval seeds, so evaluating on the "seen" cell never replays a training episode.
DEMO_SEED_BASE = 1_000_000

EnvApi = tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]


def env_api() -> EnvApi:
    """(make_env, policy_obs, make_oracle) from the src.env / src.data.oracle contract."""
    from src.data.oracle import make_oracle
    from src.env import make_env, policy_obs

    return make_env, policy_obs, make_oracle


class LazyRecorder:
    """Creates the dataset at the first observation, so its H x W come from the env itself."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.rec: DemoRecorder | None = None

    def get(self, image_hw: tuple[int, int]) -> DemoRecorder:
        if self.rec is None:
            size = int(self.cfg["env"]["image_size"])
            if image_hw != (size, size):
                print(
                    f"WARNING: env renders {image_hw}, cfg env.image_size={size}; recording {image_hw}"
                )
            self.rec = DemoRecorder.from_config(self.cfg, image_hw=image_hw, overwrite=True)
        elif self.rec.image_hw != image_hw:
            raise ValueError(f"image size changed between tasks: {self.rec.image_hw} -> {image_hw}")
        return self.rec


def run_task(
    task: str,
    cfg: dict[str, Any],
    api: EnvApi,
    recorders: LazyRecorder,
    *,
    demos: int,
    eval_episodes: int,
    allow_failed: bool,
) -> dict[str, Any]:
    make_env, policy_obs, make_oracle = api
    t_start = time.monotonic()
    env = make_env(cfg, task, cell="seen", render=True)
    oracle = make_oracle(task, env)
    n = int(getattr(env.unwrapped, "num_envs", 1))
    max_steps = int(cfg["env"]["max_episode_steps"])
    max_episodes = max(eval_episodes, 3 * demos, n)
    seed0 = DEMO_SEED_BASE + int(cfg.get("seed", 0)) * 100_000

    n_run = n_success = recorded = env_steps = 0
    lengths: list[int] = []
    failed_pool: list[tuple[int, str, list[Any], list[Any], list[Any]]] = []
    while (recorded < demos or n_run < eval_episodes) and n_run < max_episodes:
        seeds = [seed0 + n_run + i for i in range(n)]
        obs, _ = env.reset(seed=seeds if n > 1 else seeds[0])
        h, w = policy_obs(obs)["image"].shape[1:3]
        recorder = recorders.get((int(h), int(w)))
        instructions = list(env.unwrapped.get_instructions())
        oracle.reset(obs)
        keep = recorded < demos or (allow_failed and len(failed_pool) < demos)
        images: list[np.ndarray[Any, Any]] = []
        states: list[np.ndarray[Any, Any]] = []
        actions: list[np.ndarray[Any, Any]] = []
        done = torch.zeros(n, dtype=torch.bool)
        success = torch.zeros(n, dtype=torch.bool)
        length = torch.zeros(n, dtype=torch.long)
        for _ in range(max_steps):
            po = policy_obs(obs)
            action = oracle.act(obs)
            if keep:
                images.append(po["image"].detach().cpu().numpy())
                states.append(po["state"].detach().cpu().float().numpy())
                actions.append(torch.as_tensor(action).detach().cpu().float().numpy())
            obs, _, term, trunc, info = env.step(action)
            env_steps += n
            length += (~done).long()
            succ = torch.as_tensor(info["success"]).detach().cpu().bool().reshape(n)
            success |= succ & ~done
            ended = torch.as_tensor(term).cpu().bool() | torch.as_tensor(trunc).cpu().bool()
            done |= succ | ended.reshape(-1).expand(n)
            if bool(done.all()):
                break

        for i in range(n):
            ep_len, ok = int(length[i]), bool(success[i])
            n_run += 1
            n_success += int(ok)
            lengths.append(ep_len)
            if not keep:
                continue
            frames = (
                [im[i] for im in images[:ep_len]],
                [st[i] for st in states[:ep_len]],
                [ac[i] for ac in actions[:ep_len]],
            )
            if ok and recorded < demos:
                for img, st, ac in zip(*frames, strict=True):
                    recorder.add_frame(img, st, ac, instructions[i])
                recorder.save_episode(
                    {
                        "task_family": task,
                        "seed": seeds[i],
                        "cell": "seen",
                        "oracle_success": True,
                    }
                )
                recorded += 1
            elif not ok and allow_failed and len(failed_pool) < demos:
                failed_pool.append((seeds[i], instructions[i], *frames))

    filled = 0
    if allow_failed:
        for seed, instruction, imgs, sts, acs in failed_pool[: max(0, demos - recorded)]:
            for img, st, ac in zip(imgs, sts, acs, strict=True):
                recorder.add_frame(img, st, ac, instruction)
            recorder.save_episode(
                {"task_family": task, "seed": seed, "cell": "seen", "oracle_success": False}
            )
            filled += 1
    env.close()
    wall = time.monotonic() - t_start
    return {
        "episodes": n_run,
        "oracle_successes": n_success,
        "oracle_success_rate": n_success / max(1, n_run),
        "demos_recorded": recorded,
        "failed_demos_filled": filled,
        "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
        "env_steps": env_steps,
        "num_envs": n,
        "wall_s": round(wall, 2),
    }


def print_table(stats: dict[str, dict[str, Any]]) -> None:
    head = f"{'task':<10} {'episodes':>8} {'oracle':>8} {'demos':>6} {'len':>6} {'sps':>7} {'wall_s':>7}"
    print(head)
    print("-" * len(head))
    for task, s in stats.items():
        demos = f"{s['demos_recorded']}" + (
            f"+{s['failed_demos_filled']}F" if s["failed_demos_filled"] else ""
        )
        sps = s["env_steps"] / max(s["wall_s"], 1e-9)
        print(
            f"{task:<10} {s['episodes']:>8} {100 * s['oracle_success_rate']:>7.1f}% {demos:>6}"
            f" {s['mean_episode_len']:>6.1f} {sps:>7.0f} {s['wall_s']:>7.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None, help="override cfg.device"
    )
    parser.add_argument("--push", action="store_true", help="push the dataset to the Hub (private)")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="min oracle episodes per task (default: eval.episodes_per_cell)",
    )
    args = parser.parse_args()

    cfg = load_config(args.preset, args.config)
    if args.device:
        cfg["device"] = args.device
    demos = int(cfg["data"]["demos_per_task"])
    eval_episodes = args.eval_episodes
    if eval_episodes is None:
        eval_episodes = int(cfg.get("eval", {}).get("episodes_per_cell", demos))
    # tiny proves plumbing only: if its short episodes cannot succeed, fill with flagged failures
    allow_failed = cfg["preset"] == "tiny"
    api = env_api()

    with Run(
        work_order="WO-01", script=__file__, cfg=cfg, device=str(cfg.get("device", "cpu"))
    ) as run:
        recorders = LazyRecorder(cfg)
        stats: dict[str, dict[str, Any]] = {}
        for task in cfg["env"]["tasks"]:
            stats[task] = run_task(
                task,
                cfg,
                api,
                recorders,
                demos=demos,
                eval_episodes=eval_episodes,
                allow_failed=allow_failed,
            )
            run.log_metrics(task=task, **stats[task])
        recorder = recorders.rec
        if recorder is None:
            raise RuntimeError("no episodes were run; check data.demos_per_task / --eval-episodes")
        recorder.finalize()
        run.results["image_hw"] = list(recorder.image_hw)
        run.results.update(
            tasks=stats,
            dataset={
                "repo_id": recorder.repo_id,
                "root": os.path.relpath(recorder.root, REPO_ROOT),
                "episodes": recorder.num_episodes,
                "frames": recorder.num_frames,
            },
        )
        if args.push:
            run.results["dataset"]["hub_url"] = recorder.push_to_hub(private=True)

    print(f"\nrun {run.id}  preset={cfg['preset']}  wall={run.record['wall_clock_s']}s")
    print_table(stats)
    print(
        f"dataset: {recorder.repo_id} -> {os.path.relpath(recorder.root, REPO_ROOT)}"
        f" ({recorder.num_episodes} episodes, {recorder.num_frames} frames)"
    )
    if args.push:
        print(f"pushed: {run.results['dataset']['hub_url']}")
    print(f"record: {run.record_path}")
    short = [t for t, s in stats.items() if s["demos_recorded"] < demos]
    if short and not allow_failed:
        print(f"ERROR: fewer than {demos} successful demos for {short}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
