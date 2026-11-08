#!/usr/bin/env python
"""WO-01 deliverable 4: build each task's env, run the scripted oracle, record demos.

For every task in `env.tasks`: run seeded oracle episodes on the `seen` cell until the first
`data.demos_per_task` successful episodes are recorded AND at least `--eval-episodes` episodes
have run; report oracle success over all episodes run. All tasks go into ONE LeRobotDataset
(v3.0) at `<data.dataset_root>/<repo name>`, rebuilt from scratch each time.

A config can record each demo past its success step through release and retreat
(`TailSettings`, `data.post_success_steps` / `data.post_success_max_steps`) and pick the
`data.oracle_profile` oracle (src/data/oracle.py). The defaults are the 0.1-era settings the
existing sets (KudratSingh22/vla-rl-so101-demos, -demos-25) were recorded with: no tail, oracle
v1. configs/demos_v2.yaml opts in to the v2 oracle and the 15/45 tail. `--push` refuses those two
0.1 repos unless `--overwrite-hub` is passed: every reported SFT/RL number trained on them.

    uv run scripts/wo01_build_env.py --preset tiny             # CPU, < 2 min
    uv run scripts/wo01_build_env.py --preset sim --push       # GPU box; pushes privately to the Hub
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
# The 0.1-era demo sets every reported SFT/RL number trained on: --push needs --overwrite-hub
PROTECTED_HUB_REPOS = ("KudratSingh22/vla-rl-so101-demos", "KudratSingh22/vla-rl-so101-demos-25")

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


@dataclass(frozen=True)
class TailSettings:
    """How far a demo is recorded past the step its success check first fires.

    The 0.1-era demo sets stopped at that step, often mid-release (11 of 25 PickPlace and 13 of 25
    Stack demos end with the gripper at 0.22-0.29 rad, short of the 0.45 release target, with no
    retreat; review 2026-11-07 section 3.2). With `min_steps > 0` recording continues for at
    least `min_steps` frames and until the oracle has finished its script (release target
    reached, retreat done), capped at `max_steps` frames. The DEFAULTS (0 / 0 / no final check)
    reproduce the 0.1 recorder, so sim.yaml / sft_a4.yaml keep their behaviour and config hash;
    configs/demos_v2.yaml (and demos_v2_tiny.yaml) opt in with 15 / 45 / true.
    """

    min_steps: int = 0  # data.post_success_steps (demos_v2: 15)
    max_steps: int = 0  # data.post_success_max_steps (demos_v2: 45)
    require_final_success: bool = False  # data.require_final_success: success holds at the end

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> TailSettings:
        data = cfg.get("data", {})
        out = cls(
            min_steps=int(data.get("post_success_steps", cls.min_steps)),
            max_steps=int(data.get("post_success_max_steps", cls.max_steps)),
            require_final_success=bool(
                data.get("require_final_success", cls.require_final_success)
            ),
        )
        if not 0 <= out.min_steps <= out.max_steps:
            raise ValueError(f"need 0 <= post_success_steps <= post_success_max_steps: {out}")
        return out


def oracle_profile(cfg: dict[str, Any]) -> str:
    from src.data.oracle import DEFAULT_PROFILE

    return str(cfg.get("data", {}).get("oracle_profile", DEFAULT_PROFILE))


def run_task(
    task: str,
    cfg: dict[str, Any],
    api: EnvApi,
    recorders: LazyRecorder,
    *,
    demos: int,
    eval_episodes: int,
    allow_failed: bool,
    num_envs: int | None,
) -> dict[str, Any]:
    make_env, policy_obs, make_oracle = api
    t_start = time.monotonic()
    env = make_env(cfg, task, num_envs=num_envs, cell="seen", render=True)
    profile = oracle_profile(cfg)
    oracle = make_oracle(task, env, profile)
    tail = TailSettings.from_cfg(cfg)
    n = int(getattr(env.unwrapped, "num_envs", 1))
    horizon = int(cfg["env"]["max_episode_steps"])
    max_episodes = max(eval_episodes, 3 * demos, n)
    seed0 = DEMO_SEED_BASE + int(cfg.get("seed", 0)) * 100_000

    n_run = n_success = recorded = env_steps = tail_lost = 0
    lengths: list[int] = []
    rec_lengths: list[int] = []
    tails: list[int] = []
    finished_tails = 0
    failed_pool: list[tuple[int, str, list[Any], list[Any], list[Any]]] = []
    while (recorded < demos or n_run < eval_episodes) and n_run < max_episodes:
        seeds = [seed0 + n_run + i for i in range(n)]
        obs, _ = env.reset(seed=seeds if n > 1 else seeds[0])
        h, w = policy_obs(obs)["image"].shape[1:3]
        recorder = recorders.get((int(h), int(w)))
        instructions = list(env.unwrapped.get_instructions())
        oracle.reset(obs)
        keep = recorded < demos or (allow_failed and len(failed_pool) < demos)
        # batches run only for the success statistic stop at the success step
        tmin, tmax = (tail.min_steps, tail.max_steps) if keep else (0, 0)
        images: list[np.ndarray[Any, Any]] = []
        states: list[np.ndarray[Any, Any]] = []
        actions: list[np.ndarray[Any, Any]] = []
        done = torch.zeros(n, dtype=torch.bool)  # this env's recording has ended
        success = torch.zeros(n, dtype=torch.bool)  # success check fired within the horizon
        length = torch.zeros(n, dtype=torch.long)  # frames recorded
        succ_len = torch.zeros(n, dtype=torch.long)  # steps up to and including first success
        tail_n = torch.zeros(n, dtype=torch.long)  # frames recorded after the success step
        final_ok = torch.zeros(n, dtype=torch.bool)  # success check at the last recorded step
        finished = torch.zeros(n, dtype=torch.bool)  # oracle script complete when recording ended
        for t in range(horizon + tmax):
            po = policy_obs(obs)
            action = oracle.act(obs)
            if keep:
                images.append(po["image"].detach().cpu().numpy())
                states.append(po["state"].detach().cpu().float().numpy())
                actions.append(torch.as_tensor(action).detach().cpu().float().numpy())
            obs, _, term, trunc, info = env.step(action)
            env_steps += n
            live = ~done
            length += live.long()
            succ = torch.as_tensor(info["success"]).detach().cpu().bool().reshape(n)
            newly = live & ~success & succ & (t < horizon)
            success |= newly
            succ_len[newly] = t + 1
            tail_n += (live & success & ~newly).long()
            final_ok = torch.where(live, succ, final_ok)
            ended = torch.as_tensor(term).cpu().bool() | torch.as_tensor(trunc).cpu().bool()
            fail_end = live & ~success & (ended.reshape(-1).expand(n) | (t + 1 >= horizon))
            fin = oracle.finished().cpu()
            tail_end = live & success & (((tail_n >= tmin) & fin) | (tail_n >= tmax))
            finished = torch.where(tail_end, fin, finished)
            done |= fail_end | tail_end
            if bool(done.all()):
                break

        for i in range(n):
            ok = bool(success[i])
            ep_len = int(succ_len[i]) if ok else int(length[i])
            n_run += 1
            n_success += int(ok)
            lengths.append(ep_len)
            if not keep:
                continue
            rec_len = int(length[i])
            frames = (
                [im[i] for im in images[:rec_len]],
                [st[i] for st in states[:rec_len]],
                [ac[i] for ac in actions[:rec_len]],
            )
            if ok and recorded < demos:
                if tail.require_final_success and not bool(final_ok[i]):
                    tail_lost += 1  # success did not hold through release / retreat
                    continue
                for img, st, ac in zip(*frames, strict=True):
                    recorder.add_frame(img, st, ac, instructions[i])
                recorder.save_episode(
                    {
                        "task_family": task,
                        "seed": seeds[i],
                        "cell": "seen",
                        "oracle_success": True,
                        "success_step": int(succ_len[i]),
                        "post_success_frames": int(tail_n[i]),
                        "oracle_finished": bool(finished[i]),
                        "oracle_profile": profile,
                    }
                )
                recorded += 1
                rec_lengths.append(rec_len)
                tails.append(int(tail_n[i]))
                finished_tails += int(finished[i])
            elif not ok and allow_failed and len(failed_pool) < demos:
                failed_pool.append((seeds[i], instructions[i], *frames))

    filled = 0
    if allow_failed:
        for seed, instruction, imgs, sts, acs in failed_pool[: max(0, demos - recorded)]:
            for img, st, ac in zip(imgs, sts, acs, strict=True):
                recorder.add_frame(img, st, ac, instruction)
            recorder.save_episode(
                {
                    "task_family": task,
                    "seed": seed,
                    "cell": "seen",
                    "oracle_success": False,
                    "oracle_profile": profile,
                }
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
        "mean_demo_frames": float(np.mean(rec_lengths)) if rec_lengths else 0.0,
        "mean_post_success_frames": float(np.mean(tails)) if tails else 0.0,
        "demos_oracle_finished": finished_tails,
        "demos_rejected_tail_lost_success": tail_lost,
        "oracle_profile": profile,
        "tail": asdict(tail),
        "env_steps": env_steps,
        "num_envs": n,
        "wall_s": round(wall, 2),
    }


def print_table(stats: dict[str, dict[str, Any]]) -> None:
    head = (
        f"{'task':<10} {'episodes':>8} {'oracle':>8} {'demos':>6} {'len':>6} {'frames':>6}"
        f" {'tail':>5} {'sps':>7} {'wall_s':>7}"
    )
    print(head)
    print("-" * len(head))
    for task, s in stats.items():
        demos = f"{s['demos_recorded']}" + (
            f"+{s['failed_demos_filled']}F" if s["failed_demos_filled"] else ""
        )
        sps = s["env_steps"] / max(s["wall_s"], 1e-9)
        print(
            f"{task:<10} {s['episodes']:>8} {100 * s['oracle_success_rate']:>7.1f}% {demos:>6}"
            f" {s['mean_episode_len']:>6.1f} {s['mean_demo_frames']:>6.1f}"
            f" {s['mean_post_success_frames']:>5.1f} {sps:>7.0f} {s['wall_s']:>7.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None, help="override cfg.device"
    )
    parser.add_argument("--push", action="store_true", help="push the dataset to the Hub (private)")
    parser.add_argument(
        "--overwrite-hub",
        action="store_true",
        help=f"allow --push to a 0.1-era demo repo ({', '.join(PROTECTED_HUB_REPOS)})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="min oracle episodes per task (default: eval.episodes_per_cell)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="parallel envs (default: make_env's — cfg env.num_envs on cuda, 1 on cpu)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.preset, args.config)
    hub_repo = str(cfg.get("data", {}).get("hub_repo", ""))
    if args.push and hub_repo in PROTECTED_HUB_REPOS and not args.overwrite_hub:
        parser.error(
            f"refusing to --push over {hub_repo}: it holds a 0.1-era demo set that reported "
            "SFT/RL numbers trained on; set data.hub_repo to a new repo or pass --overwrite-hub"
        )
    if args.device:
        cfg["device"] = args.device
    num_envs = args.num_envs
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
                num_envs=num_envs,
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
