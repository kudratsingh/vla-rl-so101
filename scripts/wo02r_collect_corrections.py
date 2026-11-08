#!/usr/bin/env python
"""Recovery SFT, data step (docs/plans/recovery-sft.md): expert corrections from SFT failures.

    uv run scripts/wo02r_collect_corrections.py --preset tiny                  # CPU smoke, < 2 min
    uv run scripts/wo02r_collect_corrections.py --preset tiny --mode control
    uv run scripts/wo02r_collect_corrections.py --preset sim --push            # GPU pod (approved)
    uv run scripts/wo02r_collect_corrections.py --preset sim --mode control --push

--mode corrections (default): roll the SFT adapter out greedily on RECOVERY seeds (seen cell only,
src/eval/seed_ranges.py), classify each failure and pick a branch state before its decisive event
(src/data/recovery.py), restore that state (src/env/state.py), let the scripted oracle finish
from there, and keep the branches the env's own success check confirms. Recorded with the
WO-01 DemoRecorder into a SEPARATE dataset; sidecar kind=correction plus failure mode, event and
branch step. Frames start at the restore point: the SFT policy's own actions are never recorded.

--mode control: fresh oracle demos from reset on CONTROL seeds (kind=control_demo), for the
matched control (configs/sft_a5_control.yaml). No policy is loaded.

Default config: configs/sft_tiny.yaml for --preset tiny (TINY_DEFAULTS below), else
configs/sft_a5_recovery.yaml (its `recovery:` block). Writes runs/<id>/record.json.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.data.recorder import DemoRecorder, dataset_location
from src.data.recovery import (
    Branch,
    choose_branch,
    classify,
    episode_series,
    interleave,
    yield_table,
)
from src.eval import seed_ranges as SR

DEFAULTS: dict[str, Any] = {
    "hub_repo": None,  # corrections dataset (None: local/vla-rl-so101-correction-<preset>)
    "control_hub_repo": None,  # matched-control dataset
    "rollouts_per_task": 192,  # SFT rollouts on recovery seeds per task (rounded up to batches)
    "rollout_max_steps": None,  # None: env.max_episode_steps (= the protocol horizon, 150)
    "cap_per_mode": 24,  # branches tried per (task, failure mode)
    "branch_back": 1,  # chunk boundaries before the decisive event
    "oracle_max_steps": 150,
    "control_episodes_per_task": 60,
    "restore_tol": 1.0e-4,  # max |restored - saved| sim-state entry
    "allow_reset_branch": False,
}
TINY_DEFAULTS: dict[str, Any] = {
    **DEFAULTS,
    "rollouts_per_task": 2,
    "rollout_max_steps": 24,  # 3 chunks of the random tiny policy
    "cap_per_mode": 2,
    # tiny proves plumbing only: the random policy fails in the first chunk, so allow branching
    # from the reset state (never in sim: there an oracle run from reset is a control demo)
    "allow_reset_branch": True,
    "control_episodes_per_task": 2,
}


def default_config(preset: str) -> Path:
    return REPO_ROOT / "configs" / ("sft_tiny.yaml" if preset == "tiny" else "sft_a5_recovery.yaml")


def settings(cfg: Config) -> dict[str, Any]:
    base = TINY_DEFAULTS if cfg["preset"] == "tiny" else DEFAULTS
    out = {**base, **(cfg.get("recovery") or {})}
    unknown = set(out) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unknown recovery settings: {sorted(unknown)}")
    return out


def output_location(cfg: Config, kind: str, hub_repo: str | None) -> tuple[str, Path]:
    repo = hub_repo or f"local/vla-rl-so101-{kind}-{cfg.get('preset', 'dev')}"
    repo_id, root = dataset_location({**cfg, "data": {**cfg.get("data", {}), "hub_repo": repo}})
    return repo_id, root


class LazyRecorder:
    """Creates the dataset at the first kept episode, sized from the env's own images."""

    def __init__(self, repo_id: str, root: Path) -> None:
        self.repo_id, self.root = repo_id, root
        self.rec: DemoRecorder | None = None

    def get(self, image_hw: tuple[int, int]) -> DemoRecorder:
        if self.rec is None:
            self.rec = DemoRecorder(self.repo_id, self.root, image_hw=image_hw, overwrite=True)
        elif self.rec.image_hw != image_hw:
            raise ValueError(f"image size changed: {self.rec.image_hw} -> {image_hw}")
        return self.rec


def oracle_episodes(
    env: Any, oracle: Any, obs: Any, obs_fn: Callable[[Any], Any], max_steps: int
) -> tuple[torch.Tensor, torch.Tensor, list[list[tuple[Any, Any, Any]]]]:
    """Run the (already reset) oracle until every env succeeds or times out.

    Returns (success (N,), length (N,), per-env frames [(image, state, action)] up to length):
    state = qpos at t, action = the oracle command at t (the WO-01 demo convention)."""
    n = int(env.unwrapped.num_envs)
    frames: list[list[tuple[Any, Any, Any]]] = [[] for _ in range(n)]
    done = torch.zeros(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    length = torch.zeros(n, dtype=torch.long)
    for _ in range(max_steps):
        po = obs_fn(obs)
        action = oracle.act(obs)
        img = po["image"].detach().cpu().numpy()
        st = po["state"].detach().cpu().float().numpy()
        ac = torch.as_tensor(action).detach().cpu().float().numpy()
        for i in torch.nonzero(~done).flatten().tolist():
            frames[i].append((img[i], st[i], ac[i]))
        obs, _, term, trunc, info = env.step(action)
        length += (~done).long()
        succ = torch.as_tensor(info["success"]).detach().cpu().bool().reshape(n)
        success |= succ & ~done
        ended = torch.as_tensor(term).cpu().bool() | torch.as_tensor(trunc).cpu().bool()
        done |= succ | ended.reshape(-1).expand(n)
        if bool(done.all()):
            break
    return success, length, frames


def save(
    recorder: LazyRecorder, frames: list[tuple[Any, Any, Any]], instruction: str, meta: Any
) -> None:
    img0 = np.asarray(frames[0][0])
    rec = recorder.get((int(img0.shape[0]), int(img0.shape[1])))
    for img, st, ac in frames:
        rec.add_frame(img, st, ac, instruction)
    rec.save_episode(meta)


def find_branches(
    task: str,
    policy: Any,
    env: Any,
    env_mod: Any,
    state_mod: Any,
    seeds: list[int],
    max_steps: int,
    back: int,
    allow_reset: bool = False,
) -> tuple[list[Branch], list[Any], dict[str, Any]]:
    """SFT rollouts on `seeds` -> candidate branches, the traces (snapshots), rollout stats."""
    from src.eval.sim_eval import run_batch_traced

    n = int(env.unwrapped.num_envs)
    branches: list[Branch] = []
    traces: list[Any] = []
    modes: Counter[str] = Counter()
    unrecoverable: Counter[str] = Counter()
    reset_only: Counter[str] = Counter()  # recoverable only from reset: a control demo, dropped
    successes = 0
    for b, start in enumerate(range(0, len(seeds), n)):
        batch = seeds[start : start + n]
        ok, length, trace = run_batch_traced(
            policy.act_batch,
            env,
            env_mod.policy_obs,
            batch,
            max_steps,
            snapshot_fn=state_mod.snapshot,
        )
        traces.append(trace)
        steps = [bd.step for bd in trace.boundaries]
        for row, seed in enumerate(batch):
            if bool(ok[row]):
                successes += 1
                continue
            ser = episode_series(trace.steps, row)
            ep_len = int(length[row])
            mode, event = classify(task, ser, ep_len, max_steps)
            modes[mode] += 1
            k = choose_branch(task, ser, steps, event, ep_len, back, allow_reset)
            if k is None:
                if choose_branch(task, ser, steps, event, ep_len, back, allow_reset=True) is None:
                    unrecoverable[mode] += 1
                else:
                    reset_only[mode] += 1
                continue
            grasped = ser.get("is_grasped")
            branches.append(
                Branch(
                    task=task,
                    seed=seed,
                    mode=mode,
                    event_step=event,
                    boundary=k,
                    step=steps[k],
                    in_hand=bool(grasped[steps[k]]) if grasped is not None else False,
                    grip_cmd=float(trace.boundaries[k].last_action[row, 5]),
                    batch=b,
                    row=row,
                )
            )
    stats = {
        "rollouts": len(seeds),
        "sft_success_rate": successes / max(1, len(seeds)),
        "failure_modes": dict(modes),
        "unrecoverable": dict(unrecoverable),
        "reset_branch": dict(reset_only),
    }
    return branches, traces, stats


def run_corrections(
    task: str,
    cfg: Config,
    rc: dict[str, Any],
    policy: Any,
    env: Any,
    recorder: LazyRecorder,
    *,
    keep_failed: bool,
    adapter: str,
) -> dict[str, Any]:
    from src.data.oracle import make_oracle

    env_mod: Any = importlib.import_module("src.env")
    state_mod: Any = importlib.import_module("src.env.state")
    n = int(env.unwrapped.num_envs)
    n_roll = math.ceil(int(rc["rollouts_per_task"]) / n) * n
    seeds = SR.recovery_seeds(task, n_roll)
    SR.assert_seeds_in(seeds, "recovery", SR.standard_ranges())
    max_steps = int(rc["rollout_max_steps"] or cfg["env"]["max_episode_steps"])
    candidates, traces, stats = find_branches(
        task,
        policy,
        env,
        env_mod,
        state_mod,
        seeds,
        max_steps,
        int(rc["branch_back"]),
        bool(rc["allow_reset_branch"]),
    )
    chosen = interleave(candidates, int(rc["cap_per_mode"]))
    tried: list[Branch] = []
    oks: list[bool] = []
    frames_kept = kept = 0
    max_err = 0.0
    for start in range(0, len(chosen), n):
        group = chosen[start : start + n]
        padded = group + [group[-1]] * (n - len(group))
        rows = [traces[g.batch].boundaries[g.boundary].state.row(g.row) for g in padded]
        snap = state_mod.EnvSnapshot.stack(rows)
        obs = state_mod.restore(env, [g.seed for g in padded], snap)
        err = state_mod.state_error(state_mod.snapshot(env), snap)
        max_err = max(max_err, err)
        if err > float(rc["restore_tol"]):
            raise RuntimeError(f"{task}: restored state differs from the snapshot by {err:.2e}")
        instructions = list(env.unwrapped.get_instructions())
        oracle = make_oracle(task, env)
        oracle.reset(
            obs,
            start_phase="infer",
            grasped=torch.tensor([g.in_hand for g in padded]),
            grip_cmd=torch.tensor([g.grip_cmd for g in padded]),
        )
        success, length, frames = oracle_episodes(
            env, oracle, obs, env_mod.policy_obs, int(rc["oracle_max_steps"])
        )
        for i, g in enumerate(group):
            ok = bool(success[i])
            tried.append(g)
            oks.append(ok)
            if not (ok or keep_failed):
                continue
            ep = frames[i][: int(length[i])]
            meta = {
                "task_family": task,
                "seed": g.seed,
                "cell": "seen",
                "oracle_success": ok,
                "kind": "correction",
                "failure_mode": g.mode,
                "event_step": g.event_step,
                "branch_step": g.step,
                "in_hand": g.in_hand,
                "source_policy": adapter,
                "backend": snap.backend,
            }
            save(recorder, ep, instructions[i], meta)
            kept += int(ok)
            frames_kept += len(ep) if ok else 0
    return {
        **stats,
        "candidates": len(candidates),
        "tried": len(tried),
        "corrections": kept,
        "correction_frames": frames_kept,
        "yield_by_mode": yield_table(tried, oks),
        "restore_max_err": max_err,
        "seeds": [seeds[0], seeds[-1]],
    }


def run_control(
    task: str, rc: dict[str, Any], env: Any, recorder: LazyRecorder, *, keep_failed: bool
) -> dict[str, Any]:
    from src.data.oracle import make_oracle

    env_mod: Any = importlib.import_module("src.env")
    n = int(env.unwrapped.num_envs)
    want = int(rc["control_episodes_per_task"])
    pool = SR.control_seeds(task, math.ceil(max(3 * want, n) / n) * n)
    SR.assert_seeds_in(pool, "control", SR.standard_ranges())
    kept = run = frames_kept = 0
    for start in range(0, len(pool), n):
        if kept >= want:
            break
        batch = pool[start : start + n]
        obs, _ = env.reset(seed=batch)
        instructions = list(env.unwrapped.get_instructions())
        oracle = make_oracle(task, env)
        oracle.reset(obs)
        success, length, frames = oracle_episodes(
            env, oracle, obs, env_mod.policy_obs, int(rc["oracle_max_steps"])
        )
        for i, seed in enumerate(batch):
            run += 1
            ok = bool(success[i])
            if kept >= want or not (ok or keep_failed):
                continue
            meta = {
                "task_family": task,
                "seed": seed,
                "cell": "seen",
                "oracle_success": ok,
                "kind": "control_demo",
            }
            ep = frames[i][: int(length[i])]
            save(recorder, ep, instructions[i], meta)
            kept += 1
            frames_kept += len(ep)
    return {"episodes_run": run, "control_demos": kept, "control_frames": frames_kept}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--mode", choices=["corrections", "control"], default="corrections")
    parser.add_argument("--adapter", default=None, help="SFT checkpoint (default policy.adapter)")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--sim-device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--push", action="store_true", help="push the dataset to the Hub (private)")
    args = parser.parse_args(argv)
    cfg = load_config(args.preset, args.config or default_config(args.preset))
    rc = settings(cfg)
    device = str(cfg.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"preset {args.preset!r} needs a CUDA GPU (use --preset tiny on CPU)")
    SR.check_disjoint(SR.standard_ranges())
    env_mod: Any = importlib.import_module("src.env")
    runs: Any = importlib.import_module("src.runs")
    sim_cfg = {**cfg, "device": args.sim_device} if args.sim_device else cfg
    num_envs = int(args.num_envs or cfg["env"].get("num_envs", 16))
    # tiny proves plumbing only: keep (flagged) failed oracle branches so the dataset is not empty
    keep_failed = cfg["preset"] == "tiny"
    kind = "correction" if args.mode == "corrections" else "control"
    repo_id, root = output_location(
        cfg, kind, rc["hub_repo"] if args.mode == "corrections" else rc["control_hub_repo"]
    )
    recorder = LazyRecorder(repo_id, root)
    policy = None
    adapter = str(args.adapter or cfg["policy"].get("adapter") or "tiny-random")
    if args.mode == "corrections":
        from src.policy.factory import make_policy

        if adapter == "tiny-random" and not cfg["policy"].get("tiny_random", False):
            raise ValueError("--adapter (the SFT checkpoint to correct) is required")
        policy = make_policy(cfg, device, adapter=None if adapter == "tiny-random" else adapter)
        policy.eval()

    with runs.Run(work_order="WO-02R", script=__file__, cfg=cfg, device=device) as run:
        stats: dict[str, Any] = {}
        for task in cfg["env"]["tasks"]:
            env = env_mod.make_env(sim_cfg, task, num_envs=num_envs, cell="seen")
            try:
                if policy is not None:
                    stats[task] = run_corrections(
                        task,
                        cfg,
                        rc,
                        policy,
                        env,
                        recorder,
                        keep_failed=keep_failed,
                        adapter=adapter,
                    )
                else:
                    stats[task] = run_control(task, rc, env, recorder, keep_failed=keep_failed)
            finally:
                env.close()
            run.log_metrics(task=task, **{k: v for k, v in stats[task].items() if k != "seeds"})
            print(f"[wo02r] {task}: {stats[task]}", flush=True)
        dataset: dict[str, Any] = {"repo_id": repo_id, "root": os.path.relpath(root, REPO_ROOT)}
        if recorder.rec is not None:
            recorder.rec.finalize()
            dataset.update(episodes=recorder.rec.num_episodes, frames=recorder.rec.num_frames)
            if args.push:
                dataset["hub_url"] = recorder.rec.push_to_hub(private=True)
        else:
            dataset.update(episodes=0, frames=0)
        run.results.update(mode=args.mode, settings=rc, tasks=stats, dataset=dataset)
        run.results["source_policy"] = adapter if policy is not None else None
        print(
            f"[wo02r] dataset {repo_id}: {dataset['episodes']} episodes, {dataset['frames']} frames"
        )
        return dict(run.results, run_id=run.id, run_dir=str(run.dir))


if __name__ == "__main__":
    main()
