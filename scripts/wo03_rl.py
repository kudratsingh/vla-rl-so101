"""WO-03/04 entry point: GRPO post-training (ADR-0002).

    uv run scripts/wo03_rl.py --preset tiny [--seed N]      # CPU, toy env + 2-layer LoRA policy
    uv run scripts/wo03_rl.py --preset rl --estimate        # print hours / $ only, run nothing
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --probe-steps 3   # GPU probe (pod job)
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --seed N          # WO-04 run (paid)

`rl.env: toy` selects the synthetic env (src/rl/toy_env.py); anything else uses src.env.make_env
(GPU-parallel sim when `device` is cuda, serial CPU envs on cpu). The policy comes from the
registry in src/rl/policy_api.py keyed by `policy.base` (WO-04: OpenVLA-OFT, LoRA, the SFT adapter
from `policy.adapter` / `--adapter`: a local dir or a Hub `user/repo[/path][@rev]`). Real imports
are lazy so `tiny` runs without ManiSkill or VLA dependencies.
Writes runs/<id>/record.json, runs/<id>/metrics.jsonl and runs/<id>/checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, Config, add_config_args, config_hash, load_config
from src.eval import protocol
from src.rl.policy_api import build_policy
from src.rl.rollout import ObsFn
from src.rl.trainer import (
    EnvFactory,
    GRPOConfig,
    GRPOTrainer,
    eval_schedule,
    kill_switch_step,
)
from src.runs import device_info, hourly_usd

log = logging.getLogger("wo03_rl")


# --- run record: local helper, to be replaced by the shared src/runs.py (other branch) --------
def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except OSError:
        return ""


def _write_run_record(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def _work_order(cfg: Config) -> str:
    return "WO-04" if cfg["preset"] == "rl" else "WO-03"


def _new_run_record(cfg: Config, run_id: str, device: str) -> dict[str, Any]:
    dev = device_info(device)
    return {
        "run_id": run_id,
        "wo": _work_order(cfg),
        "preset": cfg["preset"],
        "config_hash": config_hash(cfg),
        "config": cfg,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "device": device,
        "gpu": dev["name"],
        "runpod_pod_id": dev["runpod_pod_id"],
        "hourly_usd": hourly_usd(dev),  # RunPod list price by GPU name; 0 off RunPod
        "start_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "end_utc": None,
        "wall_clock_h": None,
        "dollars": None,  # hourly_usd x wall clock of this process (pod setup/eval not included)
        "status": "running",
    }


def _finish_record(record: dict[str, Any], wall_s: float) -> None:
    record["end_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["wall_clock_h"] = round(wall_s / 3600.0, 4)
    record["dollars"] = round(float(record["hourly_usd"]) * wall_s / 3600.0, 4)


# --- cost estimate (cfg cost.throughput constants; WO-04 replaces them with measurements) ---
def estimate_cost(cfg: Config) -> dict[str, float]:
    """Hours and dollars for one run. The arithmetic is spelled out in configs/rl.yaml."""
    rl = GRPOConfig.from_config(cfg)
    tp = cfg["cost"]["throughput"]
    chunk = int(cfg["policy"]["action_chunk"])
    ep_len = float(cfg["env"]["max_episode_steps"]) * float(tp["mean_episode_frac"])
    trained = rl.rollout_batch * rl.samples_per_prompt
    episodes = trained * float(tp["rollout_overhead"])  # dynamic-sampling refill
    actions = episodes * ep_len
    decisions = trained * math.ceil(ep_len / chunk)
    rollout_s = actions / float(tp["rollout_actions_per_s"])
    recompute_s = (
        decisions / float(tp["forward_decisions_per_s"]) if rl.recompute_old_log_probs else 0.0
    )
    update_s = rl.ppo_epochs * decisions / float(tp["update_decisions_per_s"])
    step_s = rollout_s + recompute_s + update_s
    schedule = eval_schedule(rl)
    n = rl.rollout_envs
    eval_episodes = sum(math.ceil(ep / n) * n for ep in schedule.values())  # per task, run total
    n_evals = len(schedule)
    eval_s = len(cfg["env"]["tasks"]) * eval_episodes * ep_len
    eval_s /= float(tp["rollout_actions_per_s"])
    hours = (rl.steps * step_s + eval_s) / 3600.0
    return {
        "episodes_per_step": episodes,
        "actions_per_step": actions,
        "decisions_trained_per_step": float(decisions),
        "rollout_s_per_step": rollout_s,
        "recompute_s_per_step": recompute_s,
        "update_s_per_step": update_s,
        "step_min": step_s / 60.0,
        "evals": float(n_evals),
        "eval_episodes_per_task": float(eval_episodes),
        "eval_h": eval_s / 3600.0,
        "hours": hours,
        "dollars": hours * float(cfg["cost"]["usd_per_hour"]),
    }


def project_from_probe(cfg: Config, probe: dict[str, Any], usd_per_hour: float) -> dict[str, Any]:
    """Hours / $ for one seed from the probe's measured times (configs/rl.yaml step count)."""
    rl = GRPOConfig.from_config(cfg)
    n, tasks = rl.rollout_envs, len(cfg["env"]["tasks"])
    s_step = float(probe["s_per_step_median"])
    s_eval_batch = float(probe["eval_s_per_env_batch"])
    schedule = eval_schedule(rl)
    check = kill_switch_step(rl)

    def eval_s(upto: int) -> float:
        return sum(math.ceil(ep / n) * tasks for s, ep in schedule.items() if s <= upto) * (
            s_eval_batch
        )

    # worst case: dynamic-sampling refill hits the cap (max_rollout_rounds full rounds) every step
    round_batches = math.ceil(rl.rollout_batch / (n // rl.samples_per_prompt))
    rounds = rl.max_rollout_rounds if rl.dynamic_sampling else 1
    s_step_worst = round_batches * rounds * float(probe["rollout_s_per_env_batch"])
    s_step_worst += float(probe["update_s_median"])
    protocol_episodes = (
        len(protocol.TASKS)
        * len(protocol.CELLS)
        * len(protocol.SIM_SEEDS)
        * protocol.SIM_EPISODES_PER_CELL
    )
    out = {
        "steps": rl.steps,
        "run_h": (rl.steps * s_step + eval_s(rl.steps)) / 3600.0,
        "kill_switch_step": check,
        "kill_switch_stop_h": (check * s_step + eval_s(check)) / 3600.0,
        "worst_case_h": (rl.steps * s_step_worst + eval_s(rl.steps)) / 3600.0,
        # the final scripts/wo02_eval.py protocol eval at the probe's greedy rate (estimate)
        "protocol_eval_h": protocol_episodes * s_eval_batch / n / 3600.0,
        "usd_per_hour": usd_per_hour,
    }
    for key in ("run", "kill_switch_stop", "worst_case", "protocol_eval"):
        out[f"{key}_usd"] = out[f"{key}_h"] * usd_per_hour
    return out


def print_probe(cfg: Config, probe: dict[str, Any], proj: dict[str, Any], gpu: str) -> None:
    """The lines the pod job posts for the owner's approval (all prefixed `[wo03_rl] PROBE`)."""
    rl = GRPOConfig.from_config(cfg)
    base = probe["baseline"]
    tasks = " ".join(f"{t} {base[f'eval/success/{t}']:.3f}" for t in cfg["env"]["tasks"])
    mem = (
        f"torch peak {probe['mem/torch_peak_allocated_gb']:.1f} GB allocated / "
        f"{probe['mem/torch_peak_reserved_gb']:.1f} GB reserved; device "
        f"{probe['mem/device_used_gb']:.1f} of {probe['mem/device_total_gb']:.1f} GB in use"
        if "mem/device_used_gb" in probe
        else "n/a (no CUDA)"
    )
    ratio = probe["first_minibatch_max_abs_log_ratio"]
    speed = (
        f"s/step median {probe['s_per_step_median']:.1f} (max {probe['s_per_step_max']:.1f}; "
        f"rollout {probe['rollout_s_per_env_batch']:.1f} s per {rl.rollout_envs}-env batch, "
        f"update {probe['update_s_median']:.1f} s); rollout "
        f"{probe['rollout_episodes_per_hour']:,.0f} episodes/h; greedy eval "
        f"{probe['eval_s_per_env_batch']:.1f} s per batch; env setup {probe['env_setup_s']:.0f} s"
    )
    baseline = (
        f"SFT baseline ({rl.early_stop_eval_episodes} greedy episodes/task): {tasks} "
        f"(mean {base['eval/success']:.3f})"
    )
    checks = (
        f"checks: first-minibatch |log ratio| max {ratio:.4f} / mean "
        f"{probe['first_minibatch_mean_abs_log_ratio']:.5f} (max > 0.05 -> consider "
        "rl.recompute_old_log_probs: true), group-reset diff frac "
        f"{probe['group_reset_diff_frac']:.5f}, groups dropped {probe['frac_groups_dropped']:.0%}"
    )
    projected = (
        f"projected (estimate from {probe['steps']} probe steps, step 1 excluded): full run "
        f"{proj['steps']} steps = {proj['run_h']:.1f} h = ${proj['run_usd']:.2f} per seed on {gpu} "
        f"@ ${proj['usd_per_hour']:.2f}/h; kill-switch stop at step {proj['kill_switch_step']} = "
        f"{proj['kill_switch_stop_h']:.1f} h = ${proj['kill_switch_stop_usd']:.2f}; worst case "
        f"(refill cap every step) {proj['worst_case_h']:.1f} h = ${proj['worst_case_usd']:.2f}; "
        f"+ protocol eval {proj['protocol_eval_h']:.1f} h = ${proj['protocol_eval_usd']:.2f}"
    )
    lines = [speed, f"VRAM: {mem}", baseline, checks, projected]
    for line in lines:
        print(f"[wo03_rl] PROBE {line}", flush=True)
    if ratio > 0.05:
        print("[wo03_rl] PROBE WARNING sampler/update log-probs disagree", flush=True)


# --- wiring --------------------------------------------------------------------------------
def build_env(cfg: Config) -> tuple[EnvFactory, ObsFn, int]:
    """(env factory, obs -> policy-obs fn, max episode steps) for the configured env."""
    if cfg["rl"].get("env") == "toy":
        from src.rl.toy_env import make_toy_env_factory, policy_obs

        return make_toy_env_factory(cfg), policy_obs, int(cfg["toy"]["max_episode_steps"])

    from src.env import make_env, policy_obs  # WO-01; ManiSkill only needed here

    def factory(task: str, num_envs: int) -> Any:
        return make_env(cfg, task, num_envs=num_envs, cell="seen", render=True)

    return factory, policy_obs, int(cfg["env"]["max_episode_steps"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--device", default=None, help="override cfg device (cpu, cuda, cuda:1)")
    parser.add_argument("--seed", type=int, default=None, help="override cfg seed")
    parser.add_argument(
        "--adapter", default=None, help="override policy.adapter (SFT checkpoint dir or Hub path)"
    )
    parser.add_argument("--estimate", action="store_true", help="print hours/$ and exit")
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help="baseline eval + this many RL steps; print s/step, VRAM, projected h/$; save nothing",
    )
    args = parser.parse_args(argv)
    if args.probe_steps is not None and args.probe_steps < 1:
        parser.error("--probe-steps must be >= 1 (0 would start the full run)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.preset, args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.adapter is not None:
        cfg["policy"]["adapter"] = args.adapter
    device, seed = str(cfg["device"]), int(cfg["seed"])
    rl = GRPOConfig.from_config(cfg)  # validate before touching envs or weights
    if args.estimate:
        est = estimate_cost(cfg)
        gpu = cfg["cost"]["gpu"]
        for key, value in est.items():
            print(f"{key:>28}: {value:,.2f}")
        print(
            f"estimated: {gpu} x {est['hours']:.1f} h = ${est['dollars']:.0f} (config {config_hash(cfg)[:12]})"
        )
        return est
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device!r} has no CUDA here (use --device cpu or --preset tiny)"
        )
    seed_everything(seed)

    t_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    kind = _work_order(cfg).lower().replace("-", "") + ("probe" if args.probe_steps else "")
    run_id = f"{stamp}-{kind}-{cfg['preset']}-s{seed}-{uuid.uuid4().hex[:6]}"
    run_dir = REPO_ROOT / "runs" / run_id
    record = _new_run_record(cfg, run_id, device)
    _write_run_record(run_dir, record)
    log.info("run %s (config %s)", run_id, record["config_hash"][:12])

    env_factory, obs_fn, max_steps = build_env(cfg)
    policy = build_policy(cfg, device)
    describe = getattr(policy, "describe", None)
    record["policy"] = describe() if callable(describe) else None
    trainer = GRPOTrainer(
        rl=rl,
        policy=policy,
        env_factory=env_factory,
        obs_fn=obs_fn,
        tasks=list(cfg["env"]["tasks"]),
        max_episode_steps=max_steps,
        run_dir=run_dir,
        seed=seed,
    )
    # `timeout` (the pod job's soft deadline) sends SIGTERM: exit through the finally blocks so the
    # trainer saves an interrupted checkpoint and the run record gets its end time and dollars
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        if args.probe_steps:
            probe = trainer.probe(args.probe_steps)
            cost = cfg.get("cost", {})  # measured GPU's list price on RunPod, else the config's
            usd = float(record["hourly_usd"]) or float(cost.get("usd_per_hour", 0.0))
            gpu = str(record["gpu"]) if record["hourly_usd"] else str(cost.get("gpu", device))
            proj = project_from_probe(cfg, probe, usd)
            record["status"] = "probe"
            record["probe"] = {k: v for k, v in probe.items() if k != "rows"}
            record["projection"] = proj
            print_probe(cfg, probe, proj, gpu)
            return {**probe, "projection": proj, "run_dir": str(run_dir)}
        summary = trainer.run()
        record["status"] = "stopped_flat" if summary["stopped_flat"] else "finished"
        record["summary"] = summary
    except (KeyboardInterrupt, SystemExit):
        record["status"] = "interrupted"
        raise
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        _finish_record(record, time.perf_counter() - t_start)
        _write_run_record(run_dir, record)
    log.info(
        "done: eval success %.3f -> %.3f (best %.3f) in %d steps, %.1fs; %s",
        summary["initial_eval_success"],
        summary["final_eval_success"],
        summary["best_eval_success"],
        summary["steps_completed"],
        summary["wall_time_s"],
        run_dir,
    )
    return summary


if __name__ == "__main__":
    main()
