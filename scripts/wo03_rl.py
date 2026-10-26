"""WO-03/04 entry point: GRPO post-training (ADR-0002).

    uv run scripts/wo03_rl.py --preset tiny [--seed N]      # CPU, toy env + 2-layer LoRA policy
    uv run scripts/wo03_rl.py --preset rl                   # A100: VLA + ManiSkill (WO-04, paid)

`rl.env: toy` selects the synthetic env (src/rl/toy_env.py); anything else uses src.env.make_env.
The policy comes from the registry in src/rl/policy_api.py keyed by `policy.base`. Both real
imports are lazy so `tiny` runs without ManiSkill or VLA dependencies.
Writes runs/<id>/record.json, runs/<id>/metrics.jsonl and runs/<id>/checkpoints/.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, Config, add_config_args, config_hash, load_config
from src.rl.policy_api import build_policy
from src.rl.rollout import ObsFn
from src.rl.trainer import EnvFactory, GRPOConfig, GRPOTrainer

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


def _new_run_record(cfg: Config, run_id: str, device: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "wo": "WO-03",
        "preset": cfg["preset"],
        "config_hash": config_hash(cfg),
        "config": cfg,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "device": device,
        "start_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "end_utc": None,
        "dollars": 0.0 if device == "cpu" else None,  # GPU runs: fill from RunPod billing
        "status": "running",
    }


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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.preset, args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    device, seed = str(cfg["device"]), int(cfg["seed"])
    rl = GRPOConfig.from_config(cfg)  # validate before touching envs or weights
    seed_everything(seed)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-wo03-{cfg['preset']}-s{seed}-{uuid.uuid4().hex[:6]}"
    run_dir = REPO_ROOT / "runs" / run_id
    record = _new_run_record(cfg, run_id, device)
    _write_run_record(run_dir, record)
    log.info("run %s (config %s)", run_id, record["config_hash"][:12])

    env_factory, obs_fn, max_steps = build_env(cfg)
    policy = build_policy(cfg, device)
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
    try:
        summary = trainer.run()
        record["status"] = "stopped_flat" if summary["stopped_flat"] else "finished"
        record["summary"] = summary
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["end_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
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
