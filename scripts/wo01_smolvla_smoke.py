#!/usr/bin/env python
"""WO-01 deliverable 5: few-shot SmolVLA fine-tune on the local demo dataset, then sim eval.

Proves the dataset format and the train/eval plumbing; the benchmark is only >0% success on at
least one task. Training is lerobot 0.4.4's own `lerobot_train.train` (in-process, so the
torchcodec compat shim applies), starting from `lerobot/smolvla_base` (override: config key
`smoke.base` or `--policy-base`). Eval runs each task on the `seen` cell with seeds disjoint from
the demo seeds, success = env success check at any step.

    uv run scripts/wo01_build_env.py --preset tiny && \\
      uv run scripts/wo01_smolvla_smoke.py --preset tiny --device cpu       # < 2 min, plumbing only
    uv run scripts/wo01_smolvla_smoke.py --preset sim --device mps          # full smoke on the Mac
"""

from __future__ import annotations

import argparse
import math
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wo01_build_env import EnvApi, env_api

from src.config import Config, add_config_args, load_config
from src.data.recorder import (
    IMAGE_KEY,
    STATE_KEY,
    VIDEO_BACKEND,
    dataset_location,
    install_torchcodec_compat,
)
from src.runs import Run

DEFAULT_BASE = "lerobot/smolvla_base"
# smolvla_base was pretrained with camera1..3; our single front camera feeds camera1
RENAME_MAP = {IMAGE_KEY: "observation.images.camera1"}
EVAL_SEED_BASE = 0  # eval seeds cfg.seed * 10_000 + k; demo seeds start at 1_000_000
RTX4090_USD_PER_H = 0.89
# Rough, unmeasured: training-step speed-up of an RTX 4090 over the measuring device.
RTX4090_SPEEDUP = {"mps": 10.0, "cpu": 50.0, "cuda": 1.0}


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def train(
    cfg: Config,
    run: Run,
    *,
    base: str,
    steps: int,
    batch_size: int,
    device: str,
) -> tuple[Path, list[float]]:
    """Fine-tune `base` with lerobot's trainer; returns (checkpoint dir, per-step seconds)."""
    install_torchcodec_compat()
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train

    repo_id, root = dataset_location(cfg)
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"no dataset at {root}; run scripts/wo01_build_env.py first")
    policy_cfg = PreTrainedConfig.from_pretrained(base)
    policy_cfg.pretrained_path = Path(base)
    policy_cfg.device = device
    policy_cfg.push_to_hub = False
    out = run.dir / "train"
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=repo_id, root=str(root), video_backend=VIDEO_BACKEND),
        policy=policy_cfg,
        output_dir=out,
        job_name=f"wo01_smolvla_{cfg['preset']}",
        seed=int(cfg.get("seed", 0)),
        # macOS DataLoader workers are spawned and would not inherit the torchcodec shim
        num_workers=0 if sys.platform == "darwin" else 4,
        batch_size=batch_size,
        steps=steps,
        eval_freq=0,
        log_freq=max(1, steps // 10),
        save_checkpoint=True,
        save_freq=steps,
        wandb=WandBConfig(enable=False),
        rename_map=dict(RENAME_MAP),
    )

    step_s: list[float] = []
    original = lerobot_train.update_policy
    original_make_policy = lerobot_train.make_policy

    def make_policy(*args: Any, **kwargs: Any) -> Any:
        return _cpu_fp32(original_make_policy(*args, **kwargs), device)

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        _sync(device)
        t0 = time.perf_counter()
        tracker, output = original(*args, **kwargs)
        _sync(device)
        step_s.append(time.perf_counter() - t0)
        loss = getattr(getattr(tracker, "loss", None), "val", None)
        run.log_metrics(step=len(step_s), loss=loss, step_s=round(step_s[-1], 4))
        return tracker, output

    lerobot_train.update_policy = timed_update
    lerobot_train.make_policy = make_policy
    try:
        lerobot_train.train(train_cfg)
    finally:
        lerobot_train.update_policy = original
        lerobot_train.make_policy = original_make_policy
    return out / "checkpoints" / "last" / "pretrained_model", step_s


def _cpu_fp32(policy: Any, device: str) -> Any:
    """SmolVLA loads its VLM in bf16; on CPU that is ~30x slower than fp32 (28 s vs 0.9 s/step on
    an M3 Pro, batch 2), so CPU runs use fp32. MPS/CUDA keep lerobot's bf16."""
    if device == "cpu":
        policy.to(torch.float32)
    return policy


def load_policy(ckpt: Path, device: str, n_action_steps: int) -> tuple[Any, Any, Any]:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    pcfg = PreTrainedConfig.from_pretrained(ckpt)
    pcfg.device = device
    pcfg.pretrained_path = ckpt
    pcfg.n_action_steps = min(n_action_steps, int(pcfg.chunk_size))
    policy = get_policy_class(pcfg.type).from_pretrained(ckpt, config=pcfg)
    _cpu_fp32(policy.to(device), device).eval()
    pre, post = make_pre_post_processors(
        pcfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


@torch.inference_mode()
def evaluate(
    cfg: Config,
    api: EnvApi,
    policy: Any,
    pre: Any,
    post: Any,
    *,
    episodes: int,
    device: str,
) -> dict[str, dict[str, Any]]:
    make_env, policy_obs, _ = api
    max_steps = int(cfg["env"]["max_episode_steps"])
    seed0 = EVAL_SEED_BASE + int(cfg.get("seed", 0)) * 10_000
    out: dict[str, dict[str, Any]] = {}
    for task in cfg["env"]["tasks"]:
        t0 = time.monotonic()
        env = make_env(cfg, task, cell="seen", render=True)
        n = int(getattr(env.unwrapped, "num_envs", 1))
        n_run = n_success = 0
        chunk_s: list[float] = []
        while n_run < episodes:
            seeds = [seed0 + n_run + i for i in range(n)]
            obs, _ = env.reset(seed=seeds if n > 1 else seeds[0])
            instructions = list(env.unwrapped.get_instructions())
            policy.reset()
            success = torch.zeros(n, dtype=torch.bool)
            for _ in range(max_steps):
                po = policy_obs(obs)
                batch = {
                    IMAGE_KEY: po["image"].permute(0, 3, 1, 2).float().div(255.0),
                    STATE_KEY: po["state"].float(),
                    "task": instructions,
                }
                new_chunk = len(policy._queues["action"]) == 0
                _sync(device)
                t1 = time.perf_counter()
                action = post(policy.select_action(pre(batch)))
                _sync(device)
                if new_chunk:
                    chunk_s.append(time.perf_counter() - t1)
                obs, _, _, _, info = env.step(action.float().cpu())
                success |= torch.as_tensor(info["success"]).cpu().bool().reshape(n)
            take = min(n, episodes - n_run)
            n_success += int(success[:take].sum())
            n_run += take
        env.close()
        out[task] = {
            "episodes": n_run,
            "successes": n_success,
            "success_rate": n_success / max(1, n_run),
            # first chunk includes kernel warm-up
            "chunk_inference_s": statistics.median(chunk_s[1:] or chunk_s) if chunk_s else None,
            "wall_s": round(time.monotonic() - t0, 2),
        }
    return out


def estimate(
    measured_device: str, step_s: float, measured_batch: int, chunk_s: float | None
) -> dict[str, Any]:
    """Wall-clock / $ for the full smoke (`sim` preset) on this Mac's MPS and on an RTX 4090."""
    sim = load_config("sim")
    sm = sim.get("smoke", {})
    steps = int(sm.get("steps", sim["sft"]["steps"]))
    batch = int(sm.get("batch_size", sim["sft"]["batch_size"]))
    episodes = int(sm.get("eval_episodes", 10))
    chunks = (
        len(sim["env"]["tasks"])
        * episodes
        * math.ceil(int(sim["env"]["max_episode_steps"]) / int(sm.get("n_action_steps", 10)))
    )
    train_h = steps * step_s * batch / measured_batch / 3600.0
    eval_h = chunks * (chunk_s or 0.0) / 3600.0
    local_h = train_h + eval_h
    gpu_h = local_h / RTX4090_SPEEDUP.get(measured_device, 1.0)
    return {
        "full_smoke": {"steps": steps, "batch_size": batch, "eval_episodes_per_task": episodes},
        "measured_device": measured_device,
        "measured_step_s": step_s,
        "measured_batch_size": measured_batch,
        "local_h": round(local_h, 3),
        "rtx4090_h": round(gpu_h, 3),
        "rtx4090_usd": round(gpu_h * RTX4090_USD_PER_H, 2),
        "rtx4090_speedup_assumed": RTX4090_SPEEDUP.get(measured_device, 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument(
        "--steps", type=int, default=None, help="training steps (default: sft.steps)"
    )
    parser.add_argument("--batch-size", type=int, default=None, help="default: sft.batch_size")
    parser.add_argument("--eval-episodes", type=int, default=None, help="per task; 0 skips eval")
    parser.add_argument(
        "--policy-base", default=None, help=f"default: smoke.base or {DEFAULT_BASE}"
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="keep the tiny preset's checkpoint (deleted by default)",
    )
    args = parser.parse_args()

    cfg = load_config(args.preset, args.config)
    sm = cfg.get("smoke", {})
    device = args.device or str(cfg.get("device", "cpu"))  # policy device
    # cfg.device is the sim backend for make_env, which takes only cpu | cuda
    cfg["device"] = "cuda" if device == "cuda" else "cpu"
    base = args.policy_base or sm.get("base") or DEFAULT_BASE
    steps = int(args.steps or sm.get("steps") or cfg["sft"]["steps"])
    batch_size = int(args.batch_size or sm.get("batch_size") or cfg["sft"]["batch_size"])
    episodes = args.eval_episodes
    if episodes is None:
        episodes = int(sm.get("eval_episodes", 1 if cfg["preset"] == "tiny" else 10))
    n_action_steps = int(sm.get("n_action_steps", 10))
    cfg["smoke"] = {
        "device": device,
        "base": base,
        "steps": steps,
        "batch_size": batch_size,
        "eval_episodes": episodes,
        "n_action_steps": n_action_steps,
    }

    with Run(work_order="WO-01", script=__file__, cfg=cfg, device=device) as run:
        t0 = time.monotonic()
        ckpt, step_times = train(
            cfg, run, base=base, steps=steps, batch_size=batch_size, device=device
        )
        train_s = time.monotonic() - t0
        timed = step_times[1:] or step_times  # first step includes lazy init / kernel warm-up
        step_s = statistics.median(timed)
        run.results.update(
            checkpoint=str(ckpt.relative_to(REPO_ROOT)),
            train_wall_s=round(train_s, 1),
            step_s_median=round(step_s, 4),
        )
        results: dict[str, dict[str, Any]] = {}
        if episodes > 0:
            policy, pre, post = load_policy(ckpt, device, n_action_steps)
            results = evaluate(cfg, env_api(), policy, pre, post, episodes=episodes, device=device)
            run.results["eval"] = results
            del policy, pre, post
        chunk = [r["chunk_inference_s"] for r in results.values() if r["chunk_inference_s"]]
        est = estimate(device, step_s, batch_size, statistics.median(chunk) if chunk else None)
        run.results["estimate"] = est
        if not args.keep_checkpoint and cfg["preset"] == "tiny":
            shutil.rmtree(run.dir / "train" / "checkpoints")  # ~1-2 GB of plumbing-only weights
            run.results["checkpoint"] = None

    print(f"\nrun {run.id}  preset={cfg['preset']}  device={device}")
    print(f"train: {steps} steps x batch {batch_size} in {train_s:.1f}s; median step {step_s:.3f}s")
    if results:
        print(f"{'task':<10} {'episodes':>8} {'success':>8} {'chunk_s':>8} {'wall_s':>7}")
        for task, r in results.items():
            cs = f"{r['chunk_inference_s']:.2f}" if r["chunk_inference_s"] else "-"
            print(
                f"{task:<10} {r['episodes']:>8} {100 * r['success_rate']:>7.1f}% {cs:>8} {r['wall_s']:>7.1f}"
            )
    fs = est["full_smoke"]
    print(
        f"full smoke estimate (sim preset: {fs['steps']} steps x batch {fs['batch_size']},"
        f" {fs['eval_episodes_per_task']} eval eps/task), from {device} step time:"
    )
    print(f"  this machine ({device}): {est['local_h']:.2f} h")
    print(
        f"  RTX 4090 @ ${RTX4090_USD_PER_H}/h: {est['rtx4090_h']:.2f} h = ${est['rtx4090_usd']:.2f}"
        f"  (assumes {est['rtx4090_speedup_assumed']:.0f}x {device}; rough)"
    )
    print(f"wall {run.record['wall_clock_s']}s  record: {run.record_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
