"""WO-02 few-shot SFT of OpenVLA-OFT on the WO-01 demos (all three tasks, one checkpoint).

    uv run scripts/wo02_sft.py --preset tiny                    # CPU smoke: tiny random OFT
    uv run scripts/wo02_sft.py --preset tiny --synthetic        # ... without the WO-01 dataset
    uv run scripts/wo02_sft.py --preset sim --probe-steps 20    # GPU memory/speed probe (~$0.10)
    uv run scripts/wo02_sft.py --preset sim                     # full run (owner-approved only)
    uv run scripts/wo02_sft.py --preset sim --config configs/sft_a5_recovery.yaml   # recovery SFT

`data.extra_datasets` (configs/sft_a5_*.yaml) mixes capped correction / control datasets into the
demos (src/data/oft_dataset.py); the bounds coverage check then runs per dataset as well.

Default config: configs/sft_tiny.yaml for --preset tiny, else configs/sft.yaml. Writes
runs/<id>/record.json (+ metrics.jsonl, checkpoints/step_*/: PEFT adapter + norm_stats.json).
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

from src.config import REPO_ROOT, Config, add_config_args, load_config
from src.data.oft_dataset import (
    ChunkedDemoDataset,
    ConcatFrames,
    all_actions,
    load_sft_frames,
    synthetic_frames,
)
from src.policy.norm_bounds import demo_coverage
from src.policy.openvla_oft import build_policy
from src.policy.sft import SFTSettings, estimate_cost, train_sft

SYNTHETIC_INSTRUCTIONS = {
    "PickPlace": "pick up the red cube and place it in the bowl",
    "Push": "push the red cube to the target",
    "Stack": "stack the red cube on the blue cube",
}


def default_config(preset: str) -> Path:
    return REPO_ROOT / "configs" / ("sft_tiny.yaml" if preset == "tiny" else "sft.yaml")


def pick_device(cfg: Config) -> str:
    want = str(cfg.get("device", "cpu"))
    if want.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"preset {cfg['preset']!r} needs a CUDA GPU (use --preset tiny on CPU)")
    return want


def load_data(
    cfg: Config, chunk: int, synthetic: bool
) -> tuple[ChunkedDemoDataset, dict[str, Any]]:
    if synthetic or cfg["data"].get("synthetic", False):
        tasks = list(cfg["env"]["tasks"])
        n = int(cfg["data"]["demos_per_task"])
        frames = synthetic_frames(
            [SYNTHETIC_INSTRUCTIONS[t] for t in tasks],
            episodes_per_instruction=n,
            frames_per_episode=int(cfg["data"].get("synthetic_frames_per_episode", 20)),
            chunk_len=chunk,
            seed=int(cfg.get("seed", 0)),
        )
        return ChunkedDemoDataset(frames, chunk_len=chunk), {
            "synthetic": True,
            "num_frames": len(frames),
        }
    return load_sft_frames(cfg, chunk)


def coverage_by_part(bounds: Any, frames: Any) -> dict[str, dict[str, Any]]:
    """Bounds coverage per dataset of a mixed SFT set (demos, corrections, control demos)."""
    if not isinstance(frames, ConcatFrames):
        return {"demo": demo_coverage(bounds, all_actions(frames))}
    out: dict[str, dict[str, Any]] = {}
    for i, (name, part) in enumerate(zip(frames.names, frames.parts, strict=True)):
        out[name if name not in out else f"{name}_{i}"] = demo_coverage(bounds, all_actions(part))
    return out


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument(
        "--probe-steps", type=int, default=None, help="measure s/step + memory, save nothing"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="random frames instead of the WO-01 demos"
    )
    parser.add_argument("--max-demo-clip-frac", type=float, default=0.01)
    args = parser.parse_args(argv)
    cfg = load_config(args.preset, args.config or default_config(args.preset))
    device = pick_device(cfg)
    settings = SFTSettings.from_cfg(cfg, device)
    chunk = int(cfg["policy"]["action_chunk"])

    dataset, data_info = load_data(cfg, chunk, args.synthetic)
    policy = build_policy(cfg, device, grad_checkpointing=settings.grad_checkpointing)
    coverage = demo_coverage(policy.pre.action_bounds, all_actions(dataset.frames))
    if coverage["frac_outside"] > args.max_demo_clip_frac:
        raise ValueError(f"demo actions outside the normalisation bounds: {coverage}")
    by_part = coverage_by_part(policy.pre.action_bounds, dataset.frames)
    clipped = {k: c for k, c in by_part.items() if c["frac_outside"] > args.max_demo_clip_frac}
    if clipped:  # e.g. corrections from displaced arm states: clipping would change the labels
        raise ValueError(f"actions outside the normalisation bounds, by dataset: {clipped}")
    est = estimate_cost(cfg)
    print(
        f"[wo02_sft] {data_info.get('num_frames')} frames, device={device}, "
        f"micro_batch={settings.micro_batch_size}x{settings.accum_steps}, "
        f"grad_ckpt={settings.grad_checkpointing}, trainable="
        f"{sum(p.numel() for p in policy.trainable_parameters()) / 1e6:.1f}M; "
        f"estimate {est['hours']} h, ${est['usd']}",
        flush=True,
    )
    runs: Any = importlib.import_module("src.runs")  # WO-01
    work = "WO-02-probe" if args.probe_steps else "WO-02"
    with runs.Run(
        work_order=work, script=__file__, cfg=cfg, device=device, estimated_usd=est["usd"]
    ) as run:
        run.results.update(
            data=data_info,
            demo_coverage=coverage,
            coverage_by_dataset=by_part,
            bounds={
                "low": list(policy.pre.action_bounds.low),
                "high": list(policy.pre.action_bounds.high),
            },
            policy=policy.describe(),
            estimate=est,
        )
        results = train_sft(
            policy,
            dataset,
            settings,
            sink=run,
            out_dir=None if args.probe_steps else run.dir / "checkpoints",
            device=torch.device(device),
            probe_steps=args.probe_steps,
        )
        run.results.update(results)
        if args.probe_steps:
            run.results["extrapolated"] = estimate_cost(cfg, results["s_per_step_median"])
        print(
            json.dumps(
                {
                    "run_id": run.id,
                    **{k: run.results[k] for k in ("steps_run", "s_per_step_median", "max_mem_gb")},
                }
            )
        )
        if args.probe_steps:
            print(
                f"[wo02_sft] full run ({cfg['sft']['steps']} steps) extrapolated: {run.results['extrapolated']}"
            )
        return dict(run.results, run_id=run.id, run_dir=str(run.dir))


if __name__ == "__main__":
    main()
