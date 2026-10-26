"""Few-shot SFT loop (WO-02): AdamW + MultiStepLR, gradient accumulation, bf16 autocast (in the
policy), gradient checkpointing (Llama layers + ViT blocks), JSONL metrics, a checkpoint every N
steps, optional early stop on the normalised L1, and a short `probe` mode that measures s/step and
peak memory for the cost estimate. Hyper-parameter defaults follow OFT's LoRA recipe.
"""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.oft_dataset import collate_sft
from src.policy.base import SFTBatch, VLAPolicy

LARGE_GPU_GB = 40.0  # A100-80GB: no grad checkpointing needed at micro-batch 8 (~41 GB est.)


class MetricsSink(Protocol):
    def log_metrics(self, step: int | None = None, **metrics: Any) -> None: ...


@dataclass(frozen=True)
class SFTSettings:
    steps: int
    batch_size: int
    micro_batch_size: int
    grad_checkpointing: bool
    lr: float
    lr_milestones: tuple[int, ...]
    lr_gamma: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    image_aug: bool
    checkpoint_every: int
    log_every: int
    num_workers: int
    stop_l1: float | None
    seed: int

    @property
    def accum_steps(self) -> int:
        return self.batch_size // self.micro_batch_size

    @classmethod
    def from_cfg(cls, cfg: Config, device: str) -> SFTSettings:
        s = cfg["sft"]
        batch = int(s["batch_size"])
        micro, ckpt = resolve_micro_batch(s, batch, device)
        if batch % micro:
            raise ValueError(f"sft.batch_size {batch} not divisible by micro_batch_size {micro}")
        stop = s.get("stop_l1")
        return cls(
            steps=int(s["steps"]),
            batch_size=batch,
            micro_batch_size=micro,
            grad_checkpointing=ckpt,
            lr=float(s["lr"]),
            lr_milestones=tuple(int(m) for m in s.get("lr_milestones", [])),
            lr_gamma=float(s.get("lr_gamma", 0.1)),
            warmup_steps=int(s.get("warmup_steps", 0)),
            weight_decay=float(s.get("weight_decay", 0.0)),
            grad_clip=float(s.get("grad_clip", 1.0)),
            image_aug=bool(s.get("image_aug", True)),
            checkpoint_every=int(s.get("checkpoint_every", 2500)),
            log_every=int(s.get("log_every", 10)),
            num_workers=int(s.get("num_workers", 0)),
            stop_l1=None if stop is None else float(stop),
            seed=int(cfg.get("seed", 0)),
        )


def resolve_micro_batch(s: Config, batch: int, device: str) -> tuple[int, bool]:
    """`auto`: full batch without checkpointing on >= 40 GB GPUs (A100/H100), else 4 + ckpt."""
    micro_cfg, ckpt_cfg = s.get("micro_batch_size", "auto"), s.get("grad_checkpointing", "auto")
    if micro_cfg == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 2**30
            micro = batch if gb >= LARGE_GPU_GB else min(4, batch)
        else:
            micro = batch
    else:
        micro = int(micro_cfg)
    ckpt = micro < batch if ckpt_cfg == "auto" else bool(ckpt_cfg)
    return micro, ckpt


def make_optimizer(
    params: list[torch.nn.Parameter], s: SFTSettings
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    opt = torch.optim.AdamW(params, lr=s.lr, weight_decay=s.weight_decay)
    decay = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[m - s.warmup_steps for m in s.lr_milestones], gamma=s.lr_gamma
    )
    if s.warmup_steps <= 0:
        return opt, decay
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=s.warmup_steps)
    seq = torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warm, decay], milestones=[s.warmup_steps]
    )
    return opt, seq


def batches(
    dataset: Dataset[dict[str, Any]], micro_batch: int, num_workers: int, seed: int
) -> Iterator[SFTBatch]:
    """Endless shuffled micro-batches (reshuffled every epoch)."""
    g = torch.Generator().manual_seed(seed)
    while True:
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            shuffle=True,
            drop_last=len(dataset) >= micro_batch,  # type: ignore[arg-type]
            num_workers=num_workers,
            collate_fn=collate_sft,
            generator=g,
            persistent_workers=False,
        )
        yield from loader


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_mem_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    return 0.0


def train_sft(
    policy: VLAPolicy,
    dataset: Dataset[dict[str, Any]],
    s: SFTSettings,
    *,
    sink: MetricsSink,
    out_dir: Path | None,
    device: torch.device,
    probe_steps: int | None = None,
) -> dict[str, Any]:
    """Run SFT; returns results for the run record. `probe_steps` runs that many steps only,
    saves nothing, and reports timing/memory for extrapolating the full run."""
    params = policy.trainable_parameters()
    opt, sched = make_optimizer(params, s)
    data = batches(dataset, s.micro_batch_size, s.num_workers, s.seed)
    total = s.steps if probe_steps is None else min(probe_steps, s.steps)
    policy.train()
    step_times: list[float] = []
    window: dict[str, list[float]] = {}
    saved: list[str] = []
    stopped_early = False
    last: dict[str, float] = {}
    for step in range(1, total + 1):
        _sync(device)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        step_metrics: dict[str, float] = {}
        for _ in range(s.accum_steps):
            loss, m = policy.sft_loss(next(data), augment=s.image_aug)
            scaled: Any = loss / s.accum_steps
            scaled.backward()
            for k, v in m.items():
                step_metrics[k] = step_metrics.get(k, 0.0) + v / s.accum_steps
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, s.grad_clip))
        opt.step()
        sched.step()
        _sync(device)
        step_times.append(time.perf_counter() - t0)
        if not all(math.isfinite(v) for v in step_metrics.values()):
            raise FloatingPointError(f"non-finite SFT metrics at step {step}: {step_metrics}")
        for k, v in step_metrics.items():
            window.setdefault(k, []).append(v)
        if step % s.log_every == 0 or step == total:
            last = {k: statistics.fmean(v) for k, v in window.items()}
            sink.log_metrics(
                step=step,
                **last,
                lr=sched.get_last_lr()[0],
                grad_norm=grad_norm,
                s_per_step=step_times[-1],
                samples_per_s=s.batch_size / step_times[-1],
                max_mem_gb=round(_max_mem_gb(device), 3),
            )
            window = {}
            if s.stop_l1 is not None and last.get("l1_norm", math.inf) < s.stop_l1:
                stopped_early = probe_steps is None
        due = step % s.checkpoint_every == 0 or step == total or stopped_early
        if out_dir is not None and probe_steps is None and due:
            path = out_dir / f"step_{step:06d}"
            policy.save(path)
            saved.append(str(path))
        if stopped_early:
            break
    timed = step_times[2:] if len(step_times) > 4 else step_times
    return {
        "steps_run": len(step_times),
        "stopped_early": stopped_early,
        "final_metrics": last,
        "s_per_step_median": statistics.median(timed),
        "max_mem_gb": round(_max_mem_gb(device), 3),
        "checkpoints": saved,
        "settings": asdict(s),
    }


def estimate_cost(cfg: Config, s_per_step: float | None = None) -> dict[str, float]:
    """Hours and $ for `sft.steps` at `s_per_step` (default: cost.est_s_per_step) + setup."""
    c = cfg.get("cost", {})
    sps = float(s_per_step if s_per_step is not None else c.get("est_s_per_step", 1.0))
    hours = int(cfg["sft"]["steps"]) * sps / 3600.0 + float(c.get("setup_hours", 0.0))
    rate = float(c.get("usd_per_hour", 0.0))
    return {"s_per_step": sps, "hours": round(hours, 3), "usd": round(hours * rate, 2)}
