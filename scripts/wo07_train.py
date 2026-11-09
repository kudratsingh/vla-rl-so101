#!/usr/bin/env python
"""WO-07: fine-tune SmolVLA or ACT on the real SO-101 demos (LeRobot 0.4.4's own trainer).

Training is ``lerobot.scripts.lerobot_train.train`` called in-process (the path of
``scripts/wo01_smolvla_smoke.py``, so the torchcodec compat shim applies), on a LeRobotDataset in
the frozen real-arm convention ``src/robot/convention.py``: ``observation.images.front`` +
``observation.images.wrist`` (480 x 640), ``observation.state`` / ``action`` in canonical radians,
absolute targets, 15 Hz. The dataset is refused unless its ``meta/info.json`` conforms
(``convention.check_dataset_info``) and its ``meta/vla_rl_so101.json`` cites this convention's
hash. SmolVLA gets ``convention.SMOLVLA_RENAME_MAP`` (front -> camera1, wrist -> camera2, as the
LeRobot SmolVLA guide); LeRobot saves it in the checkpoint's preprocessor, so inference renames
the same way. ACT reads the dataset keys directly.

Presets (``--config`` defaults to ``configs/wo07_train.yaml`` / ``configs/wo07_train_tiny.yaml``):
  tiny   CPU, < 2 min per policy, never a result: a synthetic two-camera dataset built with the
         convention's schema and frame builder, a few steps, save, reload, and the same chunk from
         the in-memory and the reloaded policy on one observation, in eval mode with the policy
         state reset and (SmolVLA) the SAME explicit flow-matching noise for both calls.
  real   the paid run on one RTX 4090 (scripts/pod/wo07_train_job.sh); the same reload check.
``--probe``: a short timed run (no checkpoint): s/step, peak VRAM, projected hours and $ of the full
run at this batch -> ``runs/<id>/probe.json``. Exit 3 = CUDA out of memory at this batch.

    uv run scripts/wo07_train.py --preset tiny --policy smolvla
    uv run scripts/wo07_train.py --preset tiny --policy act
    uv run scripts/wo07_train.py --preset real --policy smolvla --probe --batch-size 64   # pod only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import shutil
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.data.recorder import datasets_write_compat, install_torchcodec_compat, load_demo_dataset
from src.policy import lerobot_ckpt as ck
from src.robot import convention as conv
from src.robot.units import JOINT_NAMES, URDF_LIMITS_RAD
from src.runs import Run

POLICIES = ("smolvla", "act")
DEFAULT_CONFIGS = {"tiny": "configs/wo07_train_tiny.yaml", "real": "configs/wo07_train.yaml"}
PROVENANCE = "meta/vla_rl_so101.json"
EXIT_OOM = 3
GIB = 1024**3


class TrainSetupError(ValueError):
    """A dataset or config this trainer refuses (nothing was trained)."""


class RunFailed(RuntimeError):
    """The run ended without a usable result (OOM, reload check); carries the exit code."""

    def __init__(self, rc: int, msg: str) -> None:
        super().__init__(msg)
        self.rc = rc


# ------------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------------
def resolve_config(preset: str, config: Path | None) -> Config:
    if preset not in DEFAULT_CONFIGS:
        raise TrainSetupError(
            f"--preset {preset}: this trainer trains on real-arm demos; use tiny or real"
        )
    return load_config(preset, config or REPO_ROOT / DEFAULT_CONFIGS[preset])


def policy_block(cfg: Config, name: str) -> dict[str, Any]:
    """``policies.<name>`` checked against the convention (chunk) and the trainer's needs."""
    if name not in POLICIES:
        raise TrainSetupError(f"--policy {name}: expected one of {POLICIES}")
    block = dict((cfg.get("policies") or {}).get(name) or {})
    for key in ("type", "base", "steps", "batch_size"):
        if block.get(key) in (None, ""):
            raise TrainSetupError(f"policies.{name}.{key} is missing")
    if block["type"] != name:
        raise TrainSetupError(f"policies.{name}.type is {block['type']!r}")
    ov = dict(block.get("overrides") or {})
    if int(ov.get("chunk_size", -1)) != conv.CHUNK_SIZE[name]:
        raise TrainSetupError(
            f"policies.{name}.overrides.chunk_size {ov.get('chunk_size')} != the convention's "
            f"{conv.CHUNK_SIZE[name]} (src/robot/convention.py CHUNK_SIZE)"
        )
    if int(ov.get("n_action_steps", 0)) > conv.CHUNK_SIZE[name]:
        raise TrainSetupError(f"policies.{name}.overrides.n_action_steps exceeds the chunk")
    block["overrides"] = ov
    return block


def make_policy_config(name: str, block: dict[str, Any], device: str) -> Any:
    """The LeRobot policy config: smolvla_base's own config (``policy.base``) or a fresh ACT, with
    ``overrides`` set and checked (an unknown key is refused, never ignored)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy_config as lerobot_policy_config

    base = str(block["base"])
    if name == "smolvla":
        pcfg = PreTrainedConfig.from_pretrained(base)
        if pcfg.type != "smolvla":
            raise TrainSetupError(f"policy.base {base} is a {pcfg.type!r} checkpoint")
        pcfg.pretrained_path = Path(base)
    elif base == "act":
        pcfg = lerobot_policy_config("act")
    else:
        raise TrainSetupError(f"policies.act.base must be 'act' (fresh LeRobot ACT), got {base!r}")
    for key, value in block["overrides"].items():
        if not hasattr(pcfg, key):
            raise TrainSetupError(f"policies.{name}.overrides.{key}: no such {pcfg.type} setting")
        current = getattr(pcfg, key)
        setattr(pcfg, key, tuple(value) if isinstance(current, tuple) else value)
    if pcfg.n_action_steps > pcfg.chunk_size:
        raise TrainSetupError("n_action_steps > chunk_size")
    pcfg.device = device
    pcfg.push_to_hub = False
    pcfg.repo_id = None
    return pcfg


# ------------------------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------------------------
@dataclass
class DatasetRef:
    repo_id: str
    root: Path
    revision: str | None
    synthetic: bool


def dataset_ref(cfg: Config, run_dir: Path) -> DatasetRef:
    d = cfg.get("dataset") or {}
    if d.get("synthetic"):
        return DatasetRef("local/wo07-tiny-synthetic", run_dir / "dataset", None, True)
    repo_id = str(d["repo_id"])
    root = Path(str(d.get("root", "data/real_teleop")))
    if not root.is_absolute():
        root = REPO_ROOT / root
    rev = d.get("revision")
    return DatasetRef(repo_id, root / repo_id.split("/")[-1], str(rev) if rev else None, False)


def build_synthetic_dataset(ref: DatasetRef, episodes: int, frames: int, seed: int) -> None:
    """A tiny two-camera dataset with the real recorder's schema (``convention.dataset_features``,
    v3.0, 15 fps, AV1) and frames from ``convention.recording_frame``: a smooth joint trajectory
    inside the URDF limits, action = the next state (absolute targets), and images that change with
    the state so the policy has something to fit. Plus the provenance file citing the convention."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(seed)
    lo = np.array([URDF_LIMITS_RAD[n][0] for n in JOINT_NAMES])
    hi = np.array([URDF_LIMITS_RAD[n][1] for n in JOINT_NAMES])
    mid, amp = (lo + hi) / 2, (hi - lo) / 4
    yy, xx = np.mgrid[0 : conv.IMAGE_HEIGHT, 0 : conv.IMAGE_WIDTH]
    with datasets_write_compat():
        ds = LeRobotDataset.create(
            repo_id=ref.repo_id,
            fps=conv.FPS,
            features=conv.dataset_features(),
            root=ref.root,
            robot_type=conv.ROBOT_TYPE,
            use_videos=True,
            image_writer_threads=4 * len(conv.CAMERAS),
            video_backend=conv.VIDEO_BACKEND,
            vcodec=conv.VIDEO_CODEC,
        )
        for _ in range(episodes):
            phase = rng.uniform(0, 2 * math.pi, size=6)
            q = [mid + amp * np.sin(phase + 0.2 * t) for t in range(frames + 1)]
            for t in range(frames):
                s = (q[t] - lo) / (hi - lo)  # in [0, 1]
                front = np.empty(conv.IMAGE_SHAPE_HWC, np.uint8)
                front[..., 0] = (255 * s[0] * xx / conv.IMAGE_WIDTH).astype(np.uint8)
                front[..., 1] = (255 * s[1] * yy / conv.IMAGE_HEIGHT).astype(np.uint8)
                front[..., 2] = int(255 * s[2])
                wrist = np.empty(conv.IMAGE_SHAPE_HWC, np.uint8)
                wrist[...] = (int(255 * s[3]), int(255 * s[4]), int(255 * s[5]))
                ds.add_frame(
                    conv.recording_frame(
                        {"front": front, "wrist": wrist}, q[t], q[t + 1], conv.DEMO_TASK_STRING
                    )
                )
            ds.save_episode()
        ds.finalize()
    provenance = {
        "source": "scripts/wo07_train.py --preset tiny (synthetic, never a result)",
        "convention": conv.citation(),
        "units": dict(conv.DECLARED_UNITS),
        "synthetic": True,
        "rehearsal": False,
    }
    (ref.root / PROVENANCE).write_text(json.dumps(provenance, indent=2) + "\n")


def check_dataset(root: Path, *, allow_rehearsal: bool) -> dict[str, Any]:
    """Refuse a dataset outside the convention: ``meta/info.json`` must conform and
    ``meta/vla_rl_so101.json`` must cite THIS convention version and hash."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise TrainSetupError(
            f"no dataset at {root}; on the pod: hf download <repo_id> --repo-type dataset "
            f"--revision <commit> --local-dir {root}"
        )
    info = json.loads(info_path.read_text())
    problems = conv.check_dataset_info(info)
    prov_path = root / PROVENANCE
    if not prov_path.is_file():
        problems.append(f"{PROVENANCE} missing: not recorded by wo06_teleop_record.py")
        prov: dict[str, Any] = {}
    else:
        prov = json.loads(prov_path.read_text())
        cited = prov.get("convention") or {}
        if cited.get("hash") != conv.convention_hash():
            problems.append(
                f"{PROVENANCE} cites convention {cited.get('id')} {cited.get('version')} "
                f"{cited.get('hash')}, this trainer is {conv.CONVENTION_VERSION} "
                f"{conv.convention_hash()}"
            )
        if prov.get("rehearsal") and not allow_rehearsal:
            problems.append("a rehearsal recording (dataset.allow_rehearsal is false)")
    if problems:
        raise TrainSetupError(f"dataset {root} refused:\n  " + "\n  ".join(problems))
    return {
        "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]),
        "fps": int(info["fps"]),
        "codebase_version": info["codebase_version"],
        "rehearsal": bool(prov.get("rehearsal", False)),
        "synthetic": bool(prov.get("synthetic", False)),
    }


def dataset_hash(root: Path) -> dict[str, Any]:
    """SHA-256 over ``relpath:sha256`` of every dataset file (hidden dirs such as the ``hf
    download`` cache skipped): the exact bytes trained on."""
    h = hashlib.sha256()
    n = size = 0
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = f.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        digest = hashlib.sha256()
        with f.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        h.update(f"{rel.as_posix()}:{digest.hexdigest()}\n".encode())
        n += 1
        size += f.stat().st_size
    return {"sha256": h.hexdigest(), "files": n, "bytes": size}


# ------------------------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------------------------
@dataclass
class Trained:
    checkpoint: Path | None
    step_s: list[float]
    losses: list[float]
    policy: Any = None
    pre: Any = None
    post: Any = None
    peak: dict[str, float] = field(default_factory=dict)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def train(
    cfg: Config,
    run: Run,
    name: str,
    block: dict[str, Any],
    ref: DatasetRef,
    *,
    steps: int,
    batch_size: int,
    device: str,
    save: bool,
) -> Trained:
    """``lerobot_train.train`` in-process; keeps the trained policy and processors in memory."""
    install_torchcodec_compat()
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts import lerobot_train

    tcfg = cfg.get("train") or {}
    pcfg = make_policy_config(name, block, device)
    out = run.dir / "train"
    train_cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=ref.repo_id,
            root=str(ref.root),
            revision=ref.revision,
            video_backend=conv.VIDEO_BACKEND,
        ),
        policy=pcfg,
        output_dir=out,
        job_name=f"wo07_{name}_{cfg['preset']}",
        seed=int(cfg.get("seed", 0)),
        # macOS DataLoader workers are spawned and would not inherit the torchcodec shim
        num_workers=0 if sys.platform == "darwin" else int(tcfg.get("num_workers", 4)),
        batch_size=batch_size,
        steps=steps,
        eval_freq=0,
        log_freq=max(1, min(int(tcfg.get("log_freq", 200)), steps)),
        save_checkpoint=save,
        save_freq=max(1, min(int(tcfg.get("save_freq", steps)), steps)),
        wandb=WandBConfig(enable=False),
        rename_map=dict(conv.SMOLVLA_RENAME_MAP) if name == "smolvla" else {},
    )

    res = Trained(None, [], [])
    original_update = lerobot_train.update_policy
    original_make_policy = lerobot_train.make_policy
    original_processors = lerobot_train.make_pre_post_processors

    def make_policy(*args: Any, **kwargs: Any) -> Any:
        res.policy = ck.cpu_fp32(original_make_policy(*args, **kwargs), device)
        return res.policy

    def make_processors(*args: Any, **kwargs: Any) -> Any:
        res.pre, res.post = original_processors(*args, **kwargs)
        return res.pre, res.post

    def timed_update(*args: Any, **kwargs: Any) -> Any:
        _sync(device)
        t0 = time.perf_counter()
        tracker, output = original_update(*args, **kwargs)
        _sync(device)
        res.step_s.append(time.perf_counter() - t0)
        loss = getattr(getattr(tracker, "loss", None), "val", None)
        if loss is not None:
            res.losses.append(float(loss))
        run.log_metrics(step=len(res.step_s), loss=loss, step_s=round(res.step_s[-1], 4))
        return tracker, output

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    argv = sys.argv
    lerobot_train.update_policy = timed_update
    lerobot_train.make_policy = make_policy
    lerobot_train.make_pre_post_processors = make_processors
    sys.argv = argv[:1]  # TrainPipelineConfig.validate re-reads sys.argv for --policy.path
    try:
        lerobot_train.train(train_cfg)
    finally:
        sys.argv = argv
        lerobot_train.update_policy = original_update
        lerobot_train.make_policy = original_make_policy
        lerobot_train.make_pre_post_processors = original_processors
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        res.peak = {
            "allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
            "reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
            "device_total_gib": round(props.total_memory / GIB, 2),
        }
    if save:
        res.checkpoint = (out / "checkpoints" / "last" / "pretrained_model").resolve()
    return res


def reload_check(
    trained: Trained,
    ref: DatasetRef,
    name: str,
    *,
    device: str,
    seed: int,
    atol: float,
) -> dict[str, Any]:
    """Save -> reload -> the same chunk. One dataset observation through
    ``convention.policy_observation_from_sample`` (the inference preprocessing), the in-memory
    policy and the reloaded checkpoint each in eval mode with the state reset, SmolVLA with the
    same explicit noise for both calls; plus the reloaded preprocessor's rename map."""
    assert trained.checkpoint is not None
    sample = load_demo_dataset(ref.repo_id, ref.root)[0]
    obs = conv.policy_observation_from_sample(sample)
    noise = ck.flow_noise(trained.policy, seed)
    a = ck.predict_chunk(trained.policy, trained.pre, trained.post, obs, noise)
    policy, pre, post = ck.load_checkpoint(trained.checkpoint, device)
    b = ck.predict_chunk(policy, pre, post, obs, noise)
    want_rename = dict(conv.SMOLVLA_RENAME_MAP) if name == "smolvla" else {}
    out: dict[str, Any] = {
        "observation": "dataset frame 0 via convention.policy_observation_from_sample",
        "chunk_shape": list(b.shape),
        "max_abs_diff_rad": float((a - b).abs().max()),
        "atol_rad": atol,
        "explicit_noise": noise is not None,
        "noise_seed": seed if noise is not None else None,
        "rename_map": ck.rename_map_of(pre),
        "rename_map_ok": ck.rename_map_of(pre) == want_rename,
    }
    if noise is not None:  # the noise is really used: another draw moves the chunk
        c = ck.predict_chunk(policy, pre, post, obs, ck.flow_noise(policy, seed + 1))
        out["other_noise_max_abs_diff_rad"] = float((b - c).abs().max())
    targets = conv.action_targets(b)
    out["targets_shape"] = list(targets.shape)
    out["ok"] = bool(
        out["max_abs_diff_rad"] <= atol
        and out["rename_map_ok"]
        and tuple(targets.shape) == (conv.CHUNK_SIZE[name], len(JOINT_NAMES))
        and (noise is None or out["other_noise_max_abs_diff_rad"] > atol)
    )
    del policy, pre, post
    return out


def projection(cfg: Config, block: dict[str, Any], s_per_step: float, batch: int) -> dict[str, Any]:
    """Hours and $ of the full run at the measured s/step (this batch), + setup and storage."""
    cost = cfg.get("cost") or {}
    price = float(cost.get("usd_per_hour", 0.0))
    steps = int(block["steps"])
    train_h = steps * s_per_step / 3600.0
    setup_h = float((cfg.get("probe") or {}).get("setup_h", 0.0))
    total_h = train_h + setup_h
    gb = float(cost.get("volume_gb", 0.0))
    storage = gb * float(cost.get("storage_usd_per_gb_month", 0.0)) * total_h / 730.0
    return {
        "gpu": cost.get("gpu"),
        "usd_per_hour": price,
        "steps": steps,
        "batch_size": batch,
        "s_per_step": round(s_per_step, 4),
        "train_h": round(train_h, 3),
        "setup_h": setup_h,
        "total_h": round(total_h, 3),
        "gpu_usd": round(total_h * price, 2),
        "storage_usd": round(storage, 2),
        "total_usd": round(total_h * price + storage, 2),
    }


# ------------------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    parser.add_argument("--steps", type=int, default=None, help="default: policies.<p>.steps")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="default: policies.<p>.batch_size"
    )
    parser.add_argument("--probe", action="store_true", help="timed short run, no checkpoint")
    parser.add_argument("--dataset-revision", default=None, help="Hub commit of the dataset")
    parser.add_argument(
        "--keep-checkpoint", action="store_true", help="tiny: keep the checkpoint and dataset"
    )
    args = parser.parse_args()

    try:
        cfg = resolve_config(args.preset, args.config)
        block = policy_block(cfg, args.policy)
    except TrainSetupError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    device = args.device or str(cfg.get("device", "cpu"))
    probe = cfg.get("probe") or {}
    steps = int(probe["steps"]) if args.probe else int(args.steps or block["steps"])
    batch = int(args.batch_size or block["batch_size"])
    if args.dataset_revision:
        cfg.setdefault("dataset", {})["revision"] = args.dataset_revision
    cfg["policy"] = block  # policy.base etc. in the run record (CLAUDE.md: names from config)
    cfg["run"] = {"policy": args.policy, "steps": steps, "batch_size": batch, "probe": args.probe}
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        return _run(cfg, args, block, device, steps, batch)
    except TrainSetupError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    except RunFailed as e:
        print(f"failed: {e}", file=sys.stderr)
        return e.rc


def _run(
    cfg: Config,
    args: argparse.Namespace,
    block: dict[str, Any],
    device: str,
    steps: int,
    batch: int,
) -> int:
    probe = cfg.get("probe") or {}
    with Run(work_order="WO-07", script=__file__, cfg=cfg, device=device) as run:
        ref = dataset_ref(cfg, run.dir)
        if ref.synthetic:
            d = cfg["dataset"]
            t0 = time.monotonic()
            build_synthetic_dataset(
                ref, int(d["episodes"]), int(d["frames_per_episode"]), int(cfg.get("seed", 0))
            )
            run.results["synthetic_build_s"] = round(time.monotonic() - t0, 1)
        ds_info = check_dataset(
            ref.root, allow_rehearsal=bool(cfg["dataset"].get("allow_rehearsal", False))
        )
        run.results.update(
            policy=args.policy,
            base=block["base"],
            convention=conv.citation(),
            rename_map=dict(conv.SMOLVLA_RENAME_MAP) if args.policy == "smolvla" else {},
            dataset={
                "repo_id": ref.repo_id,
                "root": str(ref.root),
                "revision": ref.revision,
                **ds_info,
                **dataset_hash(ref.root),
            },
        )
        run.flush()
        t0 = time.monotonic()
        try:
            trained = train(
                cfg,
                run,
                args.policy,
                block,
                ref,
                steps=steps,
                batch_size=batch,
                device=device,
                save=not args.probe,
            )
        except torch.cuda.OutOfMemoryError as e:
            first = str(e).splitlines()[0] if str(e) else "CUDA out of memory"
            run.results["oom"] = {"batch_size": batch, "error": first}
            raise RunFailed(EXIT_OOM, f"[probe] OOM at batch {batch}: {first}") from None
        train_s = time.monotonic() - t0
        warm = int(probe.get("warmup_steps", 1)) if args.probe else 1
        timed = trained.step_s[warm:] or trained.step_s
        step_s = statistics.median(timed)
        run.results.update(
            steps=steps,
            batch_size=batch,
            train_wall_s=round(train_s, 1),
            step_s_median=round(step_s, 4),
            loss_first=trained.losses[0] if trained.losses else None,
            loss_last=trained.losses[-1] if trained.losses else None,
            peak_vram=trained.peak or None,
        )
        est = projection(cfg, block, step_s, batch)
        run.results["projection_full_run"] = est
        check: dict[str, Any] = {"ok": True}
        if args.probe:
            (run.dir / "probe.json").write_text(json.dumps(run.results, indent=2) + "\n")
            print(
                f"[probe] {args.policy} batch {batch}: {step_s:.3f} s/step"
                f" (median of {len(timed)}), peak VRAM {trained.peak or 'n/a'}"
            )
            print(
                f"[probe] full run {args.policy}: {est['steps']} steps x batch {batch} ="
                f" {est['train_h']:.2f} h + setup {est['setup_h']} h, extrapolated"
                f" ${est['total_usd']:.2f} at ${est['usd_per_hour']}/h ({est['gpu']})"
            )
        else:
            assert trained.checkpoint is not None
            run.results["checkpoint"] = {
                "path": str(trained.checkpoint),
                **ck.checkpoint_identity(trained.checkpoint),
            }
            run.flush()
            tcfg = cfg.get("train") or {}
            check = reload_check(
                trained,
                ref,
                args.policy,
                device=device,
                seed=int(tcfg.get("compare_seed", 0)),
                atol=float(tcfg.get("compare_atol", 1e-3)),
            )
            run.results["reload_check"] = check
            print(f"reload check: {json.dumps(check)}")
        del trained
        gc.collect()
        if cfg["preset"] == "tiny" and not args.keep_checkpoint:
            shutil.rmtree(run.dir / "train" / "checkpoints", ignore_errors=True)
            shutil.rmtree(run.dir / "dataset", ignore_errors=True)
            if "checkpoint" in run.results:
                run.results["checkpoint"]["deleted"] = True
        if not check["ok"]:
            raise RunFailed(1, f"reload check failed: {json.dumps(check)}")
    r = run.results
    print(
        f"\nrun {run.id}  preset={cfg['preset']}  policy={args.policy}  device={device}\n"
        f"train: {steps} steps x batch {batch}; median step {r['step_s_median']:.3f}s;"
        f" loss {r['loss_first']} -> {r['loss_last']}\n"
        f"record: {run.record_path}  wall {run.record['wall_clock_s']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
