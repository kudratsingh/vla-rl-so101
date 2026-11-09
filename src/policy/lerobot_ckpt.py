"""LeRobot checkpoints of the real-arm policies (SmolVLA, ACT; WO-07): load, identify, run one chunk.

Shared by the trainer (``scripts/wo07_train.py``: the save / reload check and the run record) and
the Mac inference path. A checkpoint is LeRobot 0.4.4's ``pretrained_model`` directory
(``config.json``, ``model.safetensors``, ``train_config.json`` and the saved pre/post-processor
pipelines, which carry the dataset's normalization stats and, for SmolVLA, the
``convention.SMOLVLA_RENAME_MAP`` rename).

Inputs are the batch ``src.robot.convention.policy_observation`` builds (dataset feature names,
float32 CHW images in [0, 1], canonical radians); outputs are un-normalized absolute joint targets
in canonical radians, (1, chunk, 6), for ``convention.action_targets``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

POLICY_TYPES = ("smolvla", "act")
# Everything LeRobot's from_pretrained / make_pre_post_processors read from a checkpoint dir.
CHECKPOINT_GLOBS = ("config.json", "model.safetensors", "train_config.json", "policy_*")


def cpu_fp32(policy: Any, device: str) -> Any:
    """SmolVLA loads its VLM in bf16; on CPU that is ~30x slower than fp32 (28 s vs 0.9 s/step on
    an M3 Pro, batch 2; ``scripts/wo01_smolvla_smoke.py``), so CPU runs use fp32. MPS / CUDA keep
    LeRobot's dtypes (VLM bf16, action expert fp32)."""
    if device == "cpu":
        policy.to(torch.float32)
    return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_files(ckpt: Path) -> list[Path]:
    """The files a load reads, sorted by name (refuses a dir without config + weights)."""
    found = sorted({p for g in CHECKPOINT_GLOBS for p in ckpt.glob(g) if p.is_file()})
    names = {p.name for p in found}
    missing = {"config.json", "model.safetensors"} - names
    if missing:
        raise FileNotFoundError(f"{ckpt} is not a LeRobot checkpoint: missing {sorted(missing)}")
    return found


def checkpoint_identity(ckpt: str | Path) -> dict[str, Any]:
    """Content identity of a checkpoint dir, as ``src.robot.serve.adapter_identity`` does for the
    OFT adapters: ``checkpoint_sha256`` over ``name:sha256`` of every file a load reads, and
    ``checkpoint_revision``, the Hub snapshot commit when the dir is a ``snapshots/<sha>/``
    download (None for a local dir). Same weights + processors = same hash, wherever they live."""
    path = Path(ckpt)
    h = hashlib.sha256()
    files = checkpoint_files(path)
    for f in files:
        h.update(f"{f.name}:{_sha256(f)}\n".encode())
    parts = path.resolve().parts
    rev = parts[parts.index("snapshots") + 1] if "snapshots" in parts[:-1] else None
    return {
        "checkpoint_sha256": h.hexdigest(),
        "checkpoint_revision": rev,
        "checkpoint_files": [f.name for f in files],
        "checkpoint_bytes": sum(f.stat().st_size for f in files),
    }


def load_checkpoint(ckpt: str | Path, device: str) -> tuple[Any, Any, Any]:
    """(policy in eval mode, preprocessor, postprocessor) from a ``pretrained_model`` dir."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(ckpt)
    pcfg = PreTrainedConfig.from_pretrained(path)
    if pcfg.type not in POLICY_TYPES:
        raise ValueError(f"{path}: policy type {pcfg.type!r} is not one of {POLICY_TYPES}")
    pcfg.device = device
    pcfg.pretrained_path = path
    policy = get_policy_class(pcfg.type).from_pretrained(path, config=pcfg)
    cpu_fp32(policy.to(device), device).eval()
    if device == "cpu":
        # from_pretrained builds SmolVLA's action expert in bf16 (the VLM text config's dtype) and
        # copies the saved weights INTO it: fp32 weights trained on CPU were rounded to bf16 (max
        # 2e-4 per weight, 0.16 rad on a chunk after 3 tiny steps). Re-load them into the fp32 model.
        from safetensors.torch import load_file

        policy.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
    pre, post = make_pre_post_processors(
        pcfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def rename_map_of(preprocessor: Any) -> dict[str, str]:
    """The observation rename a (saved) preprocessor applies; {} if it has no rename step."""
    for step in getattr(preprocessor, "steps", []):
        rm = getattr(step, "rename_map", None)
        if isinstance(rm, dict):
            return {str(k): str(v) for k, v in rm.items()}
    return {}


def flow_noise(policy: Any, seed: int, batch: int = 1) -> torch.Tensor | None:
    """The explicit flow-matching noise of one SmolVLA chunk, (batch, chunk_size, max_action_dim),
    from a fixed seed (SmolVLA samples it per call otherwise). None for ACT (deterministic in eval:
    the VAE latent is zero at inference)."""
    cfg = policy.config
    if cfg.type != "smolvla":
        return None
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, int(cfg.chunk_size), int(cfg.max_action_dim))
    return torch.randn(shape, generator=gen, dtype=torch.float32)


@torch.inference_mode()
def predict_chunk(
    policy: Any,
    pre: Any,
    post: Any,
    observation: dict[str, Any],
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One action chunk for ``convention.policy_observation``'s batch: reset the policy state, run
    the preprocessor (rename, tokenize, normalize), predict in eval mode with ``noise`` (SmolVLA),
    un-normalize. Returns float32 CPU (1, chunk, 6), canonical radians."""
    policy.reset()
    policy.eval()
    batch = pre(dict(observation))
    if noise is not None:
        dev = next(policy.parameters()).device
        chunk = policy.predict_action_chunk(batch, noise=noise.to(dev))
    else:
        chunk = policy.predict_action_chunk(batch)
    out = post(chunk)
    result: torch.Tensor = out.detach().to("cpu", torch.float32)
    return result
