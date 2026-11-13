"""LeRobot checkpoints of the real-arm policies (SmolVLA, ACT; WO-07): load, identify, run one chunk.

Shared by the trainer (``scripts/wo07_train.py``: the save / reload check and the run record) and
the Mac inference path. A checkpoint is LeRobot 0.4.4's ``pretrained_model`` directory
(``config.json``, ``model.safetensors``, ``train_config.json`` and the saved pre/post-processor
pipelines, which carry the dataset's normalization stats and, for SmolVLA, the
``convention.SMOLVLA_RENAME_MAP`` rename).

Inputs are the batch ``src.robot.convention.policy_observation`` builds (dataset feature names,
float32 CHW images in [0, 1], canonical radians); outputs are un-normalized absolute joint targets
in canonical radians, (1, chunk, 6), for ``convention.action_targets``.

Provenance (``PROVENANCE_FILE``, written into every saved ``pretrained_model`` dir by the trainer,
part of the checkpoint hash; reviews conv-1, F1, F8): the training run id, the dataset's
convention citation, ``unit_map`` (``robot.units``: arm sign / offset decide what the stored
radians mean), calibration hash, dataset hash, protocol citation and, for SmolVLA, the VLM
dependency LeRobot loads from the Hub on EVERY load (``vlm_model_name``: the tokenizer the
instruction goes through, the processor and the VLM config/weights; ``processor_smolvla.py:74``,
``smolvlm_with_expert.py:76-87``), pinned to the commit it resolved to at training with the
sha256 of every non-weight file (``vlm_identity``). The deploy path (``src/robot/local_policy.py``)
loads that exact snapshot (``vlm_path`` below) and refuses any other.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

POLICY_TYPES = ("smolvla", "act")
PROVENANCE_FILE = "vla_rl_so101.json"
# Every file a load reads from a checkpoint dir (LeRobot's from_pretrained /
# make_pre_post_processors) plus this repo's provenance. SmolVLA's VLM files come from the Hub and
# are covered through the provenance's ``vlm`` block, not by these globs.
CHECKPOINT_GLOBS = (
    "config.json",
    "model.safetensors",
    "train_config.json",
    "policy_*",
    PROVENANCE_FILE,
)
# The non-weight files of a VLM snapshot (tokenizer, processor, chat template, configs)
VLM_META_PATTERNS = ("*.json", "*.txt", "*.jinja", "*.model", "tokenizer*", "merges*", "vocab*")
VLM_WEIGHT_PATTERNS = ("*.safetensors",)


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
    OFT adapters: ``checkpoint_sha256`` over ``name:sha256`` of every checkpoint file
    (``CHECKPOINT_GLOBS``: config, weights, train config, processors and their stats, the
    provenance), and ``checkpoint_revision``, the Hub snapshot commit when the dir is a
    ``snapshots/<sha>/`` download (None for a local dir). Same files = same hash, wherever they
    live. A SmolVLA load ALSO reads the VLM from the Hub; that dependency is in the hash only
    through the provenance's pinned ``vlm`` block (module doc), so a SmolVLA checkpoint without
    a provenance is not fully identified by this hash.

    The ONE checkpoint identity of PR 2: the trainer's run record, the Mac inference path's
    ``checkpoint_sha256`` (``src/robot/local_policy.py``), the real eval's ``policy_identity`` and
    the vishwaroop export's ``checkpoint_id`` (``src/eval/export_bundle.py``, which bundles exactly
    ``checkpoint_files``) all hash these files this way."""
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


def read_provenance(ckpt: str | Path) -> dict[str, Any] | None:
    """The checkpoint's ``PROVENANCE_FILE`` (None when the trainer did not write one: a Hub base
    such as smolvla_base, an untrained ACT)."""
    import json

    path = Path(ckpt) / PROVENANCE_FILE
    if not path.is_file():
        return None
    value: dict[str, Any] = json.loads(path.read_text())
    return value


def write_provenance(ckpt: str | Path, value: dict[str, Any]) -> Path:
    import json

    path = Path(ckpt) / PROVENANCE_FILE
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _files_sha(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): _sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix != ".safetensors"  # the weights: model.safetensors
    }


def vlm_snapshot(name: str, revision: str | None, *, weights: bool) -> Path:
    """The local HF-cache snapshot of the VLM at ``revision`` (None = what ``main`` resolves to
    now), downloading what is missing: the non-weight files, plus the weights when asked."""
    from huggingface_hub import snapshot_download

    patterns = list(VLM_META_PATTERNS) + (list(VLM_WEIGHT_PATTERNS) if weights else [])
    return Path(snapshot_download(name, revision=revision, allow_patterns=patterns))


def vlm_identity(name: str, revision: str | None = None) -> dict[str, Any]:
    """{name, revision (the commit), files: {path: sha256}} of the VLM's non-weight files."""
    path = vlm_snapshot(name, revision, weights=False)
    files = _files_sha(path)
    if not files:
        raise FileNotFoundError(f"{name}@{revision}: no tokenizer / config files in {path}")
    return {"name": name, "revision": path.name, "files": files}


def pinned_vlm_path(vlm: dict[str, Any], *, weights: bool) -> Path:
    """The snapshot of a provenance's ``vlm`` block, refused unless every non-weight file hashes
    as recorded at training (``ValueError``)."""
    path = vlm_snapshot(str(vlm["name"]), str(vlm["revision"]), weights=weights)
    got = _files_sha(path)
    if path.name != vlm["revision"] or got != vlm["files"]:
        changed = sorted(
            k for k in set(got) | set(vlm["files"]) if got.get(k) != vlm["files"].get(k)
        )
        raise ValueError(
            f"VLM {vlm['name']}@{vlm['revision']}: the local snapshot {path} differs from the one "
            f"trained with ({changed[:5]})"
        )
    return path


def load_checkpoint(
    ckpt: str | Path,
    device: str,
    *,
    config: Any | None = None,
    n_action_steps: int | None = None,
    preprocessor_overrides: dict[str, Any] | None = None,
    vlm_path: Path | None = None,
) -> tuple[Any, Any, Any]:
    """(policy in eval mode, preprocessor, postprocessor) from a ``pretrained_model`` dir.

    The ONE loader of the trainer's reload check and the Mac inference path
    (``src/robot/local_policy.py``). ``config``: the dir's ``PreTrainedConfig`` when the caller
    already read (and checked) it; ``n_action_steps``: override the saved value (the deploy loop
    executes whole chunks); ``preprocessor_overrides``: extra processor-step overrides (e.g. a
    rename map), merged with the device override; ``vlm_path`` (SmolVLA): load the VLM, its
    processor and the tokenizer from this local snapshot (``pinned_vlm_path``) instead of
    resolving ``vlm_model_name`` on the Hub."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    path = Path(ckpt)
    pcfg = PreTrainedConfig.from_pretrained(path) if config is None else config
    if n_action_steps is not None:
        pcfg.n_action_steps = int(n_action_steps)
    if pcfg.type not in POLICY_TYPES:
        raise ValueError(f"{path}: policy type {pcfg.type!r} is not one of {POLICY_TYPES}")
    pcfg.device = device
    pcfg.pretrained_path = path
    overrides: dict[str, Any] = dict(preprocessor_overrides or {})
    if vlm_path is not None and pcfg.type == "smolvla":
        pcfg.vlm_model_name = str(vlm_path)  # in memory only: the saved config keeps the Hub name
        overrides["tokenizer_processor"] = {"tokenizer_name": str(vlm_path)}
    policy = get_policy_class(pcfg.type).from_pretrained(path, config=pcfg)
    cpu_fp32(policy.to(device), device).eval()
    if device == "cpu":
        # from_pretrained builds SmolVLA's action expert in bf16 (the VLM text config's dtype) and
        # copies the saved weights INTO it: fp32 weights trained on CPU were rounded to bf16 (max
        # 2e-4 per weight, 0.16 rad on a chunk after 3 tiny steps). Re-load them into the fp32 model.
        from safetensors.torch import load_file

        # strict=False as LeRobot's own from_pretrained (a Hub base may carry extra buffers)
        policy.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    overrides["device_processor"] = {"device": device}
    pre, post = make_pre_post_processors(
        pcfg, pretrained_path=str(path), preprocessor_overrides=overrides
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
