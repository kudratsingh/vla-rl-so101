"""OpenVLA-OFT policy (ADR-0001): LoRA on openvla/openvla-7b, SimpleVLA-RL's CE action head.

One model, one input builder (`OFTPreprocessor.build_inputs`), one forward (`action_logits`) for
SFT, greedy eval and GRPO sampling/scoring, so the three can never disagree on token layout,
normalisation or image preprocessing.

Checkpoint format (`save`): a PEFT adapter directory (LoRA + the fully-trained proprio projector
as `modules_to_save`) plus `norm_stats.json` (OFT dataset_statistics layout) and `policy.json`
(layout version, chunk, base checkpoint). Loading an adapter always reuses ITS norm stats.

Builders: `build_policy(cfg, device)` (SFT / eval) and `build_rl_policy(cfg, device)` (the
`src.rl.policy_api.POLICY_BUILDERS["openvla_oft"]` entry: SFT adapter loaded, LoRA-only grads).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.config import Config
from src.policy.base import ActionSample, PolicyObs, SFTBatch, TokenScores
from src.policy.oft import model as oft_model
from src.policy.oft.actions import (
    N_BINS,
    NormBounds,
    bounds_from_norm_stats,
    norm_stats_dict,
)
from src.policy.oft.preprocess import (
    AugmentConfig,
    HFPromptTokenizer,
    OFTInputs,
    OFTPreprocessor,
    PromptTokenizer,
    ToyPromptTokenizer,
)

NORM_STATS_FILE = "norm_stats.json"
POLICY_FILE = "policy.json"


@dataclass(frozen=True)
class OFTPolicyConfig:
    tiny_random: bool
    checkpoint: str | None
    revision: str | None
    adapter: str | None
    chunk_len: int
    action_dim: int
    precision: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_lm_head: bool
    train_proprio_projector: bool
    crop_scale: float
    seed: int

    @classmethod
    def from_cfg(cls, cfg: Config) -> OFTPolicyConfig:
        p = cfg["policy"]
        if p.get("base") != "openvla_oft":
            raise ValueError(f"policy.base={p.get('base')!r} is not openvla_oft")
        tiny = bool(p.get("tiny_random", False))
        if not tiny and not p.get("checkpoint"):
            raise ValueError("policy.checkpoint (HF id of the base VLA) is required")
        return cls(
            tiny_random=tiny,
            checkpoint=None if tiny else str(p["checkpoint"]),
            revision=p.get("revision"),
            adapter=p.get("adapter"),
            chunk_len=int(p.get("action_chunk", 8)),
            action_dim=int(p.get("action_dim", 6)),
            precision=str(p.get("precision", "bf16")),
            lora_rank=int(p.get("lora_rank", 32)),
            lora_alpha=int(p.get("lora_alpha", 16)),
            lora_dropout=float(p.get("lora_dropout", 0.0)),
            lora_lm_head=bool(p.get("lora_lm_head", True)),
            train_proprio_projector=bool(p.get("train_proprio_projector", True)),
            crop_scale=float(p.get("center_crop_scale", 0.9)),
            seed=int(cfg.get("seed", 0)),
        )


class OpenVLAOFTPolicy(nn.Module):
    def __init__(
        self,
        peft_model: nn.Module,
        preprocessor: OFTPreprocessor,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        meta: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = peft_model
        self.pre = preprocessor
        self.chunk_len = preprocessor.chunk_len
        self.action_dim = preprocessor.action_dim
        self.n_bins = N_BINS
        self._device = device
        self._autocast_dtype = autocast_dtype
        self._meta = meta
        self._aug_gen = torch.Generator().manual_seed(int(meta.get("seed", 0)))

    # ------------------------------------------------------------------ plumbing
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def base_model(self) -> Any:
        return cast(Any, self.model).get_base_model()

    @contextlib.contextmanager
    def _autocast(self) -> Iterator[None]:
        if self._autocast_dtype is not None and self._device.type == "cuda":
            with torch.autocast("cuda", dtype=self._autocast_dtype):
                yield
        else:
            yield

    def inputs(
        self,
        obs: PolicyObs,
        instructions: Sequence[str],
        actions: Tensor | None = None,
        *,
        train: bool = False,
    ) -> OFTInputs:
        built = self.pre.build_inputs(
            list(instructions),
            obs["image"].to(self._device),  # crop/jitter/resize on the GPU when there is one
            obs["state"],
            actions,
            train=train,
            generator=self._aug_gen if train else None,
        )
        return built.to(self._device)

    def action_logits(self, inputs: OFTInputs) -> Tensor:
        """(B, C*A, 256) float32."""
        with self._autocast():
            return oft_model.action_logits(self.base_model, inputs, self.pre.num_action_tokens)

    # -------------------------------------------------------------- eval / deploy
    @torch.no_grad()
    def act_batch(self, obs: PolicyObs, instructions: Sequence[str]) -> Tensor:
        logits = self.action_logits(self.inputs(obs, instructions))
        return self.pre.decode_actions(logits.argmax(-1))

    def act(self, obs: Mapping[str, np.ndarray | Tensor], instruction: str) -> np.ndarray:
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
        return self.act_batch(batch, [instruction])[0].cpu().numpy()

    # ------------------------------------------------------------------------ SFT
    def sft_loss(self, batch: SFTBatch, *, augment: bool = True) -> tuple[Tensor, dict[str, float]]:
        """Cross-entropy on the 256-bin action slice (the distribution RL samples from)."""
        inputs = self.inputs(
            {"image": batch.images, "state": batch.states},
            batch.instructions,
            batch.actions,
            train=augment,
        )
        assert inputs.targets is not None
        logits = self.action_logits(inputs)
        loss = F.cross_entropy(logits.reshape(-1, self.n_bins), inputs.targets.reshape(-1))
        with torch.no_grad():
            pred = logits.argmax(-1)
            bins = self.pre.bins
            l1 = (bins.decode(pred) - bins.decode(inputs.targets)).abs().mean()
            metrics = {
                "loss": float(loss.detach()),
                "token_acc": float((pred == inputs.targets).float().mean()),
                "l1_norm": float(l1),
            }
        return loss, metrics

    # ------------------------------------------------------------------------- RL
    @torch.no_grad()
    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> ActionSample:
        logits = self.action_logits(self.inputs(obs, instructions))
        b = logits.shape[0]
        if temperature > 0:
            logp_all = F.log_softmax(logits / temperature, dim=-1)
            flat = torch.multinomial(logp_all.reshape(-1, self.n_bins).exp(), 1)
            tokens = flat.view(b, -1)
        else:
            logp_all = F.log_softmax(logits, dim=-1)
            tokens = logits.argmax(-1)
        log_probs = logp_all.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        shape = (b, self.chunk_len, self.action_dim)
        return ActionSample(
            actions=self.pre.decode_actions(tokens).to(torch.float32),
            tokens=tokens.view(shape),
            log_probs=log_probs.view(shape),
        )

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: Tensor, temperature: float
    ) -> TokenScores:
        if temperature <= 0:
            raise ValueError("log_prob needs the (positive) rollout temperature")
        logits = self.action_logits(self.inputs(obs, instructions))
        logp_all = F.log_softmax(logits / temperature, dim=-1)
        flat = tokens.to(self._device, torch.long).reshape(logits.shape[0], -1)
        log_probs = logp_all.gather(-1, flat.unsqueeze(-1)).squeeze(-1)
        entropy = -(logp_all.exp() * logp_all).sum(-1)
        shape = (logits.shape[0], self.chunk_len, self.action_dim)
        return TokenScores(log_probs=log_probs.view(shape), entropy=entropy.view(shape))

    def trainable_parameters(self) -> list[nn.Parameter]:
        return oft_model.trainable(self.parameters())

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # lm_head carries LoRA, but its base weights are frozen: do not copy them into the adapter
        cast(Any, self.model).save_pretrained(str(path), save_embedding_layers=False)
        stats = norm_stats_dict(self.pre.action_bounds, self.pre.proprio_bounds)
        (path / NORM_STATS_FILE).write_text(json.dumps(stats, indent=2) + "\n")
        (path / POLICY_FILE).write_text(json.dumps(self.describe(), indent=2) + "\n")

    def load_adapter(self, path: Path) -> None:
        """Load adapter weights saved by `save` into this (same-architecture) policy."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(str(Path(path) / "adapter_model.safetensors"))
        result: Any = set_peft_model_state_dict(self.model, state)
        if result.unexpected_keys:
            raise RuntimeError(f"unexpected adapter keys: {result.unexpected_keys[:5]}")

    def describe(self) -> dict[str, Any]:
        return {**self._meta, **self.pre.describe()}


# ----------------------------------------------------------------------------- builders
def resolve_adapter(spec: str | Path | None) -> Path | None:
    """Local adapter dir, or a Hub model repo: `user/repo[@rev]` (adapter at the repo root) or
    `user/repo/path/in/repo[@rev]` (only that folder is downloaded), e.g. a WO-02 checkpoint
    `KudratSingh22/vla-rl-so101-sft/wo02/<pod>/runs/<run>/checkpoints/step_004000`."""
    if spec is None or str(spec) == "":
        return None
    path = Path(spec)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    ref, _, rev = str(spec).partition("@")
    parts = [p for p in ref.split("/") if p]
    if len(parts) < 2:
        raise FileNotFoundError(f"adapter {spec!r} is neither a local dir nor a Hub user/repo id")
    repo, sub = "/".join(parts[:2]), "/".join(parts[2:])
    root = Path(
        snapshot_download(
            repo_id=repo, revision=rev or None, allow_patterns=[f"{sub}/*"] if sub else None
        )
    )
    out = root / sub if sub else root
    if not (out / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no PEFT adapter (adapter_config.json) at {spec!r} ({out})")
    return out


def resolve_bounds(cfg: Config, adapter_dir: Path | None) -> tuple[NormBounds, NormBounds]:
    """Adapter's own norm stats if present (never re-derive for a trained policy), else config."""
    if adapter_dir is not None and (adapter_dir / NORM_STATS_FILE).is_file():
        return bounds_from_norm_stats(json.loads((adapter_dir / NORM_STATS_FILE).read_text()))
    from src.policy.norm_bounds import bounds_from_cfg

    return bounds_from_cfg(cfg)


def build_policy(
    cfg: Config,
    device: str | torch.device,
    *,
    adapter: str | Path | None = None,
    is_trainable: bool = True,
    lora_only: bool = False,
    grad_checkpointing: bool = False,
) -> OpenVLAOFTPolicy:
    """Base VLA + LoRA (fresh, or `adapter` / `policy.adapter` loaded)."""
    from peft import LoraConfig, PeftModel, get_peft_model

    pc = OFTPolicyConfig.from_cfg(cfg)
    oft_model.check_constants(pc.chunk_len, pc.action_dim)
    dev = torch.device(device)
    adapter_dir = resolve_adapter(adapter if adapter is not None else pc.adapter)
    action_b, proprio_b = resolve_bounds(cfg, adapter_dir)
    use_bf16 = pc.precision == "bf16" and dev.type == "cuda"
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model = oft_model.load_model(
        checkpoint=pc.checkpoint,
        revision=pc.revision,
        action_bounds=action_b,
        proprio_bounds=proprio_b,
        dtype=dtype,
        tiny_seed=pc.seed if pc.tiny_random else None,
    )
    if grad_checkpointing:
        oft_model.enable_gradient_checkpointing(model)
    if adapter_dir is not None:
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=is_trainable)
    else:
        lora = LoraConfig(
            r=pc.lora_rank,
            lora_alpha=pc.lora_alpha,
            lora_dropout=pc.lora_dropout,
            target_modules=oft_model.lora_target_modules(model, include_lm_head=pc.lora_lm_head),
            init_lora_weights="gaussian",
            modules_to_save=["proprio_projector"] if pc.train_proprio_projector else None,
        )
        peft_model = get_peft_model(model, lora)
    for name, param in peft_model.named_parameters():
        if "modules_to_save" in name:
            param.data = param.data.float()  # the fully-trained projector copy keeps fp32 weights
            param.requires_grad_(is_trainable and not lora_only)
        elif "lora_" in name and is_trainable:
            param.data = param.data.float()
    peft_model.to(dev)
    tokenizer: PromptTokenizer = (
        ToyPromptTokenizer()
        if pc.tiny_random
        else HFPromptTokenizer(cast(str, pc.checkpoint), pc.revision)
    )
    pre = OFTPreprocessor(
        tokenizer,
        action_bounds=action_b,
        proprio_bounds=proprio_b,
        chunk_len=pc.chunk_len,
        action_dim=pc.action_dim,
        image_size=int(model.config.image_sizes[0]),
        n_towers=len(model.config.image_sizes),
        augment=AugmentConfig(crop_scale=pc.crop_scale),
    )
    meta = {
        "base": "openvla_oft",
        "checkpoint": pc.checkpoint or "tiny-random",
        "revision": pc.revision,
        "adapter": None if adapter_dir is None else str(adapter_dir),
        "lora_rank": pc.lora_rank,
        "lora_alpha": pc.lora_alpha,
        "seed": pc.seed,
    }
    return OpenVLAOFTPolicy(
        peft_model,
        pre,
        device=dev,
        autocast_dtype=torch.bfloat16 if use_bf16 else None,
        meta=meta,
    )


def assert_deterministic(model: nn.Module) -> None:
    """RL contract (src/rl/policy_api.py): `log_prob` in train() mode must equal `sample`'s eval()
    scores, so no dropout / drop-path / attention dropout may be active anywhere (incl. the LoRA
    dropout of a loaded SFT adapter). Otherwise every PPO ratio carries noise."""
    active = []
    for name, mod in model.named_modules():
        p = float(getattr(mod, "p", 0.0)) if isinstance(mod, nn.Dropout) else 0.0
        p = max(p, float(getattr(mod, "drop_prob", 0.0) or 0.0))  # timm DropPath
        attn = getattr(mod, "attention_dropout", 0.0)  # HF attention modules hold a float
        p = max(p, float(attn) if isinstance(attn, (int, float)) else 0.0)
        if p > 0.0:
            active.append(f"{name or type(mod).__name__} (p={p})")
    if active:
        raise ValueError(f"RL needs a deterministic forward; active dropout in: {active[:5]}")


def build_rl_policy(cfg: Config, device: str) -> OpenVLAOFTPolicy:
    """WO-03 builder: SFT adapter (`policy.adapter`) loaded, only LoRA weights require grad."""
    pc = OFTPolicyConfig.from_cfg(cfg)
    if pc.adapter is None and not pc.tiny_random:
        raise ValueError("RL starts from the WO-02 SFT checkpoint: set policy.adapter")
    policy = build_policy(cfg, device, is_trainable=True, lora_only=True)
    assert_deterministic(policy)
    return policy
