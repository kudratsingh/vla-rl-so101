"""Loading and running the vendored OpenVLA-OFT model on stock transformers 4.57.6.

E2 (bidirectional attention): OFT's parallel decoding needs every token to attend to every other
non-pad token. The moojink/transformers-openvla-oft fork (v4.40.1 + bc339d9) patched that into
`LlamaSdpaAttention`; here a forward pre-hook on the Llama backbone turns the 2-D padding mask into
a 4-D boolean (B, 1, L, L) "attend to all non-pad keys" mask, which transformers 4.57 passes to
SDPA unchanged. Verified bit-identical to the fork (docs/adr/0001-base-model.md). The hook only
works with `attn_implementation="sdpa"` (the fork's patch, too, left flash/eager causal), so
loading asserts it.

`action_logits` is the single forward used by SFT loss, greedy eval, RL sampling and RL scoring.
It runs the LLM backbone, gathers the K*D action-slot positions per sample (prompts may have
different lengths; right padding) and applies lm_head only there, returning the 256-wide action
slice. Parity with upstream's `generate_action_verl` / `forward` is a test (tests/test_oft_policy).
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.policy.oft.actions import LOGIT_SLICE, N_BINS, NormBounds, norm_stats_dict
from src.policy.oft.preprocess import OFTInputs
from src.policy.oft.vendor import configuration_prismatic as _cfg_mod
from src.policy.oft.vendor import constants as _const
from src.policy.oft.vendor import modeling_prismatic as _model_mod

OpenVLAConfig: Any = _cfg_mod.OpenVLAConfig
OpenVLAForActionPrediction: Any = _model_mod.OpenVLAForActionPrediction

ATTN_IMPLEMENTATION = "sdpa"
TINY_BACKBONE = "dinosiglip-tiny-test"
_HOOK_FLAG = "_vla_rl_bidirectional_hook"


def check_constants(chunk_len: int, action_dim: int) -> None:
    """The vendored code bakes chunk/dim in as module constants; refuse a mismatching config."""
    if (_const.NUM_ACTIONS_CHUNK, _const.ACTION_DIM) != (chunk_len, action_dim):
        raise ValueError(
            f"policy.action_chunk/action_dim = {chunk_len}/{action_dim} but the vendored OFT "
            f"constants are {_const.NUM_ACTIONS_CHUNK}/{_const.ACTION_DIM} "
            "(src/policy/oft/vendor/constants.py)"
        )


def _bidirectional_mask_hook(
    module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    emb = kwargs.get("inputs_embeds")
    if emb is None:
        raise ValueError("OFT's language model must be called with inputs_embeds")
    b, length = emb.shape[:2]
    am = kwargs.get("attention_mask")
    if am is None:
        am = torch.ones(b, length, dtype=torch.bool, device=emb.device)
    if am.ndim == 2:
        kwargs["attention_mask"] = am.bool()[:, None, None, :].expand(b, 1, length, length)
    return args, kwargs


def attach_bidirectional_attention(model: Any) -> None:
    """Assert SDPA and register the 4-D bidirectional-mask hook on the Llama backbone (once)."""
    lm = model.language_model
    impl = lm.config._attn_implementation
    if impl != ATTN_IMPLEMENTATION:
        raise RuntimeError(
            f"OpenVLA-OFT needs attn_implementation='sdpa' for bidirectional attention; got {impl!r}"
        )
    backbone = lm.model
    if getattr(backbone, _HOOK_FLAG, False):
        return
    backbone.register_forward_pre_hook(_bidirectional_mask_hook, with_kwargs=True)
    setattr(backbone, _HOOK_FLAG, True)


def tiny_config(action_bounds: NormBounds, proprio_bounds: NormBounds) -> Any:
    """A randomly initialised OFT small enough for CPU tests (vocab layout kept: 32064 rows)."""
    text_config: dict[str, Any] = {
        "vocab_size": 32064,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 512,
        "pad_token_id": 32000,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "torch_dtype": "float32",
    }
    cfg = OpenVLAConfig(
        vision_backbone_id=TINY_BACKBONE,
        llm_backbone_id="llama2-7b-pure",
        arch_specifier="no-align+fused-gelu-mlp",
        image_resize_strategy="resize-naive",
        text_config=text_config,
        n_action_bins=N_BINS,
        norm_stats=norm_stats_dict(action_bounds, proprio_bounds),
        pad_token_id=32000,
        pad_to_multiple_of=64,
        use_proprio=True,
        proprio_dim=proprio_bounds.dim,
    )
    cfg._attn_implementation = ATTN_IMPLEMENTATION
    return cfg


def load_model(
    *,
    checkpoint: str | None,
    revision: str | None,
    action_bounds: NormBounds,
    proprio_bounds: NormBounds,
    dtype: torch.dtype,
    tiny_seed: int | None = None,
) -> Any:
    """Base OFT model with a (new) proprio projector, bidirectional attention attached.

    `checkpoint=None` builds the tiny random model (seeded by `tiny_seed`).
    """
    if checkpoint is None:
        if tiny_seed is not None:
            torch.manual_seed(tiny_seed)
        model = OpenVLAForActionPrediction(tiny_config(action_bounds, proprio_bounds))
        model = model.to(dtype)
    else:
        config = OpenVLAConfig.from_pretrained(checkpoint, revision=revision)
        config.use_proprio = True
        config.proprio_dim = proprio_bounds.dim
        config.norm_stats = norm_stats_dict(action_bounds, proprio_bounds)
        model = OpenVLAForActionPrediction.from_pretrained(
            checkpoint,
            revision=revision,
            config=config,
            torch_dtype=dtype,
            attn_implementation=ATTN_IMPLEMENTATION,
        )
    model.vision_backbone.set_num_images_in_input(1)
    model.norm_stats = norm_stats_dict(action_bounds, proprio_bounds)
    attach_bidirectional_attention(model)
    return model


def num_prefix_tokens(model: Any, use_proprio: bool = True) -> int:
    """Multimodal tokens inserted after <s>: image patches (+1 proprio token)."""
    vb = model.vision_backbone
    return int(vb.get_num_patches() * vb.get_num_images_in_input()) + int(use_proprio)


def action_logits(model: Any, inputs: OFTInputs, num_action_tokens: int) -> Tensor:
    """(B, K*D, 256) float32 logits of the action slots (the slice OFT samples/scores)."""
    ids = inputs.input_ids
    emb = model.get_input_embeddings()(ids)
    emb = emb * (~inputs.action_mask).unsqueeze(-1).to(emb.dtype)
    patches = model.projector(model.vision_backbone(inputs.pixel_values.to(emb.dtype)))
    proprio = model.proprio_projector(inputs.proprio.to(emb.dtype)).unsqueeze(1)
    prefix = torch.cat([patches, proprio], dim=1)
    n_prefix = prefix.shape[1]
    embeds = torch.cat([emb[:, :1], prefix, emb[:, 1:]], dim=1)
    am = inputs.attention_mask
    mask = torch.cat([am[:, :1], am.new_ones(am.shape[0], n_prefix), am[:, 1:]], dim=1)
    lm = model.language_model
    hidden = lm.model(inputs_embeds=embeds, attention_mask=mask, use_cache=False).last_hidden_state
    # slot k of sample i sits at text index start_i + k, i.e. multimodal index n_prefix + start_i
    # + k; with the next-token convention it is predicted from the position before it.
    pos = n_prefix + inputs.action_start.unsqueeze(1) - 1
    pos = pos + torch.arange(num_action_tokens, device=pos.device).unsqueeze(0)
    h = hidden.gather(1, pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    return cast(Tensor, lm.lm_head(h))[..., LOGIT_SLICE].float()


def enable_gradient_checkpointing(model: Any) -> None:
    """Checkpoint every Llama decoder layer AND every ViT block.

    Prismatic runs its timm towers through `get_intermediate_layers`, which loops over the blocks
    directly, so timm's own `set_grad_checkpointing` would have no effect; wrap each block's
    forward instead (the module tree, state-dict keys and LoRA layers are unchanged).
    """
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    vb = model.vision_backbone
    towers = [vb.featurizer] + ([vb.fused_featurizer] if vb.use_fused_vision_backbone else [])
    for tower in towers:
        for blk in tower.blocks:
            if getattr(blk, "_vla_rl_ckpt", False):
                continue
            orig = blk.forward

            def ckpt_forward(x: Tensor, _orig: Any = orig, _blk: nn.Module = blk) -> Tensor:
                if _blk.training and torch.is_grad_enabled():
                    return cast(Tensor, checkpoint(_orig, x, use_reentrant=False))
                return cast(Tensor, _orig(x))

            blk.forward = functools.partial(ckpt_forward)
            blk._vla_rl_ckpt = True


def lora_target_modules(model: nn.Module, *, include_lm_head: bool) -> list[str]:
    """Every nn.Linear except the proprio projector (trained in full) -- OFT's "all-linear"
    (which in peft 0.11.1 also covered lm_head; peft 0.21 excludes it, so it is explicit here)."""
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name.startswith("proprio_projector"):
            continue
        if name.endswith("lm_head") and not include_lm_head:
            continue
        names.append(name)
    return names


def trainable(params: Iterable[nn.Parameter]) -> list[nn.Parameter]:
    return [p for p in params if p.requires_grad]
