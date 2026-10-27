"""THE input builder for OpenVLA-OFT: one `build_inputs()` used by SFT, eval and RL alike.

Token layout (LAYOUT_VERSION), per sample, right-padded with <PAD> (32000):

    <s> In: What action should the robot take to {instruction}?\\nOut: ▁ [K*D action slots] </s>

* `▁` is token 29871. OFT's training transform produces it (the prompt ends "Out: " and the action
  characters follow), and OFT's `predict_action` re-inserts it at inference; SimpleVLA-RL's rollout
  drops it (its insertion is commented out). We keep it, everywhere.
* Action slots hold a placeholder id; the model zeroes their embeddings (upstream does the same),
  so SFT targets never leak into the input.
* Text index of the first action slot = `action_start` = len(prompt ids incl. <s> and ▁).

Images: (B, H, W, 3) uint8 (env `policy_obs`) or (B, 3, H, W) float in [0, 1] (LeRobotDataset).
Crop to 90% area (centre at eval/RL, random position + colour jitter in SFT, as OFT's
`image_aug` / SimpleVLA-RL's `center_crop`), bicubic-resize to the backbone size, then normalise
once per vision tower (DINOv2 ImageNet stats, SigLIP 0.5/0.5) and stack channels -> (B, 6, S, S).
Deterministic unless `train=True`.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast

import torch
import torchvision.transforms.v2.functional as TF
from torch import Tensor

from src.policy.oft.actions import ACTION_VOCAB_BEGIN, ActionBins, NormBounds

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"
LAYOUT_VERSION = "oft-pd-v1:<s>prompt▁[K*D slots]</s>,right-pad"
BOS_ID = 1
STOP_ID = 2  # </s>
PAD_ID = 32000
SPACE_ID = 29871  # "▁"
SLOT_ID = ACTION_VOCAB_BEGIN  # any id works: slot embeddings are zeroed

# openvla/openvla-7b preprocessor_config.json `means` / `stds` (DINOv2, then SigLIP). These are what
# PrismaticImageProcessor applies (its stored tvf_normalize_params are bf16-rounded copies).
IMAGE_MEANS: tuple[tuple[float, float, float], ...] = ((0.485, 0.456, 0.406), (0.5, 0.5, 0.5))
IMAGE_STDS: tuple[tuple[float, float, float], ...] = ((0.229, 0.224, 0.225), (0.5, 0.5, 0.5))


class PromptTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        """Token ids of `text` with the leading <s> (id 1), no </s>."""
        ...


class HFPromptTokenizer:
    """The checkpoint's Llama-2 tokenizer (tokenizer.json; no sentencepiece needed)."""

    def __init__(self, checkpoint: str, revision: str | None = None) -> None:
        from transformers import AutoTokenizer

        auto: Any = AutoTokenizer
        self.tok = auto.from_pretrained(checkpoint, revision=revision)
        if self.tok.pad_token_id != PAD_ID or self.tok.bos_token_id != BOS_ID:
            raise ValueError(f"{checkpoint}: unexpected special tokens for the OFT layout")

    def encode(self, text: str) -> list[int]:
        return [int(i) for i in self.tok(text, add_special_tokens=True).input_ids]


class ToyPromptTokenizer:
    """Deterministic word-hash tokenizer for the tiny random model (CPU tests, `tiny` preset)."""

    def encode(self, text: str) -> list[int]:
        words = text.replace("\n", " \n ").split(" ")
        return [BOS_ID] + [3 + zlib.crc32(w.encode()) % 29000 for w in words if w]


@dataclass(frozen=True)
class OFTInputs:
    input_ids: Tensor  # (B, L) int64, right-padded with PAD_ID
    attention_mask: Tensor  # (B, L) int64
    action_mask: Tensor  # (B, L) bool, True on the K*D action slots
    action_start: Tensor  # (B,) int64, text index of the first action slot
    pixel_values: Tensor  # (B, 3 * n_towers, S, S) float32
    proprio: Tensor  # (B, P) float32, normalised to [-1, 1]
    targets: Tensor | None  # (B, K*D) int64 action tokens (SFT), chunk-major: c * D + d

    def to(self, device: torch.device | str) -> OFTInputs:
        return replace(
            self,
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            action_mask=self.action_mask.to(device),
            action_start=self.action_start.to(device),
            pixel_values=self.pixel_values.to(device),
            proprio=self.proprio.to(device),
            targets=None if self.targets is None else self.targets.to(device),
        )


@dataclass(frozen=True)
class AugmentConfig:
    crop_scale: float = 0.9  # fraction of image AREA kept by the (random | centre) square crop
    brightness: float = 0.2  # additive delta, OFT/dlimp random_brightness
    contrast: tuple[float, float] = (0.8, 1.2)
    saturation: tuple[float, float] = (0.8, 1.2)
    hue: float = 0.05


class OFTPreprocessor:
    def __init__(
        self,
        tokenizer: PromptTokenizer,
        *,
        action_bounds: NormBounds,
        proprio_bounds: NormBounds,
        chunk_len: int,
        action_dim: int,
        image_size: int,
        n_towers: int = 2,
        augment: AugmentConfig | None = None,
    ) -> None:
        if action_bounds.dim != action_dim or proprio_bounds.dim != action_dim:
            raise ValueError("bounds dimension must equal action_dim")
        self.tokenizer = tokenizer
        self.action_bounds = action_bounds
        self.proprio_bounds = proprio_bounds
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self.image_size = image_size
        self.n_towers = n_towers
        self.augment = augment or AugmentConfig()
        self.bins = ActionBins()
        self._prompt_cache: dict[str, list[int]] = {}

    @property
    def num_action_tokens(self) -> int:
        return self.chunk_len * self.action_dim

    def prompt_ids(self, instruction: str) -> list[int]:
        """<s> + prompt + ▁ (the text preceding the action slots)."""
        key = instruction.strip().lower()
        if key not in self._prompt_cache:
            ids = self.tokenizer.encode(PROMPT_TEMPLATE.format(instruction=key))
            self._prompt_cache[key] = ids + [SPACE_ID]
        return self._prompt_cache[key]

    # ---------------------------------------------------------------- text
    def _text(self, instructions: Sequence[str]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n_act = self.num_action_tokens
        rows = []
        starts = []
        for instr in instructions:
            p = self.prompt_ids(instr)
            starts.append(len(p))
            rows.append(p + [SLOT_ID] * n_act + [STOP_ID])
        length = max(len(r) for r in rows)
        ids = torch.full((len(rows), length), PAD_ID, dtype=torch.long)
        attn = torch.zeros((len(rows), length), dtype=torch.long)
        act = torch.zeros((len(rows), length), dtype=torch.bool)
        for i, (r, s) in enumerate(zip(rows, starts, strict=True)):
            ids[i, : len(r)] = torch.tensor(r)
            attn[i, : len(r)] = 1
            act[i, s : s + n_act] = True
        return ids, attn, act, torch.tensor(starts, dtype=torch.long)

    # -------------------------------------------------------------- images
    def _as_float_chw(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"images must be 4-D, got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            x = images.permute(0, 3, 1, 2) if images.shape[-1] == 3 else images
            return x.float() / 255.0
        x = images.float()
        return x.permute(0, 3, 1, 2) if x.shape[-1] == 3 and x.shape[1] != 3 else x

    def _jitter(self, x: Tensor, g: torch.Generator) -> Tensor:
        a = self.augment

        def u(lo: float, hi: float) -> float:
            return float(lo + (hi - lo) * torch.rand((), generator=g))

        x = (x + u(-a.brightness, a.brightness)).clamp(0.0, 1.0)
        x = TF.adjust_contrast(x, u(*a.contrast))
        x = TF.adjust_saturation(x, u(*a.saturation))
        return cast(Tensor, TF.adjust_hue(x, u(-a.hue, a.hue))).clamp(0.0, 1.0)

    def _resize_normalize(self, x: Tensor) -> Tensor:
        x = TF.resize(
            x,
            [self.image_size, self.image_size],
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        ).clamp(0.0, 1.0)
        towers = [
            TF.normalize(x, list(m), list(s))
            for m, s in zip(IMAGE_MEANS[: self.n_towers], IMAGE_STDS[: self.n_towers], strict=True)
        ]
        return torch.cat(towers, dim=-3)

    def pixel_values(
        self, images: Tensor, *, train: bool = False, generator: torch.Generator | None = None
    ) -> Tensor:
        """(B, 3 * n_towers, S, S) float32 on the images' device."""
        x = self._as_float_chw(images)
        _, _, h, w = x.shape
        side = math.sqrt(self.augment.crop_scale)
        ch, cw = max(1, round(h * side)), max(1, round(w * side))
        if not train:  # deterministic centre crop, whole batch at once
            top, left = (h - ch) // 2, (w - cw) // 2
            return self._resize_normalize(x[:, :, top : top + ch, left : left + cw])
        g = generator if generator is not None else torch.Generator().manual_seed(0)
        out = []
        for i in range(x.shape[0]):
            top = int(torch.randint(0, h - ch + 1, (), generator=g))
            left = int(torch.randint(0, w - cw + 1, (), generator=g))
            xi = self._jitter(x[i, :, top : top + ch, left : left + cw], g)
            out.append(self._resize_normalize(xi))
        return torch.stack(out)

    # ---------------------------------------------------------------- main
    def build_inputs(
        self,
        instructions: Sequence[str],
        images: Tensor,
        states: Tensor,
        actions: Tensor | None = None,
        *,
        train: bool = False,
        generator: torch.Generator | None = None,
    ) -> OFTInputs:
        """Batch of B samples. `states` (B, D) and `actions` (B, K, D) are in env units (radians)."""
        b = len(instructions)
        if images.shape[0] != b or states.shape[0] != b:
            raise ValueError("instructions / images / states batch sizes differ")
        ids, attn, act, starts = self._text(instructions)
        targets = None
        if actions is not None:
            if tuple(actions.shape[1:]) != (self.chunk_len, self.action_dim):
                raise ValueError(f"actions must be (B, {self.chunk_len}, {self.action_dim})")
            a_norm = self.action_bounds.normalize(actions.detach().cpu().float())
            targets = self.bins.encode(a_norm).reshape(b, self.num_action_tokens)
        return OFTInputs(
            input_ids=ids,
            attention_mask=attn,
            action_mask=act,
            action_start=starts,
            pixel_values=self.pixel_values(images.detach(), train=train, generator=generator),
            proprio=self.proprio_bounds.normalize(states.detach().cpu().float()),
            targets=targets,
        )

    def decode_actions(self, tokens: Tensor) -> Tensor:
        """(B, K*D) or (B, K, D) tokens -> (B, K, D) actions in env units."""
        a_norm = self.bins.decode(tokens.reshape(tokens.shape[0], self.chunk_len, self.action_dim))
        return self.action_bounds.unnormalize(a_norm)

    def describe(self) -> dict[str, Any]:
        return {
            "layout": LAYOUT_VERSION,
            "prompt_template": PROMPT_TEMPLATE,
            "chunk_len": self.chunk_len,
            "action_dim": self.action_dim,
            "image_size": self.image_size,
            "crop_scale": self.augment.crop_scale,
            "n_bins": self.bins.n_bins,
        }
