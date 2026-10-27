"""Action <-> token mapping for OpenVLA-OFT under SimpleVLA-RL's cross-entropy head.

Semantics follow moojink/openvla-oft @ e4287e9 `prismatic/vla/action_tokenizer.py` and
SimpleVLA-RL @ 7c51662 `modeling_prismatic.py` / `dp_rob.py` (both MIT), re-expressed on token
*indices* instead of decoded strings:

* A normalised action value a in [-1, 1] is digitised against 256 uniform bin EDGES
  (`np.linspace(-1, 1, 256)`), d = digitize(a) in [1, 256]; upstream's vocab id is 32000 - d.
* The policy only ever scores the 256 vocab rows 31744..31999 (the logit slice
  `[..., -320:-64]` of the 64-padded 32064 vocab). A *token* here is the index j into that slice:
  j = vocab_id - 31744 = 256 - d, so j in [0, 255].
* Decoding: upstream maps vocab id -> bin_centers[clip(32000 - id - 1, 0, 254)], i.e.
  value(j) = bin_centers[clip(255 - j, 0, 254)] with 255 bin centres.

Normalisation is OFT's "bounds" scheme: x -> clip(2 (x - lo) / (hi - lo + 1e-8) - 1, -1, 1), with
dimensions where lo == hi mapped to 0 (upstream data_utils.normalize_action_and_proprio).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

N_BINS = 256
LLAMA_VOCAB_SIZE = 32000  # tokenizer.vocab_size; the model's embedding is padded to 32064
ACTION_VOCAB_BEGIN = LLAMA_VOCAB_SIZE - N_BINS  # 31744: vocab id of token index 0
LOGIT_SLICE = slice(-N_BINS - 64, -64)  # the 256 action rows of the 32064-row lm_head
NORM_EPS = 1e-8
EMBODIMENT_KEY = "so101"  # key in OFT-style norm_stats / dataset_statistics.json


class ActionBins:
    """256-edge uniform binning of [-1, 1] with OFT's token-index conventions."""

    def __init__(self, n_bins: int = N_BINS) -> None:
        self.n_bins = n_bins
        self.edges = torch.linspace(-1.0, 1.0, n_bins, dtype=torch.float64)
        self.centers = ((self.edges[:-1] + self.edges[1:]) / 2.0).float()

    def encode(self, a_norm: Tensor) -> Tensor:
        """Normalised actions (any shape) -> token indices in [0, n_bins), int64."""
        a = a_norm.detach().to(torch.float64).clamp(-1.0, 1.0)
        d = torch.bucketize(a, self.edges.to(a.device), right=True)  # == np.digitize, in [1, 256]
        return (-d + self.n_bins).long()

    def decode(self, tokens: Tensor) -> Tensor:
        """Token indices -> normalised action values (bin centres), float32."""
        idx = (self.n_bins - 1 - tokens.long()).clamp(0, self.n_bins - 2)
        return self.centers.to(tokens.device)[idx]


@dataclass(frozen=True)
class NormBounds:
    """Per-dimension [low, high] used to map env units (radians) to [-1, 1]."""

    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) != len(self.high):
            raise ValueError("low/high length mismatch")
        if any(h < lo for lo, h in zip(self.low, self.high, strict=True)):
            raise ValueError(f"high < low in bounds {self}")

    @property
    def dim(self) -> int:
        return len(self.low)

    def _lo_hi(self, ref: Tensor) -> tuple[Tensor, Tensor]:
        lo = torch.tensor(self.low, dtype=torch.float32, device=ref.device)
        hi = torch.tensor(self.high, dtype=torch.float32, device=ref.device)
        return lo, hi

    def normalize(self, x: Tensor) -> Tensor:
        lo, hi = self._lo_hi(x)
        y = 2.0 * (x.float() - lo) / (hi - lo + NORM_EPS) - 1.0
        return torch.where(hi == lo, torch.zeros_like(y), y.clamp(-1.0, 1.0))

    def unnormalize(self, y: Tensor) -> Tensor:
        lo, hi = self._lo_hi(y)
        return 0.5 * (y.float() + 1.0) * (hi - lo + NORM_EPS) + lo

    def deg_per_bin(self, n_bins: int = N_BINS) -> list[float]:
        return [
            float(torch.rad2deg(torch.tensor((h - lo) / (n_bins - 1))))
            for lo, h in zip(self.low, self.high, strict=True)
        ]

    @classmethod
    def from_lists(cls, low: Sequence[float], high: Sequence[float]) -> NormBounds:
        return cls(tuple(float(v) for v in low), tuple(float(v) for v in high))


def norm_stats_dict(action: NormBounds, proprio: NormBounds) -> dict[str, Any]:
    """OFT `dataset_statistics.json` layout ({key: {"action": {...}, "proprio": {...}}})."""

    def block(b: NormBounds, with_mask: bool) -> dict[str, Any]:
        out: dict[str, Any] = {
            "min": list(b.low),
            "max": list(b.high),
            "q01": list(b.low),
            "q99": list(b.high),
        }
        if with_mask:
            out["mask"] = [True] * b.dim
        return out

    return {EMBODIMENT_KEY: {"action": block(action, True), "proprio": block(proprio, False)}}


def bounds_from_norm_stats(stats: dict[str, Any]) -> tuple[NormBounds, NormBounds]:
    entry = stats[EMBODIMENT_KEY]
    return (
        NormBounds.from_lists(entry["action"]["min"], entry["action"]["max"]),
        NormBounds.from_lists(entry["proprio"]["min"], entry["proprio"]["max"]),
    )
