"""π0.5 (openpi LoRA, RLinf πRL path): ADR-0001's fallback, NOT implemented.

Switch only if OpenVLA-OFT few-shot SFT stays below 20% on seen positions after 3 runs. openpi @
215abfb pins torch 2.7.1, jax 0.5.3, transformers 4.53.2 (with site-packages file overwrites),
numpy<2, tensorflow-cpu 2.15 and a git lerobot -- all conflicting with this repo's pins -- and its
PyTorch port does not support LoRA, so adopting it is its own work order (docs/adr/0001).
"""

from __future__ import annotations

from typing import NoReturn

from src.config import Config

REASON = "pi05 is ADR-0001's fallback and is not implemented (see src/policy/pi05.py)"


def build_policy(cfg: Config, device: str) -> NoReturn:
    raise NotImplementedError(REASON)


def build_rl_policy(cfg: Config, device: str) -> NoReturn:
    raise NotImplementedError(REASON)
