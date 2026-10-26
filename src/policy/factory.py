"""`make_policy(cfg, device)`: the VLA policy named by `policy.base` (SFT / eval entry point).

openvla_oft  ADR-0001 default (src/policy/openvla_oft.py)
pi05         ADR-0001 fallback, stub (src/policy/pi05.py)
smolvla      WO-01 data-path smoke only (scripts/wo01_smolvla_smoke.py); not a VLA-RL policy
toy          WO-03's RL stand-in; built by src.rl.policy_api, not here
RL builds its policy through src.rl.policy_api.POLICY_BUILDERS, which points at
`src.policy.openvla_oft:build_rl_policy` / `src.policy.pi05:build_rl_policy`.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config
from src.policy.base import VLAPolicy


def make_policy(
    cfg: Config,
    device: str,
    *,
    adapter: str | Path | None = None,
    is_trainable: bool = False,
    grad_checkpointing: bool = False,
) -> VLAPolicy:
    base = str(cfg["policy"]["base"])
    if base == "openvla_oft":
        from src.policy.openvla_oft import build_policy

        return build_policy(
            cfg,
            device,
            adapter=adapter,
            is_trainable=is_trainable,
            grad_checkpointing=grad_checkpointing,
        )
    if base == "pi05":
        from src.policy import pi05

        pi05.build_policy(cfg, device)
    if base == "smolvla":
        raise NotImplementedError(
            "smolvla is the WO-01 data-path smoke model (scripts/wo01_smolvla_smoke.py), "
            "not a WO-02/RL policy"
        )
    raise KeyError(f"unknown policy.base={base!r} for SFT/eval; expected openvla_oft | pi05")
