# Rewritten for vla-rl-so101 from PRIME-RL/SimpleVLA-RL @ 7c51662df27b586f9e8a1ab35fcf849f2b8852f9
#   verl/utils/vla_utils/openvla_oft/constants.py (MIT License, Copyright (c) 2025 PRIME-RL;
#   LICENSE-SimpleVLA-RL), itself from moojink/openvla-oft prismatic/vla/constants.py (MIT).
# [vla-rl-so101] Upstream picks the robot platform at IMPORT time by sniffing sys.argv / the
# ROBOT_PLATFORM env var and silently falls back to ALOHA (14-dim, chunk 25). We have one
# embodiment, so the SO-101 values are fixed here; src/policy/oft/model.py asserts the model
# config agrees (`policy.action_dim`, `policy.action_chunk`).
"""Token / layout constants for OpenVLA-OFT on the SO-101 (6 absolute joint targets, chunk 8)."""

from enum import Enum

# Llama 2 token constants (unchanged from upstream)
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


class NormalizationType(str, Enum):
    NORMAL = "normal"  # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"  # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"  # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]


# SO-101: absolute joint position targets (radians) for the 6 joints in src.env.JOINT_NAMES,
# min/max ("bounds") normalisation as upstream uses for absolute joint actions (ALOHA/RoboTwin2).
NUM_ACTIONS_CHUNK = 8
ACTION_DIM = 6
PROPRIO_DIM = 6
ACTION_PROPRIO_NORMALIZATION_TYPE = NormalizationType.BOUNDS
