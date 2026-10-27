"""Upstream OpenVLA-OFT model code as vendored by SimpleVLA-RL (see each file's header).

Not linted or type-checked (pyproject excludes it). Local edits are marked "[vla-rl-so101]":
  E1  modeling_prismatic.py      `_supports_sdpa` property -> class attribute (transformers 4.57)
  E3  configuration_prismatic.py tiny random vision backbone for CPU tests
  constants.py                   rewritten: fixed SO-101 values instead of argv/env sniffing
The bidirectional attention that the moojink/transformers-openvla-oft fork patched into
LlamaSdpaAttention is applied by src/policy/oft/model.py as a 4D-mask hook (E2), not here.
"""
