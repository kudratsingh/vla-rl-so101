"""OpenVLA-OFT under SimpleVLA-RL's cross-entropy action head (ADR-0001).

actions.py     256-bin token <-> action mapping, bounds normalisation
preprocess.py  THE input builder (prompt/token layout, images, proprio) for SFT, eval and RL
model.py       vendored-model loading, bidirectional-attention hook, the single action forward
vendor/        upstream SimpleVLA-RL @ 7c51662 model code (MIT), minimal marked edits
"""
