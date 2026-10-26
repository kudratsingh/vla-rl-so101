.PHONY: setup check smoke tiny

# Mac: Vulkan via MoltenVK for SAPIEN, FFmpeg 7 for torchcodec (see scripts/mac_env.sh)
ifeq ($(shell uname),Darwin)
export VK_ICD_FILENAMES ?= /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export DYLD_FALLBACK_LIBRARY_PATH ?= /opt/homebrew/opt/ffmpeg@7/lib
export MVK_CONFIG_LOG_LEVEL ?= 0
endif

setup:
	uv sync --extra dev

check:
	uv run ruff check src tests scripts
	uv run black --check src tests scripts
	uv run mypy

smoke:
	uv run pytest -q

tiny:    ## full pipeline on CPU in <2 min: env → oracle → sft → rl → eval
	uv run scripts/wo01_build_env.py --preset tiny
	uv run scripts/wo02_sft.py --preset tiny
	uv run scripts/wo03_rl.py --preset tiny
	uv run scripts/wo02_eval.py --preset tiny
