.PHONY: setup check smoke tiny

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
