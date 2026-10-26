"""Preset/config loading shared by every script: `--config` and `--preset {tiny,sim,rl,real}`."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
PRESETS = ("tiny", "sim", "rl", "real")

Config = dict[str, Any]


def _deep_merge(base: Config, override: Config) -> Config:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    parent = raw.pop("inherit", None)
    if parent is None:
        return raw
    return _deep_merge(_load_yaml(REPO_ROOT / parent), raw)


def load_config(preset: str, config: str | Path | None = None) -> Config:
    """Load `configs/<preset>.yaml` (or an explicit `config` file), resolving `inherit:` chains."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
    path = Path(config) if config is not None else CONFIG_DIR / f"{preset}.yaml"
    cfg = _load_yaml(path)
    cfg["preset"] = preset
    return cfg


def config_hash(cfg: Config) -> str:
    """SHA-256 of the canonical JSON form; cited in run records."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=PRESETS, required=True)
    parser.add_argument("--config", type=Path, default=None, help="override the preset's file")
