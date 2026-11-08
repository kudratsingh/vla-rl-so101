"""WO-01r guards that need no renderer: the render-gap comparison label and the protected-repo
push guard on the wo01r push path (every push goes through src.data.recorder.push_dataset)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.config import REPO_ROOT


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_render_gap_comparison_labels() -> None:
    """A Mac CPU run on a 0.1 set (both sides the Mac CPU renderer) is a same-renderer pipeline
    check, not a Mac-vs-GPU gap; a GPU run on the 0.1 set is the renderer gap."""
    from src.config import CONFIG_DIR, load_config

    rebuild = _load("wo01r_rebuild_demos")
    cfg = load_config("sim", CONFIG_DIR / "demos_v2.yaml")
    old, new = cfg["render_gap"]["dataset"], cfg["data"]["hub_repo"]
    label = rebuild.comparison_label
    assert label(cfg, old, "render-gap", "cpu").startswith("same-renderer pipeline check")
    assert "NOT a Mac-vs-GPU" in label(cfg, old, "render-gap", "cpu")
    assert label(cfg, old, "render-gap", "cuda").startswith("renderer gap")
    assert label(cfg, new, "render-gap", "cuda").startswith("same-renderer pipeline check")
    assert label(cfg, new, "render-gap", "cpu").startswith("renderer gap")
    assert label(cfg, new, "dry-run", "cpu").startswith("same-renderer pipeline check")


def test_wo01r_push_path_refuses_the_protected_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`wo01r_rebuild_demos.py --push` to a 0.1 repo stops before the build, and push_dataset
    itself (every push path) refuses it before any token or network use."""
    from src.data import recorder

    rebuild = _load("wo01r_rebuild_demos")
    built: list[Any] = []
    monkeypatch.setattr(rebuild, "_build_env_module", lambda: built.append(1))
    cfg = tmp_path / "protected.yaml"
    for repo in recorder.PROTECTED_HUB_REPOS:
        cfg.write_text(f"inherit: configs/demos_v2.yaml\ndata:\n  hub_repo: {repo}\n")
        with pytest.raises(SystemExit):
            rebuild.main(["--preset", "sim", "--config", str(cfg), "--mode", "build", "--push"])
        with pytest.raises(recorder.ProtectedRepoError, match="refusing to push"):
            recorder.push_dataset(repo, tmp_path / "nowhere")
    assert built == []  # nothing was built (or pushed)

    def no_token() -> None:
        raise AssertionError("a protected push must fail before the token check")

    monkeypatch.setattr(recorder, "_ensure_hf_token", no_token)
    with pytest.raises(recorder.ProtectedRepoError):
        recorder.push_dataset(recorder.PROTECTED_HUB_REPOS[1], tmp_path / "nowhere")
    recorder.check_push_target("KudratSingh22/vla-rl-so101-demos-25-v2")  # the v2 repo is fine
    recorder.check_push_target(recorder.PROTECTED_HUB_REPOS[0], allow_protected=True)
