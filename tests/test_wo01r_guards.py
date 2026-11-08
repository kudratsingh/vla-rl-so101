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


class _Env:
    """One env: success from the 3rd step on; obs state = step count."""

    def __init__(self) -> None:
        self.t = 0
        self.num_envs = 1

    @property
    def unwrapped(self) -> Any:
        return self

    def get_instructions(self) -> list[str]:
        return ["pick"]

    def reset(self, *, seed: Any = None) -> tuple[Any, dict[str, Any]]:
        self.t = 0
        self.seed = int(seed)
        return {"t": self.t}, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        import torch

        self.t += 1
        return (
            {"t": self.t},
            0.0,
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            {"success": torch.tensor([self.t >= 3])},
        )

    def close(self) -> None:
        pass


class _Oracle:
    """Finishes its script at step 5, except on BAD seeds, where the release never completes
    (the recording hits post_success_max_steps with the gripper short of the release target)."""

    def __init__(self, env: _Env, bad: set[int]) -> None:
        self.env, self.bad = env, bad

    def reset(self, obs: Any) -> None:
        pass

    def _ok(self) -> bool:
        return self.env.seed not in self.bad

    def act(self, obs: Any) -> Any:
        import torch

        from src.data.oracle import GRIPPER_RELEASE

        grip = GRIPPER_RELEASE if self._ok() and self.env.t >= 3 else 0.25
        return torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, grip]])

    def finished(self) -> Any:
        import torch

        return torch.tensor([self._ok() and self.env.t >= 5])


class _Recorder:
    def __init__(self) -> None:
        self.episodes: list[dict[str, Any]] = []
        self.frames = 0

    def get(self, hw: Any) -> _Recorder:
        return self

    def add_frame(self, *a: Any) -> None:
        self.frames += 1

    def save_episode(self, meta: dict[str, Any]) -> None:
        self.episodes.append(meta)


def test_unfinished_demos_are_rejected_individually_and_replaced() -> None:
    """Seeds 0 and 2 succeed but never finish release/retreat: each is dropped and reported on
    its own; the next seeds are recorded instead, so the set still has 3 valid demos."""
    import torch

    build = _load("wo01_build_env")
    seed0 = build.DEMO_SEED_BASE
    bad = {seed0, seed0 + 2}
    env = _Env()

    def make_env(cfg: Any, task: str, **kw: Any) -> _Env:
        return env

    def policy_obs(obs: Any) -> dict[str, Any]:
        return {"image": torch.zeros(1, 4, 4, 3), "state": torch.full((1, 6), float(obs["t"]))}

    cfg = {
        "env": {"max_episode_steps": 10, "image_size": 4},
        "data": {
            "post_success_steps": 2,
            "post_success_max_steps": 4,
            "require_final_success": True,
            "require_tail_finished": True,
            "oracle_profile": "v2",
        },
        "seed": 0,
    }
    rec = _Recorder()
    stats = build.run_task(
        "PickPlace",
        cfg,
        (make_env, policy_obs, lambda task, e, profile: _Oracle(e, bad)),
        rec,
        demos=3,
        eval_episodes=3,
        allow_failed=False,
        num_envs=None,
    )
    assert stats["demos_recorded"] == 3 and stats["demos_oracle_finished"] == 3
    assert [m["seed"] for m in rec.episodes] == [seed0 + 1, seed0 + 3, seed0 + 4]
    assert all(m["oracle_finished"] for m in rec.episodes)
    assert [r["seed"] for r in stats["rejected_episodes"]] == sorted(bad)
    assert stats["rejected_episodes"][0]["reasons"] == ["oracle_unfinished", "off_release_target"]
    assert stats["demos_rejected"] == {
        "final_success_lost": 0,
        "oracle_unfinished": 2,
        "off_release_target": 2,
    }
    # without the opt-in (0.1 recorder settings) nothing is rejected for its tail
    cfg["data"]["require_tail_finished"] = False
    rec = _Recorder()
    stats = build.run_task(
        "PickPlace",
        cfg,
        (make_env, policy_obs, lambda task, e, profile: _Oracle(e, bad)),
        rec,
        demos=3,
        eval_episodes=3,
        allow_failed=False,
        num_envs=None,
    )
    assert [m["seed"] for m in rec.episodes] == [seed0, seed0 + 1, seed0 + 2]
    assert stats["rejected_episodes"] == []
    assert build.tail_problems("Push", True, 0.0) == []  # Push keeps the gripper closed
