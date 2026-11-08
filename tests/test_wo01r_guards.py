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
    """The label follows the RENDERER, not the sim device (review 2026-11-08): a Mac run on a
    0.1 set is a same-renderer pipeline check whatever the device; a run on the Linux GPU pod is
    the Mac-vs-GPU renderer gap even with --device cpu (SAPIEN still renders on the GPU)."""
    from src.config import CONFIG_DIR, load_config

    rebuild = _load("wo01r_rebuild_demos")
    assert rebuild.render_backend("Darwin", False) == "mac"
    assert rebuild.render_backend("Linux", True) == "gpu"  # GPU renderer, any physics device
    assert rebuild.render_backend("Linux", False) == "unknown"
    cfg = load_config("sim", CONFIG_DIR / "demos_v2.yaml")
    old, new = cfg["render_gap"]["dataset"], cfg["data"]["hub_repo"]
    label = rebuild.comparison_label
    same = "same-renderer pipeline check"
    for device in ("cpu", "cuda"):
        assert label(cfg, old, "render-gap", "mac", device).startswith(same)
        assert "NOT a Mac-vs-GPU" in label(cfg, old, "render-gap", "mac", device)
        assert label(cfg, old, "render-gap", "gpu", device).startswith("renderer gap")
        assert label(cfg, new, "render-gap", "gpu", device).startswith(same)
        assert label(cfg, new, "render-gap", "mac", device).startswith("renderer gap")
        assert "UNKNOWN" in label(cfg, old, "render-gap", "unknown", device)
    pod_cpu = label(cfg, old, "render-gap", "gpu", "cpu")  # CPU physics on the 4090 pod
    assert "NVIDIA GPU" in pod_cpu and "sim device cpu" in pod_cpu and same not in pod_cpu
    assert label(cfg, new, "dry-run", "mac", "cpu").startswith(same)


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
    # Hub repo ids resolve case-insensitively: a differently cased id is the same repo
    for repo in recorder.PROTECTED_HUB_REPOS:
        for variant in (repo.lower(), repo.upper(), f" {repo} "):
            with pytest.raises(recorder.ProtectedRepoError):
                recorder.check_push_target(variant)
    recorder.check_push_target("KudratSingh22/vla-rl-so101-demos-25-v2")  # the v2 repo is fine
    recorder.check_push_target("kudratsingh22/vla-rl-so101-demos-25-v2")
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


def test_wo01_build_env_push_refuses_a_short_set_before_pushing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2 (2026-11-08): `wo01_build_env.py --push` with fewer valid demos than
    demos_per_task (e.g. after rejecting unfinished demos) must not push, then exits 1."""
    import functools
    import json

    from src.runs import Run

    build = _load("wo01_build_env")
    pushed: list[Any] = []

    class _Rec:
        image_hw = (4, 4)
        repo_id = "KudratSingh22/vla-rl-so101-demos-25-v2"
        root = tmp_path / "demos"
        num_episodes = num_frames = 1

        def finalize(self) -> None:
            pass

        def push_to_hub(self, **kw: Any) -> str:
            pushed.append(kw)
            return "https://huggingface.co/datasets/x"

    class _Lazy:
        def __init__(self, cfg: Any) -> None:
            self.rec = _Rec()

    recorded = {"Stack": 22}

    def run_task(task: str, cfg: Any, api: Any, rec: Any, *, demos: int, **kw: Any) -> Any:
        return {
            "demos_recorded": recorded.get(task, demos),
            "failed_demos_filled": 0,
            "episodes": demos,
            "oracle_success_rate": 1.0,
            "mean_episode_len": 1.0,
            "mean_demo_frames": 1.0,
            "mean_post_success_frames": 1.0,
            "env_steps": 1,
            "wall_s": 1.0,
            "rejected_episodes": [],
            "demos_rejected": {},
        }

    monkeypatch.setattr(build, "Run", functools.partial(Run, runs_dir=tmp_path / "runs"))
    monkeypatch.setattr(build, "env_api", lambda: None)
    monkeypatch.setattr(build, "run_task", run_task)
    monkeypatch.setattr(build, "LazyRecorder", _Lazy)
    argv = ["--preset", "sim", "--config", "configs/demos_v2.yaml", "--push"]
    assert build.main(argv) == 1
    assert pushed == []  # refused before the push
    (run_dir,) = (tmp_path / "runs").iterdir()
    dataset = json.loads((run_dir / "record.json").read_text())["results"]["dataset"]
    assert "Stack" in dataset["push_refused"] and "hub_url" not in dataset
    recorded.clear()  # a full set still pushes
    assert build.main(argv) == 0 and len(pushed) == 1
