"""Batched greedy eval harness on a fake vector env (deterministic success per seed), and
scripts/wo02_eval.py end to end with stand-ins for WO-01's src.env / src.eval.protocol / src.runs.
"""

from __future__ import annotations

import json
import math
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

import pytest
import torch
from torch import Tensor

import src.env.cameras  # noqa: F401  (real module cached before src.env is stubbed below)
from src.eval import protocol_v02
from src.eval.sim_eval import (
    EpisodeRecord,
    episode_seeds,
    evaluate,
    run_batch,
    summarize,
    to_markdown,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


class FakeEnv:
    """Seed s succeeds at step 3 iff s % 3 == 0; every episode is truncated at step 10."""

    def __init__(self, task: str, n: int) -> None:
        self.task, self.num_envs, self.unwrapped = task, n, self
        self.steps = 0
        self.seeds: list[int] = []
        self.actions: list[Tensor] = []
        self.closed = False

    def reset(self, seed: list[int]) -> tuple[dict[str, Tensor], dict[str, Any]]:
        assert len(seed) == self.num_envs
        self.seeds, self.steps = list(seed), 0
        return self._obs(), {}

    def get_instructions(self) -> list[str]:
        return [f"push the red cube to the target ({self.task})"] * self.num_envs

    def step(self, actions: Tensor) -> tuple[Any, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        assert actions.shape == (self.num_envs, 6)
        self.actions.append(actions.clone())
        self.steps += 1
        success = torch.tensor([s % 3 == 0 and self.steps >= 3 for s in self.seeds])
        trunc = torch.full((self.num_envs,), self.steps >= 10)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        grasped = torch.tensor([s % 2 == 1 for s in self.seeds])
        steps = torch.full((self.num_envs,), self.steps)  # not a bool flag: not an end flag
        info = {"success": success, "is_grasped": grasped, "elapsed_steps": steps}
        return self._obs(), zeros.float(), zeros, trunc, info

    def _obs(self) -> dict[str, Tensor]:
        return {
            "image": torch.zeros(self.num_envs, 32, 32, 3, dtype=torch.uint8),
            "state": torch.zeros(self.num_envs, 6),
        }

    def close(self) -> None:
        self.closed = True


def zero_act(obs: Mapping[str, Tensor], instructions: Sequence[str]) -> Tensor:
    return torch.zeros(len(instructions), 8, 6)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def test_run_batch_counts_success_and_lengths() -> None:
    env = FakeEnv("Push", 6)
    s, ln = run_batch(zero_act, env, lambda o: o, [0, 1, 2, 3, 4, 5], max_steps=150)
    assert s.tolist() == [True, False, False, True, False, False]
    assert ln.tolist() == [3, 10, 10, 3, 10, 10]  # one 8-step chunk, then truncation at 10
    s, ln = run_batch(zero_act, env, lambda o: o, [0, 1, 2, 3, 4, 5], max_steps=2)
    assert not s.any() and ln.tolist() == [2] * 6


def test_run_batch_receding_horizon_requeries_policy() -> None:
    calls: list[int] = []

    def counting_act(obs: Mapping[str, Tensor], instructions: Sequence[str]) -> Tensor:
        calls.append(len(instructions))
        return zero_act(obs, instructions)

    # seeds 1, 2 never succeed -> episodes run to truncation at step 10
    s, ln = run_batch(counting_act, FakeEnv("Push", 2), lambda o: o, [1, 2], max_steps=150)
    assert len(calls) == 2 and ln.tolist() == [10, 10]  # whole 8-step chunks: 8 + 2
    calls.clear()
    s, ln = run_batch(
        counting_act, FakeEnv("Push", 2), lambda o: o, [1, 2], max_steps=150, exec_steps=2
    )
    assert len(calls) == 5 and ln.tolist() == [10, 10]  # 2 steps per query: 10 / 2
    assert not s.any()


def ones_act(obs: Mapping[str, Tensor], instructions: Sequence[str]) -> Tensor:
    return torch.ones(len(instructions), 8, 6)  # 1 rad from the 0 rad reset pose on every joint


def test_run_batch_action_clip_and_end_flags() -> None:
    env = FakeEnv("Push", 3)
    end: dict[str, Tensor] = {}
    s, ln = run_batch(
        ones_act, env, lambda o: o, [0, 1, 2], max_steps=150, action_clip=0.25, end_info=end
    )
    sent = torch.stack(env.actions)  # (T, 3, 6): 0.25, 0.5, 0.75, then 1.0 on every joint
    assert torch.equal(sent[:4, 0, 0], torch.tensor([0.25, 0.5, 0.75, 1.0]))
    assert float((sent[1:] - sent[:-1]).abs().max()) <= 0.25
    assert s.tolist() == [True, False, False] and ln.tolist() == [3, 10, 10]
    assert end["clipped_steps"].tolist() == [3, 3, 3]
    assert end["success"].tolist() == [True, False, False]
    assert end["truncated"].tolist() == [False, True, True]
    assert end["is_grasped"].tolist() == [False, True, False]
    assert "elapsed_steps" not in end  # only bool flags
    env2 = FakeEnv("Push", 3)  # no clip: the raw chunk is executed, same outcome here
    run_batch(ones_act, env2, lambda o: o, [0, 1, 2], max_steps=150)
    assert torch.equal(torch.stack(env2.actions), torch.ones(10, 3, 6))


def test_evaluate_summarize_markdown() -> None:
    envs: list[FakeEnv] = []

    def make_env(task: str, cell: str, n: int) -> FakeEnv:
        envs.append(FakeEnv(task, n))
        return envs[-1]

    records: list[EpisodeRecord] = []
    res = evaluate(
        zero_act,
        make_env,
        lambda o: o,
        tasks=["Push"],
        cells=["seen", "shifted"],
        seeds=[0, 1],
        episodes=7,
        num_envs=4,
        max_steps=150,
        all_tasks=["PickPlace", "Push", "Stack"],
        all_cells=["seen", "shifted", "unseen_objects"],
        log_episode=records.append,
    )
    assert len(res) == 4 and all(r.episodes == 7 for r in res) and all(e.closed for e in envs)
    assert len(records) == 4 * 7  # padding episodes (2 batches of 4 for 7) are not recorded
    for r in res:
        mine = [e for e in records if (e.cell, e.seed) == (r.cell, r.seed)]
        assert [e.episode_seed for e in mine] == episode_seeds(
            r.seed, 1, ["seen", "shifted"].index(r.cell), 7
        )
        assert sum(e.success for e in mine) == r.successes
        assert all((e.end_reason == "success") == e.success for e in mine)
        assert all(e.end_reason in ("success", "truncated") and e.clipped_steps == 0 for e in mine)
    for r in res:  # successes = seeds divisible by 3 among this cell's 7 episode seeds
        seeds = episode_seeds(r.seed, 1, ["seen", "shifted"].index(r.cell), 7)
        assert r.successes == sum(s % 3 == 0 for s in seeds)
    assert len({tuple(episode_seeds(s, 1, c, 7)) for s in (0, 1) for c in (0, 1)}) == 4
    summary = summarize(res, wilson, {"policy": "zero", "protocol_complete": False})
    row = summary["rows"][0]
    assert row["episodes"] == 14 and row["wilson95"][0] <= row["pooled"] <= row["wilson95"][1]
    md = to_markdown(summary)
    assert "| Push | seen |" in md and "| Push | shifted |" in md


def _install_standins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_protocol: bool) -> Path:
    protocol = types.ModuleType("src.eval.protocol")
    protocol.__dict__.update(
        TASKS=("PickPlace", "Push", "Stack"),
        CELLS=("seen", "shifted", "unseen_objects"),
        SIM_SEEDS=(0, 1, 2),
        SIM_EPISODES_PER_CELL=100,
        PROTOCOL_VERSION="test",
        protocol_hash=lambda: "f" * 64,
        wilson_interval=wilson,
    )
    env = types.ModuleType("src.env")
    env.__dict__.update(
        make_env=lambda cfg, task, num_envs, cell: FakeEnv(task, num_envs),
        policy_obs=lambda obs: obs,
        CONTROL_HZ=15,
    )

    class Run:
        def __init__(self, **kw: Any) -> None:
            self.id, self.dir, self.results = "eval-test", tmp_path / "run", {}
            self.dir.mkdir()
            self.metrics: list[dict[str, Any]] = []

        def log_metrics(self, step: int | None = None, **m: Any) -> None:
            self.metrics.append(m)

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    runs = types.ModuleType("src.runs")
    runs.Run = Run  # type: ignore[attr-defined]
    stubs: list[tuple[str, types.ModuleType]] = [("src.env", env), ("src.runs", runs)]
    if stub_protocol:
        stubs.append(("src.eval.protocol", protocol))
    for name, mod in stubs:
        monkeypatch.setitem(sys.modules, name, mod)
    return tmp_path / "run"


@pytest.fixture()
def wo01_standins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _install_standins(tmp_path, monkeypatch, stub_protocol=True)


@pytest.fixture()
def env_standins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake env + run dir, real protocol modules."""
    return _install_standins(tmp_path, monkeypatch, stub_protocol=False)


def test_wo02_eval_tiny_end_to_end(wo01_standins: Path) -> None:
    import wo02_eval

    out = wo02_eval.main(["--preset", "tiny", "--episodes", "3", "--num-envs", "2"])
    assert out["protocol_complete"] is False and out["protocol_hash"] == "f" * 64
    assert "action_clip_rad" not in out  # 0.1-draft (default): no clip
    saved = json.loads((wo01_standins / "eval.json").read_text())
    assert [(r["task"], r["cell"], r["episodes"]) for r in saved["rows"]] == [
        ("PickPlace", "seen", 3)
    ]
    assert (wo01_standins / "eval.md").read_text().startswith("Policy `tiny-random`")
    lines = (wo01_standins / "episodes.jsonl").read_text().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["clipped_steps"] == 0 for x in lines)


def test_wo02_eval_protocol_v02_tiny(env_standins: Path) -> None:
    import wo02_eval

    argv = ["--preset", "tiny", "--episodes", "3", "--num-envs", "2", "--protocol", "0.2"]
    out = wo02_eval.main(argv)
    assert out["protocol_version"] == "0.2"
    assert out["protocol_hash"] == protocol_v02.protocol_hash()
    assert out["action_clip_rad"] == protocol_v02.action_clip().bound_rad
    assert out["protocol_complete"] is False  # tiny: CPU sim, 64 px images
    assert any("image_size" in d for d in out["protocol_deviations"])
    assert "src/env/success.py" in out["protocol_components"]["files"]
    eps = [json.loads(x) for x in (env_standins / "episodes.jsonl").read_text().splitlines()]
    assert [(e["task"], e["cell"], e["seed"]) for e in eps] == [("PickPlace", "seen", 0)] * 3
    assert {"success", "is_grasped"} <= set(eps[0]["flags"])
