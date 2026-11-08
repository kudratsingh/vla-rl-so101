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

from src.eval.sim_eval import episode_seeds, evaluate, run_batch, summarize, to_markdown

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


class FakeEnv:
    """Seed s succeeds at step 3 iff s % 3 == 0; every episode is truncated at step 10."""

    def __init__(self, task: str, n: int) -> None:
        self.task, self.num_envs, self.unwrapped = task, n, self
        self.steps = 0
        self.seeds: list[int] = []
        self.closed = False

    def reset(self, seed: list[int]) -> tuple[dict[str, Tensor], dict[str, Any]]:
        assert len(seed) == self.num_envs
        self.seeds, self.steps = list(seed), 0
        return self._obs(), {}

    def get_instructions(self) -> list[str]:
        return [f"push the red cube to the target ({self.task})"] * self.num_envs

    def step(self, actions: Tensor) -> tuple[Any, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        assert actions.shape == (self.num_envs, 6)
        self.steps += 1
        success = torch.tensor([s % 3 == 0 and self.steps >= 3 for s in self.seeds])
        trunc = torch.full((self.num_envs,), self.steps >= 10)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool)
        return self._obs(), zeros.float(), zeros, trunc, {"success": success}

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


def test_evaluate_summarize_markdown() -> None:
    envs: list[FakeEnv] = []

    def make_env(task: str, cell: str, n: int) -> FakeEnv:
        envs.append(FakeEnv(task, n))
        return envs[-1]

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
    )
    assert len(res) == 4 and all(r.episodes == 7 for r in res) and all(e.closed for e in envs)
    for r in res:  # successes = seeds divisible by 3 among this cell's 7 episode seeds
        seeds = episode_seeds(r.seed, 1, ["seen", "shifted"].index(r.cell), 7)
        assert r.successes == sum(s % 3 == 0 for s in seeds)
    assert len({tuple(episode_seeds(s, 1, c, 7)) for s in (0, 1) for c in (0, 1)}) == 4
    summary = summarize(res, wilson, {"policy": "zero", "protocol_complete": False})
    row = summary["rows"][0]
    assert row["episodes"] == 14 and row["wilson95"][0] <= row["pooled"] <= row["wilson95"][1]
    md = to_markdown(summary)
    assert "| Push | seen |" in md and "| Push | shifted |" in md


@pytest.fixture()
def wo01_standins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    for name, mod in (("src.eval.protocol", protocol), ("src.env", env), ("src.runs", runs)):
        monkeypatch.setitem(sys.modules, name, mod)
    return tmp_path / "run"


def test_wo02_eval_tiny_end_to_end(wo01_standins: Path) -> None:
    import wo02_eval

    out = wo02_eval.main(["--preset", "tiny", "--episodes", "3", "--num-envs", "2"])
    assert out["protocol_complete"] is False and out["protocol_hash"] == "f" * 64
    saved = json.loads((wo01_standins / "eval.json").read_text())
    assert [(r["task"], r["cell"], r["episodes"]) for r in saved["rows"]] == [
        ("PickPlace", "seen", 3)
    ]
    assert (wo01_standins / "eval.md").read_text().startswith("Policy `tiny-random`")


@pytest.mark.parametrize("exec_steps", [None, 3])
def test_traced_rollout_matches_run_batch(exec_steps: int | None) -> None:
    from src.eval.sim_eval import run_batch_traced

    seeds = [0, 1, 2, 3, 4, 5]
    s, ln = run_batch(zero_act, FakeEnv("Push", 6), lambda o: o, seeds, 150, exec_steps)
    calls: list[int] = []
    st, lt, trace = run_batch_traced(
        zero_act,
        FakeEnv("Push", 6),
        lambda o: o,
        seeds,
        150,
        exec_steps,
        snapshot_fn=lambda env: calls.append(env.steps) or env.steps,
    )
    assert torch.equal(s, st) and torch.equal(ln, lt)
    stride = 8 if exec_steps is None else exec_steps
    want = list(range(0, 10, stride))
    assert [b.step for b in trace.boundaries] == want and [
        b.state for b in trace.boundaries
    ] == want
    assert calls == want and trace.boundaries[0].active.all()
    assert len(trace.steps) == 11 and "action" not in trace.steps[0]
    assert "success" not in trace.steps[0]  # the fake's reset info is empty
    assert torch.stack([x["success"] for x in trace.steps[1:]]).shape == (10, 6)
    assert torch.equal(trace.boundaries[-1].last_action, trace.steps[want[-1]]["action"])
