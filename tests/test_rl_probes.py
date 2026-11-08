"""Dry runs of the Track B probe modes on the tiny preset (CPU): --lr-sweep and
--probe-temperatures (scripts/wo03_rl.py, GRPOTrainer.lr_sweep / probe_temperatures)."""

import importlib.util
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.config import Config, load_config
from src.rl.policy_api import build_policy
from src.rl.rollout import RolloutBatch
from src.rl.toy_env import make_toy_env_factory, policy_obs
from src.rl.trainer import GRPOConfig, GRPOTrainer

REPO = Path(__file__).resolve().parent.parent


def _trainer(cfg: Config, run_dir: Path) -> GRPOTrainer:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return GRPOTrainer(
        rl=GRPOConfig.from_config(cfg),
        policy=build_policy(cfg, "cpu"),
        env_factory=make_toy_env_factory(cfg),
        obs_fn=policy_obs,
        tasks=cfg["env"]["tasks"],
        max_episode_steps=int(cfg["toy"]["max_episode_steps"]),
        run_dir=run_dir,
        seed=0,
    )


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_lr_sweep_restores_state_and_movement_grows_with_lr(tmp_path: Path) -> None:
    tr = _trainer(load_config("tiny"), tmp_path)
    before = [p.detach().clone() for p in tr.params]
    out = tr.lr_sweep([1e-3, 1e-2, 5e-2], adam_steps=3)
    after = [p.detach().clone() for p in tr.params]
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))  # untouched
    assert not tr.optimizer.state  # the sweep's Adam moments did not leak
    rows = out["rows"]
    assert [r["lr"] for r in rows] == [1e-3, 1e-2, 5e-2]
    assert all(r["train/optimizer_steps"] == 3.0 for r in rows)  # 128 traj / 32 -> 3 of 4
    k3 = [r["diag/post/k3_kl"] for r in rows]
    assert 0 < k3[0] < k3[1] < k3[2]  # more lr, more on-batch movement
    flips = [r["diag/post/argmax_flip_frac"] for r in rows]
    assert flips[0] <= flips[2] and all(0 <= f <= 1 for f in flips)
    assert all(r["diag/trajectories"] == out["batch"]["trajectories"] for r in rows)  # all of it
    assert all(r["diag/post/mean_adv_dlogp"] > 0 for r in rows)
    # each lr starts from the same state: rerunning one lr alone gives the same numbers
    again = tr.lr_sweep([1e-2], adam_steps=3, batch=RolloutBatch.load(tmp_path / "sweep_batch.pt"))
    assert again["rows"][0]["diag/post/k3_kl"] == pytest.approx(k3[1], rel=1e-9)


def test_temperature_probe_pairs_prompts_across_temperatures(tmp_path: Path) -> None:
    tr = _trainer(load_config("tiny"), tmp_path)
    resets: list[list[int]] = []
    factory = tr.env_factory

    class Recording:
        def __init__(self, env: Any) -> None:
            self.env = env

        def reset(self, *, seed: Any = None, options: Any = None) -> Any:
            resets.append([int(s) for s in seed])
            return self.env.reset(seed=seed, options=options)

        def step(self, action: torch.Tensor) -> Any:
            return self.env.step(action)

        @property
        def unwrapped(self) -> Any:
            return self.env.unwrapped

    tr.env_factory = lambda task, n: Recording(factory(task, n))
    out = tr.probe_temperatures([0.5, 1.6], groups=20)  # 16 groups per env batch -> 2 batches
    assert out["groups"] == 32 and out["env_batches_per_temperature"] == 2
    assert resets[:2] == resets[2:]  # the same prompts at every temperature
    low, high = out["rows"]
    assert low["probe/entropy"] < high["probe/entropy"]
    for r in (low, high):
        assert r["probe/groups"] == 32 and r["probe/episodes"] == 128
        assert 0 <= r["probe/success_rate"] <= 1 and 0 <= r["probe/frac_groups_dropped"] <= 1
        drops = r["probe/frac_groups_all_zero"] + r["probe/frac_groups_all_one"]
        assert drops == pytest.approx(r["probe/frac_groups_dropped"])
        assert r["probe/time/policy_s"] > 0 and r["probe/env_batches_per_update"] >= 2
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["temp_probe", "temp_probe"]


def test_entry_script_probe_modes_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    out = script.main(["--preset", "tiny", "--lr-sweep", "1e-3,1e-2", "--sweep-adam-steps", "2"])
    run_dir = Path(out["run_dir"])
    assert "-wo03lrsweep-tiny-" in run_dir.name and (run_dir / "sweep_batch.pt").is_file()
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "probe" and len(record["lr_sweep"]["rows"]) == 2
    assert not (run_dir / "checkpoints").exists()
    # replaying the stored batch skips the rollout and reproduces the numbers
    replay = script.main(
        ["--preset", "tiny", "--lr-sweep", "1e-3,1e-2", "--sweep-adam-steps", "2",
         "--sweep-batch", str(run_dir / "sweep_batch.pt")]
    )  # fmt: skip
    for a, b in zip(out["rows"], replay["rows"], strict=True):
        assert a["diag/post/k3_kl"] == pytest.approx(b["diag/post/k3_kl"], rel=1e-9)
    temps = script.main(
        ["--preset", "tiny", "--probe-temperatures", "1.0,1.6", "--probe-groups", "16"]
    )
    assert "-wo03tempprobe-tiny-" in temps["run_dir"] and len(temps["rows"]) == 2
    printed = capsys.readouterr().out
    assert printed.count("[wo03_rl] LR-SWEEP lr") == 4 and "argmax flips" in printed
    assert printed.count("[wo03_rl] TEMP-PROBE T") == 2 and "groups dropped" in printed
    with pytest.raises(SystemExit):  # one mode at a time
        script.main(["--preset", "tiny", "--lr-sweep", "1e-3", "--probe-temperatures", "1.0"])
    with pytest.raises(SystemExit):
        script.main(["--preset", "tiny", "--lr-sweep", "-1e-3"])
