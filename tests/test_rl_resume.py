"""Resume smoke test (tiny preset, CPU): train N steps, resume from a mid-run checkpoint, and get
exactly the uninterrupted run's weights, optimizer state and metrics."""

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
from src.rl.toy_env import make_toy_env_factory, policy_obs
from src.rl.trainer import GRPOConfig, GRPOTrainer

REPO = Path(__file__).resolve().parent.parent


def _cfg() -> Config:
    cfg = load_config("tiny")
    cfg["rl"].update(
        steps=12,
        eval_every=4,
        checkpoint_every=6,
        gate_window=2,
        gate_check_frac=0.5,  # flat gate at step 6, decline gate on: their windows must resume
        gate_max_drop=0.02,
        gate_stop=False,  # log-only, so both runs train all 12 steps whatever fires
    )
    return cfg


def _trainer(cfg: Config, run_dir: Path, init_seed: int) -> GRPOTrainer:
    random.seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
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


def _rows(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Train rows by step, without wall-clock-dependent keys."""
    out = {}
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "train":
            out[r["step"]] = {
                k: v
                for k, v in r.items()
                if not any(t in k for t in ("time", "per_hour", "batch_s", "mem/"))
            }
    return out


def test_resume_round_trip_matches_uninterrupted_run(tmp_path: Path) -> None:
    cfg = _cfg()
    full = _trainer(cfg, tmp_path / "full", init_seed=0)
    summary_full = full.run()
    ckpt = tmp_path / "full" / "checkpoints" / "step_000006"
    for name in ("adapter", "trainable.pt", "optimizer.pt", "scheduler.pt", "rng.pt"):
        assert (ckpt / name).exists(), name
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert state["step"] == state["completed_step"] == 6 and len(state["gates"]["history"]) == 6

    # same frozen base (in a real run: the base checkpoint + SFT adapter the builder loads), but
    # scrambled trainable weights and global RNG state: both must come from the checkpoint
    resumed = _trainer(cfg, tmp_path / "resumed", init_seed=0)
    with torch.no_grad():
        for p in resumed.params:
            p.normal_()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    resumed.load_checkpoint(ckpt)
    summary = resumed.run()

    for p, q in zip(full.params, resumed.params, strict=True):
        assert torch.equal(p, q)
    opt_a, opt_b = full.optimizer.state_dict(), resumed.optimizer.state_dict()
    for i, st in opt_a["state"].items():
        for key, value in st.items():
            assert torch.equal(value, opt_b["state"][i][key]), (i, key)
    assert resumed.scheduler.last_epoch == full.scheduler.last_epoch > 0
    rows_full, rows_resumed = _rows(tmp_path / "full"), _rows(tmp_path / "resumed")
    assert sorted(rows_resumed) == list(range(7, 13))  # no step-0 eval, no repeated steps
    for step in range(7, 13):
        assert rows_resumed[step] == rows_full[step], step
    assert summary["eval_history"] == summary_full["eval_history"]
    assert summary["steps_completed"] == 12 and summary["resumed_from"] == str(ckpt)

    bad = load_config("tiny")
    bad["toy"]["hidden"] = 32  # different adapter shapes
    with pytest.raises(ValueError, match="do not match"):
        _trainer(bad, tmp_path / "bad", init_seed=0).load_checkpoint(ckpt)


def test_entry_script_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    cfg_path = tmp_path / "tiny6.yaml"
    cfg_path.write_text(
        f"inherit: {REPO / 'configs' / 'tiny.yaml'}\n"
        "rl:\n  steps: 6\n  checkpoint_every: 3\n  gate_check_frac: 0\n"
    )
    script.main(["--preset", "tiny", "--config", str(cfg_path)])
    (first,) = (tmp_path / "runs").iterdir()
    ckpt = first / "checkpoints" / "step_000003"
    summary = script.main(["--preset", "tiny", "--config", str(cfg_path), "--resume", str(ckpt)])
    (second,) = [d for d in (tmp_path / "runs").iterdir() if d != first]
    record = json.loads((second / "record.json").read_text())
    assert record["status"] == "finished" and record["resumed_from"] == str(ckpt)
    assert summary["steps_completed"] == 6
    assert sorted(_rows(second)) == [4, 5, 6]
    assert _rows(second) == {s: r for s, r in _rows(first).items() if s > 3}
    with pytest.raises(FileNotFoundError, match="trainer_state"):
        script.main(["--preset", "tiny", "--resume", str(tmp_path / "nope")])
