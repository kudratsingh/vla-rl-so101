"""Resume smoke test (tiny preset, CPU): train N steps, resume from a mid-run checkpoint, and get
exactly the uninterrupted run's weights, optimizer state and metrics."""

import importlib.util
import json
import random
import shutil
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

    # an _interrupted checkpoint (weights from inside step 7) is refused unless approximate
    part = tmp_path / "full" / "checkpoints" / "step_000007_interrupted"
    shutil.copytree(ckpt, part)
    st = json.loads((part / "trainer_state.json").read_text())
    st["step"] = 7
    st["gates"]["history"].append({**st["gates"]["history"][-1], "step": 7})
    (part / "trainer_state.json").write_text(json.dumps(st))
    with pytest.raises(ValueError, match="_interrupted.*step_000006"):
        _trainer(cfg, tmp_path / "part", init_seed=0).load_checkpoint(part)
    approx = _trainer(cfg, tmp_path / "approx", init_seed=0)
    approx.load_checkpoint(part, allow_partial=True)
    assert approx._start_step == 7 and approx.gates.history[-1].step == 6  # step 7 re-recordable

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


def test_resume_applies_this_runs_lr_and_weight_decay(tmp_path: Path) -> None:
    """Optimizer/scheduler state_dicts carry the saved lr and weight_decay; the resumed run's
    config must win (record.json shows it) while the Adam moments are kept."""
    cfg = _cfg()
    cfg["rl"].update(steps=6, lr=1e-4, gate_check_frac=0)
    _trainer(cfg, tmp_path / "a", init_seed=0).run()
    ckpt = tmp_path / "a" / "checkpoints" / "step_000006"
    cfg["rl"].update(steps=8, lr=5e-4, weight_decay=0.1)
    tr = _trainer(cfg, tmp_path / "b", init_seed=0)
    tr.load_checkpoint(ckpt)
    saved = torch.load(ckpt / "optimizer.pt", weights_only=True)
    assert tr.optimizer.state_dict()["state"][0]["step"] == saved["state"][0]["step"] > 0
    tr.run()
    group = tr.optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"]) == (5e-4, 0.1)
    assert tr.scheduler.base_lrs == [5e-4]
    assert tr.resume_overrides["lr"] == [1e-4, 5e-4]
    assert tr.resume_overrides["weight_decay"] == [0.01, 0.1]


def test_entry_script_fails_fast_and_records_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-PR-28 checkpoint is refused before any model load; a failure during the model load
    still finishes the run record (status, end time, dollars)."""
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)

    def boom(*_: object) -> None:
        raise RuntimeError("model load failed")

    monkeypatch.setattr(script, "build_policy", boom)
    old = tmp_path / "old" / "step_000025"  # main's checkpoint layout
    (old / "adapter").mkdir(parents=True)
    (old / "optimizer.pt").write_bytes(b"")
    (old / "trainer_state.json").write_text(
        json.dumps({"step": 25, "next_prompt_seed": 400, "eval_history": []})
    )
    for extra in ([], ["--resume-approximate"]):
        with pytest.raises(SystemExit):
            script.main(["--preset", "tiny", "--resume", str(old), *extra])
    with pytest.raises(SystemExit):
        script.main(["--preset", "tiny", "--lr-sweep", "1e-3", "--sweep-batch", "nope.pt"])
    assert not (tmp_path / "runs").exists()  # nothing got as far as a run record
    with pytest.raises(RuntimeError, match="model load"):
        script.main(["--preset", "tiny"])
    (run,) = (tmp_path / "runs").iterdir()
    record = json.loads((run / "record.json").read_text())
    assert record["status"] == "failed" and record["end_utc"] and record["dollars"] is not None
