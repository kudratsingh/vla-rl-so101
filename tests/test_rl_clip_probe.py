"""The paired clip on/off probe (owner 2026-11-08): `GRPOTrainer.probe_clip` and
`scripts/wo03_rl.py --probe-clip` on the CPU tiny preset, with a 6-joint test env (the toy env's
state is not a joint reading, so the clip cannot apply to it), and src/rl/paired.py."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from joint_fakes import JointEnv, JointPolicy, joint_obs

from src.config import Config, load_config
from src.rl.paired import mcnemar_exact_p, paired_group_rewards
from src.rl.trainer import GRPOConfig, GRPOTrainer

REPO = Path(__file__).resolve().parent.parent
BOUND = math.radians(4.0)


def _cfg() -> Config:
    cfg = load_config("tiny")
    cfg["rl"].update(rollout_action_clip_deg=4.0, temperature=1.0)
    return cfg  # K = 4, 64 envs -> 16 groups per env batch


def _trainer(cfg: Config, run_dir: Path, envs: list[JointEnv]) -> GRPOTrainer:
    def factory(task: str, n: int) -> JointEnv:
        envs.append(JointEnv(n, max_steps=8))
        return envs[-1]

    torch.manual_seed(0)
    return GRPOTrainer(
        rl=GRPOConfig.from_config(cfg),
        policy=JointPolicy(chunk_len=2),
        env_factory=factory,
        obs_fn=joint_obs,
        tasks=cfg["env"]["tasks"],
        max_episode_steps=8,
        run_dir=run_dir,
        seed=0,
    )


def test_paired_tables() -> None:
    off = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 0]], dtype=torch.float32)
    on = torch.tensor([[1, 0], [0, 0], [1, 1], [1, 1]], dtype=torch.float32)
    p = paired_group_rewards(off, on)
    assert p["episodes_both_success"] == 2 and p["episodes_neither"] == 2
    assert p["episodes_off_only"] == 1 and p["episodes_on_only"] == 3
    assert p["success_off"] == 3 / 8 and p["success_on"] == 5 / 8
    assert p["success_diff_on_minus_off"] == pytest.approx(0.25)
    # dropped (all-equal): off groups 0, 1, 3; on groups 1, 2, 3
    assert (p["groups_dropped_both"], p["groups_dropped_off_only"]) == (2, 1)
    assert (p["groups_dropped_on_only"], p["groups_kept_both"]) == (1, 0)
    assert p["groups_dropped_off"] == p["groups_dropped_on"] == 0.75
    assert mcnemar_exact_p(1, 3) == pytest.approx(0.625)  # 2 x (1 + 4) / 16
    assert mcnemar_exact_p(0, 0) == 1.0 and mcnemar_exact_p(0, 10) == pytest.approx(2 / 1024)
    with pytest.raises(ValueError):
        paired_group_rewards(off, on[:2])


def test_probe_clip_pairs_prompts_and_sampling_seeds(tmp_path: Path) -> None:
    envs: list[JointEnv] = []
    tr = _trainer(_cfg(), tmp_path, envs)
    out = tr.probe_clip(groups=20)  # 16 groups per env batch -> 2 batches per arm
    assert out["groups"] == 32 and out["env_batches_per_arm"] == 2
    assert out["samples_per_prompt"] == 4 and out["rollout_action_clip_deg"] == 4.0
    (env,) = envs  # tiny trains one task
    # batch order: (b0 off, b0 on, b1 off, b1 on); the arms of a batch play the same prompts
    assert env.resets[0] == env.resets[1] and env.resets[2] == env.resets[3]
    assert env.resets[0] != env.resets[2]
    assert len(set(out["sampling_seeds"])) == 2
    off, on = out["rows"]
    assert (off["arm"], off["clip_applied"], on["arm"], on["clip_applied"]) == (
        "off",
        False,
        "on",
        True,
    )
    for r in (off, on):
        assert r["probe/groups"] == 32 and r["probe/episodes"] == 128
        assert 0 <= r["probe/success_rate"] <= 1 and 0 <= r["probe/frac_groups_dropped"] <= 1
        assert r["probe/action_clip_frac"] > 0  # +-0.5 rad jumps: the clip binds
        assert 0 < r["probe/action_clip_joint_frac"] <= r["probe/action_clip_frac"]
        assert len(r["probe/action_clip_frac_by_joint"]) == 6
        assert r["probe/action_clip_max_abs_delta_deg"] > 4.0
    # arm off sent unclipped actions (jumps of up to 1 rad); arm on never moved > 4 deg per step
    p = out["paired"]
    assert p["first_decision_token_match"] == 1.0  # same scene + same RNG -> same first draw
    assert p["groups"] == 32 and p["episodes_per_arm"] == 128
    assert p["success_off"] == off["probe/success_rate"]
    assert p["success_on"] == on["probe/success_rate"]
    assert on["probe/success_rate"] < off["probe/success_rate"]  # the goal is 2+ clipped steps
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["clip_probe", "clip_probe", "clip_probe_paired"]

    # deterministic: a second probe from the same counters gives the same numbers
    envs2: list[JointEnv] = []
    again = _trainer(_cfg(), tmp_path / "again", envs2).probe_clip(groups=20)
    assert again["paired"] == p


def test_probe_clip_builds_envs_before_seeding(tmp_path: Path) -> None:
    # review R3-3: an env factory that draws from the global torch RNG while building must not
    # shift arm "off"'s first-batch draws (envs are built before any reseed or timer)
    envs: list[JointEnv] = []
    tr = _trainer(_cfg(), tmp_path, envs)
    build = tr.env_factory

    def rng_factory(task: str, n: int) -> JointEnv:
        torch.rand(1000)
        return build(task, n)

    tr.env_factory = rng_factory
    out = tr.probe_clip(groups=20)
    assert out["paired"]["first_decision_token_match"] == 1.0
    assert all(r["probe/s_per_env_batch_after_first"] > 0 for r in out["rows"])


def test_probe_clip_needs_the_clip_set(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["rl"]["rollout_action_clip_deg"] = None
    with pytest.raises(ValueError, match="rollout_action_clip_deg"):
        _trainer(cfg, tmp_path, []).probe_clip(groups=4)


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_entry_script_probe_clip_tiny_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--probe-clip` end to end on CPU (tiny preset + the preset's clip), as diag_job runs it."""
    cfg_path = tmp_path / "tiny_clip.yaml"
    cfg_path.write_text(f"""
inherit: {REPO / 'configs' / 'tiny.yaml'}
rl:
  rollout_action_clip_deg: 4.0
  temperature: 1.0
""")
    script = _script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        script,
        "build_env",
        lambda cfg: ((lambda task, n: JointEnv(n, max_steps=8)), joint_obs, 8),
    )
    monkeypatch.setattr(script, "build_policy", lambda cfg, device: JointPolicy(chunk_len=2))
    base = ["--preset", "tiny", "--config", str(cfg_path)]
    out = script.main([*base, "--probe-clip", "--probe-groups", "16"])
    run_dir = Path(out["run_dir"])
    assert "-wo03clipprobe-tiny-" in run_dir.name
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "probe" and record["clip_probe"]["groups"] == 16
    assert record["clip_probe"]["rollout_action_clip_deg"] == 4.0
    assert not (run_dir / "checkpoints").exists()
    printed = capsys.readouterr().out
    lines = [x for x in printed.splitlines() if x.startswith("[wo03_rl] CLIP-PROBE")]
    assert len(lines) == 4 and "clip 4 deg/step vs off" in lines[0]
    assert "arm off" in lines[1] and "WOULD have changed" in lines[1]
    assert "arm on" in lines[2] and "clip changed" in lines[2]
    assert "first-decision tokens identical 1.000" in lines[3]

    # the other probes read the same preset clip and print it
    temps = script.main([*base, "--probe-temperatures", "1.0", "--probe-groups", "16"])
    assert temps["rollout_action_clip_deg"] == 4.0
    assert temps["rows"][0]["probe/action_clip_frac"] > 0
    sweep = script.main([*base, "--lr-sweep", "1e-3", "--sweep-adam-steps", "1"])
    assert sweep["rollout_action_clip_deg"] == 4.0
    assert sweep["batch"]["rollout/action_clip_frac"] > 0
    printed = capsys.readouterr().out
    assert "TEMP-PROBE 16 groups" in printed and "action clip 4 deg/step:" in printed
    assert "LR-SWEEP batch" in printed and "rollouts with action clip 4 deg/step" in printed

    # --estimate on the A100 preset (configs/rl.yaml, clip 4.0): 32 groups (default) at 4 groups
    # per 32-env batch = 8 batches per arm, x 2 arms, x 32 x 120 actions at 86 actions/s
    est = script.main(["--preset", "rl", "--estimate", "--probe-clip"])
    assert est["clip_probe_env_batches_per_arm"] == 8.0
    assert est["clip_probe_s"] == pytest.approx(2 * 8 * 32 * 120 / 86)
    assert "rollout_action_clip_deg: 4.0" in capsys.readouterr().out
    # the tiny preset itself has no clip: --probe-clip refuses before building anything
    with pytest.raises(SystemExit):
        script.main(["--preset", "tiny", "--probe-clip"])
