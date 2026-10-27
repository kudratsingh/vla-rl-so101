"""Smoke tests for the GRPO loop on the toy env: env, rollout, update, trainer, entry script."""

import copy
import importlib.util
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.config import Config, load_config
from src.rl.grpo import group_advantages
from src.rl.policy_api import PolicyObs, RLPolicy, RolloutStep, TokenLogProbs, build_policy
from src.rl.rollout import check_group_reset, collect_rollouts
from src.rl.toy_env import ToyGridEnv, make_toy_env_factory, optimal_chunk, policy_obs
from src.rl.toy_policy import ToyPolicy
from src.rl.trainer import (
    EVAL_SEED_BASE,
    GRPOConfig,
    GRPOTrainer,
    eval_schedule,
    is_flat_run,
    kill_switch_step,
)

REPO = Path(__file__).resolve().parent.parent


def _cfg(**rl: Any) -> Config:
    cfg = load_config("tiny")
    cfg["rl"].update(rl)
    return cfg


def _trainer(cfg: Config, run_dir: Path, seed: int = 0) -> GRPOTrainer:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return GRPOTrainer(
        rl=GRPOConfig.from_config(cfg),
        policy=build_policy(cfg, "cpu"),
        env_factory=make_toy_env_factory(cfg),
        obs_fn=policy_obs,
        tasks=cfg["env"]["tasks"],
        max_episode_steps=int(cfg["toy"]["max_episode_steps"]),
        run_dir=run_dir,
        seed=seed,
    )


class OraclePolicy:
    """The toy env's known optimal policy behind the RLPolicy interface (3 bins: -1/0/+1)."""

    def __init__(self, chunk_len: int = 2) -> None:
        self.chunk_len = chunk_len

    def sample(
        self, obs: PolicyObs, instructions: Sequence[str], temperature: float
    ) -> RolloutStep:
        moves = optimal_chunk(obs["state"], grid_size=5, chunk_len=self.chunk_len)
        tokens = (moves + 1).long()
        return RolloutStep(
            actions=moves * (2.0 / 3.0), tokens=tokens, log_probs=torch.zeros(moves.shape)
        )

    def log_prob(
        self, obs: PolicyObs, instructions: Sequence[str], tokens: torch.Tensor, temperature: float
    ) -> TokenLogProbs:
        zeros = torch.zeros(tokens.shape)
        return TokenLogProbs(log_probs=zeros, entropy=zeros)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return []

    def train(self, mode: bool = True) -> object:
        return self

    def eval(self) -> object:
        return self

    def save(self, path: Path) -> None:
        pass


def _capture_grads_instead_of_stepping(tr: GRPOTrainer) -> list[torch.Tensor]:
    captured: list[torch.Tensor] = []
    params = tr.params

    def step(*args: Any, **kwargs: Any) -> None:
        captured.extend(p.grad.clone() for p in params if p.grad is not None)

    tr.optimizer.step = step  # type: ignore[method-assign]
    return captured


def test_toy_env_optimum_is_perfect_and_random_is_not() -> None:
    assert isinstance(OraclePolicy(), RLPolicy)
    env = ToyGridEnv(256, grid_size=5, max_episode_steps=6)
    seeds = list(range(256))
    batch, _ = collect_rollouts(
        OraclePolicy(), env, policy_obs, prompt_seeds=seeds, group_size=1,
        temperature=0.0, max_episode_steps=6, task="t",
    )  # fmt: skip
    assert float(batch.rewards.mean()) == 1.0  # optimal return is 1.0 on every episode
    torch.manual_seed(0)
    env.reset(seed=seeds)
    success = torch.zeros(256, dtype=torch.bool)
    for _ in range(6):
        _, _, _, _, info = env.step(torch.rand(256, 2) * 2 - 1)
        success |= info["success"]
    assert float(success.float().mean()) < 0.3  # uniform random policy: ~15%


def test_rollout_groups_share_prompt_and_mask_unexecuted_actions() -> None:
    k, prompts = 4, 16
    env = ToyGridEnv(k * prompts, grid_size=5, max_episode_steps=6)
    batch, diff = collect_rollouts(
        OraclePolicy(), env, policy_obs, prompt_seeds=range(100, 100 + prompts), group_size=k,
        temperature=1.6, max_episode_steps=6, task="t",
    )  # fmt: skip
    assert diff == 0.0
    first = batch.obs["state"][:, 0].view(prompts, k, -1)
    assert torch.equal(first, first[:, :1].expand_as(first))  # a group shares its initial state
    # the oracle reaches the goal in exactly Chebyshev-distance steps; nothing after is trained on
    d = (first[:, 0, 4:6] * 4).round().abs().amax(dim=1).repeat_interleave(k)
    assert batch.rewards.tolist() == [1.0] * (k * prompts)
    assert batch.episode_len.tolist() == d.long().tolist()
    assert batch.action_mask.flatten(1).sum(1).tolist() == d.long().tolist()
    flat = batch.action_mask.flatten(1).long()
    assert torch.all(flat[:, 1:] <= flat[:, :-1])  # executed actions form a prefix
    odd = int(torch.nonzero(d % 2 == 1)[0])  # success on the 1st action of a chunk
    last = int(d[odd]) // 2
    assert batch.action_mask[odd, last].tolist() == [True, False]


def test_group_reset_check_tolerances() -> None:
    """Joint state must match exactly; images may differ in <= 0.1% of values (GPU-sim pose
    rounding at silhouettes) but not by an object-sized patch."""
    g = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8, generator=g).repeat(4, 1, 1, 1)
    state = torch.rand(1, 6, generator=g).repeat(4, 1)
    assert check_group_reset({"image": image, "state": state}, 2, "error") == 0.0
    noisy = image.clone()
    noisy[1, 10, 10] ^= 1  # one pixel off by one level: 3 of 12288 values (0.024%)
    frac = check_group_reset({"image": noisy, "state": state}, 2, "error")
    assert frac == pytest.approx(3 / (64 * 64 * 3))
    moved = image.clone()
    moved[3, 20:28, 20:28] = 0  # an 8x8 px patch (a cube at 64 px): 1.6% of values
    with pytest.raises(RuntimeError, match="same seed differ"):
        check_group_reset({"image": moved, "state": state}, 2, "error")
    assert check_group_reset({"image": moved, "state": state}, 2, "warn") > 0.01
    shifted = state.clone()
    shifted[1, 0] += 1e-3  # 0.001 rad: the env's init-qpos noise is ~0.02 rad
    with pytest.raises(RuntimeError, match="state"):
        check_group_reset({"image": image, "state": shifted}, 2, "error")
    shifted[1, 0] = state[1, 0] + 5e-5  # float rounding passes
    assert check_group_reset({"image": image, "state": shifted}, 2, "error") == 0.0


def test_group_reset_check_catches_envs_that_ignore_seeds() -> None:
    class SeedIgnoringEnv(ToyGridEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> Any:
            return super().reset(seed=12345)  # per-env seeds 12345 + i: groups differ

    with pytest.raises(RuntimeError, match="same seed differ"):
        collect_rollouts(
            OraclePolicy(), SeedIgnoringEnv(16), policy_obs, prompt_seeds=range(4), group_size=4,
            temperature=1.6, max_episode_steps=6, task="t",
        )  # fmt: skip


def test_update_matches_upstream_per_trajectory_loss_and_ignores_micro_batching(
    tmp_path: Path,
) -> None:
    """Trainer gradient == literal re-implementation of upstream dp_rob aggregation, any micro."""
    cfg = _cfg(ppo_mini_batch=128, rollout_batch=32, grad_clip=1e9)  # no clipping: raw grads
    grads: dict[int, list[torch.Tensor]] = {}
    base = _trainer(cfg, tmp_path / "a")
    batch, _ = base.collect()
    assert batch is not None
    for micro in (1, 7, 4096):
        tr = _trainer(dict(cfg, rl=dict(cfg["rl"], micro_batch=micro)), tmp_path / f"m{micro}")
        tr.policy = copy.deepcopy(base.policy)
        tr.params = tr.policy.trainable_parameters()
        tr.optimizer = torch.optim.SGD(tr.params, lr=0.0)
        grads[micro] = _capture_grads_instead_of_stepping(tr)
        tr.update(batch)

    # reference: for each trajectory, masked token-mean of max(pg1, pg2); then mean over them
    ref = copy.deepcopy(base.policy)
    assert isinstance(ref, ToyPolicy)
    adv = group_advantages(batch.group_rewards).flatten()
    losses = []
    for e in range(batch.num_trajectories):
        valid = batch.action_mask[e].any(-1)
        obs = {key: v[e][valid] for key, v in batch.obs.items()}
        lp = ref.log_prob(obs, ["x"] * int(valid.sum()), batch.tokens[e][valid], 1.6).log_probs
        old = lp.detach()  # recomputed old log-probs == current params before the first step
        ratio = torch.exp(lp - old)
        pg = torch.maximum(-adv[e] * ratio, -adv[e] * ratio.clamp(0.8, 1.28))
        mask = batch.action_mask[e][valid].unsqueeze(-1).expand_as(pg)
        losses.append(pg[mask].mean())
    torch.stack(losses).mean().backward()
    ref_grads = [p.grad for p in ref.trainable_parameters()]
    for micro, got in grads.items():
        assert len(got) == len(ref_grads), micro
        for g, r in zip(got, ref_grads, strict=True):
            assert r is not None
            torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-6)


def test_tiny_loop_improves_success_deterministically(tmp_path: Path) -> None:
    cfg = _cfg(steps=40, eval_every=10)  # kill-switch check at step 10
    first = _trainer(cfg, tmp_path / "a").run()
    again = _trainer(cfg, tmp_path / "b").run()
    assert first["eval_history"] == again["eval_history"]  # same seed -> same numbers
    assert not first["stopped_flat"]
    assert first["final_eval_success"] >= first["initial_eval_success"] + 0.4
    assert first["final_eval_success"] >= 0.9
    rows = [
        json.loads(line) for line in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()
    ]
    train = [r for r in rows if r["kind"] == "train"]
    assert len(train) == 40
    for key in ("rollout/success_rate", "rollout/frac_groups_dropped", "train/clip_frac",
                "train/approx_kl", "train/entropy", "rollout/episodes_per_hour"):  # fmt: skip
        assert key in train[0], key
    # silent-bug detectors: the sampler and the scorer agree, and before the first optimizer
    # step the PPO ratio is exactly 1
    assert max(r["train/logprob_mismatch"] for r in train) < 1e-5
    assert max(r["train/first_minibatch_max_abs_log_ratio"] for r in train) < 1e-5


class _RecordingEnv:
    """Pass-through env that logs every reset's seed list."""

    def __init__(self, env: Any, resets: list[list[int]]) -> None:
        self.env, self.resets = env, resets

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        self.resets.append([int(s) for s in seed])
        return self.env.reset(seed=seed, options=options)

    def step(self, action: torch.Tensor) -> Any:
        return self.env.step(action)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped


def test_kill_switch_evals_64_episodes_on_identical_seeds_periodic_32(tmp_path: Path) -> None:
    """Owner, WO-04: step 0 and the 25% check evaluate early_stop_eval_episodes (64) per task on
    the same seeds; the periodic evals keep eval_episodes (32)."""
    cfg = _cfg(
        steps=8,
        eval_every=4,
        rollout_envs=32,
        rollout_batch=8,
        eval_episodes=32,
        early_stop_eval_episodes=64,
        early_stop_on_flat=False,  # keep evaluating after the check step
    )
    rl = GRPOConfig.from_config(cfg)
    assert eval_schedule(rl) == {0: 64, 2: 64, 4: 32, 8: 32}  # check step = ceil(0.25 * 8) = 2
    tr = _trainer(cfg, tmp_path)
    resets: list[list[int]] = []
    factory = tr.env_factory
    tr.env_factory = lambda task, n: _RecordingEnv(factory(task, n), resets)
    seeds_by_eval: list[list[int]] = []
    evaluate = tr.evaluate

    def spy(episodes: int | None = None) -> dict[str, float]:
        start = len(resets)
        out = evaluate(episodes)
        seeds_by_eval.append(sorted(s for r in resets[start:] for s in r))
        return out

    tr.evaluate = spy  # type: ignore[method-assign]
    tr.run()
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    per_task = {r["step"]: r["eval/episodes_per_task"] for r in rows if "eval/success" in r}
    assert per_task == {0: 64.0, 2: 64.0, 4: 32.0, 8: 32.0}
    seeds64 = [EVAL_SEED_BASE + i for i in range(64)]
    assert seeds_by_eval[0] == seeds_by_eval[1] == seeds64  # baseline == check-step seeds
    assert seeds_by_eval[2] == seeds_by_eval[3] == seeds64[:32]
    train_seeds = [s for r in resets for s in r if s < EVAL_SEED_BASE]
    assert train_seeds and max(train_seeds) < EVAL_SEED_BASE  # disjoint from the eval seeds


def test_kill_switch_threshold_is_three_points(tmp_path: Path) -> None:
    """Improvement 0.029 over step 0 stops the run at the check step; 0.031 continues."""
    assert is_flat_run(0.5, 0.529, 0.03) and not is_flat_run(0.5, 0.531, 0.03)
    assert GRPOConfig().early_stop_min_improvement == 0.03
    for check_value, stops in ((0.529, True), (0.531, False)):
        tr = _trainer(_cfg(steps=8, eval_every=100), tmp_path / str(check_value))
        values = iter([0.5, check_value])  # step-0 baseline, step-2 check; later evals: last step

        def fake_eval(episodes: int | None = None, it: Any = values) -> dict[str, float]:
            v = next(it, 0.9)  # the all-task mean stays flat: only the trained-task mean counts
            return {"eval/success": 0.5, "eval/success_train": v, "eval/success/PickPlace": v}

        tr.evaluate = fake_eval  # type: ignore[method-assign]
        summary = tr.run()
        assert summary["stopped_flat"] is stops
        assert summary["steps_completed"] == (2 if stops else 8)


def test_train_tasks_roll_out_only_trained_tasks_and_evaluate_all(tmp_path: Path) -> None:
    """Owner amendment (ADR-0002): RL rollouts on rl.train_tasks only (Stack eval-only); evals
    cover every env task and report the trained-task mean; rl.max_episode_steps sets the horizon."""
    cfg = _cfg(steps=2, eval_every=100, train_tasks=["PickPlace", "Push"], max_episode_steps=4)
    cfg["env"]["tasks"] = ["PickPlace", "Push", "Stack"]
    tr = _trainer(cfg, tmp_path)
    assert tr.max_episode_steps == 4  # overrides toy.max_episode_steps = 6
    collected: list[str] = []
    evaluated: list[str] = []
    phase = {"name": "eval"}
    factory = tr.env_factory

    def spy_factory(task: str, n: int) -> Any:
        return _TaskTaggedEnv(factory(task, n), task, phase, collected, evaluated)

    tr.env_factory = spy_factory
    collect, evaluate = tr.collect, tr.evaluate

    def spy_collect() -> Any:
        phase["name"] = "collect"
        try:
            return collect()
        finally:
            phase["name"] = "eval"

    tr.collect = spy_collect  # type: ignore[method-assign]
    out = evaluate(32)
    assert set(evaluated) == {"PickPlace", "Push", "Stack"}
    assert out["eval/success_train"] == pytest.approx(
        (out["eval/success/PickPlace"] + out["eval/success/Push"]) / 2
    )
    tr.run()
    assert collected and set(collected) == {"PickPlace", "Push"}
    with pytest.raises(ValueError, match="subset"):
        _trainer(_cfg(train_tasks=["Stack"]), tmp_path / "bad")


class _TaskTaggedEnv(_RecordingEnv):
    """Logs which task each reset belongs to, split by trainer phase (collect vs evaluate)."""

    def __init__(
        self,
        env: Any,
        task: str,
        phase: dict[str, str],
        collected: list[str],
        evaluated: list[str],
    ) -> None:
        super().__init__(env, [])
        self.task, self.phase, self.collected, self.evaluated = task, phase, collected, evaluated

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        (self.collected if self.phase["name"] == "collect" else self.evaluated).append(self.task)
        return super().reset(seed=seed, options=options)


def test_zero_success_baseline_aborts_before_training(tmp_path: Path) -> None:
    """CLAUDE.md rule 3: RL never starts from a 0%-success base."""
    tr = _trainer(_cfg(steps=4), tmp_path)
    tr.evaluate = lambda episodes=None: {  # type: ignore[method-assign]
        "eval/success": 0.0,
        "eval/success/PickPlace": 0.0,
        "eval/episodes": 64.0,
    }
    with pytest.raises(RuntimeError, match="rule 3"):
        tr.run()
    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(r)["kind"] for r in rows] == ["eval"]  # no training step ran


def test_probe_measures_without_saving(tmp_path: Path) -> None:
    probe = _trainer(_cfg(), tmp_path).probe(3)
    assert probe["baseline"]["eval/episodes_per_task"] == 256.0
    assert len(probe["rows"]) == 3 and probe["s_per_step_median"] > 0
    assert probe["rollout_episodes_per_hour"] > 0 and probe["eval_s_per_env_batch"] > 0
    assert probe["first_minibatch_max_abs_log_ratio"] < 1e-5
    assert not (tmp_path / "checkpoints").exists()


def test_flat_run_kill_switch_stops_the_run(tmp_path: Path) -> None:
    cfg = _cfg(steps=20, lr=0.0, eval_every=100)  # lr 0: policy cannot change -> flat
    summary = _trainer(cfg, tmp_path).run()
    assert summary["stopped_flat"]
    assert summary["steps_completed"] == 5  # ceil(0.25 * 20)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["event/flat_run"] == 1.0
    assert (tmp_path / "checkpoints" / "step_000005" / "adapter").is_dir()


def test_config_validation() -> None:
    rl = GRPOConfig.from_config(load_config("rl"))  # the WO-04 preset must parse
    assert (rl.clip_low, rl.clip_high) == (0.2, 0.28)
    # owner's K=16 rule fired (confirming probe: 72% of K=8 groups dropped); 128 trajectories
    assert (rl.samples_per_prompt, rl.rollout_batch, rl.rollout_envs) == (16, 8, 32)
    # owner amendment after the WO-04 probes (2026-10-27, ADR-0002)
    assert (rl.temperature, rl.recompute_old_log_probs, rl.steps) == (1.0, True, 150)
    assert (rl.train_tasks, rl.max_episode_steps) == (("PickPlace", "Push"), 120)
    assert kill_switch_step(rl) == 38
    # owner's WO-04 settings (2026-10-26)
    assert (rl.lr, rl.ppo_mini_batch) == (5e-5, 64)
    assert (rl.eval_episodes, rl.early_stop_eval_episodes) == (32, 64)
    assert (rl.early_stop_min_improvement, rl.early_stop_check_at_frac) == (0.03, 0.25)
    assert rl.early_stop_on_flat and rl.require_nonzero_baseline
    with pytest.raises(ValueError, match="kl_coef"):
        GRPOConfig.from_config(_cfg(kl_coef=0.01))
    with pytest.raises(ValueError, match="multiple of K"):
        GRPOConfig.from_config(_cfg(rollout_envs=6))
    with pytest.raises(ValueError, match="unknown"):
        GRPOConfig.from_config(_cfg(clip_ratio=0.2))
    with pytest.raises(ValueError, match="max_episode_steps"):
        GRPOConfig.from_config(_cfg(max_episode_steps=-1))
    # WO-02 provides the OpenVLA-OFT builder; it refuses a config without the base checkpoint
    with pytest.raises(ValueError, match="policy.checkpoint"):
        build_policy({"policy": {"base": "openvla_oft"}}, "cpu")


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_cost_estimate_matches_rl_yaml_arithmetic() -> None:
    est = _load_script().estimate_cost(load_config("rl"))
    # configs/rl.yaml (A100 throughput measured 2026-10-27; amended preset): 8 x 16 = 128 trained,
    # x 3.0 (refill cap) = 384 episodes x 120 (RL horizon) = 46,080 actions / 86 = 535.8 s;
    # 128 x 15 = 1,920 decisions / 10.3 = 186.4 s update + / 31 = 61.9 s old-log-prob recompute;
    # 8 evals = 6 x 32 + 2 x 64 (step 0 and the kill-switch step 38) = 320 episodes/task x 3 tasks
    # x 120 / 86
    assert est["actions_per_step"] == 46_080
    assert est["decisions_trained_per_step"] == 1_920
    assert est["recompute_s_per_step"] == pytest.approx(1920 / 31)
    assert est["evals"] == 8 and est["eval_episodes_per_task"] == 320
    assert est["hours"] == pytest.approx(
        (150 * (46_080 / 86 + 1920 / 10.3 + 1920 / 31) + 320 * 3 * 120 / 86) / 3600
    )
    assert est["dollars"] == pytest.approx(est["hours"] * 1.79)


def test_entry_script_tiny_writes_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    cfg_path = tmp_path / "tiny_short.yaml"
    # 3 steps puts the flat-run check at step 1, where eval noise alone can trip it (seen on the
    # Linux CI runner); this test is about the run record, so the kill switch is off
    cfg_path.write_text(
        f"inherit: {REPO / 'configs' / 'tiny.yaml'}\nrl:\n  steps: 3\n  early_stop_on_flat: false\n"
    )
    summary = script.main(["--preset", "tiny", "--config", str(cfg_path), "--seed", "1"])
    (run_dir,) = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "finished" and record["dollars"] == 0.0 and record["end_utc"]
    assert record["wo"] == "WO-03" and record["config"]["seed"] == 1
    assert record["summary"]["steps_completed"] == summary["steps_completed"] == 3
    assert (run_dir / "metrics.jsonl").exists()


def test_entry_script_probe_prints_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    out = script.main(["--preset", "tiny", "--probe-steps", "2"])
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert "-wo03probe-tiny-" in run_dir.name
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "probe" and not (run_dir / "checkpoints").exists()
    proj = record["projection"]
    # projection = configs' step count x measured median s/step + scheduled evals x s/eval batch
    rl = GRPOConfig.from_config(load_config("tiny"))
    eval_batches = sum(-(-ep // rl.rollout_envs) for ep in eval_schedule(rl).values())
    want = rl.steps * out["s_per_step_median"] + eval_batches * out["eval_s_per_env_batch"]
    assert proj["run_h"] == pytest.approx(want / 3600)
    assert proj["kill_switch_stop_h"] < proj["run_h"] <= proj["worst_case_h"]
    printed = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[wo03_rl] PROBE")
    ]
    assert any("projected" in ln and "per seed" in ln for ln in printed)


def test_sigterm_saves_interrupted_checkpoint_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod job's soft deadline (`timeout` -> SIGTERM) must not lose the steps since the last
    checkpoint, and the run record must still be finalised."""
    import os
    import signal

    script = _load_script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    train_step = GRPOTrainer._train_step
    calls: list[int] = []

    def train_step_then_sigterm(self: GRPOTrainer) -> dict[str, Any]:
        row = train_step(self)
        calls.append(1)
        if len(calls) == 3:  # the deadline hits during step 3
            os.kill(os.getpid(), signal.SIGTERM)
        return row

    monkeypatch.setattr(GRPOTrainer, "_train_step", train_step_then_sigterm)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit):
            script.main(["--preset", "tiny"])
    finally:
        signal.signal(signal.SIGTERM, previous)
    (run_dir,) = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "interrupted" and record["end_utc"] and record["dollars"] == 0.0
    assert (run_dir / "checkpoints" / "step_000003_interrupted" / "adapter").is_dir()
