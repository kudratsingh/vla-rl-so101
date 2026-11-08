"""Smoke tests for the GRPO loop on the toy env: env, rollout, update, trainer, entry script."""

import copy
import importlib.util
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
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
    gate_check_step,
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
        tr.scheduler = SimpleNamespace(step=lambda: None)  # type: ignore[assignment]
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
    cfg = _cfg(steps=40, eval_every=10)  # flat gate at step 10 (windows 1-5 vs 6-10)
    first = _trainer(cfg, tmp_path / "a").run()
    again = _trainer(cfg, tmp_path / "b").run()
    assert first["eval_history"] == again["eval_history"]  # same seed -> same numbers
    assert first["stop_reason"] is None
    assert first["final_eval_success"] >= first["initial_eval_success"] + 0.4
    assert first["final_eval_success"] >= 0.9
    rows = [
        json.loads(line) for line in (tmp_path / "a" / "metrics.jsonl").read_text().splitlines()
    ]
    train = [r for r in rows if r["kind"] == "train"]
    assert len(train) == 40
    for key in ("rollout/success_rate", "rollout/frac_groups_dropped", "rollout/entropy",
                "train/entropy", "train/grad_norm", "train/offpolicy_k3_kl", "diag/post/k3_kl",
                "diag/post/clip_frac", "diag/pre/k3_kl", "rollout/episodes_per_hour",
                "rollout/time/policy_s", "rollout/time/sim_step_s", "gate/success_window"):  # fmt: skip
        assert any(key in r for r in train), key
    for removed in ("train/clip_frac", "train/approx_kl"):  # uninformative aggregates (3.1)
        assert not any(removed in r for r in train)
    # 128 trajectories / 32 per mini-batch = 4 optimizer steps, each with its own pre-step stats
    assert len(train[0]["train/opt_step_k3_kl"]) == 4 and train[0]["train/opt_step_k3_kl"][0] < 1e-9
    # silent-bug detectors: the sampler and the scorer agree, and before the first optimizer
    # step the PPO ratio is exactly 1
    assert max(r["train/logprob_mismatch"] for r in train) < 1e-5
    assert max(r["train/first_minibatch_max_abs_log_ratio"] for r in train) < 1e-5
    # fixed-batch diagnostic: zero movement before the update (recompute on), real movement after
    # it, in the advantage's direction: the surrogate gain mean(A x dlogp) on the held 32
    # trajectories is positive on (nearly) every update -- the later mini-batches train on other
    # prompts and at tiny's lr 1e-2 can undo a little -- and on average A > 0 tokens rise while
    # A < 0 tokens fall
    assert max(r["diag/pre/mean_abs_dlogp"] for r in train) == 0.0
    diag = [r for r in train if "diag/post/k3_kl" in r]
    assert len(diag) >= 35  # every step that trained (diag_every 1)
    assert all(r["diag/post/mean_abs_dlogp"] > 0 and r["diag/post/k3_kl"] > 0 for r in diag)
    assert sum(r["diag/post/mean_adv_dlogp"] > 0 for r in diag) >= 0.9 * len(diag)
    assert statistics.mean(r["diag/post/mean_adv_dlogp"] for r in diag) > 0.01
    early = train[:10]
    assert statistics.mean(r["diag/post/mean_dlogp_pos_adv"] for r in early) > 0
    assert statistics.mean(r["diag/post/mean_dlogp_neg_adv"] for r in early) < 0
    # timing split: sections add up to the rollout time; the toy env has no renderer
    split = sum(train[0][f"rollout/time/{k}_s"] for k in ("policy", "sim_step", "render", "obs_fn",
                                                         "host_copy", "reset", "other"))  # fmt: skip
    assert split == pytest.approx(train[0]["rollout/time_s"], rel=0.05)
    assert train[0]["rollout/time/render_s"] == 0.0


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


def test_periodic_evals_use_identical_seeds_and_eval_episodes(tmp_path: Path) -> None:
    """Periodic greedy evals (sanity logs only): step 0, every eval_every and the last step, all
    eval_episodes per trained task on the same seeds, disjoint from the training prompts."""
    cfg = _cfg(steps=8, eval_every=4, rollout_envs=32, rollout_batch=8, eval_episodes=32)
    cfg["rl"]["gate_check_frac"] = 0  # 8 steps: no room for two gate windows
    rl = GRPOConfig.from_config(cfg)
    assert eval_schedule(rl) == {0: 32, 4: 32, 8: 32}
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
    assert per_task == {0: 32.0, 4: 32.0, 8: 32.0}
    seeds32 = [EVAL_SEED_BASE + i for i in range(32)]
    assert seeds_by_eval == [seeds32] * 3
    train_seeds = [s for r in resets for s in r if s < EVAL_SEED_BASE]
    assert train_seeds and max(train_seeds) < EVAL_SEED_BASE  # disjoint from the eval seeds


def test_train_tasks_roll_out_and_evaluate_only_trained_tasks(tmp_path: Path) -> None:
    """Owner amendment (ADR-0002): RL rollouts on rl.train_tasks only; since 2026-11-08 the
    periodic evals cover the trained tasks only too (review 9: no eval-only tasks in periodic
    evals; Stack is measured by the protocol eval); rl.max_episode_steps sets the horizon."""
    cfg = _cfg(steps=2, eval_every=100, train_tasks=["PickPlace", "Push"], max_episode_steps=4)
    cfg["rl"]["gate_check_frac"] = 0
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
    assert set(evaluated) == {"PickPlace", "Push"}
    assert "eval/success/Stack" not in out
    assert (
        out["eval/success_train"]
        == out["eval/success"]
        == pytest.approx((out["eval/success/PickPlace"] + out["eval/success/Push"]) / 2)
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


def test_collect_logs_exact_per_task_rollout_stats(tmp_path: Path) -> None:
    cfg = _cfg(train_tasks=["PickPlace", "Push"])
    cfg["env"]["tasks"] = ["PickPlace", "Push"]
    tr = _trainer(cfg, tmp_path)
    _, stats = tr.collect()
    trained = stats["rollout/groups_trained/PickPlace"] + stats["rollout/groups_trained/Push"]
    assert trained == stats["rollout/groups_trained"]
    for t in ("PickPlace", "Push"):
        assert 0.0 <= stats.get(f"rollout/frac_groups_dropped/{t}", 0.0) <= 1.0
    # the gate tallies cover every generated group (dropped ones too)
    groups = sum(t.groups for t in tr._last_tallies.values())
    assert groups == stats["rollout/groups_generated"]
    rate = sum(t.sum_rate for t in tr._last_tallies.values()) / groups
    assert rate == pytest.approx(stats["rollout/success_rate"])


def test_zero_success_baseline_aborts_before_training(tmp_path: Path) -> None:
    """CLAUDE.md rule 3: RL never starts from a 0%-success base."""
    tr = _trainer(_cfg(steps=4, gate_check_frac=0), tmp_path)
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
    split = probe["rollout_split_s_per_env_batch"]
    assert sum(split.values()) == pytest.approx(probe["rollout_s_per_env_batch"], rel=0.05)
    assert probe["diag_post"]["k3_kl"] > 0 and 0 <= probe["refill_cap_hit_frac"] <= 1
    assert not (tmp_path / "checkpoints").exists()


def test_flat_gate_stops_a_run_that_cannot_learn(tmp_path: Path) -> None:
    """lr 0: the policy cannot change, so windowed rollout success stays flat -> the flat gate
    stops at ceil(0.25 x 40) = 10 and checkpoints there (spend decision only)."""
    cfg = _cfg(steps=40, lr=0.0, eval_every=100)
    summary = _trainer(cfg, tmp_path).run()
    assert summary["stop_reason"] == "flat" and summary["stopped_flat"]
    assert summary["steps_completed"] == 10
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["event/flat_run"] == 1.0 and rows[-1]["gate/flat_gain"] < 0.03
    assert (tmp_path / "checkpoints" / "step_000010" / "adapter").is_dir()
    cfg["rl"]["gate_stop"] = False  # gates only log
    summary = _trainer(cfg, tmp_path / "log_only").run()
    assert summary["stop_reason"] is None and summary["steps_completed"] == 40


def test_config_validation() -> None:
    rl = GRPOConfig.from_config(load_config("rl"))  # the next-run preset must parse
    assert (rl.clip_low, rl.clip_high) == (0.2, 0.28)
    # review 3.1 / 9 (2026-11-08): 16 prompts x K=8 = 128 trajectories, lr 1e-4, 2 PPO epochs
    assert (rl.samples_per_prompt, rl.rollout_batch, rl.rollout_envs) == (8, 16, 32)
    assert (rl.lr, rl.ppo_epochs, rl.ppo_mini_batch) == (1e-4, 2, 64)
    # owner amendment after the WO-04 probes (2026-10-27, ADR-0002)
    assert (rl.temperature, rl.recompute_old_log_probs, rl.steps) == (1.0, True, 150)
    assert (rl.train_tasks, rl.max_episode_steps) == (("PickPlace", "Push"), 120)
    # spend gates on rollout windows; flat gate at step 38 compares windows 1-10 and 29-38
    assert gate_check_step(rl) == 38 and rl.gate_window == 10
    assert (rl.gate_min_gain, rl.gate_rule, rl.gate_stop) == (0.03, "any_task", True)
    assert rl.eval_episodes == 32 and rl.require_nonzero_baseline
    assert (rl.diag_every, rl.diag_trajectories) == (1, 32)
    pick = GRPOConfig.from_config(load_config("rl", REPO / "configs" / "rl_pickplace.yaml"))
    assert pick.train_tasks == ("PickPlace",) and pick.checkpoint_every == 25
    assert (pick.lr, pick.samples_per_prompt, pick.ppo_epochs) == (1e-4, 8, 2)
    with pytest.raises(ValueError, match="kl_coef"):
        GRPOConfig.from_config(_cfg(kl_coef=0.01))
    with pytest.raises(ValueError, match="multiple of K"):
        GRPOConfig.from_config(_cfg(rollout_envs=6))
    with pytest.raises(ValueError, match="unknown"):
        GRPOConfig.from_config(_cfg(clip_ratio=0.2))
    with pytest.raises(ValueError, match="max_episode_steps"):
        GRPOConfig.from_config(_cfg(max_episode_steps=-1))
    # the removed kill-switch / peak-stop keys fail loudly and name their replacement
    for key in ("early_stop_eval_episodes", "peak_stop_step", "early_stop_min_improvement"):
        with pytest.raises(ValueError, match="removed.*gate|gate.*removed|replaced"):
            GRPOConfig.from_config(_cfg(**{key: 1}))
    with pytest.raises(ValueError, match="2 x gate_window"):
        GRPOConfig.from_config(_cfg(steps=8))  # check step 2 < 2 x 5
    with pytest.raises(ValueError, match="gate_rule"):
        GRPOConfig.from_config(_cfg(gate_rule="median"))
    with pytest.raises(ValueError, match="diag_trajectories"):
        GRPOConfig.from_config(_cfg(diag_trajectories=6))  # not a multiple of K=4
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
    script = _load_script()
    est = script.estimate_cost(load_config("rl"))
    # configs/rl.yaml (A100 throughput measured 2026-10-27; 2026-11-08 shape): 16 x 8 = 128
    # trained, x 3.0 (refill cap) = 384 episodes x 120 (RL horizon) = 46,080 actions / 86 =
    # 535.8 s; 128 x 15 = 1,920 decisions: / 31 = 61.9 s recompute, 2 epochs / 10.3 = 372.8 s
    # update; diagnostic 32 x 15 = 480 decisions / 31 = 15.5 s; 7 evals (steps 0, 25, .., 150)
    # x 32 episodes x 2 trained tasks x 120 / 86
    assert est["actions_per_step"] == 46_080
    assert est["decisions_trained_per_step"] == 1_920
    assert est["recompute_s_per_step"] == pytest.approx(1920 / 31)
    assert est["update_s_per_step"] == pytest.approx(2 * 1920 / 10.3)
    assert est["diag_s_per_step"] == pytest.approx(480 / 31)
    assert est["evals"] == 7 and est["eval_episodes_per_task"] == 224 and est["eval_tasks"] == 2
    step_s = 46_080 / 86 + 2 * 1920 / 10.3 + 1920 / 31 + 480 / 31
    assert est["hours"] == pytest.approx((150 * step_s + 224 * 2 * 120 / 86) / 3600)
    assert est["dollars"] == pytest.approx(est["hours"] * 1.79)
    assert est["gate_check_step"] == 38
    assert est["gate_check_stop_h"] == pytest.approx((38 * step_s + 64 * 2 * 120 / 86) / 3600)
    # Track B probes: lr sweep = one capped collection + per lr (8 Adam steps x 64 x 15 / 10.3 +
    # 3 forwards of 1,920 decisions / 31) + one greedy forward; temperature probe = 16 env
    # batches of 32 x 120 actions / 86 per temperature
    probes = script.estimate_probes(
        load_config("rl"), lrs=[5e-5, 2e-4, 5e-4, 1e-3], adam_steps=8, stored_batch=False,
        temperatures=[0.7, 1.0, 1.3], groups=64,
    )  # fmt: skip
    per_lr = 8 * 64 * 15 / 10.3 + 3 * 1920 / 31
    assert probes["lr_sweep_h"] == pytest.approx((46_080 / 86 + 1920 / 31 + 4 * per_lr) / 3600)
    assert probes["temp_probe_h"] == pytest.approx(3 * 16 * 32 * 120 / 86 / 3600)


def test_entry_script_tiny_writes_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    cfg_path = tmp_path / "tiny_short.yaml"
    # 3 steps leaves no room for two gate windows; this test is about the run record
    cfg_path.write_text(
        f"inherit: {REPO / 'configs' / 'tiny.yaml'}\nrl:\n  steps: 3\n  gate_check_frac: 0\n"
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
    assert proj["gate_check_stop_h"] < proj["run_h"] <= proj["worst_case_h"]
    printed = [
        ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[wo03_rl] PROBE")
    ]
    assert any("projected" in ln and "per seed" in ln for ln in printed)
    assert any("sim_step" in ln and "render" in ln for ln in printed)  # rollout timing split
    assert any("post-update" in ln and "k3 KL" in ln for ln in printed)
    assert len(printed) <= 6  # scripts/pod/wo04_job.sh posts the last 6 PROBE lines


def test_entry_script_rejects_rl_seeds_on_the_protocol_range(tmp_path: Path) -> None:
    """Review 9 / PR #25: ManiSkill RL seed 10 would draw training prompts from 100M.., the
    protocol's held-out episode seeds; it is rejected before anything is built."""
    cfg_path = tmp_path / "maniskill.yaml"
    cfg_path.write_text(f"inherit: {REPO / 'configs' / 'rl.yaml'}\nseed: 10\n")
    with pytest.raises(ValueError, match=r"must be in \[0, 10\)"):
        _load_script().main(["--preset", "rl", "--config", str(cfg_path), "--estimate"])


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
    ckpt = run_dir / "checkpoints" / "step_000003_interrupted"
    assert (ckpt / "adapter").is_dir() and (ckpt / "optimizer.pt").is_file()
    state = json.loads((ckpt / "trainer_state.json").read_text())
    assert (state["step"], state["completed_step"]) == (3, 2)  # resume would redo step 3
    with pytest.raises(SystemExit):  # refused without --resume-approximate, before any build
        script.main(["--preset", "tiny", "--resume", str(ckpt)])
