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
from src.rl.rollout import collect_rollouts
from src.rl.toy_env import ToyGridEnv, make_toy_env_factory, optimal_chunk, policy_obs
from src.rl.toy_policy import ToyPolicy
from src.rl.trainer import GRPOConfig, GRPOTrainer

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
    assert (rl.samples_per_prompt, rl.clip_low, rl.clip_high, rl.temperature) == (8, 0.2, 0.28, 1.6)
    with pytest.raises(ValueError, match="kl_coef"):
        GRPOConfig.from_config(_cfg(kl_coef=0.01))
    with pytest.raises(ValueError, match="multiple of K"):
        GRPOConfig.from_config(_cfg(rollout_envs=6))
    with pytest.raises(ValueError, match="unknown"):
        GRPOConfig.from_config(_cfg(clip_ratio=0.2))
    with pytest.raises(ImportError, match="openvla_oft"):
        build_policy({"policy": {"base": "openvla_oft"}}, "cpu")


def test_entry_script_tiny_writes_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    cfg_path = tmp_path / "tiny_short.yaml"
    cfg_path.write_text(f"inherit: {REPO / 'configs' / 'tiny.yaml'}\nrl:\n  steps: 3\n")
    summary = script.main(["--preset", "tiny", "--config", str(cfg_path), "--seed", "1"])
    (run_dir,) = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "finished" and record["dollars"] == 0.0
    assert record["wo"] == "WO-03" and record["config"]["seed"] == 1
    assert record["summary"]["steps_completed"] == summary["steps_completed"] == 3
    assert (run_dir / "metrics.jsonl").exists()
