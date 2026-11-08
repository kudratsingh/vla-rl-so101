"""rl.rollout_action_clip_deg: null (off) in the code default and the tiny preset, so those RL
rollouts send the policy's actions unchanged (the WO-04 runs); 4.0 in configs/rl.yaml (owner
2026-11-08), where executed actions get protocol 0.2's per-joint step clip. The clip changes the
env action only: the batch and PPO keep the sampled tokens and their log-probs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from joint_fakes import JointEnv, JointPolicy, joint_obs

from src.config import load_config
from src.eval import protocol_v02
from src.eval.sim_eval import clip_joint_step
from src.rl.policy_api import PolicyObs, RolloutStep
from src.rl.rollout import ClipStats, collect_rollouts
from src.rl.trainer import GRPOConfig, GRPOTrainer

N, CHUNK = 4, 3
BOUND = math.radians(4.0)


class JumpPolicy:
    """Always commands every joint to 1 rad (a 57 deg jump from the reset pose)."""

    def sample(self, obs: PolicyObs, instructions: Sequence[str], temperature: float) -> Any:
        return RolloutStep(
            actions=torch.ones(N, CHUNK, 6),
            tokens=torch.zeros(N, CHUNK, 6, dtype=torch.long),
            log_probs=torch.zeros(N, CHUNK, 6),
        )

    def eval(self) -> object:
        return self


def _run(clip: float | None, apply: bool = True) -> tuple[JointEnv, ClipStats]:
    env = JointEnv(N, max_steps=6, goal_rad=math.inf)  # no success: every episode runs 6 steps
    stats = ClipStats()
    collect_rollouts(
        JumpPolicy(),  # type: ignore[arg-type]
        env,
        joint_obs,
        prompt_seeds=[0, 1],
        group_size=2,
        temperature=1.0,
        max_episode_steps=6,
        task="t",
        action_clip=clip,
        clip_stats=stats,
        clip_apply=apply,
    )
    return env, stats


def test_presets_rl_on_at_the_protocol_bound_tiny_and_default_off() -> None:
    # owner 2026-11-08: the RL presets clip at protocol 0.2's bound (configs/real.yaml)
    bound = protocol_v02.action_clip().bound_deg
    assert bound == 4.0
    for name, path in (("rl", None), ("rl", "configs/rl_pickplace.yaml")):
        cfg = load_config(name, path)
        assert GRPOConfig.from_config(cfg).rollout_action_clip_deg == bound
    # tiny (CPU toy, WO-03 benchmark runs) and the code default keep the WO-04 behaviour
    assert GRPOConfig.from_config(load_config("tiny")).rollout_action_clip_deg is None
    assert GRPOConfig.rollout_action_clip_deg is None


def test_off_sends_actions_unchanged() -> None:
    env, stats = _run(None)
    assert all(torch.equal(a, torch.ones(N, 6)) for a in env.actions)  # unclipped
    assert stats.env_steps == 0  # nothing measured when there is no clip


def test_clip_limits_each_executed_step_like_protocol_02() -> None:
    env, stats = _run(BOUND)
    prev = torch.zeros(N, 6)
    for a in env.actions:
        assert torch.allclose((a - prev).abs(), torch.full((N, 6), BOUND))  # 4 deg per step
        prev = a
    assert stats.env_steps == stats.clipped_steps == 6 * N
    assert stats.clipped_joint_steps == stats.joint_steps == 6 * N * 6
    s = stats.summary("x")
    assert s["x_frac"] == s["x_joint_frac"] == 1.0 and s["x_frac_by_joint"] == [1.0] * 6
    # step t asks for 1 rad from (t - 1) x 4 deg: |a - clip(a)| = 1 - t x 4 deg
    deltas = [1.0 - t * BOUND for t in range(1, 7)]
    assert s["x_max_abs_delta_deg"] == pytest.approx(math.degrees(deltas[0]), rel=1e-5)
    assert s["x_mean_abs_delta_deg"] == pytest.approx(math.degrees(sum(deltas) / 6), rel=1e-5)


def test_measure_only_counts_what_the_clip_would_have_changed() -> None:
    env, stats = _run(BOUND, apply=False)
    assert all(torch.equal(a, torch.ones(N, 6)) for a in env.actions)  # sent unchanged
    # reference = the executed (unclipped) action: only the first jump exceeds 4 deg
    assert stats.env_steps == 6 * N and stats.clipped_steps == N
    assert stats.summary("x")["x_max_abs_delta_deg"] == pytest.approx(
        math.degrees(1.0 - BOUND), rel=1e-5
    )


def test_clip_changes_the_env_action_only_ppo_keeps_the_sampled_tokens(tmp_path: Path) -> None:
    """Owner 2026-11-08: the clip applies ONLY to the action sent to the env; the stored tokens,
    their log-probs, PPO's old log-probs and ratios all refer to the SAMPLED tokens."""
    torch.manual_seed(0)
    n, k, steps = 8, 2, 8
    policy, env = JointPolicy(chunk_len=2), JointEnv(n, max_steps=steps)
    batch, _ = collect_rollouts(
        policy,
        env,
        joint_obs,
        prompt_seeds=[0, 1, 2, 3],
        group_size=k,
        temperature=1.0,
        max_episode_steps=steps,
        task="t",
        action_clip=BOUND,
    )
    # stored = sampled, decision by decision (tokens AND their sampler log-probs)
    assert len(policy.samples) == batch.tokens.shape[1]
    for s, sampled in enumerate(policy.samples):
        assert torch.equal(batch.tokens[:, s], sampled.tokens)
        assert torch.equal(batch.log_probs[:, s], sampled.log_probs)
    # ... while the env received the clipped action, which differs from the sampled one
    sampled_actions = torch.cat([st.actions for st in policy.samples], dim=1)  # (n, S*C, 6)
    executed = torch.stack(env.actions, dim=1)  # (n, T, 6)
    t_steps = executed.shape[1]
    assert not torch.equal(executed, sampled_actions[:, :t_steps])
    prev = torch.zeros(n, 6)
    for t in range(t_steps):
        expect = clip_joint_step(sampled_actions[:, t], prev, BOUND)
        assert torch.equal(executed[:, t], expect)  # protocol 0.2's function, sampled -> env
        prev = expect
    assert float((executed[:, 1:] - executed[:, :-1]).abs().max()) <= BOUND + 1e-6

    # PPO on that batch: old log-probs (sampler or recomputed) and ratios are on the sampled tokens
    cfg = load_config("tiny")
    cfg["rl"].update(
        samples_per_prompt=k, rollout_envs=n, rollout_batch=4, ppo_mini_batch=8,
        ppo_epochs=1, micro_batch=64, diag_every=0, rollout_action_clip_deg=4.0, temperature=1.0,
    )  # fmt: skip
    for recompute in (False, True):
        cfg["rl"]["recompute_old_log_probs"] = recompute
        tr = GRPOTrainer(
            rl=GRPOConfig.from_config(cfg),
            policy=policy,
            env_factory=lambda task, num: JointEnv(num, steps),
            obs_fn=joint_obs,
            tasks=["t"],
            max_episode_steps=steps,
            run_dir=tmp_path / f"r{int(recompute)}",
            seed=0,
        )
        tr.optimizer.param_groups[0]["lr"] = 0.0  # score only: the weights stay put
        policy.scored.clear()
        stats = tr.update(batch)
        # old log-probs: the sampler's (recompute off) or a forward on the stored = sampled
        # tokens (recompute on); either way they match the sampled log-probs, and the first
        # mini-batch's ratio is exactly on-policy
        assert stats["train/first_minibatch_max_abs_log_ratio"] < 1e-5
        if recompute:
            assert stats["train/logprob_mismatch"] < 1e-5
        sampled_rows = {tuple(r) for r in batch.tokens.reshape(-1, 6).tolist()}
        scored_rows = {tuple(r) for t in policy.scored for r in t.reshape(-1, 6).tolist()}
        assert policy.scored and scored_rows <= sampled_rows


def test_config_key() -> None:
    cfg = load_config("tiny")
    cfg["rl"]["rollout_action_clip_deg"] = 4
    rl = GRPOConfig.from_config(cfg)
    assert rl.rollout_action_clip_deg == 4.0
    assert rl.action_clip_rad == pytest.approx(math.radians(4.0))
    cfg["rl"]["rollout_action_clip_deg"] = None
    assert GRPOConfig.from_config(cfg).action_clip_rad is None
    cfg["rl"]["rollout_action_clip_deg"] = 0
    with pytest.raises(ValueError, match="rollout_action_clip_deg"):
        GRPOConfig.from_config(cfg)
