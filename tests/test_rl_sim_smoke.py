"""WO-04 integration smoke: the real `scripts/wo03_rl.py` path on CPU -- WO-02's `build_rl_policy`
(tiny random OpenVLA-OFT under LoRA, SFT adapter loaded from `policy.adapter`) + WO-01's
`make_env` (ManiSkill, serial CPU envs, rendered) + WO-03's rollout / GRPO update -- for 2 steps.

The tiny random policy never succeeds, so every group would be all-zero and the update a no-op.
`_ScriptedSuccess` (test double only; the env's success check is untouched) reports success for
the first sample of every group after 3 env steps, giving mixed groups and a nonzero advantage.
4 serial envs in total: on macOS, MoltenVK's descriptor pools fragment beyond ~6 fresh renderers
in one process (ErrorFragmentedPool); the GPU sim has one env per task and is not affected.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
import torch

from src.config import load_config
from src.policy.openvla_oft import OpenVLAOFTPolicy, build_policy
from src.rl.rollout import TIMING_KEYS, RolloutBatch
from src.rl.trainer import GRPOTrainer

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.needs_render


class _ScriptedSuccess:
    def __init__(self, env: Any) -> None:
        self.env, self.t = env, 0

    def reset(self, *, seed: Any = None, options: Any = None) -> Any:
        self.t = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: torch.Tensor) -> Any:
        obs, rew, term, trunc, info = self.env.step(action)
        self.t += 1
        success = torch.as_tensor(info["success"]).bool()
        forced = (torch.arange(success.shape[0]) % 2 == 0) & (self.t >= 3)
        return obs, rew, term, trunc, {**info, "success": success | forced}

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def close(self) -> None:
        self.env.close()


class _Spy(GRPOTrainer):
    """Records each update's batch + stats, and scores the stored rollout inputs with the
    pre-update policy in several micro-batch layouts and both modes (sample/log_prob parity)."""

    instances: ClassVar[list[_Spy]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _Spy.instances.append(self)
        self.oft = self.policy
        assert isinstance(self.oft, OpenVLAOFTPolicy)
        self.init_params = {n: p.detach().clone() for n, p in self.oft.named_parameters()}
        self.updates: list[tuple[RolloutBatch, dict[str, Any], float]] = []

    @torch.no_grad()
    def _parity(self, batch: RolloutBatch) -> float:
        valid = batch.action_mask.any(-1)
        e_idx, s_idx = valid.nonzero(as_tuple=True)
        worst = 0.0
        for mode in (False, True):
            self.oft.train(mode)
            for micro in (1, 3, len(e_idx)):
                for i in range(0, len(e_idx), micro):
                    e, s = e_idx[i : i + micro], s_idx[i : i + micro]
                    obs = {k: v[e, s] for k, v in batch.obs.items()}
                    instr = [batch.instructions[j] for j in e.tolist()]
                    lp = self.oft.log_prob(obs, instr, batch.tokens[e, s], self.rl.temperature)
                    diff = (lp.log_probs - batch.log_probs[e, s]).abs()[batch.action_mask[e, s]]
                    worst = max(worst, float(diff.max()))
        return worst

    def update(self, batch: RolloutBatch) -> dict[str, Any]:
        parity = self._parity(batch)
        stats = super().update(batch)
        self.updates.append((batch, stats, parity))
        return stats


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("wo03_rl", REPO / "scripts" / "wo03_rl.py")
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_rl_preset_path_on_cpu_with_tiny_oft_and_maniskill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a WO-02-style SFT adapter (tiny random OFT, LoRA + trained proprio projector, perturbed)
    sft_cfg = load_config("tiny", REPO / "configs" / "sft_tiny.yaml")
    sft = build_policy(sft_cfg, "cpu")
    with torch.no_grad():
        for p in sft.trainable_parameters():
            p.add_(0.02 * torch.randn_like(p))
    sft.save(tmp_path / "sft")

    cfg_path = tmp_path / "rl_smoke.yaml"
    cfg_path.write_text(f"""
inherit: {REPO / 'configs' / 'sft_tiny.yaml'}
env:
  tasks: [PickPlace, Push]
  image_size: 64
  max_episode_steps: 16       # 2 decisions of C=8 actions
policy:
  adapter: {tmp_path / 'sft'}
rl:
  env: maniskill              # src.env.make_env, not the toy env
  samples_per_prompt: 2
  rollout_envs: 2             # 1 group per env batch
  rollout_batch: 2            # 2 env batches per step: PickPlace + Push
  max_rollout_rounds: 1
  steps: 2
  lr: 5e-5
  ppo_mini_batch: 4           # one mini-batch holding both tasks' instructions
  micro_batch: 3              # crosses trajectory, decision and task boundaries
  recompute_old_log_probs: false   # as configs/rl.yaml: sampler log-probs are the old ones
  eval_every: 2
  eval_episodes: 2
  gate_check_frac: 0          # 2 steps: no room for two gate windows
  diag_trajectories: 2        # the fixed-batch diagnostic on the first group
  diag_argmax: true
  checkpoint_every: 2
""")
    script = _load_script()
    monkeypatch.setattr(script, "REPO_ROOT", tmp_path)
    build_env = script.build_env

    def scripted_build_env(cfg: Any) -> Any:
        factory, obs_fn, max_steps = build_env(cfg)
        return (lambda task, n: _ScriptedSuccess(factory(task, n))), obs_fn, max_steps

    monkeypatch.setattr(script, "build_env", scripted_build_env)
    monkeypatch.setattr(script, "GRPOTrainer", _Spy)
    _Spy.instances.clear()
    summary = script.main(["--preset", "tiny", "--config", str(cfg_path)])

    (tr,) = _Spy.instances
    # the RL policy started from the SFT adapter: every weight equal, only LoRA trainable
    for name, p in sft.named_parameters():
        assert torch.equal(tr.init_params[name], p), name
    trainable = {n for n, p in tr.oft.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    assert len(tr.updates) == 2
    for batch, stats, parity in tr.updates:
        # shapes the trainer expects: E = 2 groups x K 2, S = 16 / 8 decisions, C = 8, A = 6
        assert batch.tokens.shape == batch.log_probs.shape == (4, 2, 8, 6)
        assert batch.tokens.dtype == torch.int64 and batch.log_probs.dtype == torch.float32
        assert 0 <= int(batch.tokens.min()) and int(batch.tokens.max()) < 256
        assert bool((batch.log_probs <= 0).all()) and bool(batch.log_probs.isfinite().all())
        assert batch.obs["image"].shape == (4, 2, 64, 64, 3)
        assert batch.obs["image"].dtype == torch.uint8
        assert batch.obs["state"].shape == (4, 2, 6)
        assert batch.action_mask.shape == (4, 2, 8)
        assert batch.rewards.tolist() == [1.0, 0.0] * 2
        assert batch.episode_len.tolist() == [3, 16] * 2  # scripted success ends k=0 at step 3
        assert batch.action_mask.flatten(1).sum(1).tolist() == batch.episode_len.tolist()
        assert batch.tasks == ["PickPlace"] * 2 + ["Push"] * 2
        assert len(set(batch.instructions)) == 2
        # sample() and log_prob() agree on the stored inputs, any micro-batch layout, both modes
        assert parity < 1e-5
        assert stats["train/first_minibatch_max_abs_log_ratio"] < 1e-5
        assert stats["train/optimizer_steps"] == 1.0 and stats["train/grad_norm"] > 0
        assert stats["train/adv_abs_mean"] > 0
        # fixed-batch diagnostic on the real policy: the pre-update scores match the sampler's
        # log-probs (reused as old ones here), the post-update ones moved
        assert stats["diag/pre/max_abs_dlogp"] < 1e-5 and stats["diag/post/mean_abs_dlogp"] > 0
        assert 0 <= stats["diag/post/argmax_flip_frac"] <= 1

    # the update moved LoRA weights only (base, proprio projector frozen)
    changed = {
        n for n, p in tr.oft.named_parameters() if not torch.equal(p.detach(), tr.init_params[n])
    }
    assert changed and changed <= trainable

    (run_dir,) = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "record.json").read_text())
    assert record["status"] == "finished" and record["dollars"] == 0.0
    assert record["policy"]["adapter"] == str(tmp_path / "sft")
    assert summary["steps_completed"] == 2
    rows = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evals = {r["step"]: r["eval/episodes_per_task"] for r in rows if "eval/success" in r}
    assert evals == {0: 2.0, 2: 2.0}  # step-0 baseline + periodic, eval_episodes each
    train = [r for r in rows if r["kind"] == "train"]
    assert [r["rollout/group_reset_diff_frac"] for r in train] == [0.0, 0.0]  # groups share scenes
    # rollout timing split: the render hook reaches the serial CPU envs' renderers
    for r in train:
        assert r["rollout/time/render_s"] > 0 and r["rollout/time/sim_step_s"] > 0
        assert r["rollout/time/policy_s"] > 0
        parts = sum(r[f"rollout/time/{k}_s"] for k in TIMING_KEYS)
        assert parts == pytest.approx(r["rollout/time_s"], rel=0.05)

    # the final RL checkpoint loads the way scripts/wo02_eval.py loads it, with identical outputs
    ckpt = run_dir / "checkpoints" / "step_000002" / "adapter"
    again = build_policy(sft_cfg, "cpu", adapter=ckpt, is_trainable=False)
    batch = tr.updates[-1][0]
    obs = {k: v[:, 0] for k, v in batch.obs.items()}
    tr.oft.eval()
    with torch.no_grad():
        assert torch.allclose(
            tr.oft.act_batch(obs, batch.instructions), again.act_batch(obs, batch.instructions)
        )
