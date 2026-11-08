"""Rollout worker: N parallel envs, K samples per prompt, binary terminal reward (ADR-0002).

Loop logic adapted from PRIME-RL/SimpleVLA-RL @ 7c51662df27b586f9e8a1ab35fcf849f2b8852f9 (MIT;
file header Apache-2.0, Copyright 2024 Bytedance Ltd.), verl/workers/rollout/rob_rollout.py
`RobHFRollout._generate_minibatch_libero` + `env_worker` and verl/trainer/main_ppo.py
`RobRewardManager.verify`:
  * a group = one prompt (task + reset seed = initial state) repeated K times in consecutive envs
    (upstream: `trial_id.repeat_interleave(n_samples)`); trajectory index e = g * K + k;
  * every decision samples one action chunk for all envs, which is executed step by step; an env
    whose episode ended (success, env termination/truncation, or the step limit) is no longer
    counted; `finish_step` = env steps executed, and only the tokens of executed actions are
    trained on (upstream mask `steps < finish_step` at action granularity);
  * reward = 1.0 if `info["success"]` fired before the episode ended, else 0.0, one scalar per
    trajectory (upstream `complete`).
Differences: envs are one batched (GPU) env instead of one process per env, so finished envs keep
receiving (ignored) actions; and the loop stops as soon as every env is done.

Action clip (`action_clip`, rad; rl.rollout_action_clip_deg): None (the code default, the tiny
preset, and every RL run so far, WO-04 included) sends the policy's actions to env.step unchanged.
When set (configs/rl.yaml: 4.0 deg/step since owner 2026-11-08), every executed action is clipped
per joint to within that bound of the previous executed action (of the policy-obs joint state at
reset) by protocol 0.2's own `clip_joint_step` (src/eval/sim_eval.py, imported, not copied), as
protocol 0.2's eval does. The clip is part of the ENVIRONMENT: it changes only the action sent to
env.step. The batch stores the SAMPLED tokens and their sampler log-probs, so PPO's old log-probs,
ratios and advantages all refer to what the policy sampled, never to the clipped action
(tests/test_rl_action_clip.py). `clip_apply=False` computes the clip without applying it (the
paired probe's off arm: how often the clip WOULD have changed the executed action). `ClipStats`
accumulates the intervention counts.

The env must reset deterministically per seed: identical seeds must give identical initial states,
otherwise a "group" does not share a prompt and the group baseline is wrong. `check_group_reset`
verifies this on every batch at no cost (tolerances in its docstring).

Timing split (`RolloutTimer`, review 5: the 44.5 s per 32-env batch was not instrumented). Wall
seconds per section, summed over a call; with CUDA initialised every section boundary calls
torch.cuda.synchronize(), so asynchronous GPU work is charged to the section that queued it:
  policy       policy.sample(): the wrapper's preprocessing (resize/crop/normalise), host->device
               copy of the inputs, the forward pass, sampling and token->action decoding. The
               policy API does not expose preprocessing separately, so it is inside this number.
  sim_step     env.step() minus `render`: action clipping / PD targets, physics substeps, success
               evaluation (get_info) and the non-image part of the observation.
  render       ManiSkill's `_get_obs_sensor_data` inside env.step(): update_render, camera
               capture, picture readout and its own CUDA sync. Measured by wrapping that method on
               the unwrapped env (`instrument_render`); 0.0 where the env has no such method (toy
               env). Approximate on the GPU sim: physics kernels still in flight when rendering
               starts are synchronised inside it and counted here, not in sim_step. ManiSkill
               renders on every env step, while the policy only uses the observation at decision
               boundaries (every C = 8 steps).
  obs_fn       env observation -> policy observation (src.env.policy_obs), at decision boundaries.
  host_copy    device->host copies into the CPU rollout buffers (observations, tokens, log-probs)
               and of the per-step success / termination flags.
  reset        env.reset() including its render, plus the group-reset check.
  other        the rest (loop bookkeeping, buffer stacking; in the trainer's rollout/time/other_s
               also its dynamic-sampling filter and entropy tally between env batches):
               wall time - sum of the above, so the sections sum to rollout/time_s.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from src.eval.sim_eval import clip_joint_step
from src.rl.policy_api import RLPolicy

log = logging.getLogger(__name__)

ObsFn = Callable[[Any], dict[str, Tensor]]
STORAGE = torch.device("cpu")  # rollout buffers live on CPU; the policy moves inputs as needed
# check_group_reset tolerances (see its docstring)
RESET_FLOAT_ATOL = 1e-4  # joint positions, rad
RESET_IMAGE_DIFF_FRAC = 1e-3  # of an image's uint8 values


class VecEnv(Protocol):
    """The slice of the gymnasium / ManiSkill batched env API the rollout uses."""

    def reset(self, *, seed: Any = None, options: Any = None) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Tensor) -> tuple[Any, Any, Any, Any, dict[str, Any]]: ...

    @property
    def unwrapped(self) -> Any: ...


TIMING_KEYS = ("policy", "sim_step", "render", "obs_fn", "host_copy", "reset", "other")


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()


class RolloutTimer:
    """Wall seconds per rollout section (module docstring). One timer may span several
    `collect_rollouts` calls; `totals()` sums them."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = dict.fromkeys(TIMING_KEYS, 0.0)
        self.total_s = 0.0
        self.pending_render = 0.0  # filled by the render hook during env.step / env.reset

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.seconds[name] += time.perf_counter() - t0

    def env_step(self, env: VecEnv, action: Tensor) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """env.step(action), split into sim_step and render (the render hook's share)."""
        self.pending_render = 0.0
        _sync()
        t0 = time.perf_counter()
        out = env.step(action)
        _sync()
        dt = time.perf_counter() - t0
        render = min(self.pending_render, dt)
        self.seconds["render"] += render
        self.seconds["sim_step"] += dt - render
        self.pending_render = 0.0
        return out

    def totals(self, wall_s: float | None = None) -> dict[str, float]:
        """Seconds per section. `other` = `wall_s` (default: the time inside collect_rollouts)
        minus the named sections, so a caller that passes its own wall time (the trainer: its
        bookkeeping between env batches too) gets parts that sum to it."""
        out = dict(self.seconds)
        wall = self.total_s if wall_s is None else wall_s
        out["other"] = max(0.0, wall - sum(v for k, v in out.items() if k != "other"))
        return out


_RENDER_HOOK = "_rl_render_timer"


def _render_bases(env: Any) -> list[Any]:
    """The ManiSkill envs that render for `env`: its unwrapped env (GPU sim), or each sub-env's
    (src.env.vector.SerialVecEnv on the CPU sim)."""
    base = getattr(env, "unwrapped", None)
    if base is not None and callable(getattr(base, "_get_obs_sensor_data", None)):
        return [base]
    subs = getattr(base, "envs", None)
    if isinstance(subs, list):
        return [b for sub in subs for b in _render_bases(sub)]
    return []


def instrument_render(env: VecEnv, timer: RolloutTimer | None) -> bool:
    """Point the env's render hook at `timer` (None detaches it). The first call installs the hook:
    the unwrapped ManiSkill env's `_get_obs_sensor_data` is wrapped at instance level, so env.step
    and env.reset call the timed version. False if the env has nothing to instrument (toy env)."""
    bases = _render_bases(env)
    for base in bases:
        holder: dict[str, RolloutTimer | None] | None = getattr(base, _RENDER_HOOK, None)
        if holder is None:
            holder = {"timer": None}
            base._get_obs_sensor_data = _timed(base._get_obs_sensor_data, holder)
            setattr(base, _RENDER_HOOK, holder)
        holder["timer"] = timer
    return bool(bases)


def _timed(original: Callable[..., Any], holder: dict[str, RolloutTimer | None]) -> Any:
    def timed(*args: Any, **kwargs: Any) -> Any:
        t = holder["timer"]
        if t is None:
            return original(*args, **kwargs)
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            t.pending_render += time.perf_counter() - t0

    return timed


@dataclass
class ClipStats:
    """Action-clip intervention counts over env steps of RUNNING episodes (finished envs that
    still receive ignored actions are not counted). "Changed" = the clip moved the action
    (`clip_joint_step(a, ref, bound) != a`), whether or not it was applied (`clip_apply`)."""

    env_steps: int = 0  # running-episode env steps seen
    clipped_steps: int = 0  # ... on which the clip changed at least one joint
    joint_steps: int = 0  # env_steps x joints
    clipped_joint_steps: int = 0
    clipped_by_joint: list[int] = field(default_factory=list)  # per joint
    delta_sum_rad: float = 0.0  # sum of |a - clip(a)| over the changed entries
    delta_max_rad: float = 0.0

    def add(self, changed: Tensor, delta: Tensor, active: Tensor) -> None:
        """changed / delta (T, N, A) (already zero where the episode was not running), active
        (T, N) bool."""
        by_joint = changed.sum(dim=(0, 1)).tolist()
        if not self.clipped_by_joint:
            self.clipped_by_joint = [0] * len(by_joint)
        self.clipped_by_joint = [
            a + int(b) for a, b in zip(self.clipped_by_joint, by_joint, strict=True)
        ]
        steps = int(active.sum())
        self.env_steps += steps
        self.clipped_steps += int(changed.any(-1).sum())
        self.joint_steps += steps * changed.shape[-1]
        self.clipped_joint_steps += int(changed.sum())
        self.delta_sum_rad += float(delta.double().sum())
        self.delta_max_rad = max(self.delta_max_rad, float(delta.max()) if delta.numel() else 0.0)

    def summary(self, prefix: str) -> dict[str, Any]:
        """`<prefix>_frac` (env steps changed), `_joint_frac` (joint entries changed),
        `_frac_by_joint` (per joint, of env steps), `_mean_abs_delta_deg` / `_max_abs_delta_deg`
        (|a - clip(a)| over the changed entries) and `_env_steps`."""
        steps = max(self.env_steps, 1)
        return {
            f"{prefix}_frac": self.clipped_steps / steps,
            f"{prefix}_joint_frac": self.clipped_joint_steps / max(self.joint_steps, 1),
            f"{prefix}_frac_by_joint": [c / steps for c in self.clipped_by_joint],
            f"{prefix}_mean_abs_delta_deg": math.degrees(
                self.delta_sum_rad / max(self.clipped_joint_steps, 1)
            ),
            f"{prefix}_max_abs_delta_deg": math.degrees(self.delta_max_rad),
            f"{prefix}_env_steps": float(self.env_steps),
        }


@dataclass
class RolloutBatch:
    """E = G * K trajectories, S = ceil(max_episode_steps / C) decision slots. All on CPU."""

    obs: dict[str, Tensor]  # each (E, S, ...): observation at each decision
    instructions: list[str]  # (E,)
    tasks: list[str]  # (E,)
    tokens: Tensor  # (E, S, C, A) int64
    log_probs: Tensor  # (E, S, C, A) float32, from the sampler
    action_mask: Tensor  # (E, S, C) bool: action executed while the episode was running
    rewards: Tensor  # (E,) float32 in {0, 1}
    episode_len: Tensor  # (E,) int64 env steps executed (upstream finish_step)
    group_size: int  # K

    @property
    def num_trajectories(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def num_groups(self) -> int:
        return self.num_trajectories // self.group_size

    @property
    def group_rewards(self) -> Tensor:
        return self.rewards.view(self.num_groups, self.group_size)

    def select_groups(self, keep: Tensor) -> RolloutBatch:
        """Keep whole groups (keep: (G,) bool), preserving the contiguous group layout."""
        traj = keep.repeat_interleave(self.group_size).nonzero(as_tuple=True)[0]
        return self._index(traj)

    def _index(self, idx: Tensor) -> RolloutBatch:
        ids = idx.tolist()
        return RolloutBatch(
            obs={k: v[idx] for k, v in self.obs.items()},
            instructions=[self.instructions[i] for i in ids],
            tasks=[self.tasks[i] for i in ids],
            tokens=self.tokens[idx],
            log_probs=self.log_probs[idx],
            action_mask=self.action_mask[idx],
            rewards=self.rewards[idx],
            episode_len=self.episode_len[idx],
            group_size=self.group_size,
        )

    def save(self, path: Path) -> None:
        """torch.save every field (CPU tensors, lists, ints): the lr sweep's replayable batch."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f.name: getattr(self, f.name) for f in fields(self)}, path)

    @staticmethod
    def load(path: Path) -> RolloutBatch:
        data = torch.load(path, map_location=STORAGE, weights_only=True)
        return RolloutBatch(**data)

    @staticmethod
    def cat(batches: Sequence[RolloutBatch]) -> RolloutBatch:
        if not batches:
            raise ValueError("nothing to concatenate")
        k = batches[0].group_size
        if any(b.group_size != k for b in batches):
            raise ValueError("cannot concatenate batches with different group sizes")
        return RolloutBatch(
            obs={key: torch.cat([b.obs[key] for b in batches]) for key in batches[0].obs},
            instructions=[s for b in batches for s in b.instructions],
            tasks=[s for b in batches for s in b.tasks],
            tokens=torch.cat([b.tokens for b in batches]),
            log_probs=torch.cat([b.log_probs for b in batches]),
            action_mask=torch.cat([b.action_mask for b in batches]),
            rewards=torch.cat([b.rewards for b in batches]),
            episode_len=torch.cat([b.episode_len for b in batches]),
            group_size=k,
        )


def _flag(x: Any, n: int) -> Tensor:
    """Env flag (bool tensor on any device, or a scalar bool) -> (n,) bool on CPU."""
    t = torch.as_tensor(x).to(STORAGE, torch.bool)
    return t.expand(n).clone() if t.numel() == 1 else t.reshape(n)


def check_group_reset(
    obs: dict[str, Tensor], group_size: int, mode: Literal["error", "warn", "off"]
) -> float:
    """Largest fraction of one env's reset-observation values that differ from its group's first
    env (max over groups, envs and keys; 0.0 = every group shares its initial observation).

    Fails ("error" raises, "warn" logs) if any float value (joint positions) differs by more than
    RESET_FLOAT_ATOL, or more than RESET_IMAGE_DIFF_FRAC of an integer (uint8 image) observation
    differs at all. Measured on the CPU sim (WO-04, 3 tasks, 64 and 224 px): identical seeds are
    bit-identical; different seeds differ in >= 4.5% of the pixel values and >= 0.03 rad of qpos
    (the init-qpos noise alone). The image allowance is for the GPU sim, which keeps every
    sub-scene at its own world offset, so float32 pose rounding (~1e-6 m) can flip a few
    silhouette pixels between identical sub-scenes; it is ~45x below a real scene difference.
    """
    if mode == "off" or group_size == 1:
        return 0.0
    worst = 0.0
    bad = []
    for key, value in obs.items():
        v = value.to(STORAGE)
        grouped = v.reshape(-1, group_size, *v.shape[1:])
        first = grouped[:, :1]
        if v.is_floating_point():
            differs = (grouped.float() - first.float()).abs() > RESET_FLOAT_ATOL
            tol = 0.0
        else:
            differs = grouped != first
            tol = RESET_IMAGE_DIFF_FRAC
        frac = float(differs.reshape(grouped.shape[0], group_size, -1).float().mean(-1).max())
        worst = max(worst, frac)
        if frac > tol:
            bad.append(f"{key}: {frac:.2%} of values differ (allowed {tol:.2%})")
    if bad:
        msg = (
            f"envs reset with the same seed differ ({'; '.join(bad)}): a GRPO group would not "
            "share its prompt. The env must draw its initial state from the per-env seed."
        )
        if mode == "error":
            raise RuntimeError(msg + " Set rl.group_reset_check=warn to override.")
        log.warning(msg)
    return worst


@torch.no_grad()
def collect_rollouts(
    policy: RLPolicy,
    env: VecEnv,
    obs_fn: ObsFn,
    *,
    prompt_seeds: Sequence[int],
    group_size: int,
    temperature: float,
    max_episode_steps: int,
    task: str,
    group_reset_check: Literal["error", "warn", "off"] = "error",
    timer: RolloutTimer | None = None,
    action_clip: float | None = None,
    clip_stats: ClipStats | None = None,
    clip_apply: bool = True,
) -> tuple[RolloutBatch, float]:
    """Run one batch of episodes: len(prompt_seeds) prompts x group_size samples each.

    The env must have exactly len(prompt_seeds) * group_size parallel envs. Returns the batch and
    the group-reset max diff (0.0 when groups share their initial observation). With `timer`, the
    call's wall time is added to its sections (module docstring). `action_clip` (rad, None = off):
    the per-joint step clip of the module docstring, applied to the env action only (the batch
    keeps the sampled tokens / log-probs); `clip_apply=False` measures it without applying it.
    `clip_stats`, when given, accumulates the intervention counts (`ClipStats`).
    """
    tm = timer if timer is not None else RolloutTimer()
    instrument_render(env, tm)
    t_call = time.perf_counter()
    try:
        return _collect(
            policy, env, obs_fn, prompt_seeds, group_size, temperature, max_episode_steps, task,
            group_reset_check, tm, action_clip, clip_stats, clip_apply,
        )  # fmt: skip
    finally:
        instrument_render(env, None)
        tm.total_s += time.perf_counter() - t_call


def _collect(
    policy: RLPolicy,
    env: VecEnv,
    obs_fn: ObsFn,
    prompt_seeds: Sequence[int],
    group_size: int,
    temperature: float,
    max_episode_steps: int,
    task: str,
    group_reset_check: Literal["error", "warn", "off"],
    tm: RolloutTimer,
    action_clip: float | None = None,
    clip_stats: ClipStats | None = None,
    clip_apply: bool = True,
) -> tuple[RolloutBatch, float]:
    seeds = [int(s) for s in prompt_seeds for _ in range(group_size)]
    n = len(seeds)
    policy.eval()
    with tm.section("reset"):
        raw_obs, _ = env.reset(seed=seeds)
        tm.pending_render = 0.0  # reset's render stays in "reset"
    with tm.section("obs_fn"):
        pobs = obs_fn(raw_obs)
    with tm.section("reset"):
        reset_diff = check_group_reset(pobs, group_size, group_reset_check)
    instructions = list(env.unwrapped.get_instructions())
    if len(instructions) != n:
        raise ValueError(f"env returned {len(instructions)} instructions for {n} envs")

    active = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    episode_len = torch.zeros(n, dtype=torch.long)
    obs_steps: list[dict[str, Tensor]] = []
    token_steps: list[Tensor] = []
    logp_steps: list[Tensor] = []
    mask_steps: list[Tensor] = []
    chunk_len = 0
    # protocol 0.2's clip reference: the joint reading at reset, then the last executed action
    ref = pobs["state"].detach().float().clone() if action_clip is not None else None
    clip_changed: list[Tensor] = []  # per env step (N, A): entries the clip changed, running envs
    clip_delta: list[Tensor] = []  # per env step (N, A): |a - clip(a)| on those entries
    clip_active: list[Tensor] = []  # per env step (N,): running envs
    while bool(active.any()):
        with tm.section("policy"):
            step = policy.sample(pobs, instructions, temperature)
        chunk_len = int(step.tokens.shape[1])
        with tm.section("host_copy"):
            obs_steps.append({k: v.to(STORAGE).clone() for k, v in pobs.items()})
            token_steps.append(step.tokens.to(STORAGE, torch.long))
            logp_steps.append(step.log_probs.to(STORAGE, torch.float32))
        executed = torch.zeros(n, chunk_len, dtype=torch.bool)
        for c in range(chunk_len):
            if not bool(active.any()):
                break
            # the SAMPLED action; `step.tokens` / `step.log_probs` (stored above) belong to it
            action = step.actions[:, c]
            if ref is not None and action_clip is not None:
                if ref.shape != action.shape:
                    raise ValueError(
                        f"rl.rollout_action_clip_deg needs the policy-obs state to be the joint "
                        f"reading, shaped like the action: state {tuple(ref.shape)} vs action "
                        f"{tuple(action.shape)}"
                    )
                ref = ref.to(action.device, action.dtype)
                safe = clip_joint_step(action, ref, action_clip)  # protocol 0.2's function
                if clip_stats is not None:
                    changed = (safe != action) & active.to(action.device).unsqueeze(-1)
                    clip_changed.append(changed)
                    clip_delta.append((action - safe).abs() * changed)
                    clip_active.append(active.clone())
                if clip_apply:  # env action only: the stored tokens / log-probs stay the sampled
                    action = safe
                ref = action  # the executed action
            raw_obs, _, terminated, truncated, info = tm.env_step(env, action)
            executed[:, c] = active
            episode_len += active.long()
            with tm.section("host_copy"):
                hit = _flag(info["success"], n) & active
                ended = hit | _flag(terminated, n) | _flag(truncated, n)
            success |= hit
            ended |= episode_len >= max_episode_steps
            active &= ~ended
        mask_steps.append(executed)
        with tm.section("obs_fn"):
            pobs = obs_fn(raw_obs)

    if clip_stats is not None and clip_changed:  # one device -> host copy per call
        clip_stats.add(
            torch.stack(clip_changed).to(STORAGE),
            torch.stack(clip_delta).to(STORAGE),
            torch.stack(clip_active),
        )
    num_slots = math.ceil(max_episode_steps / chunk_len)
    if len(token_steps) > num_slots:
        raise RuntimeError(f"{len(token_steps)} decisions exceed {num_slots} slots")
    pad = num_slots - len(token_steps)

    def stack(steps: list[Tensor]) -> Tensor:
        out = torch.stack(steps, dim=1)
        if pad:
            out = torch.cat([out, out.new_zeros((n, pad, *out.shape[2:]))], dim=1)
        return out

    batch = RolloutBatch(
        obs={k: stack([o[k] for o in obs_steps]) for k in obs_steps[0]},
        instructions=instructions,
        tasks=[task] * n,
        tokens=stack(token_steps),
        log_probs=stack(logp_steps),
        action_mask=stack(mask_steps),
        rewards=success.float(),
        episode_len=episode_len,
        group_size=group_size,
    )
    return batch, reset_diff
