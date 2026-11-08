"""WO-03/04 entry point: GRPO post-training (ADR-0002).

    uv run scripts/wo03_rl.py --preset tiny [--seed N]      # CPU, toy env + 2-layer LoRA policy
    uv run scripts/wo03_rl.py --preset rl --estimate        # print hours / $ only, run nothing
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --probe-steps 3   # GPU probe (pod job)
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --seed N          # WO-04 run (paid)
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --seed N --resume <run>/checkpoints/step_N
    # Track B diagnostics (review section 8), each a paid GPU job needing its own owner go:
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --lr-sweep 5e-5,2e-4,5e-4,1e-3 \\
        [--sweep-adam-steps 8] [--sweep-batch <run>/sweep_batch.pt]
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --probe-temperatures 0.7,1.0,1.3 \\
        [--probe-groups 64]
    uv run scripts/wo03_rl.py --preset rl --adapter <sft> --probe-clip [--probe-groups 32]
    # add --estimate to any probe command for its projected hours / $ without running it
Every probe uses the preset's rl.rollout_action_clip_deg (configs/rl.yaml: 4.0, owner 2026-11-08)
for its rollouts and prints it; --probe-clip compares that clip against none on paired rollouts.

`rl.env: toy` selects the synthetic env (src/rl/toy_env.py); anything else uses src.env.make_env
(GPU-parallel sim when `device` is cuda, serial CPU envs on cpu). The policy comes from the
registry in src/rl/policy_api.py keyed by `policy.base` (WO-04: OpenVLA-OFT, LoRA, the SFT adapter
from `policy.adapter` / `--adapter`: a local dir or a Hub `user/repo[/path][@rev]`). Real imports
are lazy so `tiny` runs without ManiSkill or VLA dependencies.
Writes runs/<id>/record.json, runs/<id>/metrics.jsonl and runs/<id>/checkpoints/.

Record status: finished | stopped_flat (flat spend gate: no protocol eval, diagnose) |
stopped_peak (decline or entropy spend gate: evaluated like a finished run; `stop_reason` names
the gate) | interrupted | failed | probe. The status words are the ones scripts/pod/wo04_job.sh
dispatches on.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import REPO_ROOT, Config, add_config_args, config_hash, load_config
from src.eval import protocol
from src.eval.seed_ranges import check_rl_seed
from src.rl.policy_api import build_policy
from src.rl.rollout import TIMING_KEYS, ObsFn, RolloutBatch
from src.rl.trainer import (
    EnvFactory,
    GRPOConfig,
    GRPOTrainer,
    eval_schedule,
    gate_check_step,
    ppo_minibatches,
    read_checkpoint_state,
)
from src.runs import device_info, hourly_usd

log = logging.getLogger("wo03_rl")

STATUS_BY_STOP = {"flat": "stopped_flat", "decline": "stopped_peak", "entropy": "stopped_peak"}


# --- run record: local helper, to be replaced by the shared src/runs.py (other branch) --------
def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except OSError:
        return ""


def _write_run_record(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def _work_order(cfg: Config) -> str:
    return "WO-04" if cfg["preset"] == "rl" else "WO-03"


def _new_run_record(cfg: Config, run_id: str, device: str) -> dict[str, Any]:
    dev = device_info(device)
    return {
        "run_id": run_id,
        "wo": _work_order(cfg),
        "preset": cfg["preset"],
        "config_hash": config_hash(cfg),
        "config": cfg,
        "git_commit": _git("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "device": device,
        "gpu": dev["name"],
        "runpod_pod_id": dev["runpod_pod_id"],
        "hourly_usd": hourly_usd(dev),  # RunPod list price by GPU name; 0 off RunPod
        "start_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "end_utc": None,
        "wall_clock_h": None,
        "dollars": None,  # hourly_usd x wall clock of this process (pod setup/eval not included)
        "status": "running",
    }


def _finish_record(record: dict[str, Any], wall_s: float) -> None:
    record["end_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["wall_clock_h"] = round(wall_s / 3600.0, 4)
    record["dollars"] = round(float(record["hourly_usd"]) * wall_s / 3600.0, 4)


# --- cost estimate (cfg cost.throughput constants; the probes replace them with measurements) --
def _shape(cfg: Config) -> dict[str, float]:
    """Per-step sizes shared by the estimates (see the arithmetic in configs/rl.yaml)."""
    rl = GRPOConfig.from_config(cfg)
    tp = cfg["cost"]["throughput"]
    chunk = int(cfg["policy"]["action_chunk"])
    horizon = rl.max_episode_steps or int(cfg["env"]["max_episode_steps"])  # RL horizon
    ep_len = float(horizon) * float(tp["mean_episode_frac"])
    per_traj = math.ceil(ep_len / chunk)
    return {
        "ep_len": ep_len,
        "decisions_per_traj": float(per_traj),
        "trained": float(rl.rollout_batch * rl.samples_per_prompt),
        "actions_per_s": float(tp["rollout_actions_per_s"]),
        "forward_per_s": float(tp["forward_decisions_per_s"]),
        "update_per_s": float(tp["update_decisions_per_s"]),
        "overhead": float(tp["rollout_overhead"]),
    }


def estimate_cost(cfg: Config) -> dict[str, float]:
    """Hours and dollars for one run. The arithmetic is spelled out in configs/rl.yaml."""
    rl = GRPOConfig.from_config(cfg)
    sh = _shape(cfg)
    trained = sh["trained"]
    episodes = trained * sh["overhead"]  # dynamic-sampling refill
    actions = episodes * sh["ep_len"]
    decisions = trained * sh["decisions_per_traj"]
    rollout_s = actions / sh["actions_per_s"]
    recompute_s = decisions / sh["forward_per_s"] if rl.recompute_old_log_probs else 0.0
    update_s = rl.ppo_epochs * decisions / sh["update_per_s"]
    diag_s = 0.0
    if rl.diag_every:
        traj = rl.diag_trajectories or trained
        forwards = 1 + (0 if rl.recompute_old_log_probs else 1) + (2 if rl.diag_argmax else 0)
        diag_s = forwards * traj * sh["decisions_per_traj"] / sh["forward_per_s"] / rl.diag_every
    step_s = rollout_s + recompute_s + update_s + diag_s
    schedule = eval_schedule(rl)
    n = rl.rollout_envs
    eval_tasks = len(rl.train_tasks or cfg["env"]["tasks"])  # periodic evals: trained tasks only

    def eval_seconds(upto: int) -> float:
        episodes = sum(math.ceil(ep / n) * n for s, ep in schedule.items() if s <= upto)
        return eval_tasks * episodes * sh["ep_len"] / sh["actions_per_s"]

    eval_episodes = sum(math.ceil(ep / n) * n for ep in schedule.values())  # per task, run total
    hours = (rl.steps * step_s + eval_seconds(rl.steps)) / 3600.0
    check = gate_check_step(rl)
    usd = float(cfg["cost"]["usd_per_hour"])
    out = {
        "episodes_per_step": episodes,
        "actions_per_step": actions,
        "decisions_trained_per_step": decisions,
        "rollout_s_per_step": rollout_s,
        "recompute_s_per_step": recompute_s,
        "update_s_per_step": update_s,
        "diag_s_per_step": diag_s,
        "step_min": step_s / 60.0,
        "evals": float(len(schedule)),
        "eval_tasks": float(eval_tasks),
        "eval_episodes_per_task": float(eval_episodes),
        "eval_h": eval_seconds(rl.steps) / 3600.0,
        "hours": hours,
        "dollars": hours * usd,
    }
    if check:
        out["gate_check_step"] = float(check)
        out["gate_check_stop_h"] = (check * step_s + eval_seconds(check)) / 3600.0
        out["gate_check_stop_dollars"] = out["gate_check_stop_h"] * usd
    return out


def estimate_probes(
    cfg: Config,
    *,
    lrs: list[float] | None,
    adam_steps: int,
    stored_batch: bool,
    temperatures: list[float] | None,
    groups: int,
    clip_groups: int | None = None,
) -> dict[str, float]:
    """Projected GPU time of the Track B probes from cost.throughput (not included: setup, and
    the 7B load + GPU env build of each separate invocation).

    lr sweep drivers: one rollout collection (worst case: the refill cap), 2 no-grad forwards of
    the whole batch before the first lr (greedy tokens; log-probs, which serve every lr as old
    log-probs and diag reference), then per lr `adam_steps` mini-batch forward+backward passes
    and 2 no-grad forwards of the whole batch (post scores + greedy tokens).
    Temperature probe drivers: env stepping + rendering + sampling forwards of
    ceil(groups / groups-per-env-batch) env batches per temperature (no update).
    Clip probe (`clip_groups`): the same per env batch, ceil(clip_groups / groups-per-env-batch)
    env batches per arm, 2 arms (the clip itself is a per-step clamp of 6 joints: not costed)."""
    rl = GRPOConfig.from_config(cfg)
    sh = _shape(cfg)
    out: dict[str, float] = {}
    usd = float(cfg["cost"]["usd_per_hour"])
    batch_actions = rl.rollout_envs * sh["ep_len"]
    gpb = rl.rollout_envs // rl.samples_per_prompt
    if lrs:
        cap_batches = math.ceil(rl.rollout_batch / gpb) * rl.max_rollout_rounds
        collect_s = 0.0 if stored_batch else cap_batches * batch_actions / sh["actions_per_s"]
        batch_dec = sh["trained"] * sh["decisions_per_traj"]
        # the largest of the trainer's equal whole-group mini-batches of a full batch
        k = rl.samples_per_prompt
        mb_groups = max(len(g) for g in ppo_minibatches(rl.rollout_batch, rl.ppo_mini_batch // k))
        mb_dec = mb_groups * k * sh["decisions_per_traj"]
        fwd = batch_dec / sh["forward_per_s"]
        per_lr = adam_steps * mb_dec / sh["update_per_s"] + 2 * fwd
        total = collect_s + 2 * fwd + len(lrs) * per_lr
        out.update(
            {
                "lr_sweep_collect_s": collect_s,
                "lr_sweep_s_per_lr": per_lr,
                "lr_sweep_adam_s_per_lr": adam_steps * mb_dec / sh["update_per_s"],
                "lr_sweep_h": total / 3600.0,
                "lr_sweep_dollars": total / 3600.0 * usd,
            }
        )
    if temperatures:
        batches = math.ceil(groups / gpb)
        per_t = batches * batch_actions / sh["actions_per_s"]
        total = per_t * len(temperatures)
        out.update(
            {
                "temp_probe_env_batches_per_t": float(batches),
                "temp_probe_s_per_t": per_t,
                "temp_probe_h": total / 3600.0,
                "temp_probe_dollars": total / 3600.0 * usd,
            }
        )
    if clip_groups:
        batches = math.ceil(clip_groups / gpb)
        total = 2 * batches * batch_actions / sh["actions_per_s"]
        out.update(
            {
                "clip_probe_env_batches_per_arm": float(batches),
                "clip_probe_s": total,
                "clip_probe_h": total / 3600.0,
                "clip_probe_dollars": total / 3600.0 * usd,
            }
        )
    if out:
        out["setup_h"] = float(cfg["cost"].get("setup_hours", 0.0))
    return out


def project_from_probe(cfg: Config, probe: dict[str, Any], usd_per_hour: float) -> dict[str, Any]:
    """Hours / $ for one seed from the probe's measured times (configs/rl.yaml step count)."""
    rl = GRPOConfig.from_config(cfg)
    n = rl.rollout_envs
    tasks = len(rl.train_tasks or cfg["env"]["tasks"])  # periodic evals: trained tasks only
    env_steps = int(cfg["env"]["max_episode_steps"])
    horizon_scale = env_steps / (rl.max_episode_steps or env_steps)
    s_step = float(probe["s_per_step_median"])
    s_eval_batch = float(probe["eval_s_per_env_batch"])
    schedule = eval_schedule(rl)
    check = gate_check_step(rl)

    def eval_s(upto: int) -> float:
        return sum(math.ceil(ep / n) * tasks for s, ep in schedule.items() if s <= upto) * (
            s_eval_batch
        )

    # worst case: dynamic-sampling refill hits the cap (max_rollout_rounds full rounds) every step
    round_batches = math.ceil(rl.rollout_batch / (n // rl.samples_per_prompt))
    rounds = rl.max_rollout_rounds if rl.dynamic_sampling else 1
    s_step_worst = round_batches * rounds * float(probe["rollout_s_per_env_batch"])
    s_step_worst += float(probe["update_s_median"])
    protocol_episodes = (
        len(protocol.TASKS)
        * len(protocol.CELLS)
        * len(protocol.SIM_SEEDS)
        * protocol.SIM_EPISODES_PER_CELL
    )
    out = {
        "steps": rl.steps,
        "run_h": (rl.steps * s_step + eval_s(rl.steps)) / 3600.0,
        "gate_check_step": check,
        "gate_check_stop_h": (check * s_step + eval_s(check)) / 3600.0,
        "worst_case_h": (rl.steps * s_step_worst + eval_s(rl.steps)) / 3600.0,
        # the final scripts/wo02_eval.py protocol eval at the probe's greedy rate (estimate),
        # scaled from the RL horizon to the protocol's env.max_episode_steps
        "protocol_eval_h": protocol_episodes * s_eval_batch / n / 3600.0 * horizon_scale,
        "usd_per_hour": usd_per_hour,
    }
    for key in ("run", "gate_check_stop", "worst_case", "protocol_eval"):
        out[f"{key}_usd"] = out[f"{key}_h"] * usd_per_hour
    return out


def print_probe(cfg: Config, probe: dict[str, Any], proj: dict[str, Any], gpu: str) -> None:
    """The lines the pod job posts for the owner's approval (all prefixed `[wo03_rl] PROBE`;
    scripts/pod/wo04_job.sh shows the last 6, so this stays at <= 6 lines)."""
    rl = GRPOConfig.from_config(cfg)
    base = probe["baseline"]
    tasks = " ".join(f"{t} {base[f'eval/success/{t}']:.3f}" for t in probe["train_tasks"])
    mem = (
        f"torch peak {probe['mem/torch_peak_allocated_gb']:.1f} GB allocated / "
        f"{probe['mem/torch_peak_reserved_gb']:.1f} GB reserved; device "
        f"{probe['mem/device_used_gb']:.1f} of {probe['mem/device_total_gb']:.1f} GB in use"
        if "mem/device_used_gb" in probe
        else "n/a (no CUDA)"
    )
    split = probe["rollout_split_s_per_env_batch"]
    batch_s = max(float(probe["rollout_s_per_env_batch"]), 1e-12)
    split_txt = ", ".join(f"{k} {split[k]:.1f} s ({split[k] / batch_s:.0%})" for k in TIMING_KEYS)
    ratio = probe["first_minibatch_max_abs_log_ratio"]
    speed = (
        f"s/step median {probe['s_per_step_median']:.1f} (max {probe['s_per_step_max']:.1f}; "
        f"rollout {probe['rollout_s_per_env_batch']:.1f} s per {rl.rollout_envs}-env batch = "
        f"{split_txt}; update {probe['update_s_median']:.1f} s); rollout "
        f"{probe['rollout_episodes_per_hour']:,.0f} episodes/h; greedy eval "
        f"{probe['eval_s_per_env_batch']:.1f} s per batch (first, warm-up: "
        f"{probe['eval_s_first_env_batch']:.1f} s); env setup {probe['env_setup_s']:.0f} s"
    )
    baseline = (
        f"SFT baseline ({rl.eval_episodes} greedy episodes/trained task, a sanity check; the "
        f"protocol eval is the measurement): {tasks} (mean {base['eval/success']:.3f}, horizon "
        f"{probe.get('max_episode_steps', '?')})"
    )
    d = probe.get("diag_post", {})
    diag = (
        f"post-update on {rl.diag_trajectories or 'all'} held trajectories (median): |dlogp| "
        f"{d['mean_abs_dlogp']:.4f} (A>0 {d['mean_dlogp_pos_adv']:+.4f}, A<0 "
        f"{d['mean_dlogp_neg_adv']:+.4f}), k3 KL {d['k3_kl']:.2e}, clip {d['clip_frac']:.4f}"
        if d
        else "no fixed-batch diagnostic (rl.diag_every 0 or every update skipped)"
    )
    checks = (
        f"checks: first-minibatch |log ratio| max {ratio:.4f} / mean "
        f"{probe['first_minibatch_mean_abs_log_ratio']:.5f} (max > 0.05 -> consider "
        "rl.recompute_old_log_probs: true), group-reset diff frac "
        f"{probe['group_reset_diff_frac']:.5f}, groups dropped {probe['frac_groups_dropped']:.0%} "
        f"(refill cap hit on {probe['refill_cap_hit_frac']:.0%} of steps); "
        f"{_clip_text(rl.rollout_action_clip_deg, probe.get('action_clip_frac'))}; {diag}"
    )
    flat_stop = (
        "flat-gate stop" if rl.gate_stop else "flat-gate check (logged only, gate_stop false)"
    )
    projected = (
        f"projected (estimate from {probe['steps']} probe steps, step 1 excluded): full run "
        f"{proj['steps']} steps = {proj['run_h']:.1f} h = ${proj['run_usd']:.2f} per seed on {gpu} "
        f"@ ${proj['usd_per_hour']:.2f}/h; {flat_stop} at step {proj['gate_check_step']} = "
        f"{proj['gate_check_stop_h']:.1f} h = ${proj['gate_check_stop_usd']:.2f}; worst case "
        f"(refill cap every step) {proj['worst_case_h']:.1f} h = ${proj['worst_case_usd']:.2f}; "
        f"+ protocol eval {proj['protocol_eval_h']:.1f} h = ${proj['protocol_eval_usd']:.2f}"
    )
    lines = [speed, f"VRAM: {mem}", baseline, checks, projected]
    for line in lines:
        print(f"[wo03_rl] PROBE {line}", flush=True)
    if ratio > 0.05:
        print("[wo03_rl] PROBE WARNING sampler/update log-probs disagree", flush=True)


def _clip_text(deg: float | None, frac: float | None = None, verb: str = "changed") -> str:
    """The rollout action clip as the probe lines print it."""
    if deg is None:
        return "action clip off (rl.rollout_action_clip_deg null)"
    txt = f"action clip {deg:g} deg/step"
    return txt if frac is None else f"{txt} ({verb} {frac:.1%} of env steps)"


def _clip_detail(row: dict[str, Any], prefix: str, verb: str) -> str:
    return (
        f"clip {verb} {row[f'{prefix}_frac']:.1%} of env steps / "
        f"{row[f'{prefix}_joint_frac']:.1%} of joint-steps (mean |delta| "
        f"{row[f'{prefix}_mean_abs_delta_deg']:.2f} deg, max {row[f'{prefix}_max_abs_delta_deg']:.1f})"
    )


def print_lr_sweep(out: dict[str, Any]) -> None:
    b = out["batch"]
    clip = out.get("rollout_action_clip_deg")
    clipped = (
        f"; {_clip_detail(b, 'rollout/action_clip', 'changed')}"
        if "rollout/action_clip_frac" in b
        else ""
    )
    print(
        f"[wo03_rl] LR-SWEEP batch: {b['trajectories']} trajectories ({b['groups']} groups, "
        f"{b['decisions']} decisions, tasks {b['tasks']}); {out['adam_steps']} Adam steps per lr; "
        f"rollouts with {_clip_text(clip)}{clipped}; on-batch movement after the update (all "
        "tokens of the batch):",
        flush=True,
    )
    for r in out["rows"]:
        print(
            f"[wo03_rl] LR-SWEEP lr {r['lr']:.1e}: k3 KL {r['diag/post/k3_kl']:.3e}, clip "
            f"{r['diag/post/clip_frac']:.4f} (outside {r['diag/post/frac_outside_clip']:.4f}), "
            f"|dlogp| {r['diag/post/mean_abs_dlogp']:.4f} (A>0 {r['diag/post/mean_dlogp_pos_adv']:+.4f}"
            f", A<0 {r['diag/post/mean_dlogp_neg_adv']:+.4f}), argmax flips "
            f"{r['diag/post/argmax_flip_frac']:.4f}, grad_norm {r['train/grad_norm']:.3f}, "
            f"{r['time/lr_s']:.0f} s",
            flush=True,
        )


def print_temperature_probe(out: dict[str, Any]) -> None:
    print(
        f"[wo03_rl] TEMP-PROBE {out['groups']} groups per temperature (same prompts each time), "
        f"rollouts only, {_clip_text(out.get('rollout_action_clip_deg'))}:",
        flush=True,
    )
    for r in out["rows"]:
        clip = (
            f", {_clip_detail(r, 'probe/action_clip', 'changed')}"
            if "probe/action_clip_frac" in r
            else ""
        )
        print(
            f"[wo03_rl] TEMP-PROBE T {r['temperature']:.2f}: sampled success "
            f"{r['probe/success_rate']:.3f} (SE {r['probe/success_rate_se']:.3f}), groups dropped "
            f"{r['probe/frac_groups_dropped']:.3f} (all-0 {r['probe/frac_groups_all_zero']:.3f}, "
            f"all-1 {r['probe/frac_groups_all_one']:.3f}), entropy {r['probe/entropy']:.3f} "
            f"nats/token, {r['probe/env_batches_per_update']:.1f} env batches per update"
            f"{' (refill cap binds)' if r['probe/refill_cap_binds'] else ''}{clip}, "
            f"{r['probe/s_per_env_batch']:.1f} s per env batch",
            flush=True,
        )


def print_clip_probe(out: dict[str, Any]) -> None:
    """4 lines prefixed `[wo03_rl] CLIP-PROBE` (scripts/pod/diag_job.sh shows the last 4)."""
    print(
        f"[wo03_rl] CLIP-PROBE {out['groups']} groups x K={out['samples_per_prompt']} per arm, "
        f"T {out['temperature']:.2f}, clip {out['rollout_action_clip_deg']:g} deg/step vs off; "
        "same prompts and the same sampling seeds per env batch in both arms; rollouts only:",
        flush=True,
    )
    for r in out["rows"]:
        verb = "changed" if r["clip_applied"] else "WOULD have changed"
        print(
            f"[wo03_rl] CLIP-PROBE arm {r['arm']}: sampled success {r['probe/success_rate']:.3f} "
            f"(SE {r['probe/success_rate_se']:.3f}), groups dropped "
            f"{r['probe/frac_groups_dropped']:.3f} (all-0 {r['probe/frac_groups_all_zero']:.3f}, "
            f"all-1 {r['probe/frac_groups_all_one']:.3f}), entropy {r['probe/entropy']:.3f}, mean "
            f"episode {r['probe/mean_episode_len']:.1f} steps, "
            f"{_clip_detail(r, 'probe/action_clip', verb)}, "
            f"{r['probe/s_per_env_batch']:.1f} s per env batch",
            flush=True,
        )
    p = out["paired"]
    print(
        f"[wo03_rl] CLIP-PROBE paired: success on - off {p['success_diff_on_minus_off']:+.3f} (SE "
        f"{p['success_diff_se']:.3f} over {p['groups']:.0f} groups); episodes both "
        f"{p['episodes_both_success']:.0f} / off only {p['episodes_off_only']:.0f} / on only "
        f"{p['episodes_on_only']:.0f} / neither {p['episodes_neither']:.0f} (exact McNemar p "
        f"{p['episodes_mcnemar_exact_p']:.3f}, episodes in a group are not independent); groups "
        f"dropped both {p['groups_dropped_both']:.0f} / off only "
        f"{p['groups_dropped_off_only']:.0f} / on only {p['groups_dropped_on_only']:.0f} / kept "
        f"both {p['groups_kept_both']:.0f}; first-decision tokens identical "
        f"{p['first_decision_token_match']:.3f} (pairing check)",
        flush=True,
    )


# --- wiring --------------------------------------------------------------------------------
def build_env(cfg: Config) -> tuple[EnvFactory, ObsFn, int]:
    """(env factory, obs -> policy-obs fn, max episode steps) for the configured env."""
    if cfg["rl"].get("env") == "toy":
        from src.rl.toy_env import make_toy_env_factory, policy_obs

        return make_toy_env_factory(cfg), policy_obs, int(cfg["toy"]["max_episode_steps"])

    from src.env import make_env, policy_obs  # WO-01; ManiSkill only needed here

    def factory(task: str, num_envs: int) -> Any:
        return make_env(cfg, task, num_envs=num_envs, cell="seen", render=True)

    return factory, policy_obs, int(cfg["env"]["max_episode_steps"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _floats(text: str) -> list[float]:
    try:
        values = [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from exc
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive numbers, got {text!r}")
    return values


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--device", default=None, help="override cfg device (cpu, cuda, cuda:1)")
    parser.add_argument("--seed", type=int, default=None, help="override cfg seed")
    parser.add_argument(
        "--adapter", default=None, help="override policy.adapter (SFT checkpoint dir or Hub path)"
    )
    parser.add_argument("--estimate", action="store_true", help="print hours/$ and exit")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue from a local checkpoint dir (runs/<id>/checkpoints/step_N): adapter, "
        "optimizer, scheduler, step counter and RNG state. Prefer the newest step_N checkpoint "
        "without a suffix: an _interrupted one is refused unless --resume-approximate",
    )
    parser.add_argument(
        "--resume-approximate",
        action="store_true",
        help="with --resume: accept a step_N_interrupted checkpoint, whose weights may already "
        "hold part of step N; step N is re-run on top (record.resume_approximate = true)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help="baseline eval + this many RL steps; print s/step, VRAM, projected h/$; save nothing",
    )
    mode.add_argument(
        "--lr-sweep",
        type=_floats,
        default=None,
        metavar="LR1,LR2,...",
        help="collect one rollout batch, then per lr restore the adapter + optimizer and run "
        "--sweep-adam-steps updates; log on-batch k3 KL, clip, dlogp by advantage sign, flips",
    )
    mode.add_argument(
        "--probe-temperatures",
        type=_floats,
        default=None,
        metavar="T1,T2,...",
        help="rollouts only: group-drop rate and sampled success per temperature",
    )
    mode.add_argument(
        "--probe-clip",
        action="store_true",
        help="rollouts only, paired: the preset's rl.rollout_action_clip_deg vs no clip on the "
        "same prompts and sampling seeds; success, group-drop rate, clip intervention per arm",
    )
    parser.add_argument("--sweep-adam-steps", type=int, default=8)
    parser.add_argument(
        "--sweep-batch", type=Path, default=None, help="replay a stored sweep_batch.pt"
    )
    parser.add_argument(
        "--probe-groups",
        type=int,
        default=None,
        help="groups (prompts) per probed temperature (default 64) or per clip-probe arm "
        "(default 32)",
    )
    args = parser.parse_args(argv)
    if args.probe_groups is None:
        args.probe_groups = 32 if args.probe_clip else 64
    if args.probe_steps is not None and args.probe_steps < 1:
        parser.error("--probe-steps must be >= 1 (0 would start the full run)")
    if args.sweep_adam_steps < 1 or args.probe_groups < 1:
        parser.error("--sweep-adam-steps and --probe-groups must be >= 1")
    if args.sweep_batch is not None and args.lr_sweep is None:
        parser.error("--sweep-batch only applies to --lr-sweep")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.preset, args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.adapter is not None:
        cfg["policy"]["adapter"] = args.adapter
    device, seed = str(cfg["device"]), int(cfg["seed"])
    rl = GRPOConfig.from_config(cfg)  # validate before touching envs or weights
    if cfg["rl"].get("env") != "toy":  # ManiSkill prompts must not reuse held-out episode seeds
        check_rl_seed(seed)
    bad_tasks = sorted(set(rl.train_tasks) - set(cfg["env"]["tasks"]))
    if bad_tasks:  # GRPOTrainer checks this too, but only after the 7B load
        parser.error(f"rl.train_tasks {bad_tasks} not in env.tasks {list(cfg['env']['tasks'])}")
    if args.probe_clip and rl.rollout_action_clip_deg is None:
        parser.error("--probe-clip needs rl.rollout_action_clip_deg set in the preset (it is null)")
    if args.estimate:
        gpu = cfg["cost"]["gpu"]
        if args.lr_sweep or args.probe_temperatures or args.probe_clip:
            est = estimate_probes(
                cfg,
                lrs=args.lr_sweep,
                adam_steps=args.sweep_adam_steps,
                stored_batch=args.sweep_batch is not None,
                temperatures=args.probe_temperatures,
                groups=args.probe_groups,
                clip_groups=args.probe_groups if args.probe_clip else None,
            )
            for key, value in est.items():
                print(f"{key:>28}: {value:,.2f}")
            print(f"{'rollout_action_clip_deg':>28}: {rl.rollout_action_clip_deg}")
            hours = sum(est.get(f"{k}_h", 0.0) for k in ("lr_sweep", "temp_probe", "clip_probe"))
            print(
                f"estimated: {gpu} x {hours:.2f} h = ${hours * float(cfg['cost']['usd_per_hour']):.2f}"
                f" + setup {est['setup_h']:.1f} h per pod (config {config_hash(cfg)[:12]})"
            )
            return est
        est = estimate_cost(cfg)
        for key, value in est.items():
            print(f"{key:>28}: {value:,.2f}")
        print(
            f"estimated: {gpu} x {est['hours']:.1f} h = ${est['dollars']:.0f} (config {config_hash(cfg)[:12]})"
        )
        return est
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {device!r} has no CUDA here (use --device cpu or --preset tiny)"
        )
    # every check below fails before the run record, the env build and the 7B load
    if args.resume_approximate and args.resume is None:
        parser.error("--resume-approximate needs --resume")
    if args.sweep_batch is not None and not args.sweep_batch.is_file():
        parser.error(f"--sweep-batch {args.sweep_batch}: no such file")
    if args.resume is not None:
        try:
            st = read_checkpoint_state(args.resume)  # FileNotFoundError: no trainer_state.json
        except ValueError as exc:
            parser.error(f"--resume: {exc}")
        if not args.resume_approximate and int(st["step"]) != int(st["completed_step"]):
            parser.error(
                f"--resume {args.resume} is an _interrupted checkpoint (step {st['step']}, "
                f"completed_step {st['completed_step']}): resume from the newest step_N "
                "checkpoint without a suffix, or pass --resume-approximate"
            )
    seed_everything(seed)

    t_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    kind = _work_order(cfg).lower().replace("-", "")
    if args.probe_steps:
        kind += "probe"
    elif args.lr_sweep:
        kind += "lrsweep"
    elif args.probe_temperatures:
        kind += "tempprobe"
    elif args.probe_clip:
        kind += "clipprobe"
    run_id = f"{stamp}-{kind}-{cfg['preset']}-s{seed}-{uuid.uuid4().hex[:6]}"
    run_dir = REPO_ROOT / "runs" / run_id
    record = _new_run_record(cfg, run_id, device)
    # `timeout` (the pod job's soft deadline) sends SIGTERM: exit through the finally blocks so the
    # trainer saves an interrupted checkpoint and the run record gets its end time and dollars,
    # also when it arrives during the env build or the 7B load (installed before the first write)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    _write_run_record(run_dir, record)
    log.info("run %s (config %s)", run_id, record["config_hash"][:12])
    trainer: GRPOTrainer | None = None
    try:
        env_factory, obs_fn, max_steps = build_env(cfg)
        policy = build_policy(cfg, device)
        describe = getattr(policy, "describe", None)
        record["policy"] = describe() if callable(describe) else None
        trainer = GRPOTrainer(
            rl=rl,
            policy=policy,
            env_factory=env_factory,
            obs_fn=obs_fn,
            tasks=list(cfg["env"]["tasks"]),
            max_episode_steps=max_steps,
            run_dir=run_dir,
            seed=seed,
        )
        if args.resume is not None:
            # restores RNG state too: after seed_everything
            trainer.load_checkpoint(args.resume, allow_partial=args.resume_approximate)
            record["resumed_from"] = str(args.resume)
            if args.resume_approximate:
                record["resume_approximate"] = True
            if trainer.resume_overrides:  # rl key -> [checkpoint, this run]; this run's apply
                record["resume_overrides"] = trainer.resume_overrides
        if args.probe_steps:
            probe = trainer.probe(args.probe_steps)
            cost = cfg.get("cost", {})  # measured GPU's list price on RunPod, else the config's
            usd = float(record["hourly_usd"]) or float(cost.get("usd_per_hour", 0.0))
            gpu = str(record["gpu"]) if record["hourly_usd"] else str(cost.get("gpu", device))
            proj = project_from_probe(cfg, probe, usd)
            record["status"] = "probe"
            record["probe"] = {k: v for k, v in probe.items() if k != "rows"}
            record["projection"] = proj
            print_probe(cfg, probe, proj, gpu)
            return {**probe, "projection": proj, "run_dir": str(run_dir)}
        if args.lr_sweep:
            stored = RolloutBatch.load(args.sweep_batch) if args.sweep_batch else None
            sweep = trainer.lr_sweep(args.lr_sweep, args.sweep_adam_steps, stored)
            record["status"] = "probe"
            record["lr_sweep"] = sweep
            print_lr_sweep(sweep)
            return {**sweep, "run_dir": str(run_dir)}
        if args.probe_temperatures:
            temps = trainer.probe_temperatures(args.probe_temperatures, args.probe_groups)
            record["status"] = "probe"
            record["temperature_probe"] = temps
            print_temperature_probe(temps)
            return {**temps, "run_dir": str(run_dir)}
        if args.probe_clip:
            clip_out = trainer.probe_clip(args.probe_groups)
            record["status"] = "probe"
            record["clip_probe"] = clip_out
            print_clip_probe(clip_out)
            return {**clip_out, "run_dir": str(run_dir)}
        summary = trainer.run()
        reason = summary["stop_reason"]
        record["status"] = STATUS_BY_STOP[reason] if reason else "finished"
        record["stop_reason"] = reason
        record["summary"] = summary
    except (KeyboardInterrupt, SystemExit):
        record["status"] = "interrupted"
        raise
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        if trainer is not None and not (
            args.probe_steps or args.lr_sweep or args.probe_temperatures or args.probe_clip
        ):
            # when each spend gate would have stopped the run (also with rl.gate_stop false, and
            # for an interrupted run): first step per gate, the flat-gate decision at its check
            record["gate_decisions"] = trainer.gate_summary()
        _finish_record(record, time.perf_counter() - t_start)
        _write_run_record(run_dir, record)
    log.info(
        "done: greedy eval (sanity) %s -> %s (best %s) in %d steps, %.1fs, stop %s; %s",
        summary["initial_eval_success"],
        summary["final_eval_success"],
        summary["best_eval_success"],
        summary["steps_completed"],
        summary["wall_time_s"],
        summary["stop_reason"],
        run_dir,
    )
    return summary


if __name__ == "__main__":
    main()
