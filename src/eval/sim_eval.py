"""Batched greedy sim evaluation: tasks x cells x seeds, success with 95% Wilson intervals.

Env contract (src/env): `make_env(cfg, task, num_envs=, cell=)` -> vector env with
`reset(seed=list)`, `step(actions (N, 6))` -> (obs, reward, terminated, truncated, info) where
`info["success"]` is (N,) bool, and `env.unwrapped.get_instructions()`; `policy_obs(obs)` ->
{"image", "state"}. An episode succeeds if `success` fires before it ends (same rule as the RL
rollout in src/rl/rollout.py). A policy decision executes its whole action chunk open loop.

Protocol constants (tasks, cells, seeds, episodes per cell, Wilson z, hash) come from
src/eval/protocol.py and are passed in, so this module stays testable on its own.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

ActFn = Callable[[Mapping[str, Tensor], Sequence[str]], Tensor]  # obs, instructions -> (N, C, A)
ObsFn = Callable[[Any], Mapping[str, Tensor]]
MakeEnv = Callable[[str, str, int], Any]  # task, cell, num_envs -> env
Interval = Callable[[int, int], tuple[float, float]]

EVAL_SEED_BASE = 100_000_000  # far above demo / RL training seeds


def episode_seeds(protocol_seed: int, task_index: int, cell_index: int, n: int) -> list[int]:
    """Deterministic, disjoint per (protocol seed, task, cell) episode seeds."""
    base = EVAL_SEED_BASE + 1_000_000 * protocol_seed + 10_000 * (task_index * 8 + cell_index)
    return [base + i for i in range(n)]


def _flag(x: Any, n: int) -> Tensor:
    t = torch.as_tensor(x).reshape(-1).to("cpu", torch.bool)
    return t.expand(n) if t.numel() == 1 else t


def clip_joint_step(target: Tensor, ref: Tensor, bound: float) -> Tensor:
    """Per-joint step clip: each entry of `target` moved to within `bound` of `ref` (protocol 0.2;
    the step clip src/robot/safety.py applies on the arm)."""
    return ref + (target - ref).clamp(-bound, bound)


def _record_end(
    end_info: dict[str, Tensor], info: Any, rows: Tensor, n: int, term: Any, trunc: Any
) -> None:
    """Copy the per-env bool flags of `info` (+ terminated/truncated) for the rows that just ended."""
    flags = {"terminated": _flag(term, n), "truncated": _flag(trunc, n)}
    if isinstance(info, Mapping):
        for k, v in info.items():
            if isinstance(v, Tensor) and v.dtype == torch.bool and v.reshape(-1).shape[0] == n:
                flags[str(k)] = v.reshape(-1).to("cpu")
    for k, v in flags.items():
        end_info.setdefault(k, torch.zeros(n, dtype=torch.bool))[rows] = v[rows]


@torch.no_grad()
def run_batch(
    act: ActFn,
    env: Any,
    obs_fn: ObsFn,
    seeds: Sequence[int],
    max_steps: int,
    exec_steps: int | None = None,
    frames: list[Tensor] | None = None,
    *,
    action_clip: float | None = None,
    end_info: dict[str, Tensor] | None = None,
    instructions_out: list[str] | None = None,
) -> tuple[Tensor, Tensor]:
    """One batch of len(seeds) parallel episodes -> (success (N,) bool, length (N,) int).

    `frames`, when given, receives the policy camera image (N, H, W, 3) at reset and after every
    env step (for videos; scripts/record_videos.py).

    `exec_steps` executes only the first N actions of each predicted chunk before re-querying the
    policy (receding horizon); None executes the whole chunk. A policy setting, not protocol.

    `action_clip` (rad, protocol 0.2) clips every executed action per joint to within that bound
    of the previous executed action (of the joint reading obs["state"] at reset); None = no clip.
    `end_info`, when given, receives per-episode (N,) tensors: every bool flag of the env `info`
    at the step the episode ended (success, is_grasped, obj_in_bowl, ...), "terminated",
    "truncated", and "clipped_steps" (steps on which the clip changed the action).
    `instructions_out`, when given, receives the reset's per-episode instructions (they name the
    episode's object, e.g. which unseen object an unseen_objects episode drew).
    """
    n = len(seeds)
    raw, _ = env.reset(seed=list(seeds))
    obs = obs_fn(raw)
    if frames is not None:
        frames.append(obs["image"].detach().cpu())
    instructions = list(env.unwrapped.get_instructions())
    if len(instructions) != n:
        raise ValueError(f"env gave {len(instructions)} instructions for {n} episodes")
    if instructions_out is not None:
        instructions_out.extend(instructions)
    active = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    length = torch.zeros(n, dtype=torch.long)
    clipped = torch.zeros(n, dtype=torch.long)
    ref = obs["state"].detach().float().clone() if action_clip is not None else None
    while bool(active.any()):
        chunk = act(obs, instructions)
        horizon = chunk.shape[1] if exec_steps is None else min(exec_steps, chunk.shape[1])
        for c in range(horizon):
            if not bool(active.any()):
                break
            action = chunk[:, c]
            if ref is not None and action_clip is not None:
                ref = ref.to(action.device, action.dtype)
                safe = clip_joint_step(action, ref, action_clip)
                clipped += (safe != action).any(-1).to("cpu") & active
                action = ref = safe
            raw, _, terminated, truncated, info = env.step(action)
            if frames is not None:
                frames.append(obs_fn(raw)["image"].detach().cpu())
            length += active.long()
            hit = _flag(info["success"], n) & active
            success |= hit
            ended = hit | _flag(terminated, n) | _flag(truncated, n) | (length >= max_steps)
            if end_info is not None and bool((ended & active).any()):
                _record_end(end_info, info, ended & active, n, terminated, truncated)
            active &= ~ended
        obs = obs_fn(raw)
    if end_info is not None:
        end_info["clipped_steps"] = clipped
    return success, length


@dataclass(frozen=True)
class CellSeedResult:
    task: str
    cell: str
    seed: int
    episodes: int
    successes: int
    mean_length: float


@dataclass(frozen=True)
class EpisodeRecord:
    """One evaluated episode (runs/<id>/episodes.jsonl). `seed` is the protocol seed;
    `instruction` the reset's language instruction (it names the manipulated object, so the
    unseen_objects cell splits by object). `end_reason`: success | terminated | horizon, where
    horizon is the time limit, whether the env's truncation or the eval's `max_steps` cap ended
    it (scripts/wo02_eval.py sets both to env.max_episode_steps, so ManiSkill truncates on that
    same step; the raw flag stays in `flags["truncated"]`). With the current tasks "terminated"
    does not occur: none defines a fail state, so ManiSkill's terminated equals success. `flags`:
    the env's bool info flags at the last step (see `run_batch` end_info) + truncated;
    `clipped_steps` is 0 without an action clip."""

    task: str
    cell: str
    seed: int
    episode_seed: int
    instruction: str
    success: bool
    length: int
    end_reason: str
    clipped_steps: int
    flags: dict[str, bool]


def _episode_records(
    task: str,
    cell: str,
    seed: int,
    seeds: Sequence[int],
    s: Tensor,
    ln: Tensor,
    end: dict[str, Any],
    instructions: Sequence[str],
) -> list[EpisodeRecord]:
    out = []
    skip = {"terminated", "clipped_steps"}
    for i, ep in enumerate(seeds):
        term = bool(end["terminated"][i])
        reason = "success" if bool(s[i]) else "terminated" if term else "horizon"
        flags = {k: bool(v[i]) for k, v in sorted(end.items()) if k not in skip}
        out.append(
            EpisodeRecord(
                task=task,
                cell=cell,
                seed=int(seed),
                episode_seed=int(ep),
                instruction=str(instructions[i]),
                success=bool(s[i]),
                length=int(ln[i]),
                end_reason=reason,
                clipped_steps=int(end["clipped_steps"][i]),
                flags=flags,
            )
        )
    return out


def evaluate(
    act: ActFn,
    make_env: MakeEnv,
    obs_fn: ObsFn,
    *,
    tasks: Sequence[str],
    cells: Sequence[str],
    seeds: Sequence[int],
    episodes: int,
    num_envs: int,
    max_steps: int,
    all_tasks: Sequence[str] | None = None,
    all_cells: Sequence[str] | None = None,
    log: Callable[[CellSeedResult], None] | None = None,
    exec_steps: int | None = None,
    action_clip: float | None = None,
    log_episode: Callable[[EpisodeRecord], None] | None = None,
) -> list[CellSeedResult]:
    """`episodes` per (task, cell, seed); one env per (task, cell), reused across seeds/batches.

    `action_clip`: see `run_batch` (protocol 0.2). `log_episode` receives one EpisodeRecord per
    evaluated episode (padding episodes excluded)."""
    task_ids, cell_ids = list(all_tasks or tasks), list(all_cells or cells)
    batch = max(1, min(num_envs, episodes))
    out = []
    for task in tasks:
        for cell in cells:
            env = make_env(task, cell, batch)
            try:
                for seed in seeds:
                    want = episode_seeds(seed, task_ids.index(task), cell_ids.index(cell), episodes)
                    succ: list[Tensor] = []
                    lens: list[Tensor] = []
                    for start in range(0, episodes, batch):
                        chunk = want[start : start + batch]
                        pad = chunk + [chunk[-1] + episodes + i for i in range(batch - len(chunk))]
                        end: dict[str, Tensor] = {}
                        instr: list[str] = []
                        s, ln = run_batch(
                            act,
                            env,
                            obs_fn,
                            pad,
                            max_steps,
                            exec_steps,
                            action_clip=action_clip,
                            end_info=end if log_episode is not None else None,
                            instructions_out=instr,
                        )
                        if log_episode is not None:
                            recs = _episode_records(task, cell, seed, chunk, s, ln, end, instr)
                            for rec in recs:
                                log_episode(rec)
                        succ.append(s[: len(chunk)])
                        lens.append(ln[: len(chunk)])
                    s_all, l_all = torch.cat(succ), torch.cat(lens)
                    res = CellSeedResult(
                        task,
                        cell,
                        int(seed),
                        episodes,
                        int(s_all.sum()),
                        float(l_all.float().mean()),
                    )
                    out.append(res)
                    if log is not None:
                        log(res)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()
    return out


def summarize(
    results: Sequence[CellSeedResult], interval: Interval, meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Per (task, cell): per-seed rates, mean ± std over seeds, pooled rate with Wilson CI."""
    cells: dict[tuple[str, str], list[CellSeedResult]] = {}
    for r in results:
        cells.setdefault((r.task, r.cell), []).append(r)
    rows = []
    for (task, cell), rs in cells.items():
        rates = [r.successes / r.episodes for r in rs]
        k, n = sum(r.successes for r in rs), sum(r.episodes for r in rs)
        lo, hi = interval(k, n)
        rows.append(
            {
                "task": task,
                "cell": cell,
                "seeds": [r.seed for r in rs],
                "per_seed": rates,
                "mean": statistics.fmean(rates),
                "std": statistics.pstdev(rates) if len(rates) > 1 else 0.0,
                "successes": k,
                "episodes": n,
                "pooled": k / n,
                "wilson95": [lo, hi],
                "mean_length": statistics.fmean(r.mean_length for r in rs),
            }
        )
    return {**meta, "rows": rows, "raw": [asdict(r) for r in results]}


def to_markdown(summary: Mapping[str, Any]) -> str:
    title = (
        f"Policy `{summary.get('policy', '?')}` | protocol `{summary.get('protocol_version', '?')}`"
        f" hash `{str(summary.get('protocol_hash', '?'))[:12]}` | complete: "
        f"{summary.get('protocol_complete', False)}"
    )
    head = [
        title,
        "",
        "| task | cell | success (pooled) | 95% Wilson | per seed | episodes |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        per = ", ".join(f"{100 * v:.0f}" for v in r["per_seed"])
        lo, hi = r["wilson95"]
        head.append(
            f"| {r['task']} | {r['cell']} | {100 * r['pooled']:.1f}% | "
            f"[{100 * lo:.1f}, {100 * hi:.1f}] | {per} | {r['episodes']} |"
        )
    return "\n".join(head) + "\n"
