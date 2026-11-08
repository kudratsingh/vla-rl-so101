"""Batched greedy sim evaluation: tasks x cells x seeds, success with 95% Wilson intervals.

Env contract (src/env): `make_env(cfg, task, num_envs=, cell=)` -> vector env with
`reset(seed=list)`, `step(actions (N, 6))` -> (obs, reward, terminated, truncated, info) where
`info["success"]` is (N,) bool, and `env.unwrapped.get_instructions()`; `policy_obs(obs)` ->
{"image", "state"}. An episode succeeds if `success` fires before it ends (same rule as the RL
rollout in src/rl/rollout.py). A policy decision executes its whole action chunk open loop.
`run_batch_traced` is the same loop plus per-step privileged data and chunk-boundary snapshots
(recovery SFT); it is never used for reported numbers.

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


@torch.no_grad()
def run_batch(
    act: ActFn,
    env: Any,
    obs_fn: ObsFn,
    seeds: Sequence[int],
    max_steps: int,
    exec_steps: int | None = None,
    frames: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """One batch of len(seeds) parallel episodes -> (success (N,) bool, length (N,) int).

    `frames`, when given, receives the policy camera image (N, H, W, 3) at reset and after every
    env step (for videos; scripts/record_videos.py).

    `exec_steps` executes only the first N actions of each predicted chunk before re-querying the
    policy (receding horizon); None executes the whole chunk. A policy setting, not protocol.
    """
    n = len(seeds)
    raw, _ = env.reset(seed=list(seeds))
    obs = obs_fn(raw)
    if frames is not None:
        frames.append(obs["image"].detach().cpu())
    instructions = list(env.unwrapped.get_instructions())
    if len(instructions) != n:
        raise ValueError(f"env gave {len(instructions)} instructions for {n} episodes")
    active = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    length = torch.zeros(n, dtype=torch.long)
    while bool(active.any()):
        chunk = act(obs, instructions)
        horizon = chunk.shape[1] if exec_steps is None else min(exec_steps, chunk.shape[1])
        for c in range(horizon):
            if not bool(active.any()):
                break
            raw, _, terminated, truncated, info = env.step(chunk[:, c])
            if frames is not None:
                frames.append(obs_fn(raw)["image"].detach().cpu())
            length += active.long()
            hit = _flag(info["success"], n) & active
            success |= hit
            ended = hit | _flag(terminated, n) | _flag(truncated, n) | (length >= max_steps)
            active &= ~ended
        obs = obs_fn(raw)
    return success, length


@dataclass(frozen=True)
class ChunkBoundary:
    """State just before a policy decision. `step` = env steps executed so far (0 = reset);
    `last_action` (N, 6) = the command last sent (reset qpos at step 0); `state` = the env
    snapshot from `snapshot_fn` (None without one)."""

    step: int
    active: Tensor
    last_action: Tensor
    state: Any = None


@dataclass
class RolloutTrace:
    """Per-step privileged data and chunk-boundary snapshots of one `run_batch_traced` batch.

    `steps[t]` holds batched CPU tensors after t env steps (t = 0: reset): "qpos", the env's
    privileged `obs["extra"]` entries, every per-env `info` flag (success, is_grasped, ...), and
    "action" (N, 6) = the command that produced step t (absent at t = 0)."""

    seeds: list[int]
    steps: list[dict[str, Tensor]]
    boundaries: list[ChunkBoundary]

    def series(self, key: str) -> Tensor:
        """(T + 1, N, ...) stack of `key` over steps 0..T (missing at a step -> error)."""
        return torch.stack([s[key] for s in self.steps])


def _privileged(raw: Any, info: Any, n: int) -> dict[str, Tensor]:
    """Batched CPU copies of qpos, obs["extra"] and the per-env entries of `info`."""
    out: dict[str, Tensor] = {}
    if isinstance(raw, Mapping):
        agent = raw.get("agent")
        if isinstance(agent, Mapping) and "qpos" in agent:
            out["qpos"] = torch.as_tensor(agent["qpos"]).detach().to("cpu").float()
        extra = raw.get("extra")
        if isinstance(extra, Mapping):
            for k, v in extra.items():
                t = torch.as_tensor(v).detach().to("cpu")
                if t.ndim > 0 and t.shape[0] == n:
                    out[k] = t
    if isinstance(info, Mapping):
        for k, v in info.items():
            if isinstance(v, Tensor) and v.ndim > 0 and v.shape[0] == n:
                out[k] = v.detach().to("cpu")
    return out


@torch.no_grad()
def run_batch_traced(
    act: ActFn,
    env: Any,
    obs_fn: ObsFn,
    seeds: Sequence[int],
    max_steps: int,
    exec_steps: int | None = None,
    snapshot_fn: Callable[[Any], Any] | None = None,
) -> tuple[Tensor, Tensor, RolloutTrace]:
    """`run_batch` plus a `RolloutTrace` (recovery-SFT failure mining, docs/plans/recovery-sft.md).

    Same episode loop and the same (success, length) outcome rule as `run_batch`; the trace only
    reads the env. `snapshot_fn(env)` (e.g. `src.env.state.snapshot`) runs at every chunk
    boundary. Tests check the outcomes match `run_batch` exactly.
    """
    n = len(seeds)
    raw, info = env.reset(seed=list(seeds))
    obs = obs_fn(raw)
    instructions = list(env.unwrapped.get_instructions())
    if len(instructions) != n:
        raise ValueError(f"env gave {len(instructions)} instructions for {n} episodes")
    steps = [_privileged(raw, info, n)]
    last = steps[0].get("qpos", torch.zeros(n, 6))
    boundaries: list[ChunkBoundary] = []
    active = torch.ones(n, dtype=torch.bool)
    success = torch.zeros(n, dtype=torch.bool)
    length = torch.zeros(n, dtype=torch.long)
    while bool(active.any()):
        state = snapshot_fn(env) if snapshot_fn is not None else None
        boundaries.append(ChunkBoundary(len(steps) - 1, active.clone(), last.clone(), state))
        chunk = act(obs, instructions)
        horizon = chunk.shape[1] if exec_steps is None else min(exec_steps, chunk.shape[1])
        for c in range(horizon):
            if not bool(active.any()):
                break
            action = chunk[:, c]
            raw, _, terminated, truncated, info = env.step(action)
            last = torch.as_tensor(action).detach().to("cpu").float()
            steps.append({**_privileged(raw, info, n), "action": last})
            length += active.long()
            hit = _flag(info["success"], n) & active
            success |= hit
            ended = hit | _flag(terminated, n) | _flag(truncated, n) | (length >= max_steps)
            active &= ~ended
        obs = obs_fn(raw)
    return success, length, RolloutTrace(list(seeds), steps, boundaries)


@dataclass(frozen=True)
class CellSeedResult:
    task: str
    cell: str
    seed: int
    episodes: int
    successes: int
    mean_length: float


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
) -> list[CellSeedResult]:
    """`episodes` per (task, cell, seed); one env per (task, cell), reused across seeds/batches."""
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
                        s, ln = run_batch(act, env, obs_fn, pad, max_steps, exec_steps)
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
