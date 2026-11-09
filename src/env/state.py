"""Episode snapshots and restores for recovery SFT (docs/plans/recovery-sft.md).

`snapshot(env)` captures every sub-env's sim state; `restore(env, seeds, snap)` puts a batch of
saved rows back so the scripted oracle can continue from a policy's failure state. Works on the
GPU env (`make_env` on cuda) and on the CPU `SerialVecEnv`.

ManiSkill 3.0.1 caveats, and how this module handles them:

- `get_state_dict` covers dynamic/kinematic actors (pose + velocities) and articulations (root,
  qpos, qvel). Task bookkeeping is outside it (active candidate, object half size, instructions,
  goal pose of the static parts): `restore` first resets with the SAME episode seeds, which
  rebuilds all of it, then overwrites the sim state.
- Push keeps an episode latch `_ever_lifted` outside the sim state; reset clears it, so it is
  saved in `EnvSnapshot.task` and written back.
- Controller state is saved by `get_state_dict` but `set_state_dict` never restores it. Harmless
  for the absolute `pd_joint_pos` controller (every action sets the drive targets); any other
  control mode is refused.
- With `env_idx` on GPU, `set_state_dict` writes only the masked rows and expects exactly those
  rows. `restore` always resets the full batch and writes every row, without `env_idx`.
- The action-latency randomization history (`_act_hist`) is not restored: refused when it is on.
- Contact-derived flags (`is_grasped`, touching) in the first observation after a restore are
  stale until one physics step has run. Callers use the flag recorded with the snapshot.
- `elapsed_steps` restarts at 0, so a restored branch gets a fresh `max_episode_steps` budget.
- CPU and GPU PhysX differ: a snapshot only restores into the backend it came from. Replaying the
  same actions after a CPU restore drifts by ~1e-5 rad (solver warm start is not part of the state).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

TASK_STATE_ATTRS: tuple[str, ...] = ("_ever_lifted",)  # Push latch (src/env/tasks.py)


def _tree(tree: Any, fn: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _tree(v, fn) for k, v in tree.items()}
    return fn(tree)


def _cpu(t: Any) -> Any:
    return t.detach().to("cpu").clone() if isinstance(t, torch.Tensor) else t


def _leaves(tree: Any) -> list[torch.Tensor]:
    if isinstance(tree, dict):
        return [x for v in tree.values() for x in _leaves(v)]
    return [tree] if isinstance(tree, torch.Tensor) else []


@dataclass(frozen=True)
class EnvSnapshot:
    """Batched sim state (CPU tensors, one row per sub-env) + task bookkeeping + backend."""

    sim: dict[str, Any]
    backend: str
    task: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        leaves = _leaves(self.sim)
        return int(leaves[0].shape[0]) if leaves else 0

    def row(self, i: int) -> EnvSnapshot:
        """Sub-env i alone (batch dim 1)."""

        def pick(t: Any) -> Any:
            return t[i : i + 1].clone() if isinstance(t, torch.Tensor) and t.ndim > 0 else t

        return EnvSnapshot(
            _tree(self.sim, pick), self.backend, {k: pick(v) for k, v in self.task.items()}
        )

    @staticmethod
    def stack(rows: Sequence[EnvSnapshot]) -> EnvSnapshot:
        """Rows (e.g. from different episodes) -> one batched snapshot, in order."""
        if not rows:
            raise ValueError("need at least one snapshot row")
        backends = {r.backend for r in rows}
        if len(backends) != 1:
            raise ValueError(f"snapshots from different backends: {sorted(backends)}")

        def cat(items: list[Any]) -> Any:
            first = items[0]
            if isinstance(first, dict):
                return {k: cat([it[k] for it in items]) for k in first}
            if isinstance(first, torch.Tensor):
                return torch.cat(items, dim=0)
            return first

        task = {k: torch.cat([r.task[k] for r in rows]) for k in rows[0].task}
        return EnvSnapshot(cat([r.sim for r in rows]), rows[0].backend, task)


def bases(env: Any) -> list[Any]:
    """The ManiSkill BaseEnv(s) behind `env`: one per sub-env on CPU, one batched env on GPU."""
    envs = getattr(env, "envs", None)
    if envs is not None:
        return [e.unwrapped for e in envs]
    return [env.unwrapped]


def backend(env: Any) -> str:
    return "physx_cuda" if bool(getattr(bases(env)[0], "gpu_sim_enabled", False)) else "physx_cpu"


def _check_restorable(env: Any) -> None:
    for b in bases(env):
        if getattr(b, "control_mode", "pd_joint_pos") != "pd_joint_pos":
            raise NotImplementedError(
                f"restore needs control_mode pd_joint_pos (controller state is not restored), "
                f"got {b.control_mode!r}"
            )
        dr = getattr(b, "dr", None)
        if dr is not None and dr.enabled and dr.action_latency_max_steps > 0:
            raise NotImplementedError("restore with action-latency randomization (history lost)")


def snapshot(env: Any) -> EnvSnapshot:
    """Current state of every sub-env (copied to CPU)."""
    sim: dict[str, Any] = _tree(env.unwrapped.get_state_dict(), _cpu)
    sim.pop("controller", None)  # never restored by ManiSkill 3.0.1 (see module docstring)
    task: dict[str, torch.Tensor] = {}
    for attr in TASK_STATE_ATTRS:
        vals = [getattr(b, attr, None) for b in bases(env)]
        if all(isinstance(v, torch.Tensor) for v in vals):
            task[attr] = torch.cat([_cpu(v).reshape(-1) for v in vals])
    return EnvSnapshot(sim, backend(env), task)


def restore(env: Any, seeds: Sequence[int], snap: EnvSnapshot) -> Any:
    """Reset `env` with the snapshot rows' own episode `seeds`, write the rows back, and return a
    freshly rendered observation of the restored state. `len(seeds)` = sub-envs = snapshot rows.
    """
    _check_restorable(env)
    n = int(getattr(env.unwrapped, "num_envs", 1))
    if len(seeds) != n or snap.num_rows != n:
        raise ValueError(f"restore needs {n} seeds and rows, got {len(seeds)} / {snap.num_rows}")
    if snap.backend != backend(env):
        raise ValueError(f"snapshot from {snap.backend} cannot restore into {backend(env)}")
    env.reset(seed=list(seeds))
    env.unwrapped.set_state_dict(snap.sim)
    bs = bases(env)
    for attr, values in snap.task.items():
        if len(bs) == 1:
            cur = getattr(bs[0], attr)
            cur[:] = values.to(cur.device, cur.dtype)
        else:
            for i, b in enumerate(bs):
                cur = getattr(b, attr)
                cur[:] = values[i : i + 1].to(cur.device, cur.dtype)
    return env.unwrapped.get_obs()


def state_error(a: EnvSnapshot, b: EnvSnapshot) -> float:
    """Max abs difference over every sim-state entry (restore round-trip check)."""
    la, lb = _leaves(a.sim), _leaves(b.sim)
    if len(la) != len(lb):
        raise ValueError("snapshots have different layouts")
    return max((float((x - y).abs().max()) for x, y in zip(la, lb, strict=True)), default=0.0)
