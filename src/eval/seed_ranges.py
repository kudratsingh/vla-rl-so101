"""Every episode-seed range the project uses, and a guard that they never overlap.

A seed is a ManiSkill reset seed for one sub-env (it fixes object, goal and arm noise). Held-out
evaluation is only held out if no training, collection or checkpoint-selection run ever touches
its seeds. Ranges are half-open [start, stop):

| name        | seeds                                                  | source |
|-------------|--------------------------------------------------------|--------|
| demo        | 1_000_000 + 100_000 * cfg.seed + k (cfg.seed 0 used)   | scripts/wo01_build_env.py |
| rl_train    | rl_seed * 10_000_000 + prompt counter (< 1_000_000)    | src/rl/trainer.py |
| protocol    | 100_000_000 + 1_000_000 * s + ..., s in SIM_SEEDS      | src/eval/sim_eval.py, protocol.py |
| selection   | the same formula with protocol seeds 7, 8 (not SIM_SEEDS) | docs/plans/recovery-sft.md |
| recovery    | 3_000_000_000 + 1_000_000 * task_index + k             | this file |
| control     | 3_100_000_000 + 1_000_000 * task_index + k             | this file |
| rl_eval     | 4_000_000_000 + i                                      | src/rl/trainer.py |

The rl_train span bound: 150 steps x 8 prompts x <= 3 refill rounds is ~3,600 prompts per seed.
RL seeds >= 10 would land on the protocol range (100M+); `check_rl_seed` rejects them.
All seeds stay below 2**32 (numpy RandomState).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import SupportsInt

from src.eval.protocol import SIM_SEEDS, TASKS
from src.eval.sim_eval import EVAL_SEED_BASE
from src.rl.trainer import EVAL_SEED_BASE as RL_EVAL_SEED_BASE
from src.rl.trainer import TRAIN_SEED_STRIDE

DEMO_SEED_BASE = 1_000_000  # = scripts/wo01_build_env.py DEMO_SEED_BASE (a test checks)
DEMO_SEED_STRIDE = 100_000  # per cfg.seed
DEMO_CFG_SEEDS: tuple[int, ...] = (0,)  # both demo sets (10 and 25 per task) used cfg.seed 0
RL_TRAIN_SPAN = 1_000_000
RL_EVAL_SPAN = 1_000_000
PROTOCOL_SEED_SPAN = 1_000_000  # one protocol seed's 27 (task, cell) blocks fit in 180_100
SELECTION_PROTOCOL_SEEDS: tuple[int, ...] = (7, 8)  # checkpoint selection, never SIM_SEEDS
RECOVERY_SEED_BASE = 3_000_000_000
CONTROL_SEED_BASE = 3_100_000_000
TASK_STRIDE = 1_000_000  # recovery / control seeds per task
MAX_SEED = 2**32


@dataclass(frozen=True)
class SeedRange:
    name: str
    start: int
    stop: int

    def __contains__(self, seed: object) -> bool:
        return isinstance(seed, SupportsInt) and self.start <= int(seed) < self.stop

    def overlaps(self, other: SeedRange) -> bool:
        return self.start < other.stop and other.start < self.stop


def _protocol_block(protocol_seed: int, name: str) -> SeedRange:
    start = EVAL_SEED_BASE + PROTOCOL_SEED_SPAN * protocol_seed
    return SeedRange(name, start, start + PROTOCOL_SEED_SPAN)


def rl_train_range(rl_seed: int) -> SeedRange:
    start = rl_seed * TRAIN_SEED_STRIDE
    return SeedRange(f"rl_train[{rl_seed}]", start, start + RL_TRAIN_SPAN)


def standard_ranges(
    *,
    rl_seeds: Iterable[int] = (0, 1, 2),
    selection_seeds: Iterable[int] = SELECTION_PROTOCOL_SEEDS,
    demo_cfg_seeds: Iterable[int] = DEMO_CFG_SEEDS,
) -> list[SeedRange]:
    """All ranges in use: protocol eval, checkpoint selection, demos, recovery, control, RL."""
    sel = list(selection_seeds)
    if set(sel) & set(SIM_SEEDS):
        raise ValueError(f"selection seeds {sel} include protocol SIM_SEEDS {SIM_SEEDS}")
    out = [_protocol_block(s, f"protocol[{s}]") for s in SIM_SEEDS]
    out += [_protocol_block(s, f"selection[{s}]") for s in sel]
    for s in demo_cfg_seeds:
        start = DEMO_SEED_BASE + DEMO_SEED_STRIDE * s
        out.append(SeedRange(f"demo[{s}]", start, start + DEMO_SEED_STRIDE))
    out.append(SeedRange("recovery", RECOVERY_SEED_BASE, RECOVERY_SEED_BASE + TASK_STRIDE * 3))
    out.append(SeedRange("control", CONTROL_SEED_BASE, CONTROL_SEED_BASE + TASK_STRIDE * 3))
    out += [rl_train_range(s) for s in rl_seeds]
    out.append(SeedRange("rl_eval", RL_EVAL_SEED_BASE, RL_EVAL_SEED_BASE + RL_EVAL_SPAN))
    return out


def check_disjoint(ranges: Sequence[SeedRange]) -> None:
    """Raise if any two ranges overlap or a range leaves [0, 2**32)."""
    bad = [r.name for r in ranges if r.start < 0 or r.stop > MAX_SEED or r.stop <= r.start]
    if bad:
        raise ValueError(f"seed ranges out of [0, 2**32): {bad}")
    clash = [
        f"{a.name} [{a.start}, {a.stop}) x {b.name} [{b.start}, {b.stop})"
        for i, a in enumerate(ranges)
        for b in ranges[i + 1 :]
        if a.overlaps(b)
    ]
    if clash:
        raise ValueError("overlapping seed ranges: " + "; ".join(clash))


def assert_seeds_in(seeds: Iterable[int], name: str, ranges: Sequence[SeedRange]) -> None:
    """The ranges are pairwise disjoint and every seed lies in the range(s) called `name` (prefix
    match: "protocol" covers protocol[0..2]), hence in no other range."""
    check_disjoint(ranges)
    own = [r for r in ranges if r.name == name or r.name.startswith(name + "[")]
    if not own:
        raise KeyError(f"no seed range named {name!r}")
    for s in seeds:
        if not any(s in r for r in own):
            raise ValueError(f"seed {s} is outside the {name!r} range(s)")


def _task_seeds(base: int, task: str, n: int, start: int) -> list[int]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if start < 0 or start + n > TASK_STRIDE:
        raise ValueError(f"seed offsets [{start}, {start + n}) exceed the per-task stride")
    first = base + TASK_STRIDE * TASKS.index(task) + start
    return list(range(first, first + n))


def recovery_seeds(task: str, n: int, start: int = 0) -> list[int]:
    """Episode seeds for SFT failure mining (seen cell only)."""
    return _task_seeds(RECOVERY_SEED_BASE, task, n, start)


def control_seeds(task: str, n: int, start: int = 0) -> list[int]:
    """Episode seeds for the matched control's fresh oracle demos (seen cell only)."""
    return _task_seeds(CONTROL_SEED_BASE, task, n, start)


def check_rl_seed(rl_seed: int) -> None:
    """Guard for a ManiSkill GRPO run: its training-prompt seeds must not touch any other range."""
    others = [r for r in standard_ranges(rl_seeds=()) if not r.name.startswith("rl_train")]
    check_disjoint([rl_train_range(rl_seed), *others])
