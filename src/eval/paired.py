"""Paired comparison of two eval runs from their per-episode records (docs/plans/recovery-sft.md
section 10): both runs play the same episode seeds, so arm-vs-base differences are read episode
by episode with a one-sided exact McNemar test on the discordant episodes.

Input: two `scripts/wo02_eval.py` run dirs (`eval.json` + `episodes.jsonl`, sim_eval.EpisodeRecord
per line). The runs must cite the same protocol version and hash and cover the same episodes of
the selected (cells, protocol seeds); anything else is refused, not silently intersected.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Key = tuple[str, str, int, int]  # task, cell, protocol seed, episode seed


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[Key, bool]]:
    """(eval.json, {episode key: success}) of one eval run dir."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "eval.json").read_text())
    out: dict[Key, bool] = {}
    for line in (run_dir / "episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        key = (str(r["task"]), str(r["cell"]), int(r["seed"]), int(r["episode_seed"]))
        if key in out:
            raise ValueError(f"{run_dir}: duplicate episode {key}")
        out[key] = bool(r["success"])
    return meta, out


def mcnemar_one_sided(arm_only: int, base_only: int) -> float:
    """P(X >= arm_only), X ~ Binomial(arm_only + base_only, 1/2): exact one-sided McNemar p for
    "the arm succeeds more often than the base" (1.0 with no discordant episodes)."""
    n = arm_only + base_only
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(arm_only, n + 1)) / 2**n)


def _select(eps: dict[Key, bool], cells: Sequence[str], seeds: Sequence[int] | None) -> set[Key]:
    return {k for k in eps if k[1] in cells and (seeds is None or k[2] in seeds)}


def compare(
    base_dir: Path,
    arm_dir: Path,
    *,
    cells: Sequence[str] = ("seen",),
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Pooled and per-task paired readout of `arm` minus `base` over `cells` x `seeds` (None: every
    protocol seed in the records)."""
    base_meta, base = load_run(base_dir)
    arm_meta, arm = load_run(arm_dir)
    for field in ("protocol_version", "protocol_hash"):
        if base_meta.get(field) != arm_meta.get(field):
            raise ValueError(f"{field} differs: {base_meta.get(field)} vs {arm_meta.get(field)}")
    keys = _select(base, cells, seeds)
    if keys != _select(arm, cells, seeds):
        raise ValueError("the runs did not play the same episodes (task, cell, seed, episode seed)")
    if not keys:
        raise ValueError(f"no episodes for cells {list(cells)} seeds {seeds}")

    def readout(ks: list[Key]) -> dict[str, Any]:
        arm_only = sum(arm[k] and not base[k] for k in ks)
        base_only = sum(base[k] and not arm[k] for k in ks)
        n = len(ks)
        return {
            "episodes": n,
            "base_rate": sum(base[k] for k in ks) / n,
            "arm_rate": sum(arm[k] for k in ks) / n,
            "diff_points": 100.0 * (arm_only - base_only) / n,
            "arm_only": arm_only,
            "base_only": base_only,
            "p_one_sided": mcnemar_one_sided(arm_only, base_only),
        }

    tasks = sorted({k[0] for k in keys})
    return {
        "protocol_version": base_meta.get("protocol_version"),
        "protocol_hash": base_meta.get("protocol_hash"),
        "protocol_complete": bool(base_meta.get("protocol_complete"))
        and bool(arm_meta.get("protocol_complete")),
        "base": str(base_meta.get("policy")),
        "arm": str(arm_meta.get("policy")),
        "cells": list(cells),
        "seeds": sorted({k[2] for k in keys}),
        "pooled": readout(sorted(keys)),
        "per_task": {t: readout(sorted(k for k in keys if k[0] == t)) for t in tasks},
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """uv run python -m src.eval.paired runs/<base> runs/<arm> [--seeds 1,2] [--cells seen]"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="eval run dir of the base checkpoint")
    parser.add_argument("arm", type=Path, help="eval run dir of the arm checkpoint")
    parser.add_argument("--cells", default="seen", help="comma-separated (default seen)")
    parser.add_argument("--seeds", default=None, help="comma-separated protocol seeds (all)")
    args = parser.parse_args(argv)
    seeds = None if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    out = compare(args.base, args.arm, cells=args.cells.split(","), seeds=seeds)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
