"""src/eval/paired.py: paired readout of two eval runs from their per-episode records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.paired import main, mcnemar_one_sided


def _run(root: Path, name: str, successes: list[bool], hash_: str = "h") -> Path:
    d = root / name
    d.mkdir()
    meta = {"protocol_version": "0.2", "protocol_hash": hash_, "protocol_complete": True}
    (d / "eval.json").write_text(json.dumps({**meta, "policy": name}))
    rows = [
        {"task": "PickPlace" if i % 2 else "Push", "cell": "seen", "seed": i % 3,
         "episode_seed": 100 + i, "success": ok}
        for i, ok in enumerate(successes)
    ]  # fmt: skip
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_paired_compare_reads_discordant_episodes(tmp_path: Path) -> None:
    assert mcnemar_one_sided(0, 0) == 1.0 and mcnemar_one_sided(5, 0) == 1 / 32
    base = str(_run(tmp_path, "base", [False] * 6 + [True] * 2))
    arm = str(_run(tmp_path, "arm", [True] * 5 + [False] + [True, False]))
    out = main([base, arm])
    pooled = out["pooled"]
    assert (pooled["episodes"], pooled["arm_only"], pooled["base_only"]) == (8, 5, 1)
    assert pooled["diff_points"] == pytest.approx(50.0)
    assert pooled["p_one_sided"] == pytest.approx(7 / 64)  # P(X >= 5), X ~ Bin(6, 1/2)
    assert set(out["per_task"]) == {"PickPlace", "Push"} and out["protocol_complete"]
    assert main([base, arm, "--seeds", "1,2"])["pooled"]["episodes"] == 5
    with pytest.raises(ValueError, match="protocol_hash"):
        main([base, str(_run(tmp_path, "other", [True] * 8, hash_="x"))])
    with pytest.raises(ValueError, match="same episodes"):
        main([base, str(_run(tmp_path, "short", [True] * 7))])
