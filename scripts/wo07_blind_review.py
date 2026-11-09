#!/usr/bin/env python
"""WO-07 (optional): blinded clip review of recorded evaluation trials (final2 label provenance).

Pools the captured episodes of one or more real-0.1 evaluation sessions (e.g. the SmolVLA and the
ACT session), shuffles them in a seeded random order and shows the reviewer one packet at a time,
WITHOUT the policy identity: a packet is a folder named by a random blind id holding a contact
sheet of the trial (front and wrist frames decoded with torchcodec from the session's
LeRobotDataset) and the end-state stills (both cameras, read after the settle,
``src/robot/recording.py``). The prompt shows only the item number, the blind id, the instruction
and the rubric's success condition (``src/eval/protocol_real.py``, R1); the answer is y or n
(doubt = n). ``--partial-credit`` also asks whether the object was lifted (a SECONDARY column:
1 success, 0.5 lifted, 0 otherwise; never the headline).

The output ``review.json`` records the reviewer id, the seed and order, every verdict with its
(session, episode) key, and ``method: blinded_review``. Only such a file lets
``scripts/wo07_export_bundle.py eval --review`` declare ``labels.method = blinded_review``. Live
operator scores and automatic fault outcomes keep their own provenance in the trial records and
are never relabelled. Rubric R2 (a fault after the start is a failure) is applied by the export
from the recorded fault, not by the reviewer.

    uv run scripts/wo07_blind_review.py --preset real --session <smolvla run> --session <act run> \\
        --reviewer <name>

Packets open in Finder (``open``) unless ``--no-open``. Ctrl-C or a closed stdin ends the review;
the verdicts so far are written (``complete: false``) and the export refuses an incomplete review.
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args
from src.eval import protocol_real as PR
from src.robot import convention as conv
from src.robot.recording import PROVENANCE
from src.runs import RUNS_DIR

SHEET_FRAMES = 8
_DATASETS: dict[str, Any] = {}  # capture dir -> its LeRobotDataset (decoded with torchcodec)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def session_items(capture: Path) -> list[dict[str, Any]]:
    """Every captured episode of a session's capture dir, with its trial id and stills."""
    prov = json.loads((capture / "dataset" / PROVENANCE).read_text())
    if (prov.get("protocol") or {}).get("version") != PR.PROTOCOL_VERSION:
        sys.exit(f"{capture}: not a {PR.PROTOCOL_VERSION} capture")
    sid = str(prov["session_id"])
    return [
        {
            "session_id": sid,
            "episode_index": int(i),
            "trial_id": ep.get("trial_id"),
            "end_state": ep.get("end_state"),
            "capture": str(capture),
        }
        for i, ep in sorted(prov.get("episodes", {}).items(), key=lambda kv: int(kv[0]))
    ]


def contact_sheet(capture: Path, episode_index: int, out: Path, n: int = SHEET_FRAMES) -> str:
    """n evenly spaced frames of both cameras (rows: front, wrist), decoded with torchcodec."""
    import cv2

    from src.data.recorder import load_demo_dataset

    key = str(capture)
    if key not in _DATASETS:
        _DATASETS[key] = load_demo_dataset(
            f"local/review-{capture.parent.name}", capture / "dataset"
        )
    ds = _DATASETS[key]
    ep = ds.meta.episodes[int(episode_index)]
    lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    picks = sorted({int(v) for v in np.linspace(lo, hi - 1, num=min(n, hi - lo))})
    rows = []
    for cam in conv.CAMERAS:
        tiles = []
        for i in picks:
            img = ds[i][cam.key]  # float CHW in [0, 1]
            a = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            tiles.append(cv2.resize(a, (320, 240)))
        rows.append(np.concatenate(tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    task = ds.meta.episodes[int(episode_index)]["tasks"][0]
    return str(task)


def make_packet(item: dict[str, Any], packet: Path) -> str:
    packet.mkdir(parents=True)
    capture = Path(item["capture"])
    task = contact_sheet(capture, item["episode_index"], packet / "trial_contact_sheet.png")
    stills = item.get("end_state")
    if isinstance(stills, dict):
        for name, rel in stills.items():
            src = capture / rel
            if src.is_file():
                shutil.copyfile(src, packet / f"end_state_{name}.png")
    return task


def review(
    sessions: Sequence[Path],
    reviewer: str,
    out_dir: Path,
    *,
    seed: int | None = None,
    partial_credit: bool = False,
    ask: Callable[[str], str] = input,
    show: Callable[[Path], None] | None = None,
) -> Path:
    items = [it for cap in sessions for it in session_items(cap)]
    if not items:
        sys.exit("no captured episodes to review")
    seed = int(seed if seed is not None else secrets.randbits(32))
    order = [int(i) for i in np.random.default_rng(seed).permutation(len(items))]
    out_dir.mkdir(parents=True, exist_ok=False)
    rec: dict[str, Any] = {
        "method": "blinded_review",
        "policy_identity_hidden": True,
        "reviewer_id": reviewer,
        "seed": seed,
        "rubric": {"id": PR.RUBRIC_ID, "version": PR.RUBRIC_VERSION, "rule": "R1_end_state"},
        "criterion": PR.JUDGING_RULE[PR.TASK],
        "partial_credit": partial_credit,
        "sessions": sorted({it["session_id"] for it in items}),
        "started_utc": _now(),
        "items": [],
        "complete": False,
    }
    path = out_dir / "review.json"
    print(
        f"{len(items)} trials, shuffled (seed {seed}). Judge each by the end state, as at the arm: "
        f"y only if {PR.JUDGING_RULE[PR.TASK]}. Doubt = n."
    )
    try:
        for n, i in enumerate(order, start=1):
            it = items[i]
            blind = secrets.token_hex(4)
            packet = out_dir / "packets" / blind
            task = make_packet(it, packet)
            if show is not None:
                show(packet)
            verdict = None
            while verdict is None:
                a = ask(f"[{n}/{len(items)}] {blind}: '{task}'. success? [y/n] ").strip().lower()
                verdict = {"y": "success", "n": "failure"}.get(a)
            partial = None
            if partial_credit:
                if verdict == "success":
                    partial = 1.0
                else:
                    lifted = ""
                    while lifted not in ("y", "n"):
                        lifted = (
                            ask("  was the object lifted off the table? [y/n] ").strip().lower()
                        )
                    partial = 0.5 if lifted == "y" else 0.0
            rec["items"].append(
                {
                    "blind_id": blind,
                    "session_id": it["session_id"],
                    "episode_index": it["episode_index"],
                    "trial_id": it["trial_id"],
                    "verdict": verdict,
                    "partial": partial,
                    "t_utc": _now(),
                }
            )
        rec["complete"] = True
    except (KeyboardInterrupt, EOFError):
        print("\nreview stopped: the verdicts so far are written (complete: false)")
    finally:
        rec["ended_utc"] = _now()
        path.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {path} ({len(rec['items'])}/{len(items)} reviewed)")
    return path


def _capture(runs_dir: Path, ref: str) -> Path:
    run_dir = Path(ref) if Path(ref).is_dir() else runs_dir / ref
    results = json.loads((run_dir / "record.json").read_text()).get("results") or {}
    sid = (results.get("session") or {}).get("session_id")
    if not sid:
        sys.exit(f"{run_dir} is not a wo07_real_eval session")
    return run_dir.parent / str(sid) / "capture"


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--session", action="append", required=True, help="run id of a session")
    parser.add_argument("--reviewer", required=True, help="the reviewer's id (recorded)")
    parser.add_argument("--seed", type=int, default=None, help="order seed (default: random)")
    parser.add_argument("--partial-credit", action="store_true", help="also the secondary column")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true", help="do not open packets in Finder")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("the blinded review is a real-arm step: --preset real")
    caps = [_capture(args.runs_dir, s) for s in args.session]
    out = args.out or args.runs_dir / f"review-{_now().replace(':', '')}-{args.reviewer}"

    def show(p: Path) -> None:
        subprocess.run(["open", str(p)], check=False)

    return review(
        caps,
        args.reviewer,
        out,
        seed=args.seed,
        partial_credit=args.partial_credit,
        show=None if args.no_open else show,
    )


if __name__ == "__main__":
    main()
