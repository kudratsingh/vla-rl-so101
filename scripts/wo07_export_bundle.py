#!/usr/bin/env python
"""WO-07: assemble the vishwaroop export inputs of one finished recording (``src/eval/export_bundle.py``).

One command per recording session; it refuses on any missing or unknown field (nothing guessed)
and writes the bundle next to the LeRobot dataset, then prints the owner's seal command. Sealing
is vishwaroop's tool (``docs/SO101_EXPORT.md`` there), never this repo's.

    # demos: a dataset recorded with scripts/wo06_teleop_record.py --demo-schedule
    uv run scripts/wo07_export_bundle.py --preset real demo --dataset data/real_teleop/<name>
    # an evaluation session of scripts/wo07_real_eval.py (protocol real-0.1); its newest run
    uv run scripts/wo07_export_bundle.py --preset real eval --session <run id> \\
        [--checkpoint-dir <dir>] [--review review-<reviewer>.json ...]

Demo bundle: ``<dataset>.export/``. Evaluation bundle: ``<runs>/<session>/capture/export/``
(``--out`` overrides; an existing bundle is never overwritten). Labels are declared
``blinded_review`` only with ``--review`` files from ``scripts/wo07_blind_review.py`` covering every
episode; otherwise the declaration has ``labels: null`` and the live scores stay in
``trials.json`` with their own provenance. Mock sessions need ``--allow-mock`` (a format test, not
data); development sessions ``--include-dev`` (never reported); an incomplete session
``--allow-incomplete``. A scored trial without its recording (``recording_missing``, or staged and
not yet recovered: resume the session first) refuses the export; ``--allow-missing-recordings``
builds it declared ``real-0.1-incomplete`` with ``status: INCOMPLETE`` (``src/eval/export_bundle.py``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args
from src.eval import export_bundle as X
from src.runs import RUNS_DIR


def _eval_script() -> Any:
    spec = importlib.util.spec_from_file_location(
        "wo07_real_eval_export", REPO_ROOT / "scripts" / "wo07_real_eval.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    ev = _eval_script()
    run_dir = Path(args.session) if Path(args.session).is_dir() else args.runs_dir / args.session
    settings, trials = ev.load_prior(run_dir)  # record.json + metrics + started-without-record
    ev.check_newest(run_dir, settings)  # refuses an older run of a continued session
    results = json.loads((run_dir / "record.json").read_text()).get("results") or {}
    mode = str(results.get("mode"))
    if mode == "mock" and not args.allow_mock:
        sys.exit(
            "refused: a mock session (fake arm and policy) needs --allow-mock: format test only"
        )
    if settings.get("schedule") != "eval" and not args.include_dev:
        sys.exit("refused: a development session is never reported; --include-dev to bundle it")
    summary = ev.table(trials, settings, _dcfg_stub())
    if not summary["complete"] and not args.allow_incomplete:
        sys.exit("refused: the session is incomplete (resume it first, or --allow-incomplete)")
    sid = str(settings["session_id"])
    capture = run_dir.parent / sid / "capture"
    ckpt = args.checkpoint_dir or (settings.get("policy_identity") or {}).get("checkpoint_dir")
    if ckpt is None:
        sys.exit("refused: --checkpoint-dir is required (the session recorded none)")
    reviews = [json.loads(Path(p).read_text()) for p in args.review or []]
    out = Path(args.out) if args.out else capture / "export"
    index = X.build_eval_bundle(
        capture / "dataset",
        settings,
        trials,
        out,
        checkpoint_dir=Path(ckpt),
        reviews=reviews,
        mode=mode,
        allow_missing_recordings=args.allow_missing_recordings,
    )
    return index, out


class _dcfg_stub:
    latency_target_ms = 150.0
    capture_offset_ms = None


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    sub = parser.add_subparsers(dest="kind", required=True)
    d = sub.add_parser("demo", help="a teleop demo recording")
    d.add_argument("--dataset", type=Path, required=True)
    d.add_argument("--out", type=Path, default=None)
    e = sub.add_parser("eval", help="an evaluation session (its newest run)")
    e.add_argument("--session", required=True, help="run id or run dir of the session")
    e.add_argument("--checkpoint-dir", type=Path, default=None)
    e.add_argument("--review", action="append", default=None, help="blinded review file(s)")
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--allow-mock", action="store_true")
    e.add_argument("--include-dev", action="store_true")
    e.add_argument("--allow-incomplete", action="store_true")
    e.add_argument(
        "--allow-missing-recordings",
        action="store_true",
        help="build an INCOMPLETE bundle (declared real-0.1-incomplete) when a scored trial has "
        "no recording",
    )
    e.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("exports are real-arm artefacts: --preset real")
    try:
        if args.kind == "demo":
            out = args.out or args.dataset.with_name(args.dataset.name + ".export")
            index = X.build_demo_bundle(args.dataset, out)
        else:
            index, out = eval_bundle(args)
    except X.ExportRefused as err:
        sys.exit(f"REFUSED: {err}")
    print(json.dumps(index, indent=2, default=str))
    if index.get("status") == "INCOMPLETE":
        print(
            f"\nINCOMPLETE EXPORT: {len(index['recording_missing'])} scored trial(s) have no "
            f"recording; declared protocol_version {index['protocol_version']}"
        )
    print("\nseal it (owner, in the vishwaroop repo):")
    print("  " + X.seal_command(index, out, "<new export id>", "<export root>"))
    return {**index, "bundle": str(out)}


if __name__ == "__main__":
    main()
