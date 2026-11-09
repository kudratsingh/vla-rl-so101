#!/usr/bin/env python
"""WO-07 real evaluation: 50 trials per protocol cell on the real SO-101, operator-scored.

One task per session, cells in protocol order (seen, shifted, unseen_objects), ``eval.trials_per_cell``
(= protocol REAL_TRIALS_PER_CELL, 50) trials per cell. Per trial: the arm goes home through the
safety filter, the operator places the objects on the slots this script prints (placement grid
``docs/assets/placement-grid-a3.pdf``, slots sampled by ``src/eval/real_grid.py`` from
``--slot-seed``: the same seed gives SFT and SFT+RL sessions the same placements, trial by trial),
the policy runs (``src/robot/client.py``), the arm holds still for the settling interval, then
the operator scores it y/n under the judging rule below. Every trial record carries protocol 0.2's
composite hash (``src/eval/protocol_v02.py``) with the 0.1-draft version and hash as a reference,
the placement and the placement plan's hash, the judging-rule version, the settling interval, the
real-eval file hashes and the session fingerprint. ``record.json`` holds the Wilson table.

Judging rule ``real-judge-2`` (``judging_rule()``; project review 2026-11-07 section 3.2: a real
judge must be defined before WO-07; real-judge-1 approved by the owner 2026-11-08, real-judge-2
implements the owner's trial-scoring decision of 2026-11-08): after the trial has ended, the arm
holds still for ``eval.settle_s`` seconds (the settling interval: objects stop rolling or
tipping), then it is judged ONCE (the prompt only appears then), by eye, without touching
anything. Real trials have no success latch: the sim stops at its first success step, the arm
keeps acting until ``trial_max_steps`` or a stop, so a success undone before the end is a failure
(stricter than sim). Doubt counts as a failure.

What real-judge-2 changes against real-judge-1 (which had the operator score faulted trials after
supporting the arm, and left trials with a failed torque-off unscored): a trial counts once its
policy STARTED (``policy_started``: the goal write of the trial's first action from the policy
server was begun, ``So101Driver.goal_write_attempted``; recorded with its UTC time; a first goal
blocked before its write, e.g. by the pre-write stop check after a hang-up, never started).
Every started trial is scored exactly once and is never redone. A fault after the start is a
FAILURE in the primary metric without a judgement, the fault recorded separately
(``fault: {kind, stop_reason, reason, where}``; kinds: servo, serial,
safety = a safety-layer or deadman stop (joint / workspace violation, lost frame, stall, stale
chunk), server, operator_abort (Ctrl-C during motion), client_error, torque_off_unconfirmed),
also when torque off is unconfirmed; a later operator 'resume' does not change it. The operator
still gets the support prompt (at the shutdown that follows: a torque-off fault ends the session).
A score prompt that ends without a judgement (Ctrl-C, closed stdin, no answer in
``deploy.prompt_torque_timeout_s``) is a failure, ``outcome_source: unjudged`` with the reason
(doubt = n applied to a missing judgement; owner check). An attempt whose policy never started
(the move home failed / faulted / was Ctrl-C'd, a server failure or a stale first chunk before the
first action, the first goal stopped before its write, a quit) is ``not_started``: kept in the
attempts log, never scored, and retried with the same placement on a resume. Primary metric:
successes / started trials, per cell, with the Wilson interval; ``diagnostic_excluding_faulted``
(success rate over started trials without a fault) and ``faults_by_kind`` are a clearly labelled
SECONDARY diagnostic. The rule's version and text hash are in the session fingerprint, so a
session scored under real-judge-1 can be neither resumed nor pooled with real-judge-2 records.

Start-rule wording change inside real-judge-2 (2026-11-09, PR fix/deploy-prewrite-stop; before
the real-eval protocol is frozen, no real session recorded): "started" moved from "the first
action reached the safety filter" to "the bus write of the first action was begun". Deliberate
reclassification: a first step that ends before its write (the pre-write stop check, a deadman
trip at its frame check or in the filter, a joint-read serial fault) is now ``not_started`` and
retried instead of a started failure. The text hash changed with it; the first FROZEN real-eval
protocol (PR #32, ``src/eval/protocol_real.py``) must carry this wording, and real-judge-2 is
cited only from that frozen version.

Paired placements (owner 2026-11-08): the SFT and the SFT+RL sessions use the SAME slot sequence.
``--hardware`` refuses a ``--slot-seed`` other than ``eval.slot_seed``, and the plan is
prefix-stable in the trial count (``src/eval/real_grid.py``), so trial k of a cell gets the same
placement in every session; every trial record carries ``placement_plan_sha256`` (the cell's
plan over max(trials, 50), ``plan_sha256``) to check the pairing afterwards.

Session fingerprint (owner 2026-11-08). A session is bound to the effective settings it ran with,
in six components hashed separately (``fingerprint_values``): ``protocol`` (protocol 0.2 hash, the
0.1-draft reference, judging rule and grid hashes), ``policy`` (the server's policy_id,
``server.policy_checkpoint`` and the served adapter's content hash ``adapter_sha256``, so a
re-pushed or overwritten adapter under the same name is a different policy; ``--hardware``
refuses a server that reports none), ``camera`` (the ``cameras`` block: index, resolution, fps,
intrinsics, crop, fov, pose), ``calibration`` (the calibration the driver loaded: per motor id,
drive mode, homing offset, range; the file's sha256; ``robot.units``), ``safety`` (the effective
``SafetyConfig``, ``robot.safety``, ``robot.servo_monitor``) and ``motion`` (control rate, the
effective deploy settings: chunk length and execution, timeouts, chunk age, trial steps, home and
park settings incl. ``robot.park.enabled``, the gripper floor, the settling interval). Every trial
record stores it. ``--resume`` refuses when any component changed, naming the component and the
changed keys, and the table refuses to pool trials whose fingerprints differ.

Resumable: ``--resume <run id>`` continues a session (any earlier run of it). The task, cells,
trials per cell, slot seed, label, policy id and fingerprint come from that run and must still
match. A slot with any attempt whose policy started is done (``done_indices``) and never re-run;
a slot whose attempts all ended before the policy started is retried with the same placement.
The start is durable: the moment the policy starts, a ``trial_started`` row goes to
metrics.jsonl, and a resume scores a started trial that has no record (the process was killed,
crashed or lost power mid-trial) as a FAILURE with fault kind ``session_lost`` (``load_prior``).
A first goal blocked after that row (a stop recorded during its I/O, or the row failed to save:
the session then ends ``start_marker_failed``) was never sent: a ``trial_start_retracted`` row
cancels the marker, and the slot is retried with the same placement.
Each invocation is its own run record, holding every trial of the session so far; record.json is
written when the session is bound and after every trial, so a run killed without exiting
(SIGKILL, power loss) can be resumed itself. Closing the Terminal (SIGHUP / SIGTERM) is not a
kill: the session takes the closed-stdin path (grace period, torque off, released_unconfirmed,
cause hangup) and writes its record. Ctrl-Z never suspends a hardware session (ignored with a
message); the session must run in the terminal's foreground, and losing it takes the same path
(cause lost_foreground).
``--resume`` refuses a run that a later run of its session continued (a resume of an older
run would fork the session and drop the newer runs' scored trials), and warns (``--hardware``: asks for
'checked') when an earlier run ended with TORQUE OFF UNCONFIRMED, in a trial or at shutdown.

--mock      CPU, in-process, safe anywhere: MockPolicy behind gRPC on 127.0.0.1, the fake arm and
            camera from tests/robot_fakes.py, the real safety filter and sim FK. Scripted scores.
            NOT a result.
--hardware  OWNER ONLY, never run by tests or agents: gated exactly like scripts/wo07_pilot.py
            (refuses without --i-am-present, or a null / non-positive
            deploy.prompt_torque_timeout_s, then needs 'go' typed). Every prompt answered with
            torque on reads servo health before and after and waits at most that timeout (then
            "support the arm now", the grace period, torque off, the session ends). A normal end
            holds the arm, asks for support, and turns torque off only after ENTER (Ctrl-C there
            is ignored; a closed stdin or the timeout is recorded as released_unconfirmed,
            support_confirmed false); robot.park.enabled (default false) parks first. Ctrl-C
            during motion holds (outside prompts Ctrl-C is a flag: motion stops between ticks,
            never inside a bus transfer or the shutdown); a servo over-temperature / over-current fault cuts torque at
            once; a failed torque-off blocks all motion and says to switch off the inline power
            switch.

    uv run scripts/wo07_real_eval.py --preset real --mock --fake-clock --trials 2 --max-steps 16
    .venv/bin/python scripts/wo07_real_eval.py --preset real --hardware --i-am-present \\
        --task PickPlace --label sft
    .venv/bin/python scripts/wo07_real_eval.py --preset real --hardware --i-am-present --resume <id>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.eval import protocol as P1
from src.eval import protocol_v02 as P2
from src.eval import real_grid as G
from src.robot.camera import PolicyImagePipeline
from src.robot.client import (
    ConsoleOperator,
    DeployConfig,
    DeployLoop,
    Operator,
    PolicyClient,
    ScriptedOperator,
    TrialResult,
    check_server,
    final_session_end,
    latency_lines,
    run_trials,
    shutdown_or_record,
    shutdown_results,
    summarize,
    trial_object,
)
from src.robot.driver import So101Driver
from src.robot.prompts import TorqueGuard, TorqueWait, operator_signals, require_terminal
from src.robot.safety import FkFn, load_sim_fk, make_safety
from src.robot.serve import MockPolicy, PolicyServer, build_server
from src.robot.servo_health import monitor_from_config
from src.robot.units import UnitMap
from src.runs import RUNS_DIR, Run, new_run_id

JUDGING_RULE_VERSION = "real-judge-2"
DEFAULT_SETTLE_S = 3.0  # eval.settle_s
JUDGING_GENERAL = (
    "Judge ONCE, after the trial has ended and the arm has then held still for the settling "
    "interval of {settle_s:g} s (eval.settle_s; this prompt only appears then), by eye, touching "
    "nothing. y only if ALL of the task's conditions hold NOW; a success that was undone before "
    "the end is n. Doubt = n. A trial counts once its policy started (the bus write of its first "
    "action from the policy server was begun; one stopped before its write never started) and is "
    "then scored exactly once, never redone. A "
    "fault after the start (servo, serial, a safety-layer or deadman stop: joint / workspace "
    "violation, lost frame, stall, stale chunk; a policy-server failure; Ctrl-C during motion; a "
    "client error; torque off unconfirmed; inside the trial, at the settle read or while this "
    "prompt waits) makes it a failure without a judgement (the fault is recorded; a later "
    "'resume' does not change it). A score prompt that ends without a judgement (Ctrl-C, closed "
    "input, no answer in deploy.prompt_torque_timeout_s) scores it n. An attempt whose policy "
    "never started is not scored and is redone with the same placement."
)
JUDGING_RULE: dict[str, str] = {  # mirrors src/env/success.py, read only
    "PickPlace": "the object rests inside the bowl (over the bowl floor, below the rim), the bowl "
    "is upright, and the object touches neither gripper jaw",
    "Push": "the object's centre is inside the 5 cm target disc, the object is on the table, it "
    "was never lifted off the table during the trial, and the gripper is not holding it",
    "Stack": "the object rests on top of the blue cube (centred within ~1.5 cm, flat), the blue "
    "cube is still on the table, and the object touches neither gripper jaw",
}

MOCK_LABEL = "MOCK: fake arm, fake camera, mock policy; exercises the eval loop only, NOT a result"
HARDWARE_LABEL = "real arm, operator-scored, single session per checkpoint"

HARDWARE_REFUSAL = """--hardware drives the real SO-101 and refuses to run without --i-am-present.

Only the owner runs it, standing at the arm with a hand on the power switch, after WO-06
(calibration, intrinsics, camera match), the WO-07 pilot, the placement grid printed and its
10 cm scale bar measured, and with the policy server reachable through the tunnel."""

PRIOR_TORQUE_OFF_UNCONFIRMED = (
    "WARNING: an earlier run of this session ended with TORQUE OFF UNCONFIRMED (in a trial or "
    "at shutdown). Check the bus, the USB cable and every servo before powering the arm."
)

HARDWARE_BANNER = """
==========================================================================================
 THIS WILL MOVE THE REAL ARM, up to 50 trials per cell.
 Before every trial it drives to the home pose (deploy.home_qpos) through the safety filter,
 then the policy moves it. Keep the workspace clear and a hand on the power switch.
 After each trial the arm holds still for eval.settle_s, then you score it (doubt = n).
 A fault after the policy started is a failure (no score asked); support the arm.
 No answer within deploy.prompt_torque_timeout_s at a prompt with torque on: SUPPORT THE
 ARM, torque goes OFF after the grace period, the session ends (resume later).
 A normal end (all trials, or q / Ctrl-C at a prompt) holds the arm: support it, press
 ENTER, torque goes off (robot.park.enabled true: it PARKS first, it moves). Ctrl-C while
 the arm moves holds it; after ENTER torque goes off and the arm drops: support it. A servo
 over-temperature / over-current fault cuts torque AT ONCE: the arm drops, support it.
 TORQUE OFF UNCONFIRMED: switch off the servo supply at the inline power switch.
==========================================================================================
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def settle_s_from_config(cfg: Config) -> float:
    v = float((cfg.get("eval") or {}).get("settle_s", DEFAULT_SETTLE_S))
    if not (math.isfinite(v) and 0.0 <= v <= 30.0):
        raise ValueError(f"eval.settle_s must be in [0, 30] s, got {v}")
    return v


def judging_rule(settle_s: float) -> dict[str, Any]:
    """The judging rule (``JUDGING_RULE_VERSION``), with its settling interval."""
    return {
        "version": JUDGING_RULE_VERSION,
        "general": JUDGING_GENERAL.format(settle_s=settle_s),
        "tasks": JUDGING_RULE,
        "settle_s": float(settle_s),
    }


def protocol_fields(settle_s: float = DEFAULT_SETTLE_S) -> dict[str, Any]:
    """Per-trial protocol citation: 0.2 composite hash, 0.1-draft as a reference, real-eval files."""
    rule_text = json.dumps(judging_rule(settle_s), sort_keys=True)
    return {
        "protocol_version": P2.PROTOCOL_VERSION,
        "protocol_hash": P2.protocol_hash(),
        "protocol_ref": {"version": P1.PROTOCOL_VERSION, "hash": P1.protocol_hash()},
        "real_eval": {
            "judging_rule": JUDGING_RULE_VERSION,
            "judging_rule_sha256": hashlib.sha256(rule_text.encode()).hexdigest(),
            "settle_s": float(settle_s),
            "grid_sha256": _sha256(REPO_ROOT / "src" / "eval" / "real_grid.py"),
            "grid_pdf_sha256": _sha256(REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf"),
        },
    }


class JudgeOperator(ConsoleOperator):
    """The person at the arm, scoring under ``judging_rule(settle_s)``."""

    def __init__(self, guard: TorqueGuard, task: str, settle_s: float = DEFAULT_SETTLE_S):
        super().__init__(guard)
        self.task = task
        self.settle_s = settle_s

    def outcome(self, result: TrialResult) -> str:
        rule = judging_rule(self.settle_s)
        print(f"\ntrial ended ({result.stop_reason}, {result.steps} steps). {rule['general']}")
        print(f"  {self.task} success = {JUDGING_RULE[self.task]}.")
        return super().outcome(result)


# ------------------------------------------------------------------------------------------------
# Session fingerprint and paired placements
# ------------------------------------------------------------------------------------------------
FINGERPRINT_COMPONENTS = ("protocol", "policy", "camera", "calibration", "safety", "motion")
# deploy fields that only label the latency report; they never change what the arm does
_REPORT_ONLY = ("latency_target_ms", "capture_offset_ms", "policy_checkpoint")


def _canon(v: Any) -> Any:
    """JSON-normal form: tuples -> lists, keys sorted, non-JSON values as strings."""
    return json.loads(json.dumps(v, sort_keys=True, default=str))


def _digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibration_values(arm: So101Driver, cfg: Config) -> dict[str, Any]:
    """The calibration the driver loaded (lerobot reads ``<calibration_dir>/<id>.json`` when the
    follower is built, before any port is opened), the file's sha256, and ``robot.units``."""
    fol = arm.follower
    cal = getattr(fol, "calibration", None) or {}
    keys = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
    path = getattr(fol, "calibration_fpath", None)
    file_sha = _sha256(Path(path)) if path is not None and Path(path).is_file() else None
    return {
        "motors": {m: {k: int(getattr(c, k)) for k in keys} for m, c in sorted(cal.items())},
        "file_sha256": file_sha,
        "units": cfg["robot"].get("units"),
    }


def fingerprint_values(
    cfg: Config,
    arm: So101Driver,
    info: dict[str, Any],
    dcfg: DeployConfig,
    *,
    max_steps: int | None,
    settle_s: float,
) -> dict[str, Any]:
    """The effective settings a session is bound to, per component (module doc)."""
    r = cfg["robot"]
    sc = arm.safety.config
    deploy = {
        k: v for k, v in dataclasses.asdict(dcfg).items() if k not in _REPORT_ONLY
    }  # control_hz, chunk_len, exec_steps, timeouts, chunk age, trial steps, home, park
    return _canon(
        {
            "protocol": protocol_fields(settle_s),
            "policy": {
                "policy_id": info.get("policy_id"),
                "policy_checkpoint": dcfg.policy_checkpoint,
                "adapter_sha256": info.get("adapter_sha256"),  # the weights, not the name
                "adapter_revision": info.get("adapter_revision"),
            },
            "camera": {"cameras": cfg["cameras"]},
            "calibration": calibration_values(arm, cfg),
            "safety": {
                "effective": dataclasses.asdict(sc),
                "robot.safety": r.get("safety"),
                "robot.servo_monitor": r.get("servo_monitor"),
            },
            "motion": {
                "robot.control_hz": r.get("control_hz"),
                "deploy": deploy,
                "trial_max_steps_effective": (
                    dcfg.trial_max_steps if max_steps is None else int(max_steps)
                ),
                "robot.park": r.get("park"),
                "park_enabled": dcfg.park is not None,
                "gripper_floor_deg": round(math.degrees(sc.lower_rad[-1]), 4),
                "settle_s": float(settle_s),
            },
        }
    )


def make_fingerprint(values: dict[str, Any]) -> dict[str, Any]:
    comps = {c: _digest(values[c]) for c in FINGERPRINT_COMPONENTS}
    return {"sha256": _digest(comps), "components": comps}


def _flat(v: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(v, dict):
        out: dict[str, Any] = {}
        for k, x in v.items():
            out.update(_flat(x, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: v}


def fingerprint_diff(
    old: dict[str, Any], new_values: dict[str, Any], max_keys: int = 6
) -> list[str]:
    """One line per changed component, naming up to ``max_keys`` changed settings."""
    new = make_fingerprint(new_values)
    old_vals = old.get("values") or {}
    lines = []
    for c in FINGERPRINT_COMPONENTS:
        if old.get("components", {}).get(c) == new["components"][c]:
            continue
        a, b = _flat(old_vals.get(c, {})), _flat(new_values[c])
        changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        detail = ", ".join(f"{k}: {a.get(k)!r} -> {b.get(k)!r}" for k in changed[:max_keys])
        more = f" (+{len(changed) - max_keys} more)" if len(changed) > max_keys else ""
        lines.append(f"{c} changed: {detail or 'no recorded values to compare'}{more}")
    return lines


def check_poolable(trials: Sequence[dict[str, Any]], fingerprint: dict[str, Any]) -> None:
    """Refuse to pool trials recorded under a different session fingerprint (or none)."""
    want = fingerprint["sha256"]
    bad = [t for t in trials if (t.get("session_fingerprint") or {}).get("sha256") != want]
    if not bad:
        return
    lines = []
    for t in bad[:5]:
        got = t.get("session_fingerprint") or {}
        comps = [
            c
            for c in FINGERPRINT_COMPONENTS
            if (got.get("components") or {}).get(c) != fingerprint["components"][c]
        ]
        lines.append(
            f"{t.get('cell')} trial {t.get('trial_idx')} ({t.get('start_utc')}): "
            f"{'no fingerprint' if not got else 'differs in ' + ', '.join(comps)}"
        )
    raise PoolingRefused(
        f"refusing to pool {len(bad)} trial(s) recorded under different session settings:\n  "
        + "\n  ".join(lines)
    )


class PoolingRefused(RuntimeError):
    """Trials from different effective settings would be pooled into one table."""


def plan_sha256(task: str, cell: str, slot_seed: int, n: int = P1.REAL_TRIALS_PER_CELL) -> str:
    """Hash of the cell's placement plan over max(n, REAL_TRIALS_PER_CELL) trials (the paired
    slot sequence). The plan is prefix-stable, so a session of up to 50 trials carries the same
    hash as a 50-trial one, and a longer session's hash covers every placement it used."""
    n_hash = max(int(n), P1.REAL_TRIALS_PER_CELL)
    return _digest(_canon(G.sample_plan(task, cell, n_hash, slot_seed)))


# ------------------------------------------------------------------------------------------------
# Session state (resume)
# ------------------------------------------------------------------------------------------------
SCORED = ("success", "failure")


def load_prior(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(session settings, every trial record so far) from an earlier run of the session. Trials
    come from record.json, plus any metrics.jsonl trial rows it lacks (a run killed before exit),
    plus every ``trial_started`` row without its trial: the policy started and the process died
    before the trial was recorded (killed, crashed, power lost). That row is the record as it
    stood at the start, a FAILURE with fault kind ``session_lost``: the started slot is done and
    is never re-run (real-judge-2). A ``trial_started`` row with a ``trial_start_retracted`` row
    of the same attempt (cell, trial_idx, start_utc) is dropped: the first goal was blocked after
    the marker (a stop recorded during its I/O, a frame / chunk gone stale during it, or the
    marker failed), nothing was sent, the attempt never started and the slot is retried
    (``DeployLoop._first_goal_check``)."""
    rec_path = run_dir / "record.json"
    if not rec_path.is_file():
        sys.exit(f"--resume: no {rec_path}")
    results = json.loads(rec_path.read_text()).get("results", {})
    settings = results.get("session")
    if not settings:
        sys.exit(f"--resume: {rec_path} is not a wo07_real_eval session")
    if prior_run_torque_off_unconfirmed(results):  # carried by every later run of the session
        settings["prior_torque_off_unconfirmed"] = True
    trials: list[dict[str, Any]] = list(results.get("trials") or [])
    seen = {(t["cell"], t["trial_idx"], t["start_utc"]) for t in trials}
    metrics = run_dir / "metrics.jsonl"
    started: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    retracted: set[tuple[Any, Any, Any]] = set()
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # a line cut short by the kill
                continue
            if row.get("kind") not in ("trial", "trial_started", "trial_start_retracted"):
                continue
            key = (row["cell"], row["trial_idx"], row["start_utc"])
            if row["kind"] == "trial_start_retracted":
                retracted.add(key)
                continue
            rec = {k: v for k, v in row.items() if k not in ("kind", "t", "step")}
            if row["kind"] == "trial_started":
                started.setdefault(key, rec)
            elif key not in seen:
                seen.add(key)
                trials.append(rec)
    for key, rec in started.items():
        if key in retracted and key not in seen:
            print(
                f"--resume: {key[0]} trial {key[1]}: its start was retracted (the first goal was "
                "blocked before its write): not started, the slot is retried"
            )
        elif key not in seen:
            seen.add(key)
            print(
                f"--resume: {key[0]} trial {key[1]} started at {rec.get('policy_started_utc')} and "
                "has no record (the run ended mid-trial): scored a FAILURE (session_lost)"
            )
            trials.append(rec)
    return settings, trials


def prior_run_torque_off_unconfirmed(results: dict[str, Any]) -> bool:
    """The run's own session end / shutdown shows the latch (it may have fired after the last
    trial, in safe_shutdown, so no trial carries it), or an earlier run of the session did."""
    return bool(
        results.get("torque_off_unconfirmed")
        or str(results.get("session_end") or "").startswith("torque_off_unconfirmed")
        or str(results.get("shutdown") or "").startswith("torque_off_unconfirmed")
        or (results.get("session") or {}).get("prior_torque_off_unconfirmed")
    )


def session_runs(runs_dir: Path, session_id: str) -> dict[str, str | None]:
    """{run id: the run it resumed from} for every run whose record holds this session."""
    out: dict[str, str | None] = {}
    for rec in sorted(runs_dir.glob("*/record.json")):
        try:
            results = json.loads(rec.read_text()).get("results") or {}
        except (OSError, ValueError):
            continue
        if (results.get("session") or {}).get("session_id") == session_id:
            out[rec.parent.name] = results.get("resumed_from")
    return out


def check_newest(prior_dir: Path, settings: dict[str, Any]) -> list[str]:
    """Refuse to resume from a run that a later run of the same session continued (a fork would
    drop that run's scored trials and re-roll them). Returns the chain of runs up to the chosen
    one, oldest first."""
    sid = str(settings.get("session_id"))
    runs = session_runs(prior_dir.parent, sid)
    children: dict[str, list[str]] = {}
    for rid, parent in runs.items():
        if parent:
            children.setdefault(parent, []).append(rid)
    me = prior_dir.name
    if me in children:
        tip = me
        while tip in children:
            tip = max(children[tip])
        sys.exit(
            f"--resume refused: run {min(children[me])} continued session {sid} after {me}; "
            f"resume {tip} (its newest run), or its scored trials would be dropped and re-run"
        )
    chain = [me]
    while (parent := runs.get(chain[0])) and parent not in chain:
        chain.insert(0, parent)
    return chain


def done_indices(trials: Sequence[dict[str, Any]], cell: str) -> set[int]:
    """Slots of ``cell`` with an attempt whose policy started: scored once, never re-run."""
    return {int(t["trial_idx"]) for t in trials if t["cell"] == cell and t.get("policy_started")}


def table(
    trials: Sequence[dict[str, Any]], settings: dict[str, Any], cfg_deploy: DeployConfig
) -> dict[str, Any]:
    """client.summarize over the first started record per (cell, trial_idx) (a started slot is
    never re-run, so there is one) plus every not-started attempt, and completeness. Raises
    PoolingRefused when a trial's session fingerprint differs from the session's."""
    if settings.get("fingerprint"):
        check_poolable(trials, settings["fingerprint"])
    first: dict[tuple[str, int], dict[str, Any]] = {}
    attempts = []
    for t in trials:
        if t.get("policy_started") and t["outcome"] in SCORED:
            first.setdefault((t["cell"], int(t["trial_idx"])), t)
        else:
            attempts.append(t)
    scored = list(first.values())
    s = summarize(scored + attempts, cfg_deploy.latency_target_ms, cfg_deploy.capture_offset_ms)
    n = int(settings["trials_per_cell"])
    for c in s["cells"].values():
        c["target_n"] = n
        c["complete"] = c["n"] == n
    s["complete"] = all(
        s["cells"].get(f"{settings['task']}/{cell}", {}).get("complete", False)
        for cell in settings["cells"]
    )
    # a protocol table needs every protocol cell at REAL_TRIALS_PER_CELL scored trials
    s["protocol_complete"] = (
        s["complete"] and n == P1.REAL_TRIALS_PER_CELL and set(settings["cells"]) == set(P1.CELLS)
    )
    s.update(
        protocol_version=P2.PROTOCOL_VERSION,
        protocol_hash=settings["protocol"]["protocol_hash"],
        protocol_ref=settings["protocol"]["protocol_ref"],
        label=settings["label"],
        judging_rule=JUDGING_RULE_VERSION,
        session_fingerprint=(settings.get("fingerprint") or {}).get("sha256"),
    )
    return s


def print_table(
    run_id: str, s: dict[str, Any], end: str, shutdown: dict[str, Any] | None = None
) -> None:
    print(f"\nrun {run_id}  (session end: {end})  label {s['label']}")
    if shutdown is not None:
        cause = f" ({shutdown['cause']})" if shutdown.get("cause") else ""
        print(
            f"shutdown: {shutdown['end']}{cause}  support confirmed: {shutdown['support_confirmed']}"
        )
    print("PRIMARY: successes / started trials (a fault after the start is a failure)")
    print("| cell | successes / started | rate | Wilson 95% | faulted | unjudged | not started |")
    print("|---|---|---|---|---|---|---|")
    for name, c in s["cells"].items():
        rate = "n/a" if c["success_rate"] is None else f"{100 * c['success_rate']:.1f}%"
        lo, hi = c["wilson_95"]
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(c["faults_by_kind"].items()))
        print(
            f"| {name} | {c['successes']} / {c['n']} (of {c['target_n']}) | {rate} | "
            f"[{100 * lo:.1f}, {100 * hi:.1f}] | {c['faulted']}{f' ({kinds})' if kinds else ''} "
            f"| {c['unjudged']} | {c['not_started']} |"
        )
    print("SECONDARY diagnostic (NOT the result): success rate excluding faulted trials")
    for name, c in s["cells"].items():
        d = c["diagnostic_excluding_faulted"]
        d_rate = "n/a" if d["success_rate"] is None else f"{100 * d['success_rate']:.1f}%"
        print(f"  {name}: {d['successes']} / {d['n']} = {d_rate}")
    for line in latency_lines(s):
        print(line)
    print(
        f"  protocol {s['protocol_version']} {s['protocol_hash'][:12]} (ref "
        f"{s['protocol_ref']['version']} {s['protocol_ref']['hash'][:12]}), judge {s['judging_rule']}"
    )
    if not s["protocol_complete"]:
        print(
            f"  NOT a protocol table: needs {P1.REAL_TRIALS_PER_CELL} scored trials in each of "
            f"{', '.join(P1.CELLS)}"
        )
    if s.get("torque_off_unconfirmed") or end.startswith("torque_off_unconfirmed"):
        print("  TORQUE OFF UNCONFIRMED: keep the servo supply OFF at the inline power switch")


# ------------------------------------------------------------------------------------------------
# Session
# ------------------------------------------------------------------------------------------------
def session(
    cfg: Config,
    loop: DeployLoop,
    operator: Operator,
    info: dict[str, Any],
    *,
    mode: str,
    settings: dict[str, Any],
    prior_trials: list[dict[str, Any]],
    resumed_from: str | None,
    max_steps: int | None,
    camera_calibrated: bool,
    runs_dir: Path,
    on_trial_start: Callable[[int], None] | None = None,
    prior_runs: Sequence[str] = (),
    torque_wait: TorqueWait | None = None,
) -> Path:
    task, n = str(settings["task"]), int(settings["trials_per_cell"])
    settle_s = float(settings["settle_s"])
    trials: list[dict[str, Any]] = list(prior_trials)
    end = "error"
    fingerprint = settings["fingerprint"]
    extra = {
        **settings["protocol"],
        "session_id": settings["session_id"],
        "label": settings["label"],
        "mode": mode,
        "session_fingerprint": {k: fingerprint[k] for k in ("sha256", "components")},
    }
    with Run(
        work_order="WO-07",
        script=__file__,
        cfg=cfg,
        device="cpu",
        runs_dir=runs_dir,
        run_id=new_run_id("WO-07", f"realeval-{mode}"),
    ) as run:
        if not settings.get("session_id"):
            settings["session_id"] = extra["session_id"] = run.id
        run.results.update(
            mode=mode,
            label=MOCK_LABEL if mode == "mock" else HARDWARE_LABEL,
            session=settings,
            resumed_from=resumed_from,
            server_info=info,
            judging_rule=judging_rule(settle_s),
            session_runs=[*prior_runs, run.id],
            torque_wait=None if torque_wait is None else torque_wait.record(),
            trials=trials,  # the same list: every flush carries the trials so far
        )
        run.flush()  # the session is on disk before the first motion: a killed run resumes

        def log_row(**row: Any) -> None:
            run.log_metrics(**row)
            if row.get("kind") == "trial":
                run.flush()

        print(f"{judging_rule(settle_s)['general']}\n  {task} success = {JUDGING_RULE[task]}.")
        try:
            for cell in settings["cells"]:
                done = done_indices(trials, cell)
                todo = [k for k in range(n) if k not in done]
                if not todo:
                    continue
                plan = G.sample_plan(task, cell, n, int(settings["slot_seed"]))

                def place(
                    k: int, cell: str = cell, plan: list[dict[str, Any]] = plan
                ) -> tuple[str, dict[str, Any]]:
                    obj = trial_object(task, cell, k)
                    return G.placement_prompt(obj.label, plan[k]), plan[k]

                _, why = run_trials(
                    loop,
                    operator,
                    task=task,
                    cell=cell,
                    n_trials=n,
                    info=info,
                    camera_calibrated=camera_calibrated,
                    log_row=log_row,
                    max_steps=max_steps,
                    trials=trials,
                    on_trial_start=on_trial_start,
                    indices=todo,
                    placement_fn=place,
                    record_extra={
                        **extra,
                        "placement_plan_sha256": plan_sha256(
                            task, cell, int(settings["slot_seed"]), n
                        ),
                    },
                    settle_s=settle_s,
                )
                if why != "completed":
                    end = why
                    break
            else:
                end = "completed"
        except KeyboardInterrupt:
            end = "operator_ctrl_c"
        except Exception as e:  # never a healthy end: hold + support, no park
            end = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            shutdown = shutdown_or_record(run.results, loop, operator, end)
            end = final_session_end(loop.arm, end)  # a failed torque-off wins over any end
            run.results.update(
                session_end=end,
                **shutdown_results(shutdown),
                torque_off_unconfirmed=loop.arm.torque_off_unconfirmed,
                trials=trials,
            )
            summary = table(trials, settings, loop.dcfg)
            run.results.update(summary=summary)
        print_table(run.id, summary, end, shutdown.record())
        if not summary["complete"]:
            print(f"  session incomplete: resume with --resume {run.id}")
    return run.record_path


def new_settings(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    ecfg = cfg.get("eval") or {}
    n = int(
        args.trials
        if args.trials is not None
        else ecfg.get("trials_per_cell", P1.REAL_TRIALS_PER_CELL)
    )
    return {
        "session_id": None,  # the first run's id
        "task": args.task,
        "cells": list(args.cells or ecfg.get("cells") or P1.CELLS),
        "trials_per_cell": n,
        "protocol_trials_per_cell": P1.REAL_TRIALS_PER_CELL,
        "slot_seed": int(
            args.slot_seed if args.slot_seed is not None else ecfg.get("slot_seed", 0)
        ),
        "label": args.label,
        "policy_id": None,  # from the server's Info
        "settle_s": settle_s_from_config(cfg),
        "protocol": protocol_fields(settle_s_from_config(cfg)),
        "fingerprint": None,  # bind_session, once the rig and the server are known
    }


def check_resume(
    settings: dict[str, Any], info: dict[str, Any], values: dict[str, Any] | None = None
) -> None:
    """Refuse to resume when the protocol, the policy or (with ``values``) any fingerprint
    component differs from the session's."""
    now = protocol_fields(float(settings.get("settle_s", DEFAULT_SETTLE_S)))
    bad = [
        f"{k}: session {settings['protocol'][k]!r} != now {now[k]!r}"
        for k in ("protocol_hash", "protocol_ref", "real_eval")
        if settings["protocol"][k] != now[k]
    ]
    if settings.get("policy_id") not in (None, info.get("policy_id")):
        bad.append(
            f"policy_id: session {settings['policy_id']!r} != server {info.get('policy_id')!r}"
        )
    if values is not None:
        old = settings.get("fingerprint")
        if not old:
            bad.append("fingerprint: the session has none (recorded before fingerprints)")
        elif old.get("sha256") != make_fingerprint(values)["sha256"]:
            bad += fingerprint_diff(old, values)
    if bad:
        sys.exit("--resume refused, the session's evaluation changed:\n  " + "\n  ".join(bad))


def bind_session(
    cfg: Config,
    settings: dict[str, Any],
    prior: Sequence[dict[str, Any]],
    resumed_from: str | None,
    arm: So101Driver,
    info: dict[str, Any],
    dcfg: DeployConfig,
    max_steps: int | None,
    acknowledge: Callable[[str], str] | None = None,
) -> None:
    """Fingerprint the effective settings; on a resume refuse any change, and refuse to pool
    earlier trials with a different fingerprint. Runs before the arm is connected on hardware.
    An earlier TORQUE OFF UNCONFIRMED (any trial, or a run's session end / shutdown) is warned
    about; with ``acknowledge`` (hardware) the operator must type 'checked'."""
    values = fingerprint_values(
        cfg, arm, info, dcfg, max_steps=max_steps, settle_s=settle_s_from_config(cfg)
    )
    if resumed_from:
        check_resume(settings, info, values)
        try:
            check_poolable(prior, settings["fingerprint"])
        except PoolingRefused as e:
            sys.exit(f"--resume refused: {e}")
        if settings.get("prior_torque_off_unconfirmed") or any(
            t.get("torque_off_unconfirmed") for t in prior
        ):
            print(PRIOR_TORQUE_OFF_UNCONFIRMED)
            if (
                acknowledge is not None
                and acknowledge(
                    "type 'checked' once the bus, the USB cable and every servo are checked: "
                ).strip()
                != "checked"
            ):
                sys.exit("not acknowledged; nothing was opened")
    else:
        settings["fingerprint"] = {**make_fingerprint(values), "values": values}
    settings["policy_id"] = info.get("policy_id")


@contextmanager
def mock_rig(
    cfg: Config, args: argparse.Namespace, fk: FkFn | None
) -> Iterator[tuple[DeployLoop, dict[str, Any], Callable[[int], None], bool]]:
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from robot_fakes import FakeCam, FakeClock, FakeFollower

    spec = importlib.util.spec_from_file_location(
        "wo07_pilot", REPO_ROOT / "scripts" / "wo07_pilot.py"
    )
    assert spec is not None and spec.loader is not None
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    if args.fake_clock:
        fake = FakeClock()
        clock, sleep = fake, fake.sleep
    else:
        clock, sleep = time.perf_counter, time.sleep
    dcfg = DeployConfig.from_config(cfg)
    policy = MockPolicy(
        pilot.mock_waypoints(dcfg.home_qpos), chunk_len=dcfg.chunk_len, infer_s=0.06, sleep=sleep
    )
    servicer = PolicyServer(
        policy,
        image_size=dcfg.image_size,
        policy_id="mock:pickplace-waypoints",
        clock=clock,
        adapter={"adapter_sha256": "mock:no-weights", "adapter_revision": None},
    )
    server, port = build_server(servicer, "127.0.0.1", 0)
    try:
        arm = So101Driver(
            FakeFollower(),
            UnitMap.from_config(cfg),
            make_safety(cfg, fk or load_sim_fk(), clock=clock),
            clock=clock,
            monitor=monitor_from_config(cfg),
            sleep=sleep,
        )
        pipe = PolicyImagePipeline.from_config(cfg)
        with (
            PolicyClient(
                f"127.0.0.1:{port}",
                timeout_s=dcfg.request_timeout_s,
                chunk_len=dcfg.chunk_len,
                action_dim=dcfg.action_dim,
                clock=clock,
            ) as client,
            arm,
        ):
            info = client.info()
            check_server(info, dcfg)
            loop = DeployLoop(arm, FakeCam(clock), pipe, client, dcfg, clock=clock, sleep=sleep)
            yield loop, info, lambda _k: policy.reset(), pipe.calibrated
    finally:
        policy.release.set()
        server.stop(grace=0)


def run(
    cfg: Config,
    args: argparse.Namespace,
    *,
    fk: FkFn | None = None,
    operator: Operator | None = None,
) -> Path:
    if args.resume:
        prior_dir = Path(args.resume) if Path(args.resume).is_dir() else args.runs_dir / args.resume
        settings, prior = load_prior(prior_dir)
        prior_runs = check_newest(prior_dir, settings)
        resumed_from: str | None = prior_dir.name
    else:
        if args.task is None:
            sys.exit("--task is required for a new session")
        settings, prior, resumed_from, prior_runs = new_settings(args, cfg), [], None, []
    if args.hardware:
        return run_hardware(cfg, args, settings, prior, resumed_from, prior_runs)
    torque_wait = TorqueWait.from_config(cfg, "deploy", allow_none=True)  # validated; mock: unused
    with mock_rig(cfg, args, fk) as (loop, info, on_start, calibrated):
        bind_session(cfg, settings, prior, resumed_from, loop.arm, info, loop.dcfg, args.max_steps)
        return session(
            cfg,
            loop,
            operator or ScriptedOperator(),
            info,
            mode="mock",
            settings=settings,
            prior_trials=prior,
            resumed_from=resumed_from,
            max_steps=args.max_steps,
            camera_calibrated=calibrated,
            runs_dir=args.runs_dir,
            on_trial_start=on_start,
            prior_runs=prior_runs,
            torque_wait=torque_wait,
        )


def run_hardware(
    cfg: Config,
    args: argparse.Namespace,
    settings: dict[str, Any],
    prior: list[dict[str, Any]],
    resumed_from: str | None,
    prior_runs: Sequence[str] = (),
) -> Path:
    if not args.i_am_present:
        print(HARDWARE_REFUSAL)
        sys.exit(2)
    try:  # a null or non-positive torque-on bound is refused before anything is opened
        torque_wait = TorqueWait.from_config(cfg, "deploy", allow_none=False)
    except ValueError as e:
        sys.exit(f"{e}; nothing was opened")
    assert torque_wait is not None
    require_terminal("wo07_real_eval --hardware")  # prompts need an operator; before anything opens
    print(HARDWARE_BANNER)
    if input("type 'go' to continue: ").strip() != "go":
        sys.exit("not confirmed; nothing was opened")
    cam_cfg = cfg["cameras"]["front"]
    if not cam_cfg.get("intrinsics"):
        sys.exit("cameras.front.intrinsics is null: run scripts/wo06_camera_intrinsics.py first")
    cfg_seed = int((cfg.get("eval") or {}).get("slot_seed", 0))
    if int(settings["slot_seed"]) != cfg_seed:
        sys.exit(
            f"slot seed {settings['slot_seed']} != eval.slot_seed {cfg_seed}: the SFT and SFT+RL "
            "sessions must use the same placements (owner 2026-11-08); drop --slot-seed"
        )
    from src.robot.camera import OpenCVFrameSource

    dcfg = DeployConfig.from_config(cfg)
    fk = load_sim_fk()
    pipe = PolicyImagePipeline.from_config(cfg)
    target = f"{cfg['server']['host']}:{cfg['server']['port']}"
    with PolicyClient(
        target,
        timeout_s=dcfg.request_timeout_s,
        chunk_len=dcfg.chunk_len,
        action_dim=dcfg.action_dim,
    ) as client:
        info = client.info()  # before the arm is connected: a wrong server never moves it
        check_server(info, dcfg)
        if not info.get("adapter_sha256"):
            sys.exit(
                "policy server reports no adapter_sha256: the served weights cannot be bound to "
                "the session (update the server to this branch); nothing was opened"
            )
        # builds the follower (lerobot reads the calibration file; no port is opened) so the
        # fingerprint and the resume check run before anything can move
        arm = So101Driver.from_config(cfg, fk)
        bind_session(cfg, settings, prior, resumed_from, arm, info, dcfg, None, acknowledge=input)
        print(f"policy server OK: {info['policy_id']} (server run {info.get('server_run')})")
        if dcfg.policy_checkpoint is None:
            print("WARNING: server.policy_checkpoint is null: the served adapter is not checked")
        guard = TorqueGuard(arm, torque_wait)
        operator = JudgeOperator(guard, str(settings["task"]), settle_s_from_config(cfg))
        # Ctrl-C a flag outside prompts, SIGHUP / SIGTERM / a lost foreground the closed-stdin
        # path, Ctrl-Z ignored: from before torque on to after the final torque off (prompts.py)
        with operator_signals() as interrupts, OpenCVFrameSource(cam_cfg) as cam, arm:
            loop = DeployLoop(arm, cam, pipe, client, dcfg, interrupts=interrupts)
            return session(
                cfg,
                loop,
                operator,
                info,
                mode="hardware",
                settings=settings,
                prior_trials=prior,
                resumed_from=resumed_from,
                max_steps=args.max_steps,
                camera_calibrated=pipe.calibrated,
                runs_dir=args.runs_dir,
                prior_runs=prior_runs,
                torque_wait=torque_wait,
            )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="in-process fakes (CPU, safe)")
    mode.add_argument("--hardware", action="store_true", help="REAL ARM; owner only")
    parser.add_argument("--i-am-present", action="store_true", help="owner at the arm")
    parser.add_argument("--task", choices=P1.TASKS, default=None)
    parser.add_argument("--cells", choices=P1.CELLS, nargs="+", default=None)
    parser.add_argument(
        "--trials", type=int, default=None, help="per cell; default eval.trials_per_cell (50)"
    )
    parser.add_argument(
        "--slot-seed", type=int, default=None, help="placement plan; default eval.slot_seed"
    )
    parser.add_argument("--label", default="unlabelled", help="checkpoint label, e.g. sft / sft_rl")
    parser.add_argument(
        "--resume", default=None, help="run id (or run dir) of the session to continue"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="default deploy.trial_max_steps"
    )
    parser.add_argument("--fake-clock", action="store_true", help="mock: instant, exact latency")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error(
            "WO-07 runs with --preset real (configs/real.yaml robot / deploy / eval blocks)"
        )
    if args.hardware and args.max_steps is not None:
        parser.error("--max-steps is for mock runs; the real eval uses deploy.trial_max_steps")
    cfg = load_config(args.preset, args.config)
    return run(cfg, args)


if __name__ == "__main__":
    main()
