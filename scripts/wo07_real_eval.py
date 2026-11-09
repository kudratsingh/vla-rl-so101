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

Judging rule ``real-judge-3`` (``judging_rule()``; project review 2026-11-07 section 3.2: a real
judge must be defined before WO-07; real-judge-1 approved by the owner 2026-11-08, real-judge-2
implements the owner's trial-scoring decision of 2026-11-08, real-judge-3 the start rule of PR
#33, owner 2026-11-09): after the trial has ended, the arm
holds still for ``eval.settle_s`` seconds (the settling interval: objects stop rolling or
tipping), then it is judged ONCE (the prompt only appears then), by eye, without touching
anything. Real trials have no success latch: the sim stops at its first success step, the arm
keeps acting until ``trial_max_steps`` or a stop, so a success undone before the end is a failure
(stricter than sim). Doubt counts as a failure.

What real-judge-2 changed against real-judge-1 (which had the operator score faulted trials after
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
session scored under an earlier rule (real-judge-1 or real-judge-2) can be neither resumed nor
pooled with real-judge-3 records.

What real-judge-3 changes against real-judge-2 (owner 2026-11-09, after PR #33; before the
real-eval protocol is frozen, no real session recorded): "started" moved from "the first action
reached the safety filter" to "the bus write of the first action was begun". Deliberate
reclassification: a first step that ends before its write (the pre-write stop check, a stop or a
frame / chunk gone stale while the start marker was saved, a marker that could not be saved or
synced, a deadman trip at its frame check or in the filter, a joint-read serial fault) is now
``not_started`` and retried instead of a started failure; a write that began and failed still
counts as started. The rule text lives in ``src/eval/protocol_real.py`` (real-0.1 hashes it).

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
A first goal blocked after that row (a stop recorded during its I/O, a frame or chunk gone stale
during it, or the row failed to save or to sync: the session then ends ``start_marker_failed``)
was never sent: a ``trial_start_retracted`` row cancels the marker, and the slot is retried with
the same placement.
Each invocation is its own run record, holding every trial of the session so far; record.json is
written when the session is bound and after every trial, so a run killed without exiting
(SIGKILL, power loss) can be resumed itself.
Power loss (owner 2026-11-09): the ``DURABLE_ROWS`` (``session``, ``trial_started``,
``trial_start_retracted``) are synced to stable storage before the call that wrote them returns
(``Run.log_durable``: ``F_FULLFSYNC`` on macOS, ``fsync`` elsewhere), and every record.json write
is atomic and synced (``src/runs.py``); other metrics rows (decisions, trial and capture rows)
are plain appends, each trial / capture row followed by a synced record.json holding it. The
start marker's sync is part of the marker: a failed sync blocks the first goal like a failed
write, and the freshness and final stop checks run after it. ``load_prior`` rebuilds the session
from metrics.jsonl's ``session`` row when record.json is missing or unreadable; a cut
metrics.jsonl line is skipped, and record.json's trials stand without metrics.jsonl. Closing the Terminal (SIGHUP / SIGTERM) is not a
kill: the session takes the closed-stdin path (grace period, torque off, released_unconfirmed,
cause hangup) and writes its record. Ctrl-Z never suspends a hardware session (ignored with a
message); the session must run in the terminal's foreground, and losing it takes the same path
(cause lost_foreground).
``--resume`` refuses a run that a later run of its session continued (a resume of an older
run would fork the session and drop the newer runs' scored trials), and warns (``--hardware``: asks for
'checked') when an earlier run ended with TORQUE OFF UNCONFIRMED, in a trial or at shutdown.

Protocol real-0.1 (``--protocol real-0.1``, the default; ``src/eval/protocol_real.py``; owner
2026-11-09, final2 item 6 and amendments D1-D5). PickPlace only. The frozen schedule (seen = 10 at
each of B2 B5 E2 E5 C4, shifted = 50 over the 20 ring slots, unseen = 25 purple + 25 green cube at
the seen slots, the bowl fixed at (0.195, +0.205) m) runs in ONE seeded order across all three
cells: the same placements and the same order for every policy (SmolVLA and ACT), every record
carrying ``schedule_pos``, ``schedule_sha256`` and the real-0.1 composite hash. ``--schedule dev``
runs the disjoint 30-trial development schedule (checkpoint selection, the 100-demo failure
analysis): ``reportable: false``, never a result. The horizon is the protocol's
(``TRIAL_MAX_STEPS`` 450 = 30 s at 15 Hz), the settle 3 s; ``--hardware`` refuses ``--trials``,
``--cells`` and ``--slot-seed`` (the schedule is frozen), and needs ``--labeler`` (the named person
who scores), ``--family`` / ``--training-run-id`` / ``--checkpoint-dir`` (the policy's stable family
id, its training run and the content hash of the checkpoint files, recorded in the session's
fingerprint). Every trial is captured (``src/robot/recording.py``): both cameras, state and action
at the control rate into one LeRobotDataset v3.0 per session (``<runs>/<session>/capture/dataset``,
episode = trial; ``capture.episode_index`` in the trial record, the trial id in the dataset's
``meta/vla_rl_so101.json``); a not-started attempt is never an episode. The summary reports seen,
shifted, unseen per object and unseen pooled (the headline). ``--report RUN RUN`` prints two
sessions side by side and refuses unless they ran the same protocol hash and schedule.
Baseline cadence (owner 2026-11-09): 15 Hz nominal control, 30 fps cameras, the SAME 450-command
cap for SmolVLA and ACT, synchronous inference (no overlap) and ``max_chunk_age_ms`` 800. Budget
per 150-trial session (single-run MOCK-RIG rates with the capture on, 450-command trials, M3 Pro;
docs/STATUS.md "PR 2: owner follow-up"): POLICY time (the policy commanding) = 450 / the achieved
write rate: SmolVLA 6.27 writes/s -> ~72 s per trial, ~3.0 h per session; ACT 12.86 writes/s ->
~35 s, ~1.5 h. BENCH time (everything else) ~36 s per trial, ~1.5 h per session: measured staging
0.26 s and episode save + checkpoint ~7.5-7.9 s, the 3 s settle, and ESTIMATES (not measurable
on the mock) of home ~5 s, placement / reset ~15 s, score ~5 s. Totals ~4.5 h (SmolVLA) and
~3.0 h (ACT); the SmolVLA session takes two bench sittings through ``--resume``.

Audit against final2 item 5 (where each requirement lives):
* randomized trial order, seeded and recorded, the same for both policies: real-0.1 =
  ``protocol_real.eval_schedule`` (one permutation of all 150) + ``schedule_pos`` /
  ``schedule_sha256`` per record; protocol 0.2 = per-cell seeded slot permutation
  (``real_grid.sample_plan``), cells in protocol order, ``placement_plan_sha256`` per record and
  ``--hardware`` refusing a ``--slot-seed`` other than ``eval.slot_seed`` (``run_hardware``).
* initial object pose per trial from the grid: ``placement`` in every record (``run_trials`` in
  ``src/robot/client.py``); real-0.1 names them ``planned_*`` with ``pose_source``.
* named labeler and success rubric in the record: real-0.1 ``labeler`` and ``rubric`` (id,
  version, judging rule) per record; the judging rule text and hash in every record
  (``protocol_fields``).
* fault after the policy started = failure with the reason: ``client._score`` / ``trial_fault``.
* resume skips started attempts and retries only never-started slots: ``done_indices``,
  ``load_prior`` (a started trial without a record is a ``session_lost`` failure).

Policy backend (``--backend``). ``local``: the SmolVLA / ACT checkpoint in ``--checkpoint-dir`` runs
in process on the Mac (``src/robot/local_policy.py``, MPS with CPU fallback) inside the same
deploy loop; it is loaded only if it hashes to the session's ``policy_identity.checkpoint_sha256``
(``src.policy.lerobot_ckpt.checkpoint_identity``, the trainer's hash and the export's
``checkpoint_id``), it reads the ``CameraRig`` itself (front + wrist, uncropped, through
``convention.policy_observation``), and its chunk / deadline come from the ``local_policy`` block.
This is the only backend of a real-0.1 session on the arm. ``grpc``: the policy server (protocol
0.2, and the mock's default MockPolicy). A resume keeps the session's backend.

--mock      CPU, in-process, safe anywhere: MockPolicy behind gRPC on 127.0.0.1 (or, with
            ``--backend local``, the checkpoint in process), the fake arm and two fake cameras
            from tests/robot_fakes.py, the real safety filter and sim FK. Scripted scores.
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

    uv run scripts/wo07_real_eval.py --preset real --mock --fake-clock --trials 2 --max-steps 16 \\
        --task PickPlace
    uv run scripts/wo07_real_eval.py --preset real --mock --fake-clock --trials 1 --max-steps 8 \\
        --task PickPlace --backend local --labeler mock --family act \\
        --training-run-id <train run id> --checkpoint-dir <pretrained_model dir>
    .venv/bin/python scripts/wo07_real_eval.py --preset real --hardware --i-am-present \\
        --task PickPlace --label smolvla-20k --labeler <name> --family smolvla \\
        --training-run-id <train run id> --checkpoint-dir <pretrained_model dir>
    uv run scripts/wo07_real_eval.py --preset real --report <smolvla run> <act run>
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.eval import protocol as P1
from src.eval import protocol_real as PR
from src.eval import protocol_v02 as P2
from src.eval import real_grid as G
from src.eval.bowl_fit import bowl_box_problem
from src.robot import convention as conv
from src.robot.camera import (
    CameraRig,
    CameraRoleError,
    PolicyImagePipeline,
    confirm_camera_roles,
)
from src.robot.client import (
    DURABLE_ROW_KINDS,
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
    utc_now,
)
from src.robot.driver import So101Driver
from src.robot.prompts import TorqueGuard, TorqueWait, operator_signals, require_terminal
from src.robot.recording import (
    NO_POLICY_WRITE,
    STAGED_NOT_STARTED,
    STAGED_UNREADABLE,
    EvalCapture,
    RigFrameSource,
    check_capture_resumable,
    has_staged,
    recording_required,
)
from src.robot.safety import FkFn, load_sim_fk, make_safety
from src.robot.serve import MockPolicy, PolicyServer, build_server
from src.robot.servo_health import monitor_from_config
from src.robot.units import UnitMap
from src.runs import RUNS_DIR, Run, new_run_id

PROTOCOLS = (PR.PROTOCOL_VERSION, P2.PROTOCOL_VERSION)

JUDGING_RULE_VERSION = PR.JUDGING_RULE_VERSION  # "real-judge-3"
# metrics rows synced to stable storage before log_row returns (module doc, Run.log_durable)
DURABLE_ROWS: tuple[str, ...] = (*DURABLE_ROW_KINDS, "session")
DEFAULT_SETTLE_S = 3.0  # eval.settle_s
# the rule's text lives in src/eval/protocol_real.py (hashed by real-0.1), with #33's start
# wording (real-judge-3); protocol 0.2 sessions record the same version and text hash
JUDGING_GENERAL = PR.JUDGING_GENERAL
JUDGING_RULE: dict[str, str] = PR.JUDGING_RULE

MOCK_LABEL = "MOCK: fake arm, fake camera, mock policy; exercises the eval loop only, NOT a result"
HARDWARE_LABEL = "real arm, operator-scored, single session per checkpoint"

HARDWARE_REFUSAL = """--hardware drives the real SO-101 and refuses to run without --i-am-present.

Only the owner runs it, standing at the arm with a hand on the power switch, after WO-06
(calibration, intrinsics, camera match), the WO-07 pilot, the placement grid printed and its
10 cm scale bar measured. Protocol real-0.1 runs the SmolVLA / ACT checkpoint in process on the
Mac (--backend local); protocol 0.2 needs the policy server reachable through the tunnel."""

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


def protocol_fields(
    settle_s: float = DEFAULT_SETTLE_S,
    protocol: str = P2.PROTOCOL_VERSION,
    schedule: str | None = None,
) -> dict[str, Any]:
    """Per-trial protocol citation. 0.2: its composite hash, 0.1-draft as a reference, real-eval
    files. real-0.1: its composite hash (slots, bowl, objects, schedules, grid, rubric, convention),
    the schedule run and its hash, the same reference and files."""
    rule_text = json.dumps(judging_rule(settle_s), sort_keys=True)
    real_eval: dict[str, Any] = {
        "judging_rule": JUDGING_RULE_VERSION,
        "judging_rule_sha256": hashlib.sha256(rule_text.encode()).hexdigest(),
        "settle_s": float(settle_s),
        "grid_sha256": _sha256(REPO_ROOT / "src" / "eval" / "real_grid.py"),
        "grid_pdf_sha256": _sha256(REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf"),
    }
    if protocol == PR.PROTOCOL_VERSION:
        sched = schedule or "eval"
        return {
            "protocol_version": PR.PROTOCOL_VERSION,
            "protocol_hash": PR.protocol_hash(),
            "protocol_ref": {"version": P1.PROTOCOL_VERSION, "hash": P1.protocol_hash()},
            "real_eval": {
                **real_eval,
                "schedule": sched,
                "schedule_sha256": PR.schedule_sha256(sched),
                "rubric": {"id": PR.RUBRIC_ID, "version": PR.RUBRIC_VERSION},
                "convention": conv.citation(),
            },
        }
    if protocol != P2.PROTOCOL_VERSION:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return {
        "protocol_version": P2.PROTOCOL_VERSION,
        "protocol_hash": P2.protocol_hash(),
        "protocol_ref": {"version": P1.PROTOCOL_VERSION, "hash": P1.protocol_hash()},
        "real_eval": real_eval,
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
    protocol: str = P2.PROTOCOL_VERSION,
    schedule: str | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The effective settings a session is bound to, per component (module doc). ``identity``
    (real-0.1): the policy's family id, training run and checkpoint content hash."""
    r = cfg["robot"]
    sc = arm.safety.config
    deploy = {
        k: v for k, v in dataclasses.asdict(dcfg).items() if k not in _REPORT_ONLY
    }  # control_hz, chunk_len, exec_steps, timeouts, chunk age, trial steps, home, park
    return _canon(
        {
            "protocol": protocol_fields(settle_s, protocol, schedule),
            "policy": {
                "policy_id": info.get("policy_id"),
                "policy_checkpoint": dcfg.policy_checkpoint,
                "adapter_sha256": info.get("adapter_sha256"),  # the weights, not the name
                "adapter_revision": info.get("adapter_revision"),
                **({} if identity is None else {"identity": _identity_key(identity)}),
                **_local_runtime(info),
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


def _local_runtime(info: dict[str, Any]) -> dict[str, Any]:
    """The in-process policy's device and dtype (review F2): a session that ran on MPS never
    pools with or resumes on a CPU fallback. Empty for the gRPC server (unchanged fingerprints)."""
    if info.get("transport") != "in_process":
        return {}
    d = info.get("describe") or {}
    return {"runtime": {"device": d.get("device"), "dtype": d.get("dtype")}}


def _identity_key(identity: dict[str, Any]) -> dict[str, Any]:
    """What binds a session to a policy (the checkpoint's path is not part of it)."""
    return {k: identity.get(k) for k in ("family_id", "training_run_id", "checkpoint_sha256")}


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
    is never re-run (real-judge-3). A ``trial_started`` row with a ``trial_start_retracted`` row
    of the same attempt is dropped: the first goal was blocked after the marker (a stop recorded
    during its I/O, a frame / chunk gone stale during it, or the marker failed), nothing was sent,
    the attempt never started and the slot is retried (``DeployLoop._first_goal_check``); a
    ``capture`` row of a retracted attempt is dropped too (it has no recording).

    Rows are keyed by (cell, trial_idx, start_utc, attempt): ``start_utc`` has one-second
    resolution, so a not-started attempt and its in-place retry can share it (review F4). A
    ``capture`` row is merged only into a STARTED record of the same key (a not-started attempt
    never has an episode). A real-0.1 record without ``trial_id`` (rows written before the
    start marker carried it) gets the one ``trial_id()`` derives (review F1)."""
    rec_path = run_dir / "record.json"
    try:
        results = json.loads(rec_path.read_text()).get("results", {})
    except (OSError, ValueError) as e:  # missing / cut short (power loss): the synced rows
        results = results_from_metrics(run_dir, f"{type(e).__name__}: {e}")
    settings = results.get("session")
    if not settings:
        sys.exit(f"--resume: {rec_path} is not a wo07_real_eval session")
    if prior_run_torque_off_unconfirmed(results):  # carried by every later run of the session
        settings["prior_torque_off_unconfirmed"] = True
    trials: list[dict[str, Any]] = list(results.get("trials") or [])
    seen = {_row_key(t) for t in trials}
    metrics = run_dir / "metrics.jsonl"
    started: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    captures: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    retracted: set[tuple[Any, Any, Any, Any]] = set()
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:  # a line cut short by the kill
                continue
            if row.get("kind") == "capture":  # the episode saved after the trial's record
                captures[_row_key(row)] = row["capture"]
                continue
            if row.get("kind") not in ("trial", "trial_started", "trial_start_retracted"):
                continue
            key = _row_key(row)
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
    for key in retracted:  # a retracted attempt never started: it has no recording
        captures.pop(key, None)
    real = settings.get("protocol_id") == PR.PROTOCOL_VERSION and settings.get("session_id")
    for t in trials:
        if real and t.get("trial_id") is None:
            t["trial_id"] = trial_id(str(settings["session_id"]), t["cell"], t["trial_idx"])
        got = captures.get(_row_key(t))
        if (
            got is not None
            and t.get("policy_started")
            and (t.get("capture") or {}).get("episode_index") is None
        ):
            t["capture"] = got
    return settings, trials


def _last_session_row(run_dir: Path) -> dict[str, Any] | None:
    """The last ``session`` row of a run's metrics.jsonl (cut lines skipped), or None."""
    metrics = run_dir / "metrics.jsonl"
    found: dict[str, Any] | None = None
    try:
        lines = metrics.read_text().splitlines() if metrics.is_file() else []
    except OSError:
        return None
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") == "session" and isinstance(row.get("session"), dict):
            found = row
    return found


def results_from_metrics(run_dir: Path, why: str) -> dict[str, Any]:
    """The session of a run whose record.json is missing or unreadable (``why``), rebuilt from
    the last synced ``session`` row of its metrics.jsonl (``DURABLE_ROWS``); its trials then come
    from the metrics rows alone (``load_prior``). How that run ended is unknown, so a resume
    treats its torque off as unconfirmed (the operator checks the bus). Exits when neither
    survives."""
    found = _last_session_row(run_dir)
    if found is None:
        sys.exit(
            f"--resume: {run_dir / 'record.json'} unreadable ({why}) and no session row in "
            f"{run_dir / 'metrics.jsonl'}"
        )
    print(
        f"--resume: {run_dir / 'record.json'} unreadable ({why}): the session is rebuilt from "
        "metrics.jsonl; that run's shutdown is unknown, so its torque off counts as unconfirmed"
    )
    settings = dict(found["session"])
    settings["prior_torque_off_unconfirmed"] = True
    return {
        "session": settings,
        "trials": [],
        "resumed_from": found.get("resumed_from"),
        "rebuilt_from": f"metrics.jsonl ({why})",
    }


def _row_key(row: Mapping[str, Any]) -> tuple[Any, Any, Any, Any]:
    """A trial / attempt's key in the session's rows (``load_prior``)."""
    return (row["cell"], row["trial_idx"], row["start_utc"], row.get("attempt"))


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
    """{run id: the run it resumed from} for every run whose record holds this session (or,
    record.json lost, whose synced ``session`` row does)."""
    out: dict[str, str | None] = {}
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        try:
            results = json.loads((d / "record.json").read_text()).get("results") or {}
        except (OSError, ValueError):
            results = _last_session_row(d) or {}
        if (results.get("session") or {}).get("session_id") == session_id:
            out[d.name] = results.get("resumed_from")
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
    real = settings.get("protocol_id") == PR.PROTOCOL_VERSION
    if real:
        # binary success per reporting row; unseen per object and pooled (the headline)
        s["real_rows"] = PR.cell_tables(scored)
        s["schedule"] = settings.get("schedule")
        s["reportable"] = settings.get("schedule") == "eval"
        s["protocol_complete"] = bool(s["protocol_complete"] and s["reportable"])
        s["policy_identity"] = settings.get("policy_identity")
    s.update(
        protocol_version=settings["protocol"]["protocol_version"],
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
    if s.get("real_rows"):
        tag = "" if s.get("reportable") else "  (DEVELOPMENT schedule: never reported)"
        print(f"{PR.PROTOCOL_VERSION} rows, binary success (unseen pooled = headline){tag}")
        for name, r in s["real_rows"].items():
            rate = "n/a" if r["success_rate"] is None else f"{100 * r['success_rate']:.1f}%"
            lo, hi = r["wilson_95"]
            print(
                f"  {name}: {r['successes']} / {r['n']} = {rate}  [{100 * lo:.1f}, {100 * hi:.1f}]"
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
    camera_check: dict[str, Any] | None = None,
) -> Path:
    task = str(settings["task"])
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
        # where the policy runs in this process (in-process SmolVLA / ACT: mps | cpu); the gRPC
        # path's policy runs on the server, this process only on the CPU
        device=str(_local_runtime(info).get("runtime", {}).get("device") or "cpu"),
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
            camera_check=camera_check,
            trials=trials,  # the same list: every flush carries the trials so far
        )
        run.flush()  # the session is on disk before the first motion: a killed run resumes

        def log_row(**row: Any) -> None:
            if row.get("kind") in DURABLE_ROWS:  # synced before it returns (module doc)
                run.log_durable(**row)
            else:
                run.log_metrics(**row)
            if row.get("kind") in ("trial", "capture"):
                run.flush()

        # the session in metrics.jsonl too, synced: load_prior rebuilds from it without
        # record.json (power loss), and record.json carries which sync ran
        log_row(kind="session", session=settings, resumed_from=resumed_from, run_id=run.id)
        run.results["durability"] = {
            "synced_rows": list(DURABLE_ROWS),
            "record_json": "atomic: tmp + sync + rename + directory sync, every write",
            "sync": run.sync_method,
        }

        print(f"{judging_rule(settle_s)['general']}\n  {task} success = {JUDGING_RULE[task]}.")
        real = settings.get("protocol_id") == PR.PROTOCOL_VERSION
        capture: EvalCapture | None = None
        try:
            if real and settings.get("capture"):
                capture = open_capture(cfg, settings, run, loop, resumed=resumed_from is not None)
                capture.provenance.setdefault("camera_checks", []).append(
                    {"run": run.id, **(camera_check or {"skipped": f"{mode} rig"})}
                )
                loop.on_write = capture.on_write  # every policy write: both cameras, state, action
                loop.on_settled = capture.on_settled  # the end-state stills, before the score
                run.results["capture"] = {
                    "root": str(capture.root),
                    "dataset": str(capture.dataset_root),
                }
                if resumed_from is not None:  # before any new trial: earlier runs' gaps
                    run.results["capture"]["reconciled"] = reconcile_captures(
                        capture, trials, log_row
                    )
                run.flush()
            if real:
                end = _run_schedule(
                    loop,
                    operator,
                    settings,
                    info,
                    trials,
                    extra,
                    capture,
                    log_row=log_row,
                    max_steps=max_steps,
                    camera_calibrated=camera_calibrated,
                    on_trial_start=on_trial_start,
                )
            else:
                end = _run_cells_02(
                    loop,
                    operator,
                    settings,
                    info,
                    trials,
                    extra,
                    log_row=log_row,
                    max_steps=max_steps,
                    camera_calibrated=camera_calibrated,
                    on_trial_start=on_trial_start,
                )
        except KeyboardInterrupt:
            end = "operator_ctrl_c"
        except Exception as e:  # never a healthy end: hold + support, no park
            end = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            loop.on_write = None
            loop.on_settled = None
            shutdown = shutdown_or_record(run.results, loop, operator, end)
            end = final_session_end(loop.arm, end)  # a failed torque-off wins over any end
            if capture is not None:
                stage_unsaved(capture, trials)  # a trial the session ended in, before its save
                try:
                    capture.close()  # Parquet footers: the dataset is readable and resumable
                    run.results["capture"]["episodes"] = len(capture.provenance["episodes"])
                except Exception as e:  # noqa: BLE001 - the trial records stand
                    run.results["capture"]["close_error"] = f"{type(e).__name__}: {e}"
                run.results["capture"].update(capture_gaps(capture.root, trials))
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
        gaps = (run.results.get("capture") or {}) if capture is not None else {}
        if gaps.get("recoverable_on_resume"):
            print(
                f"  {len(gaps['recoverable_on_resume'])} scored trial(s) have a staged, unsaved "
                f"recording: --resume {run.id} recovers them before any new trial"
            )
        if gaps.get("recording_missing"):
            print(
                f"  RECORDING MISSING for {gaps['recording_missing']}: their scores stand; the "
                "export refuses a complete bundle (src/eval/export_bundle.py)"
            )
    return run.record_path


def _run_cells_02(
    loop: DeployLoop,
    operator: Operator,
    settings: dict[str, Any],
    info: dict[str, Any],
    trials: list[dict[str, Any]],
    extra: dict[str, Any],
    *,
    log_row: Callable[..., None],
    max_steps: int | None,
    camera_calibrated: bool,
    on_trial_start: Callable[[int], None] | None,
) -> str:
    """Protocol 0.2: cells in protocol order, the per-cell seeded slot plan (real_grid)."""
    task, n = str(settings["task"]), int(settings["trials_per_cell"])
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
                "placement_plan_sha256": plan_sha256(task, cell, int(settings["slot_seed"]), n),
            },
            settle_s=float(settings["settle_s"]),
        )
        if why != "completed":
            return why
    return "completed"


def session_items(settings: dict[str, Any]) -> list[dict[str, Any]]:
    """The real-0.1 schedule items this session runs, in schedule order: the whole schedule, or
    (mock subsets) the first ``trials_per_cell`` items of each chosen cell."""
    items = PR.schedule(str(settings["schedule"]))
    n = int(settings["trials_per_cell"])
    out, taken = [], {c: 0 for c in settings["cells"]}
    for it in items:
        c = it["cell"]
        if c in taken and taken[c] < n:
            taken[c] += 1
            out.append(it)
    return out


def trial_id(session_id: str, cell: str, trial_idx: int) -> str:
    return f"{session_id}/{cell}/{int(trial_idx):03d}"


def _run_schedule(
    loop: DeployLoop,
    operator: Operator,
    settings: dict[str, Any],
    info: dict[str, Any],
    trials: list[dict[str, Any]],
    extra: dict[str, Any],
    capture: EvalCapture | None,
    *,
    log_row: Callable[..., None],
    max_steps: int | None,
    camera_calibrated: bool,
    on_trial_start: Callable[[int], None] | None,
) -> str:
    """real-0.1: the frozen schedule in its seeded order, one trial at a time across the cells;
    a started (cell, trial_idx) is never re-run, a never-started one is retried IN ITS PLACE
    (review F3: the same item again before the next, up to ``MAX_START_ATTEMPTS`` per run, then
    the session ends so a resume retries it first; every record carries its ``attempt``). The
    capture of a trial is saved AFTER its scored record is logged and flushed (review F7), then
    a ``capture`` row logs the episode index."""
    task, n = str(settings["task"]), int(settings["trials_per_cell"])
    items = session_items(settings)
    total = len(items)
    sched = str(settings["schedule"])
    rubric = {
        "id": PR.RUBRIC_ID,
        "version": PR.RUBRIC_VERSION,
        "judging_rule": JUDGING_RULE_VERSION,
    }
    base = {
        **extra,
        "schedule": sched,
        "schedule_sha256": PR.schedule_sha256(sched),
        "reportable": sched == "eval",
        "labeler": settings.get("labeler"),
        "rubric": rubric,
        "policy_identity": settings.get("policy_identity"),
    }
    sid = str(settings["session_id"])
    for item in items:
        cell, k = str(item["cell"]), int(item["trial_idx"])
        if k in done_indices(trials, cell):
            continue
        pos = int(item["schedule_pos"])
        tid = trial_id(sid, cell, k)
        obj = PR.object_by_label(str(item["object"]))

        def place(
            _k: int, item: dict[str, Any] = item, pos: int = pos
        ) -> tuple[str, dict[str, Any]]:
            return f"(session trial {pos + 1}/{total}) {PR.placement_prompt(item)}", dict(item)

        def start(_k: int, tid: str = tid, item: dict[str, Any] = item) -> None:
            if capture is not None:
                meta = {
                    key: item[key]
                    for key in ("cell", "trial_idx", "schedule_pos", "object", "object_slot")
                }
                capture.begin(tid, str(item["instruction"]), meta)
            if on_trial_start is not None:
                on_trial_start(_k)

        def hook(rec: dict[str, Any], tid: str = tid) -> None:
            rec["trial_id"] = tid
            if capture is None:
                rec["capture"] = {
                    "episode_index": None,
                    "reason": "capture off (mock --no-capture)",
                }
                return
            rec["capture"] = {"episode_index": None, "reason": CAPTURE_PENDING}
            if rec.get("policy_started"):  # journal it BEFORE the record is logged
                rec["capture"]["stage"] = capture.stage(rec)

        while True:
            n0 = len(trials)
            attempt = 1 + sum(1 for t in trials if t.get("trial_id") == tid)
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
                on_trial_start=start,
                indices=[k],
                placement_fn=place,
                # trial_id in every row, the start marker too: a session_lost record is
                # staged at the session end and recovered by its id (review F1)
                record_extra={**base, "schedule_pos": pos, "attempt": attempt, "trial_id": tid},
                settle_s=float(settings["settle_s"]),
                object_fn=lambda _k, obj=obj: obj,
                record_hook=hook,
            )
            if capture is not None:
                for rec in trials[n0:]:
                    if rec.get("trial_id") == tid:
                        save_capture(capture, rec, log_row)
            if why != "completed":
                return why
            if k in done_indices(trials, cell):
                break
            if attempt >= MAX_START_ATTEMPTS:
                return (
                    f"not_started: schedule item {pos + 1} did not start in {attempt} attempts "
                    "this run; resume to retry it in its place"
                )
    return "completed"


def save_capture(capture: EvalCapture, rec: dict[str, Any], log_row: Callable[..., None]) -> None:
    """After the trial record is logged: save (or discard) its episode, put the capture block in
    the record and log it as a ``capture`` row (merged into the record by ``load_prior``)."""
    try:
        block = capture.finish(rec)
    except Exception as e:  # noqa: BLE001 - never let the capture end the session
        block = {"episode_index": None, "error": f"{type(e).__name__}: {e}"}
    rec["capture"] = block
    log_row(
        kind="capture",
        trial_id=rec.get("trial_id"),
        cell=rec["cell"],
        trial_idx=rec["trial_idx"],
        start_utc=rec["start_utc"],
        attempt=rec.get("attempt"),
        capture=block,
    )


def reconcile_captures(
    capture: EvalCapture, trials: list[dict[str, Any]], log_row: Callable[..., None]
) -> dict[str, Any]:
    """On a resume, BEFORE any new trial (owner 2026-11-09, the capture / resume gap): every
    scored trial of the earlier runs that has no episode gets one, never by re-running it:

    1. its trial id is in the capture provenance: the episode was saved, the run ended before
       its ``capture`` row; attached (``filled_from``);
    2. else its staged journal is whole: the episode is rebuilt and saved under the next free
       index (``recording.EvalCapture.recover_staged``; ``recovered`` says so); a recovery that
       fails with the journal still whole and readable (e.g. a transient save error) is
       ``recovery_failed``: NOT flagged, the journal stays and the next resume retries it
       (review F3);
    3. else (no journal, or a damaged one) it is flagged ``recording_missing`` with the reason
       (sticky: its score stands; the export refuses a complete bundle).

    Only trials that need a recording (``recording.recording_required``: not a trial no policy
    write reached). Each result is logged as a ``capture`` row (merged by ``load_prior``).
    Returns what it did."""
    by_tid: dict[str, int] = {}
    entries = capture.provenance.get("episodes") or {}
    for idx, entry in entries.items():
        if entry.get("trial_id") is not None:
            by_tid[str(entry["trial_id"])] = int(idx)
    saved_keys = (
        "frames",
        "nominal_fps",
        "write_time_span_s",
        "write_rate_hz",
        "end_state",
        "end_state_when",
    )
    claimed = {
        int(t["capture"]["episode_index"])
        for t in trials
        if (t.get("capture") or {}).get("episode_index") is not None
    }
    out: dict[str, list[Any]] = {
        "attached": [],
        "recovered": [],
        "recovery_failed": [],
        "recording_missing": [],
        "discarded": [],
    }
    started_tids = {str(t.get("trial_id")) for t in trials if t.get("policy_started")}
    for tid in sorted({str(t["trial_id"]) for t in trials if t.get("trial_id") is not None}):
        # a slot none of whose attempts started (e.g. a retracted start): no recording belongs
        # to it, and a journal must never reach its later attempt (recording module doc)
        if tid not in started_tids and has_staged(capture.root, tid):
            capture.drop_staged(tid)
            out["discarded"].append({"trial_id": tid, "reason": STAGED_NOT_STARTED})
    for t in trials:
        cap = t.get("capture") or {}
        if not recording_required(t) or cap.get("episode_index") is not None:
            continue
        if cap.get("recording_missing"):
            out["recording_missing"].append(t.get("trial_id"))
            continue
        tid = str(t.get("trial_id"))
        idx = by_tid.get(tid)
        if t.get("trial_id") is not None and idx is not None and idx not in claimed:
            entry = entries[str(idx)]
            block = {
                **{k: entry[k] for k in saved_keys if k in entry},
                "dataset": "dataset",
                "trial_id": tid,
                "episode_index": idx,
                "filled_from": "capture provenance (the run ended before its capture row)",
            }
            capture.drop_staged(tid)
            out["attached"].append(tid)
        else:
            block = capture.recover_staged(tid, expect=t)
            if (
                block.get("episode_index") is None
                and block.get("reason") != STAGED_UNREADABLE
                and has_staged(capture.root, tid)
            ):
                block = {
                    "trial_id": tid,
                    "episode_index": None,
                    "recovery_failed": True,
                    "reason": "recovery failed; the staged journal is kept: a resume retries it "
                    f"({block.get('reason')})",
                    **({"error": block["error"]} if block.get("error") else {}),
                }
                out["recovery_failed"].append(tid)
            elif block.get("episode_index") is None:
                block = {
                    "trial_id": tid,
                    "episode_index": None,
                    "recording_missing": True,
                    "reason": f"recording missing: {block.get('reason')}",
                    **({"error": block["error"]} if block.get("error") else {}),
                }
                out["recording_missing"].append(tid)
            else:
                out["recovered"].append(tid)
        if block.get("episode_index") is not None:
            claimed.add(int(block["episode_index"]))
        t["capture"] = block
        log_row(
            kind="capture",
            trial_id=t.get("trial_id"),
            cell=t["cell"],
            trial_idx=t["trial_idx"],
            start_utc=t["start_utc"],
            attempt=t.get("attempt"),
            capture=block,
        )
    if any(out.values()):
        print(f"capture reconciled on resume: {out}")
    return out


def stage_unsaved(capture: EvalCapture, trials: Sequence[dict[str, Any]]) -> None:
    """At the session end: a trial whose episode is still open (the session ended between its
    start and its save, e.g. a client error after the policy started: its ``session_lost``
    record carries its trial id, review F1) is staged, so a resume recovers it instead of
    flagging it missing; one no policy write reached gets the no-write capture block (it has no
    recording to miss). Never raises."""
    cur = capture.open_episode
    if cur is None:
        return
    for t in reversed(trials):
        if t.get("trial_id") != cur[0]:
            continue
        cap = t.get("capture") or {}
        if not t.get("policy_started") or t.get("outcome") not in SCORED:
            return
        if cap.get("episode_index") is not None:
            return
        st = capture.stage(t)  # never raises
        if st.get("staged") is None and st.get("reason") == NO_POLICY_WRITE:
            t["capture"] = {
                "trial_id": cur[0],
                "episode_index": None,
                "frames": 0,
                "reason": NO_POLICY_WRITE,
                "error": None,
            }
        else:
            t["capture"] = {
                **cap,
                "trial_id": cur[0],
                "episode_index": None,
                "reason": cap.get("reason") or "the session ended before the episode was saved",
                "stage": st,
            }
        return


def capture_gaps(root: Path, trials: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    """Scored trials that need a recording (``recording.recording_required``) without an
    episode: ``recording_missing`` (flagged: no journal or a damaged one; or nothing staged) and
    ``recoverable_on_resume`` (a whole staged journal, never flagged). The same classification
    as ``export_bundle.missing_recordings``."""
    out: dict[str, list[Any]] = {"recoverable_on_resume": [], "recording_missing": []}
    for t in trials:
        cap = t.get("capture") or {}
        if not recording_required(t) or cap.get("episode_index") is not None:
            continue
        tid = t.get("trial_id")
        if not cap.get("recording_missing") and tid is not None and has_staged(root, str(tid)):
            out["recoverable_on_resume"].append(tid)
        else:
            out["recording_missing"].append(tid)
    return out


MAX_START_ATTEMPTS = 3  # per schedule item and run (review F3)
CAPTURE_PENDING = (
    "pending: the episode is saved after this record is logged (stays pending if the run "
    "ended first: not captured)"
)


def open_capture(
    cfg: Config, settings: dict[str, Any], run: Run, loop: DeployLoop, *, resumed: bool
) -> EvalCapture:
    """The session's capture dataset (``<runs>/<session id>/capture``); a resume appends."""
    sid = str(settings["session_id"])
    root = Path(run.dir).parent / sid / "capture"
    cam = loop.cam
    if isinstance(cam, RigFrameSource):  # gRPC path: the policy gets the front view only
        rig = cam.rig
    elif isinstance(cam, CameraRig):  # in-process SmolVLA / ACT: the policy gets every camera
        rig = cam
    else:
        raise TypeError("the evaluation capture needs the two-camera rig (CameraRig)")
    provenance: dict[str, Any] = {
        "source": "scripts/wo07_real_eval.py",
        "purpose": "evaluation",
        "session_id": sid,
        "recorded_at": settings.get("recorded_at"),
        "robot_id": str(cfg["robot"]["id"]),
        "convention": conv.citation(),
        "units": dict(conv.DECLARED_UNITS),
        "protocol": PR.citation(),
        "task": {"name": PR.TASK, "id": PR.TASK_ID, "version": PR.TASK_VERSION},
        "schedule": settings.get("schedule"),
        "schedule_sha256": PR.schedule_sha256(str(settings.get("schedule"))),
        "policy_identity": settings.get("policy_identity"),
        "labeler": settings.get("labeler"),
        "cameras": {n: dict((cfg.get("cameras") or {}).get(n) or {}) for n in conv.CAMERA_NAMES},
        "runs": [run.id],
    }
    return EvalCapture.open(
        root,
        capture_repo_id(sid),
        provenance,
        resume=resumed,
        follower=loop.arm.follower,
        end_state=lambda: rig.read().images,
        clock=loop.clock,
    )


def capture_repo_id(session_id: str) -> str:
    return f"local/so101-real-eval-{session_id}"


def precheck_capture(runs_dir: Path, settings: dict[str, Any]) -> dict[str, Any] | None:
    """A resumed real-0.1 session, BEFORE the arm is connected (review F1): recover its capture
    dataset (a kill mid-save) and make sure it reopens; refuses with a clear message if not."""
    sid = settings.get("session_id")
    if not sid or not settings.get("capture"):
        return None
    root = Path(runs_dir) / str(sid) / "capture"
    try:
        report = check_capture_resumable(root, capture_repo_id(str(sid)), sid)
    except Exception as e:  # noqa: BLE001 - nothing is open yet
        sys.exit(f"--resume refused: the session's capture {root} cannot be reopened ({e})")
    if report and (report["moved"] or report["pruned"]):
        print(f"capture recovered after an interrupted save: {report}")
    return report


def policy_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    """--family / --training-run-id / --checkpoint-dir -> the policy's identity (all or none)."""
    given = [args.family, args.training_run_id, args.checkpoint_dir]
    if all(v is None for v in given):
        return None
    if any(v is None for v in given):
        sys.exit("--family, --training-run-id and --checkpoint-dir go together")
    from src.eval.export_bundle import FAMILIES, ExportRefused, checkpoint_identity
    from src.policy import lerobot_ckpt

    try:
        ident = checkpoint_identity(Path(args.checkpoint_dir))
    except ExportRefused as e:
        sys.exit(f"--checkpoint-dir refused: {e}")
    # review F8: the run id must be the one that wrote the checkpoint (its provenance is inside
    # the hash); without a provenance it is unverified (the real arm refuses that later)
    prov = lerobot_ckpt.read_provenance(Path(args.checkpoint_dir))
    recorded = None if prov is None else prov.get("training_run_id")
    if recorded is not None and recorded != str(args.training_run_id):
        sys.exit(
            f"--training-run-id {args.training_run_id} != {recorded}, the run that saved "
            f"{args.checkpoint_dir} ({lerobot_ckpt.PROVENANCE_FILE}); nothing was opened"
        )
    return {
        "family": args.family,
        "family_id": FAMILIES[args.family],
        "training_run_id": str(args.training_run_id),
        "training_run_verified": recorded is not None,
        "training_run_verified_by": (
            f"checkpoint {lerobot_ckpt.PROVENANCE_FILE}" if recorded is not None else None
        ),
        "checkpoint_sha256": ident["sha256"],
        "checkpoint_files": ident["files"],
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
    }


def check_identity(settings: dict[str, Any], args: argparse.Namespace) -> None:
    """A resume re-hashes the checkpoint (``--checkpoint-dir``, else the recorded dir) and refuses
    other files: the session is bound to the weights it started with."""
    ident = settings.get("policy_identity")
    if not ident:
        return
    from src.eval.export_bundle import ExportRefused, checkpoint_identity

    path = Path(args.checkpoint_dir or ident["checkpoint_dir"])
    try:
        now = checkpoint_identity(path)["sha256"]
    except ExportRefused as e:
        sys.exit(f"--resume refused: {e}")
    if now != ident["checkpoint_sha256"]:
        sys.exit(
            f"--resume refused: the checkpoint files changed ({path} hashes to {now[:12]}, the "
            f"session ran {str(ident['checkpoint_sha256'])[:12]})"
        )
    # review F4: the path just verified is the one the local backend loads (a moved dir)
    ident["checkpoint_dir"] = str(path.resolve())


def new_settings(args: argparse.Namespace, cfg: Config) -> dict[str, Any]:
    ecfg = cfg.get("eval") or {}
    n = int(
        args.trials
        if args.trials is not None
        else ecfg.get("trials_per_cell", P1.REAL_TRIALS_PER_CELL)
    )
    protocol = str(args.protocol)
    real = protocol == PR.PROTOCOL_VERSION
    schedule = str(args.schedule) if real else None
    if real:
        if args.task != PR.TASK:
            sys.exit(
                f"protocol {protocol} defines {PR.TASK} only (Push and Stack wait for the baseline)"
            )
        full = len(PR.schedule(str(schedule))) // len(PR.CELLS)  # equal per cell: 50 eval, 10 dev
        n = full if args.trials is None else int(args.trials)
        if not 1 <= n <= full:
            sys.exit(f"--trials must be in [1, {full}] for the {schedule} schedule")
    elif args.schedule != "eval":
        sys.exit("--schedule is a real-0.1 option")
    return {
        "session_id": None,  # the first run's id
        "protocol_id": protocol,
        "schedule": schedule,
        "recorded_at": utc_now(),
        "task": args.task,
        "cells": list(args.cells or ecfg.get("cells") or P1.CELLS),
        "trials_per_cell": n,
        "protocol_trials_per_cell": P1.REAL_TRIALS_PER_CELL,
        "slot_seed": int(
            args.slot_seed if args.slot_seed is not None else ecfg.get("slot_seed", 0)
        ),
        "label": args.label,
        "labeler": args.labeler,
        "policy_identity": policy_identity(args) if real else None,
        "capture": bool(real and not args.no_capture),
        "backend": resolve_backend(args, real),
        "policy_id": None,  # from the server's Info
        "settle_s": settle_s_from_config(cfg),
        "protocol": protocol_fields(settle_s_from_config(cfg), protocol, schedule),
        "fingerprint": None,  # bind_session, once the rig and the server are known
    }


def check_resume(
    settings: dict[str, Any], info: dict[str, Any], values: dict[str, Any] | None = None
) -> None:
    """Refuse to resume when the protocol, the policy or (with ``values``) any fingerprint
    component differs from the session's."""
    now = protocol_fields(
        float(settings.get("settle_s", DEFAULT_SETTLE_S)),
        str(settings.get("protocol_id") or P2.PROTOCOL_VERSION),
        settings.get("schedule"),
    )
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
        cfg,
        arm,
        info,
        dcfg,
        max_steps=max_steps,
        settle_s=settle_s_from_config(cfg),
        protocol=str(settings.get("protocol_id") or P2.PROTOCOL_VERSION),
        schedule=settings.get("schedule"),
        identity=settings.get("policy_identity"),
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


def resolve_backend(args: argparse.Namespace, real: bool) -> str:
    """Where the policy runs. ``local``: SmolVLA / ACT in process on the Mac through
    ``src/robot/local_policy.py`` (the checkpoint in ``--checkpoint-dir``, both cameras, the
    convention's preprocessing); the only backend of a real-0.1 hardware session (the gRPC server
    serves OpenVLA-OFT). ``grpc``: the policy server (protocol 0.2; the mock's MockPolicy).
    A hardware session's backend and identity are refused later, after the presence gate, by
    ``real_hardware_problems`` (nothing is opened before)."""
    backend = args.backend or ("local" if real and args.hardware else "grpc")
    if backend == "local" and not real:
        sys.exit("--backend local runs SmolVLA / ACT under protocol real-0.1 only")
    if backend == "local" and args.checkpoint_dir is None and not args.hardware:
        sys.exit("--backend local needs --family, --training-run-id and --checkpoint-dir")
    return backend


@contextmanager
def local_backend(
    cfg: Config,
    settings: dict[str, Any],
    clock: Callable[[], float] = time.perf_counter,
    *,
    hardware: bool = False,
) -> Iterator[tuple[Any, dict[str, Any], DeployConfig]]:
    """The in-process policy bound to the session's ``policy_identity``: the checkpoint is loaded
    only if it hashes to the recorded ``checkpoint_sha256`` (one hash definition:
    ``src.policy.lerobot_ckpt.checkpoint_identity``), its provenance fits the live convention and
    ``robot.units`` (and, for SmolVLA, its pinned VLM snapshot), then checked against the chunk and
    the deploy settings (``check_local``). ``hardware``: the provenance is required and the MPS ->
    CPU fallback refused (reviews conv-1, F2). Runs before the arm is connected."""
    from src.robot import local_policy as LP

    ident = settings.get("policy_identity") or {}
    want = ident.get("checkpoint_sha256")
    spec = LP.LocalPolicySpec.from_config(
        cfg, base=ident.get("checkpoint_dir"), family=ident.get("family")
    )
    spec = dataclasses.replace(spec, require_provenance=hardware, strict_device=hardware)
    if spec.checkpoint_sha256 is not None and spec.checkpoint_sha256 != want:
        sys.exit(
            f"local_policy.checkpoint_sha256 {spec.checkpoint_sha256[:12]} != the session's "
            f"checkpoint {str(want)[:12]}; nothing was opened"
        )
    spec = dataclasses.replace(spec, checkpoint_sha256=want)
    try:
        policy = LP.load_local_policy(
            spec, clock=clock, units=dataclasses.asdict(UnitMap.from_config(cfg))
        )
    except (LP.CheckpointMismatch, FileNotFoundError) as e:
        sys.exit(f"{e}; nothing was opened")
    dcfg = LP.local_deploy_config(cfg, policy)
    with LP.LocalPolicyClient(policy, timeout_s=dcfg.request_timeout_s, clock=clock) as client:
        info = client.info()
        LP.check_local(info, dcfg, spec)
        yield client, info, dcfg


def _mock_fakes(
    cfg: Config, args: argparse.Namespace, fk: FkFn | None
) -> tuple[Callable[[], float], Callable[[float], None], Any, Any]:
    """(clock, sleep, fake arm factory, fake two-camera rig factory) of a mock session."""
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from robot_fakes import FakeCam, FakeClock, FakeFollower

    clock: Callable[[], float]
    sleep: Callable[[float], None]
    if args.fake_clock:
        fake = FakeClock()
        clock, sleep = fake, fake.sleep
    else:
        clock, sleep = time.perf_counter, time.sleep

    def fake_arm() -> So101Driver:
        return So101Driver(
            FakeFollower(),
            UnitMap.from_config(cfg),
            make_safety(cfg, fk or load_sim_fk(), clock=clock),
            clock=clock,
            monitor=monitor_from_config(cfg),
            sleep=sleep,
        )

    def fake_rig() -> CameraRig:
        wrist = FakeCam(clock)
        wrist.image = np.ascontiguousarray(255 - wrist.image)  # a swap would show
        return CameraRig({"front": FakeCam(clock), "wrist": wrist})

    return clock, sleep, fake_arm, fake_rig


@contextmanager
def local_mock_rig(
    cfg: Config, args: argparse.Namespace, fk: FkFn | None, settings: dict[str, Any]
) -> Iterator[tuple[DeployLoop, dict[str, Any], Callable[[int], None], bool]]:
    """``--mock --backend local``: the session's checkpoint in process on the fake arm, reading
    the fake two-camera rig itself (both cameras to the policy and to the capture)."""
    from src.robot.local_policy import local_pipe

    clock, sleep, fake_arm, fake_rig = _mock_fakes(cfg, args, fk)
    with local_backend(cfg, settings, clock) as (client, info, dcfg), fake_arm() as arm:
        loop = DeployLoop(arm, fake_rig(), local_pipe, client, dcfg, clock=clock, sleep=sleep)
        yield loop, info, lambda _k: None, False  # raw frames: no undistortion, no crop


@contextmanager
def mock_rig(
    cfg: Config, args: argparse.Namespace, fk: FkFn | None
) -> Iterator[tuple[DeployLoop, dict[str, Any], Callable[[int], None], bool]]:
    spec = importlib.util.spec_from_file_location(
        "wo07_pilot", REPO_ROOT / "scripts" / "wo07_pilot.py"
    )
    assert spec is not None and spec.loader is not None
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    clock, sleep, fake_arm, fake_rig = _mock_fakes(cfg, args, fk)
    from robot_fakes import FakeCam

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
        arm = fake_arm()
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
            cam: Any = FakeCam(clock)
            if getattr(args, "protocol", P2.PROTOCOL_VERSION) == PR.PROTOCOL_VERSION:
                cam = RigFrameSource(fake_rig())
            loop = DeployLoop(arm, cam, pipe, client, dcfg, clock=clock, sleep=sleep)
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
        check_identity(settings, args)
        settings.setdefault("backend", "grpc")  # sessions recorded before --backend
        if settings.get("protocol_id") == PR.PROTOCOL_VERSION:  # before anything is opened
            precheck_capture(args.runs_dir, settings)
        if args.backend not in (None, settings["backend"]):
            sys.exit(f"--resume refused: the session ran --backend {settings['backend']}")
    else:
        if args.task is None:
            sys.exit("--task is required for a new session")
        settings, prior, resumed_from, prior_runs = new_settings(args, cfg), [], None, []
    if args.hardware:
        return run_hardware(cfg, args, settings, prior, resumed_from, prior_runs)
    torque_wait = TorqueWait.from_config(cfg, "deploy", allow_none=True)  # validated; mock: unused
    rig = (
        local_mock_rig(cfg, args, fk, settings)
        if settings["backend"] == "local"
        else mock_rig(cfg, args, fk)
    )
    max_steps: int | None = args.max_steps
    if max_steps is None and settings.get("protocol_id") == PR.PROTOCOL_VERSION:
        max_steps = PR.TRIAL_MAX_STEPS  # the protocol's shared 450-command cap, as on the arm
    with rig as (loop, info, on_start, calibrated):
        bind_session(cfg, settings, prior, resumed_from, loop.arm, info, loop.dcfg, max_steps)
        return session(
            cfg,
            loop,
            operator or ScriptedOperator(),
            info,
            mode="mock",
            settings=settings,
            prior_trials=prior,
            resumed_from=resumed_from,
            max_steps=max_steps,
            camera_calibrated=calibrated,
            runs_dir=args.runs_dir,
            on_trial_start=on_start,
            prior_runs=prior_runs,
            torque_wait=torque_wait,
        )


def real_hardware_problems(cfg: Config, settings: dict[str, Any]) -> list[str]:
    """Why a real-0.1 hardware session may not start (empty = it may)."""
    out = [f"convention: {p}" for p in conv.check_config(cfg)]  # both cameras, 15 Hz, gripper
    if settings.get("backend") != "local":
        out.append("--backend local: a real-0.1 session runs SmolVLA / ACT in process on the Mac")
    if settle_s_from_config(cfg) != PR.SETTLE_S:
        out.append(f"eval.settle_s {settle_s_from_config(cfg)} != the protocol's {PR.SETTLE_S}")
    if not settings.get("labeler"):
        out.append("--labeler is required: the named person who scores every trial")
    if not settings.get("policy_identity"):
        out.append(
            "--family, --training-run-id and --checkpoint-dir are required (policy identity)"
        )
    elif not settings["policy_identity"].get("training_run_verified"):
        out.append(
            "the checkpoint has no trainer provenance: --training-run-id cannot be verified "
            "(a checkpoint saved by scripts/wo07_train.py carries it)"
        )
    full = len(PR.schedule(str(settings.get("schedule")))) // len(PR.CELLS)
    if int(settings["trials_per_cell"]) != full or set(settings["cells"]) != set(PR.CELLS):
        out.append("--trials / --cells: a hardware session runs the whole frozen schedule")
    if not settings.get("capture"):
        out.append("the evaluation capture is required on the arm (final2 hardware floor)")
    bowl = bowl_box_problem(cfg)  # the bowl is pending (owner 2026-11-09): no session at it
    if bowl is not None:
        out.append(bowl)
    frozen = PR.freeze_problem()  # PR.FREEZE_STATUS: pending until the owner confirms the bowl
    if frozen is not None:
        out.append(frozen)
    return out


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
    real = settings.get("protocol_id") == PR.PROTOCOL_VERSION
    max_steps: int | None = PR.TRIAL_MAX_STEPS if real else None
    if real:
        problems = real_hardware_problems(cfg, settings)
        if problems:
            sys.exit("refusing a real-0.1 session; nothing was opened:\n  " + "\n  ".join(problems))
    cfg_seed = int((cfg.get("eval") or {}).get("slot_seed", 0))
    if not real and int(settings["slot_seed"]) != cfg_seed:
        sys.exit(
            f"slot seed {settings['slot_seed']} != eval.slot_seed {cfg_seed}: the SFT and SFT+RL "
            "sessions must use the same placements (owner 2026-11-08); drop --slot-seed"
        )
    from src.robot.camera import OpenCVFrameSource

    fk = load_sim_fk()
    with _hardware_backend(cfg, settings, hardware=True) as (client, info, dcfg, pipe, calibrated):
        # builds the follower (lerobot reads the calibration file; no port is opened) so the
        # fingerprint and the resume check run before anything can move
        arm = So101Driver.from_config(cfg, fk)
        bind_session(
            cfg, settings, prior, resumed_from, arm, info, dcfg, max_steps, acknowledge=input
        )
        print(f"policy OK: {info['policy_id']} (run {info.get('server_run')})")
        if settings["backend"] == "grpc" and dcfg.policy_checkpoint is None:
            print("WARNING: server.policy_checkpoint is null: the served adapter is not checked")
        guard = TorqueGuard(arm, torque_wait)
        operator = JudgeOperator(guard, str(settings["task"]), settle_s_from_config(cfg))
        # Ctrl-C a flag outside prompts, SIGHUP / SIGTERM / a lost foreground the closed-stdin
        # path, Ctrl-Z ignored: from before torque on to after the final torque off (prompts.py)
        cams: Any = (
            CameraRig.from_config(cfg, conv.CAMERA_NAMES)
            if real
            else OpenCVFrameSource(cfg["cameras"]["front"])
        )
        with operator_signals() as interrupts, cams as cam:
            camera_check = None
            if real:  # which index is which camera, before torque on (review conv-4)
                try:
                    camera_check = confirm_camera_roles(
                        cam,
                        args.runs_dir / "camera_checks" / f"{utc_now().replace(':', '')}.png",
                        input,
                        indices={n: cfg["cameras"][n].get("index") for n in conv.CAMERA_NAMES},
                        show=_show_file,
                    )
                except CameraRoleError as e:
                    sys.exit(str(e))
            with arm:
                # local: the policy reads the rig itself (every camera); gRPC real-0.1 (mock
                # only): the front view, the rig for the capture; protocol 0.2: the front camera
                source: Any = (
                    cam if settings["backend"] == "local" else RigFrameSource(cam) if real else cam
                )
                loop = DeployLoop(arm, source, pipe, client, dcfg, interrupts=interrupts)
                return session(
                    cfg,
                    loop,
                    operator,
                    info,
                    mode="hardware",
                    settings=settings,
                    prior_trials=prior,
                    resumed_from=resumed_from,
                    max_steps=max_steps,
                    camera_calibrated=calibrated,
                    runs_dir=args.runs_dir,
                    prior_runs=prior_runs,
                    torque_wait=torque_wait,
                    camera_check=camera_check,
                )


def _show_file(path: Path) -> None:
    import subprocess

    try:
        subprocess.run(["open", str(path)], check=False)
    except OSError:
        pass


@contextmanager
def _hardware_backend(
    cfg: Config, settings: dict[str, Any], *, hardware: bool = True
) -> Iterator[tuple[Any, dict[str, Any], DeployConfig, Any, bool]]:
    """(client, info, deploy config, image pipe, camera calibrated) of the session's backend,
    checked before the arm is connected: a wrong policy never moves it."""
    if settings["backend"] == "local":
        from src.robot.local_policy import local_pipe

        with local_backend(cfg, settings, hardware=hardware) as (client, info, dcfg):
            yield client, info, dcfg, local_pipe, False  # raw frames: no undistortion, no crop
        return
    dcfg = DeployConfig.from_config(cfg)
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
        yield client, info, dcfg, pipe, pipe.calibrated


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="in-process fakes (CPU, safe)")
    mode.add_argument("--hardware", action="store_true", help="REAL ARM; owner only")
    mode.add_argument(
        "--report", nargs=2, metavar="RUN", default=None, help="two sessions side by side"
    )
    parser.add_argument("--protocol", choices=PROTOCOLS, default=PR.PROTOCOL_VERSION)
    parser.add_argument(
        "--schedule", choices=sorted(PR.SCHEDULES), default="eval", help="real-0.1: eval | dev"
    )
    parser.add_argument("--labeler", default=None, help="the named person who scores (real-0.1)")
    parser.add_argument("--family", choices=sorted(("smolvla", "act")), default=None)
    parser.add_argument("--training-run-id", default=None, help="the checkpoint's training run")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None, help="the served checkpoint's files"
    )
    parser.add_argument(
        "--no-capture", action="store_true", help="mock only: skip the evaluation capture"
    )
    parser.add_argument(
        "--backend",
        choices=("grpc", "local"),
        default=None,
        help="local: SmolVLA / ACT in process from --checkpoint-dir (real-0.1; the hardware "
        "default); grpc: the policy server (0.2; the mock's default, MockPolicy)",
    )
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
        "--max-steps",
        type=int,
        default=None,
        help=f"mock only. Default: real-0.1's shared cap of {PR.TRIAL_MAX_STEPS} policy commands "
        "per trial (both policies); protocol 0.2 deploy.trial_max_steps",
    )
    parser.add_argument("--fake-clock", action="store_true", help="mock: instant, exact latency")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error(
            "WO-07 runs with --preset real (configs/real.yaml robot / deploy / eval blocks)"
        )
    if args.hardware and args.max_steps is not None:
        parser.error(
            f"--max-steps is for mock runs; the real eval's cap is fixed: real-0.1 "
            f"{PR.TRIAL_MAX_STEPS} policy commands per trial (protocol 0.2: "
            "deploy.trial_max_steps)"
        )
    if args.hardware and args.no_capture:
        parser.error("--no-capture is for mock runs: every trial on the arm is captured")
    if args.hardware and args.protocol == PR.PROTOCOL_VERSION and args.slot_seed is not None:
        parser.error("--slot-seed is a protocol 0.2 option; real-0.1's schedule is frozen")
    if args.report:
        report([_run_dir(args.runs_dir, r) for r in args.report])
        return args.runs_dir
    cfg = load_config(args.preset, args.config)
    return run(cfg, args)


def _run_dir(runs_dir: Path, r: str) -> Path:
    return Path(r) if Path(r).is_dir() else runs_dir / r


def report(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Two real-0.1 sessions (e.g. SmolVLA and ACT) side by side. Refuses unless both ran the
    same protocol hash and the same evaluation schedule, under different policies."""
    rows = []
    for d in run_dirs:
        results = json.loads((d / "record.json").read_text()).get("results") or {}
        st, summ = results.get("session") or {}, results.get("summary") or {}
        if st.get("protocol_id") != PR.PROTOCOL_VERSION:
            sys.exit(f"--report: {d.name} is not a {PR.PROTOCOL_VERSION} session")
        rows.append((d.name, st, summ))
    (_, a, sa), (_, b, sb) = rows
    pa, pb = a["protocol"], b["protocol"]
    if pa["protocol_hash"] != pb["protocol_hash"] or pa["real_eval"].get("schedule_sha256") != pb[
        "real_eval"
    ].get("schedule_sha256"):
        sys.exit("--report refused: the sessions ran different protocol hashes or schedules")
    if a.get("schedule") != "eval" or b.get("schedule") != "eval":
        sys.exit("--report refused: development sessions are never reported")
    ida, idb = a.get("policy_identity") or {}, b.get("policy_identity") or {}
    if ida and idb and ida.get("checkpoint_sha256") == idb.get("checkpoint_sha256"):
        sys.exit("--report refused: both sessions ran the same checkpoint")
    out = {
        "protocol_version": PR.PROTOCOL_VERSION,
        "protocol_hash": pa["protocol_hash"],
        "rows": {},
    }
    names = [
        f"{(st.get('policy_identity') or {}).get('family') or st.get('label')}" for _, st, _ in rows
    ]
    print(
        f"protocol {PR.PROTOCOL_VERSION} {pa['protocol_hash'][:12]}; binary success, single session each"
    )
    print(f"| row | {names[0]} | {names[1]} |")
    print("|---|---|---|")
    for row in sa.get("real_rows") or {}:
        cells = []
        for summ in (sa, sb):
            r = (summ.get("real_rows") or {}).get(row) or {}
            rate = "n/a" if r.get("success_rate") is None else f"{100 * r['success_rate']:.1f}%"
            cells.append(f"{r.get('successes', 0)}/{r.get('n', 0)} = {rate}")
        out["rows"][row] = cells
        print(f"| {row} | {cells[0]} | {cells[1]} |")
    complete = [bool(summ.get("protocol_complete")) for summ in (sa, sb)]
    if not all(complete):
        print("  NOT a protocol table: a session is incomplete")
    out["protocol_complete"] = all(complete)
    return out


if __name__ == "__main__":
    main()
