"""Export inputs for the evaluator project vishwaroop (its ``docs/SO101_EXPORT.md``, WO-023).

The owner seals an export with vishwaroop's own tool (``python -m vishwaroop.data.so101_import
seal``); this repo only produces the inputs, per recording session, next to the LeRobot dataset:

* the dataset itself, LeRobot ``codebase_version`` v3.0 (what lerobot 0.4.4 writes);
* ``declaration.json``: every field of SO101_EXPORT.md section 2 (``declaration``);
* ``calibration.json``: the follower's LeRobot calibration file as loaded for the session (the
  recorder's copy in the dataset's ``meta/calibration_follower.json``, hash-checked);
* ``rubric.json``: the versioned success rubric (``protocol_real.rubric()``);
* evaluation sessions: ``trials.json`` (every trial record and not-started attempt of the session,
  with their actual provenance: ``outcome_source`` operator / fault / unjudged, the labeler, the
  episode index of each captured trial); ``labels.json`` ONLY from a blinded clip review
  (``scripts/wo07_blind_review.py``): then declared ``blinded_review``; otherwise the ``labels``
  declaration is omitted (``null``), never misdeclared;
* trained-policy sessions: every checkpoint file (``lerobot_ckpt.checkpoint_files``: the weights
  ``model.safetensors`` and the config ``config.json`` declared, the rest alongside) with
  ``checkpoint_files.json``; the checkpoint id is the content hash over all of them
  (``checkpoint_identity``, the trainer's and the deploy path's hash), checked against
  the hash the evaluation session recorded at its start; ``family_id`` is stable per training
  family (``FAMILIES``: SmolVLA and ACT are two) and ``training_run_id`` names the training run.
* demonstration sessions: ``placements.json`` (each episode's planned placement from the frozen
  demo schedule); a rehearsal recording is refused.
* development sessions (``--include-dev``; never reported) are declared with protocol_version
  ``real-0.1-dev``, so the sealed declaration itself says they are not the evaluation; a mock
  session (``--allow-mock``: fake arm, mock policy, a format test) gets ``-mock`` right after the
  version (``real-0.1-mock``, ``real-0.1-mock-dev``, ``real-0.1-mock-incomplete``), so the sealed declaration says mock too, not only
  ``bundle.json`` / ``trials.json`` (``mock_label``; review F2 of the PR #32 hold round).
* timing: LeRobot's ``timestamp`` (what the importer reads) is nominal; every frame's real write
  and camera times are the ``timing_s`` feature (convention 1.1), each trial's achieved write rate
  is in ``trials.json`` and ``bundle.json`` warns when it is more than 10% off the nominal fps.

``object_pose`` is omitted (``null``): only PLANNED initial poses exist (grid slot centres and
yaws), and vishwaroop imports ``object_pose`` as a per-frame stream; a per-frame feature filled with
the planned start pose would claim a tracked pose. The planned poses travel in ``placements.json``
/ ``trials.json`` instead.

Missing recordings (owner 2026-11-09): every scored trial (its policy started) needs its episode.
A trial whose recording is missing (``recording_missing``: a kill between its scored record and
its save that left nothing to recover, ``src/robot/recording.py``) or still only staged (a resume
recovers it: resume first) makes the export INCOMPLETE. An episode saved but not confirmed
durable (its save's sync failed: ``pending_episodes``) refuses every export, incomplete ones too,
naming the trial, until a resume verifies and syncs it. vishwaroop's declaration has no field for
an incomplete export (``docs/SO101_EXPORT.md`` section 2; its importer, ``parse_declaration`` /
``check_export``, only checks that the declared sessions equal the dataset's episodes), so a
complete evaluation bundle is REFUSED. ``allow_missing_recordings`` builds one anyway, declared
``protocol_version`` ``real-0.1-incomplete`` (the same in-spec mechanism as ``-dev``: the sealed
declaration itself says it is not the full evaluation), with ``status: INCOMPLETE`` and the
missing trials in ``bundle.json`` and ``trials.json``. Episodes are never dropped or re-mapped:
each one keeps the trial id the capture provenance gave it. A started trial that no policy write
reached (the first write faulted: frames 0, no capture error) cannot have a recording: it is
listed under ``uncaptured_started_trials`` and does not make the export incomplete
(``recording.recording_required``, review F2).

Nothing is guessed: a missing or unknown value refuses the export (``ExportRefused``).
``validate_bundle`` mirrors the importer's checks of the declaration, the LeRobot tables and the
label file (``vishwaroop/src/vishwaroop/data/so101_import.py``: ``parse_declaration``,
``read_dataset``, ``episode_frames``, ``check_export``'s session and file checks,
``_label_outcome``), plus this repo's stricter ones (the frozen units, ISO 8601 times). Session ids
are the recording session's own id (the demo recorder's, or the evaluation session's first run),
so a session is never split.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.eval import protocol_real as PR
from src.robot import convention as conv

FORMAT = "lerobot-v3.0"
CODEBASE_VERSION = "v3.0"
DATASET_ROOT = "dataset"
UNITS_VOCAB: tuple[str, ...] = ("deg", "rad", "normalized_m100_100", "normalized_0_100")
PURPOSES: tuple[str, ...] = ("demonstration", "evaluation")
LABEL_METHODS: tuple[str, ...] = ("physical_sensor_and_review", "blinded_review")
MAX_TIMESTAMP_SLIP_FRAMES = 0.5
FAMILIES: dict[str, str] = {"smolvla": "so101-real-smolvla", "act": "so101-real-act"}
WRITE_RATE_TOLERANCE = 0.10  # bundle.json warns when an episode's write rate is further off

CALIBRATION_FILE = "calibration.json"
RUBRIC_FILE = "rubric.json"
LABELS_FILE = "labels.json"
TRIALS_FILE = "trials.json"
PLACEMENTS_FILE = "placements.json"
CHECKPOINT_INDEX = "checkpoint_files.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
DECLARATION_FILE = "declaration.json"
BUNDLE_INDEX = "bundle.json"  # local index for the owner (not sealed)
RESERVED = {
    CALIBRATION_FILE,
    RUBRIC_FILE,
    LABELS_FILE,
    TRIALS_FILE,
    PLACEMENTS_FILE,
    CHECKPOINT_INDEX,
    DECLARATION_FILE,
    BUNDLE_INDEX,
}


class ExportRefused(ValueError):
    """A missing or unknown value: the export is refused, never guessed."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_identity(ckpt: Path) -> dict[str, Any]:
    """Content identity of a checkpoint dir: sha256 over ``<name>:<sha256>`` (sorted by name) of
    its checkpoint files, ``src.policy.lerobot_ckpt.checkpoint_files`` (config, weights,
    train_config, the processor pipelines and their stats): the same hash as the trainer's run
    record and the deploy path's ``checkpoint_sha256``. Exactly those files are bundled. Needs
    ``model.safetensors`` and ``config.json``."""
    from src.policy import lerobot_ckpt

    ckpt = Path(ckpt)
    if not ckpt.is_dir():
        raise ExportRefused(f"checkpoint dir {ckpt} does not exist")
    try:
        files = lerobot_ckpt.checkpoint_files(ckpt)
    except FileNotFoundError as e:
        raise ExportRefused(str(e)) from None
    names = [p.name for p in files]
    for need in (WEIGHTS_FILE, CONFIG_FILE):
        if need not in names:  # checkpoint_files refuses this already; kept explicit
            raise ExportRefused(f"checkpoint dir {ckpt} has no {need}")
    clash = sorted(set(names) & RESERVED - {CONFIG_FILE})
    if clash:
        raise ExportRefused(f"checkpoint files {clash} collide with bundle file names")
    shas = {p.name: _sha256(p) for p in files}
    h = hashlib.sha256()
    for name, sha in sorted(shas.items()):
        h.update(f"{name}:{sha}\n".encode())
    ident = lerobot_ckpt.checkpoint_identity(ckpt)["checkpoint_sha256"]
    if h.hexdigest() != ident:  # one definition: never let the two drift
        raise ExportRefused(f"checkpoint hash {h.hexdigest()} != lerobot_ckpt's {ident}")
    return {"sha256": h.hexdigest(), "files": shas}


# ------------------------------------------------------------------------------------------------
# Validation: mirrors vishwaroop's importer (module doc)
# ------------------------------------------------------------------------------------------------
def _str(block: Mapping[str, Any], key: str, where: str) -> str:
    v = block.get(key)
    if not isinstance(v, str) or not v:
        raise ExportRefused(f"{where}.{key} must be a non-empty string")
    return v


def _rel(value: str, where: str) -> str:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ExportRefused(f"{where} must be a relative path inside the export: {value!r}")
    return value


def check_declaration(d: Mapping[str, Any]) -> dict[str, Any]:
    """The importer's ``parse_declaration`` (same refusals) plus the frozen units and ISO times.
    Returns the parsed parts: files named, sessions {int: str}, policy."""
    w = "declaration"
    if not isinstance(d, Mapping):
        raise ExportRefused("the declaration must be a JSON object")
    known = {
        "format", "dataset_root", "units", "calibration", "rubric", "task_id", "task_version",
        "protocol_version", "recorded_at", "robot_id", "purpose", "sessions", "policy", "labels",
        "object_pose",
    }  # fmt: skip
    unknown = sorted(set(d) - known)
    if unknown:
        raise ExportRefused(f"{w} has unknown fields {unknown}")
    if d.get("format") != FORMAT:
        raise ExportRefused(f"{w}.format must be {FORMAT!r}")
    units = d.get("units")
    if not isinstance(units, Mapping) or not units:
        raise ExportRefused(f"{w}.units must declare a unit for every joint")
    for name, unit in units.items():
        if unit not in UNITS_VOCAB:
            raise ExportRefused(
                f"{w}.units[{name!r}] is {unit!r}, not one of {sorted(UNITS_VOCAB)}"
            )
    if dict(units) != conv.DECLARED_UNITS:
        raise ExportRefused(
            f"{w}.units {dict(units)} != what was recorded ({conv.CONVENTION_ID}: "
            f"{conv.DECLARED_UNITS})"
        )
    purpose = _str(d, "purpose", w)
    if purpose not in PURPOSES:
        raise ExportRefused(f"{w}.purpose must be one of {PURPOSES}")
    raw = d.get("sessions")
    if not isinstance(raw, Mapping) or not raw:
        raise ExportRefused(f"{w}.sessions must map every episode index to a session")
    sessions: dict[int, str] = {}
    for k, v in raw.items():
        if not str(k).isdigit() or not isinstance(v, str) or not v:
            raise ExportRefused(f"{w}.sessions has a bad entry {k!r}: {v!r}")
        sessions[int(k)] = v
    root = _rel(_str(d, "dataset_root", w), f"{w}.dataset_root")
    if root != DATASET_ROOT:
        raise ExportRefused(
            f"{w}.dataset_root must be {DATASET_ROOT!r} (a sealed export keeps it there)"
        )
    files = [
        _rel(_str(d, "calibration", w), f"{w}.calibration"),
        _rel(_str(d, "rubric", w), f"{w}.rubric"),
    ]
    for key in ("task_id", "task_version", "protocol_version", "robot_id"):
        _str(d, key, w)
    try:
        datetime.fromisoformat(_str(d, "recorded_at", w))  # Python 3.11 reads a trailing Z
    except ValueError as e:
        raise ExportRefused(f"{w}.recorded_at is not ISO 8601: {d.get('recorded_at')!r}") from e
    pol = d.get("policy")
    if not isinstance(pol, Mapping):
        raise ExportRefused(f"{w}.policy must be an object with 'origin'")
    if pol.get("origin") == "teleop":
        extra = sorted(k for k in pol if k != "origin")
        if extra:
            raise ExportRefused(f"{w}.policy: a teleop policy carries no identities, got {extra}")
        if purpose != "demonstration":
            raise ExportRefused(f"{w}: a teleop policy records demonstrations, not {purpose}")
    elif pol.get("origin") == "trained":
        keys = ("checkpoint_id", "training_run_id", "family_id", "weights", "config")
        missing = [k for k in keys if not isinstance(pol.get(k), str) or not pol.get(k)]
        if missing:
            raise ExportRefused(f"{w}.policy: a trained policy needs {missing}")
        unknown_p = sorted(set(pol) - {"origin", *keys, "policy_run_group_id"})
        if unknown_p:
            raise ExportRefused(f"{w}.policy has unknown fields {unknown_p}")
        if pol["family_id"] not in FAMILIES.values():
            raise ExportRefused(
                f"{w}.policy.family_id {pol['family_id']!r} is not one of {FAMILIES}"
            )
        files += [
            _rel(pol["weights"], f"{w}.policy.weights"),
            _rel(pol["config"], f"{w}.policy.config"),
        ]
    else:
        raise ExportRefused(
            f"{w}.policy.origin must be 'teleop' or 'trained', not {pol.get('origin')!r}"
        )
    labels = d.get("labels", None)
    if labels is not None:
        if not isinstance(labels, Mapping):
            raise ExportRefused(f"{w}.labels must be null or an object")
        if _str(labels, "method", f"{w}.labels") not in LABEL_METHODS:
            raise ExportRefused(f"{w}.labels.method must be one of {LABEL_METHODS}")
        files.append(_rel(_str(labels, "file", f"{w}.labels"), f"{w}.labels.file"))
    pose = d.get("object_pose", None)
    if pose is not None:
        if not isinstance(pose, Mapping):
            raise ExportRefused(f"{w}.object_pose must be null or an object")
        u = pose.get("units")
        if not isinstance(u, list) or not u or not all(isinstance(x, str) for x in u):
            raise ExportRefused(f"{w}.object_pose.units must list one unit per element")
        _str(pose, "feature", f"{w}.object_pose")
        _str(pose, "frame_id", f"{w}.object_pose")
    return {"files": files, "sessions": sessions, "policy": dict(pol), "labels": labels}


def _read_parquet_dir(directory: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise ExportRefused(f"no Parquet files under {directory}")
    rows: list[dict[str, Any]] = []
    for p in files:
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def check_dataset(root: Path, declaration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The importer's ``read_dataset`` + ``episode_frames`` on every episode; returns the
    episodes table."""
    import pyarrow.parquet as pq

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError) as e:
        raise ExportRefused(f"meta/info.json unreadable: {e}") from e
    if info.get("codebase_version") != CODEBASE_VERSION:
        raise ExportRefused(
            f"codebase_version is {info.get('codebase_version')!r}; only v3.0 is read"
        )
    if info.get("storage_format", "parquet") != "parquet":
        raise ExportRefused("only Parquet-backed datasets are read")
    fps = info.get("fps")
    if not isinstance(fps, int | float) or isinstance(fps, bool) or fps <= 0:
        raise ExportRefused("meta/info.json fps must be a positive number")
    feats = info.get("features")
    if not isinstance(feats, dict):
        raise ExportRefused("meta/info.json has no features")

    def names(key: str) -> list[str]:
        f = feats.get(key)
        if not isinstance(f, dict):
            raise ExportRefused(f"meta/info.json has no {key} feature")
        n, shape = f.get("names"), f.get("shape")
        if not isinstance(n, list) or not n or shape != [len(n)]:
            raise ExportRefused(f"{key} needs one name per element and a matching shape")
        return [str(x) for x in n]

    joints = names(conv.STATE_KEY)
    if names(conv.ACTION_KEY) != joints:
        raise ExportRefused("action and observation.state must name the same joints in order")
    if set(declaration["units"]) != set(joints):
        raise ExportRefused(
            f"declared units cover {sorted(declaration['units'])}, the joints are {joints}"
        )
    cams = {k for k, f in feats.items() if isinstance(f, dict) and f.get("dtype") == "video"}
    for k in feats:
        if k.startswith("observation.images.") and k not in cams:
            raise ExportRefused(f"{k} is stored as images, not video; only video is read")
    missing = [c.key for c in conv.CAMERAS if c.key not in cams]
    if missing:
        raise ExportRefused(f"cameras {missing} missing: both are recorded in every episode")
    problems = conv.check_dataset_info(info)
    if problems:
        raise ExportRefused(f"the dataset departs from {conv.CONVENTION_ID}: {problems}")
    episodes = _read_parquet_dir(root / "meta" / "episodes")
    seen: set[int] = set()
    for ep in episodes:
        idx = ep.get("episode_index")
        if not isinstance(idx, int) or idx in seen:
            raise ExportRefused(f"bad or duplicate episode_index {idx!r}")
        seen.add(idx)
        for key in ep:
            if key.startswith("videos/") and key.split("/")[1] not in cams:
                raise ExportRefused(f"episodes table names video {key!r} absent from info.json")
        tasks = ep.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
            raise ExportRefused(f"episode {idx}: exactly one task string is required")
        rel = str(info["data_path"]).format(
            chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
        )
        table = pq.read_table(root / rel).to_pylist()
        lo, hi = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        rows = sorted(
            (r for r in table if lo <= int(r["index"]) < hi), key=lambda r: int(r["index"])
        )
        if len(rows) != int(ep["length"]) or len(rows) != hi - lo:
            raise ExportRefused(f"episode {idx}: {len(rows)} rows, length {ep['length']}")
        for k, r in enumerate(rows):
            if int(r["episode_index"]) != idx or int(r["frame_index"]) != k:
                raise ExportRefused(f"episode {idx}: frame {k} out of order")
            if abs(float(r["timestamp"]) * float(fps) - k) > MAX_TIMESTAMP_SLIP_FRAMES:
                raise ExportRefused(f"episode {idx}: frame {k}'s timestamp is off its index")
            for key in (conv.STATE_KEY, conv.ACTION_KEY):
                v = r.get(key)
                if not isinstance(v, list) or len(v) != len(joints):
                    raise ExportRefused(
                        f"episode {idx} frame {k}: {key} must have {len(joints)} values"
                    )
                if not all(math.isfinite(float(x)) for x in v):
                    raise ExportRefused(f"episode {idx} frame {k}: {key} is not finite")
    return episodes


def check_labels(path: Path, episodes: Sequence[int]) -> None:
    """The importer's ``_label_outcome`` for every episode."""
    try:
        table = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise ExportRefused(f"label file unreadable: {e}") from e
    for idx in episodes:
        entry = table.get(str(idx)) if isinstance(table, dict) else None
        if not isinstance(entry, dict):
            raise ExportRefused(f"label file has no entry for episode {idx}")
        if not isinstance(entry.get("success"), bool):
            raise ExportRefused(f"episode {idx}: label success must be true or false")
        r = entry.get("reviewer_ids")
        if not isinstance(r, list) or not r or not all(isinstance(x, str) and x for x in r):
            raise ExportRefused(f"episode {idx}: a label needs its reviewer_ids")
        for key in ("terminated", "truncated"):
            if not isinstance(entry.get(key), bool):
                raise ExportRefused(f"episode {idx}: label {key} must be true or false")
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise ExportRefused(f"episode {idx}: a label needs its reason")


def validate_bundle(bundle: Path, dataset: Path) -> dict[str, Any]:
    """Every check above on a written bundle; returns a short summary."""
    try:
        decl = json.loads((bundle / DECLARATION_FILE).read_text())
    except (OSError, ValueError) as e:
        raise ExportRefused(f"{DECLARATION_FILE} unreadable: {e}") from e
    parsed = check_declaration(decl)
    for rel in parsed["files"]:
        if not (bundle / rel).is_file():
            raise ExportRefused(f"{rel} is declared but not in the bundle")
    episodes = check_dataset(dataset, decl)
    indices = sorted(int(e["episode_index"]) for e in episodes)
    if set(indices) != set(parsed["sessions"]):
        raise ExportRefused(
            f"sessions are declared for episodes {sorted(parsed['sessions'])}, the dataset holds "
            f"{indices}"
        )
    if parsed["labels"] is not None:
        check_labels(bundle / parsed["labels"]["file"], indices)
    if parsed["policy"]["origin"] == "trained":
        index = json.loads((bundle / CHECKPOINT_INDEX).read_text())
        got = {n: _sha256(bundle / n) for n in index["files"]}
        if got != index["files"]:
            raise ExportRefused("bundled checkpoint files do not match checkpoint_files.json")
        h = hashlib.sha256()
        for n, sha in sorted(got.items()):
            h.update(f"{n}:{sha}\n".encode())
        if f"sha256:{h.hexdigest()}" != parsed["policy"]["checkpoint_id"]:
            raise ExportRefused("checkpoint_id is not the content hash of the bundled files")
    return {"episodes": len(indices), "sessions": sorted(set(parsed["sessions"].values()))}


# ------------------------------------------------------------------------------------------------
# Building
# ------------------------------------------------------------------------------------------------
def _provenance(dataset: Path) -> dict[str, Any]:
    p = dataset / "meta" / "vla_rl_so101.json"
    if not p.is_file():
        raise ExportRefused(f"{p} missing: not a recording of this repo")
    prov: dict[str, Any] = json.loads(p.read_text())
    return prov


def _need(block: Mapping[str, Any], key: str, where: str) -> Any:
    v = block.get(key)
    if v is None or v == "" or v == {} or v == []:
        raise ExportRefused(f"{where} has no {key!r}: nothing is guessed")
    return v


def _calibration(dataset: Path, prov: Mapping[str, Any], out: Path) -> None:
    cal = _need(prov, "calibration", "the recording's provenance")
    src = dataset / str(_need(cal, "file", "provenance.calibration"))
    if not src.is_file():
        raise ExportRefused(f"{src} missing: the follower's calibration was not recorded")
    if _sha256(src) != _need(cal, "sha256", "provenance.calibration"):
        raise ExportRefused(f"{src} does not match the sha256 recorded with the session")
    shutil.copyfile(src, out / CALIBRATION_FILE)


def _protocol(prov: Mapping[str, Any]) -> None:
    proto = _need(prov, "protocol", "the recording's provenance")
    if proto.get("version") != PR.PROTOCOL_VERSION:
        raise ExportRefused(
            f"recorded under protocol {proto.get('version')!r}, not {PR.PROTOCOL_VERSION}"
        )
    if proto.get("hash") != PR.protocol_hash():
        raise ExportRefused(
            f"recorded under {PR.PROTOCOL_VERSION} hash {str(proto.get('hash'))[:12]}, this code is "
            f"{PR.protocol_hash()[:12]}: the version label would not identify what was frozen"
        )
    conv_cite = _need(prov, "convention", "the recording's provenance")
    if conv_cite.get("hash") != conv.convention_hash():
        raise ExportRefused("recorded under a different observation/action convention hash")


DEV_SUFFIX = "-dev"  # a development session's protocol_version (never reported; review F9)
MOCK_SUFFIX = "-mock"  # a mock session's (fake arm, mock policy): the declaration says so too
MOCK_EXPORT_LABEL = "MOCK EXPORT: fake arm and mock policy; a format test, NOT a real-arm result"
INCOMPLETE_SUFFIX = "-incomplete"  # an export missing a scored trial's recording (module doc)


def _common(
    prov: Mapping[str, Any],
    purpose: str,
    sessions: Mapping[int, str],
    *,
    protocol_version: str = PR.PROTOCOL_VERSION,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "dataset_root": DATASET_ROOT,
        "units": dict(_need(prov, "units", "the recording's provenance")),
        "calibration": CALIBRATION_FILE,
        "rubric": RUBRIC_FILE,
        "task_id": PR.TASK_ID,
        "task_version": PR.TASK_VERSION,
        "protocol_version": protocol_version,
        "recorded_at": str(_need(prov, "recorded_at", "the recording's provenance")),
        "robot_id": str(_need(prov, "robot_id", "the recording's provenance")),
        "purpose": purpose,
        "sessions": {str(k): v for k, v in sorted(sessions.items())},
        "labels": None,
        "object_pose": None,  # only planned poses exist (module doc)
    }


def _episode_indices(dataset: Path) -> list[int]:
    return sorted(int(e["episode_index"]) for e in _read_parquet_dir(dataset / "meta" / "episodes"))


def _fresh(out: Path) -> None:
    if out.exists():
        raise ExportRefused(f"{out} exists: a bundle is written once (pick a new --out)")
    out.mkdir(parents=True)


def build_demo_bundle(dataset: Path, out: Path) -> dict[str, Any]:
    """Bundle a finished teleop demo recording (``scripts/wo06_teleop_record.py --demo-schedule``)."""
    prov = _provenance(dataset)
    _protocol(prov)
    if prov.get("rehearsal") is not False:
        raise ExportRefused("a rehearsal recording (or one that does not say) is never exported")
    if prov.get("purpose") != "demonstration" or prov.get("demo_schedule") is not True:
        raise ExportRefused("not a --demo-schedule demonstration recording")
    sid = str(_need(prov, "session_id", "the recording's provenance"))
    eps = _episode_indices(dataset)
    placements = _need(prov, "episodes", "the recording's provenance")
    missing = [i for i in eps if str(i) not in placements]
    if missing:
        raise ExportRefused(f"episodes {missing} have no recorded demo placement")
    if any(placements[str(i)].get("kind") != "demo" for i in eps):
        raise ExportRefused("a demo episode's placement is not from the frozen demo schedule")
    _fresh(out)
    _calibration(dataset, prov, out)
    _write(out / RUBRIC_FILE, PR.rubric())
    _write(out / PLACEMENTS_FILE, {str(i): placements[str(i)] for i in eps})
    decl = {**_common(prov, "demonstration", {i: sid for i in eps}), "policy": {"origin": "teleop"}}
    _write(out / DECLARATION_FILE, decl)
    extras = [CALIBRATION_FILE, RUBRIC_FILE, PLACEMENTS_FILE]
    summary = validate_bundle(out, dataset)
    index = {"dataset": str(dataset), "extras": extras, "purpose": "demonstration", **summary}
    _write(out / BUNDLE_INDEX, index)
    return index


def _review_labels(
    reviews: Sequence[Mapping[str, Any]], sid: str, by_episode: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    """labels.json from blinded reviews (scripts/wo07_blind_review.py): every review must cover
    every episode; success = every reviewer said success (R1) and the trial had no fault after
    the start (rubric R2, from the recorded fault, not a judgement)."""
    verdicts: dict[int, list[tuple[str, str, Any]]] = {i: [] for i in by_episode}
    for rv in reviews:
        if rv.get("method") != "blinded_review" or not rv.get("policy_identity_hidden"):
            raise ExportRefused("a review file is not a blinded clip review")
        policies = rv.get("session_policies") or {}
        if sid not in policies or not all(policies.values()) or len(set(policies.values())) < 2:
            raise ExportRefused(
                "a blinded review must pool clips of at least two different policies (and this "
                f"session's): {policies or 'no session_policies recorded'}"
            )
        reviewer = str(_need(rv, "reviewer_id", "a review file"))
        got = {
            int(it["episode_index"]): it
            for it in rv.get("items", [])
            if it.get("session_id") == sid and it.get("verdict") in ("success", "failure")
        }
        missing = sorted(set(by_episode) - set(got))
        if missing:
            raise ExportRefused(
                f"review by {reviewer} does not cover episodes {missing[:10]}: the blinded review "
                "is incomplete (omit --review to export without labels)"
            )
        for i in by_episode:
            verdicts[i].append((reviewer, got[i]["verdict"], got[i].get("partial")))
    labels: dict[str, Any] = {}
    for i, rec in sorted(by_episode.items()):
        fault = rec.get("fault")
        r1 = all(v == "success" for _, v, _ in verdicts[i])
        stop = str(rec.get("stop_reason") or "")
        reason = f"blinded review (R1): {'success' if r1 else 'failure'}"
        if fault:
            reason += f"; rubric R2: fault after the policy started ({fault.get('kind')}: {stop}) = failure"
        partial = [p for _, _, p in verdicts[i] if p is not None]
        labels[str(i)] = {
            "success": bool(r1 and not fault),
            "reviewer_ids": sorted({r for r, _, _ in verdicts[i]}),
            "terminated": stop != "max_steps",
            "truncated": stop == "max_steps",
            "reason": reason,
            "trial_id": rec.get("trial_id"),
            **({"secondary_partial_credit": min(partial)} if partial else {}),
        }
    return labels


def build_eval_bundle(
    capture_dataset: Path,
    settings: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
    out: Path,
    *,
    checkpoint_dir: Path,
    reviews: Sequence[Mapping[str, Any]] = (),
    mode: str = "hardware",
    allow_missing_recordings: bool = False,
) -> dict[str, Any]:
    """Bundle one finished evaluation session (``scripts/wo07_real_eval.py``, protocol real-0.1).
    A scored trial without its episode refuses the export unless ``allow_missing_recordings``
    (then declared ``-incomplete``; module doc)."""
    if mode not in ("hardware", "mock"):
        raise ExportRefused(f"session mode {mode!r} is neither hardware nor mock")
    prov = _provenance(capture_dataset)
    _protocol(prov)
    sid = str(_need(settings, "session_id", "the session"))
    if prov.get("session_id") != sid:
        raise ExportRefused(
            f"the capture belongs to session {prov.get('session_id')!r}, not {sid!r}"
        )
    if settings.get("protocol_id") != PR.PROTOCOL_VERSION:
        raise ExportRefused(f"session {sid} is not a {PR.PROTOCOL_VERSION} session")
    ident = _need(settings, "policy_identity", f"session {sid}")
    for key in ("family_id", "training_run_id", "checkpoint_sha256"):
        _need(ident, key, f"session {sid} policy_identity")
    if not ident.get("training_run_verified") and mode != "mock":
        raise ExportRefused(
            f"session {sid}: training_run_id {ident['training_run_id']!r} was not verified against "
            "the checkpoint's provenance: nothing is guessed"
        )
    got = checkpoint_identity(checkpoint_dir)
    if got["sha256"] != ident["checkpoint_sha256"]:
        raise ExportRefused(
            f"{checkpoint_dir} hashes to {got['sha256'][:12]}, the session ran "
            f"{str(ident['checkpoint_sha256'])[:12]}: not the evaluated checkpoint"
        )
    eps = _episode_indices(capture_dataset)
    recs = [dict(t) for t in trials]  # a filled capture block below never edits the caller's
    by_episode: dict[int, dict[str, Any]] = {}
    for t in recs:
        idx = (t.get("capture") or {}).get("episode_index")
        if idx is not None:
            if int(idx) in by_episode:
                raise ExportRefused(f"episode {idx} is claimed by two trial records")
            by_episode[int(idx)] = t
    pending = pending_episodes(recs, prov, eps, Path(capture_dataset).parent)
    if pending:  # never dropped, never counted: always refused, allow_missing_recordings too
        raise ExportRefused(
            f"PENDING: {len(pending)} episode(s) saved but not confirmed durable "
            f"({[(p['trial_id'], p['episode_index']) for p in pending]}): --resume the session "
            "to verify and sync them first; nothing is exported while an episode is pending"
        )
    by_trial = {str(t.get("trial_id")): t for t in recs if t.get("policy_started")}
    for i in sorted(set(eps) - set(by_episode)):  # saved, then killed before its capture row
        entry = (prov.get("episodes") or {}).get(str(i)) or {}
        rec = by_trial.get(str(entry.get("trial_id")))
        if rec is not None and (rec.get("capture") or {}).get("episode_index") is None:
            by_episode[i] = rec
            rec["capture"] = {
                **(rec.get("capture") or {}),
                "episode_index": i,
                "filled_from": "capture provenance (the run ended before its capture row)",
            }
    orphans = sorted(set(eps) - set(by_episode))
    if orphans:
        raise ExportRefused(f"episodes {orphans} have no trial record (a run killed mid-trial?)")
    unknown = sorted(set(by_episode) - set(eps))
    if unknown:
        raise ExportRefused(f"trial records name episodes {unknown} that the dataset does not hold")
    for i, t in by_episode.items():
        if not t.get("policy_started") or t.get("outcome") not in ("success", "failure"):
            raise ExportRefused(f"episode {i} is not a scored trial")
        if not t.get("labeler") and t.get("outcome_source") == "operator":
            raise ExportRefused(f"episode {i}: an operator score without a named labeler")
    missing = missing_recordings(recs, Path(capture_dataset).parent)
    if missing and not allow_missing_recordings:
        pending = [m["trial_id"] for m in missing if m["status"] == "recoverable_on_resume"]
        raise ExportRefused(
            f"INCOMPLETE: {len(missing)} scored trial(s) have no recording "
            f"({[m['trial_id'] for m in missing]}); "
            + (
                f"{pending} are staged: --resume the session to recover them first; "
                if pending
                else ""
            )
            + "vishwaroop's declaration cannot say incomplete, so a complete evaluation bundle is "
            f"refused (allow_missing_recordings declares protocol_version "
            f"{PR.PROTOCOL_VERSION}{INCOMPLETE_SUFFIX})"
        )
    _fresh(out)
    _calibration(capture_dataset, prov, out)
    _write(out / RUBRIC_FILE, PR.rubric())
    uncaptured = [
        {
            "trial_id": t.get("trial_id"),
            "cell": t.get("cell"),
            "trial_idx": t.get("trial_idx"),
            "capture": t.get("capture"),
        }
        for t in recs
        if t.get("policy_started") and (t.get("capture") or {}).get("episode_index") is None
    ]
    _write(
        out / TRIALS_FILE,
        {
            "session_id": sid,
            "mode": mode,
            "mock_label": MOCK_EXPORT_LABEL if mode == "mock" else None,
            "protocol": PR.citation(),
            "schedule": settings.get("schedule"),
            "labeler": settings.get("labeler"),
            "note": "every trial record and not-started attempt of the session with its actual "
            "provenance (outcome_source operator | scripted | fault | unjudged | not_scored); "
            "the declared labels, if any, come from the blinded review only",
            "uncaptured_started_trials": uncaptured,
            "export_status": "INCOMPLETE" if missing else "COMPLETE",
            "recording_missing": missing,
            "trials": recs,
        },
    )
    for name in got["files"]:
        shutil.copyfile(Path(checkpoint_dir) / name, out / name)
    _write(out / CHECKPOINT_INDEX, {"sha256": got["sha256"], "files": got["files"]})
    dev = settings.get("schedule") != "eval"
    decl = _common(
        prov,
        "evaluation",
        {i: sid for i in eps},
        protocol_version=PR.PROTOCOL_VERSION
        + (MOCK_SUFFIX if mode == "mock" else "")
        + (DEV_SUFFIX if dev else "")
        + (INCOMPLETE_SUFFIX if missing else ""),
    )
    decl["policy"] = {
        "origin": "trained",
        "checkpoint_id": f"sha256:{got['sha256']}",
        "training_run_id": str(ident["training_run_id"]),
        "family_id": str(ident["family_id"]),
        "weights": WEIGHTS_FILE,
        "config": CONFIG_FILE,
    }
    extras = [CALIBRATION_FILE, RUBRIC_FILE, TRIALS_FILE, CHECKPOINT_INDEX, *got["files"]]
    disagreements: list[dict[str, Any]] = []
    if reviews:
        labels = _review_labels(reviews, sid, by_episode)
        _write(out / LABELS_FILE, labels)
        decl["labels"] = {"file": LABELS_FILE, "method": "blinded_review"}
        extras.append(LABELS_FILE)
        disagreements = [
            {
                "episode_index": int(i),
                "blinded": v["success"],
                "live": by_episode[int(i)]["outcome"],
            }
            for i, v in labels.items()
            if v["success"] != (by_episode[int(i)]["outcome"] == "success")
        ]
    _write(out / DECLARATION_FILE, decl)
    summary = validate_bundle(out, capture_dataset)
    rates = {
        i: (t.get("capture") or {}).get("write_rate_hz") for i, t in sorted(by_episode.items())
    }
    off = {
        str(i): r
        for i, r in rates.items()
        if r is not None and abs(float(r) - conv.FPS) > WRITE_RATE_TOLERANCE * conv.FPS
    }
    index = {
        "dataset": str(capture_dataset),
        "extras": extras,
        "purpose": "evaluation",
        "protocol_version": decl["protocol_version"],
        "mode": mode,
        "mock_label": MOCK_EXPORT_LABEL if mode == "mock" else None,
        "timing": {
            "note": "LeRobot timestamps are nominal (frame_index / fps); the real write and camera "
            f"times are the {conv.TIMING_KEY} feature",
            "nominal_fps": conv.FPS,
            "episodes_off_nominal_rate": off,
            "warning": (
                f"{len(off)} episodes wrote at a rate more than "
                f"{WRITE_RATE_TOLERANCE:.0%} off the nominal {conv.FPS} fps"
                if off
                else None
            ),
        },
        "labels": decl["labels"],
        "uncaptured_started_trials": len(uncaptured),
        "status": "INCOMPLETE" if missing else "COMPLETE",
        "recording_missing": missing,
        "blinded_vs_live_disagreements": disagreements,
        **summary,
    }
    _write(out / BUNDLE_INDEX, index)
    return index


def pending_episodes(
    trials: Sequence[Mapping[str, Any]],
    prov: Mapping[str, Any],
    episodes: Sequence[int],
    capture_root: Path,
) -> list[dict[str, Any]]:
    """Episodes saved but not confirmed durable (``recording.capture_unconfirmed``, the rule the
    session summary uses; owner hold on PR #32): a trial's pending block, a record that claims
    its index beside the save's error, a provenance entry marked ``sync_pending``, or a saved
    episode (in the dataset, or claimed by a record) whose trial still has a whole staged journal
    (its sync failed or never ran). A resume confirms them (``EvalCapture.confirm_saved``)."""
    from src.robot.recording import capture_pending, capture_unconfirmed, has_staged

    found: dict[str, dict[str, Any]] = {}

    def add(tid: Any, idx: Any, why: str) -> None:
        found.setdefault(str(tid), {"trial_id": tid, "episode_index": idx, "why": why})

    for t in trials:
        cap = t.get("capture") or {}
        tid = t.get("trial_id")
        if capture_pending(cap):
            add(tid, cap["pending_episode_index"], str(cap.get("sync_error") or cap.get("reason")))
        elif capture_unconfirmed(cap, capture_root, tid):
            why = cap.get("sync_error") or cap.get("error") or "saved, its staged journal kept"
            add(tid, cap["episode_index"], str(why))
    entries = prov.get("episodes") or {}
    for i in episodes:
        entry = entries.get(str(i)) or {}
        tid = entry.get("trial_id")
        if entry.get("sync_pending"):
            add(tid, i, str(entry["sync_pending"]))
        elif tid is not None and has_staged(capture_root, str(tid)):
            add(tid, i, "saved, its staged journal kept")
    return list(found.values())


def missing_recordings(
    trials: Sequence[Mapping[str, Any]], capture_root: Path
) -> list[dict[str, Any]]:
    """Scored trials that need a recording (``recording.recording_required``: not one no policy
    write reached, which is listed under ``uncaptured_started_trials`` only) with no episode
    index: ``recording_missing`` when flagged (no journal, or a damaged one) or nothing is
    staged, else ``recoverable_on_resume`` (a whole staged journal under ``capture_root``: a
    resume saves it). The same classification as ``wo07_real_eval.capture_gaps``."""
    from src.robot.recording import capture_pending, has_staged, recording_required

    out = []
    for t in trials:
        cap = t.get("capture") or {}
        if not recording_required(t) or cap.get("episode_index") is not None:
            continue
        tid = t.get("trial_id")
        staged = capture_pending(cap) or (
            not cap.get("recording_missing")
            and tid is not None
            and has_staged(capture_root, str(tid))
        )
        out.append(
            {
                "trial_id": tid,
                "cell": t.get("cell"),
                "trial_idx": t.get("trial_idx"),
                "outcome": t.get("outcome"),
                "status": "recoverable_on_resume" if staged else "recording_missing",
                "reason": cap.get("reason"),
            }
        )
    return out


def seal_command(index: Mapping[str, Any], bundle: Path, export_id: str, export_root: str) -> str:
    """The owner's vishwaroop seal command for a bundle (``docs/SO101_EXPORT.md`` section 1)."""
    extras = " ".join(f"--extra {bundle / e}" for e in index["extras"])
    return (
        "python -m vishwaroop.data.so101_import seal "
        f"--dataset {index['dataset']} --declaration {bundle / DECLARATION_FILE} {extras} "
        f"--export-id {export_id} --out {export_root}"
    )
