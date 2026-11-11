"""vishwaroop export inputs (src/eval/export_bundle.py, scripts/wo07_export_bundle.py) and the blinded
clip review (scripts/wo07_blind_review.py), from fake recordings made with the REAL recorder
(demos: scripts/wo06_teleop_record.py on the fake follower and two fake cameras) and the REAL
evaluation capture (scripts/wo07_real_eval.py --mock, protocol real-0.1).

``validate_bundle`` mirrors vishwaroop's importer checks (its so101_import.py); the real importer
is not installed here (it needs Python >= 3.12 and its own lock), so the run against it is the
documented mock export in docs/STATUS.md ("PR 2: eval")."""

from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.config import REPO_ROOT
from src.eval import export_bundle as X
from src.eval import protocol_real as PR
from src.robot import convention as conv
from src.robot import recording
from src.robot.client import ScriptedOperator


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_xb", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")
xb = _load("wo07_export_bundle")
br = _load("wo07_blind_review")


@pytest.fixture(scope="module")
def demo_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two demos of the frozen demo schedule, recorded through the real teleop recorder path."""
    import test_convention as tc
    from robot_fakes import real_cfg

    root = tmp_path_factory.mktemp("demo") / "so101-demos"
    answers = ["", "y", "", "y", ""]  # scene, keep, scene, keep, support ENTER
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("builtins.input", lambda _p="": answers.pop(0))
        rig, _ = tc._rig()
        sess, fol = tc._session(tc.MovingLeader(), rig)
        dataset = recording.DurableDataset(
            tc.teleop.create_dataset("local/so101-demos", root), "local/so101-demos"
        )
        instr = tc.teleop.protocol_instructions(["PickPlace"])
        cfg = real_cfg()
        prov = tc.teleop.recording_provenance(
            cfg, sess.units, instr, dict(instr), rehearsal=False, demo_schedule=True
        )
        prov["calibration"] = recording.copy_calibration(fol, root)
        placements = PR.demo_schedule()[:2]
        with rig:
            saved = tc.teleop.record_session(
                sess, dataset, ["PickPlace"], instr, 2, 0.4, placements
            )
        dataset.close()
        assert saved == 2 and not answers
        prov["episodes"] = dict(sess.placements)
        recording.write_json(root / recording.PROVENANCE, prov)
    return root


class _Session:
    def __init__(self, runs: Path, record: Path, ckpt: Path) -> None:
        self.runs, self.record, self.ckpt = runs, record, ckpt

    @property
    def results(self) -> dict[str, Any]:
        out: dict[str, Any] = json.loads(self.record.read_text())["results"]
        return out


def _mock_session(base: Path, family: str, weights: bytes, run_id: str) -> _Session:
    runs = base / "runs"
    ckpt = base / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(weights)
    (ckpt / "config.json").write_text(f'{{"type": "{family}", "chunk_size": 50}}\n')
    (ckpt / "policy_preprocessor.json").write_text("{}\n")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "ScriptedOperator", ScriptedOperator)
        rec = ev.main(
            [
                "--preset", "real", "--mock", "--fake-clock", "--max-steps", "5",
                "--task", "PickPlace", "--trials", "1", "--labeler", "mock-labeler",
                "--family", family, "--training-run-id", run_id,
                "--checkpoint-dir", str(ckpt), "--runs-dir", str(runs),
            ]
        )  # fmt: skip
    return _Session(runs, rec, ckpt)


@pytest.fixture(scope="module")
def eval_session(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Session]:
    base = tmp_path_factory.mktemp("eval")
    yield _mock_session(base, "smolvla", b"\x00weights\x01", "20261109T000000Z-wo07-smolvla-abcd")


@pytest.fixture(scope="module")
def eval_session_act(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Session]:
    """A second session, another policy: what a blinded review pools with the first."""
    base = tmp_path_factory.mktemp("eval-act")
    yield _mock_session(base, "act", b"\x00act weights\x02", "20261109T000000Z-wo07-act-ef01")


def _export(s: _Session, out: Path, *extra: str) -> dict[str, Any]:
    argv = ["--preset", "real", "eval", "--session", s.record.parent.name]
    argv += ["--runs-dir", str(s.runs), "--out", str(out), "--allow-mock", *extra]
    result: dict[str, Any] = xb.main(argv)
    return result


def test_demo_bundle_is_complete_and_valid(demo_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "demo.export"
    index = X.build_demo_bundle(demo_dataset, out)
    assert index["episodes"] == 2 and index["purpose"] == "demonstration"
    decl = json.loads((out / X.DECLARATION_FILE).read_text())
    assert decl["format"] == "lerobot-v3.0" and decl["dataset_root"] == "dataset"
    assert decl["units"] == conv.DECLARED_UNITS and decl["policy"] == {"origin": "teleop"}
    assert decl["protocol_version"] == "real-0.1" and decl["task_version"] == PR.TASK_VERSION
    assert decl["labels"] is None and decl["object_pose"] is None  # planned poses only
    assert len(set(decl["sessions"].values())) == 1 and set(decl["sessions"]) == {"0", "1"}
    placements = json.loads((out / X.PLACEMENTS_FILE).read_text())
    assert placements["0"] == PR.demo_schedule()[0] and placements["0"]["kind"] == "demo"
    cal = json.loads((out / X.CALIBRATION_FILE).read_text())
    assert set(cal) == set(conv.JOINT_NAMES)
    assert json.loads((out / X.RUBRIC_FILE).read_text())["protocol_hash"] == PR.protocol_hash()
    with pytest.raises(X.ExportRefused, match="written once"):
        X.build_demo_bundle(demo_dataset, out)


def test_demo_without_the_demo_schedule_is_refused(demo_dataset: Path, tmp_path: Path) -> None:
    import shutil

    ds = tmp_path / "copy"
    shutil.copytree(demo_dataset, ds)
    p = ds / recording.PROVENANCE
    prov = json.loads(p.read_text())
    prov["episodes"] = {}
    p.write_text(json.dumps(prov))
    with pytest.raises(X.ExportRefused, match="no 'episodes'"):
        X.build_demo_bundle(ds, tmp_path / "b1")
    prov["episodes"] = {"0": {"kind": "demo"}}
    p.write_text(json.dumps(prov))
    with pytest.raises(X.ExportRefused, match=r"episodes \[1\] have no recorded demo placement"):
        X.build_demo_bundle(ds, tmp_path / "b2")


def test_eval_bundle_is_complete_and_valid(eval_session: _Session, tmp_path: Path) -> None:
    out = tmp_path / "eval.export"
    index = _export(eval_session, out)
    r = eval_session.results
    sid = r["session"]["session_id"]
    assert index["episodes"] == 3 and index["sessions"] == [sid] and index["mode"] == "mock"
    assert index["mock_label"] == X.MOCK_EXPORT_LABEL  # labelled mock: a format test only
    decl = json.loads((out / X.DECLARATION_FILE).read_text())
    # the sealed declaration (what vishwaroop imports) says mock too, never plain real-0.1
    assert decl["protocol_version"] == "real-0.1-mock" == index["protocol_version"]
    pol = decl["policy"]
    ident = X.checkpoint_identity(eval_session.ckpt)
    assert pol == {
        "origin": "trained",
        "checkpoint_id": f"sha256:{ident['sha256']}",
        "training_run_id": "20261109T000000Z-wo07-smolvla-abcd",
        "family_id": "so101-real-smolvla",
        "weights": "model.safetensors",
        "config": "config.json",
    }
    assert decl["purpose"] == "evaluation" and decl["labels"] is None  # no blinded review
    assert set(decl["sessions"].values()) == {sid}
    trials = json.loads((out / X.TRIALS_FILE).read_text())
    assert [t["outcome_source"] for t in trials["trials"]] == ["scripted"] * 3
    assert trials["mode"] == "mock" and trials["mock_label"] == X.MOCK_EXPORT_LABEL
    assert not (out / X.LABELS_FILE).exists()
    for name in ident["files"]:  # every checkpoint file travels, content-hashed
        assert (out / name).read_bytes() == (eval_session.ckpt / name).read_bytes()
    assert "--extra" in X.seal_command(index, out, "x", "y")
    # the CLI refuses a mock session without --allow-mock, and other weights
    with pytest.raises(SystemExit, match="allow-mock"):
        xb.main(
            ["--preset", "real", "eval", "--session", eval_session.record.parent.name]
            + ["--runs-dir", str(eval_session.runs), "--out", str(tmp_path / "x")]
        )
    saved = eval_session.record.read_text()  # a record without its mode is not hardware
    data = json.loads(saved)
    data["results"].pop("mode")
    try:
        eval_session.record.write_text(json.dumps(data))
        with pytest.raises(SystemExit, match="no recorded mode"):
            _export(eval_session, tmp_path / "z")
    finally:
        eval_session.record.write_text(saved)
    other = tmp_path / "other"
    other.mkdir()
    (other / "model.safetensors").write_bytes(b"other")
    (other / "config.json").write_text("{}")
    with pytest.raises(SystemExit, match="not the evaluated checkpoint"):
        _export(eval_session, tmp_path / "y", "--checkpoint-dir", str(other))


def _capture(s: _Session) -> Path:
    return s.runs / s.results["session"]["session_id"] / "capture"


def test_blinded_review_labels_and_provenance(
    eval_session: _Session, eval_session_act: _Session, tmp_path: Path
) -> None:
    sid = eval_session.results["session"]["session_id"]
    sid_act = eval_session_act.results["session"]["session_id"]
    answers = iter(["y", "n", "y", "y", "n", "y"])
    prompts: list[str] = []

    def ask(msg: str) -> str:
        prompts.append(msg)
        return next(answers)

    # the SmolVLA and the ACT session pooled: the reviewer cannot tell whose clip it is
    caps = [_capture(eval_session), _capture(eval_session_act)]
    path = br.review(caps, "reviewer-a", tmp_path / "rev", seed=1, ask=ask)
    rv = json.loads(path.read_text())
    assert rv["complete"] and rv["method"] == "blinded_review" and rv["policy_identity_hidden"]
    assert set(rv["session_policies"]) == {sid, sid_act}
    packets = sorted((tmp_path / "rev" / "packets").iterdir())
    assert len(packets) == 6
    for p in packets:  # nothing in a packet or a prompt names the session or the policy
        assert sid not in p.name and {f.name for f in p.iterdir()} == {
            "trial_contact_sheet.png",
            "end_state_front.png",
            "end_state_wrist.png",
        }
    assert not any(sid in m or sid_act in m or "smolvla" in m for m in prompts)
    out = tmp_path / "labelled"
    index = _export(eval_session, out, "--review", str(path))
    decl = json.loads((out / X.DECLARATION_FILE).read_text())
    assert decl["labels"] == {"file": "labels.json", "method": "blinded_review"}
    labels = json.loads((out / X.LABELS_FILE).read_text())
    mine = {it["episode_index"]: it["verdict"] for it in rv["items"] if it["session_id"] == sid}
    assert set(labels) == {str(i) for i in mine}
    for k, v in labels.items():
        assert v["success"] == (mine[int(k)] == "success")  # no fault in these trials
        assert v["reviewer_ids"] == ["reviewer-a"] and v["truncated"] is True
    # the live (scripted) scores were all success: every blinded n is a disagreement, reported
    assert sorted(d["episode_index"] for d in index["blinded_vs_live_disagreements"]) == sorted(
        i for i, v in mine.items() if v == "failure"
    )
    # an incomplete review is refused, never partially declared
    part = dict(rv, items=rv["items"][:2])
    p2 = tmp_path / "partial.json"
    p2.write_text(json.dumps(part))
    with pytest.raises(SystemExit, match="does not cover episodes"):
        _export(eval_session, tmp_path / "z", "--review", str(p2))


def test_one_session_review_is_not_blinded(eval_session: _Session, tmp_path: Path) -> None:
    """Review F2: one policy's clips alone: the reviewer knows whose they are, so the file says
    unblinded and the export refuses to declare blinded_review from it."""
    answers = iter(["y"] * 3)
    path = br.review(
        [_capture(eval_session)],
        "reviewer-a",
        tmp_path / "rev",
        seed=1,
        ask=lambda _m: next(answers),
    )
    rv = json.loads(path.read_text())
    assert rv["method"] == "unblinded_clip_review" and rv["policy_identity_hidden"] is False
    with pytest.raises(SystemExit, match="not a blinded clip review"):
        _export(eval_session, tmp_path / "x", "--review", str(path))
    forged = dict(rv, method="blinded_review", policy_identity_hidden=True)
    p2 = tmp_path / "forged.json"
    p2.write_text(json.dumps(forged))
    with pytest.raises(SystemExit, match="at least two different policies"):
        _export(eval_session, tmp_path / "y", "--review", str(p2))


def test_rubric_r2_fault_after_start_is_a_failure_label() -> None:
    rec = {"fault": {"kind": "safety"}, "stop_reason": "deadman", "trial_id": "s/seen/000"}
    rv = {
        "method": "blinded_review",
        "policy_identity_hidden": True,
        "reviewer_id": "r",
        "session_policies": {"s": "sha-a", "t": "sha-b"},
        "items": [{"session_id": "s", "episode_index": 0, "verdict": "success"}],
    }
    labels = X._review_labels([rv], "s", {0: rec})
    assert labels["0"]["success"] is False and "R2" in labels["0"]["reason"]
    assert labels["0"]["terminated"] is True and labels["0"]["truncated"] is False
    with pytest.raises(X.ExportRefused, match="not a blinded clip review"):
        X._review_labels([{**rv, "policy_identity_hidden": False}], "s", {0: rec})


@pytest.fixture(scope="module")
def valid_decl(eval_session: _Session, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp("decl") / "b"
    _export(eval_session, out)
    d: dict[str, Any] = json.loads((out / X.DECLARATION_FILE).read_text())
    return d


REQUIRED = [
    "format", "dataset_root", "units", "calibration", "rubric", "task_id", "task_version",
    "protocol_version", "recorded_at", "robot_id", "purpose", "sessions", "policy",
]  # fmt: skip


@pytest.mark.parametrize("key", REQUIRED)
def test_each_missing_field_is_refused(valid_decl: dict[str, Any], key: str) -> None:
    X.check_declaration(valid_decl)  # the bundle's own declaration passes
    bad = copy.deepcopy(valid_decl)
    del bad[key]
    with pytest.raises(X.ExportRefused):
        X.check_declaration(bad)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda d: d.update(format="lerobot-v2.1"), "format"),
        (lambda d: d["units"].update(gripper="percent"), "not one of"),
        (lambda d: d["units"].update(gripper="deg"), "what was recorded"),
        (lambda d: d["units"].pop("gripper"), "what was recorded"),
        (lambda d: d.update(purpose="training"), "purpose"),
        (lambda d: d.update(sessions={"x": "s"}), "bad entry"),
        (lambda d: d.update(sessions={}), "every episode"),
        (lambda d: d.update(recorded_at="yesterday"), "ISO 8601"),
        (lambda d: d.update(dataset_root="../ds"), "relative path"),
        (lambda d: d.update(extra_field=1), "unknown fields"),
        (lambda d: d["policy"].pop("training_run_id"), "needs"),
        (lambda d: d["policy"].update(family_id="smolvla-new"), "family_id"),
        (lambda d: d["policy"].update(origin="scripted"), "origin"),
        (lambda d: d.update(policy={"origin": "teleop"}), "demonstrations"),
        (lambda d: d.update(labels={"file": "labels.json", "method": "operator"}), "method"),
        (lambda d: d.update(object_pose={"feature": "x", "frame_id": "t", "units": []}), "units"),
    ],
)
def test_unknown_or_bad_values_are_refused(
    valid_decl: dict[str, Any], edit: Any, match: str
) -> None:
    bad = copy.deepcopy(valid_decl)
    edit(bad)
    with pytest.raises(X.ExportRefused, match=match):
        X.check_declaration(bad)


def test_dataset_checks_mirror_the_importer(eval_session: _Session, tmp_path: Path) -> None:
    import shutil

    sid = eval_session.results["session"]["session_id"]
    ds = tmp_path / "ds"
    shutil.copytree(eval_session.runs / sid / "capture" / "dataset", ds)
    decl = {"units": dict(conv.DECLARED_UNITS)}
    assert len(X.check_dataset(ds, decl)) == 3
    info_p = ds / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info_p.write_text(json.dumps({**info, "codebase_version": "v2.1"}))
    with pytest.raises(X.ExportRefused, match="only v3.0"):
        X.check_dataset(ds, decl)
    info_p.write_text(json.dumps(info))
    with pytest.raises(X.ExportRefused, match="declared units cover"):
        X.check_dataset(ds, {"units": {"j": "rad"}})
    with pytest.raises(X.ExportRefused, match="no entry for episode 2"):
        lab = tmp_path / "labels.json"
        ok = {"success": True, "reviewer_ids": ["r"], "terminated": True, "truncated": False}
        lab.write_text(json.dumps({"0": {**ok, "reason": "x"}, "1": {**ok, "reason": "x"}}))
        X.check_labels(lab, [0, 1, 2])
