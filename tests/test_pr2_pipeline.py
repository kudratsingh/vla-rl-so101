"""PR 2 end to end on CPU, fakes only: the pieces fit through ``src/robot/convention.py``.

Demos recorded by the REAL teleop recorder (scripts/wo06_teleop_record.py: fake follower, two fake
cameras) -> the tiny trainer (scripts/wo07_train.py, on THAT dataset) -> its saved checkpoint ->
the Mac inference path (src/robot/local_policy.py, one DeployLoop tick on the fakes) -> the real
eval (scripts/wo07_real_eval.py --mock --backend local, protocol real-0.1, every trial captured
into the evaluation LeRobotDataset) -> the vishwaroop export bundles (src/eval/export_bundle.py)
of that demo session and that evaluation session, checked by the importer mirror.

One checkpoint identity throughout (``lerobot_ckpt.checkpoint_identity``): the trainer's run
record, the local policy's ``checkpoint_sha256``, the session's ``policy_identity`` and the
export's ``checkpoint_id``. ACT always runs; SmolVLA needs the smolvla extra and the cached
``lerobot/smolvla_base`` (skipped otherwise, e.g. on CI). No serial port, camera or network."""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.eval import export_bundle as X
from src.eval import protocol_real as PR
from src.policy import lerobot_ckpt
from src.robot import convention as conv
from src.robot import local_policy as lp
from src.robot import recording
from src.robot.client import ScriptedOperator, run_trials
from src.runs import Run


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"{name}_pr2", REPO_ROOT / "scripts" / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_pr2"] = mod
    spec.loader.exec_module(mod)
    return mod


train = _load("wo07_train")
ev = _load("wo07_real_eval")
xb = _load("wo07_export_bundle")


def _smolvla_available() -> bool:
    from huggingface_hub import try_to_load_from_cache

    if importlib.util.find_spec("num2words") is None:  # the smolvla extra
        return False
    return all(
        isinstance(try_to_load_from_cache(repo, name), str)
        for repo, name in (
            ("lerobot/smolvla_base", "model.safetensors"),
            ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "model.safetensors"),
        )
    )


FAMILIES = [
    "act",
    pytest.param(
        "smolvla",
        marks=pytest.mark.skipif(
            not _smolvla_available(), reason="smolvla extra / smolvla_base not cached"
        ),
    ),
]


@pytest.fixture(scope="module")
def demos(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two demos of the frozen demo schedule through the real recorder path (both cameras)."""
    import test_convention as tc
    from robot_fakes import real_cfg

    root = tmp_path_factory.mktemp("demo") / "so101-demos"
    answers = ["", "y", "", "y", ""]  # scene, keep, scene, keep, support ENTER
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("builtins.input", lambda _p="": answers.pop(0))
        rig, _ = tc._rig()
        sess, fol = tc._session(tc.MovingLeader(), rig)
        dataset = tc.teleop.create_dataset("local/so101-demos", root)
        instr = tc.teleop.protocol_instructions(["PickPlace"])
        cfg = real_cfg()
        prov = tc.teleop.recording_provenance(
            cfg, sess.units, instr, dict(instr), rehearsal=False, demo_schedule=True
        )
        prov["calibration"] = recording.copy_calibration(fol, root)
        with rig:
            saved = tc.teleop.record_session(
                sess, dataset, ["PickPlace"], instr, 2, 1.0, PR.demo_schedule()[:2]
            )
        dataset.finalize()
        assert saved == 2 and not answers
        prov["episodes"] = dict(sess.placements)
        recording.write_json(root / recording.PROVENANCE, prov)
    return root


def _train(family: str, demos: Path, base: Path) -> dict[str, Any]:
    """scripts/wo07_train.py --preset tiny on the RECORDED demos, checkpoint kept."""
    cfg = base / "train.yaml"
    cfg.write_text(
        "inherit: configs/wo07_train_tiny.yaml\n"
        "dataset:\n"
        "  synthetic: false\n"
        "  repo_id: local/so101-demos\n"
        f"  root: {demos.parent}\n"
        "  revision: null\n"
    )
    runs = base / "train-runs"
    argv = ["wo07_train.py", "--preset", "tiny", "--config", str(cfg), "--policy", family]
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")  # cached weights only
        mp.setattr(train, "Run", functools.partial(Run, runs_dir=runs))
        mp.setattr(sys, "argv", [*argv, "--keep-checkpoint"])
        assert train.main() == 0
    (record,) = runs.glob("*/record.json")
    rec: dict[str, Any] = json.loads(record.read_text())
    return rec


@pytest.fixture(scope="module", params=FAMILIES)
def trained(
    request: pytest.FixtureRequest, demos: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, dict[str, Any], Path]:
    family = str(request.param)
    rec = _train(family, demos, tmp_path_factory.mktemp(f"train-{family}"))
    return family, rec, Path(rec["results"]["checkpoint"]["path"])


def test_trainer_reads_the_recorder_dataset(
    trained: tuple[str, dict[str, Any], Path], demos: Path
) -> None:
    _family, rec, ckpt = trained
    res = rec["results"]
    assert rec["status"] == "ok" and res["convention"] == conv.citation()
    ds = res["dataset"]
    assert Path(ds["root"]) == demos and not ds["synthetic"] and ds["episodes"] == 2
    assert res["reload_check"]["ok"]
    # one identity: trainer record = local path = export bundle
    sha = res["checkpoint"]["checkpoint_sha256"]
    assert lerobot_ckpt.checkpoint_identity(ckpt)["checkpoint_sha256"] == sha
    assert lp.checkpoint_identity(ckpt)["checkpoint_sha256"] == sha
    assert X.checkpoint_identity(ckpt)["sha256"] == sha
    assert set(X.checkpoint_identity(ckpt)["files"]) == set(res["checkpoint"]["checkpoint_files"])


def _local_cfg(ckpt: Path, family: str, sha: str) -> dict[str, Any]:
    from test_local_policy import _cfg

    return _cfg(
        ckpt,
        family=family,
        checkpoint_sha256=sha,
        request_timeout_ms=5000,  # SmolVLA on the CPU: ~0.9 s per chunk (the Mac uses MPS)
    )


def test_trained_checkpoint_runs_one_deploy_tick(trained: tuple[str, dict[str, Any], Path]) -> None:
    """tiny train -> save -> load in the local policy -> one DeployLoop tick on the fakes."""
    from test_local_policy import local_rig

    family, rec, ckpt = trained
    sha = rec["results"]["checkpoint"]["checkpoint_sha256"]
    cfg = _local_cfg(ckpt, family, sha)
    cfg["deploy"] = {**cfg["deploy"], "max_chunk_age_ms": 6000}
    with local_rig(cfg) as r:
        info = r.client.info()
        lp.check_local(info, r.dcfg, r.spec)
        assert info["checkpoint_sha256"] == sha and info["chunk_len"] == conv.CHUNK_SIZE[family]
        if family == "smolvla":
            assert r.policy.rename_map == conv.SMOLVLA_RENAME_MAP  # saved by the trainer
        n0 = len(r.fol.sent)
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=1,
            info=info,
            camera_calibrated=False,
            max_steps=1,
        )
        assert end == "completed"
        (t,) = trials
        assert t["policy_started"] and t["decisions"] == 1 and r.client.requests == 1
        assert len(r.fol.sent) > n0  # home, then the policy's write reached the bus
        if family == "act":  # deterministic: the deploy path = the trainer's predict_chunk
            images = {n: r.cams[n].image for n in conv.CAMERA_NAMES}
            state = np.asarray([0.0, 0.1, 0.0, 0.2, 0.0, 0.3])
            got = r.policy.act(images, state, conv.DEMO_TASK_STRING)
            pol, pre, post = lerobot_ckpt.load_checkpoint(ckpt, "cpu")
            ref = lerobot_ckpt.predict_chunk(
                pol, pre, post, conv.policy_observation(images, state, conv.DEMO_TASK_STRING)
            )
            np.testing.assert_allclose(got, conv.action_targets(ref), atol=1e-5)


def test_real_eval_local_capture_and_export(
    trained: tuple[str, dict[str, Any], Path],
    demos: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family, rec, ckpt = trained
    sha = rec["results"]["checkpoint"]["checkpoint_sha256"]
    real_load = ev.load_config

    def load_config(preset: str, config: Any = None) -> dict[str, Any]:
        cfg: dict[str, Any] = real_load(preset, config)
        cfg["local_policy"] = {
            **cfg["local_policy"],
            "device": "cpu",  # deterministic and CI-safe (the Mac default is MPS)
            "warmup": 0,
            "request_timeout_ms": 5000,  # SmolVLA on the CPU
        }
        cfg["deploy"] = {**cfg["deploy"], "max_chunk_age_ms": 6000}  # > the deadline (fake clock)
        return cfg

    monkeypatch.setattr(ev, "load_config", load_config)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    runs = tmp_path / "runs"
    run_id = rec["run_id"]  # the training run: the export's training_run_id
    record = ev.main(
        [
            "--preset", "real", "--mock", "--fake-clock", "--max-steps", "4",
            "--task", "PickPlace", "--trials", "1", "--labeler", "mock-labeler",
            "--backend", "local", "--family", family, "--training-run-id", str(run_id),
            "--checkpoint-dir", str(ckpt), "--runs-dir", str(runs),
        ]
    )  # fmt: skip
    res = json.loads(Path(record).read_text())["results"]
    info, st = res["server_info"], res["session"]
    assert info["transport"] == lp.TRANSPORT and info["checkpoint_sha256"] == sha
    assert st["backend"] == "local" and st["policy_identity"]["checkpoint_sha256"] == sha
    assert st["policy_identity"]["family_id"] == X.FAMILIES[family]
    trials = res["trials"]
    assert len(trials) == 3 and {t["cell"] for t in trials} == set(PR.CELLS)
    assert all(t["policy_started"] and t["policy_id"] == str(ckpt) for t in trials)
    assert res["capture"]["episodes"] == 3
    # every trial is one episode of the evaluation dataset, both cameras, the convention's schema
    ds_root = Path(res["capture"]["dataset"])
    conv.check_dataset_info(json.loads((ds_root / "meta" / "info.json").read_text()))
    from src.data.recorder import load_demo_dataset

    ds = load_demo_dataset(f"local/so101-real-eval-{st['session_id']}", ds_root)
    assert ds.meta.total_episodes == 3
    item = ds[0]
    for key in conv.IMAGE_KEYS.values():
        assert tuple(item[key].shape) == (
            conv.IMAGE_CHANNELS,
            conv.IMAGE_HEIGHT,
            conv.IMAGE_WIDTH,
        )
    # the eval bundle of THAT session and the demo bundle of THOSE demos pass the importer mirror
    out = tmp_path / "eval.export"
    argv = ["--preset", "real", "eval", "--session", Path(record).parent.name]
    index = xb.main([*argv, "--runs-dir", str(runs), "--out", str(out), "--allow-mock"])
    assert index["episodes"] == 3
    decl = json.loads((out / X.DECLARATION_FILE).read_text())
    assert decl["policy"]["checkpoint_id"] == f"sha256:{sha}"
    assert decl["policy"]["family_id"] == X.FAMILIES[family] and decl["labels"] is None
    assert decl["policy"]["training_run_id"] == run_id
    X.validate_bundle(out, ds_root)  # (the builders validate too; explicit here)
    demo_out = tmp_path / "demo.export"
    assert X.build_demo_bundle(demos, demo_out)["episodes"] == 2
    X.validate_bundle(demo_out, demos)


def test_backend_choice() -> None:
    """real-0.1 on the arm runs the policy in process; local is real-0.1 only and needs a
    checkpoint; the mock defaults to the gRPC MockPolicy. Hardware refusals come after the
    presence gate (``real_hardware_problems``)."""
    import argparse

    def ns(**kw: Any) -> argparse.Namespace:
        base = {"backend": None, "hardware": False, "checkpoint_dir": None}
        return argparse.Namespace(**{**base, **kw})

    ck = Path("ckpt")
    assert ev.resolve_backend(ns(), real=True) == "grpc"  # mock: MockPolicy
    assert ev.resolve_backend(ns(hardware=True), real=True) == "local"
    assert ev.resolve_backend(ns(backend="local", checkpoint_dir=ck), real=True) == "local"
    assert ev.resolve_backend(ns(hardware=True), real=False) == "grpc"  # protocol 0.2
    for kw, real, msg in (
        ({"backend": "local", "checkpoint_dir": ck}, False, "real-0.1 only"),
        ({"backend": "local"}, True, "--checkpoint-dir"),
    ):
        with pytest.raises(SystemExit, match=msg):
            ev.resolve_backend(ns(**kw), real=real)
    from robot_fakes import real_cfg

    settings = ev.new_settings(
        ev_args := argparse.Namespace(
            trials=None, protocol=PR.PROTOCOL_VERSION, schedule="eval", task=PR.TASK,
            cells=None, slot_seed=None, label="x", labeler="me", family=None,
            training_run_id=None, checkpoint_dir=None, no_capture=False, backend="grpc",
            hardware=True,
        ),
        real_cfg(),
    )  # fmt: skip
    assert ev_args.backend == "grpc" and settings["backend"] == "grpc"
    problems = ev.real_hardware_problems(real_cfg(), settings)
    assert any("--backend local" in p for p in problems)
