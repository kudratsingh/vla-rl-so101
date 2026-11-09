"""WO-07 trainer smoke (scripts/wo07_train.py, scripts/pod/wo07_train_job.sh) on CPU.

``--preset tiny`` for ACT and SmolVLA: a synthetic two-camera dataset in the frozen convention, a
few LeRobot training steps, save, reload, and the same chunk from the in-memory and the reloaded
policy (SmolVLA: the same explicit flow-matching noise). SmolVLA needs ``lerobot/smolvla_base`` in
the local HF cache and the ``smolvla`` extra (skipped otherwise, e.g. on CI). Plus the dataset
refusals and the pod job's dry run, which must never reach apt, the Hub or RunPod.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from src.robot import convention as conv
from src.runs import Run

REPO = Path(__file__).resolve().parents[1]
JOB = REPO / "scripts/pod/wo07_train_job.sh"


def _load() -> ModuleType:
    path = REPO / "scripts/wo07_train.py"
    spec = importlib.util.spec_from_file_location("wo07_train", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wo07_train"] = mod
    spec.loader.exec_module(mod)
    return mod


wo07 = _load()


def _main(monkeypatch: pytest.MonkeyPatch, runs: Path, *argv: str) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(wo07, "Run", functools.partial(Run, runs_dir=runs))
    monkeypatch.setattr(sys, "argv", ["wo07_train.py", *argv])
    rc = wo07.main()
    (record,) = runs.glob("*/record.json")
    return rc, json.loads(record.read_text())


def _check_record(rec: dict[str, Any], policy: str) -> None:
    res = rec["results"]
    assert rec["status"] == "ok" and rec["work_order"] == "WO-07"
    assert rec["config"]["policy"]["base"] == res["base"]  # policy.base from config
    assert res["convention"] == conv.citation()
    ds = res["dataset"]
    assert ds["synthetic"] and ds["episodes"] == 2 and ds["frames"] == 32
    assert len(ds["sha256"]) == 64 and ds["files"] > 0
    ck = res["checkpoint"]
    assert len(ck["checkpoint_sha256"]) == 64 and "model.safetensors" in ck["checkpoint_files"]
    assert ck["deleted"]  # tiny weights are plumbing only
    chk = res["reload_check"]
    assert chk["ok"] and chk["max_abs_diff_rad"] <= chk["atol_rad"]
    assert chk["chunk_shape"] == [1, conv.CHUNK_SIZE[policy], 6]
    assert chk["targets_shape"] == [conv.CHUNK_SIZE[policy], 6]


def test_tiny_act_train_save_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rc, rec = _main(monkeypatch, tmp_path, "--preset", "tiny", "--policy", "act")
    assert rc == 0
    _check_record(rec, "act")
    chk = rec["results"]["reload_check"]
    assert not chk["explicit_noise"] and chk["rename_map"] == {}  # ACT reads the dataset keys


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


@pytest.mark.skipif(not _smolvla_available(), reason="smolvla extra / smolvla_base not cached")
def test_tiny_smolvla_train_save_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # cached weights only, no Hub traffic
    rc, rec = _main(monkeypatch, tmp_path, "--preset", "tiny", "--policy", "smolvla")
    assert rc == 0
    _check_record(rec, "smolvla")
    chk = rec["results"]["reload_check"]
    # the SAME explicit noise for both calls; another draw really moves the chunk
    assert chk["explicit_noise"] and chk["other_noise_max_abs_diff_rad"] > 10 * chk["atol_rad"]
    # the convention's rename travels in the saved preprocessor (front -> camera1, wrist -> camera2)
    assert chk["rename_map"] == conv.SMOLVLA_RENAME_MAP == rec["results"]["rename_map"]


def test_dataset_outside_the_convention_is_refused(tmp_path: Path) -> None:
    ref = wo07.DatasetRef("local/refusals", tmp_path / "ds", None, True)
    wo07.build_synthetic_dataset(ref, episodes=1, frames=4, seed=0)
    info = wo07.check_dataset(ref.root, allow_rehearsal=False)
    assert info["frames"] == 4 and info["synthetic"]
    prov_path = ref.root / wo07.PROVENANCE
    prov = json.loads(prov_path.read_text())

    def refused(match: str) -> None:
        with pytest.raises(wo07.TrainSetupError, match=match):
            wo07.check_dataset(ref.root, allow_rehearsal=False)

    prov_path.write_text(json.dumps({**prov, "rehearsal": True}))
    refused("rehearsal")
    assert wo07.check_dataset(ref.root, allow_rehearsal=True)["rehearsal"]
    old = {**prov["convention"], "version": "0.9", "hash": "0" * 64}
    prov_path.write_text(json.dumps({**prov, "convention": old}))
    refused("cites convention")
    prov_path.unlink()
    refused("vla_rl_so101.json missing")
    prov_path.write_text(json.dumps(prov))
    info_path = ref.root / "meta" / "info.json"
    meta = json.loads(info_path.read_text())
    meta["fps"] = 30
    info_path.write_text(json.dumps(meta))
    refused("fps 30 != 15")
    with pytest.raises(wo07.TrainSetupError, match="no dataset"):
        wo07.check_dataset(tmp_path / "nowhere", allow_rehearsal=False)
    h1 = wo07.dataset_hash(ref.root)
    meta["fps"] = 15
    info_path.write_text(json.dumps(meta))
    assert wo07.dataset_hash(ref.root)["sha256"] != h1["sha256"]  # any byte moves the hash


def test_policy_config_follows_the_convention() -> None:
    cfg = wo07.resolve_config("real", None)
    for name in ("smolvla", "act"):
        block = wo07.policy_block(cfg, name)
        assert block["overrides"]["chunk_size"] == conv.CHUNK_SIZE[name]
    sm = wo07.policy_block(cfg, "smolvla")
    assert (sm["base"], sm["steps"], sm["batch_size"]) == ("lerobot/smolvla_base", 20000, 64)
    assert sm["overrides"]["num_steps"] == 10
    assert sm["overrides"]["resize_imgs_with_padding"] == [512, 512]
    act = wo07.policy_block(cfg, "act")
    assert (act["base"], act["steps"], act["batch_size"]) == ("act", 100000, 8)
    pcfg = wo07.make_policy_config("act", act, "cpu")
    assert (pcfg.chunk_size, pcfg.n_action_steps, pcfg.push_to_hub) == (50, 50, False)

    bad = json.loads(json.dumps(cfg))
    bad["policies"]["act"]["overrides"]["chunk_size"] = 100
    with pytest.raises(wo07.TrainSetupError, match="convention"):
        wo07.policy_block(bad, "act")
    bad["policies"]["act"]["overrides"] = {"chunk_size": 50, "chunk_sise": 1}
    with pytest.raises(wo07.TrainSetupError, match="chunk_sise"):
        wo07.make_policy_config("act", wo07.policy_block(bad, "act"), "cpu")
    with pytest.raises(wo07.TrainSetupError, match="real-arm"):
        wo07.resolve_config("sim", None)


def _stub_bin(tmp: Path) -> tuple[Path, Path]:
    """apt-get / curl / hf / runpodctl stubs that record any call and fail."""
    stubs, calls = tmp / "bin", tmp / "calls.log"
    stubs.mkdir()
    for name in ("apt-get", "curl", "hf", "runpodctl", "huggingface-cli"):
        (stubs / name).write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\nexit 99\n')
        (stubs / name).chmod(0o755)
    return stubs, calls


def test_pod_job_dry_run_passes_every_gate_offline(tmp_path: Path) -> None:
    stubs, calls = _stub_bin(tmp_path)
    before = set((REPO / "runs").glob("*-wo07-tiny-*"))
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "POLICY": "act",
        "PROBE_BATCHES": "2",
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "TMPDIR": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
    }
    env.pop("RUNPOD_POD_ID", None)
    try:
        p = subprocess.run(
            ["bash", str(JOB)],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    finally:
        for run in set((REPO / "runs").glob("*-wo07-tiny-*")) - before:
            subprocess.run(["rm", "-rf", str(run)], check=False)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-4000:]
    stages = [ln.split("stage: ")[1] for ln in p.stdout.splitlines() if "stage: " in ln]
    assert stages == ["setup", "dataset", "probe", "await-approval", "train", "upload", "done"]
    assert "[job] launch gate passed: pod=dryrun-" in out
    assert "[job] approved: DRY RUN: simulated owner go" in out
    assert "[probe] act batch 2:" in out and "[probe] full run act:" in out
    assert '"ok": true' in out  # the reload check of the (tiny) full run
    assert "[dry-run] would terminate pod dryrun-" in out
    assert not calls.exists(), calls.read_text()  # no apt, Hub or RunPod call
    assert not list(tmp_path.glob("wo07-dryrun.*"))  # sandbox removed


def test_pod_job_refuses_without_token_policy_or_revision() -> None:
    text = JOB.read_text()
    i = text.index("require_watchdogs || exit 8")
    assert text.index("pod_env || exit 9") < i < text.index("stage setup")
    assert text.index("trap finish EXIT") < i  # a refusal still uploads + deletes the pod
    assert text.index('await_approval "${APPROVAL_WAIT_S:-1800}" || exit 21') > text.index(
        "stage probe"
    )
    assert text.index("stage await-approval") < text.index("stage train")
    assert "DATASET_REVISION (Hub commit of the demo set) is required" in text
