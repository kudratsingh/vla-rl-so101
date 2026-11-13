"""In-process policy (src/robot/local_policy.py) through the REAL DeployLoop on the fakes.

A tiny, freshly initialised ACT (convention features, chunk 50) on the CPU; the real driver,
safety filter, deadman and run_trials; a FakeFollower arm and a CameraRig of two fake cameras on
one FakeClock. No serial port, no camera, no network.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from robot_fakes import FakeClock, FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.robot import convention as conv
from src.robot import local_policy as lp
from src.robot.camera import CameraFault, CameraRig, Frame
from src.robot.client import (
    DeployLoop,
    ScriptedOperator,
    ServerFault,
    ServerMismatch,
    run_trials,
)
from src.robot.driver import So101Driver
from src.robot.safety import make_safety
from src.robot.servo_health import monitor_from_config
from src.robot.units import JOINT_NAMES, UnitMap

TEST_HOME = [0.0, 0.1, 0.0, 0.2, 0.0, 0.3]  # inside real.yaml's box under the toy FK
TINY = {
    "dim_model": 16,
    "n_heads": 2,
    "dim_feedforward": 32,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_vae": False,
}


class Cam:
    """A fake ``FrameReader`` stamped with the shared fake clock."""

    def __init__(self, rgb: tuple[int, int, int], clock: FakeClock):
        self.image = np.empty(conv.IMAGE_SHAPE_HWC, np.uint8)
        self.image[...] = rgb
        self.clock = clock
        self.fail = False

    def connect(self) -> None:
        pass

    def read(self) -> Frame:
        if self.fail:
            raise CameraFault("timed out")
        return Frame(self.image, self.clock())

    def disconnect(self) -> None:
        pass


@pytest.fixture(scope="module")
def act_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny untrained ACT whose (placeholder) action stats sit at TEST_HOME."""
    out = tmp_path_factory.mktemp("act") / "pretrained_model"
    return lp.init_act_checkpoint(out, pose=TEST_HOME, pose_std_rad=0.01, **TINY)


def _cfg(base: Path, **local: Any) -> dict[str, Any]:
    cfg = real_cfg()
    cfg["deploy"] = {**cfg["deploy"], "home_qpos": TEST_HOME}
    cfg["local_policy"] = {
        **cfg["local_policy"],
        "family": "act",
        "base": str(base),
        "device": "cpu",
        "warmup": 0,
        **local,
    }
    return cfg


@contextmanager
def local_rig(cfg: dict[str, Any], policy: lp.LocalPolicy | None = None) -> Iterator[Any]:
    clock = FakeClock()
    spec = lp.LocalPolicySpec.from_config(cfg)
    policy = policy or lp.load_local_policy(spec)
    dcfg = lp.local_deploy_config(cfg, policy)
    fol = FakeFollower()
    arm = So101Driver(
        fol, UnitMap(), make_safety(cfg, toy_fk, clock=clock), clock=clock, sleep=clock.sleep
    )
    arm.connect()
    cams = {"front": Cam((30, 60, 200), clock), "wrist": Cam((200, 60, 30), clock)}
    client = lp.LocalPolicyClient(policy, timeout_s=dcfg.request_timeout_s, clock=clock)
    loop = DeployLoop(
        arm, CameraRig(cams), lp.local_pipe, client, dcfg, clock=clock, sleep=clock.sleep
    )
    try:
        yield SimpleNamespace(
            clock=clock, fol=fol, arm=arm, cams=cams, client=client, loop=loop, dcfg=dcfg,
            spec=spec, policy=policy,
        )  # fmt: skip
    finally:
        client.close()


def test_tiny_act_runs_through_the_real_deploy_loop(act_dir: Path) -> None:
    with local_rig(_cfg(act_dir)) as r:
        info = r.client.info()
        lp.check_local(info, r.dcfg, r.spec)
        assert info["checkpoint_sha256"] == lp.checkpoint_identity(act_dir)["checkpoint_sha256"]
        # an untrained ACT has no trainer provenance: it cites no convention (the mock allows it)
        assert info["convention"] is None and info["chunk_len"] == 50
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=1,
            info=info,
            camera_calibrated=False,
            max_steps=12,
        )
    assert end == "completed"
    (t,) = trials
    assert t["policy_started"] and t["outcome"] == "success" and t["stop_reason"] == "max_steps"
    assert t["steps"] == 12 and t["decisions"] == 1  # one 50-step chunk covers 12 writes
    assert t["policy_id"] == str(act_dir)
    assert t["safety"]["status_counts"].get("rejected", 0) == 0
    # every policy write reached the bus in LeRobot units (degrees), near the stats' pose
    sent = r.fol.sent[-13:-1]  # the 12 policy writes before the final hold
    deg = np.array([[a[f"{n}.pos"] for n in JOINT_NAMES[:5]] for a in sent])
    assert np.all(np.abs(deg - np.degrees(TEST_HOME[:5])) < 5.0)


def test_safety_clip_applies_to_local_policy_actions(tmp_path: Path) -> None:
    far = list(TEST_HOME)
    far[0] += math.radians(30.0)  # shoulder_pan 30 deg away: max_joint_step_deg is 4
    ckpt = lp.init_act_checkpoint(tmp_path / "far", pose=far, pose_std_rad=1e-4, **TINY)
    with local_rig(_cfg(ckpt)) as r:
        r.loop.go_home()
        n0 = len(r.fol.sent)
        res = r.loop.run_trial("pick up the red cube and place it in the bowl", max_steps=5)
    assert res.steps == 5 and res.safety_status["clipped"] == 5
    kinds = res.interventions_by_kind
    assert kinds["step_clip"] + kinds["velocity_cap"] >= 5
    pan = [a["shoulder_pan.pos"] for a in r.fol.sent[n0 - 1 : n0 + 5]]
    steps = np.diff(pan)
    # never more than 4 deg per write (the first write comes 0 s after home on the fake clock, so
    # the velocity cap allows no motion), always towards the 30 deg target, never reaching it
    assert np.all(np.abs(steps) <= 4.0 + 1e-6) and np.all(steps >= 0) and np.all(steps[1:] > 0)
    assert pan[-1] - pan[0] < 30.0


def test_checkpoint_hash_refusal(act_dir: Path, tmp_path: Path) -> None:
    ident = lp.checkpoint_identity(act_dir)
    assert set(ident["checkpoint_files"]) >= {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    assert any("normalizer" in f for f in ident["checkpoint_files"])  # the stats are hashed
    wrong = "0" * 64
    with pytest.raises(lp.CheckpointMismatch, match="checkpoint_sha256"):
        lp.load_local_policy(lp.LocalPolicySpec.from_config(_cfg(act_dir, checkpoint_sha256=wrong)))
    # the right hash loads; check_local refuses the same policy against another expected hash
    cfg = _cfg(act_dir, checkpoint_sha256=ident["checkpoint_sha256"])
    with local_rig(cfg) as r:
        info = r.client.info()
        lp.check_local(info, r.dcfg, r.spec)
        other = lp.LocalPolicySpec.from_config(_cfg(act_dir, checkpoint_sha256=wrong))
        with pytest.raises(ServerMismatch, match="checkpoint_sha256"):
            lp.check_local(info, r.dcfg, other)
    # changing any loaded file (here: the normalization stats) changes the hash
    import shutil

    copy = Path(shutil.copytree(act_dir, tmp_path / "copy"))
    stats = next(copy.glob("policy_preprocessor_step_*normalizer*.safetensors"))
    stats.write_bytes(stats.read_bytes() + b"\0")
    assert lp.checkpoint_identity(copy)["checkpoint_sha256"] != ident["checkpoint_sha256"]


class SlowPolicy:
    """A ``LocalPolicy`` stand-in whose chunk takes ``delay_s`` real seconds."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.release = threading.Event()
        self.chunk_len, self.action_dim = 50, 6
        self.spec = SimpleNamespace(base="slow", family="act")
        self.identity = {"checkpoint_sha256": "x", "checkpoint_revision": None}
        self.device, self.last_model_ms = "cpu", None

    def act(self, images: Any, state: Any, instruction: str) -> np.ndarray:
        self.release.wait(self.delay_s)
        return np.tile(np.asarray(state, dtype=np.float64), (50, 1))

    def describe(self) -> dict[str, Any]:
        return {"kind": "slow"}


def test_deadline_trips_the_deadman_and_a_hung_call_blocks_the_next(act_dir: Path) -> None:
    slow = SlowPolicy(delay_s=5.0)
    client = lp.LocalPolicyClient(slow, timeout_s=0.05)  # type: ignore[arg-type]
    images = {n: np.zeros(conv.IMAGE_SHAPE_HWC, np.uint8) for n in conv.CAMERA_NAMES}
    try:
        t0 = time.perf_counter()
        with pytest.raises(ServerFault) as e:
            client.act(images, np.zeros(6), "x")
        assert e.value.kind == "timeout" and time.perf_counter() - t0 < 2.0
        with pytest.raises(ServerFault) as e2:  # still running: never a second inference
            client.act(images, np.zeros(6), "x")
        assert e2.value.kind == "error"
    finally:
        slow.release.set()
        client.close()
    # in the loop: a timeout is the deadman (HOLD) before the policy started
    with local_rig(_cfg(act_dir)) as r:
        r.loop.go_home()
        hung = SlowPolicy(delay_s=5.0)
        r.loop.client = lp.LocalPolicyClient(hung, timeout_s=0.05, clock=r.clock)  # type: ignore[arg-type]
        try:
            res = r.loop.run_trial("x", max_steps=3)
        finally:
            hung.release.set()
            r.loop.client.close()  # type: ignore[attr-defined]
    assert res.stop_reason == "server_timeout" and not res.policy_started
    assert r.arm.deadman.tripped and res.deadman_response == "hold"


def test_lost_wrist_frame_trips_the_deadman(act_dir: Path) -> None:
    with local_rig(_cfg(act_dir)) as r:
        r.loop.go_home()
        r.cams["wrist"].fail = True
        res = r.loop.run_trial("x", max_steps=3)
    assert res.stop_reason == "deadman" and res.steps == 0 and r.arm.deadman.tripped


def test_servo_fault_during_a_local_trial_cuts_torque(act_dir: Path) -> None:
    with local_rig(_cfg(act_dir)) as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        r.loop.go_home()

        def overheat(timing: dict[str, float]) -> None:
            r.fol.temperature_c["gripper"] = 66.0

        r.loop.on_decision = overheat
        res = r.loop.run_trial("x", max_steps=40)
        n = len(r.fol.sent)
    assert res.stop_reason == "servo_fault" and res.deadman_response == "torque_off"
    assert res.policy_started and not r.fol.bus.torque_enabled and len(r.fol.sent) == n


def test_rename_map_for_smolvla_base_and_trained_checkpoints(tmp_path: Path) -> None:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache("lerobot/smolvla_base", "config.json")
    if not isinstance(cached, str):
        pytest.skip("lerobot/smolvla_base is not in the local Hugging Face cache")
    from lerobot.configs.policies import PreTrainedConfig

    base = Path(cached).parent
    pcfg = PreTrainedConfig.from_pretrained(base)
    rename, override = lp._rename_for(pcfg, base)
    assert rename == conv.SMOLVLA_RENAME_MAP and override  # saved map is empty
    # a fine-tune that saved the convention's map keeps it, no override
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": lp.RENAME_STEP,
                        "config": {"rename_map": conv.SMOLVLA_RENAME_MAP},
                    }
                ]
            }
        )
    )
    assert lp._rename_for(pcfg, tmp_path) == (conv.SMOLVLA_RENAME_MAP, False)
    # a swapped map (front -> camera2) still lands on camera inputs, but a map onto keys the
    # policy does not have is refused
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": lp.RENAME_STEP,
                        "config": {
                            "rename_map": {conv.IMAGE_KEYS["front"]: "observation.images.top"}
                        },
                    }
                ]
            }
        )
    )
    with pytest.raises(lp.CheckpointMismatch):
        lp._rename_for(pcfg, tmp_path)


def test_latency_script_tiny_preset(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "wo07_local_latency", REPO_ROOT / "scripts" / "wo07_local_latency.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    record = mod.main(["--preset", "tiny", "--ticks", "12", "--runs-dir", str(tmp_path)])
    lat = json.loads(Path(record).read_text())["results"]["latency"]
    assert lat["ticks"] == 12 and lat["chunk_ms"]["n"] >= 1 and lat["command_gap_ms"]["n"] >= 10
    assert lat["fresh_init"] and "EXCLUDES camera capture" in lat["label"]


def _with_provenance(src: Path, dst: Path, **prov: Any) -> Path:
    import shutil

    from src.policy import lerobot_ckpt

    out = Path(shutil.copytree(src, dst))
    lerobot_ckpt.write_provenance(
        out,
        {
            "training_run_id": "run-x",
            "convention": conv.citation(),
            "unit_map": json.loads(json.dumps(dataclasses.asdict(UnitMap()))),
            **prov,
        },
    )
    return out


def test_provenance_is_checked_against_the_live_setup(act_dir: Path, tmp_path: Path) -> None:
    """Review conv-1: the checkpoint's convention and unit_map (inside its hash) must match the
    live ones; the real arm also refuses a checkpoint without a provenance."""
    live = dataclasses.asdict(UnitMap())
    good = _with_provenance(act_dir, tmp_path / "good")
    assert "vla_rl_so101.json" in lp.checkpoint_identity(good)["checkpoint_files"]  # hashed
    spec = lp.LocalPolicySpec.from_config(_cfg(good))
    strict = dataclasses.replace(spec, require_provenance=True)
    policy = lp.load_local_policy(strict, units=live)
    with local_rig(_cfg(good), policy) as r:
        info = r.client.info()
        assert info["convention"] == conv.citation()
        lp.check_local(info, r.dcfg, strict)
    # wrist_flex mirrored after the demos were recorded: refused before the weights load
    mirrored = {**live, "arm_sign": [1.0, 1.0, 1.0, -1.0, 1.0]}
    with pytest.raises(lp.CheckpointMismatch, match="robot.units"):
        lp.load_local_policy(spec, units=mirrored)
    # another convention version / hash
    other = _with_provenance(
        act_dir, tmp_path / "conv", convention={**conv.citation(), "hash": "0" * 64}
    )
    with pytest.raises(lp.CheckpointMismatch, match="trained under convention"):
        lp.load_local_policy(lp.LocalPolicySpec.from_config(_cfg(other)), units=live)
    # no provenance: the mock runs it, the real arm refuses it
    with pytest.raises(lp.CheckpointMismatch, match="vla_rl_so101.json"):
        lp.load_local_policy(
            dataclasses.replace(
                lp.LocalPolicySpec.from_config(_cfg(act_dir)), require_provenance=True
            ),
            units=live,
        )


def test_strict_device_refuses_the_cpu_fallback(
    act_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review F2: on the real arm an unavailable MPS is refused, never a silent CPU run."""
    monkeypatch.setattr(lp.torch.backends.mps, "is_available", lambda: False)
    spec = dataclasses.replace(
        lp.LocalPolicySpec.from_config(_cfg(act_dir, device="mps")), strict_device=True
    )
    with pytest.raises(lp.DeviceUnavailable, match="not available"):
        lp.load_local_policy(spec)
    loose = dataclasses.replace(spec, strict_device=False)
    assert lp.load_local_policy(loose).device == "cpu"  # the mock / latency script: logged only


def test_resolve_checkpoint_never_fetches_a_mistyped_path(tmp_path: Path, monkeypatch: Any) -> None:
    """Review F4: a path-like spec that is not a directory is refused, not downloaded."""
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("Hub fetch")
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    for spec in ("checkpoints/smolvla", "./ckpt", "/no/such/dir", "~/nope", "a/b/c"):
        with pytest.raises(FileNotFoundError, match="not a directory"):
            lp.resolve_checkpoint(spec)


def test_a_fresh_process_resolves_a_smolvla_config(tmp_path: Path) -> None:
    """A load in a new process must register LeRobot's policy config types first (the latency
    script failed with "Couldn't find a choice class for 'smolvla'" when nothing else had)."""
    import subprocess
    import sys

    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"type": "smolvla"}))
    (tmp_path / "model.safetensors").write_bytes(b"x")
    spec = f"lp.LocalPolicySpec('smolvla', {str(tmp_path)!r}, 'cpu', None, 1.0, None, 0)"
    code = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from src.robot import local_policy as lp\n"
        f"try:\n    lp.load_local_policy({spec})\n"
        "except Exception as e:\n    print(type(e).__name__, e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=120
    )
    assert "choice class" not in out.stdout + out.stderr
