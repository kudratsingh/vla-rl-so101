"""The frozen real-arm convention (src/robot/convention.py) and the recorder check.

The recorder check records a short episode through the REAL teleop recorder path
(``scripts/wo06_teleop_record.py``: ``Session`` + ``record_session`` + ``create_dataset``, the real
``CameraRig``, ``So101Driver`` and safety filter) on the fake follower and two fake cameras, reads
it back as a LeRobotDataset (AV1 + torchcodec), runs it through the inference preprocessing
(``convention.policy_observation``) and checks the camera mapping, joint units and order, and the
action convention.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
import torch
from robot_fakes import FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.data.recorder import load_demo_dataset
from src.robot import convention as conv
from src.robot.camera import CameraFault, CameraRig, Frame
from src.robot.driver import DeadmanTripped, So101Driver
from src.robot.safety import make_safety
from src.robot.units import ARM_JOINTS, JOINT_NAMES, UnitMap

_spec = importlib.util.spec_from_file_location(
    "wo06_teleop_record_conv", REPO_ROOT / "scripts" / "wo06_teleop_record.py"
)
assert _spec is not None and _spec.loader is not None
teleop = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teleop)

# Pinned. A change to spec() fails here on purpose: bump CONVENTION_VERSION and pin the new hash
# (demos recorded earlier keep citing the hash they were recorded under).
FROZEN = {"1.0": "b8701c5784f73dc0fec0fc12bf695be8c7b7db826c2f8faae460716fe1c912b5"}

FRONT_RGB = (30, 60, 200)  # blue-dominant
WRIST_RGB = (200, 60, 30)  # red-dominant


class ColorCam:
    """One fake camera (``FrameReader``): a solid colour, a fresh stamp per read."""

    def __init__(self, rgb: tuple[int, int, int]) -> None:
        self.image = np.empty(conv.IMAGE_SHAPE_HWC, np.uint8)
        self.image[...] = rgb
        self.fail = False
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def read(self) -> Frame:
        if self.fail:
            raise CameraFault("timed out")
        return Frame(self.image.copy(), time.perf_counter())

    def disconnect(self) -> None:
        self.connected = False


class MovingLeader:
    """Leader arm in LeRobot units; every joint >= 10 deg from the others and moving at its own
    rate (the order and units check). toy_fk puts this pose inside the workspace box."""

    BASE: ClassVar[dict[str, float]] = {
        "shoulder_pan": 10.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 30.0,
        "wrist_flex": 40.0,
        "wrist_roll": -50.0,
        "gripper": 40.0,
    }
    RATE: ClassVar[dict[str, float]] = {
        "shoulder_pan": 0.30,
        "shoulder_lift": -0.20,
        "elbow_flex": 0.25,
        "wrist_flex": -0.15,
        "wrist_roll": 0.10,
        "gripper": 0.50,
    }

    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []

    def get_action(self) -> dict[str, float]:
        k = len(self.history)
        pose = {f"{n}.pos": self.BASE[n] + k * self.RATE[n] for n in JOINT_NAMES}
        self.history.append(pose)
        return dict(pose)


def _rig() -> tuple[CameraRig, dict[str, ColorCam]]:
    cams = {"front": ColorCam(FRONT_RGB), "wrist": ColorCam(WRIST_RGB)}
    return CameraRig(cams), cams


def _session(leader: Any, rig: CameraRig) -> tuple[Any, FakeFollower]:
    fol = FakeFollower()
    arm = So101Driver(fol, UnitMap(), make_safety(real_cfg(), toy_fk))
    arm.connect()
    sess = teleop.Session(arm, leader, rig, UnitMap(), read=lambda p, _t: input(p))
    return sess, fol


def test_convention_is_frozen() -> None:
    assert conv.CONVENTION_VERSION in FROZEN
    assert conv.convention_hash() == FROZEN[conv.CONVENTION_VERSION]
    s = conv.spec()
    assert s["joint_names"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert s["units"] == {n: "rad" for n in JOINT_NAMES}  # vishwaroop SO101_EXPORT.md §2
    assert set(s["units"].values()) <= {"deg", "rad", "normalized_m100_100", "normalized_0_100"}
    assert conv.SMOLVLA_RENAME_MAP == {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    assert (conv.FPS, conv.CAMERA_FPS, conv.IMAGE_SHAPE_HWC) == (15, 30, (480, 640, 3))
    assert conv.CHUNK_SIZE == {"smolvla": 50, "act": 50}
    assert conv.DEMO_TASK_STRING == "pick up the red cube and place it in the bowl"
    assert conv.citation()["hash"] == conv.convention_hash()


def test_smolvla_base_expects_the_renamed_keys() -> None:
    """The rename targets are input features of smolvla_base (its config.json, if cached)."""
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache("lerobot/smolvla_base", "config.json")
    if not isinstance(path, str):
        pytest.skip("lerobot/smolvla_base config.json not in the local HF cache")
    feats = json.loads(Path(path).read_text())["input_features"]
    assert set(conv.SMOLVLA_RENAME_MAP.values()) <= set(feats)
    assert feats[conv.STATE_KEY]["shape"] == [len(JOINT_NAMES)]


def test_real_config_conforms_and_departures_are_named() -> None:
    cfg = real_cfg()
    assert conv.check_config(cfg) == []
    bad = real_cfg()
    bad["cameras"]["wrist"]["enabled"] = False
    bad["cameras"]["front"]["width"] = 1280
    bad["robot"]["control_hz"] = 30
    bad["robot"]["units"]["gripper_open_rad"] = 1.5
    problems = "\n".join(conv.check_config(bad))
    for what in ("cameras.wrist", "cameras.front streams 1280", "control_hz", "gripper_open"):
        assert what in problems


def test_a_frame_without_both_cameras_is_never_recorded() -> None:
    img = np.zeros(conv.IMAGE_SHAPE_HWC, np.uint8)
    q = np.zeros(6)
    with pytest.raises(conv.ConventionError, match="missing \\['wrist'\\]"):
        conv.recording_frame({"front": img}, q, q, "t")
    with pytest.raises(conv.ConventionError, match="missing \\['front'\\]"):
        conv.policy_observation({"wrist": img}, q, "t")
    with pytest.raises(conv.ConventionError, match="uint8"):  # a 224 crop is not this convention
        conv.recording_frame({"front": img, "wrist": img[:224, :224]}, q, q, "t")

    # a wrist camera fault on the tick: no frame, the deadman holds, nothing reaches the dataset
    rig, cams = _rig()
    sess, fol = _session(MovingLeader(), rig)
    sess.tick()
    cams["wrist"].fail = True
    with pytest.raises(CameraFault, match="wrist camera"):
        rig.read()
    with pytest.raises(DeadmanTripped, match="no camera frame"):
        sess.tick()
    assert fol.bus.torque_enabled and fol.sent[-1] == pytest.approx(fol.pos)  # held


def test_recorder_check_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answers = ["", "y", ""]  # scene ENTER, keep the episode, ENTER at the support prompt
    monkeypatch.setattr("builtins.input", lambda _p="": answers.pop(0))
    rig, _ = _rig()
    leader = MovingLeader()
    sess, fol = _session(leader, rig)
    steps: list[Any] = []  # (StepResult, the LeRobot dict written to the bus on that tick)
    real_step = sess.arm.step

    def step(*a: Any, **k: Any) -> Any:
        out = real_step(*a, **k)
        steps.append((out, dict(fol.sent[-1])))
        return out

    monkeypatch.setattr(sess.arm, "step", step)
    repo_id, root = "local/convention-check", tmp_path / "ds"
    dataset = teleop.create_dataset(repo_id, root)
    instr = teleop.protocol_instructions(["PickPlace"])
    with rig:
        saved = teleop.record_session(sess, dataset, ["PickPlace"], instr, 1, 0.5)
    dataset.finalize()
    assert saved == 1 and not answers

    info = json.loads((root / "meta" / "info.json").read_text())
    assert conv.check_dataset_info(info) == []  # v3.0, fps 15, both cameras 480x640, names
    ds = load_demo_dataset(repo_id, root)
    n = len(ds)
    assert n >= 5 and sorted(ds.meta.video_keys) == sorted(conv.IMAGE_KEYS.values())
    rec_steps, rec_leader = steps[-n:], leader.history[-n:]  # the recorded ticks, in order

    live = rig.read()  # what the inference path would build from the same cameras
    live_batch = conv.policy_observation(
        live.images, rec_steps[0][0].measured, conv.DEMO_TASK_STRING
    )
    units = UnitMap()
    for i in range(n):
        sample = ds[i]
        assert sample["task"] == conv.DEMO_TASK_STRING
        # state / action stored bit-exact: the measured pose and the command actually written
        state, action = sample[conv.STATE_KEY].numpy(), sample[conv.ACTION_KEY].numpy()
        out, written = rec_steps[i]
        np.testing.assert_array_equal(state, out.measured.astype(np.float32))
        np.testing.assert_array_equal(action, out.safety.command.astype(np.float32))
        # UNITS both ways: the stored action is exactly what went to the bus (deg / 0-100)
        assert conv.canonical_to_lerobot(units, action) == pytest.approx(written, abs=1e-4)
        np.testing.assert_allclose(conv.lerobot_to_canonical(units, written), action, atol=1e-6)
        # ORDER: index j holds joint j (the joints are >= 10 deg apart; the safety filter's rate
        # limit may trim one tick's leader motion, < 1 deg)
        lead, tol = rec_leader[i], math.radians(1.0)
        for j, name in enumerate(ARM_JOINTS):
            assert action[j] == pytest.approx(math.radians(lead[f"{name}.pos"]), abs=tol)
        assert action[5] == pytest.approx(units.gripper_pct_to_rad(lead["gripper.pos"]), abs=tol)
        # ABSOLUTE targets, not deltas: the (perfectly tracking) follower then measures it
        if i + 1 < n:
            np.testing.assert_allclose(ds[i + 1][conv.STATE_KEY].numpy(), action, atol=1e-6)
        targets = conv.action_targets(sample[conv.ACTION_KEY])
        assert targets.shape == (1, 6) and targets.dtype == np.float64

        # the SAME preprocessing as inference; the wrist frame comes back under the wrist key
        batch = conv.policy_observation_from_sample(sample)
        assert set(batch) == set(live_batch)
        for key, value in batch.items():
            if key == conv.TASK_KEY:
                assert value == [conv.DEMO_TASK_STRING]
                continue
            assert value.shape == live_batch[key].shape and value.dtype == torch.float32
        front = batch[conv.IMAGE_KEYS["front"]][0].mean(dim=(1, 2))  # RGB means in [0, 1]
        wrist = batch[conv.IMAGE_KEYS["wrist"]][0].mean(dim=(1, 2))
        assert front[2] > front[0] + 0.3 and wrist[0] > wrist[2] + 0.3  # not swapped
        for name in conv.CAMERA_NAMES:
            key = conv.IMAGE_KEYS[name]
            assert float((batch[key] - live_batch[key]).abs().mean()) < 0.03  # AV1, lossy
        np.testing.assert_array_equal(batch[conv.STATE_KEY][0].numpy(), state)

    assert fol.bus.torque_enabled  # session ended in the hold + support prompt


def test_teleop_main_cites_the_convention(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys

    class Leader(MovingLeader):
        is_calibrated, is_connected = True, False

        def connect(self, calibrate: bool = False) -> None:
            self.is_connected = True

        def disconnect(self) -> None:
            self.is_connected = False

    class Dataset:
        num_episodes = 0

        def finalize(self) -> None:
            pass

    created: dict[str, Any] = {}

    def create(**kw: Any) -> Dataset:
        created.update(kw)
        return Dataset()

    def record_session(sess: Any, *_a: Any) -> int:
        assert isinstance(sess.cam, CameraRig) and sess.cam.names == conv.CAMERA_NAMES
        sess.session_end = "supported"
        return 0

    rig, _ = _rig()
    cfg = real_cfg()
    cfg["teleop"] = {**cfg["teleop"], "leader": {"type": "so101_leader"}}
    monkeypatch.setattr(teleop, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(teleop, "require_terminal", lambda _what: None)
    monkeypatch.setattr(teleop, "load_config", lambda *_a: cfg)
    monkeypatch.setattr(teleop, "load_sim_fk", lambda: toy_fk)
    monkeypatch.setattr(teleop, "make_leader", lambda _lc: Leader())
    monkeypatch.setattr(teleop.CameraRig, "from_config", classmethod(lambda cls, c, n: rig))
    monkeypatch.setattr(teleop, "record_session", record_session)
    monkeypatch.setattr(
        teleop.So101Driver,
        "from_config",
        classmethod(lambda cls, c, fk: cls(FakeFollower(), UnitMap(), make_safety(c, toy_fk))),
    )
    import lerobot.datasets.lerobot_dataset as lds

    monkeypatch.setattr(lds.LeRobotDataset, "create", staticmethod(create))
    monkeypatch.setattr(sys, "argv", ["wo06_teleop_record.py", "--preset", "real"])
    teleop.main()
    assert created["features"] == conv.dataset_features() and created["fps"] == conv.FPS
    root = tmp_path / cfg["teleop"]["dataset_root"] / cfg["teleop"]["hub_repo"].split("/")[-1]
    prov = json.loads((root / "meta" / "vla_rl_so101.json").read_text())
    assert prov["convention"] == conv.citation()
    assert prov["units"] == conv.DECLARED_UNITS
    assert set(prov["cameras"]) == {"front", "wrist"}

    # a config outside the convention is refused before the leader, the dataset or the follower
    cfg["cameras"]["wrist"] = {**cfg["cameras"]["wrist"], "enabled": False}
    monkeypatch.setattr(teleop, "make_leader", lambda _lc: pytest.fail("leader opened"))
    with pytest.raises(SystemExit, match="cameras.wrist is missing or disabled"):
        teleop.main()
