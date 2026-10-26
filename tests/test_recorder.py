from pathlib import Path

import numpy as np
import torch

from src.data.recorder import (
    ACTION_KEY,
    IMAGE_KEY,
    STATE_KEY,
    DemoRecorder,
    load_demo_dataset,
    read_sidecar,
    select_episodes,
)
from src.env import CONTROL_HZ, JOINT_NAMES


def test_two_episodes_roundtrip_with_torchcodec(tmp_path: Path) -> None:
    root = tmp_path / "demos"
    rec = DemoRecorder("local/test-demos", root, image_hw=(32, 32))
    lengths = {"PickPlace": 5, "Push": 4}
    for ep, (family, n) in enumerate(lengths.items()):
        for t in range(n):
            img = np.full((32, 32, 3), 40 * t, dtype=np.uint8)
            state = torch.full((6,), float(t))
            action = np.full(6, t + 0.5, dtype=np.float32)
            rec.add_frame(img, state, action, f"do {family.lower()}")
        rec.save_episode(
            {"task_family": family, "seed": 100 + ep, "cell": "seen", "oracle_success": True}
        )
    # a discarded episode leaves no trace
    rec.add_frame(np.zeros((32, 32, 3), np.uint8), np.zeros(6), np.zeros(6), "discard me")
    rec.discard_episode()
    rec.finalize()

    ds = load_demo_dataset("local/test-demos", root)
    assert ds.meta.info["codebase_version"] == "v3.0"
    assert ds.meta.robot_type == "so101_follower"
    assert ds.fps == CONTROL_HZ
    assert ds.num_episodes == 2 and ds.num_frames == 9
    assert ds.meta.video_keys == [IMAGE_KEY]
    assert ds.features[STATE_KEY]["names"] == list(JOINT_NAMES)
    assert ds.features[ACTION_KEY]["names"] == list(JOINT_NAMES)

    item = ds[3]  # episode 0, frame 3: decoded from AV1 video via torchcodec
    assert item[IMAGE_KEY].shape == (3, 32, 32) and item[IMAGE_KEY].dtype == torch.float32
    assert abs(float(item[IMAGE_KEY].mean()) * 255 - 120) < 6
    assert item[STATE_KEY].shape == (6,) and float(item[STATE_KEY][0]) == 3.0
    assert float(item[ACTION_KEY][0]) == 3.5
    assert item["task"] == "do pickplace"
    assert ds[5]["task"] == "do push" and int(ds[5]["episode_index"]) == 1

    rows = read_sidecar(root)
    assert [r["episode_index"] for r in rows] == [0, 1]
    assert rows[1] == {
        "episode_index": 1,
        "task_family": "Push",
        "instruction": "do push",
        "seed": 101,
        "cell": "seen",
        "oracle_success": True,
        "num_frames": 4,
    }
    assert select_episodes(root, "Push", 1) == [1]
    assert not (root / "images").exists()
