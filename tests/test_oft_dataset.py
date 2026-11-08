"""SFT data path on a real (tiny) LeRobotDataset v3.0 built here; the WO-01 recorder module is
stood in by a fake until it reaches this branch (its loader adds torchcodec video decode only)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from src.config import load_config
from src.data.oft_dataset import (
    ACTION_KEY,
    IMAGE_KEY,
    STATE_KEY,
    ChunkedDemoDataset,
    all_actions,
    collate_sft,
    load_sft_frames,
    synthetic_frames,
)

TASKS = {"PickPlace": "pick up the red cube and place it in the bowl", "Push": "push the red cube"}
LENGTHS = [5, 4, 6, 3]  # two episodes per task


def _make_dataset(root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    features = {
        IMAGE_KEY: {
            "dtype": "image",
            "shape": (24, 24, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (6,), "names": names},
        ACTION_KEY: {"dtype": "float32", "shape": (6,), "names": names},
    }
    ds = LeRobotDataset.create(
        "local/oft-test",
        fps=15,
        features=features,
        root=root,
        robot_type="so101_follower",
        use_videos=False,
    )
    sidecar = []
    for ep, n in enumerate(LENGTHS):
        family = list(TASKS)[ep // 2]
        for t in range(n):
            ds.add_frame(
                {
                    IMAGE_KEY: torch.full((24, 24, 3), 10 * t, dtype=torch.uint8).numpy(),
                    STATE_KEY: torch.full((6,), float(t)).numpy(),
                    ACTION_KEY: torch.full((6,), 100.0 * ep + t).numpy(),
                    "task": TASKS[family],
                }
            )
        ds.save_episode()
        sidecar.append(
            {"episode_index": ep, "task_family": family, "cell": "seen", "oracle_success": True}
        )
    ds.finalize()
    (root / "meta" / "vla_rl_so101_episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in sidecar)
    )


@pytest.fixture()
def fake_recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "oft-test"
    _make_dataset(root)

    def select_episodes(
        r: Path, family: str, n: int | None = None, *, cell: str = "seen"
    ) -> list[int]:
        rows = [
            json.loads(x)
            for x in (Path(r) / "meta/vla_rl_so101_episodes.jsonl").read_text().splitlines()
        ]
        idx = [x["episode_index"] for x in rows if x["task_family"] == family and x["cell"] == cell]
        return idx if n is None else idx[:n]

    def load_demo_dataset(
        repo_id: str, r: Path, episodes: list[int] | None = None, **kw: Any
    ) -> Any:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset(repo_id, root=Path(r), episodes=episodes, **kw)

    mod = types.ModuleType("src.data.recorder")
    mod.__dict__.update(
        IMAGE_KEY=IMAGE_KEY,
        STATE_KEY=STATE_KEY,
        ACTION_KEY=ACTION_KEY,
        dataset_location=lambda cfg: ("local/oft-test", root),
        select_episodes=select_episodes,
        load_demo_dataset=load_demo_dataset,
    )
    monkeypatch.setitem(sys.modules, "src.data.recorder", mod)
    return root


def test_lerobot_chunks_repeat_last_action(fake_recorder: Path) -> None:
    cfg = load_config("tiny", "configs/sft_tiny.yaml")
    cfg["env"]["tasks"] = list(TASKS)
    cfg["data"]["demos_per_task"] = 1  # first demo of each task: episodes 0 and 2
    ds, info = load_sft_frames(cfg, chunk_len=8)
    assert info["episodes_per_task"] == {"PickPlace": [0], "Push": [2]}
    assert len(ds) == LENGTHS[0] + LENGTHS[2] and info["fps"] == 15
    first = ds[0]
    assert first["image"].shape == (3, 24, 24) and first["image"].dtype == torch.float32
    assert first["instruction"] == TASKS["PickPlace"]
    assert torch.allclose(first["actions"][:, 0], torch.tensor([0, 1, 2, 3, 4, 4, 4, 4.0]))
    last_of_ep2 = ds[len(ds) - 1]  # frame 5 of episode 2 (6 frames): chunk = a5 repeated
    assert torch.allclose(last_of_ep2["actions"][:, 0], torch.full((8,), 205.0))
    assert last_of_ep2["instruction"] == TASKS["Push"]
    batch = collate_sft([ds[i] for i in range(3)])
    assert batch.images.shape == (3, 3, 24, 24) and batch.actions.shape == (3, 8, 6)
    acts = all_actions(ds.frames)
    assert acts.shape == (len(ds), 6)
    with pytest.raises(ValueError, match="need 3"):
        cfg["data"]["demos_per_task"] = 3
        load_sft_frames(cfg, chunk_len=8)


def test_synthetic_frames_feed_the_policy() -> None:
    from src.policy.openvla_oft import build_policy

    frames = synthetic_frames(
        list(TASKS.values()), episodes_per_instruction=1, frames_per_episode=5, chunk_len=8
    )
    ds = ChunkedDemoDataset(frames, chunk_len=8)
    assert len(ds) == 10 and torch.equal(ds[4]["actions"][3], ds[4]["actions"][0])
    pol = build_policy(load_config("tiny", "configs/sft_tiny.yaml"), "cpu")
    loss, metrics = pol.sft_loss(collate_sft([ds[i] for i in range(4)]))
    assert torch.isfinite(loss) and 0.0 <= metrics["token_acc"] <= 1.0
    with pytest.raises(ValueError, match="delta_timestamps"):
        ChunkedDemoDataset(frames, chunk_len=4)[0]


def _record(root: Path, repo: str, episodes: list[tuple[int, str, bool]]) -> None:
    """A real WO-01-recorder dataset: (frames, sidecar kind, oracle_success) per episode."""
    from src.data.recorder import DemoRecorder

    rec = DemoRecorder(repo, root / repo.split("/")[-1], image_hw=(32, 32))
    for ep, (n, kind, ok) in enumerate(episodes):
        for t in range(n):
            img = torch.full((32, 32, 3), 20 * t, dtype=torch.uint8).numpy()
            act = torch.full((6,), 0.01 * t + 0.1 * ep)
            rec.add_frame(img, torch.zeros(6), act, TASKS["PickPlace"])
        rec.save_episode(
            {"task_family": "PickPlace", "seed": ep, "cell": "seen", "oracle_success": ok}
            | ({} if kind == "demo" else {"kind": kind})
        )
    rec.finalize()


def test_extra_datasets_capped_and_frame_matched(tmp_path: Path) -> None:
    """Demos + capped corrections (recovery arm), and demos + frame-matched control demos."""
    from src.data.oft_dataset import ConcatFrames, select_budget

    _record(tmp_path, "local/d", [(5, "demo", True), (6, "demo", True)])
    corr = [(4, "correction", True), (2, "correction", False), (3, "correction", True)]
    _record(tmp_path, "local/c", [*corr, (5, "correction", True)])
    _record(tmp_path, "local/k", [(3, "control_demo", True)] * 4)
    base = (
        f"inherit: configs/sft_tiny.yaml\ndata:\n  dataset_root: {tmp_path}\n  hub_repo: local/d\n"
    )
    recovery = tmp_path / "recovery.yaml"
    recovery.write_text(
        base + "  extra_datasets:\n    - {hub_repo: local/c, kind: correction, "
        "max_episodes_per_task: 5, max_frames_per_task: 8}\n"
    )
    control = tmp_path / "control.yaml"
    control.write_text(
        base + "  extra_datasets:\n    - {hub_repo: local/k, kind: control_demo, "
        f"match_frames_of: {recovery}}}\n"
    )

    ds, info = load_sft_frames(load_config("tiny", recovery), chunk_len=4)
    assert isinstance(ds.frames, ConcatFrames) and ds.frames.names == ["demo", "correction"]
    extra = info["extra_datasets"][0]
    # failed branch skipped; 4 + 3 = 7 frames, the next (5) would pass the 8-frame cap
    assert extra["episodes_per_task"] == {"PickPlace": [0, 2]} and extra["num_frames"] == 7
    assert info["demo_frames"] == 11 and len(ds) == 18 and info["episodes_per_task"]["PickPlace"]
    assert all_actions(ds.frames).shape == (18, 6)
    assert torch.allclose(ds[11]["actions"][:, 0], torch.tensor([0.0, 0.01, 0.02, 0.03]))

    ds, info = load_sft_frames(load_config("tiny", control), chunk_len=4)
    extra = info["extra_datasets"][0]
    assert extra["kind"] == "control_demo" and extra["frames_per_task"] == {"PickPlace": 6}
    assert len(ds) == 11 + 6

    rows = [{"task_family": "PickPlace", "cell": "seen", "oracle_success": True,
             "episode_index": i, "num_frames": 10} for i in range(3)]  # fmt: skip
    assert select_budget(rows, "PickPlace", kind="demo", max_episodes=2) == ([0, 1], 20)
    assert select_budget(rows, "PickPlace", kind="correction") == ([], 0)
