"""SFT data for OpenVLA-OFT straight from the WO-01 LeRobotDataset v3.0 (no TensorFlow / RLDS).

Each sample is one frame t of a demo: the front image and joint state at t, the action chunk
a[t : t + K] (LeRobot `delta_timestamps`; indices past the episode end are clamped, i.e. the last
action repeats -- exactly OFT's RLDS `chunk_act_obs`), and the per-frame language `task`.
Everything model-specific (crop/augment/resize/normalise/tokenise) happens later in the policy's
single `build_inputs`, so SFT, eval and RL share one preprocessing path.

Reading goes through `src.data.recorder.load_demo_dataset` (torchcodec video decode + the
lerobot-0.4.4 / torchcodec-0.2.1 compat shim, never PyAV); that module lands with WO-01, so it is
imported lazily here.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.config import Config
from src.policy.base import SFTBatch

# LeRobot feature keys used by the WO-01 recorder (src.data.recorder IMAGE_KEY/STATE_KEY/ACTION_KEY)
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"


class FrameSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


class ChunkedDemoDataset(Dataset[dict[str, Any]]):
    """LeRobot-style frames (with an action chunk per frame) -> SFT samples."""

    def __init__(
        self,
        frames: FrameSource,
        *,
        chunk_len: int,
        image_key: str = IMAGE_KEY,
        state_key: str = STATE_KEY,
        action_key: str = ACTION_KEY,
    ) -> None:
        self.frames = frames
        self.chunk_len = chunk_len
        self.image_key, self.state_key, self.action_key = image_key, state_key, action_key

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.frames[idx]
        actions = torch.as_tensor(item[self.action_key]).float()
        if actions.ndim != 2 or actions.shape[0] != self.chunk_len:
            raise ValueError(
                f"frame {idx}: action must be a ({self.chunk_len}, D) chunk; load the dataset "
                f"with delta_timestamps for {self.action_key!r} (got {tuple(actions.shape)})"
            )
        image = torch.as_tensor(item[self.image_key])
        if image.ndim == 4:  # (1, 3, H, W) if an image delta was requested
            image = image[-1]
        return {
            "image": image,
            "state": torch.as_tensor(item[self.state_key]).float(),
            "actions": actions,
            "instruction": str(item[TASK_KEY]),
        }


def collate_sft(items: Sequence[dict[str, Any]]) -> SFTBatch:
    return SFTBatch(
        images=torch.stack([it["image"] for it in items]),
        states=torch.stack([it["state"] for it in items]),
        actions=torch.stack([it["actions"] for it in items]),
        instructions=[it["instruction"] for it in items],
    )


def all_actions(frames: FrameSource, action_key: str = ACTION_KEY) -> Tensor:
    """Every per-frame action (N, D) -- for the normalisation-bounds coverage check."""
    hf = getattr(frames, "hf_dataset", None)
    if hf is not None:
        return torch.stack([torch.as_tensor(a) for a in hf[action_key]]).float()
    rows = [torch.as_tensor(frames[i][action_key]).float() for i in range(len(frames))]
    return torch.stack([r[0] if r.ndim == 2 else r for r in rows])


def synthetic_frames(
    instructions: Sequence[str],
    *,
    episodes_per_instruction: int,
    frames_per_episode: int,
    chunk_len: int,
    image_hw: tuple[int, int] = (96, 96),
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Random frames in the LeRobot item format (tiny preset `--synthetic`, tests). Actions vary
    smoothly per episode and image brightness tracks time, so a tiny model can fit them."""
    g = torch.Generator().manual_seed(seed)
    out: list[dict[str, Any]] = []
    for instr in instructions:
        for _ in range(episodes_per_instruction):
            start = (torch.rand(6, generator=g) - 0.5) * 0.8  # inside every joint's bounds
            step = (torch.rand(6, generator=g) - 0.5) * 0.02
            acts = torch.stack([start + t * step for t in range(frames_per_episode)])
            for t in range(frames_per_episode):
                idx = [min(t + k, frames_per_episode - 1) for k in range(chunk_len)]
                img = torch.full((3, *image_hw), t / max(1, frames_per_episode - 1))
                out.append(
                    {
                        IMAGE_KEY: img,
                        STATE_KEY: acts[max(t - 1, 0)].clone(),
                        ACTION_KEY: acts[idx].clone(),
                        f"{ACTION_KEY}_is_pad": torch.tensor(
                            [t + k >= frames_per_episode for k in range(chunk_len)]
                        ),
                        TASK_KEY: instr,
                    }
                )
    return out


def _ensure_local(repo_id: str, root: Path) -> None:
    """The episode sidecar is read before LeRobot opens the dataset: pull it from the Hub once."""
    if (root / "meta" / "info.json").is_file():
        return
    if repo_id.startswith("local/"):
        raise FileNotFoundError(
            f"no demo dataset at {root}; record it first: uv run scripts/wo01_build_env.py "
            "--preset <preset> (or pass --synthetic for a smoke run)"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(root))


def load_sft_frames(cfg: Config, chunk_len: int) -> tuple[ChunkedDemoDataset, dict[str, Any]]:
    """The few-shot SFT set: the first `data.demos_per_task` successful `data.cell` demos per task."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    rec: Any = importlib.import_module("src.data.recorder")  # WO-01

    repo_id, root = rec.dataset_location(cfg)
    _ensure_local(repo_id, Path(root))
    n = int(cfg["data"]["demos_per_task"])
    cell = str(cfg["data"].get("cell", "seen"))
    episodes: list[int] = []
    per_task: dict[str, list[int]] = {}
    for task in cfg["env"]["tasks"]:
        eps = rec.select_episodes(root, task, n, cell=cell)
        if len(eps) < n:
            raise ValueError(f"{repo_id}: {len(eps)} {cell!r} demos for {task}, need {n}")
        per_task[task] = eps
        episodes.extend(eps)
    fps = int(LeRobotDatasetMetadata(repo_id, root=root).fps)
    delta = {rec.ACTION_KEY: [k / fps for k in range(chunk_len)]}
    frames = rec.load_demo_dataset(repo_id, root, episodes=sorted(episodes), delta_timestamps=delta)
    ds = ChunkedDemoDataset(
        frames,
        chunk_len=chunk_len,
        image_key=rec.IMAGE_KEY,
        state_key=rec.STATE_KEY,
        action_key=rec.ACTION_KEY,
    )
    info = {
        "repo_id": repo_id,
        "root": str(root),
        "fps": fps,
        "cell": cell,
        "episodes_per_task": per_task,
        "num_frames": len(ds),
    }
    return ds, info
