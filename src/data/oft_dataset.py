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

import bisect
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


class ConcatFrames:
    """Several frame sources (the demos + extra datasets) behind one index, in order."""

    def __init__(self, parts: Sequence[FrameSource], names: Sequence[str]) -> None:
        if len(parts) != len(names) or not parts:
            raise ValueError("need one name per part and at least one part")
        self.parts, self.names = list(parts), list(names)
        self.offsets = [0]
        for p in self.parts:
            self.offsets.append(self.offsets[-1] + len(p))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        part = bisect.bisect_right(self.offsets, idx) - 1
        return self.parts[part][idx - self.offsets[part]]


def all_actions(frames: FrameSource, action_key: str = ACTION_KEY) -> Tensor:
    """Every per-frame action (N, D) -- for the normalisation-bounds coverage check."""
    if isinstance(frames, ConcatFrames):
        return torch.cat([all_actions(p, action_key) for p in frames.parts])
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


def select_budget(
    rows: Sequence[dict[str, Any]],
    task: str,
    *,
    kind: str,
    cell: str = "seen",
    max_episodes: int | None = None,
    max_frames: int | None = None,
) -> tuple[list[int], int]:
    """Episodes of `task` / `kind` (sidecar "kind", default "demo") in recording order: the
    longest prefix with <= max_episodes episodes and <= max_frames frames. Only oracle-successful
    episodes count. Returns (episode indices, frames)."""
    out: list[int] = []
    frames = 0
    for r in rows:
        if r["task_family"] != task or r["cell"] != cell or not r["oracle_success"]:
            continue
        if str(r.get("kind", "demo")) != kind:
            continue
        if max_episodes is not None and len(out) >= max_episodes:
            break
        n = int(r["num_frames"])
        if max_frames is not None and frames + n > max_frames:
            break
        out.append(int(r["episode_index"]))
        frames += n
    return out, frames


def _cap(a: int | None, b: int | None) -> int | None:
    return b if a is None else a if b is None else min(a, b)


def _extra_selection(
    cfg: Config, rec: Any, spec: dict[str, Any], tasks: Sequence[str], cell: str
) -> tuple[str, Path, dict[str, list[int]], dict[str, int]]:
    """(repo_id, root, episodes per task, frames per task) of one `data.extra_datasets` entry.

    `match_frames_of: <config>` caps each task's frames at what that config's extra dataset of
    kind `match_kind` (default "correction") selects: the matched control."""
    repo_id, root = rec.dataset_location(
        {**cfg, "data": {**cfg["data"], "hub_repo": str(spec["hub_repo"])}}
    )
    root = Path(root)
    _ensure_local(str(repo_id), root)
    rows = rec.read_sidecar(root)
    max_eps = spec.get("max_episodes_per_task")
    max_frames = spec.get("max_frames_per_task")
    target: dict[str, int] = {}
    if spec.get("match_frames_of"):
        from src.config import REPO_ROOT, load_config

        other = load_config(str(cfg["preset"]), REPO_ROOT / str(spec["match_frames_of"]))
        want = str(spec.get("match_kind", "correction"))
        refs = [e for e in other["data"].get("extra_datasets") or [] if e.get("kind") == want]
        if len(refs) != 1:
            raise ValueError(f"{spec['match_frames_of']}: need exactly one {want!r} extra dataset")
        _, _, _, target = _extra_selection(other, rec, refs[0], tasks, cell)
    eps: dict[str, list[int]] = {}
    frames: dict[str, int] = {}
    for task in tasks:
        eps[task], frames[task] = select_budget(
            rows,
            task,
            kind=str(spec.get("kind", "correction")),
            cell=cell,
            max_episodes=None if max_eps is None else int(max_eps),
            max_frames=_cap(None if max_frames is None else int(max_frames), target.get(task)),
        )
    return str(repo_id), root, eps, frames


def load_sft_frames(cfg: Config, chunk_len: int) -> tuple[ChunkedDemoDataset, dict[str, Any]]:
    """The few-shot SFT set: the first `data.demos_per_task` successful `data.cell` demos per task,
    plus any `data.extra_datasets` (recovery corrections, matched-control demos), each capped per
    task. Frames are sampled uniformly, so each part's share of the mix is its frame share."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    rec: Any = importlib.import_module("src.data.recorder")  # WO-01

    repo_id, root = rec.dataset_location(cfg)
    _ensure_local(repo_id, Path(root))
    n = int(cfg["data"]["demos_per_task"])
    cell = str(cfg["data"].get("cell", "seen"))
    tasks = list(cfg["env"]["tasks"])
    episodes: list[int] = []
    per_task: dict[str, list[int]] = {}
    for task in tasks:
        eps = rec.select_episodes(root, task, n, cell=cell)
        if len(eps) < n:
            raise ValueError(f"{repo_id}: {len(eps)} {cell!r} demos for {task}, need {n}")
        per_task[task] = eps
        episodes.extend(eps)
    parts: list[tuple[str, str, Path, list[int]]] = [("demo", repo_id, Path(root), episodes)]
    extra_info: list[dict[str, Any]] = []
    for spec in cfg["data"].get("extra_datasets") or []:
        kind = str(spec.get("kind", "correction"))
        x_repo, x_root, x_eps, x_frames = _extra_selection(cfg, rec, spec, tasks, cell)
        flat = [e for t in tasks for e in x_eps[t]]
        if not flat:
            raise ValueError(f"{x_repo}: no {kind!r} episodes selected")
        parts.append((kind, x_repo, x_root, flat))
        extra_info.append(
            {
                "kind": kind,
                "repo_id": x_repo,
                "episodes_per_task": x_eps,
                "frames_per_task": x_frames,
                "num_frames": sum(x_frames.values()),
            }
        )
    sources: list[FrameSource] = []
    fps_all: set[int] = set()
    for _, p_repo, p_root, p_eps in parts:
        fps = int(LeRobotDatasetMetadata(p_repo, root=p_root).fps)
        fps_all.add(fps)
        delta = {rec.ACTION_KEY: [k / fps for k in range(chunk_len)]}
        sources.append(
            rec.load_demo_dataset(p_repo, p_root, episodes=sorted(p_eps), delta_timestamps=delta)
        )
    if len(fps_all) != 1:
        raise ValueError(f"datasets disagree on fps: {sorted(fps_all)}")
    frames: FrameSource = (
        sources[0] if len(sources) == 1 else ConcatFrames(sources, [p[0] for p in parts])
    )
    ds = ChunkedDemoDataset(
        frames,
        chunk_len=chunk_len,
        image_key=rec.IMAGE_KEY,
        state_key=rec.STATE_KEY,
        action_key=rec.ACTION_KEY,
    )
    info: dict[str, Any] = {
        "repo_id": repo_id,
        "root": str(root),
        "fps": fps_all.pop(),
        "cell": cell,
        "episodes_per_task": per_task,
        "num_frames": len(ds),
    }
    if extra_info:
        info["demo_frames"] = len(sources[0])
        info["extra_datasets"] = extra_info
    return ds, info
