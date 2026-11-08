"""Demo recording into ONE multi-task LeRobotDataset (lerobot 0.4.4, CODEBASE_VERSION v3.0).

Layout on disk (`root = <data.dataset_root>/<repo name>`):

    meta/info.json, meta/stats.json, meta/tasks.parquet, meta/episodes/...   (LeRobot)
    meta/vla_rl_so101_episodes.jsonl                                       (our sidecar)
    data/chunk-000/file-000.parquet                                         (state, action, ...)
    videos/observation.images.front/chunk-000/file-000.mp4                  (AV1, decoded by torchcodec)

LeRobot trains from a single repo_id, so all task families share one dataset; the per-frame
`task` is the language instruction and the sidecar records, per episode, the task family, seed,
eval cell and oracle success so WO-05's 1/5/10-demo ablation can select episodes per task
(`select_episodes`).

Video decode goes through torchcodec only (CLAUDE.md rule 6). lerobot 0.4.4's decoder cache hands
torchcodec an fsspec file object, which torchcodec 0.2.1 (our torch-2.6 pin) rejects;
`install_torchcodec_compat()` patches it to pass the local path instead. Call it (or
`load_demo_dataset`, which does) before reading any dataset.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import REPO_ROOT, Config
from src.env import CONTROL_HZ, JOINT_NAMES

IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ROBOT_TYPE = "so101_follower"
SIDECAR = "meta/vla_rl_so101_episodes.jsonl"
VIDEO_BACKEND = "torchcodec"
REQUIRED_META = ("task_family", "seed", "cell", "oracle_success")
# The 0.1-era demo sets every reported SFT/RL number trained on. Every push goes through
# `push_dataset`, which refuses them unless `allow_protected` (wo01_build_env.py --overwrite-hub;
# scripts/wo01r_rebuild_demos.py has no override).
PROTECTED_HUB_REPOS = ("KudratSingh22/vla-rl-so101-demos", "KudratSingh22/vla-rl-so101-demos-25")


class ProtectedRepoError(ValueError):
    """A push to one of PROTECTED_HUB_REPOS without an explicit override."""


def check_push_target(repo_id: str, *, allow_protected: bool = False) -> None:
    """Raise ProtectedRepoError for a protected repo (before any data or network is touched)."""
    if repo_id in PROTECTED_HUB_REPOS and not allow_protected:
        raise ProtectedRepoError(
            f"refusing to push to {repo_id}: it holds a 0.1-era demo set that reported SFT/RL "
            "numbers trained on; set data.hub_repo to a new repo"
        )


ArrayLike = np.ndarray[Any, Any] | torch.Tensor


class _NoHandle:
    def close(self) -> None:
        return None


_compat_lock = threading.Lock()
_compat_installed = False


def install_torchcodec_compat() -> None:
    """Make lerobot 0.4.4's torchcodec decoder cache work with torchcodec 0.2.1 (idempotent).

    Local files are opened by path (supported by every torchcodec release); remote URLs keep
    lerobot's original fsspec path.
    """
    global _compat_installed
    with _compat_lock:
        if _compat_installed:
            return
        try:
            import torchcodec.decoders  # noqa: F401  (loads the FFmpeg shared libs)
        except (ImportError, OSError, RuntimeError) as e:
            raise RuntimeError(
                "torchcodec cannot load FFmpeg 4-7. On macOS: brew install ffmpeg@7 and "
                "`source scripts/mac_env.sh`; on Linux: apt-get install ffmpeg"
            ) from e
        from lerobot.datasets import video_utils

        original = video_utils.VideoDecoderCache.get_decoder

        def get_decoder(self: Any, video_path: str) -> Any:
            path = str(video_path)
            if "://" in path:
                return original(self, video_path)
            from torchcodec.decoders import VideoDecoder

            with self._lock:
                if path not in self._cache:
                    decoder = VideoDecoder(path, seek_mode="approximate")
                    self._cache[path] = (decoder, _NoHandle())
                return self._cache[path][0]

        video_utils.VideoDecoderCache.get_decoder = get_decoder
        _compat_installed = True


@contextlib.contextmanager
def _datasets_write_compat() -> Iterator[None]:
    """Let `datasets` write rows after torchcodec was imported in this process.

    Once torchcodec is in sys.modules (any dataset was decoded, e.g. the render-gap check or an
    earlier test), `datasets` imports `torchcodec.decoders.AudioDecoder` while encoding rows;
    torchcodec 0.2.1 (our torch-2.6 pin) has no AudioDecoder. Our rows never hold decoder
    objects (video goes to mp4 through LeRobot's encoder), so writes skip that branch.
    """
    import datasets.config

    prev = datasets.config.TORCHCODEC_AVAILABLE
    datasets.config.TORCHCODEC_AVAILABLE = False
    try:
        yield
    finally:
        datasets.config.TORCHCODEC_AVAILABLE = prev


def dataset_location(cfg: Config) -> tuple[str, Path]:
    """(repo_id, local root) for the preset's demo dataset.

    Presets without `data.hub_repo` (e.g. `tiny`) get a local-only repo id, so a CPU smoke never
    overwrites the real dataset's directory.
    """
    data = cfg.get("data", {})
    repo_id = data.get("hub_repo") or f"local/vla-rl-so101-demos-{cfg.get('preset', 'dev')}"
    root = Path(data.get("dataset_root", "data/demos"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    return str(repo_id), root / str(repo_id).split("/")[-1]


def demo_features(
    image_hw: tuple[int, int], joint_names: Sequence[str] = JOINT_NAMES
) -> dict[str, dict[str, Any]]:
    h, w = image_hw
    names = list(joint_names)
    return {
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
        STATE_KEY: {"dtype": "float32", "shape": (len(names),), "names": names},
        ACTION_KEY: {"dtype": "float32", "shape": (len(names),), "names": names},
    }


def _to_numpy(x: ArrayLike) -> np.ndarray[Any, Any]:
    if isinstance(x, torch.Tensor):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


class DemoRecorder:
    """Thin writer over `LeRobotDataset.create` with an episode-level sidecar.

    One episode at a time: `add_frame` ... then `save_episode(meta)` or `discard_episode()`.
    Call `finalize()` once at the end (parquet footers are only written then).
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        image_hw: tuple[int, int],
        fps: int = CONTROL_HZ,
        joint_names: Sequence[str] = JOINT_NAMES,
        robot_type: str = ROBOT_TYPE,
        overwrite: bool = False,
        vcodec: str = "libsvtav1",
    ) -> None:
        import datasets
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        datasets.disable_progress_bars()  # one "Map" bar per saved episode otherwise
        os.environ.setdefault("SVT_LOG", "1")  # libsvtav1: errors only, not its config banner
        self.repo_id = repo_id
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"{self.root} exists; pass overwrite=True to rebuild it")
            shutil.rmtree(self.root)
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.num_joints = len(joint_names)
        with _datasets_write_compat():
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=demo_features(self.image_hw, joint_names),
                root=self.root,
                robot_type=robot_type,
                use_videos=True,
                video_backend=VIDEO_BACKEND,
                vcodec=vcodec,
            )
        self.sidecar_path = self.root / SIDECAR
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_path.touch()
        self._num_frames = 0
        self._instruction: str | None = None
        self._finalized = False

    @classmethod
    def from_config(
        cls, cfg: Config, *, image_hw: tuple[int, int] | None = None, overwrite: bool = False
    ) -> DemoRecorder:
        """`image_hw` defaults to cfg env.image_size (square); pass the observed (H, W) to be exact."""
        repo_id, root = dataset_location(cfg)
        if image_hw is None:
            size = int(cfg["env"]["image_size"])
            image_hw = (size, size)
        return cls(repo_id, root, image_hw=image_hw, overwrite=overwrite)

    @property
    def num_episodes(self) -> int:
        return int(self.dataset.meta.total_episodes)

    @property
    def num_frames(self) -> int:
        return int(self.dataset.meta.total_frames)

    def add_frame(
        self, image: ArrayLike, state: ArrayLike, action: ArrayLike, instruction: str
    ) -> None:
        if self._finalized:
            raise RuntimeError("recorder is finalized")
        img = _to_numpy(image)
        if img.shape != (*self.image_hw, 3) or img.dtype != np.uint8:
            raise ValueError(
                f"image must be uint8 {(*self.image_hw, 3)}, got {img.dtype} {img.shape}"
            )
        st = _to_numpy(state).astype(np.float32).reshape(self.num_joints)
        act = _to_numpy(action).astype(np.float32).reshape(self.num_joints)
        if self._instruction is None:
            self._instruction = instruction
        elif instruction != self._instruction:
            raise ValueError("instruction changed within an episode")
        with _datasets_write_compat():
            self.dataset.add_frame(
                {IMAGE_KEY: img, STATE_KEY: st, ACTION_KEY: act, "task": instruction}
            )
        self._num_frames += 1

    def save_episode(self, meta: dict[str, Any]) -> int:
        """Write the buffered episode; returns its episode_index. `meta` needs REQUIRED_META."""
        missing = [k for k in REQUIRED_META if k not in meta]
        if missing:
            raise ValueError(f"episode meta missing {missing}")
        if self._num_frames == 0 or self._instruction is None:
            raise RuntimeError("no frames buffered")
        episode_index = self.num_episodes
        with _datasets_write_compat():
            self.dataset.save_episode()
        row = {
            "episode_index": episode_index,
            "task_family": str(meta["task_family"]),
            "instruction": self._instruction,
            "seed": int(meta["seed"]),
            "cell": str(meta["cell"]),
            "oracle_success": bool(meta["oracle_success"]),
            "num_frames": self._num_frames,
        }
        row.update({k: v for k, v in meta.items() if k not in row})
        with self.sidecar_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        self._num_frames = 0
        self._instruction = None
        return episode_index

    def discard_episode(self) -> None:
        """Drop the buffered episode (and its temporary PNG frames)."""
        buf = self.dataset.episode_buffer
        if buf is not None and buf.get("size", 0) > 0:
            ep = int(buf["episode_index"])
            for key in self.dataset.meta.video_keys:
                img_dir = self.dataset._get_image_file_dir(ep, key)
                if Path(img_dir).is_dir():
                    shutil.rmtree(img_dir)
            self.dataset.clear_episode_buffer(delete_images=False)
        self._num_frames = 0
        self._instruction = None

    def finalize(self) -> None:
        if self._finalized:
            return
        self.discard_episode()
        with _datasets_write_compat():
            self.dataset.finalize()
        images = self.root / "images"
        if images.is_dir() and not any(p.is_file() for p in images.rglob("*")):
            shutil.rmtree(images)
        self._finalized = True

    def push_to_hub(self, *, private: bool = True, allow_protected: bool = False) -> str:
        """Push the finalized dataset (sidecar included) to the Hub. Needs HF_TOKEN."""
        check_push_target(self.repo_id, allow_protected=allow_protected)
        self.finalize()
        return push_dataset(
            self.repo_id, self.root, private=private, allow_protected=allow_protected
        )


def _ensure_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN" and value.strip():
                os.environ["HF_TOKEN"] = value.strip().strip("'\"")
                return
    from huggingface_hub import get_token

    if not get_token():
        raise RuntimeError("no Hugging Face token: set HF_TOKEN in the environment or .env")


def push_dataset(
    repo_id: str, root: str | Path, *, private: bool = True, allow_protected: bool = False
) -> str:
    """Push an on-disk demo dataset to `repo_id` (private by default); returns its URL. Refuses
    PROTECTED_HUB_REPOS unless `allow_protected` (ProtectedRepoError, before the token check)."""
    check_push_target(repo_id, allow_protected=allow_protected)
    _ensure_hf_token()
    ds = load_demo_dataset(repo_id, root)
    ds.push_to_hub(private=private, tags=["vla-rl-so101", "so101", "maniskill3", "lerobot"])
    return f"https://huggingface.co/datasets/{repo_id}"


def load_demo_dataset(
    repo_id: str, root: str | Path, episodes: list[int] | None = None, **kwargs: Any
) -> Any:
    """LeRobotDataset reading with torchcodec (never PyAV)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    install_torchcodec_compat()
    kwargs.setdefault("video_backend", VIDEO_BACKEND)
    return LeRobotDataset(repo_id, root=Path(root), episodes=episodes, **kwargs)


def read_sidecar(root: str | Path) -> list[dict[str, Any]]:
    path = Path(root) / SIDECAR
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_episodes(
    root: str | Path,
    task_family: str,
    n: int | None = None,
    *,
    cell: str = "seen",
    successful_only: bool = True,
) -> list[int]:
    """First `n` episode indices of `task_family` (in recording order), e.g. for 1/5/10-demo SFT."""
    rows = [
        r
        for r in read_sidecar(root)
        if r["task_family"] == task_family
        and r["cell"] == cell
        and (r["oracle_success"] or not successful_only)
    ]
    idx = [int(r["episode_index"]) for r in rows]
    return idx if n is None else idx[:n]
