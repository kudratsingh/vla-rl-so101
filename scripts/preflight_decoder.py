#!/usr/bin/env python
"""Decoder preflight: torchcodec + FFmpeg load, and a two-camera clip the demo recorder would write
decodes across every frame. No port, no camera, no arm: run it before any recording session.

    source scripts/mac_env.sh
    .venv/bin/python scripts/preflight_decoder.py --preset real

What it does, in order (exit 0 and ``PASS`` only when every stage passes; exit 1 and ``FAIL`` with
the fix otherwise, never a traceback):

1. ``torchcodec``: imports ``torchcodec.decoders`` (which dlopens ``libtorchcodec<N>`` and the FFmpeg
   4-7 shared libraries it needs) and prints its version, the FFmpeg version it bound to and the
   library files the process loaded. On macOS the FFmpeg 7 libraries come from Homebrew's keg-only
   ``ffmpeg@7``, found through ``DYLD_FALLBACK_LIBRARY_PATH`` (``scripts/mac_env.sh``).
2. ``encoder``: PyAV (LeRobot 0.4.4 encodes the dataset videos with it, ``encode_video_frames``) and
   the convention's codec ``libsvtav1`` (``src/robot/convention.py`` ``VIDEO_CODEC``). Decoding never
   uses PyAV (CLAUDE.md rule 6).
3. ``config``: ``convention.check_config`` on the preset's config (both cameras 640 x 480 at 30 fps,
   control 15 Hz), the check the recorder refuses on.
4. ``record + decode``: a throwaway dataset in a temp directory, written by the recorder's own code
   path (``recording.create_dataset``, ``recording.DurableDataset`` (save, finalize, reopen through
   LeRobot's resume path), ``convention.recording_frame``), two episodes of synthetic frames for
   every camera of the convention, then ``recording.episode_problem`` and
   ``recording.verify_episode_video`` (torchcodec, every frame of both clips) per episode, and a
   pixel check of the first and last decoded frame of each clip against what was written. The
   temp directory is removed at the end.

``--dataset <dir>`` checks a RECORDED dataset instead (stages 1 and 5; read only, nothing written):

5. ``dataset``: every episode readable and whole (``recording.episode_problem``) and both cameras'
   clips decoded across every frame (``recording.verify_episode_video``), the episode count
   (``--episodes N``: FAIL unless exactly N), and from ``meta/vla_rl_so101.json`` the recording
   kind and how each sitting ended; for a ``--demo-schedule`` recording also the real-0.1 protocol
   hash and the convention hash against this code (``scripts/wo07_export_bundle.py demo`` refuses
   on either) and that episode i is demo i of the frozen schedule. The demo recorder itself does
   not decode its saved episodes; this is that check.

    .venv/bin/python scripts/preflight_decoder.py --preset real \\
        --dataset data/real_teleop/vla-rl-so101-real-teleop --episodes 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import add_config_args, load_config

BREW_FFMPEG7_LIB = "/opt/homebrew/opt/ffmpeg@7/lib"
EPISODES = 2
FRAMES = 20  # per episode: 1.3 s at the convention's 15 fps
MAX_PIXEL_DIFF = 12.0  # mean |decoded - written| per channel, 0-255; AV1 at crf 30 on a gradient
LIB_MARKERS = ("libtorchcodec", "libavcodec", "libavformat", "libavutil", "libsvtav1")


class StageFailed(Exception):
    """A stage failed; ``fix`` is what the operator does about it."""

    def __init__(self, detail: str, fix: str):
        super().__init__(detail)
        self.fix = fix


def loaded_libraries() -> list[str]:
    """The torchcodec / FFmpeg library files this process has loaded (macOS: dyld's image list;
    Linux: /proc/self/maps). Empty where neither is available."""
    names: list[str] = []
    try:
        if sys.platform == "darwin":
            import ctypes

            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            libc._dyld_image_count.restype = ctypes.c_uint32
            libc._dyld_get_image_name.restype = ctypes.c_char_p
            libc._dyld_get_image_name.argtypes = [ctypes.c_uint32]
            for i in range(libc._dyld_image_count()):
                raw = libc._dyld_get_image_name(i)
                if raw:
                    names.append(raw.decode())
        elif Path("/proc/self/maps").is_file():
            for line in Path("/proc/self/maps").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 6 and parts[5].startswith("/"):
                    names.append(parts[5])
    except Exception:  # noqa: BLE001 - a listing is a nicety, never a failure
        return []
    return sorted({n for n in names if any(m in Path(n).name for m in LIB_MARKERS)})


def ffmpeg_fix() -> str:
    """What to do when torchcodec cannot load FFmpeg, for this machine."""
    if sys.platform != "darwin":
        return "install FFmpeg 4-7 shared libraries: apt-get install -y ffmpeg"
    dyld = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
    if not Path(BREW_FFMPEG7_LIB).is_dir():
        return (
            "brew install ffmpeg@7   (keg-only; torchcodec 0.2.1 needs FFmpeg 4-7, the default "
            "Homebrew ffmpeg is too new), then: source scripts/mac_env.sh"
        )
    if dyld != BREW_FFMPEG7_LIB:
        return (
            f"export DYLD_FALLBACK_LIBRARY_PATH={BREW_FFMPEG7_LIB}   (or: source "
            f"scripts/mac_env.sh, in a shell where it is not already set; now: {dyld!r}), and "
            "start Python as .venv/bin/python or uv run (not via /usr/bin/env, which macOS "
            "strips DYLD_* from)"
        )
    return (
        f"{BREW_FFMPEG7_LIB} is set but did not load: brew reinstall ffmpeg@7; then rerun "
        "uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla "
        "(= make setup; torchcodec 0.2.1)"
    )


def stage_torchcodec() -> str:
    try:
        import torchcodec
        import torchcodec.decoders  # dlopens libtorchcodec<N> + FFmpeg
        from torchcodec.decoders._core import get_ffmpeg_library_versions
    except (ImportError, OSError, RuntimeError) as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        first = first.split(" Likely causes:")[0]  # torchcodec's list follows on later lines
        raise StageFailed(f"torchcodec cannot load FFmpeg: {first}", ffmpeg_fix()) from e
    versions = get_ffmpeg_library_versions()
    libs = loaded_libraries()  # before PyAV and LeRobot load their own copies
    lines = [
        f"torchcodec {torchcodec.__version__} from {Path(torchcodec.__file__).parent}",
        (
            f"FFmpeg {versions.get('ffmpeg_version')} "
            f"(libavcodec {'.'.join(map(str, versions.get('libavcodec', [])))})"
        ),
        *[f"  loaded {p}" for p in libs],
    ]
    return "\n".join(lines)


def stage_encoder() -> str:
    from src.robot import convention as conv

    try:
        import av
        from av.codec import Codec

        Codec(conv.VIDEO_CODEC, "w")
    except Exception as e:
        raise StageFailed(
            f"PyAV encoder {conv.VIDEO_CODEC!r} unavailable: {type(e).__name__}: {e}",
            "uv sync --frozen --extra dev --extra gpu --extra robot --extra smolvla   "
            "(= make setup; the pinned PyAV wheel bundles libsvtav1)",
        ) from e
    libav = av.library_versions.get("libavcodec", ())
    return (
        f"PyAV {av.__version__} (bundled libavcodec {'.'.join(map(str, libav))}) encodes "
        f"{conv.VIDEO_CODEC}; LeRobot writes yuv420p (encode_video_frames default)"
    )


def stage_config(preset: str, config: Path | None) -> str:
    from src.robot import convention as conv

    cfg = load_config(preset, config)
    problems = conv.check_config(cfg)
    if problems:
        raise StageFailed(
            "the config departs from the recording convention: " + "; ".join(problems),
            "fix configs/real.yaml (the recorder refuses on the same check)",
        )
    cams = cfg["cameras"]
    return (
        ", ".join(
            f"{n} {cams[n]['width']}x{cams[n]['height']} at {cams[n]['fps']} fps"
            for n in conv.CAMERA_NAMES
        )
        + f"; dataset {conv.FPS} fps, {conv.VIDEO_CODEC}, backend {conv.VIDEO_BACKEND}"
    )


def synthetic(camera: int, i: int) -> np.ndarray:
    """A smooth frame that differs per camera and per frame (a moving gradient)."""
    from src.robot import convention as conv

    h, w = conv.IMAGE_HEIGHT, conv.IMAGE_WIDTH
    x = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    shift = i / FRAMES
    r = 255.0 * ((x + shift) % 1.0) * np.ones_like(y)
    g = 255.0 * y * np.ones_like(x)
    b = np.full((h, w), 60.0 + 120.0 * camera, np.float32)
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def pixel_problems(root: Path, index: int) -> list[str]:
    """First and last decoded frame of each clip of episode ``index`` vs the frames written."""
    import pyarrow.parquet as pq
    from torchcodec.decoders import VideoDecoder

    from src.robot import convention as conv

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    ep = next(
        r
        for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))
        for r in pq.read_table(f).to_pylist()
        if int(r["episode_index"]) == index
    )
    out = []
    for c, name in enumerate(conv.CAMERA_NAMES):
        key = conv.IMAGE_KEYS[name]
        rel = str(info["video_path"]).format(
            video_key=key,
            chunk_index=ep[f"videos/{key}/chunk_index"],
            file_index=ep[f"videos/{key}/file_index"],
        )
        t0 = float(ep[f"videos/{key}/from_timestamp"])
        dec = VideoDecoder(str(root / rel), seek_mode="exact", dimension_order="NHWC")
        ids = [0, int(ep["length"]) - 1]
        got = dec.get_frames_played_at([t0 + (i + 0.5) / fps for i in ids]).data.numpy()
        for j, i in enumerate(ids):
            diff = float(np.abs(got[j].astype(np.float32) - synthetic(c, i)).mean())
            if diff > MAX_PIXEL_DIFF:
                out.append(
                    f"{key} episode {index} frame {i}: mean pixel difference {diff:.1f} > "
                    f"{MAX_PIXEL_DIFF} (the decode returns other pixels than were written)"
                )
    return out


def stage_record_decode(tmp: Path) -> str:
    from src.robot import convention as conv
    from src.robot import recording

    root = tmp / "dataset"
    t = time.perf_counter()
    try:
        ds = recording.DurableDataset(recording.create_dataset("preflight/decoder", root), "x")
        try:
            for _ in range(EPISODES):
                for i in range(FRAMES):
                    images = {n: synthetic(c, i) for c, n in enumerate(conv.CAMERA_NAMES)}
                    row = conv.recording_frame(
                        images,
                        np.zeros(6, np.float32),
                        np.zeros(6, np.float32),
                        conv.DEMO_TASK_STRING,
                        write_time_s=i / conv.FPS,
                        frame_time_s=i / conv.FPS,
                    )
                    ds.add_frame(row)
                ds.save_episode()  # encode, finalize, reopen (LeRobot's resume path)
        finally:
            ds.close()
    except Exception as e:
        raise StageFailed(
            f"writing the two-camera dataset failed: {type(e).__name__}: {e}",
            "send the agent this output; the demo recorder writes through the same code",
        ) from e
    t_write = time.perf_counter() - t
    problems: list[str] = []
    for index in range(EPISODES):
        bad = recording.episode_problem(root, index, FRAMES)
        problems += [bad] if bad else []
        problems += recording.verify_episode_video(root, index)
    if not problems:
        try:
            for index in range(EPISODES):
                problems += pixel_problems(root, index)
        except Exception as e:  # noqa: BLE001 - the answer is the failure
            problems.append(f"pixel check: {type(e).__name__}: {e}")
    if problems:
        raise StageFailed(
            "the clips do not decode: " + " | ".join(problems),
            "send the agent this output (torchcodec / FFmpeg / the encoder disagree)",
        )
    return (
        f"{EPISODES} episodes x {FRAMES} frames x {len(conv.CAMERA_NAMES)} cameras "
        f"({', '.join(conv.IMAGE_KEYS.values())}, {conv.IMAGE_HEIGHT}x{conv.IMAGE_WIDTH}) written "
        f"in {t_write:.1f} s; verify_episode_video: every frame of every clip decodes; first and "
        f"last frames match what was written"
    )


def _jsonable(v: object) -> object:
    return json.loads(json.dumps(v))


def stage_dataset(root: Path, expect: int | None) -> str:
    """Stage 5 (module doc): a recorded dataset, read only."""
    from src.eval import protocol_real as PR
    from src.robot import convention as conv
    from src.robot import recording

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise StageFailed(f"{info_path} missing: not a LeRobot dataset", "check the --dataset path")
    n = int(json.loads(info_path.read_text()).get("total_episodes", 0))
    problems = [f"{n} episodes, expected {expect}"] if expect is not None and n != expect else []
    for index in range(n):
        bad = recording.episode_problem(root, index)
        problems += [bad] if bad else recording.verify_episode_video(root, index)
    notes = [f"{n} episodes; both cameras decode across every frame" if not problems else ""]
    prov_path = root / recording.PROVENANCE
    if not prov_path.is_file():
        problems.append(f"{prov_path} missing: not a recording of this repo")
    else:
        prov = json.loads(prov_path.read_text())
        kind = (
            "rehearsal"
            if prov.get("rehearsal")
            else "demo schedule" if prov.get("demo_schedule") else "sanity / free teleop"
        )
        ends = [s.get("session_end") for s in prov.get("sittings") or []]
        notes.append(f"{kind} recording, session {prov.get('session_id')}, sittings ended {ends}")
        if prov.get("demo_schedule"):
            got = (prov.get("protocol") or {}).get("hash")
            if got != PR.protocol_hash():
                problems.append(f"protocol hash {got} != this code's {PR.protocol_hash()}")
            if (prov.get("convention") or {}).get("hash") != conv.convention_hash():
                problems.append("convention hash differs from this code's")
            sched = _jsonable(PR.demo_schedule())
            eps = prov.get("episodes") or {}
            off = [
                i
                for i in range(n)
                if not isinstance(sched, list)
                or i >= len(sched)
                or _jsonable(eps.get(str(i))) != sched[i]
            ]
            if off:
                problems.append(f"episodes {off} are not demos {off} of the frozen schedule")
            else:
                notes.append(f"episodes 0-{n - 1} = demos 1-{n} of the frozen schedule")
            notes.append(f"{PR.PROTOCOL_VERSION} hash {PR.protocol_hash()[:12]}")
    if problems:
        raise StageFailed(
            " | ".join(problems),
            "do not export or train on it; send the agent this output (a clip that does not "
            "decode is re-recorded: move the dataset aside first, never --overwrite it)",
        )
    return "; ".join(x for x in notes if x)


def run(
    preset: str, config: Path | None, dataset: Path | None = None, expect: int | None = None
) -> bool:
    print(f"python {sys.version.split()[0]} {sys.executable}")
    print(f"DYLD_FALLBACK_LIBRARY_PATH={os.environ.get('DYLD_FALLBACK_LIBRARY_PATH')!r}")
    ok = True
    with tempfile.TemporaryDirectory(prefix="vla-rl-preflight-") as tmp:
        stages: list[tuple[str, Callable[[], str]]] = [
            ("torchcodec", stage_torchcodec),
            ("encoder", stage_encoder),
            ("config", lambda: stage_config(preset, config)),
            ("record + decode", lambda: stage_record_decode(Path(tmp))),
        ]
        if dataset is not None:
            stages = [stages[0], ("dataset", lambda: stage_dataset(dataset, expect))]
        for name, fn in stages:
            try:
                print(f"[{name}] OK: {fn()}", flush=True)
            except StageFailed as e:
                ok = False
                print(f"[{name}] FAIL: {e}\n  fix: {e.fix}", flush=True)
                if name == "torchcodec":  # nothing after it can decode
                    break
            except Exception as e:  # noqa: BLE001 - a FAIL line, never a traceback
                ok = False
                print(f"[{name}] FAIL: {type(e).__name__}: {e}\n  fix: send the agent this output")
    if dataset is None:
        print(f"temp dataset removed: {tmp}")
        print("PASS: torchcodec + FFmpeg decode the recorder's two-camera clips" if ok else "FAIL")
    else:
        print(f"PASS: {dataset} checks out" if ok else "FAIL")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    parser.add_argument(
        "--dataset", type=Path, default=None, help="check a recorded dataset instead (read only)"
    )
    parser.add_argument(
        "--episodes", type=int, default=None, help="with --dataset: FAIL unless exactly N"
    )
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("the decoder preflight checks the real-arm recording: --preset real")
    if args.episodes is not None and args.dataset is None:
        parser.error("--episodes needs --dataset")
    return 0 if run(args.preset, args.config, args.dataset, args.episodes) else 1


if __name__ == "__main__":
    sys.exit(main())
