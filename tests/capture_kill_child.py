"""TEST ONLY: a real-0.1 MOCK session (scripts/wo07_real_eval.py --mock, capture on) that SIGKILLs
itself at one point of its Nth trial's capture (tests/test_capture_resume_gap.py; N = 3 of 3 by
default, or 2: the resume then recovers it AND records the 3rd trial).

    python capture_kill_child.py RUNS_DIR CKPT_DIR {after_record,mid_save,after_save} [N]

* ``after_record``: the scored record is logged and flushed, the episode save has not started
  (the owner's 2026-11-09 interruption);
* ``mid_save``: inside the save: LeRobot has written the episode's data and video, nothing is
  finalized and the provenance entry is not written;
* ``after_save``: the episode is saved, finalized and in the provenance; the ``capture`` row that
  would name its index is not logged.

SIGKILL: no ``finally``, no shutdown, no flush; what is on disk is what a power cut leaves. Fakes
only (fake arm, fake rig, MockPolicy behind gRPC on 127.0.0.1); fake clock.

Review F5: every camera read is stamped (a 96 x 96 block of a per-read value), so no two frames of
the session share pixels, and just before the kill the episode as captured (state, action, timing,
the first and last frame of each camera, its trial id) is written to ``RUNS_DIR/expected.npz`` /
``expected.json``: the test compares the recovered episode with it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "tests")]

STAMP = (slice(0, 96), slice(0, 96))  # the stamped block of every camera read


def stamp_value(n: int) -> int:
    return (7 * n) % 256  # 7 is coprime to 256: 256 distinct values before a repeat


def _stamp_cameras() -> None:
    import robot_fakes

    from src.robot.camera import Frame

    read = robot_fakes.FakeCam.read
    made: list[Any] = []  # every stamped frame (kept alive: identity checks stay valid)

    def stamped(self: Any) -> Any:
        f = read(self)
        if any(f is m for m in made[-4:]):  # a stalled stream repeats its last frame
            return f
        img = np.array(f.image, copy=True)
        img[STAMP] = stamp_value(len(made) + 1)
        out = Frame(img, f.capture_time)
        made.append(out)
        self.last = out
        return out

    robot_fakes.FakeCam.read = stamped


def _dump_expected(capture: Any, runs: str) -> None:
    """The open episode as captured, before the kill (review F5)."""
    from PIL import Image

    from src.robot import convention as conv

    ds = capture.dataset
    wait = getattr(ds, "_wait_image_writer", None)
    if wait is not None:
        wait()
    buf = ds.episode_buffer
    arrays = {
        "state": np.stack(buf[conv.STATE_KEY]).astype(np.float32),
        "action": np.stack(buf[conv.ACTION_KEY]).astype(np.float32),
        "timing": np.stack(buf[conv.TIMING_KEY]).astype(np.float32),
    }
    for name, key in conv.IMAGE_KEYS.items():
        for tag, i in (("first", 0), ("last", -1)):
            with Image.open(buf[key][i]) as im:
                arrays[f"{name}_{tag}"] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    np.savez(Path(runs) / "expected.npz", **arrays)
    tid = capture.open_episode[0]
    (Path(runs) / "expected.json").write_text(json.dumps({"trial_id": tid}))


def _kill() -> None:
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def main() -> None:
    runs, ckpt, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    kill_at = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    _stamp_cameras()
    spec = importlib.util.spec_from_file_location(
        "wo07_real_eval_kill", REPO / "scripts" / "wo07_real_eval.py"
    )
    assert spec is not None and spec.loader is not None
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    from src.robot import recording

    seen = {"saves": 0}
    save_capture = ev.save_capture

    def killing_save_capture(capture: Any, rec: dict[str, Any], log_row: Any) -> None:
        seen["saves"] += 1
        if seen["saves"] == kill_at:
            _dump_expected(capture, runs)  # before any save: the buffer as captured
            if mode == "after_record":
                _kill()
            if mode == "after_save":
                capture.finish(rec)  # saved + checkpointed + provenance; no capture row
                _kill()
        save_capture(capture, rec, log_row)

    ev.save_capture = killing_save_capture
    if mode == "mid_save":
        from src.data.recorder import datasets_write_compat

        orig = recording.DurableDataset.save_episode

        def killing_save(self: Any, before_checkpoint: Any = None) -> int:
            if seen["saves"] == kill_at:
                with datasets_write_compat():
                    self.inner.save_episode()  # data + video written, nothing finalized
                _kill()
            return int(orig(self, before_checkpoint))

        recording.DurableDataset.save_episode = killing_save  # type: ignore[method-assign]
    if mode not in ("after_record", "mid_save", "after_save"):
        raise SystemExit(f"unknown mode {mode}")
    ev.main(
        [
            "--preset", "real", "--mock", "--fake-clock", "--max-steps", "5",
            "--task", "PickPlace", "--trials", "1", "--labeler", "mock-labeler",
            "--family", "act", "--training-run-id", "run-act", "--checkpoint-dir", ckpt,
            "--runs-dir", runs,
        ]
    )  # fmt: skip
    print("NOT KILLED", flush=True)
    sys.exit(7)


if __name__ == "__main__":
    main()
