"""TEST ONLY: a real-0.1 MOCK session (scripts/wo07_real_eval.py --mock, capture on) that SIGKILLs
itself at one point of its 3rd trial's capture (tests/test_capture_resume_gap.py).

    python capture_kill_child.py RUNS_DIR CKPT_DIR {after_record,mid_save,after_save}

* ``after_record``: the scored record is logged and flushed, the episode save has not started
  (the owner's 2026-11-09 interruption);
* ``mid_save``: inside the save: LeRobot has written the episode's data and video, nothing is
  finalized and the provenance entry is not written;
* ``after_save``: the episode is saved, finalized and in the provenance; the ``capture`` row that
  would name its index is not logged.

SIGKILL: no ``finally``, no shutdown, no flush; what is on disk is what a power cut leaves. Fakes
only (fake arm, fake rig, MockPolicy behind gRPC on 127.0.0.1); fake clock.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "tests")]

KILL_AT_TRIAL = 3


def _kill() -> None:
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def main() -> None:
    runs, ckpt, mode = sys.argv[1], sys.argv[2], sys.argv[3]
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
        if seen["saves"] == KILL_AT_TRIAL:
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
            if seen["saves"] == KILL_AT_TRIAL:
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
