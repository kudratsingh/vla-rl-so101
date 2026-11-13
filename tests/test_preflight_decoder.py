"""Smoke test of scripts/preflight_decoder.py: the recorder's two-camera clip decodes (PASS, exit 0),
and a torchcodec that cannot load FFmpeg is a FAIL line with the fix (exit 1), never a traceback."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "preflight_decoder.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("preflight_decoder", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preflight_passes_and_fails_cleanly(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pf = _load()
    assert pf.main(["--preset", "real"]) == 0
    out = capsys.readouterr().out
    assert "[torchcodec] OK: torchcodec 0.2.1" in out and "FFmpeg " in out
    assert "[record + decode] OK" in out and "every frame of every clip decodes" in out
    assert out.rstrip().endswith("PASS: torchcodec + FFmpeg decode the recorder's two-camera clips")
    tmp = out.split("temp dataset removed: ")[1].split()[0]
    assert not Path(tmp).exists()  # cleaned up
    # the pixel check can tell frames apart (a decode of another frame would fail it)
    diff = abs(pf.synthetic(0, 0).astype(float) - pf.synthetic(0, pf.FRAMES // 2)).mean()
    assert diff > pf.MAX_PIXEL_DIFF

    monkeypatch.setitem(sys.modules, "torchcodec.decoders", None)  # import fails like a dlopen
    assert pf.main(["--preset", "real"]) == 1
    out = capsys.readouterr().out
    assert "[torchcodec] FAIL: torchcodec cannot load FFmpeg" in out and "  fix: " in out
    assert out.rstrip().endswith("FAIL") and "Traceback" not in out


def test_dataset_mode_checks_a_recorded_demo_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.eval import protocol_real as PR
    from src.robot import convention as conv

    pf = _load()
    pf.stage_record_decode(tmp_path)  # 2 episodes through the recorder's path, kept
    ds = tmp_path / "dataset"
    args = ["--preset", "real", "--dataset", str(ds), "--episodes", "2"]
    assert pf.main(args) == 1  # no provenance: not a recording of this repo
    assert "vla_rl_so101.json missing" in capsys.readouterr().out
    sched = PR.demo_schedule()
    prov = {
        "demo_schedule": True,
        "rehearsal": False,
        "session_id": "s",
        "protocol": PR.citation(),
        "convention": conv.citation(),
        "episodes": {"0": sched[0], "1": sched[1]},
        "sittings": [{"session_end": "supported"}],
    }
    (ds / "meta" / "vla_rl_so101.json").write_text(json.dumps(prov))
    assert pf.main(args) == 0
    out = capsys.readouterr().out
    assert "2 episodes; both cameras decode" in out and "= demos 1-2 of the frozen schedule" in out
    assert pf.main([*args[:-1], "50"]) == 1  # the count is checked
    assert "2 episodes, expected 50" in capsys.readouterr().out
    prov["episodes"] = {"0": sched[1], "1": sched[0]}
    (ds / "meta" / "vla_rl_so101.json").write_text(json.dumps(prov))
    assert pf.main(args) == 1
    assert "not demos [0, 1] of the frozen schedule" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform != "darwin", reason="DYLD_FALLBACK_LIBRARY_PATH is macOS only")
def test_preflight_without_the_ffmpeg7_path_names_the_export() -> None:
    env = {k: v for k, v in os.environ.items() if k != "DYLD_FALLBACK_LIBRARY_PATH"}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--preset", "real"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO,
    )
    assert r.returncode == 1, r.stdout + r.stderr
    assert "[torchcodec] FAIL" in r.stdout
    assert (
        "export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib" in r.stdout
        or "brew install ffmpeg@7" in r.stdout
    )
    assert "Traceback" not in r.stdout + r.stderr
