"""Smoke test of scripts/preflight_decoder.py: the recorder's two-camera clip decodes (PASS, exit 0),
and a torchcodec that cannot load FFmpeg is a FAIL line with the fix (exit 1), never a traceback."""

from __future__ import annotations

import importlib.util
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
