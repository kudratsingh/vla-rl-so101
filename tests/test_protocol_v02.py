"""Protocol 0.2: composite hash sensitivity, 0.1-draft untouched, clip read from real.yaml, and the
run-settings conformance check."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from src.config import CONFIG_DIR, REPO_ROOT, load_config
from src.eval import protocol as P01
from src.eval import protocol_v02 as P02

FROZEN_01 = "e9ce5bb47a88da03d3cad77c4084b32d7184eeb2d30efa4f93f6f487be7e517c"


def _copy_inputs(root: Path) -> Path:
    """Every file the 0.2 hash reads, copied under `root` at the same relative path."""
    for rel in [*P02.hashed_files(), P02.REAL_CONFIG, "uv.lock"]:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, dst)
    return root


def _edit(path: Path, pattern: str, repl: str) -> None:
    text, n = re.subn(pattern, repl, path.read_text(), count=1, flags=re.MULTILINE)
    assert n == 1, f"{pattern!r} not found in {path}"
    path.write_text(text)


def test_v02_hash_covers_env_success_and_clip(tmp_path: Path) -> None:
    ref = P02.protocol_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", ref) and ref != FROZEN_01
    root = _copy_inputs(tmp_path / "repo")
    assert P02.protocol_hash(root) == ref  # the copy is complete: nothing else is read

    edits = [
        ("src/env/success.py", r"^REST_LIN_VEL = 0\.03", "REST_LIN_VEL = 0.04"),
        ("src/env/tasks.py", r'^MIN_SEP = \{"PickPlace": 0\.11', 'MIN_SEP = {"PickPlace": 0.12'),
        ("src/env/cameras.py", r'"front_left_45": \(45\.0,', '"front_left_45": (46.0,'),
        (P02.REAL_CONFIG, r"max_joint_step_deg: 4\.0", "max_joint_step_deg: 3.0"),
        ("src/eval/sim_eval.py", r"clamp\(-bound, bound\)", "clamp(-2 * bound, 2 * bound)"),
        ("src/eval/sim_eval.py", r"success \|= hit", "success = hit"),
    ]
    seen = {ref}
    for rel, pattern, repl in edits:
        original = (root / rel).read_bytes()
        _edit(root / rel, pattern, repl)
        changed = P02.protocol_hash(root)
        assert changed not in seen, f"editing {rel} did not change the 0.2 hash"
        seen.add(changed)
        (root / rel).write_bytes(original)
    assert P02.protocol_hash(root) == ref
    # 0.1-draft hashes protocol.py's bytes only: the env edits above never reach it
    assert P01.protocol_hash() == FROZEN_01 and P01.PROTOCOL_VERSION == "0.1-draft"


def test_v02_keeps_01_semantics_and_reads_the_real_clip() -> None:
    for name in ("TASKS", "CELLS", "SIM_SEEDS", "SIM_EPISODES_PER_CELL", "REAL_TRIALS_PER_CELL"):
        assert getattr(P02, name) == getattr(P01, name)
    assert P02.wilson_interval is P01.wilson_interval
    safety = yaml.safe_load((CONFIG_DIR / "real.yaml").read_text())["robot"]["safety"]
    clip = P02.action_clip()
    assert clip.max_joint_step_deg == float(safety["max_joint_step_deg"])
    assert abs(clip.bound_deg - 4.0) < 1e-9  # 4 deg/step = 60 deg/s at 15 Hz
    parts = P02.hash_components()
    assert parts["action_clip"]["bound_deg"] == clip.bound_deg
    assert "src/env/success.py" in parts["files"] and "src/env/tasks.py" in parts["files"]
    assert "src/eval/sim_eval.py" in parts["files"]  # the episode loop and the clip itself
    assert any(f.endswith(".urdf") for f in parts["files"])


def test_v02_deviations() -> None:
    sft = load_config("sim", CONFIG_DIR / "sft.yaml")  # the config the reported evals ran with
    assert P02.deviations(sft, sim_device="cuda", exec_steps=None) == []
    off = P02.deviations(sft, sim_device="cpu", exec_steps=4)
    assert any("backend" in d for d in off) and any("exec_steps" in d for d in off)
    tiny = load_config("tiny")
    assert any("image_size" in d for d in P02.deviations(tiny, sim_device="cuda", exec_steps=None))
    moved = {**sft, "env": {**sft["env"], "camera": {"pose": "front_left_45", "fov_deg": 45}}}
    assert any("camera" in d for d in P02.deviations(moved, sim_device="cuda", exec_steps=None))
