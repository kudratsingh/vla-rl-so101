#!/usr/bin/env python
"""WO-06 / WO-07: in-process SmolVLA / ACT latency on the MOCK rig (fake arm, two fake cameras).

The real deploy loop (``src/robot/client.py`` ``DeployLoop`` + ``run_trials`` with a scripted
operator), the real driver, safety filter and sim FK, the real policy on this Mac's MPS (CPU
fallback) through ``src/robot/local_policy.py``; only the arm (``FakeFollower``) and the cameras
(two fake 640x480 readers in a ``CameraRig``) are fakes. No serial port, no camera, no network
(beyond a one-time Hub download of public weights). Measures, over ``--ticks`` bus writes:

* chunk latency: the in-process query per decision (``server_ms``: convention preprocessing,
  LeRobot processors, the model, postprocessing) and the model call alone (``infer_ms``);
* command gap: time between consecutive goal writes to the (fake) bus, stamped inside
  ``FakeFollower.send_action``, within each trial.

Camera capture is EXCLUDED (the fake cameras return at once). Single run, mock rig: NOT a result
and never an arm number. ``--capture`` (review F3) runs it the way a real-0.1 session does: the
evaluation capture (``src/robot/recording.py`` ``EvalCapture``: both cameras + state + action
into a LeRobotDataset, LeRobot's 8 PNG writer threads during the trial, the AV1 save and the
per-episode checkpoint after it) on TEXTURED frames (a ring of 30 random-noise images per
camera, the worst case for PNG / AV1), and also reports the per-episode save time. Every action goes through the safety filter, so the pretrained base's
(or an untrained ACT's) actions, which are not ours, are clipped or rejected like any others. To
time a full chunk rather than one write per trial, this script (only) sets
``deploy.stop_on_reject: false``: a rejected action still writes the filter's hold command on its
tick, so the loop's pacing and the bus writes are unchanged; the safety status counts are kept.

    uv run scripts/wo07_local_latency.py --preset real --family smolvla          # MPS, smolvla_base
    uv run scripts/wo07_local_latency.py --preset real --family act              # fresh ACT
    uv run scripts/wo07_local_latency.py --preset real --family smolvla --capture  # as the eval
    uv run scripts/wo07_local_latency.py --preset tiny                           # CPU, tiny ACT, <2 min

Writes runs/<id>/record.json (results.latency: p50 / p95 / max of both, plus every trial record)
and runs/<id>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CONFIG_DIR, Config, add_config_args, load_config
from src.robot import convention as conv
from src.robot.camera import CameraFault, CameraRig, Frame
from src.robot.client import DeployLoop, ScriptedOperator, run_trials, summarize
from src.robot.driver import So101Driver
from src.robot.local_policy import (
    LocalPolicyClient,
    LocalPolicySpec,
    check_local,
    init_act_checkpoint,
    load_local_policy,
    local_deploy_config,
    local_pipe,
)
from src.robot.recording import EvalCapture
from src.robot.safety import FkFn, load_sim_fk, make_safety
from src.robot.servo_health import monitor_from_config
from src.robot.units import UnitMap
from src.runs import RUNS_DIR, Run, new_run_id

LABEL = (
    "MOCK RIG, single run: fake arm + two fake cameras, real deploy loop / safety filter / model "
    "on this Mac. EXCLUDES camera capture (exposure, USB, driver buffering). Not an arm number"
)
# tiny preset: an ACT small enough for a CPU smoke (< 2 min), same features and chunk
TINY_ACT = {
    "dim_model": 32,
    "n_heads": 2,
    "dim_feedforward": 64,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "use_vae": False,
}
TINY_TICKS = 40


class StillCam:
    """A fake ``FrameReader``: a fixed 640x480 RGB noise image (``moving``: the next of a ring of
    30), a fresh ``clock`` stamp per read."""

    def __init__(
        self, seed: int, clock: Callable[[], float] = time.perf_counter, moving: bool = False
    ):
        rng = np.random.default_rng(seed)
        n = 30 if moving else 1
        self.ring = [rng.integers(0, 256, conv.IMAGE_SHAPE_HWC, dtype=np.uint8) for _ in range(n)]
        self.reads = 0
        self.clock = clock
        self.connected = False
        self.fail = False

    @property
    def image(self) -> np.ndarray:
        return self.ring[self.reads % len(self.ring)]

    def connect(self) -> None:
        self.connected = True

    def read(self) -> Frame:
        if self.fail:
            raise CameraFault("timed out waiting for a frame")
        img = self.image
        self.reads += 1
        return Frame(img, self.clock())

    def disconnect(self) -> None:
        self.connected = False


def mock_rig(clock: Callable[[], float] = time.perf_counter, moving: bool = False) -> CameraRig:
    return CameraRig({name: StillCam(i, clock, moving) for i, name in enumerate(conv.CAMERA_NAMES)})


def pct(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "max": round(float(a.max()), 3),
    }


def mac_info() -> dict[str, Any]:
    def sysctl(key: str) -> str | None:
        try:
            out = subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    mem = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "model": sysctl("hw.model"),
        "memory_gb": None if mem is None else round(int(mem) / 2**30, 1),
        "macos": platform.mac_ver()[0] or None,
        "platform": platform.platform(),
    }


def tiny_overrides(cfg: Config) -> Config:
    """tiny: CPU, a fresh tiny ACT, a short run."""
    lp = dict(cfg.get("local_policy") or {})
    lp.update(family="act", device="cpu", warmup=1, request_timeout_ms=700)
    cfg["local_policy"] = lp
    return cfg


def run(
    cfg: Config,
    *,
    family: str,
    checkpoint: str | None,
    ticks: int,
    runs_dir: Path,
    fk: FkFn | None = None,
    act_overrides: dict[str, Any] | None = None,
    capture: bool = False,
) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from robot_fakes import FakeFollower

    cfg["deploy"] = {**cfg["deploy"], "stop_on_reject": False}  # see the module doc
    with Run(
        work_order="WO-07",
        script=__file__,
        cfg=cfg,
        device=str(cfg["local_policy"].get("device", "mps")),
        runs_dir=runs_dir,
        run_id=new_run_id("WO-07", f"local-latency-{cfg['preset']}"),
    ) as run_:
        base = checkpoint or (cfg["local_policy"].get("latency_bases") or {}).get(family)
        fresh = base is None
        if fresh:
            if family != "act":
                raise SystemExit(f"no checkpoint for {family}: pass --checkpoint")
            home = cfg["deploy"]["home_qpos"]  # placeholder stats: actions scatter around home
            base = str(
                init_act_checkpoint(
                    run_.dir / "act_init", pose=home, pose_std_rad=0.05, **(act_overrides or {})
                )
            )
        spec = LocalPolicySpec.from_config(cfg, base=base, family=family)
        t_load = time.perf_counter()
        policy = load_local_policy(spec)
        load_s = time.perf_counter() - t_load
        dcfg = local_deploy_config(cfg, policy)
        writes: list[float] = []
        fol = FakeFollower()
        send = fol.send_action

        def stamped_send(action: dict[str, Any]) -> dict[str, Any]:
            out = send(action)
            writes.append(time.perf_counter())  # after the (fake) bus accepted the goal
            return out

        fol.send_action = stamped_send  # type: ignore[method-assign]
        arm = So101Driver(
            fol,
            UnitMap.from_config(cfg),
            make_safety(cfg, fk or load_sim_fk()),
            monitor=monitor_from_config(cfg),
        )
        trials: list[dict[str, Any]] = []
        marks: list[int] = []
        decisions: list[dict[str, Any]] = []

        def log_row(**row: Any) -> None:
            if row.get("kind") == "decision":
                decisions.append(row)
            run_.log_metrics(**row)

        saves: list[float] = []
        with (
            LocalPolicyClient(policy, timeout_s=dcfg.request_timeout_s, run_id=run_.id) as client,
            mock_rig(moving=capture) as rig,
            arm,
        ):
            info = client.info()
            check_local(info, dcfg, spec)
            loop = DeployLoop(arm, rig, local_pipe, client, dcfg)
            cap: EvalCapture | None = None
            if capture:
                cap = EvalCapture.open(
                    run_.dir / "capture",
                    "local/latency-capture",
                    {"source": __file__, "purpose": "latency measurement", "session_id": run_.id},
                    resume=False,
                    follower=fol,
                    end_state=lambda: rig.read().images,
                )
                loop.on_write = cap.on_write
                loop.on_settled = cap.on_settled

            def trial_start(_k: int) -> None:
                marks.append(len(writes))
                if cap is not None:
                    cap.begin(f"latency/{_k}", conv.DEMO_TASK_STRING)

            k, end = 0, "completed"
            while sum(t["steps"] for t in trials) < ticks and end == "completed":
                _, end = run_trials(
                    loop,
                    ScriptedOperator(),
                    task=conv.DEMO_TASK,
                    cell="seen",
                    n_trials=k + 1,
                    indices=[k],
                    info=info,
                    camera_calibrated=False,
                    log_row=log_row,
                    trials=trials,
                    on_trial_start=trial_start,
                    max_steps=min(dcfg.trial_max_steps, ticks - sum(t["steps"] for t in trials)),
                )
                if cap is not None and trials:
                    t0 = time.perf_counter()
                    trials[-1]["capture"] = cap.finish(trials[-1])
                    saves.append(time.perf_counter() - t0)
                k += 1
                if trials and not trials[-1]["policy_started"]:
                    break  # the policy never started: more trials would only repeat it
            if cap is not None:
                cap.close()
        gaps: list[float] = []
        for mark, t in zip(marks, trials, strict=False):
            w = writes[mark : mark + int(t["steps"])]
            gaps += [(b - a) * 1e3 for a, b in itertools.pairwise(w)]
        summary = summarize(trials, dcfg.latency_target_ms, dcfg.capture_offset_ms)
        server_ms = [float(r["server_ms"]) for r in decisions]
        model_ms = [float(r["infer_ms"]) for r in decisions]
        steps = sum(int(t["steps"]) for t in trials)
        n_dec = sum(int(t["decisions"]) for t in trials)
        latency = {
            "label": LABEL,
            "family": family,
            "checkpoint": base if not fresh else "fresh ACT (untrained, placeholder stats)",
            "fresh_init": fresh,
            "device": policy.device,
            "dtype": policy.describe()["dtype"],
            "mac": mac_info(),
            "load_s": round(load_s, 2),
            "warmup_ms": policy.warmup_ms,
            "chunk_ms": pct(server_ms),
            "model_ms": pct(model_ms),
            "command_gap_ms": pct(gaps),
            "ticks": steps,
            "decisions": n_dec,
            "actions_per_chunk_mean": round(steps / n_dec, 2) if n_dec else None,
            "trials": len(trials),
            "stop_reasons": dict(Counter(str(t["stop_reason"]) for t in trials)),
            "chunk_len": dcfg.chunk_len,
            "exec_steps": dcfg.exec_steps,
            "request_timeout_ms": dcfg.request_timeout_s * 1e3,
            "max_chunk_age_ms": dcfg.max_chunk_age_s * 1e3,
            "control_hz": dcfg.control_hz,
            "stop_on_reject": dcfg.stop_on_reject,
            "safety_status": dict(
                sum((Counter(t["safety"]["status_counts"]) for t in trials), Counter())
            ),
            "capture": (
                {
                    "on": True,
                    "frames": "textured: a ring of 30 random-noise 640x480 images per camera",
                    "episode_save_s": pct(saves),
                    "frames_per_episode": [int(t["steps"]) for t in trials],
                }
                if capture
                else {"on": False}
            ),
        }
        run_.results.update(
            label=LABEL,
            policy_info=info,
            latency=latency,
            summary=summary,
            session_end=end,
            trials=trials,
        )
    if fresh:  # untrained weights: only their hash (results.policy_info) is worth keeping
        shutil.rmtree(run_.dir / "act_init", ignore_errors=True)
    shutil.rmtree(run_.dir / "capture", ignore_errors=True)  # noise frames: only the timing
    _print(run_.id, latency)
    return run_.record_path


def _print(run_id: str, lat: dict[str, Any]) -> None:
    print(f"\nrun {run_id}  {lat['family']} on {lat['device']} ({lat['dtype']})  {LABEL}")
    for key in ("chunk_ms", "model_ms", "command_gap_ms"):
        s = lat[key]
        print(f"  {key:<15} n={s['n']:<4} p50 {s['p50']}  p95 {s['p95']}  max {s['max']}")
    print(
        f"  {lat['ticks']} writes, {lat['decisions']} chunks, {lat['actions_per_chunk_mean']} "
        f"actions/chunk, stops {lat['stop_reasons']}, capture {lat['capture']}"
    )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    parser.add_argument("--family", choices=("smolvla", "act"), default=None)
    parser.add_argument(
        "--checkpoint", default=None, help="default local_policy.latency_bases.<family>"
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default=None)
    parser.add_argument("--ticks", type=int, default=None, help="bus writes to collect (300)")
    parser.add_argument(
        "--capture", action="store_true", help="with the evaluation capture on textured frames"
    )
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset not in ("real", "tiny"):
        parser.error("--preset real (the Mac) or tiny (CPU smoke)")
    # tiny reads configs/real.yaml (robot / cameras / deploy blocks) with tiny overrides
    config = args.config or (CONFIG_DIR / "real.yaml" if args.preset == "tiny" else None)
    cfg = load_config(args.preset, config)
    tiny = args.preset == "tiny"
    if tiny:
        cfg = tiny_overrides(cfg)
    if args.device is not None:
        cfg["local_policy"]["device"] = args.device
    family = args.family or str(cfg["local_policy"]["family"])
    ticks = args.ticks if args.ticks is not None else (TINY_TICKS if tiny else 300)
    return run(
        cfg,
        family=family,
        checkpoint=args.checkpoint,
        ticks=ticks,
        runs_dir=args.runs_dir,
        act_overrides=TINY_ACT if tiny else None,
        capture=args.capture,
    )


if __name__ == "__main__":
    main()
