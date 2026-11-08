#!/usr/bin/env python
"""WO-07 pilot: a few PickPlace trials through the full deploy client (src/robot/client.py).

--mock      CPU, in-process, safe anywhere: MockPolicy behind a real gRPC server on 127.0.0.1,
            the FakeFollower arm and FakeCam camera from tests/robot_fakes.py, the real safety
            filter and FK. No serial port, no camera, no network beyond localhost. Outcomes are
            scripted (success = ran to the step limit with no stop). NOT a result.
--hardware  OWNER ONLY, never run by tests or agents: real arm, real camera, the policy server
            through an SSH tunnel (configs/real.yaml server). Refuses without --i-am-present, then
            needs 'go' typed. Moves the arm to the home pose before every trial; the operator
            scores every trial y/n; Ctrl-C holds the arm, then torque goes off after ENTER.

    uv run scripts/wo07_pilot.py --preset real --mock                     # real-time, ~1 min
    uv run scripts/wo07_pilot.py --preset real --mock --fake-clock --trials 3
    uv run scripts/wo07_pilot.py --preset real --hardware --i-am-present  # owner at the arm

Writes runs/<id>/record.json (every trial record + the summary with Wilson intervals) and
runs/<id>/metrics.jsonl (one row per decision and per trial). This is the Mac process (device
cpu, $0); the paid GPU side is the policy server's own run record (src/robot/serve.py), whose
run id is stored here under results.server_info.server_run.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.eval.protocol import CELLS, TASKS
from src.robot.camera import PolicyImagePipeline
from src.robot.client import (
    ConsoleOperator,
    DeployConfig,
    DeployLoop,
    Operator,
    PolicyClient,
    ScriptedOperator,
    check_server,
    run_trials,
    safe_shutdown,
    summarize,
)
from src.robot.driver import So101Driver
from src.robot.safety import FkFn, load_sim_fk, make_safety
from src.robot.serve import MockPolicy, PolicyServer, build_server
from src.robot.units import UnitMap
from src.runs import RUNS_DIR, Run, new_run_id

MOCK_LABEL = "MOCK: fake arm, fake camera, mock policy; exercises the client only, NOT a result"
HARDWARE_LABEL = "pilot, single run, real arm; fewer than 50 trials per cell, not a protocol table"

HARDWARE_REFUSAL = """--hardware drives the real SO-101 and refuses to run without --i-am-present.

Only the owner runs it, standing at the arm with a hand on the power switch, after WO-06
(calibration, intrinsics, camera match) and with the policy server reachable through the tunnel."""

HARDWARE_BANNER = """
==========================================================================================
 THIS WILL MOVE THE REAL ARM.
 Before every trial it drives to the home pose (deploy.home_qpos) through the safety filter,
 then the policy moves it. Keep the workspace clear and a hand on the power switch.
 Ctrl-C holds the arm; after ENTER, torque goes off and the arm drops: support it.
==========================================================================================
"""


def mock_waypoints(home: Sequence[float]) -> list[np.ndarray]:
    """A PickPlace-shaped joint path from home: hover, descend, close, lift, carry, open.
    Every TCP is inside workspace_box_m with the sim FK. "close" commands the sim's -0.17 rad,
    which the real gripper floor (-5 deg) clips, so every full-length trial (the walk restarts
    per trial) logs gripper joint_limit interventions."""
    h = np.asarray(home, dtype=np.float64)
    hover = np.array([0.3, -0.8, h[2], h[3], h[4], 0.6])
    descend = np.array([0.3, -0.55, h[2], h[3], h[4], 0.6])
    close = np.array([0.3, -0.55, h[2], h[3], h[4], -0.17])
    lift = np.array([0.3, -1.0, h[2], h[3], h[4], -0.17])
    carry = np.array([-0.3, -1.0, h[2], h[3], h[4], -0.17])
    release = np.array([-0.3, -0.8, h[2], h[3], h[4], 0.6])
    return [hover, descend, close, lift, carry, release]


def session(
    cfg: Config,
    loop: DeployLoop,
    operator: Operator,
    info: dict[str, Any],
    *,
    mode: str,
    task: str,
    cell: str,
    n_trials: int,
    max_steps: int | None,
    camera_calibrated: bool,
    runs_dir: Path,
    on_trial_start: Callable[[int], None] | None = None,
) -> Path:
    trials: list[dict[str, Any]] = []
    end = "error"
    with Run(
        work_order="WO-07",
        script=__file__,
        cfg=cfg,
        device="cpu",
        runs_dir=runs_dir,
        run_id=new_run_id("WO-07", f"pilot-{mode}"),
    ) as run:
        run.results.update(
            mode=mode, label=MOCK_LABEL if mode == "mock" else HARDWARE_LABEL, server_info=info
        )
        try:
            _, end = run_trials(
                loop,
                operator,
                task=task,
                cell=cell,
                n_trials=n_trials,
                info=info,
                camera_calibrated=camera_calibrated,
                log_row=run.log_metrics,
                max_steps=max_steps,
                trials=trials,
                on_trial_start=on_trial_start,
            )
        except KeyboardInterrupt:
            end = "operator_ctrl_c"
        finally:
            safe_shutdown(loop.arm, operator)
            summary = summarize(trials, loop.dcfg.latency_target_ms)
            run.results.update(session_end=end, trials=trials, summary=summary)
        print_summary(run.id, summary, end)
    return run.record_path


def print_summary(run_id: str, summary: dict[str, Any], end: str) -> None:
    print(f"\nrun {run_id}  (session end: {end})")
    for name, c in summary["cells"].items():
        rate = "n/a" if c["success_rate"] is None else f"{100 * c['success_rate']:.0f}%"
        lo, hi = c["wilson_95"]
        print(
            f"  {name}: {c['successes']}/{c['n']} = {rate}  Wilson95 [{100 * lo:.0f}, {100 * hi:.0f}]"
            f"  aborted {c['aborted']}  stops {c['stop_reasons']}"
        )
    lat = summary["capture_to_first_write_ms"]
    print(
        f"  capture->first write ms: median {lat['median']}  p95 {lat['p95']}  (n {lat['n']}, "
        f"target < {summary['latency_target_ms']:.0f})"
    )
    print(f"  realised write Hz per trial: median {summary['write_hz']['median']}")
    print(f"  protocol {summary['protocol_version']} {summary['protocol_hash'][:12]}")


def run_mock(
    cfg: Config,
    args: argparse.Namespace,
    *,
    fk: FkFn | None = None,
    operator: Operator | None = None,
) -> Path:
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from robot_fakes import FakeCam, FakeClock, FakeFollower

    if args.fake_clock:
        fake = FakeClock()
        clock, sleep = fake, fake.sleep
    else:
        clock, sleep = time.perf_counter, time.sleep
    dcfg = DeployConfig.from_config(cfg)
    policy = MockPolicy(
        mock_waypoints(dcfg.home_qpos),
        chunk_len=dcfg.chunk_len,
        infer_s=float(args.mock_infer_ms) / 1e3,
        sleep=sleep,
    )
    servicer = PolicyServer(
        policy, image_size=dcfg.image_size, policy_id="mock:pickplace-waypoints", clock=clock
    )
    server, port = build_server(servicer, "127.0.0.1", 0)
    try:
        arm = So101Driver(
            FakeFollower(),
            UnitMap.from_config(cfg),
            make_safety(cfg, fk or load_sim_fk(), clock=clock),
            clock=clock,
        )
        pipe = PolicyImagePipeline.from_config(cfg)
        with (
            PolicyClient(
                f"127.0.0.1:{port}",
                timeout_s=dcfg.request_timeout_s,
                chunk_len=dcfg.chunk_len,
                action_dim=dcfg.action_dim,
                clock=clock,
            ) as client,
            arm,
        ):
            info = client.info()
            check_server(info, dcfg)
            loop = DeployLoop(arm, FakeCam(clock), pipe, client, dcfg, clock=clock, sleep=sleep)
            return session(
                cfg,
                loop,
                operator or ScriptedOperator(),
                info,
                mode="mock",
                task=args.task,
                cell=args.cell,
                n_trials=args.trials,
                max_steps=args.max_steps,
                camera_calibrated=pipe.calibrated,
                runs_dir=args.runs_dir,
                on_trial_start=lambda _k: policy.reset(),  # the waypoint walk restarts per trial
            )
    finally:
        policy.release.set()
        server.stop(grace=0)


def run_hardware(cfg: Config, args: argparse.Namespace) -> Path:
    if not args.i_am_present:
        print(HARDWARE_REFUSAL)
        sys.exit(2)
    print(HARDWARE_BANNER)
    if input("type 'go' to continue: ").strip() != "go":
        sys.exit("not confirmed; nothing was opened")
    cam_cfg = cfg["cameras"]["front"]
    if not cam_cfg.get("intrinsics") and not args.allow_uncalibrated_camera:
        sys.exit(
            "cameras.front.intrinsics is null: the policy would see an uncalibrated crop. Run "
            "scripts/wo06_camera_intrinsics.py first (or pass --allow-uncalibrated-camera)"
        )
    from src.robot.camera import OpenCVFrameSource

    dcfg = DeployConfig.from_config(cfg)
    fk = load_sim_fk()
    pipe = PolicyImagePipeline.from_config(cfg)
    target = f"{cfg['server']['host']}:{cfg['server']['port']}"
    with PolicyClient(
        target,
        timeout_s=dcfg.request_timeout_s,
        chunk_len=dcfg.chunk_len,
        action_dim=dcfg.action_dim,
    ) as client:
        info = client.info()  # before the arm is connected: a wrong server never moves it
        check_server(info, dcfg)
        print(f"policy server OK: {info['policy_id']} (server run {info.get('server_run')})")
        if dcfg.policy_checkpoint is None:
            print("WARNING: server.policy_checkpoint is null: the served adapter is not checked")
        operator = ConsoleOperator()
        with OpenCVFrameSource(cam_cfg) as cam, So101Driver.from_config(cfg, fk) as arm:
            loop = DeployLoop(arm, cam, pipe, client, dcfg)
            return session(
                cfg,
                loop,
                operator,
                info,
                mode="hardware",
                task=args.task,
                cell=args.cell,
                n_trials=args.trials,
                max_steps=args.max_steps,
                camera_calibrated=pipe.calibrated,
                runs_dir=args.runs_dir,
            )


def main(argv: Sequence[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    add_config_args(parser)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="in-process fakes (CPU, safe)")
    mode.add_argument("--hardware", action="store_true", help="REAL ARM; owner only")
    parser.add_argument("--i-am-present", action="store_true", help="owner at the arm")
    parser.add_argument("--allow-uncalibrated-camera", action="store_true")
    parser.add_argument("--task", choices=TASKS, default=None)
    parser.add_argument("--cell", choices=CELLS, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="default deploy.trial_max_steps"
    )
    parser.add_argument("--fake-clock", action="store_true", help="mock: instant, exact latency")
    parser.add_argument("--mock-infer-ms", type=float, default=60.0)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.preset != "real":
        parser.error("WO-07 runs with --preset real (configs/real.yaml robot / deploy blocks)")
    cfg = load_config(args.preset, args.config)
    pilot = cfg["deploy"].get("pilot") or {}
    args.task = args.task or str(pilot.get("task", "PickPlace"))
    args.cell = args.cell or str(pilot.get("cell", "seen"))
    args.trials = int(args.trials if args.trials is not None else pilot.get("trials", 5))
    if args.hardware:
        return run_hardware(cfg, args)
    return run_mock(cfg, args)


if __name__ == "__main__":
    main()
