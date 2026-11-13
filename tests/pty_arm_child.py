"""TEST ONLY: the WO-07 pilot's or real eval's real ``--hardware`` path on fakes, on a pty.

    python pty_arm_child.py {pilot,real_eval} MARK_FD RUNS_DIR [--wait-foreground]
        [--timeout S] [--grace S] [--steps N]

Started by tests/pty_shell.py (tests/test_job_control.py). Everything the operator sees and
types goes through the real code on the real terminal: ``require_terminal``, the ``type 'go'``
prompt, ``operator_signals`` / ``job_control``, ``DeployLoop``, ``ConsoleOperator`` ->
``TorqueGuard`` -> ``timed_input``, the shutdown and the run record. Only the hardware is fake:
the follower (tests/robot_fakes.FakeFollower; no serial port), the camera (FakeCam), the FK
(toy_fk) and the policy (MockPolicy behind a real gRPC server on 127.0.0.1). Real clocks.

MARK_FD gets one line per event the test waits on: ``CONNECT`` (the follower's port opens and
torque goes on), ``WRITE <n>`` (every goal write), ``DONE <record path>``.
``--wait-foreground``: wait (<= 20 s) for tests/pty_shell.py to hand this process group the
terminal before starting, as a shell does before it runs a foreground job.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "tests")]

HOME = [0.0, 0.1, 0.0, 0.2, 0.0, 0.3]  # inside real.yaml's workspace box under toy_fk
TARGET = [0.1, 0.2, 0.1, 0.3, 0.0, 0.5]
INTRINSICS = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "dist": [0.0] * 5}
INTRINSICS.update(width=640, height=480)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", choices=["pilot", "real_eval"])
    ap.add_argument("mark_fd", type=int)
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("--wait-foreground", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--grace", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    if a.wait_foreground:
        t_end = time.monotonic() + 20.0
        while os.tcgetpgrp(0) != os.getpgrp():
            if time.monotonic() > t_end:
                sys.exit("pty_arm_child: never got the terminal's foreground")
            time.sleep(0.01)

    from real_fixture import load_config_placeholders
    from robot_fakes import FakeCam, FakeFollower, toy_fk

    from src.robot import camera
    from src.robot.client import DeployConfig
    from src.robot.driver import So101Driver
    from src.robot.safety import make_safety
    from src.robot.serve import MockPolicy, PolicyServer, build_server
    from src.robot.servo_health import monitor_from_config
    from src.robot.units import UnitMap

    def mark(line: str) -> None:
        os.write(a.mark_fd, (line + "\n").encode())

    cfg = load_config_placeholders("real")
    cfg["deploy"] = {
        **cfg["deploy"],
        "home_qpos": HOME,
        "trial_max_steps": a.steps,
        "prompt_torque_timeout_s": a.timeout,
        "prompt_support_grace_s": a.grace,
    }
    cfg["robot"]["park"] = {**cfg["robot"]["park"], "enabled": False}
    cfg["cameras"]["front"] = {**cfg["cameras"]["front"], "intrinsics": INTRINSICS}
    cfg["eval"] = {**(cfg.get("eval") or {}), "settle_s": 0.2}
    dcfg = DeployConfig.from_config(cfg)
    policy = MockPolicy([TARGET], chunk_len=dcfg.chunk_len, infer_s=0.005)
    servicer = PolicyServer(
        policy,
        image_size=dcfg.image_size,
        policy_id="mock:pty",
        adapter={"adapter_sha256": "mock:no-weights", "adapter_revision": None},
    )
    server, port = build_server(servicer, "127.0.0.1", 0)
    cfg["server"] = {**cfg["server"], "host": "127.0.0.1", "port": port}

    class MarkedFollower(FakeFollower):
        def connect(self, calibrate: bool = True) -> None:
            mark("CONNECT")
            super().connect(calibrate)

        def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
            out = super().send_action(action)
            mark(f"WRITE {len(self.sent)}")
            return out

    class Driver(So101Driver):
        @classmethod
        def from_config(cls, cfg: Any, fk: Any) -> Driver:
            return cls(
                MarkedFollower(),
                UnitMap(),
                make_safety(cfg, toy_fk),
                monitor=monitor_from_config(cfg),
            )

    class Cam:
        def __init__(self, _cfg: Any) -> None:
            self.cam = FakeCam()

        def __enter__(self) -> FakeCam:
            return self.cam

        def __exit__(self, *_exc: object) -> None:
            return None

    camera.OpenCVFrameSource = Cam  # type: ignore[misc,assignment]
    try:
        if a.entry == "pilot":
            pilot = _load("wo07_pilot")
            pilot.So101Driver, pilot.load_sim_fk = Driver, lambda: toy_fk
            args = argparse.Namespace(
                i_am_present=True,
                allow_uncalibrated_camera=True,
                task="PickPlace",
                cell="seen",
                trials=1,
                max_steps=None,
                runs_dir=a.runs_dir,
            )
            path = pilot.run_hardware(cfg, args)
        else:
            ev = _load("wo07_real_eval")
            ev.So101Driver, ev.load_sim_fk = Driver, lambda: toy_fk
            ev.load_config = lambda *_a, **_k: cfg
            argv = ["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"]
            argv += ["--protocol", "0.2", "--cells", "seen", "--trials", "1"]
            argv += ["--runs-dir", str(a.runs_dir)]
            path = ev.main(argv)
        mark(f"DONE {path}")
    finally:
        policy.release.set()
        server.stop(grace=0)


if __name__ == "__main__":
    main()
