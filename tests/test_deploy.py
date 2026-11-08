"""WO-07 deploy client + policy server against a fake arm, fake camera and a mock policy.

Real gRPC on 127.0.0.1 (ephemeral port), the real safety filter; one shared FakeClock for the
driver, deadman, camera, server and loop, so latencies are exact. No serial port, no camera.
"""

from __future__ import annotations

import importlib.util
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from robot_fakes import FakeCam, FakeClock, FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT
from src.eval import protocol
from src.robot.camera import PolicyImagePipeline
from src.robot.client import (
    DeployConfig,
    DeployLoop,
    PolicyClient,
    ScriptedOperator,
    ServerFault,
    ServerMismatch,
    TrialResult,
    check_server,
    run_trials,
    safe_shutdown,
    summarize,
)
from src.robot.driver import So101Driver
from src.robot.safety import make_safety
from src.robot.serve import DEFAULT_HOST, MockPolicy, PolicyServer, build_server, pack, unpack
from src.robot.units import UnitMap

_spec = importlib.util.spec_from_file_location(
    "wo07_pilot", REPO_ROOT / "scripts" / "wo07_pilot.py"
)
assert _spec is not None and _spec.loader is not None
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

# Inside real.yaml's workspace box under the TEST-ONLY toy FK (sim REST_QPOS is not).
TEST_HOME = [0.0, 0.1, 0.0, 0.2, 0.0, 0.3]
TEST_TARGET = [0.1, 0.2, 0.1, 0.3, 0.0, 0.5]
INSTRUCTION = protocol.instruction("PickPlace", protocol.RED_CUBE)


@dataclass
class Rig:
    clock: FakeClock
    fol: FakeFollower
    arm: So101Driver
    cam: FakeCam
    policy: MockPolicy
    client: PolicyClient
    loop: DeployLoop
    dcfg: DeployConfig
    server_info: dict[str, Any]


@contextmanager
def rig(
    waypoints: list[list[float]] | None = None,
    infer_s: float = 0.05,
    server_chunk_len: int = 8,
    **deploy: Any,
) -> Iterator[Rig]:
    clock = FakeClock()
    cfg = real_cfg()
    cfg["deploy"] = {**cfg["deploy"], "home_qpos": TEST_HOME, **deploy}
    dcfg = DeployConfig.from_config(cfg)
    policy = MockPolicy(
        waypoints or [TEST_TARGET], chunk_len=server_chunk_len, infer_s=infer_s, sleep=clock.sleep
    )
    servicer = PolicyServer(policy, image_size=dcfg.image_size, policy_id="mock:test", clock=clock)
    server, port = build_server(servicer, port=0)
    fol = FakeFollower()
    arm = So101Driver(fol, UnitMap(), make_safety(cfg, toy_fk, clock=clock), clock=clock)
    arm.connect()
    cam = FakeCam(clock)
    client = PolicyClient(
        f"127.0.0.1:{port}", timeout_s=dcfg.request_timeout_s, chunk_len=dcfg.chunk_len, clock=clock
    )
    pipe = PolicyImagePipeline.from_config(cfg)
    loop = DeployLoop(arm, cam, pipe, client, dcfg, clock=clock, sleep=clock.sleep)
    try:
        yield Rig(clock, fol, arm, cam, policy, client, loop, dcfg, servicer.info())
    finally:
        policy.release.set()
        client.close()
        server.stop(grace=0)


def _deg(fol: FakeFollower, key: str) -> list[float]:
    return [a[key] for a in fol.sent]


def test_wire_and_server_info_check() -> None:
    img = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    head, arrays = unpack(pack({"a": 1}, {"img": img, "s": np.ones(6, np.float32)}))
    assert (
        head == {"a": 1} and np.array_equal(arrays["img"], img) and arrays["s"].dtype == np.float32
    )
    assert DEFAULT_HOST == "127.0.0.1"  # reached through an SSH tunnel, never a public interface
    with rig() as r:
        info = r.client.info()
        check_server(info, r.dcfg)
        assert info["protocol_hash"] == protocol.protocol_hash()
        with pytest.raises(ServerMismatch, match="protocol_hash"):
            check_server({**info, "protocol_hash": "0" * 64}, r.dcfg)
        bad = np.zeros((64, 64, 3), np.uint8)  # wrong image size: server says INVALID_ARGUMENT
        with pytest.raises(ServerFault, match="INVALID_ARGUMENT"):
            r.client.act(bad, np.zeros(6, np.float32), INSTRUCTION)
    with rig(server_chunk_len=4) as r, pytest.raises(ServerMismatch, match="chunk_len"):
        check_server(r.client.info(), r.dcfg)


def test_trial_goes_through_safety_and_measures_latency() -> None:
    with rig(infer_s=0.05) as r:
        home = r.loop.go_home()
        assert home["reached"] and max(home["residual_deg"]) <= 3.0
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION, max_steps=24)
        assert res.stop_reason == "max_steps" and res.steps == 24 and res.decisions == 3
        assert len(r.fol.sent) - n0 == 24 + 1  # 24 policy writes + the end-of-trial hold
        # A whole 8-action chunk (8 / 15 Hz = 533 ms) runs without tripping the 300 ms deadman.
        assert not r.arm.deadman.tripped and res.stale_chunk_drops == 0
        assert max(abs(np.diff(_deg(r.fol, "elbow_flex.pos")))) <= 4.0 + 1e-9
        assert "velocity_cap" not in res.interventions_by_kind  # writes are a full period apart
        assert res.latency_ms() == pytest.approx([50.0] * 3)  # capture -> first bus write
        assert res.latency_ms("infer_ms") == pytest.approx([50.0] * 3)
        assert set(r.policy.instructions) == {INSTRUCTION}


def test_out_of_bounds_action_is_clipped_and_logged() -> None:
    gripper_oob = [*TEST_TARGET[:5], -0.5]  # below the -5 deg gripper floor
    with rig(waypoints=[gripper_oob]) as r:
        res = r.loop.run_trial(INSTRUCTION, max_steps=48)  # 45 deg -> floor at 2 deg per step
        assert res.stop_reason == "max_steps"
        assert res.interventions_by_kind["joint_limit"] > 0
        assert res.interventions_by_group["gripper"] > 0
        iv = next(i for i in res.first_interventions if i["joint"] == "gripper")
        assert iv["kind"] == "joint_limit" and iv["applied_deg"] == pytest.approx(-5.0, abs=1e-3)
        floor = UnitMap().gripper_rad_to_pct(math.radians(-5.0))
        assert min(_deg(r.fol, "gripper.pos")) >= floor - 1e-4
    below_table = [0.0, 1.2, 0.0, 0.0, 0.0, 0.3]  # toy FK: TCP z < 0
    with rig(waypoints=[below_table]) as r:
        res = r.loop.run_trial(INSTRUCTION, max_steps=150)
        assert res.stop_reason == "safety_reject" and res.interventions_by_kind["workspace_reject"]
        for sent in r.fol.sent:
            assert toy_fk(UnitMap().to_canonical(sent))[2] >= 0.0


def test_stalled_or_lost_camera_trips_the_deadman_within_the_timeout() -> None:
    with rig() as r:
        r.cam.stall_after = 10  # the stream freezes: same frame, same stamp
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION)
        assert res.stop_reason == "deadman" and "stale" in res.stop_detail
        assert res.deadman_response == "hold" and r.fol.bus.torque_enabled
        timeout = r.arm.safety.config.frame_timeout_s
        assert r.cam.last_fresh_time is not None
        assert res.t_end - r.cam.last_fresh_time <= timeout + 1 / r.dcfg.control_hz + 1e-9
        assert len(r.fol.sent) - n0 == res.steps + 1  # policy writes + one hold, then nothing
        n = len(r.fol.sent)
        assert r.loop.run_trial(INSTRUCTION).stop_reason == "deadman"  # latched
        assert len(r.fol.sent) == n
    with rig() as r:
        r.cam.fail_after = 5  # CameraFault: a lost frame
        res = r.loop.run_trial(INSTRUCTION)
        assert res.stop_reason == "deadman" and "no camera frame" in res.stop_detail


def test_serial_error_stops_the_loop_and_the_session() -> None:
    with rig() as r:

        def unplug(timing: dict[str, float]) -> None:
            if timing["decision"] == 2:
                r.fol.fail_write = True

        r.loop.on_decision = unplug
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=3,
            info=r.server_info,
            camera_calibrated=False,
        )
        assert len(trials) == 1 and end.startswith("torque_off")
        t = trials[0]
        assert t["stop_reason"] == "serial_fault" and t["deadman_response"] == "torque_off"
        assert not r.fol.bus.torque_enabled and r.policy.calls == 2


@pytest.mark.parametrize("failure", ["error", "hang", "late", "bad_shape"])
def test_server_failure_holds_and_stops(failure: str) -> None:
    timeout_ms = 200 if failure == "hang" else 400  # "hang" waits on the real gRPC deadline
    with rig(request_timeout_ms=timeout_ms) as r:

        def break_server(timing: dict[str, float]) -> None:
            if failure == "late":
                r.policy.infer_s = 0.5  # > 400 ms on the loop's clock
            else:
                r.policy.mode = failure

        r.loop.on_decision = break_server
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION)
        expected = {
            "error": "server_error",
            "hang": "server_timeout",
            "late": "server_timeout",
            "bad_shape": "server_bad_reply",
        }
        assert res.stop_reason == expected[failure] and res.deadman_response == "hold"
        assert res.decisions == 1 and res.steps == 8  # the first chunk, then the stop
        assert len(r.fol.sent) - n0 == res.steps + 1  # one hold write, nothing after it
        assert r.fol.sent[-1] == pytest.approx(r.fol.pos) and r.fol.bus.torque_enabled
        n = len(r.fol.sent)
        r.loop.run_trial(INSTRUCTION)
        assert len(r.fol.sent) == n


def test_stale_chunk_tail_is_dropped_and_requeried() -> None:
    with rig(infer_s=0.2, max_chunk_age_ms=500) as r:
        res = r.loop.run_trial(INSTRUCTION, max_steps=20)
        # actions at 200 + i * 66.7 ms after capture: i = 0..4 are within 500 ms
        assert res.stop_reason == "max_steps" and res.decisions == 4
        assert res.stale_chunk_drops == 3 and not r.arm.deadman.tripped


def test_ctrl_c_holds_then_torque_off_after_the_operator() -> None:
    class Op(ScriptedOperator):
        torque_at_prompt: bool | None = None

        def wait(self, msg: str) -> None:
            self.torque_at_prompt = r.fol.bus.torque_enabled

    with rig() as r:
        r.cam.interrupt_after = 6
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION)
        assert res.stop_reason == "operator_abort" and res.deadman_response == "hold"
        assert len(r.fol.sent) - n0 == res.steps + 1 and r.fol.bus.torque_enabled
        op = Op()
        safe_shutdown(r.arm, op)
        assert op.torque_at_prompt is True and not r.fol.bus.torque_enabled


def test_session_resumes_after_an_acknowledged_hold() -> None:
    class Op(ScriptedOperator):
        def ack_stop(self, reason: str) -> str | None:
            r.cam.fail = False  # the operator fixed the camera
            return super().ack_stop(reason)

    with rig() as r:

        def lose_camera(timing: dict[str, float]) -> None:
            if r.policy.calls == 2:
                r.cam.fail = True

        r.loop.on_decision = lose_camera
        trials, end = run_trials(
            r.loop,
            Op(),
            task="PickPlace",
            cell="seen",
            n_trials=2,
            info=r.server_info,
            camera_calibrated=False,
            max_steps=24,
        )
        assert end == "completed" and [t["stop_reason"] for t in trials] == ["deadman", "max_steps"]
        assert [t["outcome"] for t in trials] == ["failure", "success"]
        s = summarize(trials, 150.0)["cells"]["PickPlace/seen"]
        assert (s["n"], s["successes"]) == (2, 1)
        assert s["wilson_95"] == pytest.approx(protocol.wilson_interval(1, 2), abs=1e-4)


def test_mock_pilot_writes_trial_records(tmp_path: Path) -> None:
    rec_path = pilot.main(
        ["--preset", "real", "--mock", "--fake-clock", "--trials", "3", "--max-steps", "40"]
        + ["--runs-dir", str(tmp_path)]
    )
    res = json.loads(rec_path.read_text())["results"]
    assert res["mode"] == "mock" and res["session_end"] == "completed"
    trials = res["trials"]
    assert [t["trial_idx"] for t in trials] == [0, 1, 2]
    for t in trials:
        assert t["instruction"] == INSTRUCTION  # protocol string, not teleop.instructions
        assert t["protocol_hash"] == protocol.protocol_hash()
        assert t["protocol_version"] == protocol.PROTOCOL_VERSION
        assert t["policy_id"] == "mock:pickplace-waypoints" and t["home"]["reached"]
        assert t["outcome"] == "success" and t["outcome_source"] == "scripted"
        assert t["latency"]["capture_to_first_write_ms"]["median"] == pytest.approx(60.0)
    cell = res["summary"]["cells"]["PickPlace/seen"]
    assert cell["wilson_95"] == pytest.approx(protocol.wilson_interval(3, 3), abs=1e-4)
    rows = [
        json.loads(line) for line in (rec_path.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(row["kind"] == "trial" for row in rows) == 3
    # an aborted trial is reported but not scored
    mixed = [trials[0], {**trials[1], "outcome": "failure"}, {**trials[2], "outcome": "aborted"}]
    s = summarize(mixed, 150.0)["cells"]["PickPlace/seen"]
    assert (s["n"], s["successes"], s["aborted"]) == (2, 1, 1)
    assert isinstance(TrialResult().latency_ms(), list)


def test_hardware_mode_refuses_without_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        pilot.main(["--preset", "real", "--hardware"])
    assert e.value.code == 2
    monkeypatch.setattr("builtins.input", lambda _msg="": "no")
    with pytest.raises(SystemExit, match="nothing was opened"):
        pilot.main(["--preset", "real", "--hardware", "--i-am-present"])
