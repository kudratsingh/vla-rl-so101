"""WO-07 deploy client + policy server against a fake arm, fake camera and a mock policy.

Real gRPC on 127.0.0.1 (ephemeral port), the real safety filter; one shared FakeClock for the
driver, deadman, camera, server and loop, so latencies are exact. No serial port, no camera.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from robot_fakes import (
    LAST_TORQUE_OFF_CALL,
    FakeCam,
    FakeClock,
    FakeFollower,
    FakeReader,
    real_cfg,
    toy_fk,
)

from src.config import REPO_ROOT, load_config
from src.eval import protocol
from src.robot.camera import PolicyImagePipeline
from src.robot.client import (
    ConsoleOperator,
    DeployConfig,
    DeployLoop,
    PolicyClient,
    ScriptedOperator,
    ServerFault,
    ServerMismatch,
    TrialResult,
    check_server,
    latency_lines,
    run_trials,
    safe_shutdown,
    shutdown_session,
    summarize,
)
from src.robot.driver import So101Driver
from src.robot.prompts import TorqueGuard, TorqueWait
from src.robot.safety import DeadmanResponse, make_safety
from src.robot.serve import DEFAULT_HOST, MockPolicy, PolicyServer, build_server, pack, unpack
from src.robot.servo_health import monitor_from_config
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
    track: bool = True,
    policy_id: str = "mock:test",
    park_enabled: bool = False,
    **deploy: Any,
) -> Iterator[Rig]:
    clock = FakeClock()
    cfg = real_cfg()
    cfg["deploy"] = {**cfg["deploy"], "home_qpos": TEST_HOME, **deploy}
    cfg["robot"]["park"] = {**cfg["robot"]["park"], "enabled": park_enabled}
    dcfg = DeployConfig.from_config(cfg)
    policy = MockPolicy(
        waypoints or [TEST_TARGET], chunk_len=server_chunk_len, infer_s=infer_s, sleep=clock.sleep
    )
    servicer = PolicyServer(policy, image_size=dcfg.image_size, policy_id=policy_id, clock=clock)
    server, port = build_server(servicer, port=0)
    fol = FakeFollower(track=track)
    arm = So101Driver(
        fol, UnitMap(), make_safety(cfg, toy_fk, clock=clock), clock=clock, sleep=clock.sleep
    )
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
        for missing in (None, "", "None", "null"):  # str(None) from a server without --adapter
            with pytest.raises(ServerMismatch, match="policy_id"):
                check_server({**info, "policy_id": missing}, r.dcfg)
        pinned = dataclasses.replace(r.dcfg, policy_checkpoint="user/repo/sft_rl")
        with pytest.raises(ServerMismatch, match="policy_checkpoint"):
            check_server(info, pinned)  # serving mock:test, configured for the SFT+RL adapter
        check_server({**info, "policy_id": "user/repo/sft_rl"}, pinned)
        with pytest.raises(ValueError, match="policy_id"):
            PolicyServer(r.policy, image_size=r.dcfg.image_size, policy_id="None")
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
        # The query (50 ms) counts against the boundary tick's period: the next chunk's first
        # write lands one period after the last one, as in sim, not period + query after it.
        period_ms = 1e3 / r.dcfg.control_hz
        assert res.latency_ms("boundary_gap_ms") == pytest.approx([period_ms] * 2)
        assert res.write_hz == pytest.approx(r.dcfg.control_hz)
        # frame stamp -> first bus write: the query, then the wait for the write tick
        assert res.latency_ms() == pytest.approx([50.0, period_ms, period_ms])
        assert res.latency_ms("pace_wait_ms") == pytest.approx(
            [0.0, period_ms - 50, period_ms - 50]
        )
        assert res.latency_ms("infer_ms") == pytest.approx([50.0] * 3)
        assert set(r.policy.instructions) == {INSTRUCTION}
    with rig(infer_s=0.1) as r:  # slower than the period: the gap is the query, not their sum
        res = r.loop.run_trial(INSTRUCTION, max_steps=24)
        assert res.latency_ms("boundary_gap_ms") == pytest.approx([100.0] * 2)
        assert res.latency_ms("pace_wait_ms") == pytest.approx([0.0] * 3)


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
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION)
        assert res.stop_reason == "deadman" and "no camera frame" in res.stop_detail
        # read 1 = decision tick, reads 2-5 written, read 6 lost: held on that very tick
        assert res.steps == 4 and res.decisions == 1 and res.deadman_response == "hold"
        assert len(r.fol.sent) - n0 == res.steps + 1
    with rig(infer_s=0.39) as r:
        # The stream freezes right after decision 2's frame (read 10; chunk 1 = reads 2-8
        # written, read 9 dropped as stale). No frame check while the 390 ms query is in flight,
        # so the trip comes at the first write tick after the reply: past timeout + one tick,
        # within timeout + request_timeout + one tick, and nothing is written with that frame.
        r.cam.stall_after = 10
        res = r.loop.run_trial(INSTRUCTION)
        assert res.stop_reason == "deadman" and "stale" in res.stop_detail
        assert res.steps == 7 and res.decisions == 2
        assert r.cam.last_fresh_time is not None
        late = res.t_end - r.cam.last_fresh_time
        timeout, tick = r.arm.safety.config.frame_timeout_s, 1 / r.dcfg.control_hz
        assert timeout + tick < late <= timeout + r.dcfg.request_timeout_s + tick + 1e-9
        assert res.max_frame_age_ms <= timeout * 1e3 + 1e-6


def test_lost_frame_on_a_stale_drop_tick_trips_the_deadman() -> None:
    with rig(infer_s=0.2, max_chunk_age_ms=500) as r:
        r.cam.fail_reads = {7}  # the tick whose action (i = 5) is dropped as stale
        n0 = len(r.fol.sent)
        res = r.loop.run_trial(INSTRUCTION, max_steps=20)
        assert res.stop_reason == "deadman" and "no camera frame" in res.stop_detail
        assert r.arm.deadman.tripped and res.stale_chunk_drops == 0
        assert res.steps == 5 and len(r.fol.sent) - n0 == res.steps + 1


class PromptLog(ScriptedOperator):
    """Scripted, and logs the acknowledgement / support prompts (answered ENTER)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def acknowledge(self, msg: str) -> str:
        self.prompts.append(msg)
        return "enter"

    def support(self, msg: str) -> str:
        self.prompts.append(msg)
        return "enter"


def test_serial_error_stops_the_loop_and_the_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with rig() as r:

        def unplug_on_decision_2(k: int) -> None:
            logged = r.loop.on_decision  # run_trials' metrics logger

            def on_decision(timing: dict[str, float]) -> None:
                if logged is not None:
                    logged(timing)
                if timing["decision"] == 2:
                    r.fol.fail_write = True  # the next write fails: torque off, latched

            r.loop.on_decision = on_decision

        op = PromptLog()
        rec_path = pilot.session(  # run_trials, then safe_shutdown in its finally
            real_cfg(),
            r.loop,
            op,
            r.server_info,
            mode="mock",
            task="PickPlace",
            cell="seen",
            n_trials=3,
            max_steps=None,
            camera_calibrated=False,
            runs_dir=tmp_path,
            on_trial_start=unplug_on_decision_2,
        )
        res = json.loads(rec_path.read_text())["results"]
        assert len(res["trials"]) == 1 and res["session_end"].startswith("torque_off")
        t = res["trials"][0]
        assert t["stop_reason"] == "serial_fault" and t["deadman_response"] == "torque_off"
        # the policy started, then the bus failed: a failure, never judged, never a free redo
        assert t["policy_started"] and t["outcome"] == "failure" and t["outcome_source"] == "fault"
        assert t["fault"]["kind"] == "serial" and "Goal_Position" in t["fault"]["reason"]
        assert not r.fol.bus.torque_enabled and r.policy.calls == 2
        # torque already off: the operator supports the arm in the shutdown (one ack), no
        # HOLDING prompt, torque off retried; the ack is recorded with how it was given
        assert op.prompts == ["press ENTER once the arm rests safely (supported or on the table): "]
        assert res["shutdown"] == "torque_was_off" and res["support_confirmed"] is True
        assert res["shutdown_detail"]["ack"]["how"] == "enter"
        assert "FAULT (serial error" in (out := capsys.readouterr().out) and "already OFF" in out
        assert r.fol.bus.calls[-1] == LAST_TORQUE_OFF_CALL
        # the bus answers again: still nothing is written while the latch is TORQUE_OFF
        r.fol.fail_write = False
        n = len(r.fol.sent)
        safe_shutdown(r.arm, op)
        r.arm.hold()
        assert len(r.fol.sent) == n and not r.fol.bus.torque_enabled


@pytest.mark.parametrize("failure", ["error", "hang", "late", "bad_shape", "nan"])
def test_server_failure_holds_and_stops(failure: str) -> None:
    timeout_ms = 200 if failure == "hang" else 400  # "hang" waits on the real gRPC deadline
    # track=False: the servo never moves, so the measured pose (the hold) differs from every
    # policy goal and the last-write check below can tell a hold from a policy write
    with rig(request_timeout_ms=timeout_ms, track=False) as r:
        measured = dict(r.fol.pos)
        holds: list[int] = []
        hold = r.arm.hold

        def counted_hold(*a: Any, **k: Any) -> None:
            holds.append(len(r.fol.sent))
            hold(*a, **k)

        r.arm.hold = counted_hold  # type: ignore[method-assign]

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
            "nan": "server_bad_reply",  # a server fault, not a policy safety rejection
        }
        assert res.stop_reason == expected[failure] and res.deadman_response == "hold"
        assert res.decisions == 1 and res.steps == 8  # the first chunk, then the stop
        assert len(r.fol.sent) - n0 == res.steps + 1  # one hold write, nothing after it
        assert holds == [n0 + res.steps]  # exactly one hold, and it made the last write
        assert r.fol.sent[-1] == pytest.approx(measured)
        assert r.fol.sent[-2] != pytest.approx(measured)
        assert r.fol.bus.torque_enabled
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

        def support(self, msg: str) -> str:
            self.torque_at_prompt = r.fol.bus.torque_enabled
            return "enter"

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
    rec_path = pilot.main(  # full 150-step trials: the waypoint walk reaches "close"
        ["--preset", "real", "--mock", "--fake-clock", "--trials", "3"]
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
        # the 60 ms query beats the 66.7 ms period: writes stay at 15 Hz, as in sim
        assert t["latency"]["frame_return_to_first_write_ms"]["median"] == pytest.approx(
            1e3 / 15, abs=1e-3
        )
        assert t["latency"]["boundary_gap_ms"]["max"] == pytest.approx(1e3 / 15, abs=1e-3)
        assert t["latency"]["write_hz"] == pytest.approx(15.0, abs=1e-3)
        # the walk restarts every trial, so every trial commands the clipped gripper close
        assert t["safety"]["interventions_by_group"].get("gripper", 0) > 0
    cell = res["summary"]["cells"]["PickPlace/seen"]
    assert cell["wilson_95"] == pytest.approx(protocol.wilson_interval(3, 3), abs=1e-4)
    rows = [
        json.loads(line) for line in (rec_path.parent / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(row["kind"] == "trial" for row in rows) == 3
    assert all(t["policy_started"] and t["fault"] is None for t in trials)
    assert res["shutdown"] == "supported" and res["support_confirmed"] is True  # scripted ENTER
    # an attempt whose policy never started is reported but not scored; a faulted one counts
    fault = {"kind": "servo", "stop_reason": "servo_fault", "reason": "hot", "where": "trial"}
    mixed = [
        trials[0],
        {**trials[1], "outcome": "failure", "fault": fault},
        {**trials[2], "outcome": "not_started", "policy_started": False},
    ]
    s = summarize(mixed, 150.0)["cells"]["PickPlace/seen"]
    assert (s["n"], s["successes"], s["not_started"], s["faulted"]) == (2, 1, 1, 1)
    assert s["success_rate"] == 0.5 and s["faults_by_kind"] == {"servo": 1}
    assert s["diagnostic_excluding_faulted"]["success_rate"] == 1.0  # secondary only
    assert isinstance(TrialResult().latency_ms(), list)


def test_hardware_mode_refuses_without_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        pilot.main(["--preset", "real", "--hardware"])
    assert e.value.code == 2
    monkeypatch.setattr(pilot, "require_terminal", lambda _what: None)  # as if in Terminal
    monkeypatch.setattr("builtins.input", lambda _msg="": "no")
    with pytest.raises(SystemExit, match="not confirmed; nothing was opened"):
        pilot.main(["--preset", "real", "--hardware", "--i-am-present"])


def test_serve_main_names_its_adapter_and_keeps_a_run_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.robot import serve

    with pytest.raises(SystemExit) as e:  # sft.yaml: policy.adapter null, no --adapter
        serve.main(["--preset", "sim", "--config", str(REPO_ROOT / "configs" / "sft.yaml")])
    assert e.value.code == 2
    loaded: list[str | None] = []
    seen: dict[str, Any] = {}

    def fake_oft(cfg: Any, device: str, adapter: str | None) -> MockPolicy:
        loaded.append(adapter)
        return MockPolicy([TEST_HOME])

    real_build = serve.build_server

    def build(servicer: PolicyServer, host: str, port: int) -> tuple[Any, int]:
        server, bound = real_build(servicer, host, 0)

        class Server:
            def wait_for_termination(self) -> None:
                with PolicyClient(f"127.0.0.1:{bound}", timeout_s=1.0) as c:
                    seen.update(c.info())
                raise KeyboardInterrupt  # the operator's Ctrl-C (SIGTERM is mapped to it)

            def stop(self, grace: float) -> None:
                server.stop(grace=0)

        return Server(), bound

    monkeypatch.setattr(serve, "load_oft", fake_oft)
    monkeypatch.setattr(  # no Hub access in tests: adapter_identity is tested on a local dir
        serve,
        "served_adapter_identity",
        lambda a: {"adapter_sha256": f"sha-of:{a}", "adapter_revision": None},
    )
    monkeypatch.setattr(serve, "build_server", build)
    monkeypatch.setattr(serve.signal, "signal", lambda *a: None)
    rec_path = serve.main(["--preset", "rl", "--device", "cpu", "--runs-dir", str(tmp_path)])
    adapter = load_config("rl")["policy"]["adapter"]  # no --adapter: the config's is loaded
    assert loaded == [adapter] and seen["policy_id"] == adapter
    rec = json.loads(rec_path.read_text())
    assert rec["status"] == "ok" and rec["end_utc"] and rec["actual_usd"] is not None
    assert rec["results"]["policy_id"] == adapter and rec["results"]["requests"] == 0
    assert seen["server_run"]["run_id"] == rec["run_id"]
    assert seen["server_run"]["device"] == "cpu"
    assert seen["adapter_sha256"] == rec["results"]["adapter_sha256"] == f"sha-of:{adapter}"


def test_adapter_identity_hashes_the_weights_not_the_name(tmp_path: Path) -> None:
    from src.robot import serve

    snap = tmp_path / "models--u--r" / "snapshots" / "abc123" / "ckpt"
    snap.mkdir(parents=True)
    (snap / "adapter_config.json").write_text('{"r": 8}')
    (snap / "adapter_model.safetensors").write_bytes(b"weights-v1")
    (snap / "optimizer.pt").write_bytes(b"not hashed")
    a = serve.adapter_identity(snap)
    assert a["adapter_revision"] == "abc123"
    assert a["adapter_files"] == ["adapter_config.json", "adapter_model.safetensors"]
    (snap / "optimizer.pt").write_bytes(b"changed")
    assert serve.adapter_identity(snap)["adapter_sha256"] == a["adapter_sha256"]
    (snap / "adapter_model.safetensors").write_bytes(b"weights-v2")  # re-pushed, same name
    assert serve.adapter_identity(snap)["adapter_sha256"] != a["adapter_sha256"]
    local = tmp_path / "local"
    local.mkdir()
    with pytest.raises(FileNotFoundError):
        serve.adapter_identity(local)


def test_healthy_end_parks_then_torque_off_a_motion_abort_does_not() -> None:
    with rig() as r:  # robot.park.enabled false (the default): hold + support prompt, no park
        assert r.dcfg.park is None
        _, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=1,
            info=r.server_info,
            camera_calibrated=False,
            max_steps=16,
        )
        op = PromptLog()
        n = len(r.fol.sent)
        assert shutdown_session(r.loop, op, end).end == "supported"
        assert len(r.fol.sent) == n + 1 and r.fol.sent[-1] == pytest.approx(r.fol.pos)  # a hold
        assert op.prompts and "support it" in op.prompts[0] and not r.fol.bus.torque_enabled
    with rig(park_enabled=True) as r:
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=1,
            info=r.server_info,
            camera_calibrated=False,
            max_steps=16,
        )
        assert end == "completed"
        op = PromptLog()
        sd = shutdown_session(r.loop, op, end)
        assert sd.end == "parked" and sd.support_confirmed is None  # nobody was asked
        park = r.dcfg.park
        assert park is not None and park.qpos == tuple(TEST_HOME)  # robot.park.qpos null -> home
        assert not r.fol.bus.torque_enabled and not op.prompts  # no support prompt needed
        q = UnitMap().to_canonical(r.fol.sent[-1])
        np.testing.assert_allclose(q, TEST_HOME, atol=math.radians(3.0))
    with rig() as r:  # Ctrl-C while the arm drives home: hold, support prompt, never a park
        r.cam.interrupt_after = 3
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=2,
            info=r.server_info,
            camera_calibrated=False,
            max_steps=16,
        )
        assert trials[0]["stop_reason"] == "operator_abort"
        assert trials[0]["outcome"] == "not_started" and not trials[0]["policy_started"]
        assert r.arm.deadman.tripped and r.arm.deadman.response is not None
        op = PromptLog()
        n = len(r.fol.sent)
        assert shutdown_session(r.loop, op, end).end == "supported"
        assert len(r.fol.sent) == n + 1 and r.fol.sent[-1] == pytest.approx(r.fol.pos)  # a hold
        assert op.prompts and "support it" in op.prompts[0] and not r.fol.bus.torque_enabled


def test_servo_fault_mid_trial_is_torque_off_with_support_prompt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())

        def overheat(timing: dict[str, float]) -> None:
            r.fol.temperature_c["gripper"] = 66.0

        r.loop.on_decision = overheat
        res = r.loop.run_trial(INSTRUCTION, max_steps=40)
        assert res.stop_reason == "servo_fault" and res.deadman_response == "torque_off"
        assert not r.fol.bus.torque_enabled
        n = len(r.fol.sent)
        op = PromptLog()
        sd = shutdown_session(r.loop, op, "completed")
        assert sd.end == "torque_was_off" and sd.support_confirmed is True  # no park, no hold
        assert len(r.fol.sent) == n and "SERVO FAULT" in capsys.readouterr().out
        assert op.prompts and "servo supply is off" in op.prompts[0]


def test_failed_torque_off_is_recorded_and_blocks_motion_to_the_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A servo fault whose torque-off write raises: 'torque off unconfirmed' in the operator
    message, the trial record, the shutdown result and the session end; nothing moves after."""
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())

        def overheat_with_a_dead_bus(k: int) -> None:
            def on_decision(timing: dict[str, float]) -> None:
                r.fol.temperature_c["gripper"] = 66.0
                r.fol.bus.fail_disable = True  # every torque-off write now raises

            r.loop.on_decision = on_decision

        class Op(PromptLog):
            def ack_stop(self, reason: str) -> str | None:
                return "scripted resume"  # would continue after a HOLD: must not here

        op = Op()
        rec_path = pilot.session(
            real_cfg(),
            r.loop,
            op,
            r.server_info,
            mode="mock",
            task="PickPlace",
            cell="seen",
            n_trials=3,
            max_steps=40,
            camera_calibrated=False,
            runs_dir=tmp_path,
            on_trial_start=overheat_with_a_dead_bus,
        )
        n_sent = len(r.fol.sent)
        res = json.loads(rec_path.read_text())["results"]
        assert len(res["trials"]) == 1  # the session ended at the fault
        t = res["trials"][0]
        assert t["stop_reason"] == "torque_off_unconfirmed" and t["torque_off_unconfirmed"]
        # the policy had started: a failure in the primary metric, fault recorded separately
        assert t["stop_reason_before"] == "servo_fault" and t["outcome"] == "failure"
        assert t["policy_started"] and t["fault"]["kind"] == "torque_off_unconfirmed"
        assert t["fault"]["fault_before"]["kind"] == "servo"
        assert res["summary"]["cells"]["PickPlace/seen"]["n"] == 1
        assert "bus gone" in t["torque_off_unconfirmed_detail"]
        assert res["shutdown"] == "torque_off_unconfirmed" and res["support_confirmed"] is False
        assert res["session_end"].startswith("torque_off_unconfirmed")
        assert res["torque_off_unconfirmed"] and res["summary"]["torque_off_unconfirmed"]
        out = capsys.readouterr().out
        assert "TORQUE OFF UNCONFIRMED" in out and "inline power switch" in out
        assert op.prompts and "inline power switch" in op.prompts[-1]
        # later code paths cannot move the arm or clear the flag
        r.fol.bus.fail_disable = False
        r.arm.deadman.reset("operator typed resume")
        sd = safe_shutdown(r.arm, op, healthy=True, park=r.dcfg.park)
        assert sd.end == "torque_off_unconfirmed" and sd.ack and sd.ack["given"]
        trials, end = run_trials(
            r.loop,
            ScriptedOperator(),
            task="PickPlace",
            cell="seen",
            n_trials=1,
            info=r.server_info,
            camera_calibrated=False,
        )
        assert trials == [] and end.startswith("torque_off_unconfirmed")
        assert len(r.fol.sent) == n_sent


def test_latency_is_never_labelled_end_to_end_without_a_measured_offset() -> None:
    lat = [50.0, 60.0, 70.0]
    trials = [
        {
            "task": "PickPlace",
            "cell": "seen",
            "outcome": "success",
            "stop_reason": "max_steps",
            "latency": {"per_decision_frame_return_to_first_write_ms": lat, "write_hz": 15.0},
        }
    ]
    s = summarize(trials, 150.0)
    assert s["frame_return_to_first_write_ms"]["median"] == 60.0
    assert s["end_to_end_estimate_ms"] is None and "capture_to_first_write_ms" not in s
    assert "EXCLUDES capture" in s["latency_note"]
    assert any("NOT measured" in line for line in latency_lines(s))
    s2 = summarize(trials, 150.0, capture_offset_ms=40.0)
    assert s2["end_to_end_estimate_ms"]["median"] == 100.0
    assert s2["end_to_end_estimate_ms"]["capture_offset_ms"] == 40.0
    assert any("measured capture offset" in line for line in latency_lines(s2))


# -- scoring after the trial: settle, faults, Ctrl-C at the score prompt (PR 29 review) -----------
class EventOp(ScriptedOperator):
    """Records prompts and scores in order, with the fake clock and the bus health reads."""

    def __init__(self, r: Rig, interrupt_at_score: int | None = None) -> None:
        self.r, self.events, self.scores = r, [], 0
        self.interrupt_at_score = interrupt_at_score

    def _reads(self) -> int:
        regs = ("Present_Temperature", "Present_Current")
        return sum(c[0] == "sync_read" and c[1] in regs for c in self.r.fol.bus.calls)

    def acknowledge(self, msg: str) -> str:
        self.events.append(("wait", msg))
        return "enter"

    def support(self, msg: str) -> str:
        self.events.append(("wait", msg))
        return "enter"

    def outcome(self, result: Any) -> str:
        self.scores += 1
        if self.scores == self.interrupt_at_score:
            raise KeyboardInterrupt
        self.events.append(("outcome", result.stop_reason, self.r.clock(), self._reads()))
        return super().outcome(result)


def _trials(r: Rig, op: Any, n: int = 1, **kw: Any) -> tuple[list[dict[str, Any]], str]:
    return run_trials(
        r.loop,
        op,
        task="PickPlace",
        cell="seen",
        n_trials=n,
        info=r.server_info,
        camera_calibrated=False,
        max_steps=16,
        **kw,
    )


def _wrap_run_trial(r: Rig, after: Any) -> list[tuple[float, int]]:
    """Calls ``after()`` when each trial returns; records (clock, health reads) at that moment."""
    ends: list[tuple[float, int]] = []
    real = r.loop.run_trial

    def run_trial(*a: Any, **k: Any) -> Any:
        res = real(*a, **k)
        ends.append((r.clock(), EventOp(r)._reads()))
        after()
        return res

    r.loop.run_trial = run_trial  # type: ignore[method-assign]
    return ends


def test_score_comes_after_the_settle_and_a_servo_read() -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        ends = _wrap_run_trial(r, lambda: None)
        op = EventOp(r)
        trials, end = _trials(r, op, settle_s=3.0)
        assert end == "completed" and trials[0]["settle"] == {"settle_s": 3.0, "measured_s": 3.0}
        (t_end, reads_end), (_, _, t_score, reads_score) = ends[0], op.events[-1]
        assert t_score >= t_end + 3.0  # judged only after the arm held still for settle_s
        assert reads_score == reads_end + 2  # temperature + current read in between


def test_servo_fault_at_the_settle_read_is_a_failure_without_a_judgement() -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        _wrap_run_trial(r, lambda: r.fol.temperature_c.update(gripper=61.0))  # only after it
        op = EventOp(r)
        trials, end = _trials(r, op, n=2, settle_s=3.0)
        t = trials[0]
        assert len(trials) == 1 and end.startswith("torque_off: servo fault")
        assert t["stop_reason"] == "servo_fault" and t["outcome"] == "failure"  # never redone
        assert t["outcome_source"] == "fault" and t["fault"]["where"] == "settle"
        assert t["fault"]["kind"] == "servo" and t["stop_detail"].startswith("after the settling")
        assert t["settle"]["measured_s"] == 3.0 and "over-temperature" in t["settle"]["fault"]
        assert op.events == []  # not judged; the support prompt comes with the shutdown
        sd = shutdown_session(r.loop, op, end)
        assert sd.end == "torque_was_off" and sd.support_confirmed is True
        assert [e[0] for e in op.events] == ["wait"] and "rests safely" in op.events[0][1]
        assert not r.fol.bus.torque_enabled


def test_mid_trial_servo_fault_skips_the_settle_and_is_never_judged() -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())

        def overheat(k: int) -> None:
            def on_decision(timing: dict[str, float]) -> None:
                if timing["decision"] == 1:
                    r.fol.temperature_c["gripper"] = 66.0

            r.loop.on_decision = on_decision

        sleeps: list[float] = []
        real_sleep = r.loop.sleep
        r.loop.sleep = lambda s: (sleeps.append(s), real_sleep(s))[1]  # type: ignore[assignment]
        op = EventOp(r)
        trials, end = _trials(r, op, settle_s=3.0, on_trial_start=overheat)
        t = trials[0]
        assert t["stop_reason"] == "servo_fault" and t["outcome"] == "failure"
        assert t["fault"]["kind"] == "servo" and t["fault"]["where"] == "trial"
        assert t["settle"] == {"settle_s": 3.0, "measured_s": None, "skipped": "fault: servo"}
        assert 3.0 not in sleeps  # no settling interval with the arm limp
        assert op.events == [] and op.scores == 0  # the operator is not asked to judge it
        assert end.startswith("torque_off: servo fault") and not r.fol.bus.torque_enabled


def test_ctrl_c_at_the_score_prompt_records_a_failure_never_a_redo() -> None:
    with rig() as r:
        op = EventOp(r, interrupt_at_score=2)
        trials, end = _trials(r, op, n=3, settle_s=0.0)
        assert end == "operator_ctrl_c" and [t["trial_idx"] for t in trials] == [0, 1]
        t = trials[1]
        assert t["outcome"] == "failure" and t["outcome_source"] == "unjudged"
        assert t["unjudged"]["how"] == "ctrl_c" and "doubt = n" in t["unjudged"]["reason"]
        assert {int(x["trial_idx"]) for x in trials if x["policy_started"]} == {
            0,
            1,
        }  # a resume (done_indices) never re-rolls trial 1


# -- bounded torque-on prompts, honest shutdown (owner decisions 2026-11-08) ---------------------
def _console(r: Rig, script: list[Any]) -> tuple[ConsoleOperator, FakeReader]:
    """The real ConsoleOperator on a scripted stdin. The guard's own fake clock: a 60 s wait
    does not age the camera frames."""
    gclock = FakeClock()
    reader = FakeReader(script, gclock, r.fol.bus)
    guard = TorqueGuard(
        r.arm, TorqueWait(60.0, 10.0), read=reader, clock=gclock, sleep=gclock.sleep
    )
    return ConsoleOperator(guard), reader


def _lose_camera_after_the_first_action(r: Rig) -> Any:
    def hook(k: int) -> None:
        logged = r.loop.on_decision

        def on_decision(timing: dict[str, float]) -> None:
            if logged is not None:
                logged(timing)
            r.cam.fail = True  # the next tick has no frame: deadman HOLD after the start

        r.loop.on_decision = on_decision

    return hook


ENTER_ALL = {"home": [""], "placement": [""], "score": ["y"], "resume": [], "support": [""]}
ORDER = ("home", "placement", "score", "support")
# what each answer at each pilot prompt does: (session_end prefix, shutdown, support_confirmed)
PILOT_CASES = {
    ("home", "timeout"): ("prompt_timeout", "released_unconfirmed", False),
    ("placement", "timeout"): ("prompt_timeout", "released_unconfirmed", False),
    ("score", "timeout"): ("prompt_timeout", "released_unconfirmed", False),
    ("resume", "timeout"): ("prompt_timeout", "released_unconfirmed", False),
    ("support", "timeout"): ("completed", "released_unconfirmed", False),
    ("home", "ctrl_c"): ("operator_ctrl_c", "supported", True),
    ("placement", "ctrl_c"): ("operator_ctrl_c", "supported", True),
    ("score", "ctrl_c"): ("operator_ctrl_c", "supported", True),
    ("resume", "ctrl_c"): ("operator_ctrl_c", "supported", True),
    ("support", "ctrl_c"): ("completed", "released_unconfirmed", False),  # then no answer
    ("home", "eof"): ("error", "released_unconfirmed", False),
    ("placement", "eof"): ("error", "released_unconfirmed", False),
    ("score", "eof"): ("error", "released_unconfirmed", False),
    ("resume", "eof"): ("error", "released_unconfirmed", False),
    ("support", "eof"): ("completed", "released_unconfirmed", False),
}
ANSWER = {"timeout": None, "ctrl_c": KeyboardInterrupt, "eof": EOFError}


@pytest.mark.parametrize(("prompt", "how"), sorted(PILOT_CASES))
def test_pilot_torque_on_prompts_are_bounded_and_end_honestly(
    tmp_path: Path, prompt: str, how: str
) -> None:
    """Timeout, Ctrl-C and closed stdin at every pilot prompt answered with torque on: servo
    health before the wait, the wait bounded, and the shutdown says "supported" only after an
    explicit ENTER at the support prompt."""
    end_prefix, shutdown, confirmed = PILOT_CASES[(prompt, how)]
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        order = ("home", "placement", "resume", "support") if prompt == "resume" else ORDER
        script: list[Any] = []
        for p in order[: order.index(prompt)]:
            script += ENTER_ALL[p]
        script.append(ANSWER[how])
        if how == "ctrl_c":
            script.append(None if prompt == "support" else "")  # then the support prompt
        op, reader = _console(r, script)
        hook = _lose_camera_after_the_first_action(r) if prompt == "resume" else None
        kw: dict[str, Any] = {
            "mode": "mock",
            "task": "PickPlace",
            "cell": "seen",
            "n_trials": 1,
            "max_steps": 16,
            "camera_calibrated": False,
            "runs_dir": tmp_path,
            "on_trial_start": hook,
            "torque_wait": TorqueWait(60.0, 10.0),
        }
        if how == "eof" and prompt != "support":
            with pytest.raises(EOFError):
                pilot.session(real_cfg(), r.loop, op, r.server_info, **kw)
        else:
            pilot.session(real_cfg(), r.loop, op, r.server_info, **kw)
        (rec,) = tmp_path.glob("*/record.json")
        res = json.loads(rec.read_text())["results"]
        assert res["session_end"].startswith(end_prefix), res["session_end"]
        assert res["shutdown"] == shutdown and res["support_confirmed"] is confirmed
        assert res["torque_wait"] == {"timeout_s": 60.0, "grace_s": 10.0}
        if shutdown == "released_unconfirmed":
            assert res["shutdown_detail"]["cause"] == ("timeout" if how == "ctrl_c" else how)
            assert r.arm.deadman.response is DeadmanResponse.TORQUE_OFF
        assert not r.fol.bus.torque_enabled  # torque off at the end, whichever way
        # every read while torque was on was bounded: nothing waits forever under torque
        assert all(t is not None for _, t, on in reader.calls if on)
        trials = res["trials"]
        if prompt == "home":
            assert trials == []  # the arm never moved: no attempt
        elif prompt == "placement":  # the arm went home, then the prompt ended: an attempt
            (t,) = trials
            assert (t["outcome"], t["outcome_source"]) == ("not_started", "not_scored")
            assert not t["policy_started"] and t["home"]["reached"]
            stop = {"timeout": "prompt_timeout", "ctrl_c": "operator_ctrl_c", "eof": "stdin_closed"}
            assert t["stop_reason"] == stop[how]
            assert t["stop_detail"].startswith("placement prompt")
            if how == "timeout":
                assert t["fault"]["kind"] == "prompt_timeout"
                assert t["fault"]["where"] == "placement_prompt"
            else:
                assert t["fault"] is None
            assert res["summary"]["cells"]["PickPlace/seen"]["not_started"] == 1
        else:
            (t,) = trials
            assert t["policy_started"]
            if prompt == "score":
                assert t["outcome"] == "failure" and t["outcome_source"] == "unjudged"
                assert t["unjudged"]["how"] == how
            elif prompt == "resume":  # a HOLD stop after the start: a failure, never judged
                assert t["outcome"] == "failure" and t["fault"]["kind"] == "safety"
            else:
                assert t["outcome"] == "success" and t["outcome_source"] == "operator"


def test_pilot_prompts_read_servo_health_before_and_after_each_wait(tmp_path: Path) -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        op, reader = _console(r, ["", "", "y", ""])
        reads: list[int] = []
        real_read = reader.__call__

        def counted(prompt: str, timeout_s: float | None) -> str | None:
            reads.append(EventOp(r)._reads())
            return real_read(prompt, timeout_s)

        op.guard.read_line = counted
        pilot.session(
            real_cfg(),
            r.loop,
            op,
            r.server_info,
            mode="mock",
            task="PickPlace",
            cell="seen",
            n_trials=1,
            max_steps=16,
            camera_calibrated=False,
            runs_dir=tmp_path,
        )
        # home: 2 before; placement: 2 after home's answer + 2 before (+ the trial's own reads);
        # each prompt is preceded by a fresh temperature + current read
        assert all(b - a >= 2 for a, b in zip([0, *reads], reads, strict=False))
        assert [c[1] for c in reader.calls] == [60.0] * 4
        res = json.loads(next(tmp_path.glob("*/record.json")).read_text())["results"]
        assert res["shutdown"] == "supported" and res["support_confirmed"] is True


@pytest.mark.parametrize(
    ("answer", "how", "given"),
    [("", "enter", True), (KeyboardInterrupt, "ctrl_c", False), (EOFError, "eof", False)]
    + [(None, "timeout", False)],
)
def test_fault_acknowledgements_record_how_they_were_given(
    answer: Any, how: str, given: bool
) -> None:
    with rig() as r:  # a servo fault: torque already off, the operator supports the arm
        r.arm.monitor = monitor_from_config(real_cfg())
        r.fol.temperature_c["gripper"] = 66.0
        with pytest.raises(Exception, match="SERVO FAULT"):
            r.arm.check_servos()
        op, reader = _console(r, [answer])
        sd = safe_shutdown(r.arm, op)
        assert sd.end == "torque_was_off" and sd.support_confirmed is given
        assert sd.ack is not None and (sd.ack["how"], sd.ack["given"]) == (how, given)
        assert reader.calls[0][1] == 60.0 and reader.calls[0][2] is False  # torque off: no release
    with rig() as r:  # a failed torque-off: the power-switch acknowledgement
        r.fol.bus.fail_disable = True
        assert r.arm.torque_off() is False
        op, reader = _console(r, [answer])
        sd = safe_shutdown(r.arm, op)
        assert sd.end == "torque_off_unconfirmed" and sd.support_confirmed is False
        assert sd.ack is not None and (sd.ack["how"], sd.ack["given"]) == (how, given)
        assert "inline power switch" in sd.ack["prompt"]


def test_policy_started_is_the_first_action_handed_to_the_safety_filter() -> None:
    with rig() as r:  # the first query fails: no action ever reached the filter
        r.policy.mode = "error"
        res = r.loop.run_trial(INSTRUCTION, max_steps=16)
        assert res.stop_reason == "server_error" and not res.policy_started and res.steps == 0
    with rig() as r:  # no camera frame at the decision tick: the deadman holds before any action
        r.cam.fail = True
        res = r.loop.run_trial(INSTRUCTION, max_steps=16)
        assert res.stop_reason == "deadman" and not res.policy_started
    with rig() as r:
        res = r.loop.run_trial(INSTRUCTION, max_steps=16)
        assert res.policy_started and res.policy_started_utc and res.steps == 16
        assert res.t_first_write is not None and res.policy_started_t is not None
        assert res.policy_started_t <= res.t_first_write  # stamped as the action was handed over


# -- review round 3: signals as flags, hang-up, Ctrl-C outside a read, durable records -----------
def _pilot_kw(tmp_path: Path, **extra: Any) -> dict[str, Any]:
    return {
        "mode": "mock",
        "task": "PickPlace",
        "cell": "seen",
        "n_trials": 1,
        "max_steps": 16,
        "camera_calibrated": False,
        "runs_dir": tmp_path,
        "torque_wait": TorqueWait(60.0, 10.0),
        **extra,
    }


def _record(tmp_path: Path) -> dict[str, Any]:
    (rec,) = tmp_path.glob("*/record.json")
    return dict(json.loads(rec.read_text())["results"])


def _after_first_action(r: Rig, act: Any) -> Any:
    def hook(k: int) -> None:
        logged = r.loop.on_decision

        def on_decision(timing: dict[str, float]) -> None:
            if logged is not None:
                logged(timing)
            if timing["decision"] == 1:
                act()

        r.loop.on_decision = on_decision

    return hook


def test_ctrl_c_flag_during_motion_stops_between_ticks_and_the_trial_fails() -> None:
    from src.robot.prompts import Interrupts

    with rig() as r:
        flags = Interrupts()
        r.loop.interrupts = flags
        op, _reader = _console(r, ["", "", ""])  # home, placement, support
        hook = _after_first_action(r, lambda: setattr(flags, "ctrl_c", True))
        trials, end = _trials(r, op, on_trial_start=hook)
        (t,) = trials
        assert end == "operator_abort" and t["stop_reason"] == "operator_abort"
        assert t["policy_started"] and t["outcome"] == "failure"
        assert t["fault"]["kind"] == "operator_abort"  # never a redo, never judged
        assert r.arm.deadman.response is DeadmanResponse.HOLD and r.fol.bus.torque_enabled
        assert not r.fol.bus.port_handler.is_using  # stopped between ticks, not mid-transfer
        sd = shutdown_session(r.loop, op, end)
        assert sd.end == "supported" and sd.support_confirmed is True


def test_hangup_during_motion_takes_the_closed_stdin_path_and_records_it(tmp_path: Path) -> None:
    from src.robot import prompts as P

    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        r.loop.interrupts = P.INTERRUPTS
        op, reader = _console(r, ["", ""])  # home, placement; then the terminal closes
        fake = op.guard.read_line

        def read(prompt: str, timeout_s: float | None) -> str | None:
            if P.INTERRUPTS.hangup is not None:  # what timed_input does after a SIGHUP
                reader.calls.append((prompt, timeout_s, r.fol.bus.torque_enabled))
                raise P.Hangup("SIGHUP: the operator's terminal closed")
            return fake(prompt, timeout_s)

        op.guard.read_line = read
        hook = _after_first_action(r, lambda: setattr(P.INTERRUPTS, "hangup", "SIGHUP"))
        t0 = op.guard.clock()
        pilot.session(
            real_cfg(), r.loop, op, r.server_info, **_pilot_kw(tmp_path), on_trial_start=hook
        )
        res = _record(tmp_path)
        (t,) = res["trials"]
        assert t["policy_started"] and t["outcome"] == "failure"
        assert t["stop_reason"] == "hangup" and t["fault"]["kind"] == "hangup"
        assert res["session_end"].startswith("hangup: SIGHUP")
        assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
        assert res["shutdown_detail"]["cause"] == "hangup"
        assert op.guard.clock() - t0 == pytest.approx(10.0)  # the grace period, then torque off
        assert not r.fol.bus.torque_enabled
        assert r.arm.deadman.response is DeadmanResponse.TORQUE_OFF


def test_lost_foreground_during_motion_is_polled_per_tick_and_takes_the_hangup_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job control (owner 2026-11-08): the process group loses the terminal's foreground while
    the arm moves (``tcgetpgrp`` polled by ``DeployLoop`` every tick; faked here). Motion stops
    at the next tick, the started trial is a failure (fault kind ``lost_foreground``), then the
    grace period, torque off, ``released_unconfirmed`` with cause ``lost_foreground``. The real
    prompt reader and a real terminal are exercised in tests/test_job_control.py."""
    from src.robot import prompts as P

    lost = [False]
    monkeypatch.setattr(P, "not_foreground", lambda _fd: "pgrp 1 != 2" if lost[0] else None)
    with rig() as r, P.job_control():
        P.INTERRUPTS.tty_fd = 0  # as with stdin on the terminal
        r.loop.interrupts = P.INTERRUPTS
        op, reader = _console(r, ["", ""])  # home, placement; then the foreground is lost
        fake = op.guard.read_line

        def read(prompt: str, timeout_s: float | None) -> str | None:
            with P._reading():  # the real check timed_input makes before every read
                return fake(prompt, timeout_s)

        op.guard.read_line = read
        writes: list[int] = []

        def lose() -> None:
            lost[0] = True
            writes.append(len(r.fol.sent))

        hook = _after_first_action(r, lose)
        t0 = op.guard.clock()
        pilot.session(
            real_cfg(), r.loop, op, r.server_info, **_pilot_kw(tmp_path), on_trial_start=hook
        )
        res = _record(tmp_path)
        (t,) = res["trials"]
        assert t["policy_started"] and t["outcome"] == "failure"
        assert t["stop_reason"] == "lost_foreground" and t["fault"]["kind"] == "lost_foreground"
        assert res["session_end"].startswith("lost_foreground: lost the terminal's foreground")
        assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
        assert res["shutdown_detail"]["cause"] == "lost_foreground"
        assert op.guard.clock() - t0 == pytest.approx(10.0)  # the grace period, then torque off
        assert not r.fol.bus.torque_enabled
        assert r.arm.deadman.response is DeadmanResponse.TORQUE_OFF
        assert len(r.fol.sent) - writes[0] <= 2  # the next tick's check stopped it (+ the hold)
        assert len(reader.calls) == 2  # the support prompt was never read from the terminal


@pytest.mark.parametrize("where", ["servo_read", "eof_grace"])
def test_ctrl_c_in_the_shutdown_never_skips_it(tmp_path: Path, where: str) -> None:
    """Without the flag handler (a fallback): a Ctrl-C at the home prompt ends the session, then
    a second, panicked press lands in the shutdown's servo read or the closed-stdin grace. The
    shutdown still finishes and the record still says how it ended."""
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        script: list[Any] = ["", "", "y", KeyboardInterrupt]  # trial 1 scored, Ctrl-C at home 2
        script.append("" if where == "servo_read" else EOFError)
        op, _reader = _console(r, script)
        hit: list[str] = []
        real = r.arm.check_servos

        def check_servos() -> None:
            stack = "".join(traceback.format_stack())
            if where == "servo_read" and not hit and "safe_shutdown" in stack:
                hit.append(stack)  # the shutdown's read, right before the support wait
                raise KeyboardInterrupt
            real()

        r.arm.check_servos = check_servos  # type: ignore[method-assign]
        if where == "eof_grace":
            gsleep = op.guard.sleep
            pressed = {"n": 0}

            def sleep(dt: float) -> None:
                if pressed["n"] == 0:
                    pressed["n"] += 1
                    raise KeyboardInterrupt
                gsleep(dt)

            op.guard.sleep = sleep
        pilot.session(real_cfg(), r.loop, op, r.server_info, **_pilot_kw(tmp_path, n_trials=2))
        res = _record(tmp_path)
        assert [t["outcome"] for t in res["trials"]] == ["success"]  # the scored trial is kept
        assert res["session_end"] == "operator_ctrl_c"
        if where == "servo_read":
            assert len(hit) == 1  # the press landed in the shutdown's servo read
            assert res["shutdown"] == "supported" and res["support_confirmed"] is True
        else:
            assert res["shutdown"] == "released_unconfirmed" and res["support_confirmed"] is False
            assert res["shutdown_detail"]["cause"] == "eof" and pressed["n"] == 1
        assert not r.fol.bus.torque_enabled


def test_a_shutdown_that_raises_still_leaves_the_trials_and_says_so(tmp_path: Path) -> None:
    with rig() as r:
        op, _reader = _console(r, ["", "", "y"])

        def support(msg: str) -> str:
            raise RuntimeError("bug in the shutdown")

        op.support = support  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="bug in the shutdown"):
            pilot.session(real_cfg(), r.loop, op, r.server_info, **_pilot_kw(tmp_path))
        res = _record(tmp_path)
        assert [t["outcome"] for t in res["trials"]] == ["success"]
        assert res["shutdown"].startswith("error: RuntimeError")
        assert res["support_confirmed"] is False and res["session_end"] == "completed"


def test_a_started_trial_is_recorded_even_when_the_session_dies_before_its_record() -> None:
    rows: list[dict[str, Any]] = []
    with rig() as r:

        def die() -> None:
            raise SystemExit("killed mid-trial")

        hook = _after_first_action(r, die)
        trials: list[dict[str, Any]] = []
        with pytest.raises(SystemExit):
            _trials(
                r,
                ScriptedOperator(),
                trials=trials,
                on_trial_start=hook,
                log_row=lambda **row: rows.append(row),
            )
        (t,) = trials
        assert t["policy_started"] and t["outcome"] == "failure"
        assert t["fault"]["kind"] == "session_lost" and "SystemExit" in t["fault"]["reason"]
        kinds = [row["kind"] for row in rows]
        assert kinds.index("trial_started") < kinds.index("decision") < kinds.index("trial")
        started = next(row for row in rows if row["kind"] == "trial_started")
        assert started["fault"]["kind"] == "session_lost" and started["policy_started"]
        assert all(row["cell"] == "seen" for row in rows if row["kind"] == "decision")
