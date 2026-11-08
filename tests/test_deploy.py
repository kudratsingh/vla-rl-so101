"""WO-07 deploy client + policy server against a fake arm, fake camera and a mock policy.

Real gRPC on 127.0.0.1 (ephemeral port), the real safety filter; one shared FakeClock for the
driver, deadman, camera, server and loop, so latencies are exact. No serial port, no camera.
"""

from __future__ import annotations

import dataclasses
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
from robot_fakes import LAST_TORQUE_OFF_CALL, FakeCam, FakeClock, FakeFollower, real_cfg, toy_fk

from src.config import REPO_ROOT, load_config
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
    latency_lines,
    run_trials,
    safe_shutdown,
    shutdown_session,
    summarize,
)
from src.robot.driver import So101Driver
from src.robot.safety import make_safety
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
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def wait(self, msg: str) -> None:
        self.prompts.append(msg)


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
        assert t["outcome"] == "failure"  # it ran: scored (doubt = n), never a free redo
        assert not r.fol.bus.torque_enabled and r.policy.calls == 2
        # torque already off: the operator supports the arm before the score (one ack), no
        # HOLDING prompt, torque off retried
        assert op.prompts == ["press ENTER once the arm rests safely (supported or on the table): "]
        assert "SERIAL FAULT" in (out := capsys.readouterr().out) and "already OFF" in out
        assert r.fol.bus.calls[-1] == LAST_TORQUE_OFF_CALL
        # the bus answers again: still nothing is written while the latch is TORQUE_OFF
        r.fol.fail_write = False
        n = len(r.fol.sent)
        safe_shutdown(r.arm, op)
        r.arm.hold()
        assert len(r.fol.sent) == n and len(op.prompts) == 1 and not r.fol.bus.torque_enabled


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
        assert shutdown_session(r.loop, op, end) == "supported"
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
        assert shutdown_session(r.loop, op, end) == "parked"
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
        assert trials[0]["stop_reason"] == "operator_abort" and trials[0]["outcome"] == "aborted"
        assert r.arm.deadman.tripped and r.arm.deadman.response is not None
        op = PromptLog()
        n = len(r.fol.sent)
        assert shutdown_session(r.loop, op, end) == "supported"
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
        assert shutdown_session(r.loop, op, "completed") == "torque_was_off"  # no park, no hold
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
        assert t["stop_reason_before"] == "servo_fault" and t["outcome"] == "aborted"
        assert "bus gone" in t["torque_off_unconfirmed_detail"]
        assert res["shutdown"] == "torque_off_unconfirmed"
        assert res["session_end"].startswith("torque_off_unconfirmed")
        assert res["torque_off_unconfirmed"] and res["summary"]["torque_off_unconfirmed"]
        out = capsys.readouterr().out
        assert "TORQUE OFF UNCONFIRMED" in out and "inline power switch" in out
        assert op.prompts and "inline power switch" in op.prompts[-1]
        # later code paths cannot move the arm or clear the flag
        r.fol.bus.fail_disable = False
        r.arm.deadman.reset("operator typed resume")
        assert safe_shutdown(r.arm, op, healthy=True, park=r.dcfg.park) == "torque_off_unconfirmed"
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

    def wait(self, msg: str) -> None:
        self.events.append(("wait", msg))

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


def test_servo_fault_at_the_settle_read_is_scored_after_the_support_prompt() -> None:
    with rig() as r:
        r.arm.monitor = monitor_from_config(real_cfg())
        _wrap_run_trial(r, lambda: r.fol.temperature_c.update(gripper=61.0))  # only after it
        op = EventOp(r)
        trials, end = _trials(r, op, n=2, settle_s=3.0)
        t = trials[0]
        assert len(trials) == 1 and end.startswith("torque_off: servo fault")
        assert t["stop_reason"] == "servo_fault" and t["outcome"] == "failure"  # scored, not redone
        assert t["stop_detail"].startswith("after the settling interval")
        assert t["settle"]["measured_s"] == 3.0 and "over-temperature" in t["settle"]["fault"]
        assert [e[0] for e in op.events] == ["wait", "outcome"]  # support the arm, then score
        assert "rests safely" in op.events[0][1] and not r.fol.bus.torque_enabled


def test_mid_trial_servo_fault_skips_the_settle_and_supports_before_scoring() -> None:
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
        assert t["settle"] == {"settle_s": 3.0, "measured_s": None, "skipped": "fault: torque off"}
        assert 3.0 not in sleeps  # no settling interval with the arm limp
        assert [e[0] for e in op.events] == ["wait", "outcome"]  # support first, then the score
        assert end.startswith("torque_off: servo fault") and not r.fol.bus.torque_enabled


def test_ctrl_c_at_the_score_prompt_records_a_failure_never_a_redo() -> None:
    with rig() as r:
        op = EventOp(r, interrupt_at_score=2)
        trials, end = _trials(r, op, n=3, settle_s=0.0)
        assert end == "operator_ctrl_c" and [t["trial_idx"] for t in trials] == [0, 1]
        t = trials[1]
        assert t["outcome"] == "failure" and "Ctrl-C" in t["score_interrupted"]
        assert {int(x["trial_idx"]) for x in trials if x["outcome"] in ("success", "failure")} == {
            0,
            1,
        }  # a resume (scored_indices) never re-rolls trial 1
