"""WO-07 real eval (scripts/wo07_real_eval.py), its placement grid (src/eval/real_grid.py) and the
printed grid (scripts/make_placement_grid.py), in mock mode: fake arm, fake camera, mock policy."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.config import REPO_ROOT
from src.eval import protocol as P1
from src.eval import protocol_v02 as P2
from src.eval import real_grid as G
from src.robot.client import ScriptedOperator


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("wo07_real_eval")
grid_pdf = _load("make_placement_grid")


def test_grid_lattice_matches_the_protocol_regions() -> None:
    slots = G.all_slots()
    assert len(slots) == 36 and slots[0].id == "A1" and slots[-1].id == "F6"
    seen, shifted = G.cell_slots("seen"), G.cell_slots("shifted")
    assert len(seen) == 16 and len(shifted) == 20 and G.cell_slots("unseen_objects") == seen
    assert all(P1.SEEN_REGION.contains(s.x, s.y) for s in seen)
    assert all(
        P1.SHIFTED_REGION.contains(s.x, s.y) and not P1.SEEN_REGION.contains(s.x, s.y)
        for s in shifted
    )
    a1 = slots[0]  # A = nearest the base (smallest x), 1 = robot's left (largest y)
    assert a1.x == min(s.x for s in slots) and a1.y == max(s.y for s in slots)


@pytest.mark.parametrize("task", P1.TASKS)
def test_sample_plan_is_seeded_balanced_and_feasible(task: str) -> None:
    from src.env import tasks as T

    plan = G.sample_plan(task, "seen", 50, seed=0)
    assert plan == G.sample_plan(task, "seen", 50, seed=0)  # SFT and SFT+RL sessions pair up
    assert plan != G.sample_plan(task, "seen", 50, seed=1)
    seen_ids = {s.id for s in G.cell_slots("seen")}
    counts: dict[str, int] = {}
    for p in plan:
        assert p["object_slot"] in seen_ids and p["goal_slot"] in seen_ids
        assert abs(p["object_yaw_deg"]) <= 45 and p["object_yaw_deg"] % 15 == 0
        d = math.dist(p["object_xy_m"], p["goal_xy_m"])
        if task == "Push":
            assert T.PUSH_DIST[0] <= d <= T.PUSH_DIST[1]
        else:
            assert d >= T.MIN_SEP[task]
        counts[p["object_slot"]] = counts.get(p["object_slot"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1  # every valid slot used ~equally
    assert "slot " + plan[0]["object_slot"] in G.placement_prompt("red cube", plan[0])


def test_placement_pdf_is_committed_a3_and_current() -> None:
    assert (
        grid_pdf.main(["--preset", "real", "--check"]) == 0
    )  # the committed PDF is what the script draws now
    data = (REPO_ROOT / "docs" / "assets" / "placement-grid-a3.pdf").read_bytes()
    assert data.startswith(b"%PDF-1.4") and b"/MediaBox [0 0 841.89 1190.55]" in data  # A3 pt
    assert b"10 cm scale bar" in data
    assert all(f"({s.id})".encode() in data for s in G.all_slots())  # ids = the sampler's
    ops = grid_pdf.draw()
    mm = grid_pdf.PT_PER_MM
    bar = f"{14.0 * mm:.2f} {360.0 * mm:.2f} m {114.0 * mm:.2f} {360.0 * mm:.2f} l S"
    assert bar in ops  # a 100 mm line at 1:1 (72 pt per inch)
    x, y = grid_pdf.page(0.1, 0.0)  # 10 cm along +x from the base origin is 100 mm up the page
    assert (x, y) == (grid_pdf.BASE_X_MM, grid_pdf.BASE_Y_MM + 100.0)


class QuitAfter(ScriptedOperator):
    def __init__(self, confirms: int) -> None:
        self.left = confirms

    def confirm(self, msg: str) -> bool:
        self.left -= 1
        return self.left >= 0


def _args(tmp: Path, *extra: str) -> list[str]:
    return ["--preset", "real", "--mock", "--fake-clock", "--max-steps", "16"] + [
        "--runs-dir",
        str(tmp),
        *extra,
    ]


def test_mock_session_records_protocol_02_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 2 trials per cell; the operator quits at the 5th confirm (2 confirms per trial): the seen
    # cell is done, shifted has not started
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(4))
    rec1 = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--label", "sft"))
    r1 = json.loads(rec1.read_text())["results"]
    assert r1["session_end"] == "operator_quit"
    assert r1["shutdown"] == "supported"  # robot.park.enabled false: hold + support, no park
    assert [t["cell"] for t in r1["trials"]] == ["seen", "seen"]
    assert not r1["summary"]["complete"] and not r1["summary"]["protocol_complete"]
    t0 = r1["trials"][0]
    assert t0["protocol_hash"] == P2.protocol_hash() and t0["protocol_version"] == "0.2"
    assert t0["protocol_ref"] == {"version": P1.PROTOCOL_VERSION, "hash": P1.protocol_hash()}
    assert t0["real_eval"]["judging_rule"] == ev.JUDGING_RULE_VERSION
    assert t0["placement"] == G.sample_plan("PickPlace", "seen", 2, 0)[0]
    assert "frame_return_to_first_write_ms" in t0["latency"]

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(_args(tmp_path, "--resume", rec1.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert r2["resumed_from"] == rec1.parent.name and r2["session_end"] == "completed"
    assert r2["session"]["session_id"] == rec1.parent.name and r2["session"]["label"] == "sft"
    by_cell: dict[str, list[Any]] = {}
    for t in r2["trials"]:
        by_cell.setdefault(t["cell"], []).append(t["trial_idx"])
    assert by_cell == {c: [0, 1] for c in P1.CELLS}  # seen kept, the rest run once
    s = r2["summary"]
    assert s["complete"] and not s["protocol_complete"]  # 2, not 50, per cell
    for cell in P1.CELLS:
        c = s["cells"][f"PickPlace/{cell}"]
        assert c["n"] == 2 and c["target_n"] == 2
        assert c["wilson_95"] == pytest.approx(P1.wilson_interval(c["successes"], 2), abs=1e-4)
    shifted = [t for t in r2["trials"] if t["cell"] == "shifted"]
    ring = {sl.id for sl in G.cell_slots("shifted")}
    assert shifted and all(t["placement"]["object_slot"] in ring for t in shifted)
    assert np.isfinite(s["frame_return_to_first_write_ms"]["median"])

    # a changed evaluation (here: the session's recorded hash) refuses to resume
    rec = json.loads(rec2.read_text())
    rec["results"]["session"]["protocol"]["protocol_hash"] = "0" * 64
    rec2.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="resume refused"):
        ev.main(_args(tmp_path, "--resume", rec2.parent.name))


def test_hardware_mode_is_gated_like_the_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as e:
        ev.main(["--preset", "real", "--hardware", "--task", "PickPlace"])
    assert e.value.code == 2
    monkeypatch.setattr("builtins.input", lambda _msg="": "no")
    with pytest.raises(SystemExit, match="nothing was opened"):
        ev.main(["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"])


def test_trial_records_carry_settle_fingerprint_and_paired_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec = ev.main(_args(tmp_path, "--task", "Push", "--trials", "2", "--cells", "seen"))
    r = json.loads(rec.read_text())["results"]
    fp = r["session"]["fingerprint"]
    assert set(fp["components"]) == set(ev.FINGERPRINT_COMPONENTS)
    motion = fp["values"]["motion"]
    assert motion["gripper_floor_deg"] == pytest.approx(-5.0) and motion["settle_s"] == 3.0
    assert motion["park_enabled"] is False and motion["deploy"]["exec_steps"] == 8
    assert fp["values"]["calibration"]["motors"]["wrist_flex"]["homing_offset"] == 0
    assert r["judging_rule"]["settle_s"] == 3.0 and "3 s" in r["judging_rule"]["general"]
    for t in r["trials"]:
        assert t["session_fingerprint"] == {k: fp[k] for k in ("sha256", "components")}
        # the arm held still for the settling interval before the (scripted) score
        assert t["settle"] == {"settle_s": 3.0, "measured_s": 3.0}
        assert t["placement_plan_sha256"] == ev.plan_sha256("Push", "seen", 0)
        assert t["torque_off_unconfirmed"] is False
    # the slot sequence does not depend on the session's trial count: SFT and SFT+RL pair up
    for task in P1.TASKS:
        assert G.sample_plan(task, "shifted", 3, 0) == G.sample_plan(task, "shifted", 50, 0)[:3]


def _with_cfg(monkeypatch: pytest.MonkeyPatch, edit: Any) -> None:
    real_load = ev.load_config

    def load(preset: str, path: Any = None) -> Any:
        cfg = real_load(preset, path)
        edit(cfg)
        return cfg

    monkeypatch.setattr(ev, "load_config", load)


@pytest.mark.parametrize(
    ("component", "edit"),
    [
        ("camera", lambda c: c["cameras"]["front"].update(crop_to=200)),
        ("safety", lambda c: c["robot"]["safety"].update(max_velocity_deg_s=45.0)),
        ("motion", lambda c: c["deploy"].update(request_timeout_ms=300)),
        ("motion", lambda c: c["eval"].update(settle_s=1.0)),
        ("calibration", lambda c: c["robot"]["units"].update(arm_offset_deg=[1, 0, 0, 0, 0])),
    ],
)
def test_resume_refuses_changed_effective_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str, edit: Any
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))  # 1 trial, then quit
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    _with_cfg(monkeypatch, edit)
    with pytest.raises(SystemExit, match=f"resume refused(.|\n)*{component} changed: "):
        ev.main(_args(tmp_path, "--resume", rec.parent.name))


def test_resume_refuses_a_changed_calibration_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import robot_fakes

    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    real = robot_fakes.full_sweep_calibration

    def recalibrated() -> dict[str, Any]:
        cal = real()
        c = cal["elbow_flex"]
        cal["elbow_flex"] = type(c)(c.id, c.drive_mode, 12, c.range_min, c.range_max)
        return cal

    monkeypatch.setattr(robot_fakes, "full_sweep_calibration", recalibrated)
    with pytest.raises(SystemExit, match="calibration changed: motors.elbow_flex.homing_offset"):
        ev.main(_args(tmp_path, "--resume", rec.parent.name))


def test_trials_from_different_settings_are_never_pooled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    data = json.loads(rec.read_text())
    fp = data["results"]["session"]["fingerprint"]
    trial = data["results"]["trials"][0]
    trial["session_fingerprint"] = {
        "sha256": "f" * 64,
        "components": {**fp["components"], "camera": "0" * 64},
    }
    rec.write_text(json.dumps(data))
    (rec.parent / "metrics.jsonl").write_text("")  # only record.json's (tampered) trial
    with pytest.raises(SystemExit, match="refusing to pool 1 trial(.|\n)*differs in camera"):
        ev.main(_args(tmp_path, "--resume", rec.parent.name))
    settings = data["results"]["session"]
    with pytest.raises(ev.PoolingRefused, match="no fingerprint"):
        ev.check_poolable([{**trial, "session_fingerprint": None}], settings["fingerprint"])


def test_hardware_refuses_a_slot_seed_other_than_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intr = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "dist": [0.0] * 5}
    _with_cfg(monkeypatch, lambda c: c["cameras"]["front"].update(intrinsics=intr))
    monkeypatch.setattr("builtins.input", lambda _msg="": "go")
    argv = ["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"]
    with pytest.raises(SystemExit, match="must use the same placements"):
        ev.main([*argv, "--slot-seed", "7"])  # exits before any client, camera or port


# -- PR 29 review fixes ----------------------------------------------------------------------------
class EOFAtScore(ScriptedOperator):
    def outcome(self, result: Any) -> str:
        raise EOFError  # stdin closed at the score prompt


def _records(tmp: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text())["results"] for p in sorted(tmp.glob("*/record.json"))]


@pytest.mark.parametrize("where", ["score", "home"])
def test_unexpected_error_is_never_a_healthy_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    from src.robot.client import DeployLoop
    from src.robot.driver import So101Driver

    _with_cfg(monkeypatch, lambda c: c["robot"]["park"].update(enabled=True))
    parks: list[int] = []
    monkeypatch.setattr(So101Driver, "park", lambda self, *a, **k: parks.append(1) or True)
    if where == "score":
        monkeypatch.setattr(ev, "ScriptedOperator", EOFAtScore)
        err: type[BaseException] = EOFError
    else:

        def broken_home(self: Any) -> dict[str, Any]:
            raise RuntimeError("client bug in go_home")

        monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
        monkeypatch.setattr(DeployLoop, "go_home", broken_home)
        err = RuntimeError
    with pytest.raises(err):
        ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    (r,) = _records(tmp_path)
    assert not parks and r["shutdown"] == "supported"  # hold + support, never a park
    assert r["session_end"].startswith(f"error: {err.__name__}")
    if where == "score":  # the trial ran: recorded as a failure (doubt = n), not lost
        assert [(t["trial_idx"], t["outcome"]) for t in r["trials"]] == [(0, "failure")]
        assert r["trials"][0]["unjudged"]["how"] == "eof"
        assert r["trials"][0]["outcome_source"] == "unjudged"


def test_a_killed_run_resumes_from_its_own_record_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.runs import Run

    flushes: list[int] = []
    real_flush = Run.flush

    def first_flush_only(self: Any) -> None:  # the session bind; then the process "dies"
        if not flushes:
            real_flush(self)
        flushes.append(1)

    monkeypatch.setattr(Run, "flush", first_flush_only)
    monkeypatch.setattr(Run, "finish", lambda self, status="ok": self.record_path)  # no exit
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(4))  # 2 trials, then quit
    killed = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "3", "--cells", "seen"))
    on_disk = json.loads(killed.read_text())["results"]
    assert on_disk["session"]["session_id"] == killed.parent.name and on_disk["trials"] == []
    _, prior = ev.load_prior(killed.parent)
    assert [t["trial_idx"] for t in prior] == [0, 1]  # from metrics.jsonl
    monkeypatch.setattr(Run, "flush", real_flush)
    monkeypatch.undo()
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec = ev.main(_args(tmp_path, "--resume", killed.parent.name))
    r = json.loads(rec.read_text())["results"]
    assert sorted(t["trial_idx"] for t in r["trials"]) == [0, 1, 2]  # only trial 2 ran again
    assert r["summary"]["cells"]["PickPlace/seen"]["n"] == 3


def test_resume_from_an_older_run_of_the_session_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))  # 1 trial per run
    a = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "3", "--cells", "seen"))
    b = ev.main(_args(tmp_path, "--resume", a.parent.name))
    rb = json.loads(b.read_text())["results"]
    assert rb["session_runs"] == [a.parent.name, b.parent.name]
    with pytest.raises(SystemExit, match=f"continued session .* resume {b.parent.name}"):
        ev.main(_args(tmp_path, "--resume", a.parent.name))
    c = ev.main(_args(tmp_path, "--resume", b.parent.name))  # the newest run resumes
    assert len(json.loads(c.read_text())["results"]["trials"]) == 3


def test_resume_refuses_different_weights_under_the_same_policy_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.robot.serve import PolicyServer

    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    fp = json.loads(rec.read_text())["results"]["session"]["fingerprint"]
    assert fp["values"]["policy"]["adapter_sha256"] == "mock:no-weights"
    real_info = PolicyServer.info
    monkeypatch.setattr(
        PolicyServer, "info", lambda self: {**real_info(self), "adapter_sha256": "re-pushed"}
    )
    with pytest.raises(SystemExit, match="policy changed: adapter_sha256"):
        ev.main(_args(tmp_path, "--resume", rec.parent.name))
    trial = json.loads(rec.read_text())["results"]["trials"][0]
    values = {**fp["values"], "policy": {**fp["values"]["policy"], "adapter_sha256": "re-pushed"}}
    with pytest.raises(ev.PoolingRefused, match="differs in policy"):
        ev.check_poolable([trial], ev.make_fingerprint(values))


def test_resume_warns_about_a_torque_off_failure_at_the_last_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from src.robot.driver import So101Driver

    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    real_torque_off = So101Driver.torque_off

    def fail_at_shutdown(self: So101Driver) -> bool:  # every trial is clean; shutdown's fails
        self.follower.bus.fail_disable = True
        return real_torque_off(self)

    monkeypatch.setattr(ev, "shutdown_session", _shutdown_with(fail_at_shutdown, monkeypatch))
    rec = ev.main(_args(tmp_path, "--task", "Push", "--trials", "1", "--cells", "seen"))
    r = json.loads(rec.read_text())["results"]
    assert r["shutdown"] == "torque_off_unconfirmed"
    assert not any(t["torque_off_unconfirmed"] for t in r["trials"])  # no trial carries it
    monkeypatch.undo()
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    capsys.readouterr()
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name, "--trials", "1"))
    assert ev.PRIOR_TORQUE_OFF_UNCONFIRMED in capsys.readouterr().out
    s2 = json.loads(rec2.read_text())["results"]["session"]
    assert s2["prior_torque_off_unconfirmed"] is True  # carried to every later run
    with pytest.raises(SystemExit, match="not acknowledged"):  # hardware: must type 'checked'
        ev.bind_session(real_cfg_for_bind(), s2, [], "x", *_bind_rig(), acknowledge=lambda _m: "no")


def _shutdown_with(torque_off: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from src.robot.driver import So101Driver

    real = ev.shutdown_session

    def shutdown(loop: Any, operator: Any, end: str) -> str:
        monkeypatch.setattr(So101Driver, "torque_off", torque_off)
        return real(loop, operator, end)

    return shutdown


def real_cfg_for_bind() -> Any:
    return ev.load_config("real")


def _bind_rig() -> tuple[Any, ...]:
    import sys as _sys

    _sys.path.insert(0, str(REPO_ROOT / "tests"))
    from robot_fakes import FakeFollower, toy_fk

    from src.robot.client import DeployConfig
    from src.robot.driver import So101Driver
    from src.robot.safety import make_safety
    from src.robot.units import UnitMap

    cfg = real_cfg_for_bind()
    arm = So101Driver(FakeFollower(), UnitMap(), make_safety(cfg, toy_fk))
    info = {"policy_id": "mock:pickplace-waypoints", "adapter_sha256": "mock:no-weights"}
    return arm, info, DeployConfig.from_config(cfg), 16


def test_placement_plan_is_prefix_stable_beyond_50_trials() -> None:
    for task in P1.TASKS:
        long = G.sample_plan(task, "seen", 80, 0)
        assert long[:50] == G.sample_plan(task, "seen", 50, 0)
        assert G.sample_plan(task, "seen", 65, 0) == long[:65]
    assert ev.plan_sha256("Push", "seen", 0, 2) == ev.plan_sha256("Push", "seen", 0, 50)
    assert ev.plan_sha256("Push", "seen", 0, 80) != ev.plan_sha256("Push", "seen", 0, 50)


# -- owner decisions 2026-11-08: bounded prompts, honest shutdown, fault scoring ------------------
def _judge_with(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> Any:
    """The real-eval JudgeOperator on a scripted stdin, bound to the mock rig's arm."""
    from contextlib import contextmanager

    from robot_fakes import FakeClock, FakeReader

    from src.robot.prompts import TorqueGuard, TorqueWait

    gclock = FakeClock()
    reader = FakeReader(script, gclock)
    guard = TorqueGuard(None, TorqueWait(60.0, 10.0), read=reader, clock=gclock, sleep=gclock.sleep)  # type: ignore[arg-type]
    op = ev.JudgeOperator(guard, "PickPlace", 3.0)
    real_rig = ev.mock_rig

    @contextmanager
    def rig(cfg: Any, args: Any, fk: Any) -> Any:
        with real_rig(cfg, args, fk) as out:
            guard.arm = out[0].arm
            reader.bus = out[0].arm.follower.bus
            yield out

    monkeypatch.setattr(ev, "mock_rig", rig)
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: op)
    return reader


@pytest.mark.parametrize("how", ["timeout", "ctrl_c", "eof"])
def test_real_eval_score_prompt_without_a_judgement_is_a_failure_never_redone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    answer = {"timeout": None, "ctrl_c": KeyboardInterrupt, "eof": EOFError}[how]
    script: list[Any] = ["", "", answer] + ([""] if how == "ctrl_c" else [])
    reader = _judge_with(monkeypatch, script)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen")
    if how == "eof":
        with pytest.raises(EOFError):
            ev.main(argv)
    else:
        ev.main(argv)
    (r,) = _records(tmp_path)
    (t,) = r["trials"]
    assert t["policy_started"] and t["outcome"] == "failure"
    assert t["outcome_source"] == "unjudged" and t["unjudged"]["how"] == how
    assert t["settle"]["measured_s"] == 3.0  # judged (or not) only after the settling interval
    assert all(timeout is not None for _, timeout, on in reader.calls if on)  # all bounded
    expected = {
        "timeout": ("prompt_timeout", "released_unconfirmed", False),
        "ctrl_c": ("operator_ctrl_c", "supported", True),
        "eof": ("error: EOFError", "released_unconfirmed", False),
    }[how]
    assert r["session_end"].startswith(expected[0]) and r["shutdown"] == expected[1]
    assert r["support_confirmed"] is expected[2]
    c = r["summary"]["cells"]["PickPlace/seen"]
    assert (c["n"], c["successes"], c["unjudged"]) == (1, 0, 1)
    # a resume never re-runs the started slot: only trial 1 runs
    monkeypatch.undo()
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    rec2 = ev.main(_args(tmp_path, "--resume", (tmp_path / r["session"]["session_id"]).name))
    r2 = json.loads(rec2.read_text())["results"]
    assert [x["trial_idx"] for x in r2["trials"]] == [0, 1]
    assert r2["summary"]["cells"]["PickPlace/seen"]["n"] == 2


@pytest.mark.parametrize("how", ["timeout", "ctrl_c", "eof"])
def test_real_eval_home_and_support_prompts_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    answer = {"timeout": None, "ctrl_c": KeyboardInterrupt, "eof": EOFError}[how]
    # the first home prompt, then (Ctrl-C) the support prompt: Ctrl-C there too, then silence
    script: list[Any] = [answer] + ([KeyboardInterrupt, None] if how == "ctrl_c" else [])
    reader = _judge_with(monkeypatch, script)
    argv = _args(tmp_path, "--task", "PickPlace", "--trials", "1", "--cells", "seen")
    if how == "eof":
        with pytest.raises(EOFError):
            ev.main(argv)
    else:
        ev.main(argv)
    (r,) = _records(tmp_path)
    assert r["trials"] == []  # the policy never started
    assert r["shutdown"] == "released_unconfirmed" and r["support_confirmed"] is False
    assert r["shutdown_detail"]["cause"] == ("timeout" if how == "ctrl_c" else how)
    on = [c for c in reader.calls if c[2]]
    assert on and all(timeout is not None for _, timeout, _ in on)
    if how == "ctrl_c":  # Ctrl-C at the support prompt kept the arm holding (torque on)
        assert reader.calls[1][2] is True and reader.calls[2][2] is True
        assert reader.calls[2][1] == pytest.approx(55.0)  # the bounded wait kept running


def test_faults_after_the_start_fail_and_unstarted_attempts_are_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attempt 1 (slot 0): the server fails on the first query, before any action: not started,
    unscored, retried with the same placement. Attempt 2 (slot 1): the server fails on the second
    query, after the start: a failure (fault kind server), not judged, a later 'resume' does not
    change it. Slot 2 runs clean."""
    from src.robot.serve import MockPolicy

    state = {"attempt": 0}
    fail_at = {1: 1, 2: 2}  # attempt -> the call that fails
    real_reset, real_act = MockPolicy.reset, MockPolicy.act

    def reset(self: Any) -> None:
        state["attempt"] += 1
        real_reset(self)

    def act(self: Any, obs: Any, instruction: str) -> Any:
        if fail_at.get(state["attempt"]) == self.calls + 1:
            self.calls += 1
            raise RuntimeError("mock server failure")
        return real_act(self, obs, instruction)

    monkeypatch.setattr(MockPolicy, "reset", reset)
    monkeypatch.setattr(MockPolicy, "act", act)
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)  # resumes after HOLD stops
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "3", "--cells", "seen"))
    r = json.loads(rec.read_text())["results"]
    t0, t1, t2 = r["trials"]
    assert (t0["trial_idx"], t0["outcome"], t0["policy_started"]) == (0, "not_started", False)
    assert t0["stop_reason"] == "server_error" and t0["fault"] is None
    assert (t1["trial_idx"], t1["outcome"], t1["outcome_source"]) == (1, "failure", "fault")
    assert t1["policy_started"] and t1["policy_started_utc"]
    assert t1["fault"]["kind"] == "server" and t1["fault"]["stop_reason"] == "server_error"
    assert (t2["trial_idx"], t2["outcome"]) == (2, "success")
    c = r["summary"]["cells"]["PickPlace/seen"]
    assert (c["n"], c["successes"], c["not_started"], c["faulted"]) == (2, 1, 1, 1)
    assert c["success_rate"] == 0.5 and c["faults_by_kind"] == {"server": 1}
    assert c["diagnostic_excluding_faulted"] == {
        "n": 1,
        "successes": 1,
        "success_rate": 1.0,
        "note": c["diagnostic_excluding_faulted"]["note"],
    }
    assert "SECONDARY" in c["diagnostic_excluding_faulted"]["note"]
    assert not r["summary"]["complete"]  # slot 0 still has no started attempt
    rec2 = ev.main(_args(tmp_path, "--resume", rec.parent.name))
    r2 = json.loads(rec2.read_text())["results"]
    assert [(t["trial_idx"], t["outcome"]) for t in r2["trials"][3:]] == [(0, "success")]
    assert r2["trials"][3]["placement"] == t0["placement"]  # the same placement
    c2 = r2["summary"]["cells"]["PickPlace/seen"]
    assert (c2["n"], c2["successes"], c2["not_started"]) == (3, 2, 1) and r2["summary"]["complete"]


def test_judging_rule_bump_refuses_a_real_judge_1_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ev.JUDGING_RULE_VERSION == "real-judge-2"
    assert "policy started" in ev.judging_rule(3.0)["general"]
    monkeypatch.setattr(ev, "ScriptedOperator", lambda: QuitAfter(2))
    rec = ev.main(_args(tmp_path, "--task", "PickPlace", "--trials", "2", "--cells", "seen"))
    data = json.loads(rec.read_text())
    session = data["results"]["session"]
    for holder in (session["protocol"], session["fingerprint"]["values"]["protocol"]):
        holder["real_eval"]["judging_rule"] = "real-judge-1"  # recorded under the old rule
    rec.write_text(json.dumps(data))
    monkeypatch.setattr(ev, "ScriptedOperator", ScriptedOperator)
    with pytest.raises(SystemExit, match="resume refused(.|\n)*real_eval"):
        ev.main(_args(tmp_path, "--resume", rec.parent.name))


@pytest.mark.parametrize("bad", [None, 0, -1])
def test_hardware_refuses_an_unbounded_torque_on_wait(
    monkeypatch: pytest.MonkeyPatch, bad: Any
) -> None:
    pilot = _load("wo07_pilot")
    asked: list[str] = []
    monkeypatch.setattr("builtins.input", lambda msg="": asked.append(msg) or "go")
    for mod, argv in (
        (ev, ["--preset", "real", "--hardware", "--i-am-present", "--task", "PickPlace"]),
        (pilot, ["--preset", "real", "--hardware", "--i-am-present"]),
    ):
        _with_cfg_in(mod, monkeypatch, lambda c: c["deploy"].update(prompt_torque_timeout_s=bad))
        with pytest.raises(SystemExit, match="prompt_torque_timeout_s(.|\n)*nothing was opened"):
            mod.main(argv)
    assert asked == []  # refused before the 'go' prompt: nothing was opened


def _with_cfg_in(mod: ModuleType, monkeypatch: pytest.MonkeyPatch, edit: Any) -> None:
    real_load = mod.load_config

    def load(preset: str, path: Any = None) -> Any:
        cfg = real_load(preset, path)
        edit(cfg)
        return cfg

    monkeypatch.setattr(mod, "load_config", load)
