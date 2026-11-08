#!/usr/bin/env python
"""WO-01r: rebuild the 25 demos/task on the GPU renderer, and measure the Mac-vs-GPU render gap.

Review 2026-11-07 section 3.2 / fix list: every 0.1-era demo was rendered on the Mac CPU renderer
(MoltenVK), truncated at the success step and recorded with the fast v1 oracle, while every eval
and RL rollout renders on the NVIDIA GPU sim. Two modes:

  build       scripts/wo01_build_env.py with the fixed recorder (release + retreat recorded) and the
              v2 oracle (configs/demos_v2.yaml), then a tail check of the written dataset (every
              PickPlace/Stack demo ends at the release target, every demo's oracle script ran to
              its end, TCP rise after success). `--push` uploads it (private) only if that passes.
  render-gap  re-render the stored demos on this machine's backend: reset each demo's seed, replay
              its recorded actions open loop, and diff every fresh frame against the stored
              (AV1-decoded) frame; the stored-vs-replayed joint state shows where physics diverges.
              Then compare the SFT policy's greedy action tokens on stored vs fresh frames (same
              stored joint state, so only the image differs). A codec floor -- the first
              `--floor-episodes` fresh episodes per task written through the same AV1 recorder and
              decoded again -- gives the pixel and token differences that are codec noise alone.
  dry-run     (tiny default) build, then render-gap against the dataset just built: CPU on both
              sides, so the gap must be codec noise only.

    uv run scripts/wo01r_rebuild_demos.py --preset tiny                           # CPU, < 2 min
    uv run scripts/wo01r_rebuild_demos.py --preset sim --mode render-gap          # GPU box (~5 min)
    uv run scripts/wo01r_rebuild_demos.py --preset sim --mode build --push        # GPU box
    uv run scripts/wo01r_rebuild_demos.py --preset sim --device cpu --mode render-gap \
        --skip-tokens --gap-episodes 3                                            # Mac: CPU-vs-CPU floor

Non-tiny presets default to configs/demos_v2.yaml (render_gap.* there: the 0.1 dataset and the
WO-02 attempt-4 adapter, a Hub path). tiny uses configs/tiny.yaml and the tiny random OFT policy
(configs/sft_tiny.yaml). Writes runs/<id>/record.json and runs/<id>/render_gap.json.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import Config, add_config_args, load_config
from src.runs import Run

RELEASE_TOL = 1e-3  # rad: last gripper command counts as "at the release target"
PIXEL_THRESH = 16  # |diff| (0-255) above which a pixel counts as changed
MODES = ("build", "render-gap", "dry-run")


def default_config(preset: str) -> Path:
    return REPO_ROOT / "configs" / ("tiny.yaml" if preset == "tiny" else "demos_v2.yaml")


def _build_env_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "wo01_build_env", REPO_ROOT / "scripts" / "wo01_build_env.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- dataset reading
def read_episodes(root: Path) -> dict[int, dict[str, np.ndarray[Any, Any]]]:
    """{episode_index: {"state": (T, 6), "action": (T, 6)}} from the parquet files (no video)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet data under {root / 'data'}")
    table = pa.concat_tables([pq.read_table(f) for f in files])
    ep = table.column("episode_index").to_numpy()
    fi = table.column("frame_index").to_numpy()
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    out: dict[int, dict[str, np.ndarray[Any, Any]]] = {}
    for e in np.unique(ep):
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(fi[rows])]
        out[int(e)] = {"state": state[rows], "action": action[rows]}
    return out


def tail_report(root: Path, tasks: list[str]) -> dict[str, dict[str, Any]]:
    """Per task: how the successful demos end (last gripper command, oracle finished, TCP rise)."""
    from src.data.oracle import GRIPPER_RELEASE
    from src.data.recorder import read_sidecar
    from src.env.kinematics import ArmKinematics

    kin = ArmKinematics.from_urdf()
    eps = read_episodes(root)
    side = [r for r in read_sidecar(root) if r["oracle_success"]]
    out: dict[str, dict[str, Any]] = {}
    for task in tasks:
        rows = [r for r in side if r["task_family"] == task]
        last_grip, rise, finished, tails = [], [], [], []
        for r in rows:
            e = eps[int(r["episode_index"])]
            last_grip.append(float(e["action"][-1, 5]))
            finished.append(bool(r.get("oracle_finished", False)))
            tails.append(int(r.get("post_success_frames", 0)))
            k = int(r.get("success_step", len(e["state"]))) - 1
            z = kin.fk(torch.as_tensor(e["state"][[k, -1]], dtype=torch.float32))[:, 2, 3]
            rise.append(100.0 * float(z[1] - z[0]))
        grasp = task != "Push"  # Push keeps the gripper closed throughout
        at_release = [abs(g - GRIPPER_RELEASE) < RELEASE_TOL for g in last_grip]
        out[task] = {
            "demos": len(rows),
            "end_below_release": (
                sum(g < GRIPPER_RELEASE - RELEASE_TOL for g in last_grip) if grasp else None
            ),
            "end_at_release": sum(at_release) if grasp else None,
            "oracle_finished": sum(finished),
            "mean_post_success_frames": round(float(np.mean(tails)), 2) if tails else 0.0,
            "mean_tcp_rise_after_success_cm": round(float(np.mean(rise)), 2) if rise else 0.0,
            "min_tcp_rise_after_success_cm": round(float(np.min(rise)), 2) if rise else 0.0,
        }
    return out


def tail_ok(report: dict[str, dict[str, Any]], demos: int) -> list[str]:
    """Problems that block a push: missing demos, unfinished release/retreat."""
    bad = []
    for task, r in report.items():
        if r["demos"] < demos:
            bad.append(f"{task}: {r['demos']} demos < {demos}")
        if r["end_at_release"] is not None and r["end_at_release"] < r["demos"]:
            bad.append(
                f"{task}: {r['demos'] - r['end_at_release']} demos end off the release target"
            )
        if r["oracle_finished"] < r["demos"]:
            bad.append(
                f"{task}: {r['demos'] - r['oracle_finished']} demos stop before retreat ends"
            )
    return bad


# ------------------------------------------------------------------------------ render gap
def _to_uint8_hwc(chw: torch.Tensor) -> torch.Tensor:
    """LeRobot decoded frame (3, H, W) float in [0, 1] -> (H, W, 3) uint8."""
    return (chw.float() * 255.0).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)


def _pixel_stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """(mean |diff| in 0-255, PSNR dB, fraction of pixels with any channel |diff| > threshold)."""
    d = (a.float() - b.float()).abs()
    mse = float((d**2).mean())
    psnr = 99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse)
    return float(d.mean()), psnr, float((d.amax(-1) > PIXEL_THRESH).float().mean())


@torch.no_grad()
def _greedy(
    policy: Any, images: torch.Tensor, states: torch.Tensor, instr: list[str], batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    toks, acts = [], []
    for i in range(0, len(images), batch):
        s = policy.sample(
            {"image": images[i : i + batch], "state": states[i : i + batch]},
            instr[i : i + batch],
            temperature=0.0,
        )
        toks.append(s.tokens.cpu())
        acts.append(s.actions.cpu())
    return torch.cat(toks), torch.cat(acts)


def _token_stats(
    policy: Any,
    a: torch.Tensor,
    b: torch.Tensor,
    states: torch.Tensor,
    instr: list[str],
    batch: int,
) -> dict[str, float]:
    ta, aa = _greedy(policy, a, states, instr, batch)
    tb, ab = _greedy(policy, b, states, instr, batch)
    return {
        "frames": len(a),
        "token_agreement": round(float((ta == tb).float().mean()), 4),
        "first_action_token_agreement": round(float((ta[:, 0] == tb[:, 0]).float().mean()), 4),
        "chunk_identical": round(float((ta == tb).flatten(1).all(-1).float().mean()), 4),
        "mean_abs_bin_diff": round(float((ta - tb).abs().float().mean()), 3),
        "mean_abs_action_diff_rad": round(float((aa - ab).abs().mean()), 5),
    }


def _summary(rows: list[tuple[float, float, float]]) -> dict[str, float | None]:
    if not rows:
        return {"frames": 0, "mean_abs": None, "psnr_db": None, "frac_pixels_changed": None}
    arr = np.asarray(rows)
    return {
        "frames": len(rows),
        "mean_abs": round(float(arr[:, 0].mean()), 3),
        "psnr_db": round(float(arr[:, 1].mean()), 2),
        "frac_pixels_changed": round(float(arr[:, 2].mean()), 5),
    }


def render_gap(
    cfg: Config,
    repo_id: str,
    root: Path,
    policy: Any | None,
    *,
    episodes_per_task: int | None,
    token_stride: int,
    floor_episodes: int,
    work_dir: Path,
    batch: int = 16,
) -> dict[str, Any]:
    """Per task: pixel diffs (t = 0 and all frames), joint-state divergence, token agreement, and
    the codec floor of the same quantities. See the module docstring."""
    from src.data.recorder import IMAGE_KEY, load_demo_dataset, read_sidecar
    from src.env import make_env, policy_obs

    ds = load_demo_dataset(repo_id, root)
    eps = read_episodes(root)
    side = [r for r in read_sidecar(root) if r["oracle_success"]]
    start = {
        int(e): int(s)
        for e, s in zip(
            ds.meta.episodes["episode_index"], ds.meta.episodes["dataset_from_index"], strict=True
        )
    }
    cuda = str(cfg.get("device", "cpu")).startswith("cuda")
    floor: list[dict[str, Any]] = []  # fresh episodes to push through the codec afterwards
    out: dict[str, Any] = {"dataset": repo_id, "tasks": {}}
    for task in cfg["env"]["tasks"]:
        rows = [r for r in side if r["task_family"] == task][:episodes_per_task]
        if not rows:
            continue
        floor_ids = {int(r["episode_index"]) for r in rows[:floor_episodes]}
        t0 = time.monotonic()
        pix_t0: list[tuple[float, float, float]] = []
        pix_all: list[tuple[float, float, float]] = []
        dq_t0: list[float] = []
        dq_all: list[float] = []
        tok: dict[str, list[Any]] = {"stored": [], "fresh": [], "state": [], "instr": []}
        for cell in sorted({str(r["cell"]) for r in rows}):
            todo = [r for r in rows if str(r["cell"]) == cell]
            n = len(todo) if cuda else 1
            env = make_env(cfg, task, num_envs=n, cell=cell, render=True)
            env_device = getattr(env.unwrapped, "device", "cpu")
            for b0 in range(0, len(todo), n):
                group = todo[b0 : b0 + n]
                real = len(group)
                group = group + [group[-1]] * (n - real)  # pad the last batch (results unused)
                seeds = [int(r["seed"]) for r in group]
                obs, _ = env.reset(seed=seeds if n > 1 else seeds[0])
                instr = [str(r["instruction"]) for r in group]  # what SFT trained with
                ep_ids = [int(r["episode_index"]) for r in group]
                lens = [len(eps[e]["action"]) for e in ep_ids]
                buf = {
                    i: {
                        "task": task,
                        "seed": seeds[i],
                        "cell": cell,
                        "instr": instr[i],
                        "frames": [],
                    }
                    for i in range(real)
                    if ep_ids[i] in floor_ids
                }
                for t in range(max(lens[:real])):
                    po = policy_obs(obs)
                    fresh, fstate = po["image"].cpu(), po["state"].cpu()
                    for i in range(real):
                        if t >= lens[i]:
                            continue
                        e = ep_ids[i]
                        stored = _to_uint8_hwc(ds[start[e] + t][IMAGE_KEY])
                        st = torch.as_tensor(eps[e]["state"][t])
                        stats = _pixel_stats(stored, fresh[i])
                        dq = float((fstate[i] - st).abs().max())
                        pix_all.append(stats)
                        dq_all.append(dq)
                        if t == 0:
                            pix_t0.append(stats)
                            dq_t0.append(dq)
                        if policy is not None and t % token_stride == 0:
                            tok["stored"].append(stored)
                            tok["fresh"].append(fresh[i].clone())
                            tok["state"].append(st)
                            tok["instr"].append(instr[i])
                        if i in buf:
                            buf[i]["frames"].append((fresh[i].clone(), st, eps[e]["action"][t], t))
                    act = np.stack(
                        [eps[e]["action"][min(t, ln - 1)] for e, ln in zip(ep_ids, lens)]
                    )
                    obs, *_ = env.step(torch.as_tensor(act, device=env_device))
                floor += list(buf.values())
            env.close()
        res: dict[str, Any] = {
            "episodes": len(rows),
            "pixels_t0": _summary(pix_t0),
            "pixels_all": _summary(pix_all),
            "state_max_abs_diff_t0_rad": round(max(dq_t0), 6) if dq_t0 else None,
            "state_mean_max_abs_diff_rad": round(float(np.mean(dq_all)), 6) if dq_all else None,
        }
        if policy is not None and tok["stored"]:
            res["tokens_stored_vs_fresh"] = _token_stats(
                policy,
                torch.stack(tok["stored"]),
                torch.stack(tok["fresh"]),
                torch.stack(tok["state"]),
                tok["instr"],
                batch,
            )
        res["seconds"] = round(time.monotonic() - t0, 1)
        out["tasks"][task] = res
    if floor:
        for task, fl in codec_floor(floor, policy, token_stride, work_dir, batch).items():
            out["tasks"][task]["codec_floor"] = fl
    return out


def codec_floor(
    floor: list[dict[str, Any]], policy: Any | None, token_stride: int, work_dir: Path, batch: int
) -> dict[str, dict[str, Any]]:
    """Write fresh episodes through the demo recorder's AV1 path, decode them, and diff against
    the raw frames: the pixel / token differences that are codec noise alone."""
    from src.data.recorder import IMAGE_KEY, DemoRecorder, load_demo_dataset

    h, w = (int(x) for x in floor[0]["frames"][0][0].shape[:2])
    rec = DemoRecorder("local/render-gap-floor", work_dir / "codec_floor", image_hw=(h, w))
    for ep in floor:
        for img, st, act, _ in ep["frames"]:
            rec.add_frame(img, st, act, ep["instr"])
        rec.save_episode(
            {
                "task_family": ep["task"],
                "seed": ep["seed"],
                "cell": ep["cell"],
                "oracle_success": True,
            }
        )
    rec.finalize()
    dec = load_demo_dataset("local/render-gap-floor", work_dir / "codec_floor")
    out: dict[str, dict[str, Any]] = {}
    k = 0
    for task in dict.fromkeys(ep["task"] for ep in floor):
        pix: list[tuple[float, float, float]] = []
        tok: dict[str, list[Any]] = {"raw": [], "dec": [], "state": [], "instr": []}
        for ep in (e for e in floor if e["task"] == task):
            for img, st, _, t in ep["frames"]:
                d = _to_uint8_hwc(dec[k][IMAGE_KEY])
                k += 1
                pix.append(_pixel_stats(img, d))
                if policy is not None and t % token_stride == 0:
                    tok["raw"].append(img)
                    tok["dec"].append(d)
                    tok["state"].append(st)
                    tok["instr"].append(ep["instr"])
        res: dict[str, Any] = {"pixels": _summary(pix)}
        if policy is not None and tok["raw"]:
            res["tokens_raw_vs_decoded"] = _token_stats(
                policy,
                torch.stack(tok["raw"]),
                torch.stack(tok["dec"]),
                torch.stack(tok["state"]),
                tok["instr"],
                batch,
            )
        out[task] = res
    return out


def print_gap(gap: dict[str, Any]) -> None:
    head = (
        f"{'task':<10} {'eps':>4} {'t0 |d|':>7} {'all |d|':>8} {'PSNR':>6} {'chg%':>6}"
        f" {'dq0':>8} {'tok=':>6} {'tok0=':>6} {'floor |d|':>9} {'floor tok=':>10}"
    )
    print(head)
    print("-" * len(head))
    for task, r in gap["tasks"].items():
        tk = r.get("tokens_stored_vs_fresh", {})
        fl = r.get("codec_floor", {})
        ft = fl.get("tokens_raw_vs_decoded", {})

        def f(x: Any, spec: str) -> str:
            return "-" if x is None else format(x, spec)

        print(
            f"{task:<10} {r['episodes']:>4} {f(r['pixels_t0']['mean_abs'], '7.3f')}"
            f" {f(r['pixels_all']['mean_abs'], '8.3f')} {f(r['pixels_all']['psnr_db'], '6.1f')}"
            f" {f(100 * r['pixels_all']['frac_pixels_changed'], '6.2f')}"
            f" {f(r['state_max_abs_diff_t0_rad'], '8.1e')}"
            f" {f(tk.get('token_agreement'), '6.3f')}"
            f" {f(tk.get('first_action_token_agreement'), '6.3f')}"
            f" {f(fl.get('pixels', {}).get('mean_abs'), '9.3f')}"
            f" {f(ft.get('token_agreement'), '10.3f')}"
        )


# ------------------------------------------------------------------------------------ main
def build_policy_for_gap(cfg: Config, args: argparse.Namespace, device: str) -> Any | None:
    if args.skip_tokens:
        return None
    from src.policy.openvla_oft import build_policy

    gap_cfg = cfg.get("render_gap", {})
    pcfg_path = (
        args.policy_config
        or gap_cfg.get("policy_config")
        or ("configs/sft_tiny.yaml" if cfg["preset"] == "tiny" else "configs/sft.yaml")
    )
    pcfg = load_config(cfg["preset"], REPO_ROOT / str(pcfg_path))
    adapter = args.adapter if args.adapter is not None else gap_cfg.get("adapter")
    if cfg["preset"] != "tiny" and adapter is None:
        raise SystemExit("render-gap needs the SFT adapter (--adapter or render_gap.adapter)")
    policy = build_policy(pcfg, device, adapter=adapter, is_trainable=False)
    policy.eval()
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    add_config_args(parser)
    parser.add_argument("--mode", choices=MODES, default=None, help="tiny: dry-run, else build")
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None, help="override cfg.device"
    )
    parser.add_argument("--push", action="store_true", help="build: push the dataset if it passes")
    parser.add_argument("--num-envs", type=int, default=None, help="build: parallel envs")
    parser.add_argument("--gap-dataset", default=None, help="render-gap: Hub id (default cfg)")
    parser.add_argument("--adapter", default=None, help="render-gap: SFT adapter dir or Hub path")
    parser.add_argument("--policy-config", default=None, help="render-gap: policy config file")
    parser.add_argument("--gap-episodes", type=int, default=None, help="render-gap: per task")
    parser.add_argument(
        "--token-stride", type=int, default=None, help="render-gap: every k-th frame"
    )
    parser.add_argument("--floor-episodes", type=int, default=2, help="render-gap: codec floor")
    parser.add_argument("--skip-tokens", action="store_true", help="render-gap: pixels only")
    args = parser.parse_args(argv)

    from src.data.recorder import dataset_location, push_dataset

    cfg_path = args.config or default_config(args.preset)
    cfg = load_config(args.preset, cfg_path)
    if args.device:
        cfg["device"] = args.device
    device = str(cfg.get("device", "cpu"))
    mode = args.mode or ("dry-run" if args.preset == "tiny" else "build")
    if mode == "dry-run" and args.preset != "tiny":
        raise SystemExit(
            "--mode dry-run is the CPU tiny check; use build / render-gap on a GPU box"
        )
    gap_cfg = cfg.get("render_gap", {})
    rc = 0
    with Run(work_order="WO-01r", script=__file__, cfg=cfg, device=device) as run:
        run.results["mode"] = mode
        if mode in ("build", "dry-run"):
            build_argv = ["--preset", args.preset, "--config", str(cfg_path)]
            if args.device:
                build_argv += ["--device", args.device]
            if args.num_envs:
                build_argv += ["--num-envs", str(args.num_envs)]
            build_rc = int(_build_env_module().main(build_argv))
            repo_id, root = dataset_location(cfg)
            report = tail_report(root, list(cfg["env"]["tasks"]))
            run.results.update(build_rc=build_rc, dataset=repo_id, tail=report)
            print(f"\ntail check ({repo_id}):")
            for task, r in report.items():
                print(f"  {task:<10} {json.dumps(r)}")
            problems = (
                [] if args.preset == "tiny" else tail_ok(report, int(cfg["data"]["demos_per_task"]))
            )
            run.results["tail_problems"] = problems
            if build_rc or problems:
                print(
                    "NOT pushed: " + "; ".join(problems or [f"build rc={build_rc}"]),
                    file=sys.stderr,
                )
                rc = 1
            elif args.push:
                run.results["hub_url"] = push_dataset(repo_id, root, private=True)
                print(f"pushed: {run.results['hub_url']}")
        if mode in ("render-gap", "dry-run") and rc == 0:
            if mode == "dry-run":
                repo_id, root = dataset_location(cfg)
            else:
                repo_id = args.gap_dataset or gap_cfg.get("dataset") or dataset_location(cfg)[0]
                root = dataset_location({**cfg, "data": {**cfg["data"], "hub_repo": repo_id}})[1]
                from src.data.oft_dataset import _ensure_local

                _ensure_local(repo_id, root)
            policy = build_policy_for_gap(cfg, args, device)
            episodes = args.gap_episodes or gap_cfg.get("episodes_per_task")
            stride = int(args.token_stride or gap_cfg.get("token_stride", 1))
            gap = render_gap(
                cfg,
                repo_id,
                root,
                policy,
                episodes_per_task=None if episodes is None else int(episodes),
                token_stride=stride,
                floor_episodes=args.floor_episodes,
                work_dir=run.dir,
            )
            gap["backend"] = {"device": device, "render": run.record["device"]}
            (run.dir / "render_gap.json").write_text(json.dumps(gap, indent=2))
            run.results["render_gap"] = gap
            print(f"\nrender gap: {repo_id} stored frames vs this backend ({device})")
            print_gap(gap)
            print(f"wrote {run.dir / 'render_gap.json'}")
    print(f"record: {run.record_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
