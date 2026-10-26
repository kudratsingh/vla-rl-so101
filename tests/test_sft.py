"""scripts/wo02_sft.py end to end on CPU (tiny random OFT, synthetic frames), plus the settings.

`src.runs` lands with WO-01; until then a stand-in Run (same interface) is injected."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar, Self

import pytest
import torch

from src.config import load_config
from src.policy.openvla_oft import build_policy
from src.policy.sft import SFTSettings, estimate_cost, make_optimizer, resolve_micro_batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


class FakeRun:
    instances: ClassVar[list[FakeRun]] = []

    def __init__(
        self,
        *,
        work_order: str,
        script: str,
        cfg: dict[str, Any],
        device: str,
        estimated_usd: float = 0.0,
        runs_dir: Path | None = None,
    ) -> None:
        self.id = f"test-{len(FakeRun.instances)}"
        self.dir = FakeRun.base / self.id
        self.dir.mkdir(parents=True)
        self.results: dict[str, Any] = {}
        self.metrics: list[dict[str, Any]] = []
        self.work_order, self.estimated_usd = work_order, estimated_usd
        FakeRun.instances.append(self)

    base: ClassVar[Path] = Path(".")

    def log_metrics(self, step: int | None = None, **m: Any) -> None:
        self.metrics.append({"step": step, **m})

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture()
def fake_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[FakeRun]:
    FakeRun.base, FakeRun.instances = tmp_path, []
    mod = types.ModuleType("src.runs")
    mod.Run = FakeRun  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.runs", mod)
    return FakeRun


def test_settings_auto_micro_batch_and_schedule() -> None:
    cfg = load_config("sim", "configs/sft.yaml")
    assert resolve_micro_batch(cfg["sft"], 8, "cpu") == (8, False)
    s = SFTSettings.from_cfg(cfg, "cpu")
    assert (s.lr, s.lr_milestones, s.warmup_steps, s.steps) == (5e-5, (), 100, 2000)
    assert resolve_micro_batch(
        {"micro_batch_size": 4, "grad_checkpointing": "auto"}, 8, "cuda"
    ) == (4, True)
    p = torch.nn.Parameter(torch.zeros(1))
    opt, sched = make_optimizer([p], SFTSettings(**{**s.__dict__, "warmup_steps": 2, "steps": 6}))
    lrs = []
    for _ in range(6):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    # linear warmup from 0.1 * lr over 2 steps (0.1 + 0.9 * 1/2 = 0.55 after one), then constant:
    # no decay milestones inside the 2000-step budget
    assert lrs[0] == pytest.approx(5e-5 * 0.55) and lrs[1:] == pytest.approx([5e-5] * 5)
    est = estimate_cost(cfg)
    assert est["hours"] == pytest.approx(2000 * 1.8 / 3600 + 0.5, abs=1e-3)
    assert est["usd"] == pytest.approx(est["hours"] * 0.89, abs=0.01)


def test_wo02_sft_tiny_end_to_end(fake_runs: type[FakeRun]) -> None:
    import wo02_sft

    out = wo02_sft.main(["--preset", "tiny", "--synthetic"])
    run = fake_runs.instances[-1]
    assert out["steps_run"] == 20 and run.work_order == "WO-02"
    ckpts = sorted((run.dir / "checkpoints").iterdir())
    assert [c.name for c in ckpts] == ["step_000010", "step_000020"]
    assert (ckpts[-1] / "adapter_model.safetensors").is_file()
    assert (ckpts[-1] / "norm_stats.json").is_file() and (ckpts[-1] / "policy.json").is_file()
    losses = [m["loss"] for m in run.metrics]
    assert losses[-1] < losses[0]
    assert out["demo_coverage"]["frac_outside"] == 0.0
    # the checkpoint loads back into a policy that acts
    cfg = load_config("tiny", "configs/sft_tiny.yaml")
    pol = build_policy(cfg, "cpu", adapter=ckpts[-1])
    chunk = pol.act(
        {
            "image": torch.zeros(64, 64, 3, dtype=torch.uint8).numpy(),
            "state": torch.zeros(6).numpy(),
        },
        "push the red cube to the target",
    )
    assert chunk.shape == (8, 6)


def test_wo02_sft_probe_saves_nothing(fake_runs: type[FakeRun]) -> None:
    import wo02_sft

    out = wo02_sft.main(["--preset", "tiny", "--synthetic", "--probe-steps", "6"])
    run = fake_runs.instances[-1]
    assert out["steps_run"] == 6 and run.work_order == "WO-02-probe"
    assert not (run.dir / "checkpoints").exists()
    assert out["s_per_step_median"] > 0 and out["extrapolated"]["hours"] > 0
