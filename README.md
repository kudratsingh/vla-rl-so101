# SO-101 Robot Learning with SmolVLA and ACT

**PyTorch · LeRobot · SmolVLA · ACT · Apple Silicon MPS · ManiSkill3**

A robot-learning stack for the **SO-101 arm**, connecting leader-arm demonstrations,
single-GPU fine-tuning, local policy inference and reproducible evaluation. The current focus is
**SmolVLA with an ACT baseline** on real PickPlace tasks, using front and wrist cameras and a
shared evaluation protocol.

The engineering runs from model training to the servo bus: **command filtering, fault shutdown,
checkpoint identity, recoverable recordings and protocol-hashed results**. The earlier
**OpenVLA-OFT, LoRA and GRPO** simulation work remains the experimental foundation, with further
work deferred until after full evaluation of the real arm and the SmolVLA/ACT baseline.

## Engineering highlights

- **Local inference and matched baselines.** SmolVLA and ACT share demonstrations, action conventions
  and evaluation schedules. The implementation under review targets Mac MPS inference and
  fine-tuning on a single 24 GB NVIDIA GPU.
- **Robot-side command safety.** Joint, workspace, velocity and tracking-error limits, deadman
  checks and servo monitoring guard commands. Stop checks run immediately before motion writes;
  fault responses include controlled shutdown and torque-off.
- **Reproducible evaluation.** Seen positions, shifted positions and unseen objects use versioned,
  hashed protocols, with checkpoint provenance and explicit rules for started and interrupted trials.
- **Recoverable data.** The implementation under review records both cameras, state, actions and
  timing, recovers interrupted captures and exports evaluation bundles with trial identity and
  label provenance.
- **Measured simulation foundation.** ManiSkill3 tasks, scripted IK demonstrations, OpenVLA-OFT
  LoRA fine-tuning and GRPO experiments retain their configs, results and negative findings.

**Current stage:** hardware bring-up and physical placement checks. The
[SmolVLA/ACT implementation is under review in PR #32](https://github.com/kudratsingh/vla-rl-so101/pull/32),
with CPU training/reload checks and Mac MPS mock benchmarks. Real-arm policy success remains to
be measured; the placement protocol awaits physical measurements before it is frozen.

## Current evaluation plan

| Area | Plan |
|---|---|
| Real data | 50 PickPlace demonstrations, leader-arm teleop, front and wrist cameras |
| Policies | SmolVLA and ACT trained on the same demonstration set |
| Compute | Single-GPU fine-tuning; local Mac inference over MPS |
| Evaluation | 50 binary trials per cell: seen positions, shifted positions and unseen objects |
| Controls | Shared seeded schedules, fixed rubric, protocol hashes and checkpoint identity |
| Next experiment | Reward-filtered behaviour cloning in two-camera simulation; transfer tested only after a measured sim gain |

The [plan](docs/PLAN.md), [bring-up checklist](docs/bringup/WO-06-checklist.md) and
[status log](docs/STATUS.md) track implementation and verification.

## OpenVLA research and retained results

**OpenVLA-OFT and GRPO are deferred follow-ups after the full real-arm and SmolVLA/ACT
evaluation.** The simulation environments, LoRA training, grouped rollouts and evaluation tooling
remain available for that work. The [pivot decision](docs/adr/0003-pivot-to-smolvla.md) records
the sequencing and retained foundation.

Seen-position success under protocol `0.1-draft`, hash `e9ce5bb47a88`. Each entry pools
**3 evaluation seeds × 100 episodes**. Each trained checkpoint comes from **one training seed**;
these are not three independent training runs.

| Task | SFT only, 25 demos/task | Two-task GRPO, final checkpoint | Change |
|---|---:|---:|---:|
| PickPlace | 17.0% | 22.0% | +5.0 points |
| Push | 10.0% | 7.7% | −2.3 points |
| Stack, evaluation only during RL | 2.7% | 2.3% | −0.3 points |

A separate PickPlace-only RL run finished at **20.7%**, below its **27.0%** acceptance bar.
Neither final PickPlace gain was distinguishable from SFT by the reported unpaired test. Two
additional two-task training seeds stopped at their early gates; the intended three-training-seed
comparison is incomplete.

The original target of a **20-point gain on at least two tasks across three training seeds**
remains unmet. [Full results](docs/results.md) retain shifted-position and unseen-object scores,
Wilson intervals, checkpoint identities, costs and failure analysis. Future OpenVLA experiments
will build on this evidence and the completed real-arm baseline.

## Run the local checks

Python **3.11** and [uv](https://docs.astral.sh/uv/). On Apple Silicon, install the separate
rendering/video prerequisites in the [environment guide](docs/ENVIRONMENT.md), including MoltenVK
and FFmpeg 7.

```bash
make setup
make check
make smoke
```

`make check` runs lint, formatting and types; `make smoke` runs the test suite, with
rendering-dependent tests skipped when the device is absent. `make tiny` exercises the CPU
development pipeline, including a toy RL problem; it does not reproduce the trained VLA results.

GPU training has its own pinned Linux environment, checkpoint/data prerequisites and per-run
approval. Follow the [environment guide](docs/ENVIRONMENT.md) and [work orders](docs/work-orders/)
before reproducing an experiment. Datasets and model checkpoints are stored outside this
repository, with some Hub artifacts private.

## Explore the implementation

| Area | Entry point |
|---|---|
| SO-101 driver, command filtering and bring-up | [src/robot](src/robot/), [bring-up docs](docs/bringup/) |
| SmolVLA, ACT, local inference and real-data capture | [PR #32](https://github.com/kudratsingh/vla-rl-so101/pull/32), under review |
| Simulation, oracle and demonstrations | [src/env](src/env/), [src/data](src/data/) |
| Retained OpenVLA and GRPO implementation | [src/policy](src/policy/), [src/rl](src/rl/) |
| Evaluation and experimental record | [src/eval](src/eval/), [results](docs/results.md), [decision records](docs/adr/) |
| Roadmap and current state | [Plan](docs/PLAN.md), [status](docs/STATUS.md), [agent guide](CLAUDE.md) |
