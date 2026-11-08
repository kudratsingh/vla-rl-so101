# SO-101 — VLA Reinforcement Learning & Sim-to-Real

A **vision-language-action (VLA) reinforcement-learning and sim-to-real manipulation stack** for the **SO-101 robot arm**. It connects few-shot **OpenVLA-OFT LoRA supervised fine-tuning** to single-GPU **GRPO post-training**, using camera-conditioned policies and task-success rewards in **ManiSkill3**.

The pipeline spans scripted IK demonstrations, LeRobotDataset recording, grouped rollouts, action-token log-probability updates, checkpoint evaluation and robot-side command filtering. Protocol-hashed evaluations separate seen positions, shifted positions and unseen objects, while joint, workspace, velocity and deadman checks define the hardware control boundary.

## Latest results

Seen-position success under protocol `0.1-draft`, hash `e9ce5bb47a88`. Each entry pools **3 evaluation seeds × 100 episodes**. Each trained checkpoint comes from **one training seed**; these are not three independent training runs.

| Task | SFT only, 25 demos/task | Two-task GRPO, final checkpoint | Change |
|---|---:|---:|---:|
| PickPlace | 17.0% | 22.0% | +5.0 points |
| Push | 10.0% | 7.7% | −2.3 points |
| Stack, evaluation only during RL | 2.7% | 2.3% | −0.3 points |

A separate PickPlace-only RL run finished at **20.7%**, below its **27.0%** acceptance bar. Neither final PickPlace gain was distinguishable from SFT by the reported unpaired test. Two additional two-task training seeds stopped at their early gates; the intended three-training-seed comparison is incomplete.

The original target was a **20-point gain on at least two tasks across three training seeds**. It remains unmet. Shifted-position and unseen-object results, Wilson intervals, checkpoint identities, costs and negative findings are in [results.md](docs/results.md). The [status log](docs/STATUS.md) records the current hold and remaining decisions.

## What is implemented

- **Three simulated tasks:** PickPlace, Push and Stack, with explicit success checks, joint-position actions and camera inputs.
- **Demonstration collection:** scripted IK oracle, LeRobotDataset recording, and task/seed/cell metadata. The oracle's recorded seen-position check reached 100%, 99% and 100% over 100 episodes per task; those are controller results, not learned-policy scores.
- **Few-shot SFT:** OpenVLA-OFT LoRA, action-token/proprioception handling, checkpoint sweeps and full protocol evaluation.
- **GRPO:** grouped rollouts, binary task-success rewards, all-equal-group filtering, recomputed log probabilities, clipped policy updates and bounded runs. A toy CPU environment checks the learning loop separately from the VLA experiment.
- **Robot-side groundwork:** SO-101 units and driver glue, joint/workspace/velocity limits, tracking-error guards and a deadman. Hardware bring-up and policy deployment remain unverified.

## Run the local checks

Python **3.11** and [uv](https://docs.astral.sh/uv/). On Apple Silicon, install the separate rendering/video prerequisites described in the [environment guide](docs/ENVIRONMENT.md), including MoltenVK and FFmpeg 7.

```bash
make setup
make check
make smoke
```

`make check` runs lint, formatting and types; `make smoke` runs the test suite, with rendering-dependent tests skipped when the device is absent. `make tiny` exercises the CPU development pipeline, including a toy RL problem; it does not reproduce the trained VLA results.

GPU training has its own pinned Linux environment, checkpoint/data prerequisites and per-run approval. Follow the existing [environment guide](docs/ENVIRONMENT.md) and [work orders](docs/work-orders/) before reproducing an experiment. Datasets and model checkpoints are stored outside this repository, with some Hub artifacts private.

## Explore the implementation

| Area | Entry point |
|---|---|
| Tasks, oracle and demonstrations | [src/env](src/env/), [src/data](src/data/) |
| VLA wrappers and action representation | [src/policy](src/policy/) |
| GRPO math and training | [src/rl](src/rl/), [decision records](docs/adr/) |
| Frozen evaluation tasks and cells | [protocol.py](src/eval/protocol.py), [results](docs/results.md) |
| Robot driver and command filtering | [src/robot](src/robot/) |
| Project scope and current state | [Plan](docs/PLAN.md), [status](docs/STATUS.md), [agent guide](CLAUDE.md) |

Real-robot evaluation, sim-to-real improvement and the final SFT-versus-RL comparison remain future deliverables.
