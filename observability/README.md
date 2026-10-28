# SO-101 local experiment observability (first slice)

Generate an **offline, read-only dashboard** from the project's existing `runs/<id>/record.json` and `runs/<id>/metrics.jsonl` files:

```bash
uv run python -m observability.dashboard \
  --runs-root runs \
  --output runs/observability/index.html
# Open runs/observability/index.html in a browser.
```

This does **not** import the VLA policy, torch, ManiSkill, `src.robot`, any serial drivers or network clients. It does not connect to the arm, start a GPU, request RunPod, modify training, attach to another process or change safety behavior. It reads individual run logs (even when a write is in progress), skips incomplete JSONL lines, and only writes the destination HTML. By default that file is in `runs/observability/`, which `.gitignore` already excludes. **Never commit or share locally generated reports without review.**

The first dashboard contains a comparative SFT/GRPO metric explorer, read-only run inventory, and a *sim-only* two-run evaluation matrix using the existing `scripts/wo02_eval.py` per-task/per-cell/per-seed counts and 95% Wilson confidence intervals. The display does not infer that a chosen run is SFT or RL; select the correct checkpoints yourself. It never combines real-arm trials with sim trials, and shows missing observations as missing, not zero. It does not fetch the private Hugging Face Hub or RunPod artifacts; copy approved `metrics.jsonl` and `record.json` files to a local `runs/` directory if needed.

Only known numeric training metrics and aggregate simulated success counts are embedded. No frames, commands, full configs, servo serial numbers, robot addresses, raw camera footage, instructions or credentials are embedded. Read cap: 50 run folders, 32 MiB per metric log, 120 metric types and 2,400 plotted samples per run/metric. Larger logs are skipped, not modified. Recorded costs are estimates from the run metadata, **not verified provider billing**.

## Next independent PRs (not implemented here)

- Live experiment tracking forwarding to W&B, in offline mode by default and with no change to existing training gates.
- Rerun/LeRobot episode trace export for synchronized camera, joint targets and task geometry, using recorded datasets only.
- Safety/status panels for a *separate* real-arm deployment session after hardware is available, with the current local safety filters retaining exclusive authority.

These follow-ons require separate validation and should never be injected into an active paid training session.
