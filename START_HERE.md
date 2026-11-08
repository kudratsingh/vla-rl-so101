# Start here (owner)

1. Create the GitHub repo (private until WO-05), push this kit, protect `main` (PR required).
2. Order hardware this week: SO-101 follower (pair if you want teleop/HIL-SERL later), 2 USB
   webcams, table clamps, 3 cm cubes + bowl. Lead time 2–4 weeks.
3. Point the coding agent at `CLAUDE.md` with: "Start WO-01."
4. You launch every paid run. The agent gives you config hash + cost; you run it on RunPod
   (dev = RTX 4090, rl = A100-80GB) and paste the run id back.
5. Review only WO-03 and WO-04 PRs. Everything else: merge when the benchmark line is met.
6. See the [compute cap](docs/PLAN.md#budget) and [spend ledger](docs/STATUS.md#spend).

Kill switches:
- A flat RL run at 25% of steps → stop it, don't let it finish.
- SFT below 20% after 3 tries → switch base model (ADR-0001), don't start RL.
- Two gap-closing retrains without improvement (WO-08) → stop, write it up as a finding.
