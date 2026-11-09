# WO-09 — Writeup

Branch: `wo-09-release`. Tier: none. Rewritten 2026-11-09 for the SmolVLA pivot (ADR-0003).

## Done when

README table, model cards, reproduce commands, ADR index, results.md with every number and
protocol hash.

## Deliverables

1. README: claim, the real table (SmolVLA vs ACT, seen / shifted / unseen objects, unseen per
   object and pooled), the WO-08 sim table (control vs filtered BC), cost table, hardware list,
   reproduce commands per number, limitations, what did not work (the OFT / GRPO partial result
   included, labelled as such).
2. Model cards for the released SmolVLA and ACT checkpoints: base model, data (demo count,
   positions, cameras), protocol hash, intended use, known failure modes.
3. `docs/adr/README.md` index; all ADRs at *accepted* or *superseded*.
4. `docs/results.md` complete; every number has a run id and protocol hash; sim and real never in
   one table; single runs labelled.
5. License decision recorded (owner's call; vendored code licenses listed in `THIRD_PARTY.md`).

## Benchmark

A clean clone reproduces the `tiny` presets and the table generation from published run
artifacts; README numbers match results.md; STATUS.md shows final spend.
