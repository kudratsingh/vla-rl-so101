# WO-09 — Final writeup and release

Branch: `wo-09-release`. Tier: none.

## Deliverables

1. README: claim, sim table, real table, cost table, hardware list, reproduce commands per
   number, limitations, what did not work.
2. Model card for the released LoRA weights (base model, data, protocol hash, intended use,
   known failure modes).
3. `docs/adr/README.md` index; all ADRs at *accepted* or *superseded*.
4. `docs/results.md` complete; every number has a run id and protocol hash.
5. Blog post final (`docs/blog.md`), with the ten-trial video embedded.
6. License decision recorded (owner's call; vendored code licenses listed in `THIRD_PARTY.md`).

## Benchmark

A clean clone reproduces the `tiny` preset and the eval table generation from published run
artifacts; README numbers match results.md; STATUS.md shows final spend.
