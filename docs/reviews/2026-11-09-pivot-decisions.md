# Owner decisions on top of the SmolVLA pivot handoff (2026-11-09)

The spec is `docs/reviews/2026-11-09-smolvla-pivot-handoff.md` (the "final2" version). This file
records the owner's decisions made in the same session that refine it. Where they differ, this
file wins.

## D1. Seen cell and demo layout: five positions x 10 (owner choice, supersedes an earlier note)

Earlier in the session the owner briefly asked for all 16 inner slots with 3-4 demos per slot.
After the final2 handoff the owner chose the paper's collection recipe instead (SmolVLA,
arXiv 2506.01844 §4.1): five positions, ten demonstrations each, so SmolVLA and ACT both get ten
demonstrations at every training position. The 16-slot version is withdrawn.

- **Choosing the five slots.** They come from the existing 16-slot inner block of
  `src/eval/real_grid.py` (the grid itself is unchanged). Spread them across the block: one near
  each corner and one in the middle, so the shifted ring is an extrapolation in every direction.
  The 4 x 4 block has no single centre slot, so one of the four central slots is chosen and the
  reason recorded.
- **Freezing the seen cell.** The five slot ids are frozen as the seen cell.
- **Demos.** 50 PickPlace demos, 10 at each of the five slots, with both cameras. The other 11
  inner slots and the 20-slot shifted ring are never used by demos or camera-match frames.

## D2. Bowl fixed

The bowl stays at one fixed position across all demos and all three eval cells. Shifted then tests
object position, and unseen tests object identity, with everything else held constant. The bowl
position must satisfy the scene's placement constraints for every object slot used: the five seen
slots and the 20 ring slots. Those constraints are MIN_SEP, the base keep-out and reach. The bowl
position is frozen with the slots.

## D3. Eval cells, 50 trials each

| Cell | Trials |
|---|---|
| Seen | 10 trials at each of the five selected slots |
| Shifted | 50 trials balanced across the 20 outer-ring slots (2 or 3 per slot, fixed in the schedule) |
| Unseen objects | 50 trials at the same five seen slots: 25 with the purple cube and 25 with the green cube, five per slot per object |

- Report the two unseen objects separately and pooled. The pooled 50 is the headline.
- The green cube and purple cube replace the green cylinder for the real arm. This is a protocol
  change, so the real evaluation gets its own version and hash. The sim protocols 0.1-draft and
  0.2 stay byte-identical.

## D4. Freeze before recording

Before the first demo is recorded, the new real protocol version freezes and hashes:
- the five slot ids
- the bowl position
- the seeded trial schedule (both cell membership and trial order)

The same schedule is used for both policies. Grid coordinates in every record are planned poses.
Scoring is binary success (CLAUDE.md rule 4). The paper's partial-credit score may be logged as a
secondary column, never as the headline.

## D5. Development trials and the 100-demo decision (as in final2)

Start at 50 demos. Going to 100 is conditional on a failure analysis of development trials that
shows more data addresses the observed failures. Development trials use their own small schedule,
disjoint from the frozen evaluation schedule, and are never reported as results. The 50-per-cell
evaluation is reserved for selected checkpoints.

## D6. Mock export against the real importer

Before the 50 demos, a mock demo export and a mock evaluation export are run through vishwaroop's
own seal and import tools. The run uses a scratch copy of that repo; the original directory is
read-only and never written to. Label provenance follows final2:
- `blinded_review` is declared only when a reviewer scored the recorded clips with the policy
  identity hidden.
- Otherwise the labels declaration is omitted rather than misdeclared.
