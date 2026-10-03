# Fix loop: before, after, and what fell short

The declaration (worst gate, hypothesis, fix, predicted number) is the one page in `docs/fix_declaration.md`; it was written before the fixed
pipeline was run. This document holds the measured before and after, the prediction against the actual numbers, and why the fix fell short of the
gate. A second, smaller fix (Fix B) followed on the same benchmark and is described the same way.

All numbers are agreement between the three captures of one property (there is no ground truth), pooled over the three pairs of captures.

## Regenerate, and read the diff

```
uv run python scripts/run_benchmark.py --variant before     # the pipeline before the fixes
uv run python scripts/run_benchmark.py --variant refined    # after Fix A
uv run python scripts/run_benchmark.py --variant final      # after Fix A and Fix B: the shipped pipeline
```
`before` switches Fix A and Fix B off; the opening fixes (section "Other changes") stay on in every variant. Re-running `final` reproduces the committed
results exactly (checked). Outputs: `bench/results/<variant>/`, logs `out/final_bench_<variant>.txt`. Two alternatives that were tested and not shipped are
`corridor_split` and `median_visits`. The code change of each fix is one commit; tags mark the states:

```
git diff fix-a-before fix-a-after -- src configs      # Fix A: room-local wall refinement
git diff fix-b-before fix-b-after -- src configs      # Fix B: keep corridors as rooms
```

## Before the fixes

| Measure | Value |
|---|---|
| Room widths (wall to opposite wall) within 1 cm or 0.5% | **1 of 10**, median difference 7.7 cm, 90th percentile 13.0 cm |
| Same wall's length within 1 cm or 0.5% | **2 of 34**, median difference 29.7 cm, 90th percentile 236 cm |
| Room area, median difference | 2.03 m2 (25%) |

## Fix A: room-local wall refinement

Hypothesis, evidence and prediction are in the declaration. Evidence tests (`scripts/m7_diagnose.py`, logs `out/m7_diagnose*.txt`):

| Hypothesis | Test | Result |
|---|---|---|
| Drift blur | position of one wall measured chunk by chunk (about 9 s each) | robust spread 3.4 / 3.6 / 3.8 cm, range about 14 cm: **supported** |
| Drift blur | widths measured only in chunks that see both walls, biggest pair | median difference 4.9 -> 2.6 cm, gate pass 29% -> 43%: **supported** |
| Plane-fit noise | correlation of the difference with the fit sigma | r = 0.12: rejected |
| Per-segment vs per-plane offset | offset of each segment from its own points | 0.2-0.4 cm from the plane: rejected |
| Skirting and furniture bias | densest-position estimator instead of the median | worse (20% vs 25%): rejected |
| Wrong wall correspondences | unambiguous pairs only | not better (21% vs 29%): rejected |

**Result.** The same 9 room widths measured before and after:

| Capture pair | Room, walls | Before | After |
|---|---|---|---|
| floor_only vs with_ceiling | R1 W0-W10 (1.1 m) | -10.5 cm | -0.4 cm |
| floor_only vs with_ceiling | R1 W10-W12 (3.0 m) | -2.6 cm | +0.6 cm |
| floor_only vs with_ceiling | R2 W1-W3 (3.0 m) | -12.7 cm | **-17.4 cm** |
| floor_only vs with_ceiling | R3 W3-W5 (2.9 m) | -6.1 cm | -5.3 cm |
| floor_only vs with_ceiling | R6 W1-W3 (3.1 m) | -12.2 cm | -11.3 cm |
| floor_only vs with_ceiling | R7 W1-W7 (3.9 m) | -15.1 cm | -8.5 cm |
| floor_only vs with_ceiling | R7 W2-W10 (3.4 m) | -4.0 cm | -1.8 cm |
| single_room vs with_ceiling | R2 W1-W3 (3.0 m) | -0.7 cm | +0.1 cm |
| single_room vs floor_only | R1 W4-W6 (2.9 m) | +9.3 cm | **+14.4 cm** |

7 of 9 widths better, 2 worse; median difference 9.3 -> 5.3 cm; within the gate 1/9 -> 3/9. Counting every comparable width (the set changes,
because refined walls match differently): 1/10 -> 3/12 within the gate, median 7.7 -> 7.7 cm, 90th percentile 13.0 -> 17.1 cm. The control (the
plane-pair metric, which the fix does not touch) is unchanged at 10/40 within the gate, median 4.8 cm, as predicted.

**Prediction against actual.**
- Median about 4 cm: 5.3 cm on the paired widths, 7.7 cm unpaired. Partly met.
- Pass rate 20-40%: 33% paired, 25% unpaired. Met.
- Footprints unchanged, overlaps of a few hundredths of a m2: **missed**. floor_only's footprint grew 2.7 m2 (4%) and rooms overlap by up to 0.14 m2
  where they share a wall.
- The gate still fails, as predicted.

**Why it fell short.** Refinement removes drift between visits, not within a visit. Both regressions involve the bottom-right room of floor_only (R1),
whose walls changed from chunks 0-3 alone (x width 4.595 -> 4.642 m); its only other visit is too short to compare. In with_ceiling R2, two visits of
the same room differ by up to 7 cm (3.037 vs 3.108 m, `scripts/refine_visits.py`), so one visit can carry its own error. That is the likely, not proven,
cause.

**Alternative tested: median over all visits** (`refine.visits: median`, variant `median_visits`). Adoption rule set before running: only if room
widths improve and no more get worse. Result: within the gate 3/12 -> 2/11, median 7.7 -> 10.1 cm. Rejected.

## Fix B: keep corridors as rooms

After Fix A the largest remaining failure is the same wall's length (2 of 34, median 29.7 cm).

**Hypothesis: rooms are cut at different places in different captures.** With the three captures registered on each other
(`out/seg_zoom_right.png`), a corridor about 0.9 m wide belongs to the top room in with_ceiling and to the bottom room in floor_only and
single_room, so both rooms' wall lengths change by 0.7-1.6 m. A room is seeded only where the floor is at least 0.5 m from every wall
(`rooms.seed_min_dist_m`); a narrow corridor never gets a seed and the flood gives it to whichever neighbour reaches it first. Rejected on the way:
merged partitions (wall-plane runs inside room interiors are stubs, at most 26% of the room's width, `scripts/seg_interior.py`) and unsupported
cuts (shared boundaries are 50-100% wall-plane backed, `scripts/seg_boundaries.py`).

**Fix.** Declared as: cut corridors out of rooms (`rooms.split_connectors`). On its own this had no effect, because the polygon step swallowed narrow
spaces (no grid line along their edges, below the minimum room area). Keeping narrow labelled spaces (`rooms.keep_narrow_rooms`) is what worked.

**Prediction (before running).** Same-wall median 29.7 cm -> 10-20 cm, within the gate 2/34 -> 3-6, room area median about 1 m2, one extra room in each capture.

**Result:**

| Variant | Same-wall length: median, 90th pct, within gate | Room area: median | Two whole-house footprints |
|---|---|---|---|
| Fix A only (`refined`) | 29.7 cm, 236 cm, 2/34 | 2.03 m2 (25%) | 72.7 vs 68.7 m2 (5.4% apart) |
| Fix A + keep narrow rooms (`final`, shipped) | **24.7 cm**, 236 cm, 2/34 | **1.04 m2 (15%)** | 71.7 vs 68.7 m2 (4.1% apart) |
| Fix A + keep narrow rooms + corridor split (`corridor_split`) | 41.8 cm, 225 cm, 1/45 | 1.03 m2 (22%) | 68.4 vs 68.2 m2 (0.3% apart) |

The corridor split fires in all three captures, but it also cut a notch off one of with_ceiling's rooms that no other capture has, which made wall
lengths worse, so it ships off. The choice between the last two rows was made after seeing these numbers.

**Prediction against actual.** Same-wall median 10-20 cm: 24.7 cm (**missed**, right direction). Within the gate 3-6 of 34: 2 (**missed**). Room
area about 1 m2: 1.04 m2 (met). One extra room in each capture: only with_ceiling gained one (**missed** for the other two).

**Why it fell short.** In floor_only and single_room the corridor is still inside the bottom room. Cutting it out reliably needs a corridor test that
does not also cut notches; none was found.

## Other changes measured with the same benchmark

- **Openings:** a door seen on both faces of one wall was reported twice, and a see-through with the same room on both sides (a half wall) was
  reported as an opening. Both are removed: openings found in the three captures 4 / 9 / 1 -> 3 / 7 / 1. A stricter see-through distance (50 cm) was
  tried and reverted: it did not remove the 1.9 m doorway and turned real doors into raised openings.
- **Interval calibration:** the intervals contained the difference between captures for 8/34 walls and 3/10 room areas. An extent term fitted on the
  capture pairs raises this to 27/34 and 10/10 (`scripts/calibrate_intervals.py`; `docs/benchmark_report.md` section 4).
