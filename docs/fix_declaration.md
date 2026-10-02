# Fix declaration (M7)

Written BEFORE running the fixed pipeline. Numbers regenerate with `uv run python scripts/run_benchmark.py --variant before|after`.

## Worst gate and its failing number
Repeatability: two captures of the same property must agree within 1 cm or 0.5% per wall.
On the plan output (room dimensions = distance between two opposite plane-backed walls of a room, rooms matched across captures by IoU >= 0.5),
pooled over the three capture pairs: **1 of 10 dimensions pass (10%), median disagreement 7.7 cm, 90th percentile 13.0 cm**
(`bench/results/before/benchmark.json`). The older all-plane-pairs metric (M3): 10 of 40 (25%), median 4.8 cm.

## Root-cause hypothesis and evidence
Hypothesis: **drift blur of wall positions.** A wall plane is fitted to all of its points from the whole capture; the same wall seen at different
times sits at different positions (pose drift), so the plane is an average, and a room dimension inherits the drift between the moments its two
walls were seen.
Evidence (`scripts/m7_diagnose.py`, `out/m7_diagnose*.txt`):
- H1: the position of one wall measured chunk by chunk (~9 s chunks) has a robust spread of 3.4 / 3.6 / 3.8 cm and a range of ~14 cm (median over
  walls) in the three captures. (A first version of this test searched only +-3 cm around the plane and wrongly showed ~1 cm.)
- H3: on the largest capture pair (floor_only vs with_ceiling, 14 width pairs seen together in both), widths measured only from chunks that see BOTH
  walls agree to a median 2.6 cm vs 4.9 cm from the whole-capture planes, gate pass 43% vs 29%. The two small pairs have 1 and 3 such pairs (no evidence either way).
- Rejected: plane-fit noise (H2, correlation of error with fit sigma r = 0.12), per-run offset vs whole-plane offset (H4, runs sit 0.2-0.4 cm from their
  plane), skirting/furniture mixing via a mode estimator (H5, worse), ambiguous matches (H6, unambiguous pairs are not better).
- Pictures of the worst pairs (`out/m7_look_worst.png`): whole walls of floor_only displaced 10-20 cm from with_ceiling, and doubled walls inside with_ceiling.

## Fix
Room-local wall refinement (`src/floorplan/rooms/refine.py`, switch `refine.enabled`): each room's plane-backed walls are re-measured (densest 1 cm
position within +-15 cm of the global plane) from the room's longest single visit (consecutive chunks with the cameras inside the room), so the
room's walls come from the same tens of seconds. No threshold was tuned on the benchmark; values are copied from the diagnosis and config.

## Prediction (made before the run)
- Room dimensions, pooled: median disagreement from 7.7 cm to **about 4 cm**; pass rate from 10% to **20-40%**. The gate (1 cm / 0.5%) will
  still FAIL: H3 shows ~2.6 cm median even for co-visible widths.
- M3 plane-pair metric: **exactly unchanged** (25%, 4.8 cm), because the fix does not touch the wall planes. This is a control: if it moves, the
  fix leaks somewhere it should not.
- Footprints and room count unchanged; small room overlaps (<= a few hundredths of a m2) may appear where rooms share a wall.
- Risk: only 10 room dimensions are comparable, so one or two dimensions change the pass rate by 10-20 points.

## Result (after the run)
`bench/results/after/benchmark.json`, `out/m7_bench_before.txt`, `out/m7_bench_after.txt`.

Paired comparison (the same room dimension compared in both runs, 9 of them):
| capture pair | room, walls | before | after |
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
- 7 of 9 better, 2 worse. Median |d| 9.3 -> 5.3 cm. Gate pass 1/9 -> 3/9.
- Unpaired, as the benchmark prints it (the set changes because refined walls match differently): 1/10 -> 3/12 pass, median 7.7 -> 7.7 cm,
  90th percentile 13.0 -> **17.1 cm (worse)**. Three of the new dimensions (single_room vs floor_only) disagree by 6.9-17.7 cm.
- Control: the M3 plane-pair metric is unchanged (10/40, 4.8 cm), as predicted.
- Cost: rooms now overlap by up to 0.10 / 0.11 / 0.14 m2 where they share a wall (was 0); footprints +0.2 / +2.7 / +0.4 m2.

## Prediction vs actual
- Predicted median about 4 cm: actual 5.3 cm paired, 7.7 cm unpaired (unchanged). Partly met.
- Predicted pass 20-40%: actual 33% paired, 25% unpaired. Met. The gate still FAILS, as predicted.
- Not predicted: the single_room vs floor_only dimensions get worse. That area is where floor_only is rotated +2.27 deg and shifted 26 cm locally
  (M3). Refinement uses one visit; if the visit itself is rotated (heading drift within the visit, or a rotated frame there), it cannot fix that, and
  dropping the averaging over other visits makes those walls worse.
- Predicted footprints unchanged and overlaps of a few hundredths of a m2: MISSED. floor_only's footprint grew 2.7 m2 (66.0 -> 68.7, 4%) and overlaps
  reach 0.14 m2. The two big captures' footprints are now 72.7 vs 68.7 m2 (6% apart; 9% before).
- Decision: refinement is ON by default (paired evidence, 7 of 9 better), with the regression and overlaps stated in the plan limitations and the report.

---

# Fix declaration, round 2 (written BEFORE running the fixed pipeline)
Regenerate: `uv run python scripts/run_benchmark.py --variant after` (before = round-1 fix) and `--variant after2` (round 2 on top).

## Worst gate and its failing number
Still repeatability, now measured on the quantity the walk-in laser checks, the length of the same wall in two captures:
**2 of 34 walls within 1 cm or 0.5%, median disagreement 29.7 cm, 90th percentile 236 cm**; room areas differ by a median 2.03 m2 (25%).

## Root-cause hypothesis and evidence
Rooms are cut at different places in different captures. Picture `out/seg_zoom_right.png` (all three captures registered): a ~0.9 m wide
corridor (x 5.4-6.3 m, z 3-4.8 m) between the two right-hand rooms is part of the TOP room in with_ceiling and of the BOTTOM room in floor_only
and single_room, so both rooms' walls change length by 0.7-1.6 m. Cause in the code: a room is seeded only where floor is >= 0.5 m from every
wall (`rooms.seed_min_dist_m`, chosen so doorways do not become rooms); a corridor narrower than 1 m never gets a seed and the flood gives it to
whichever neighbour reaches it first. Checked and rejected: merged partitions (wall-plane runs crossing room interiors are stubs, at most 26% of
the room's width, `scripts/seg_interior.py`); unsupported cuts (shared boundaries are 50-100% wall-plane backed, `scripts/seg_boundaries.py`).

## Fix
`rooms.split_connectors`: inside each room, the part narrower than 1.2 m, at least 1.5 m long and touching another room becomes its own room
(a connector). Doorways (one wall thick) fail the length test; dead-end alcoves fail the touching test (synthetic test `tests/test_connectors.py`).
Thresholds are from building dimensions (corridor width, door depth), not tuned on the benchmark.

## Prediction
- Same-wall lengths: median disagreement from 29.7 cm to **10-20 cm**; walls within the gate from 2/34 to **3-6** (still FAILS: the plane
  positions themselves differ by ~5 cm, as round 1 showed).
- Room areas: median disagreement from 2.03 m2 to **about 1 m2**.
- One more room in each capture (the corridor); room dimensions and the M3 plane-pair metric about unchanged.
- Risk: other narrow places (closets, the strip beside with_ceiling R1) may split in one capture and not in another, which would make some
  walls worse.

## Round 2 result (after the run)
`out/final_bench_after.txt` (before), `out/final_bench_after2.txt`, `out/final_bench_after2a.txt`; regenerate with `--variant after|after2|after2a`.

What happened, in order:
1. First run of the declared fix: **it did not fire**: room counts unchanged. Debugging (`scripts/seg_debug.py`, `seg_debug2.py`) showed that in
   with_ceiling the corridor already had its own label, but the polygon step gave its cells to the neighbouring room's larger rectangle (no grid
   line along the corridor's edge) and the remainder fell under the 1.5 m2 room minimum. Fix inside the fix: narrow labelled spaces get grid lines
   along their own edges and the 1.0 m2 connector minimum (`rooms.keep_narrow_rooms`).
2. With that, the corridor survived in with_ceiling, but the split still did not fire in floor_only and single_room (their corridor sits inside the
   bottom room and its "touches another room" test looked only 15 cm out, less than a wall). Widening it to the wall-thickness distance already used
   for adjacency (35 cm) made the split fire everywhere, but it also cut a notch off with_ceiling's top-left room that no other capture has.
3. The two parts measured separately (pooled over the three capture pairs):

| Variant | Same-wall length: median, p90, gate | Room area: median | Two whole-property footprints |
|---|---|---|---|
| before (round-1 fix only) | 29.7 cm, 236 cm, 2/34 | 2.03 m2 (25%) | 72.7 vs 68.7 m2 (5.4% apart) |
| (a) keep narrow rooms | **24.7 cm**, 236 cm, 2/34 | **1.04 m2 (15%)** | 71.7 vs 68.7 m2 (4.1%) |
| (a) + (b) split corridors | 41.8 cm, 225 cm, 1/45 | 1.03 m2 (22%) | 68.4 vs 68.2 m2 (0.3%) |

Shipped: **(a) on, (b) off** (`rooms.keep_narrow_rooms: true`, `rooms.split_connectors: false`), chosen on the declared metric (same-wall length).
Choosing between (a) and (a)+(b) after seeing these numbers is a decision informed by the benchmark; it is logged here for that reason.

## Round 2: prediction vs actual
- Same-wall length median: predicted 10-20 cm, actual 24.7 cm. **Missed** (moved in the right direction, less than predicted).
- Walls within the gate: predicted 3-6 of 34, actual 2 of 34. **Missed**; the gate still fails.
- Room-area median: predicted about 1 m2, actual 1.04 m2. **Met.**
- "One more room in each capture": only with_ceiling gained one (the corridor). Missed for the other two captures, because the corridor split (b) is off.
- Why it fell short: the corridor is now one room in with_ceiling but still inside the bottom room in floor_only and single_room, so those walls
  still differ; cutting it out reliably needs a corridor test that does not also cut notches (not found in the time available).
