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
