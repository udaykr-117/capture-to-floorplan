# Benchmark report

Regenerate everything with `bash scripts/reproduce.sh` (logs in `out/`, benchmark JSON in `bench/results/`).

## 0. What the benchmark set is, and what it is not
The brief asks for a self-built set (multi-room capture, a furnished room with staged damage, all rooms at all three tiers, one room twice, laser
or tape ground truth). We had no iPhone, so the set is the three Stray Scanner captures we were given, all of **one property**:
| Brief requirement | What we have | Gap |
|---|---|---|
| Multi-room capture, 3+ rooms + connector | `single_scan_with_ceiling` (6 rooms), `single_scan_floor_only` (6 rooms) | none |
| Furnished room with staged damage, 2 classes | none of the captures shows damage | **missing**: damage recall untested |
| Same rooms at all three tiers | LiDAR = the captures; video = each capture's own `rgb.mp4`; photo = stills chosen from those videos (`m5_make_photos.py`) | photo sets are simulated, not shot as photos |
| A room captured twice at the same tier | three overlapping LiDAR captures of the same rooms | none |
| Laser or tape ground truth | none | **missing**: every "accuracy" below is agreement between captures |
| Head-to-head vs a consumer app | not possible without an iPhone | **missing** |

## 1. Gates
| Gate | Tier | Number | Verdict |
|---|---|---|---|
| Opening widths <= 2 cm on >= 85%, misses and phantoms count | LiDAR | no ground truth. Openings found: 4 / 9 / 1 in the three captures (same property, so detection is not repeatable); 2 openings matched across captures, widths differ by a median 6.0 cm | **untested**, almost certainly fails (detection disagrees between captures) |
| Ceiling <= 1.5 cm per room; spread across captures <= 1 cm | LiDAR | measured in 6 of 6 with_ceiling rooms (intervals ±1 to ±5.5 cm); the other captures did not sweep the ceiling, so no spread can be computed | accuracy **untested**; spread **unmeasurable** (neither "repeatable-but-biased" nor "unrepeatable" can be stated) |
| Repeatability: same room, same tier, within 1 cm or 0.5% per wall | LiDAR | room dimensions 3/12 pass (paired 3/9, median 5.3 cm); same wall's length 2/34 pass, median 25-31 cm | **fails** |
| Drift accountability: correction + footprint ablation on/off | LiDAR | correction implemented; ablation in §3 | **met** as a requirement (correction shipped, ablation shown) |
| Photo-tier whole-property stitch, footprint ±8%, no overlaps | photo | no plan (0 of 8 and 0 of 31 simulated photos registered) | **fails** |
| Photo wall lengths ±8%, calibrated | photo | no plan | **fails** |
| Video wall lengths ±3% | video | no plan (4-18% of frames registered) | **fails** |
| Calibration at every tier | all | LiDAR: see §4; image tiers: no plans to calibrate | partial |

## 2. Repeatability table (LiDAR, plan output, after the fix; `out/final_bench_after.txt`)
| Capture pair | Rooms matched (IoU >= 0.5) | Room dimensions: n, median abs diff, pass | Same wall length: n, median abs diff, pass | Room area: n, median abs diff |
|---|---|---|---|---|
| floor_only vs with_ceiling | 6 | 7, 5.3 cm, 2/7 | 20, 31.5 cm, 0/20 | 6, 2.11 m2 |
| single_room vs with_ceiling | 2 | 1, 0.1 cm, 1/1 | 5, 62.7 cm, 1/5 | 2, 3.15 m2 |
| single_room vs floor_only | 2 | 4, 12.0 cm, 0/4 | 9, 18.1 cm, 1/9 | 2, 1.04 m2 |
| **pooled** | 10 | **12, 7.7 cm, 3/12 (25%)** | **34, 2/34 (6%)** | 10 |
Wall lengths disagree far more than room dimensions because the rooms are cut at different places in different captures (one wall in a capture
can span two rooms of another): this is room segmentation, not measurement noise, and is the main open problem of the LiDAR tier.

## 3. Drift ablation (stitched footprint, correction off vs on; `out/final_drift_ablation.txt`)
| Capture | Footprint off | Footprint on | Median plane spread off -> on | Start-end gap off -> on | Correction applied |
|---|---|---|---|---|---|
| with_ceiling | 71.7 ± 3.9 m2 | 72.7 ± 4.2 m2 | 1.92 -> 1.74 cm | 35.2 -> 34.5 cm | heading (max 1.2°) + floor (-7.0..+1.7 cm) |
| floor_only | 61.1 ± 2.5 m2 | 68.7 ± 3.5 m2 | 1.61 -> 1.29 cm | 18.2 -> 18.7 cm | heading (max 3.1°) + floor + 58.7 cm jump spread over 5199 frames |
| single_room | 27.1 ± 2.0 m2 | 25.9 ± 1.8 m2 | 1.67 -> 1.72 cm | n/a (not a loop) | heading (max 0.5°) + floor |
The two whole-property captures disagree on the footprint by 10.6 m2 (16%) with correction off and 4.0 m2 (6%) with it on. Cross-capture wall
agreement, floor_only on with_ceiling: 74.7% -> 81.3% of wall points within 5 cm. (Footprint intervals in this table are before the calibrated
room-extent term of §4, which widens them.)

## 4. Calibration (LiDAR)
Without ground truth, the test is: does the interval of the difference between two captures contain the difference (~95% expected)?
`scripts/calibrate_intervals.py`, `out/calibrate_intervals.txt`:
| Quantity | Before calibration | Extent term added | After, in-sample | After, leave-one-capture-pair-out |
|---|---|---|---|---|
| Wall length | 8/34 (24%) inside | ±0.54 m (robust 2σ) | 24/34 (71%) | 24/34 (71%) |
| Room area | 3/10 (30%) inside | ±25.5% of area (conformal 95%) | 10/10 | 9/10 |
| Opening width | 1/2 | none (2 pairs) | 1/2 | n/a |
Wall intervals still under-cover (71% vs 95%) by choice: covering the remaining walls needs ±3.4 m, set by one wall that spans two rooms in another
capture. The fitted terms come from the same three captures (no independent data), so they are a lower bound on the true uncertainty.
Benchmark re-run with both terms in the pipeline (`out/final_bench_after.txt`): walls 24/34 (71%) and room areas 10/10 inside, as fitted.

## 5. Head-to-head vs a consumer app
Not done: it needs an iPhone with LiDAR running the app on the same rooms, and we had none. No numbers are claimed.

## 6. Timing (laptop CPU, no GPU; `bench/results/*/benchmark.json`, `out/final_*`)
| Step | Time |
|---|---|
| LiDAR plan, single_room / floor_only / with_ceiling | 21-26 s / 84-90 s / 168-174 s |
| Damage pass (keyframe every 25 frames) | 98 s for 69 keyframes (single_room); ~1.0-1.6 s per keyframe |
| Video tier: frame extraction + SfM | 327 s for 650 frames (with_ceiling, 3 fps) |
| Video tier: depth model | ~1.4 s per registered frame (141 s for 25 frames) |
| Photo tier: SfM | 13 s for 31 photos |

## 7. Damage
| Capture | Detections at 0.10 (model-card threshold) | Regions at 0.10 | Regions at the shipped 0.33 |
|---|---|---|---|
| single_room | 65 | 1 | 0 |
| floor_only | 148 | 3 | 0 |
| with_ceiling | 356 | 5 | 0 |
All 11 boxes we looked at were false (seams, lamp, mat, glass panel). 0.33 is above every false box on these clean captures (negative control,
`out/m6_negative_control.txt`). Recall on real damage: **untested** (no damaged room in the set).
