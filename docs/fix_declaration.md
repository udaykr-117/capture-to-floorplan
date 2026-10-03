# Fix declaration

Written before the fixed pipeline was run. Measured before and after, and the prediction against the actual numbers: `docs/fix_loop.md`.

## 1. The worst-performing gate and its failing number

**Repeatability**: two captures of the same rooms must agree within 1 cm or 0.5% per wall. Measured on the plan output, pooled over the three pairs
of captures of our one property (a room's width is the distance between two opposite wall planes; rooms matched across captures by overlap):

> **1 of 10 room widths within the gate; median difference 7.7 cm; 90th percentile 13.0 cm.**

(The photo and video tiers fail harder, but they produce no plan on the samples, so there is no number to move; they are reported in
`docs/benchmark_report.md` section 1.)

## 2. Root-cause hypothesis and evidence

**Hypothesis: drift blur of wall positions.** A wall plane is fitted to every point of the whole capture. The same wall seen at different times sits
at different positions because of pose drift, so the fitted plane is an average, and a room width inherits the drift between the moments its two
walls were seen.

**Evidence** (`scripts/m7_diagnose.py`):
- One wall's position measured chunk by chunk (about 9 s each) has a robust spread of 3.4 / 3.6 / 3.8 cm and a range of about 14 cm in the three captures.
- Widths measured only in chunks that see both walls agree better across captures than widths from whole-capture planes: median 2.6 cm vs 4.9 cm on
  the biggest pair (gate pass 43% vs 29%).
- Four other causes were tested and rejected: plane-fit noise, per-segment offsets, skirting and furniture bias, wrong wall correspondences.

## 3. The fix and the predicted number

**Fix: room-local wall refinement.** Re-measure each room's walls from that room's longest single visit (consecutive chunks with the cameras inside
the room), so the walls of one room come from the same tens of seconds (`src/floorplan/rooms/refine.py`, config `refine.*`).

**Predicted after the fix:** room widths, median difference **about 4 cm** (from 7.7 cm), **20-40%** within the gate (from 10%). The gate will
still fail. Control: the plane-pair agreement (10 of 40, median 4.8 cm), which the fix does not touch, stays exactly unchanged.
