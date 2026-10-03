# Compliance matrix

Requirement (from the brief) -> file path -> artifact -> status. Status words: **met** (built and evidence printed), **partial**, **fails** (built,
measured, does not reach the bar), **untested** (built, no data to measure), **missing** (not built / not possible here).

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1 | Capture route: one-page stock protocol (Route 2) | `docs/capture_protocol.md` | Stray Scanner (LiDAR), Camera app (video, photos); walk, length, avoid, hand-over | met (not yet followed by a non-engineer on a real phone) |
| 2 | Device matrix: tier x hardware x honest accuracy | `docs/device_matrix.md` | table with measured numbers and "no plan" where it fails | met |
| 3 | Photo tier: 2-8 stills per room, per-room folders, same stitched plan | `src/floorplan/tiers/images.py` (`run_photos`), `src/floorplan/sfm.py` | one SfM over all folders, doorway shots link rooms; HEIC read | partial: runs; **no plan** on simulated photo sets (0 registered) |
| 4 | Video tier: hand-held walkthrough | `src/floorplan/tiers/images.py` (`run_video`), `src/floorplan/depth.py`, `src/floorplan/scale.py` | SfM + monocular depth + scale with uncertainty; optional learned matcher (off) | partial: runs; **no plan** on the samples (4-18% of frames registered) |
| 5 | LiDAR tier: depth, poses, intrinsics | `src/floorplan/io/stray.py`, `src/floorplan/pipeline.py` | full plan on all three captures | met |
| 6 | Same output contract from every tier, intervals widen as data thins | `src/floorplan/schema.py`, `src/floorplan/intervals.py` | one `Plan` schema; image tiers add a >= 20% scale term | met (contract); image tiers return no rooms |
| 7 | Per-room walls, ceiling height, floor area, openings | `src/floorplan/pipeline.py`, `rooms/ceiling.py`, `rooms/openings.py` | `plan.json` rooms[].walls/area/ceiling_height, openings[] | met (LiDAR); accuracy untested |
| 8 | Stitched multi-room plan with correct adjacency | `src/floorplan/pipeline.py` (stitched, adjacency) | `plan.json` stitched; `plan.png` | partial: built; correctness untested; rooms segment differently between captures |
| 9 | Per-surface damage regions with class and metric extent | `src/floorplan/damage/` | `plan.json` damage.regions (surface id, class, area upper bound, evidence image) | partial: built; recall **untested** (no damaged room); clean samples report 0 |
| 10 | Concealed-damage flags with the rule that fired | `configs/rules.yaml`, `src/floorplan/damage/rules.py` | damage.concealed_damage_flags[].rule_id / rule / hypothesis / inspect | partial: built, tested on synthetic regions only |
| 11 | Scope line items keyed to surfaces | `src/floorplan/damage/rules.py`, `configs/rules.yaml` | damage.scope_items[] (surface_id, action, quantity with interval) | partial: built, no prices (none given) |
| 12 | Confidence interval on every measurement | `src/floorplan/intervals.py`, `schema.Measurement` | every value has low/high/method; null interval when unmeasurable | met; calibration partial (row 23) |
| 13 | One command per capture | `src/floorplan/cli.py` | `uv run --group models plan run <capture>` | met |
| 14 | JSON to the published schema | `src/floorplan/schema.py` | own pydantic schema (no published Round 1 schema was available to us) | partial |
| 15 | Rendered plan | `src/floorplan/report/render.py` | `plan.png` | met |
| 16 | Benchmark set composition | `docs/benchmark_report.md` §0 | supplied captures only | partial: no staged damage, no ground truth, photos simulated |
| 17 | Gate: opening widths <= 2 cm on >= 85% | `docs/benchmark_report.md` §1 | detection counts disagree 3/7/1 between captures | untested (no ground truth) |
| 18 | Gate: ceiling <= 1.5 cm, spread <= 1 cm, report says which failure | `docs/benchmark_report.md` §1 | 7 rooms measured in one capture | untested; spread unmeasurable |
| 19 | Gate: repeatability 1 cm or 0.5% per wall | `docs/benchmark_report.md` §2 | room dims 3/12, same-wall lengths 2/34 (median 24.7 cm) | fails |
| 20 | Gate: drift accountability, footprint ablation on/off | `src/floorplan/drift/`, `scripts/m4_ablation.py`, `docs/benchmark_report.md` §3 | footprints 61.1 -> 68.7 and 71.2 -> 71.7 m2; the two whole-property captures 14% -> 4% apart | met |
| 21 | Gate: photo-tier whole-property stitch, ±8%, no overlaps | `docs/benchmark_report.md` §1 | no plan | fails |
| 22 | Gates: photo walls ±8%, video walls ±3% | `docs/benchmark_report.md` §1 | no plan | fails |
| 23 | Calibration scored at every tier | `scripts/calibrate_intervals.py`, `docs/benchmark_report.md` §4 | LiDAR walls 79% / areas 100% covered (held-out: 71% / 90%) | partial |
| 24 | Head-to-head vs a consumer app on 2 rooms | `docs/benchmark_report.md` §5 | none | missing (no iPhone) |
| 25 | Fix loop: one-page declaration, before/after regenerable, readable diff | `docs/fix_declaration.md`, `docs/fix_loop.md`, `scripts/run_benchmark.py --variant before/refined/final`, `git diff fix-a-before fix-a-after` | Fix A: paired room widths 1/9 -> 3/9 within the gate, median 9.3 -> 5.3 cm. Fix B: same-wall median 29.7 -> 24.7 cm, room areas 2.03 -> 1.04 m2 | met (gate still fails; reasons stated) |
| 26 | Process evidence: commit history | `git log` | commits per step since the scaffold | met |
| 27 | README to running in < 15 min on a clean machine | `README.md` | install, fetch, one command | met: tested from a fresh clone on Windows, offline run completes; install time on a cold network not measured |
| 28 | Reproduction bundle | `scripts/reproduce.sh`, `bench/results/` | regenerates every reported number from raw captures | met (live paths; no cached model outputs needed) |
| 29 | Benchmark report: gates at all tiers, repeatability, head-to-head, timing | `docs/benchmark_report.md` | sections 1-7 | met (head-to-head missing, row 24) |
| 30 | Technical report <= 6 pages | `docs/technical_report.md` | architecture, tiers, drift, error budget, calibration, fix loop, failure modes | met |
| 31 | Raw benchmark data: sensor logs, ground truth, app exports | not in repo (supplied captures; we have no ground truth or app exports) | — | missing |
| 32 | Pretrained models disclosed; runs without our infrastructure | `DISCLOSURE.md`, `scripts/fetch_models.sh` | pinned revisions; offline after fetch | met |
| 33 | Weights fetched by script, not committed | `scripts/fetch_models.sh`, `.gitignore` | Hugging Face snapshots at pinned revisions | met |
| 34 | Mirrors, glass, wet-look surfaces, low light covered | `docs/technical_report.md` §7, `docs/capture_protocol.md` "Avoid" | failure modes stated; protocol avoids them | partial: stated, not handled in code, not tested |
| 35 | Walk-in test: all three tiers ready to run cold | `src/floorplan/cli.py` | all tiers run; image tiers may return "no plan" | partial |
