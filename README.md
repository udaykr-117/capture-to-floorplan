# floorplan: phone captures to a measured floor plan with intervals

One command turns one capture (LiDAR scan, video clip or photo folders) into `plan.json` (pydantic schema, `src/floorplan/schema.py`) and
`plan.png`: rooms with walls, wall lengths, floor area, ceiling height and openings, all with an interval; the stitched multi-room plan with adjacency;
damage regions, concealed-damage flags naming the rule that fired, and scope items keyed to surfaces.

## Status at a glance
No iPhone, no GPU and no tape measurements were available: the benchmark is the three Stray Scanner captures supplied with the brief (one
property), all on CPU. Every "accuracy" number below is agreement between those repeat captures, not error against a tape.

| Tier | Runs on | Result on the samples |
|---|---|---|
| LiDAR | Stray Scanner export (iPhone/iPad Pro) | Full plans in 25-165 s (+2-10 min for damage). Repeatability gate **fails** (see below). Opening and ceiling accuracy untested. |
| Video | any clip | Runs; **no plan** on the samples: structure from motion places 4-18% of the frames (plain walls, blur). Says so with the reason, never invents rooms. |
| Photo | room folders (JPEG, HEIC) | Runs; **no plan** on photo sets simulated from the videos (0 frames placed). Real photo sets untested. |

| What the three captures show (shipped pipeline, `out/final_bench_after2a.txt`) | Value |
|---|---|
| Room widths agreeing within 1 cm or 0.5% | 3 of 12 (median difference 7.7 cm) |
| Same wall's length, median difference between captures | 24.7 cm (was 29.7 cm before the second fix) |
| Room area, median difference | 1.04 m2, 15% (was 2.03 m2) |
| Two whole-house footprints, drift correction off / on | 14% / 4% apart |
| Intervals that contain the real difference: walls / room areas | 79% / 100% (was 24% / 30% before calibration) |
| Openings found in the three captures | 3 / 7 / 1 (detection disagrees: gate untested, almost surely fails) |
| Damage regions on the clean samples | 0 (threshold set by negative control; recall on real damage untested) |

Gates that fail or are untested, and why: `docs/benchmark_report.md` §1. Requirement-by-requirement status: `docs/compliance_matrix.md`.

## Install (clean machine, about 5-15 minutes, mostly downloads)
Tested from a fresh `git clone` on Windows 11: `uv sync --group models` + `fetch_models.sh` took 38 s with a warm download cache, then
`plan run` on `single_room` ran in 172 s **offline** (`HF_HUB_OFFLINE=1`), damage pass included. A cold machine adds the downloads
(the installed environment is 1.4 GB on Windows; Linux takes the CPU-only torch; weights 0.7 GB). Cold-network time was not measured.
1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/ (one command; it brings its own Python 3.12).
2. Get the code and the dependencies:
   ```
   git clone <this repo> floorplan && cd floorplan
   uv sync --group models
   ```
   (`uv sync` alone is enough for the LiDAR tier without damage detection.) On Windows, clone into a short path such as `C:\fp`: a deep
   folder makes some installed file paths exceed Windows' 260-character limit and the install breaks.
3. Fetch the model weights once (Depth Anything V2 small, OWL-ViT base, DISK + LightGlue; pinned revisions, about 0.8 GB; no network needed afterwards):
   ```
   bash scripts/fetch_models.sh
   ```
   On Windows run it from Git Bash.

## Run: one command per capture
Capture with `docs/capture_protocol.md`, copy the files to the computer, then:
```
uv run --group models plan run <capture>
```
- `<capture>` = a Stray Scanner folder (LiDAR tier), a video file (video tier) or a folder of room folders (photo tier); the tier is detected,
  or force it with `--tier lidar|video|photo`.
- Output: `out/<tier>_<name>/plan.json`, `plan.png`, `damage/*.jpg` (evidence images, if any). The terminal prints every room, the damage summary
  and the plan's limitations.
- `--no-damage` skips damage detection. `--out` and `--config` choose the output folder and the config file.
- Video and photo captures that cannot be reconstructed return a plan with **no rooms and the reason** (`limitations`, `frame.tier_report`).
- Switches worth knowing (all in `configs/default.yaml`, each with its justification): `drift.*` (drift correction), `refine.*` (room-local wall
  refinement), `rooms.keep_narrow_rooms` / `rooms.split_connectors` (corridors), `openings.*`, `damage.score_min`, `sfm.matcher: sift|learned`.

## Data
The three sample captures are not in the repo (size). Put them in the repo root as supplied:
`single_room/c00a170fe1/`, `single_scan_floor_only/1a8384c3f6/`, `single_scan_with_ceiling/c7d28f72c6/`.
Example: `uv run --group models plan run single_scan_with_ceiling/c7d28f72c6`.

## Reproduce every reported number
```
bash scripts/reproduce.sh          # about 1.5-2.5 h on a laptop CPU; logs in out/
```
Single steps:
- `uv run python scripts/run_benchmark.py --variant before|after|after2a|after2|after3` (fix loops, ~5 min each; `after2a` is what ships)
- `uv run python scripts/m4_ablation.py` (drift off/on), `scripts/calibrate_intervals.py` (interval calibration)
- `scripts/m5_tier_eval.py`, `scripts/m5_photo_eval.py` (video and photo tiers), `scripts/m6_negative_control.py` (damage threshold)
- `scripts/m7_diagnose.py`, `scripts/seg_*.py`, `scripts/refine_visits.py`, `scripts/openings_xcap.py` (the diagnoses behind the fixes)

Benchmark outputs that the reports quote are committed in `bench/results/`.

## The fix loop in one paragraph
Worst gate = repeatability. Round 1: wall positions drift by ~3.5 cm over a capture, so each room's walls are re-measured from its longest single
visit (paired room widths 1/9 -> 3/9 within the gate, median 9.3 -> 5.3 cm). Round 2: rooms are cut at different places in different captures
(a corridor belongs to different rooms), so narrow spaces are kept as rooms (same-wall length 29.7 -> 24.7 cm, areas 2.03 -> 1.04 m2). Both
declarations, the predictions made before running, and where each prediction missed are in `docs/fix_declaration.md`; the gate still fails.

## Tests
```
uv run --group models pytest -q          # 74 tests
```

## Layout
- `src/floorplan/`: `io/` (Stray Scanner, video frames, image source), `geometry/` (alignment, walls, gravity), `rooms/` (free space, polygons,
  ceilings, openings, room-local refinement), `drift/` (chunk correction, odometry jumps), `tiers/` (video/photo), `damage/` (detector, projection,
  rules), `sfm.py`, `depth.py`, `scale.py`, `intervals.py`, `register.py`, `pipeline.py`, `cli.py`, `report/render.py`.
- `configs/default.yaml`: every threshold with a one-line justification. `configs/rules.yaml`: concealed-damage rules and scope actions.
- `docs/`: `capture_protocol.md`, `device_matrix.md`, `technical_report.md`, `benchmark_report.md`, `compliance_matrix.md`, `fix_declaration.md`.
- `scripts/`: benchmark, reproduction, calibration and diagnosis scripts. `tests/`: pytest. `bench/results/`: benchmark outputs.
- `DISCLOSURE.md`: every external model, library and reference.
