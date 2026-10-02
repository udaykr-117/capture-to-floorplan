# floorplan: phone captures to a measured floor plan with intervals

One command turns one capture (LiDAR scan, video clip or photo folders) into `plan.json` (pydantic schema, `src/floorplan/schema.py`) and
`plan.png`: rooms with walls, wall lengths, floor area, ceiling height and openings, all with an interval; the stitched multi-room plan with adjacency;
damage regions, concealed-damage flags naming the rule that fired, and scope items keyed to surfaces.

**Honest status** (details in `docs/benchmark_report.md`): the LiDAR tier produces full plans on the three sample captures; there is no ground
truth, so accuracy is untested, and the repeatability gate fails. The video and photo tiers run but produce an explicit "no plan" result on the
samples (structure from motion places too few frames). The damage pass reports nothing on the clean samples; whether it finds real damage is untested.

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
3. Fetch the model weights once (Depth Anything V2 small, OWL-ViT base; pinned revisions, about 0.7 GB; no network needed afterwards):
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
- `--no-damage` skips damage detection (LiDAR tier: 25-165 s instead of +2-10 min). `--out`, `--config` choose the output folder and config.

## Data
The three sample captures are not in the repo (size). Put them in the repo root as supplied:
`single_room/c00a170fe1/`, `single_scan_floor_only/1a8384c3f6/`, `single_scan_with_ceiling/c7d28f72c6/`.
Example: `uv run --group models plan run single_scan_with_ceiling/c7d28f72c6`.

## Reproduce every reported number
```
bash scripts/reproduce.sh          # about 1.5-2.5 h on a laptop CPU; logs in out/
```
Single steps: `uv run python scripts/run_benchmark.py --variant before|after` (fix loop, ~5 min each), `uv run python scripts/m4_ablation.py`
(drift on/off), `scripts/m5_tier_eval.py`, `scripts/m5_photo_eval.py` (video/photo tiers), `scripts/m6_negative_control.py` (damage).
Benchmark outputs that the reports quote are committed in `bench/results/`.

## Tests
```
uv run --group models pytest -q
```

## Layout
- `src/floorplan/`: `io/` (Stray Scanner, video frames, image source), `geometry/` (alignment, walls, gravity), `rooms/` (free space, polygons,
  ceilings, openings, room-local refinement), `drift/` (chunk correction, odometry jumps), `tiers/` (video/photo), `damage/` (detector, projection,
  rules), `sfm.py`, `depth.py`, `scale.py`, `intervals.py`, `register.py`, `pipeline.py`, `cli.py`, `report/render.py`.
- `configs/default.yaml`: every threshold with a one-line justification. `configs/rules.yaml`: concealed-damage rules and scope actions.
- `docs/`: capture protocol, device matrix, technical report, benchmark report, compliance matrix, fix declaration.
- `scripts/`: benchmark, reproduction, diagnosis scripts per stage. `tests/`: pytest. `DISCLOSURE.md`: every external model, library and reference.
