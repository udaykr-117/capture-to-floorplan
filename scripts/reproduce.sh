#!/usr/bin/env bash
# Regenerate every number in docs/benchmark_report.md and docs/technical_report.md from the raw captures.
# Needs the three sample captures in the repo root (see README "Data"). CPU only. Total about 1.5-2.5 hours, mostly the video tier and damage detection.
# Each step writes its own log in out/; nothing is read from a previous run except the explicitly named caches (deleted first).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p out
rm -rf out/cache

echo "== 1. fix loop: before / after (repeatability, wall lengths, areas, openings, calibration, timing)"
uv run python scripts/run_benchmark.py --variant before | tee out/final_bench_before.txt
uv run python scripts/run_benchmark.py --variant after  | tee out/final_bench_after.txt     # round 1 after = round 2 before
uv run python scripts/run_benchmark.py --variant after2a | tee out/final_bench_after2a.txt  # round 2, shipped
uv run python scripts/run_benchmark.py --variant after2  | tee out/final_bench_after2.txt   # round 2 with the corridor split (not shipped)
uv run python scripts/run_benchmark.py --variant after3  | tee out/final_bench_after3.txt   # median over visits (tested for the round-1 regressions, rejected)

echo "== 2. drift on/off ablation (stitched footprint, wall sharpness, loop gap, cross-capture agreement)"
uv run python scripts/m4_ablation.py | tee out/final_drift_ablation.txt

echo "== 3. fix-loop diagnosis (hypotheses H1-H6)"
uv run python scripts/m7_diagnose.py     | tee out/m7_diagnose.txt
uv run python scripts/m7_diagnose.py h45 | tee out/m7_diagnose_h45.txt
uv run python scripts/m7_diagnose.py h6  | tee out/m7_diagnose_h6.txt

echo "== 4. video tier on the three sample videos (video only: no depth, no poses), compared with the LiDAR plan"
uv run --group models python scripts/m5_tier_eval.py single_room sequential 5 | tee out/final_video_single_room.txt   # 5 fps: the reported single_room run
for c in floor_only with_ceiling; do
  uv run --group models python scripts/m5_tier_eval.py "$c" sequential 3 | tee "out/final_video_$c.txt"
done

echo "== 5. photo tier on photo sets simulated from the videos"
for c in single_room with_ceiling; do
  uv run python scripts/m5_make_photos.py "$c" 5
  uv run --group models python scripts/m5_photo_eval.py "$c" | tee "out/final_photo_$c.txt"
done

echo "== 6. damage: live detection on keyframes, then the negative control (score threshold sweep)"
for c in single_room floor_only with_ceiling; do
  uv run --group models python scripts/m6_detect_explore.py "$c" | tee "out/m6_detect_$c.txt"
done
uv run python scripts/m6_negative_control.py | tee out/m6_negative_control.txt

echo "== 7. tests"
uv run --group models pytest -q
