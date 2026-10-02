"""M6 negative control: the three samples contain no visible damage (11 of 11 detector boxes looked at were false), so any region reported on
them is a false positive. Sweep the score threshold and print, per capture, how many regions survive the full pipeline (projection, surfaces,
min views, size caps). The smallest threshold with zero regions on all three is the false-positive floor.
Usage: uv run python scripts/m6_negative_control.py   (needs out/m6/<key>/detections.json from m6_detect_explore.py, made at score >= 0.10)
"""
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from floorplan.config import load_config
from floorplan.damage.detect import Detection
from floorplan.damage.project import box_points, merge_patches, project_detection
from floorplan.damage.run import room_table
from floorplan.io.stray import StraySource, pose
from floorplan.pipeline import build_plan

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}
THRESHOLDS = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32]


def patches_for(key, cfg):
    dets = [Detection(d["frame"], d["cls"], d["prompt"], d["score"], tuple(d["box"])) for d in json.loads((ROOT / "out" / "m6" / key / "detections.json").read_text())]
    src = StraySource(SAMPLES[key], cfg)
    plan, inter = build_plan(src, cfg)
    cap, fr, corr, s = src.cap, inter["al"].frame, inter["corr"], cfg["stray"]
    rooms, out = room_table(plan, inter), []
    for i in sorted({d.frame for d in dets}):
        depth = cv2.imread(str(cap.dir / "depth" / f"{i:06d}.png"), -1).astype(np.float32) / 1000.0
        conf = cv2.imread(str(cap.dir / "confidence" / f"{i:06d}.png"), -1)
        R, t = pose(cap.odo.iloc[i])
        ci = min((i // s["frame_stride"]) // cfg["drift"]["chunk_frames"], len(inter["chunks"]) - 1)
        to_plan = lambda p, R=R, t=t, ci=ci: corr.apply(ci, fr.to_aligned(p @ R.T + t))
        cam = to_plan(np.zeros((1, 3)))[0]
        for d in (d for d in dets if d.frame == i):
            p, _ = project_detection(d, box_points(d, depth, conf, cap.K_depth, (s["rgb_width"], s["rgb_height"]), cfg), to_plan, cam, cap.K_depth, rooms, cfg)
            if p is not None:
                out.append(p)
    return dets, out


def main():
    cfg = load_config()
    rows = {}
    for key in SAMPLES:
        dets, ps = patches_for(key, cfg)
        print(f"{key}: {len(dets)} detections, {len(ps)} on a wall/ceiling; highest detection score {max(d.score for d in dets):.3f}")
        rows[key] = []
        for t in THRESHOLDS:
            c = copy.deepcopy(cfg)
            c["damage"]["score_min"] = t
            regions, _ = merge_patches([p for p in ps if p.det.score >= t], c)
            rows[key].append((len(regions), max((r.score_max for r in regions), default=0.0)))
    print("\nregions reported on clean captures (all false positives) by score threshold:")
    print("threshold  " + "  ".join(f"{k:>14s}" for k in SAMPLES))
    for j, t in enumerate(THRESHOLDS):
        print(f"  {t:.2f}     " + "  ".join(f"{rows[k][j][0]:>14d}" for k in SAMPLES))
    floor = next((t for j, t in enumerate(THRESHOLDS) if all(rows[k][j][0] == 0 for k in SAMPLES)), None)
    print(f"\nsmallest threshold with zero regions on all three: {floor}")


if __name__ == "__main__":
    main()
