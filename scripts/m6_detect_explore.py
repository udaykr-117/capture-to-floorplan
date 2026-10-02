"""M6 step 1: run the detector on keyframes of each sample and list every detection; save crops to look at.
Usage: uv run --group models python scripts/m6_detect_explore.py <capture key> [keyframe_every]
Output: out/m6/<key>/detections.json, out/m6/<key>/crops/*.jpg, out/m6_detect_<key>.txt (printed)
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from floorplan.config import load_config
from floorplan.damage.detect import Detector

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}


def main(key, every=None):
    cfg = load_config()
    every = every or cfg["damage"]["keyframe_every"]
    out = ROOT / "out" / "m6" / key
    (out / "crops").mkdir(parents=True, exist_ok=True)
    det = Detector(cfg)
    cap = cv2.VideoCapture(str(SAMPLES[key] / "rgb.mp4"))
    n = int(cap.get(7))
    all_d, times = [], []
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        if i % every:
            continue
        t0 = time.time()
        ds = det.detect(bgr, i)
        times.append(time.time() - t0)
        for d in ds:
            x0, y0, x1, y1 = map(int, d.box)
            img = cv2.resize(bgr, (960, 720))
            cv2.rectangle(img, (x0 // 2, y0 // 2), (x1 // 2, y1 // 2), (0, 0, 255), 3)
            cv2.putText(img, f"{d.cls} {d.score:.2f}", (x0 // 2 + 4, y0 // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(out / "crops" / f"f{i:05d}_{d.cls}_{int(100 * d.score):02d}.jpg"), img)
            all_d.append(d)
            print(f"frame {i:5d} {d.cls:14s} {d.score:.3f} box {x0},{y0},{x1},{y1} ({100 * (x1 - x0) * (y1 - y0) / (1920 * 1440):.1f}% of image) prompt '{d.prompt}'")
    (out / "detections.json").write_text(json.dumps([asdict(d) for d in all_d], indent=1))
    by = {}
    for d in all_d:
        by[d.cls] = by.get(d.cls, 0) + 1
    print(f"{key}: {len(times)} keyframes, {sum(times) / max(len(times), 1):.2f} s per keyframe, {len(all_d)} detections {by}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
