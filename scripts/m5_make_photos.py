"""M5: simulate the photo protocol from a sample video. There are no real photo folders, so frames are chosen as a person following the
protocol would shoot: per room about 5 sharp stills at well-spread headings, plus doorway shots looking through each door into the next room.

The ARKit poses and the LiDAR room outlines are used ONLY to choose frames (a stand-in for the person's choice); the photo tier itself gets
nothing but the images. Usage: uv run python scripts/m5_make_photos.py <capture> [photos per room]
Output: out/m5/photos_<capture>/<room>/<name>.jpg and manifest.json. File names end in the video frame index (used to score poses afterwards).
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot
from shapely.geometry import Point, Polygon

from floorplan.config import load_config
from floorplan.io.frames import sharpness
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}
CFG = load_config()


def lidar_plan_and_frame(name):
    f = OUT / "cache" / f"m5_lidarplan_{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    plan, inter = build_plan(StraySource(SAMPLES[name], CFG), CFG)
    d = dict(plan=plan, frame=inter["al"].frame)
    f.write_bytes(pickle.dumps(d))
    return d


def pick_spread(cands, heading, sharp, k, min_sep_deg=25.0):
    """Greedy: sharpest first, then the candidate whose heading is farthest from those already chosen."""
    if not cands:
        return []
    chosen = [max(cands, key=lambda i: sharp[i])]
    while len(chosen) < k:
        def sep(i):
            return min(abs((np.degrees(heading[i] - heading[j]) + 180) % 360 - 180) for j in chosen)
        best = max((i for i in cands if i not in chosen), key=sep, default=None)
        if best is None or sep(best) < min_sep_deg:
            break
        chosen.append(best)
    return chosen


def main(name, per_room=5):
    ref = lidar_plan_and_frame(name)
    plan, frame = ref["plan"], ref["frame"]
    odo = pd.read_csv(SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    n = len(odo)
    cams_a = frame.to_aligned(odo[["x", "y", "z"]].values)
    Rm = Rot.from_quat(odo[["qx", "qy", "qz", "qw"]].values).as_matrix()
    fwd = frame.rotate(Rm[:, :, 2])                       # camera forward (OpenCV z) in the aligned frame
    heading, pitch = np.arctan2(fwd[:, 2], fwd[:, 0]), fwd[:, 1]
    stride = 3
    sharp = {}
    cap = cv2.VideoCapture(str(SAMPLES[name] / "rgb.mp4"))
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            sharp[i] = sharpness(bgr)
    cap.release()

    chosen: dict[int, str] = {}
    rooms = {r.id: Polygon(r.polygon) for r in plan.rooms}
    for rid, poly in rooms.items():
        inner = poly.buffer(-0.35)
        cands = [i for i in sharp if inner.contains(Point(cams_a[i, 0], cams_a[i, 2])) and abs(pitch[i]) < 0.5]
        if not cands:
            print(f"  {rid}: no camera position inside this room, no photos")
            continue
        med = np.median([sharp[i] for i in cands])
        cands = [i for i in cands if sharp[i] >= med]
        for i in pick_spread(cands, heading, sharp, per_room):
            chosen[i] = f"{rid}/p{i:06d}"
    n_rooms_photos = len({v.split('/')[0] for v in chosen.values()})
    for o in plan.openings:
        if o.kind != "door" or None in o.rooms or o.rooms[0] == o.rooms[1]:
            continue
        mid = np.array([(o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2])
        for x, y in ((o.rooms[0], o.rooms[1]), (o.rooms[1], o.rooms[0])):
            px = rooms[x]
            c = []
            for i in sharp:
                d = mid - cams_a[i, [0, 2]]
                dist = np.linalg.norm(d)
                if px.buffer(0.1).contains(Point(cams_a[i, 0], cams_a[i, 2])) and 0.8 < dist < 2.8 and abs(pitch[i]) < 0.5 \
                        and (fwd[i, [0, 2]] / max(np.linalg.norm(fwd[i, [0, 2]]), 1e-9)) @ (d / dist) > 0.8:
                    c.append(i)
            if c:
                i = max(c, key=lambda j: sharp[j])
                chosen[i] = f"{x}/door_{y}_{i:06d}"
    out = OUT / "m5" / f"photos_{name}"
    for p in out.glob("*/*.jpg"):
        p.unlink()
    cap = cv2.VideoCapture(str(SAMPLES[name] / "rgb.mp4"))
    for i in range(n):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in chosen:
            p = out / f"{chosen[i]}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.resize(bgr, (1440, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()
    (out / "manifest.json").write_text(json.dumps({v: k for k, v in chosen.items()}, indent=1))
    per = {}
    for v in chosen.values():
        per[v.split("/")[0]] = per.get(v.split("/")[0], 0) + 1
    print(f"{name}: {len(chosen)} photos in {len(per)} room folders {per}; {sum('door_' in v for v in chosen.values())} doorway shots -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "single_room", int(sys.argv[2]) if len(sys.argv) > 2 else 5)
