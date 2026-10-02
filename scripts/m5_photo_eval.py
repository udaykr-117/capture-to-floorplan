"""M5: run the photo tier on the simulated photo folders (m5_make_photos.py) and score it against the ARKit poses and the LiDAR plan.

Usage: uv run --group models python scripts/m5_photo_eval.py <capture>
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import m5_tier_eval as T
from floorplan.sfm import umeyama
from floorplan.tiers.images import run_photos


def main(name):
    folder = T.OUT / "m5" / f"photos_{name}"
    n_photos = len(list(folder.glob("*/*.jpg")))
    rooms = sorted(p.name for p in folder.iterdir() if p.is_dir())
    print(f"{name}: {n_photos} photos in {len(rooms)} room folders ({rooms})")
    t0 = time.time()
    plan, inter = run_photos(folder, T.OUT / "m5" / f"photo_run_{name}", T.CFG)
    print(f"photo tier finished in {time.time() - t0:.0f}s: {plan.stitched.n_rooms} rooms; timing {plan.timing_s}")
    for line in plan.limitations:
        print("  note:", line)
    if not inter:
        return
    sfm = inter["sfm"]
    odo = pd.read_csv(T.D.SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    idx = np.array([int(re.search(r"(\d{6})$", i.name[:-4]).group(1)) for i in sfm.images])
    per_room = {}
    for i in sfm.images:
        per_room[i.name.split("__")[0]] = per_room.get(i.name.split("__")[0], 0) + 1
    print(f"SfM: {sfm.n_registered}/{sfm.n_input} photos registered in the largest of {sfm.n_models} models; per room folder: {per_room}")
    if sfm.n_registered >= 4:
        cams = np.array([i.center for i in sfm.images])
        ark = odo.loc[idx, ["x", "y", "z"]].values.astype(float)
        s_true, R, t = umeyama(cams, ark)
        rmse = np.sqrt((np.linalg.norm(s_true * cams @ R.T + t - ark, axis=1) ** 2).mean()) * 100
        sc = inter["source"].info["scale"]
        print(f"pose check against ARKit: scale {s_true:.4f} m/unit, residual {rmse:.1f} cm over {sfm.n_registered} photos")
        print(f"  scale estimate: depth {sc.from_depth:.4f} ({100 * (sc.from_depth / s_true - 1):+.1f}%), height prior "
              f"{sc.from_height:.4f} ({100 * (sc.from_height / s_true - 1):+.1f}%), combined {sc.metres_per_unit:.4f} ({100 * (sc.metres_per_unit / s_true - 1):+.1f}%, sigma {100 * sc.rel_sigma:.0f}%)")
    if plan.rooms:
        T.compare_to_lidar(name, plan, inter, plan.timing_s.get("sfm", 0), time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "single_room")
