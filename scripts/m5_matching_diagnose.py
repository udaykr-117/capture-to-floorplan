"""M5: can exhaustive or much longer-range matching link the separate models of a low-texture video? Usage: m5_matching_diagnose.py [capture]"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import m5_sfm_diagnose as D
from floorplan.sfm import run_sfm, umeyama


def main(name):
    odo = pd.read_csv(D.SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    d, fr = D.frames_for(name, 5, True)
    print(f"{name}: {len(fr)} frames (5 fps, sharpest in window); video has {len(odo)} frames")
    print(f"{'matching':50s} {'models':>6s} {'reg/in':>9s} {'points':>7s} {'reproj':>7s} {'time s':>7s} {'span of video':>14s} {'scale m/unit':>13s} {'RMSE vs ARKit':>14s}")
    for label, matcher, ov in (("sequential, overlap 10 (default)", "sequential", {}), ("sequential, overlap 50", "sequential", dict(seq_overlap=50)),
                               ("exhaustive", "exhaustive", {})):
        t0 = time.time()
        sfm = run_sfm(d, D.OUT / name / "work", D.CFG, matcher, ov)
        dt = time.time() - t0
        idx = np.array([int(i.name[1:-4]) for i in sfm.images])
        cams = np.array([i.center for i in sfm.images])
        s, R, t = umeyama(cams, odo.loc[idx, ["x", "y", "z"]].values.astype(float))
        rmse = np.sqrt((np.linalg.norm(s * cams @ R.T + t - odo.loc[idx, ["x", "y", "z"]].values, axis=1) ** 2).mean()) * 100
        print(f"{label:50s} {sfm.n_models:6d} {sfm.n_registered:4d}/{sfm.n_input:<4d} {sfm.n_points:7d} {sfm.reproj_err:6.2f}p {dt:7.0f} {idx.min() / len(odo):5.0%}..{idx.max() / len(odo):<5.0%}   {s:12.3f} {rmse:11.1f} cm", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "single_room")
