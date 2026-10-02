"""M5 step 1: why does COLMAP fragment on single_room video, and which setting helps? One hypothesis per variant.

Compares each reconstruction with the ARKit trajectory (Sim(3) fit of camera centres): scale, residual and time span covered.
Usage: uv run --group models python scripts/m5_sfm_diagnose.py [capture]     (capture: single_room | floor_only | with_ceiling)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from floorplan.config import load_config
from floorplan.io.frames import extract_video_frames
from floorplan.sfm import run_sfm, umeyama

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "m5"
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}
CFG = load_config()
VARIANTS = {
    "baseline (COLMAP defaults)": {},
    "longer matching window (overlap 25)": dict(seq_overlap=25),
    "more features on blank walls (peak 0.003, 12k)": dict(sift_peak_threshold=0.003, sift_max_features=12000),
    "small baselines allowed (init tri angle 4 deg)": dict(init_min_tri_angle_deg=4.0),
    "guided matching": dict(guided_matching=True),
    "all four together": dict(seq_overlap=25, sift_peak_threshold=0.003, sift_max_features=12000, init_min_tri_angle_deg=4.0, guided_matching=True),
}


def frames_for(name, fps=None, sharpest=False):
    fps = fps or CFG["sfm"]["video_fps"]
    d = OUT / name / f"frames_{fps:g}fps{'_sharpest' if sharpest else ''}"
    existing = sorted(d.glob("f*.jpg")) if d.exists() else []
    if existing:
        return d, [(int(p.stem[1:]), p) for p in existing]
    return d, extract_video_frames(SAMPLES[name] / "rgb.mp4", d, fps, CFG["sfm"]["frame_width_px"], sharpest)


def report(name, odo, d, ov, label):
    t0 = time.time()
    sfm = run_sfm(d, OUT / name / "work", CFG, "sequential", ov)
    dt = time.time() - t0
    if sfm is None:
        print(f"{label:50s} no reconstruction")
        return
    idx = np.array([int(i.name[1:-4]) for i in sfm.images])
    cams = np.array([i.center for i in sfm.images])
    ark = odo.loc[idx, ["x", "y", "z"]].values.astype(float)
    s, R, t = umeyama(cams, ark)
    rmse = np.sqrt((np.linalg.norm(s * cams @ R.T + t - ark, axis=1) ** 2).mean()) * 100
    print(f"{label:50s} {sfm.n_models:6d} {sfm.n_registered:4d}/{sfm.n_input:<4d} {sfm.n_points:7d} {sfm.reproj_err:6.2f}p {dt:7.0f} {idx.min() / len(odo):5.0%}..{idx.max() / len(odo):<5.0%}   {s:12.3f} {rmse:11.1f} cm", flush=True)


def frame_selection(name):
    odo = pd.read_csv(SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    print(f"{name}: frame selection with default COLMAP settings (video has {len(odo)} frames)")
    print(f"{'frames':50s} {'models':>6s} {'reg/in':>9s} {'points':>7s} {'reproj':>7s} {'time s':>7s} {'span of video':>14s} {'scale m/unit':>13s} {'RMSE vs ARKit':>14s}")
    for fps, sharp in ((5, False), (5, True), (10, True), (10, False)):
        d, fr = frames_for(name, fps, sharp)
        report(name, odo, d, {}, f"{fps} fps, {'sharpest in each window' if sharp else 'fixed times'} ({len(fr)} frames)")


def main(name):
    odo = pd.read_csv(SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    d, fr = frames_for(name)
    print(f"{name}: {len(fr)} frames at {CFG['sfm']['video_fps']} fps, {CFG['sfm']['frame_width_px']} px wide (video has {len(odo)} frames)")
    print(f"{'variant':50s} {'models':>6s} {'reg/in':>9s} {'points':>7s} {'reproj':>7s} {'time s':>7s} {'span of video':>14s} {'scale m/unit':>13s} {'RMSE vs ARKit':>14s}")
    for label, ov in VARIANTS.items():
        report(name, odo, d, ov, label)


if __name__ == "__main__":
    cap = sys.argv[1] if len(sys.argv) > 1 else "single_room"
    frame_selection(cap) if len(sys.argv) > 2 and sys.argv[2] == "frames" else main(cap)
