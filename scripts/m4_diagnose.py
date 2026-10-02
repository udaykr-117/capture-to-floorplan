"""M4 step 1: what does drift look like? Odometry continuity, and per-chunk wall direction and floor height over time.

Usage: uv run python scripts/m4_diagnose.py [name ...]   (names: single_room floor_only with_ceiling)
Chunks are cached in out/cache (delete after changing the chunk code).
"""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from floorplan.config import load_config
from floorplan.drift.chunks import analyze_chunk, build_chunks
from floorplan.geometry.align import estimate_frame
from floorplan.io.stray import StraySource

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}
CFG = load_config()


def chunks_for(name):
    key = hashlib.md5(json.dumps(dict(s=CFG["stray"], d=CFG["drift"]["chunk_frames"]), sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"chunks_{name}_{key}.pkl"
    src = StraySource(SAMPLES[name], CFG)
    if f.exists():
        return src, pickle.loads(f.read_bytes())
    chunks = build_chunks(src.frames(), CFG)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(chunks))
    return src, chunks


def odometry_continuity(src):
    odo = src.cap.odo
    stride = CFG["stray"]["frame_stride"]
    p = odo[["x", "y", "z"]].values
    t = odo["timestamp"].values
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    from scipy.spatial.transform import Rotation as R

    q = R.from_quat(odo[["qx", "qy", "qz", "qw"]].values)
    w = np.degrees((q[:-1].inv() * q[1:]).magnitude()) / np.diff(t)
    print(f"  odometry (every frame, {len(odo)} rows, {t[-1] - t[0]:.0f} s): speed m/s median {np.median(v):.2f}, p99 {np.percentile(v, 99):.2f}, max {v.max():.2f}; "
          f"rotation rate deg/s median {np.median(w):.0f}, p99 {np.percentile(w, 99):.0f}, max {w.max():.0f}")
    print(f"  steps above 3 m/s: {(v > 3).sum()}, above 180 deg/s: {(w > 180).sum()}; largest single-step jump {np.linalg.norm(np.diff(p, axis=0), axis=1).max() * 100:.1f} cm "
          f"(dt {np.diff(t)[np.argmax(np.linalg.norm(np.diff(p, axis=0), axis=1))] * 1000:.0f} ms)")
    path = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    print(f"  path length {path:.0f} m, start-end distance {np.linalg.norm(p[-1] - p[0]):.1f} m")


def correct_stage(name):
    from floorplan.drift.chunks import analyze_aligned
    from floorplan.drift.correct import estimate

    src, chunks = chunks_for(name)
    pts = np.concatenate([c.pts for c in chunks])
    frame, *_ = estimate_frame(pts, src.camera_positions(), CFG)
    an = [analyze_chunk(c, frame, CFG) for c in chunks]
    corr = estimate(an, CFG)
    dg = corr.diag
    print(f"  wall associations across chunks: {dg['n_assoc']} from {dg['n_segments']} wall segments; residual RMS before {dg['rms_before_cm']:.1f} cm, after {dg['rms_after_cm']:.1f} cm; "
          f"{dg['inliers']} within 5 cm after")
    print("  chunk  yaw dev -> rotation (deg)   shift x', z' (cm)      floor (cm) -> level (cm)")
    for i in range(len(chunks)):
        dv = dg["yaw_dev_deg"][i]
        print(f"  {i:4d}   {dv:+6.2f} -> {dg['theta_deg'][i]:+6.2f}          {dg['shift_cm'][i][0]:+7.1f} {dg['shift_cm'][i][1]:+7.1f}        {dg['floor_cm'][i]:+6.1f} -> {dg['dy_cm'][i]:+6.1f}")
    after = []
    for c, a in zip(chunks, an):
        pa = corr.apply(c.idx, frame.to_aligned(c.pts))
        ca = corr.apply(c.idx, frame.to_aligned(c.cams))
        after.append(analyze_aligned(c.idx, pa, ca, CFG))
    dv0, dv1 = np.array([a.yaw_dev_deg for a in an]), np.array([a.yaw_dev_deg for a in after])
    f0, f1 = np.array([a.floor_y for a in an]) * 100, np.array([a.floor_y for a in after]) * 100
    print(f"  chunk wall direction deviation: std before {np.nanstd(dv0):.2f} deg, after {np.nanstd(dv1):.2f} deg (max abs {np.nanmax(np.abs(dv0)):.2f} -> {np.nanmax(np.abs(dv1)):.2f})")
    print(f"  chunk floor height: std before {np.nanstd(f0):.2f} cm, after {np.nanstd(f1):.2f} cm (range {np.nanmin(f0):+.1f}..{np.nanmax(f0):+.1f} -> {np.nanmin(f1):+.1f}..{np.nanmax(f1):+.1f})")


def main(names, stage="diagnose"):
    if stage == "correct":
        for name in names:
            print(f"\n=================== {name}: correction ===================")
            correct_stage(name)
        return
    for name in names:
        print(f"\n=================== {name} ===================")
        src, chunks = chunks_for(name)
        odometry_continuity(src)
        pts = np.concatenate([c.pts for c in chunks])
        frame, floor, ev, _, _ = estimate_frame(pts, src.camera_positions(), CFG)
        print(f"  global alignment: yaw {frame.yaw_deg:.2f}, floor world y {frame.floor_y:.3f}; {len(chunks)} chunks of {CFG['drift']['chunk_frames']} sampled frames "
              f"(= {CFG['drift']['chunk_frames'] * CFG['stray']['frame_stride']} raw frames)")
        print("  chunk  sampled-frames   cam(x',z') m       vertical pts  yaw dev (deg)  resultant   floor y (cm)  floor pts")
        devs, floors = [], []
        for ch in chunks:
            a = analyze_chunk(ch, frame, CFG)
            cx, cz = a.cams_a[:, 0].mean(), a.cams_a[:, 2].mean()
            devs.append(a.yaw_dev_deg)
            floors.append(a.floor_y)
            dv = "   n/a" if np.isnan(a.yaw_dev_deg) else f"{a.yaw_dev_deg:+6.2f}"
            fy = "   n/a" if np.isnan(a.floor_y) else f"{100 * a.floor_y:+6.1f}"
            print(f"  {ch.idx:4d}  {ch.idx * CFG['drift']['chunk_frames']:5d}..{ch.idx * CFG['drift']['chunk_frames'] + ch.n_frames - 1:<5d}     ({cx:6.2f},{cz:6.2f})   {a.n_vertical:9d}      {dv}        {a.yaw_resultant:5.2f}      {fy}       {a.n_floor:6d}")
        d, f = np.array(devs), np.array(floors) * 100
        ok = ~np.isnan(d)
        print(f"  summary: yaw dev over {ok.sum()} usable chunks: mean {d[ok].mean():+.2f}, std {d[ok].std():.2f}, min {d[ok].min():+.2f}, max {d[ok].max():+.2f} deg; "
              f"floor height over {(~np.isnan(f)).sum()} chunks: std {np.nanstd(f):.2f} cm, range {np.nanmin(f):+.1f}..{np.nanmax(f):+.1f} cm")


if __name__ == "__main__":
    args = sys.argv[1:]
    stage = "correct" if args and args[0] == "correct" else "diagnose"
    names = [a for a in args if a in SAMPLES] or list(SAMPLES)
    main(names, stage)
