"""M5: run the video tier on a sample (video only: no depth, no poses) and compare it with the LiDAR plan of the same capture.

Usage: uv run --group models python scripts/m5_tier_eval.py <capture> [sequential|exhaustive|learned] [fps]
Cached in out/cache: per-image depth (the slow part) and the LiDAR plan. Delete out/cache after changing the depth or SfM code.
"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).parent))
import m5_sfm_diagnose as D
from floorplan.config import load_config
from floorplan.io.imagesource import build_image_source, depth_for_images
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import match_segments, register, segments, to_other, to_other_polygon, wall_points, width_pairs
from floorplan.sfm import run_sfm, run_sfm_learned, umeyama

CFG = load_config()
OUT = D.OUT.parent


def lidar_reference(name):
    f = OUT / "cache" / f"m5_lidar_{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    plan, inter = build_plan(StraySource(D.SAMPLES[name], CFG), CFG)
    d = dict(plan=plan, planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], CFG))
    f.write_bytes(pickle.dumps(d))
    return d


def main(name, matcher="sequential", fps=5.0):
    odo = pd.read_csv(D.SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    t_all = time.time()
    fdir, fr = D.frames_for(name, fps, True)
    t0 = time.time()
    sfm = run_sfm_learned(fdir, OUT / "m5" / name / "work_learned_disk", CFG) if matcher == "learned" else run_sfm(fdir, OUT / "m5" / name / f"work_{matcher}", CFG, matcher)
    t_sfm = time.time() - t0
    print(f"SfM ({matcher}): {sfm.n_registered}/{sfm.n_input} images in the largest of {sfm.n_models} models, {sfm.n_points} points, reprojection {sfm.reproj_err:.2f}px, {t_sfm:.0f}s")
    idx = np.array([int(i.name[1:-4]) for i in sfm.images])
    cams = np.array([i.center for i in sfm.images])
    ark = odo.loc[idx, ["x", "y", "z"]].values.astype(float)
    s_true, _, _ = umeyama(cams, ark)

    key = hashlib.md5((str(fdir) + matcher + json.dumps(CFG["depth"], sort_keys=True) + str(sfm.n_registered)).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m5_depth_{name}_{key}.pkl"
    t0 = time.time()
    if f.exists():
        per_image = pickle.loads(f.read_bytes())
    else:
        per_image = depth_for_images(sfm, fdir, CFG)
        f.write_bytes(pickle.dumps(per_image))
    t_depth = time.time() - t0
    src = build_image_source(f"{name} (video only)", per_image, CFG, sfm.n_input, sfm.n_registered, dict(sfm=t_sfm, depth=t_depth))
    info, sc = src.info, src.info["scale"]
    print(f"depth aligned for {info['n_aligned']} images ({t_depth:.0f}s); per-frame factor spread (std of ln) {info['per_frame_a_spread']:.2f}; gravity refined by {info['gravity']['shift_deg']:.1f} deg; camera height {info['camera_height_m']:.2f} m")
    print("\nSCALE (metres per SfM unit) against the true value from the ARKit trajectory:")
    print(f"  true (ARKit)          {s_true:.4f}")
    print(f"  depth model           {sc.from_depth:.4f}   error {100 * (sc.from_depth / s_true - 1):+.1f}%   (assumed sigma {100 * sc.sigma_depth:.0f}%)")
    if sc.from_height:
        print(f"  camera-height prior   {sc.from_height:.4f}   error {100 * (sc.from_height / s_true - 1):+.1f}%   (assumed sigma {100 * sc.sigma_height:.0f}%)")
    print(f"  combined              {sc.metres_per_unit:.4f}   error {100 * (sc.metres_per_unit / s_true - 1):+.1f}%   (sigma {100 * sc.rel_sigma:.0f}%)  [{sc.note}]")

    t0 = time.time()
    plan, inter = build_plan(src, CFG, tier="video")
    compare_to_lidar(name, plan, inter, time.time() - t0, time.time() - t_all)


def compare_to_lidar(name, plan, inter, t_plan, t_total):
    ref = lidar_reference(name)
    print(f"\nPLAN from the video only: {len(plan.rooms)} rooms, footprint {plan.stitched.footprint_area.value} m2; LiDAR plan of the same capture: {len(ref['plan'].rooms)} rooms, "
          f"footprint {ref['plan'].stitched.footprint_area.value} m2. Pipeline {t_plan:.0f}s; total {t_total:.0f}s")
    W = wall_points(inter["Pa"], inter["Na"], CFG)
    reg = register(ref["W"], W, CFG)
    print(f"registration of the video-tier walls onto the LiDAR walls: {100 * reg.fitness_5cm:.1f}% within 5 cm, median {reg.median_cm:.1f} cm, null {100 * reg.null_fitness_5cm:.0f}% (accepted: {reg.accepted})")
    SA, SB = segments(ref["planes"]), [to_other(s, reg) for s in segments(inter["planes"])]
    m = match_segments(SA, SB, CFG)
    pairs = width_pairs(SA, SB, m, CFG)
    pct = np.array([100 * p["delta"] / p["width_a"] for p in pairs])
    print(f"wall segments: video {len(SB)}, LiDAR {len(SA)}, matched {len(m)}; facing-wall widths compared: {len(pairs)}")
    if len(pairs):
        print(f"  video-tier width vs LiDAR width: median signed {np.median(pct):+.1f}%, median |error| {np.median(np.abs(pct)):.1f}%, 90th pct {np.percentile(np.abs(pct), 90):.1f}%; "
              f"within 3%: {100 * (np.abs(pct) <= 3).mean():.0f}% (video gate), within 8%: {100 * (np.abs(pct) <= 8).mean():.0f}% (photo gate)")
    pa = {r.id: Polygon(r.polygon) for r in ref["plan"].rooms}
    pb = unary_union([to_other_polygon(Polygon(r.polygon), reg) for r in plan.rooms]) if plan.rooms else Polygon()
    ref_union = unary_union(list(pa.values()))
    print(f"coverage: the video-tier rooms overlap {100 * pb.intersection(ref_union).area / ref_union.area:.0f}% of the LiDAR footprint; "
          f"{100 * (1 - pb.intersection(ref_union).area / max(pb.area, 1e-9)):.0f}% of the video-tier footprint lies outside it")
    for r in plan.rooms:
        c = r.ceiling_height
        print(f"  {r.id}: area {r.area.value} m2 [{r.area.interval.low}, {r.area.interval.high}], ceiling " + (f"{c.value} m ({c.status})" if c.value else "unmeasurable"))




if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "single_room", a[1] if len(a) > 1 else "sequential", float(a[2]) if len(a) > 2 else 5.0)
