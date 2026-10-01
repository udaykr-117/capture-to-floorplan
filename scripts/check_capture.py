"""Print evidence about one capture: up axis, planes, floor/ceiling, wall directions; save a top-down image.

Usage: uv run python scripts/check_capture.py <capture_dir> [name]
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from floorplan.config import load_config
from floorplan.io.stray import cloud, open_capture

OUT = Path(__file__).resolve().parents[1] / "out"


def main(path: str, name: str) -> None:
    cfg = load_config()
    ck = cfg["checks"]
    o3d.utility.random.seed(0)
    OUT.mkdir(exist_ok=True)
    print(f"==================== {name} ({path}) ====================")

    cap = open_capture(path, cfg)
    xyz = cap.odo[["x", "y", "z"]].values
    rng = np.ptp(xyz, axis=0)
    up = int(np.argmin(rng))
    print(f"frames {len(cap.odo)}, depth {cap.depth_hw}, K_depth fx,fy,cx,cy = {np.round([cap.K_depth[0,0], cap.K_depth[1,1], cap.K_depth[0,2], cap.K_depth[1,2]], 2)}")
    print("trajectory range (x,y,z) m:", np.round(rng, 3), f"-> smallest on '{'xyz'[up]}' (hypothesis; planes below confirm or reject)")

    pts = cloud(cap, cfg)
    print(f"points: {len(pts):,}; extent (x,y,z) m: {np.round(np.ptp(pts, axis=0), 2)}")

    rest = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    planes = []
    print("\n#  unit normal              d (n.x+d=0)   inliers   deg to nearest axis [axis]")
    for i in range(ck["planes_reported"]):
        model, inl = rest.segment_plane(ck["ransac_distance_m"], 3, ck["ransac_iterations"])
        n = np.array(model[:3]); s = np.linalg.norm(n); n /= s
        ang = np.degrees(np.arccos(np.clip(np.abs(n), 0, 1)))
        ax = int(np.argmin(ang))
        planes.append((n, model[3] / s, len(inl), ax, float(ang[ax])))
        print(f"{i}  {np.round(n, 3)!s:24} {model[3] / s:9.3f} {len(inl):9,}   {ang[ax]:6.2f} [{'xyz'[ax]}]")
        rest = rest.select_by_index(inl, invert=True)

    votes = {c: sum(k for _, _, k, ax, ang in planes if "xyz"[ax] == c and ang < 10) for c in "xyz"}
    print("inlier votes per axis (planes within 10 deg of an axis):", votes)

    near_up = [(-d * (1 if n[up] > 0 else -1), k, a) for n, d, k, ax, a in planes if ax == up and a < 10]
    print(f"planes near axis '{'xyz'[up]}': height along axis (m), inliers, deg:", [(round(float(h), 3), k, round(a, 2)) for h, k, a in near_up])
    if not near_up:
        print("no plane near the hypothesised up axis in the top planes; stopping")
        return
    floor_h = near_up[0][0]
    print(f"floor height (largest near-up plane): {floor_h:.3f} m; floor-like plane heights span {max(h for h, _, _ in near_up if h < floor_h + 0.3) - min(h for h, _, _ in near_up):.3f} m")

    hi = pts[pts[:, up] > floor_h + ck["ceiling_probe_min_above_floor_m"]]
    print(f"\nceiling probe: {len(hi):,} points >= {ck['ceiling_probe_min_above_floor_m']} m above floor ({100 * len(hi) / len(pts):.1f}% of cloud)")
    if len(hi) > 1000:
        hp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hi))
        model, inl = hp.segment_plane(ck["ransac_distance_m"], 3, 3000)
        n = np.array(model[:3]); n /= np.linalg.norm(n)
        hh = hi[inl][:, up]
        print(f"  best plane up there: {np.degrees(np.arccos(abs(n[up]))):.2f} deg from up, inliers {len(inl):,} ({100 * len(inl) / len(hi):.1f}%), "
              f"height above floor {np.median(hh) - floor_h:.3f} m (IQR {np.percentile(hh, 25) - floor_h:.3f}..{np.percentile(hh, 75) - floor_h:.3f})")
        lo = floor_h + ck["ceiling_probe_min_above_floor_m"]
        hist, e = np.histogram(hi[:, up], bins=np.arange(lo, hi[:, up].max() + 0.02, 0.02))
        pk = np.argsort(hist)[-5:][::-1]
        print("  top 2 cm bins (height above floor, count):", [(round(float(e[t] - floor_h), 2), int(hist[t])) for t in pk],
              "<- a peak at the lowest bin means a cutoff artifact, not a ceiling")

    ij = [a for a in range(3) if a != up]
    lo_h, hi_h = ck["wall_slab_above_floor_m"]
    slab = pts[(pts[:, up] > floor_h + lo_h) & (pts[:, up] < floor_h + hi_h)]
    sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slab)).voxel_down_sample(0.04)
    sp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=40))
    N = np.asarray(sp.normals)
    vert = np.abs(N[:, up]) < ck["wall_normal_vertical_max"]
    yaw = np.degrees(np.arctan2(N[vert, ij[1]], N[vert, ij[0]])) % 180
    h90, _ = np.histogram(yaw % 90, bins=90, range=(0, 90))
    sm = np.array([h90[np.arange(i - 3, i + 4) % 90].sum() for i in range(90)])
    pk = int(np.argmax(sm))
    d180 = (yaw - (pk + 0.5)) % 180
    near0 = int(((d180 < 3.5) | (d180 > 176.5)).sum()); near90 = int(((d180 > 86.5) & (d180 < 93.5)).sum())
    print(f"\nwall probe: slab {lo_h}..{hi_h} m above floor, {vert.sum():,} vertical-surface points")
    print(f"  wall-normal yaw (mod 90) peak {pk + 0.5:.1f} deg; {100 * sm[pk] / h90.sum():.1f}% within +-3.5 deg of it (uniform {100 * 7 / 90:.1f}%)")
    print(f"  facing the peak {100 * near0 / vert.sum():.1f}%, facing 90 deg to it {100 * near90 / vert.sum():.1f}%, elsewhere {100 * (vert.sum() - near0 - near90) / vert.sum():.1f}%")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist2d(pts[:, ij[0]], pts[:, ij[1]], bins=400, norm=matplotlib.colors.LogNorm())
    ax.set_aspect("equal"); ax.set_xlabel("xyz"[ij[0]]); ax.set_ylabel("xyz"[ij[1]])
    ax.set_title(f"{name}: top-down (axes {'xyz'[ij[0]]},{'xyz'[ij[1]]}; up='{'xyz'[up]}')")
    fig.savefig(OUT / f"density_{name}.png", dpi=100, bbox_inches="tight")
    print(f"\nsaved {OUT / f'density_{name}.png'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else Path(sys.argv[1]).parent.name)
