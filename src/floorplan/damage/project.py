"""Detection boxes -> patches on room surfaces (wall edge, ceiling, floor) -> merged regions with a metric extent.

A box is projected through the keyframe's depth: its depth pixels are back-projected into the corrected plan frame, a plane is fitted to
them, and only pixels on that plane count (the rest is background seen through the box). The patch lives on one surface of one room; the same
patch seen in several keyframes lands in the same raster cells of that surface and is merged there. Area = occupied cells x cell^2.
"""
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy import ndimage

from floorplan.damage.detect import Detection


@dataclass
class Patch:
    det: Detection
    kind: str                 # wall | ceiling | floor
    surface_id: str
    room_id: str
    uv: np.ndarray            # (n, 2) surface coordinates of the on-plane pixels
    xz: np.ndarray            # (n, 2) plan-frame position of the same pixels


def fit_plane(P: np.ndarray, tol: float) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares plane, refitted once on its inliers. Returns (unit normal, offset d with n.p = d, inlier mask)."""
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    for _ in range(2):
        d = float(n @ c)
        inl = np.abs(P @ n - d) < tol
        if inl.sum() < 10:
            break
        c = P[inl].mean(0)
        n = np.linalg.svd(P[inl] - c, full_matrices=False)[2][-1]
    d = float(n @ c)
    return n, d, np.abs(P @ n - d) < tol


def box_points(det: Detection, depth_m: np.ndarray, conf: np.ndarray, K: np.ndarray, rgb_wh: tuple[int, int], cfg: dict):
    """Camera-frame points and per-pixel z of the confident depth pixels inside the box."""
    s = cfg["stray"]
    sx, sy = depth_m.shape[1] / rgb_wh[0], depth_m.shape[0] / rgb_wh[1]
    x0, y0, x1, y1 = det.box
    c0, c1 = int(np.floor(x0 * sx)), int(np.ceil(x1 * sx))
    r0, r1 = int(np.floor(y0 * sy)), int(np.ceil(y1 * sy))
    sub_d, sub_c = depth_m[r0:r1, c0:c1], conf[r0:r1, c0:c1]
    v, u = np.mgrid[r0:r1, c0:c1]
    m = (sub_d > s["depth_min_m"]) & (sub_d < s["depth_max_m"]) & (sub_c >= s["confidence_min"])
    z = sub_d[m]
    return np.stack([(u[m] - K[0, 2]) * z / K[0, 0], (v[m] - K[1, 2]) * z / K[1, 1], z], 1)


def project_detection(det: Detection, p_cam: np.ndarray, to_plan, cam_plan: np.ndarray, K: np.ndarray, rooms: list, cfg: dict):
    """-> (Patch | None, reason). `to_plan` maps camera-frame points to the corrected plan frame (x', y' above the global floor, z').
    `rooms` = [(room id, shapely polygon, floor offset, [(wall id, axis, offset, s0, s1), ...])]."""
    d = cfg["damage"]
    if len(p_cam) < 30:
        return None, "too few confident depth pixels in the box"
    P = to_plan(p_cam)
    n, off, inl = fit_plane(P, d["plane_inlier_m"])
    if inl.mean() < d["min_plane_fraction"]:
        return None, "box is not on one plane"
    Pi, ci = P[inl], p_cam[inl]
    n = n if n @ (cam_plan - Pi.mean(0)) > 0 else -n          # normal points toward the camera
    med = np.median(Pi, 0)
    ny = abs(n[1])
    if ny >= d["horizontal_normal_min"]:
        kind = "ceiling" if n[1] < 0 else "floor"             # a ceiling faces down
    elif ny <= d["vertical_normal_max"]:
        kind = "wall"
    else:
        return None, "slanted surface"
    if kind not in d["surfaces"]:
        return None, f"surface kind '{kind}' not reported"
    if kind == "wall":
        h = n[[0, 2]] / np.linalg.norm(n[[0, 2]])
        probe = med[[0, 2]] + 0.12 * h                        # step off the wall toward the camera to find the room it belongs to
    else:
        probe = med[[0, 2]]
    room = next((r for r in rooms if shapely.contains_xy(r[1], probe[0], probe[1])), None)
    if room is None:
        return None, "not inside any room polygon"
    rid, _, floor_off, edges = room
    y_rel = med[1] - floor_off
    if kind != "wall":
        if kind == "floor" and y_rel > 0.5:
            return None, "horizontal surface above the floor (furniture)"
        if kind == "ceiling" and y_rel < 1.8:
            return None, "horizontal surface below ceiling height (furniture)"
        sid = f"{rid}.{kind}"
        uv = Pi[:, [0, 2]]
    else:
        axis = 0 if abs(n[0]) > abs(n[2]) else 2
        along = 2 if axis == 0 else 0
        e = [(wid, o, s0, s1) for wid, a, o, s0, s1 in edges if a == axis and abs(o - med[axis]) <= d["wall_match_m"]
             and s0 - 0.1 <= med[along] <= s1 + 0.1]
        if not e:
            return None, "no room wall edge at this plane"
        sid = min(e, key=lambda t: abs(t[1] - med[axis]))[0]
        uv = np.stack([Pi[:, along], Pi[:, 1] - floor_off], 1)
    return Patch(det, kind, sid, rid, uv, Pi[:, [0, 2]]), "ok"


@dataclass
class Region:
    cls: str
    kind: str
    surface_id: str
    room_id: str
    cells: int
    cells_low: int
    cells_high: int
    extent: tuple[float, float, float, float]
    xz: tuple[float, float]
    score_max: float
    frames: list[int]
    best: Detection = field(default=None)


def merge_patches(patches: list[Patch], cfg: dict) -> tuple[list[Region], list[str]]:
    """Group patches by (surface, class), rasterise on the surface, connected components = regions. Components seen in fewer than
    `min_views` keyframes are dropped (returned as reasons)."""
    cell = cfg["damage"]["cell_m"]
    groups: dict[tuple, list[Patch]] = {}
    for p in patches:
        groups.setdefault((p.surface_id, p.det.cls), []).append(p)
    regions, dropped = [], []
    for (sid, cls), ps in groups.items():
        allq = np.concatenate([p.uv for p in ps])
        o = np.floor(allq.min(0) / cell).astype(int) - 2
        shape = tuple(np.ceil(allq.max(0) / cell).astype(int) - o + 3)
        grids = []
        for p in ps:
            g = np.zeros(shape, bool)
            ij = np.floor(p.uv / cell).astype(int) - o
            g[ij[:, 0], ij[:, 1]] = True
            grids.append(ndimage.binary_closing(g, iterations=1))    # depth pixels are sparser than cells at long range: close gaps between them
        union = np.any(grids, axis=0)
        lab, n = ndimage.label(ndimage.binary_dilation(union, iterations=1))   # patches that touch within a cell are one region
        for k in range(1, n + 1):
            comp = (lab == k) & union
            members = [p for p, g in zip(ps, grids) if (g & comp).any()]
            frames = sorted({p.det.frame for p in members})
            if len(frames) < cfg["damage"]["min_views"]:
                dropped.append(f"{sid} {cls}: seen in {len(frames)} keyframe(s), needs {cfg['damage']['min_views']}")
                continue
            ij = np.argwhere(comp)
            u0, v0 = (ij.min(0) + o) * cell
            u1, v1 = (ij.max(0) + 1 + o) * cell
            cap = cfg["damage"]["max_extent"][cls]
            size = max(u1 - u0, v1 - v0) if cls == "crack" else int(comp.sum()) * cell**2
            if size > cap:
                dropped.append(f"{sid} {cls}: extent {size:.2f} exceeds the plausible maximum {cap}")
                continue
            xz = np.concatenate([p.xz for p in members])
            best = max((p.det for p in members), key=lambda d: d.score)
            regions.append(Region(cls, ps[0].kind, sid, ps[0].room_id, int(comp.sum()), int(ndimage.binary_erosion(comp).sum()),
                                  int(ndimage.binary_dilation(comp).sum()), (float(u0), float(u1), float(v0), float(v1)),
                                  (float(np.median(xz[:, 0])), float(np.median(xz[:, 1]))), best.score, frames, best))
    return regions, dropped
