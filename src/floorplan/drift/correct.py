"""Plane-anchored drift correction, one rigid correction per time chunk (aligned frame).

Assumptions (stated in the plan limitations): walls are rectilinear (Manhattan), the floor is flat, and drift is slow compared with a chunk.
Per chunk i: a rotation theta_i about the chunk's own camera centre (a heading error rotates what the sensor sees about the sensor; so its walls line up with the axes), a shift t_i (so a wall seen from the same side
at two different times lands at one offset), and a height shift dy_i (so the floor is level).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from floorplan.drift.chunks import ChunkAnalysis
from floorplan.geometry.walls import find_wall_planes
from floorplan.register import rot2


@dataclass
class Correction:
    pivot: np.ndarray      # (N, 2): per-chunk rotation centre = mean camera position of the chunk
    theta: np.ndarray      # radians
    t: np.ndarray          # (N, 2) metres, in x', z'
    dy: np.ndarray         # metres
    diag: dict = field(default_factory=dict)

    @staticmethod
    def identity(n: int) -> "Correction":
        return Correction(np.zeros((n, 2)), np.zeros(n), np.zeros((n, 2)), np.zeros(n))

    def apply(self, i: int, P: np.ndarray) -> np.ndarray:
        """Corrected copy of aligned-frame points P (n, 3) belonging to chunk i."""
        out = np.array(P, dtype=float, copy=True)
        xz = (P[:, [0, 2]] - self.pivot[i]) @ rot2(self.theta[i]).T + self.pivot[i] + self.t[i]
        out[:, 0], out[:, 2], out[:, 1] = xz[:, 0], xz[:, 1], P[:, 1] - self.dy[i]
        return out


def _fill(x: np.ndarray) -> np.ndarray:
    """Replace NaN by linear interpolation in chunk order (0 if nothing is valid)."""
    ok = ~np.isnan(x)
    if not ok.any():
        return np.zeros_like(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])


def _smooth(x: np.ndarray, size: int) -> np.ndarray:
    return median_filter(_fill(x), size=size, mode="nearest") if size > 1 else _fill(x)


def _side(cams: np.ndarray, axis: int, offset: float, s0: float, s1: float) -> int:
    b = 2 if axis == 0 else 0
    near = (cams[:, b] > s0 - 1.0) & (cams[:, b] < s1 + 1.0) & (np.abs(cams[:, axis] - offset) < 6.0)
    return 0 if not near.any() else int(np.sign(np.median(cams[near, axis] - offset)))


def chunk_planes(a: ChunkAnalysis, theta: float, pivot: np.ndarray, cfg: dict) -> list[dict]:
    """Wall segments of one chunk after rotating it by theta, with the side of the wall its cameras were on."""
    R = rot2(theta)
    P = np.array(a.Pa, copy=True)
    N = np.array(a.Na, copy=True)
    P[:, [0, 2]] = (a.Pa[:, [0, 2]] - pivot) @ R.T + pivot
    N[:, [0, 2]] = a.Na[:, [0, 2]] @ R.T
    cams = (a.cams_a[:, [0, 2]] - pivot) @ R.T + pivot
    c3 = np.c_[cams[:, 0], np.zeros(len(cams)), cams[:, 1]]
    out = []
    for p in find_wall_planes(P, N, cfg):
        for r in p.runs:
            side = _side(c3, p.axis, p.offset, r.s0, r.s1)
            if side:
                out.append(dict(axis=p.axis, offset=p.offset, s0=r.s0, s1=r.s1, side=side, chunk=a.idx))
    return out


def associate(segs: list[dict], cfg: dict) -> list[tuple[int, int, int, float, float]]:
    """(chunk i, chunk j, axis, offset_i - offset_j, weight) for same-axis, same-side, overlapping segments of different chunks."""
    d = cfg["drift"]
    out = []
    by_chunk: dict[int, list[dict]] = {}
    for s in segs:
        by_chunk.setdefault(s["chunk"], []).append(s)
    ids = sorted(by_chunk)
    for ii, i in enumerate(ids):
        for j in ids[ii + 1:]:
            cand = {}
            for x, a in enumerate(by_chunk[i]):
                best = None
                for y, b in enumerate(by_chunk[j]):
                    ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
                    if a["axis"] == b["axis"] and a["side"] == b["side"] and ov >= d["assoc_min_overlap_m"] and abs(a["offset"] - b["offset"]) <= d["assoc_tol_m"]:
                        if best is None or abs(a["offset"] - b["offset"]) < abs(a["offset"] - by_chunk[j][best]["offset"]):
                            best = y
                if best is not None:
                    cand[x] = best
            back = {}
            for y, b in enumerate(by_chunk[j]):
                best = None
                for x, a in enumerate(by_chunk[i]):
                    ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
                    if a["axis"] == b["axis"] and a["side"] == b["side"] and ov >= d["assoc_min_overlap_m"] and abs(a["offset"] - b["offset"]) <= d["assoc_tol_m"]:
                        if best is None or abs(a["offset"] - b["offset"]) < abs(by_chunk[i][best]["offset"] - b["offset"]):
                            best = x
                if best is not None:
                    back[y] = best
            for x, y in cand.items():
                if back.get(y) == x:
                    a, b = by_chunk[i][x], by_chunk[j][y]
                    ov = min(a["s1"], b["s1"]) - max(a["s0"], b["s0"])
                    out.append((i, j, a["axis"], a["offset"] - b["offset"], min(ov, 3.0) / 3.0))
    return out


def solve_shifts(n: int, assoc: list, cfg: dict) -> tuple[np.ndarray, dict]:
    """Least squares for the per-chunk shifts t (n, 2): matched walls coincide, neighbours move together, weak pull to zero. Huber IRLS."""
    d = cfg["drift"]
    rows, rhs, base_w = [], [], []
    for i, j, axis, diff, w in assoc:
        r = np.zeros(2 * n)
        r[2 * i + (0 if axis == 0 else 1)] = 1.0     # axis 0 -> x' component, axis 2 -> z' component
        r[2 * j + (0 if axis == 0 else 1)] = -1.0
        rows.append(r)
        rhs.append(-diff)                             # (o_i + t_i) - (o_j + t_j) = 0  ->  t_i - t_j = o_j - o_i = -(o_i - o_j)
        base_w.append(w)
    n_assoc = len(rows)
    for i in range(n - 1):
        for c in range(2):
            r = np.zeros(2 * n)
            r[2 * i + c], r[2 * (i + 1) + c] = -1.0, 1.0
            rows.append(r)
            rhs.append(0.0)
            base_w.append(d["smooth_weight"])
    for i in range(2 * n):
        r = np.zeros(2 * n)
        r[i] = 1.0
        rows.append(r)
        rhs.append(0.0)
        base_w.append(d["prior_weight"])
    A, b, w0 = np.array(rows), np.array(rhs), np.array(base_w)
    w = w0.copy()
    x = np.zeros(2 * n)
    for _ in range(d["irls_iters"]):
        x, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        res = A @ x - b
        w = w0 * np.minimum(1.0, d["huber_m"] / np.maximum(np.abs(res), 1e-9))
    res = (A @ x - b)[:n_assoc]
    diag = dict(n_assoc=n_assoc, rms_before_cm=float(np.sqrt(np.mean(np.array(rhs[:n_assoc]) ** 2)) * 100) if n_assoc else 0.0,
                rms_after_cm=float(np.sqrt(np.mean(res**2)) * 100) if n_assoc else 0.0,
                inliers=int((np.abs(res) <= d["huber_m"]).sum()) if n_assoc else 0)
    return x.reshape(n, 2), diag


def estimate(analyses: list[ChunkAnalysis], cfg: dict) -> Correction:
    d = cfg["drift"]
    n = len(analyses)
    pivot = np.array([a.cams_a[:, [0, 2]].mean(0) for a in analyses])
    dev = np.array([a.yaw_dev_deg for a in analyses])
    theta_deg = -np.clip(_smooth(dev, d["yaw_median_chunks"]), -d["max_yaw_deg"], d["max_yaw_deg"])
    if not d["use_yaw"]:
        theta_deg = np.zeros_like(theta_deg)
    theta = np.radians(theta_deg)
    segs = [s for a in analyses for s in chunk_planes(a, theta[a.idx], pivot[a.idx], cfg)]
    assoc = associate(segs, cfg)
    t, diag = solve_shifts(n, assoc, cfg)
    diag.update(_holdout(n, assoc, cfg))
    if not d["use_shift"]:
        t = np.zeros_like(t)
    floor = np.array([a.floor_y for a in analyses])
    dy = np.clip(_smooth(floor, d["floor_median_chunks"]), -d["max_floor_m"], d["max_floor_m"])
    if not d["use_floor"]:
        dy = np.zeros_like(dy)
    diag.update(yaw_dev_deg=dev.tolist(), theta_deg=theta_deg.tolist(), floor_cm=(floor * 100).tolist(), dy_cm=(dy * 100).tolist(),
                shift_cm=(t * 100).tolist(), n_segments=len(segs))
    return Correction(pivot, theta, t, dy, diag)


def _holdout(n: int, assoc: list, cfg: dict) -> dict:
    """Solve again without a random fifth of the associations and measure how well those unseen ones are explained."""
    if len(assoc) < 10:
        return {}
    rng = np.random.default_rng(0)
    held = rng.random(len(assoc)) < cfg["drift"]["holdout_fraction"]
    t, _ = solve_shifts(n, [a for a, h in zip(assoc, held) if not h], cfg)
    before, after = [], []
    for (i, j, axis, diff, _), h in zip(assoc, held):
        if h:
            c = 0 if axis == 0 else 1
            before.append(diff)
            after.append(diff + t[i, c] - t[j, c])
    return dict(holdout_n=int(held.sum()), holdout_before_cm=float(np.sqrt(np.mean(np.square(before))) * 100), holdout_after_cm=float(np.sqrt(np.mean(np.square(after))) * 100))
