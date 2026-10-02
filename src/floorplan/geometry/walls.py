"""Axis-aligned wall planes in the aligned frame (Y up, walls along x' and z').

A WallPlane with axis=0 is the plane x' = offset (normal along x'), extending along z'; axis=2 is z' = offset.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Run:
    s0: float
    s1: float
    n_points: int
    coverage: float  # fraction of slab height filled, averaged over supported columns

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass
class WallPlane:
    axis: int
    offset: float
    n_points: int
    runs: list[Run] = field(default_factory=list)
    sigma: float = 0.0  # robust spread (m) of the wall points around the plane: noise plus drift smear


def _runs(occupied: np.ndarray, edges: np.ndarray, gap_cells: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(occupied)
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > gap_cells:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def find_wall_planes(Pa: np.ndarray, Na: np.ndarray, cfg: dict) -> list[WallPlane]:
    w, s = cfg["walls"], cfg["scene"]
    lo, hi = w["slab_above_floor_m"]
    horiz = np.hypot(Na[:, 0], Na[:, 2])
    sel = (np.abs(Na[:, 1]) < s["vertical_normal_max"]) & (Pa[:, 1] > lo) & (Pa[:, 1] < hi) & (horiz > 1e-6)
    cos_min = np.cos(np.radians(w["normal_axis_max_deg"]))
    min_pts = w["peak_min_support_m2"] / s["normal_voxel_m"] ** 2
    planes: list[WallPlane] = []
    for a, b in ((0, 2), (2, 0)):
        m = sel & (np.abs(Na[:, a]) / np.maximum(horiz, 1e-9) > cos_min)
        c, t, h = Pa[m, a], Pa[m, b], Pa[m, 1]
        if len(c) == 0:
            continue
        edges = np.arange(c.min() - w["hist_bin_m"], c.max() + 2 * w["hist_bin_m"], w["hist_bin_m"])
        hist, _ = np.histogram(c, edges)
        sm = np.convolve(hist, np.ones(3) / 3, mode="same")
        dist = max(1, int(round(w["plane_min_separation_m"] / w["hist_bin_m"])))
        peaks, _ = find_peaks(sm, height=min_pts, distance=dist)
        for p in peaks:
            near = np.abs(c - (edges[p] + w["hist_bin_m"] / 2)) < w["plane_tol_m"]
            plane = WallPlane(a, float(np.median(c[near])), int(near.sum()), sigma=float(1.4826 * np.median(np.abs(c[near] - np.median(c[near])))))
            tn, hn = t[near], h[near]
            cell = w["along_cell_m"]
            e = np.arange(tn.min() - cell, tn.max() + 2 * cell, cell)
            cnt, _ = np.histogram(tn, e)
            for i0, i1 in _runs(cnt >= w["along_cell_min_points"], e, int(round(w["gap_merge_m"] / cell))):
                s0, s1 = float(e[i0]), float(e[i1 + 1])
                if s1 - s0 < w["min_run_length_m"]:
                    continue
                inr = (tn >= s0) & (tn <= s1)
                cc = w["coverage_cell_m"]
                H, _, _ = np.histogram2d(tn[inr], hn[inr], bins=[np.arange(s0, s1 + cc, cc), np.arange(lo, hi + cc, cc)])
                cols = H.sum(1) > 0
                cov = float(((H[cols] > 0).mean(1)).mean()) if cols.any() else 0.0
                plane.runs.append(Run(s0, s1, int(inr.sum()), cov))
            if plane.runs:
                planes.append(plane)
    return planes


def structural(planes: list[WallPlane], cfg: dict) -> list[WallPlane]:
    """Planes restricted to runs whose height coverage says 'wall', not low furniture."""
    mc = cfg["walls"]["min_height_coverage"]
    out = []
    for p in planes:
        runs = [r for r in p.runs if r.coverage >= mc]
        if runs:
            out.append(WallPlane(p.axis, p.offset, p.n_points, runs, p.sigma))
    return out
