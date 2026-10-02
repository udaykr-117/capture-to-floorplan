"""Ceiling height per room, measured against that room's own floor (aligned frame, global floor at y' = 0)."""
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.signal import find_peaks
from shapely.geometry import Polygon


@dataclass
class Ceiling:
    status: str                         # 'measured' | 'ambiguous' | 'unmeasurable'
    reason: str | None = None
    height: float | None = None         # m above the room's local floor
    interval: tuple[float, float] | None = None
    floor_offset: float = 0.0           # local floor minus global floor (m)
    floor_local: bool = True            # False: too few floor points, global floor used
    coverage: float = 0.0               # fraction of the room area with ceiling points at the chosen level
    levels: list[tuple[float, float]] = field(default_factory=list)  # (height above local floor, coverage) of every level found


def _robust_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else 0.0


def measure_ceiling(poly: Polygon, Pa: np.ndarray, Na: np.ndarray, cfg: dict) -> Ceiling:
    c, s, r = cfg["ceiling"], cfg["scene"], cfg["rooms"]
    inside = shapely.contains_xy(poly, Pa[:, 0], Pa[:, 2])
    horiz = inside & (np.abs(Na[:, 1]) > s["horizontal_normal_min"])
    fl = horiz & (np.abs(Pa[:, 1]) < r["floor_band_m"])
    floor_local = int(fl.sum()) >= c["floor_min_points"]
    floor_y = float(np.median(Pa[fl, 1])) if floor_local else 0.0
    floor_sigma = _robust_sigma(Pa[fl, 1]) if floor_local else 0.0
    lo, hi = c["range_above_floor_m"]
    cand = horiz & (Pa[:, 1] > floor_y + lo) & (Pa[:, 1] < floor_y + hi)
    out = Ceiling("unmeasurable", floor_offset=floor_y, floor_local=floor_local)
    if cand.sum() < 50:
        out.reason = f"only {int(cand.sum())} horizontal points between {lo} and {hi} m above the floor"
        return out
    h = Pa[cand, 1]
    edges = np.arange(h.min() - 3 * c["hist_bin_m"], h.max() + 4 * c["hist_bin_m"], c["hist_bin_m"])  # margin bins: a noise-free ceiling at one height must still have a peak
    hist, _ = np.histogram(h, edges)
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(sm, height=max(5.0, 0.1 * sm.max()), distance=max(1, int(round(c["peak_tol_m"] / c["hist_bin_m"]))))
    cells_total = max(poly.area / c["cell_m"] ** 2, 1.0)
    xz = Pa[cand][:, [0, 2]]
    levels = []
    for p in peaks:
        y0 = edges[p] + c["hist_bin_m"] / 2
        near = np.abs(h - y0) < c["peak_tol_m"]
        ij = np.unique(np.floor(xz[near] / c["cell_m"]).astype(np.int64), axis=0)
        levels.append((float(np.median(h[near])), len(ij) / cells_total, float(_robust_sigma(h[near])), int(near.sum())))
    if not levels:
        out.reason = "no ceiling-height peak"
        return out
    levels.sort(key=lambda t: -t[1])
    out.levels = [(round(y - floor_y, 3), round(cov, 3)) for y, cov, _, _ in levels]
    y, cov, sig, n = levels[0]
    out.coverage = cov
    if cov < c["min_coverage"] or cov * poly.area < c["min_area_m2"]:
        out.reason = f"best level covers {100 * cov:.0f}% of the room ({cov * poly.area:.1f} m2), below the {100 * c['min_coverage']:.0f}% / {c['min_area_m2']} m2 minimum"
        return out
    w = c["interval_sigma"] * float(np.hypot(sig, floor_sigma))
    out.height = y - floor_y
    out.interval = (out.height - w, out.height + w)
    others = [(yy - floor_y, cc) for yy, cc, _, _ in levels[1:] if cc >= c["second_peak_rel"] * cov and abs(yy - y) > c["second_peak_sep_m"]]
    if others:
        allh = [out.height] + [o[0] for o in others]
        out.status, out.reason = "ambiguous", f"{len(others) + 1} ceiling levels with similar coverage: {[round(a, 2) for a in allh]}"
        out.interval = (min(min(allh), out.interval[0]), max(max(allh), out.interval[1]))
    else:
        out.status = "measured"
    return out
