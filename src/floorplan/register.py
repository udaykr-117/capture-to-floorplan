"""Register one capture onto another (both in their aligned frames) and compare them.

The registration is constrained to the floor plane: a rotation and a 2D shift (the floors already agree in height).
Comparison metrics that do not depend on the fitted shift: distances between facing wall planes.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import Polygon


def wall_points(Pa: np.ndarray, Na: np.ndarray, cfg: dict) -> np.ndarray:
    """(x', z') of vertical-surface points in the wall slab."""
    lo, hi = cfg["walls"]["slab_above_floor_m"]
    m = (np.abs(Na[:, 1]) < cfg["scene"]["vertical_normal_max"]) & (Pa[:, 1] > lo) & (Pa[:, 1] < hi)
    return Pa[m][:, [0, 2]]


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, mq - R @ mp


@dataclass
class Registration:
    R: np.ndarray
    t: np.ndarray
    rot_k: int
    yaw_resid_deg: float
    fitness_5cm: float
    median_cm: float
    rmse_cm: float
    null_fitness_5cm: float
    null_median_cm: float
    accepted: bool

    def apply(self, xz: np.ndarray) -> np.ndarray:
        return xz @ self.R.T + self.t


def _raster(xz: np.ndarray, lo: np.ndarray, n: int, cell: float) -> np.ndarray:
    img = np.zeros((n, n), np.float32)
    idx = np.floor((xz - lo) / cell).astype(int)
    ok = (idx >= 0).all(1) & (idx < n).all(1)
    img[idx[ok, 0], idx[ok, 1]] = 1
    return img


def _coarse(A: np.ndarray, B: np.ndarray, cfg: dict) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    """For each of the four 90-degree rotations: (score, k, R, t) from a raster cross-correlation."""
    cell = cfg["register"]["raster_cell_m"]
    ca, cb = A.mean(0), B.mean(0)
    half = max(np.ptp(A, 0).max(), np.ptp(B, 0).max()) + 2.0
    n = int(np.ceil(2 * half / cell)) | 1
    lo = ca - n * cell / 2
    IA = _raster(A, lo, n, cell)
    out = []
    for k in range(4):
        Rk = rot2(k * np.pi / 2)
        IB = _raster((B - cb) @ Rk.T + ca, lo, n, cell)
        corr = correlate(IA, IB, mode="same", method="fft")
        i, j = np.unravel_index(np.argmax(corr), corr.shape)
        shift = (np.array([i, j]) - (n - 1) // 2) * cell
        out.append((float(corr[i, j] / max(IB.sum(), 1)), k, Rk, ca + shift - Rk @ cb))
    return out


def _icp(B: np.ndarray, tree: cKDTree, A: np.ndarray, R: np.ndarray, t: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    r = cfg["register"]
    for thr in r["icp_thresholds_m"]:
        for _ in range(r["icp_max_iter"]):
            s = B @ R.T + t
            d, idx = tree.query(s, distance_upper_bound=thr)
            m = np.isfinite(d)
            if m.sum() < 50:
                break
            dR, dt = _kabsch(s[m], A[idx[m]])
            R, t = dR @ R, dR @ t + dt
            if abs(np.arctan2(dR[1, 0], dR[0, 0])) < 1.7e-5 and np.linalg.norm(dt) < 1e-4:
                break
    return R, t


def _stats(B: np.ndarray, tree: cKDTree, R: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    d, _ = tree.query(B @ R.T + t)
    near = d < 0.05
    return float(near.mean()), float(np.median(d) * 100), float(np.sqrt((d[near] ** 2).mean()) * 100) if near.any() else float("nan")


def register(A: np.ndarray, B: np.ndarray, cfg: dict, seed: int = 0) -> Registration:
    """Move B's wall points (x', z') onto A's. A and B are (N, 2) arrays."""
    r = cfg["register"]
    rng = np.random.default_rng(seed)
    Bs = B[rng.choice(len(B), min(len(B), r["icp_max_points"]), replace=False)]
    tree = cKDTree(A)
    coarse = _coarse(A, Bs, cfg)
    score, k, R0, t0 = max(coarse, key=lambda c: c[0])
    R, t = _icp(Bs, tree, A, R0, t0, cfg)
    fit, med, rmse = _stats(Bs, tree, R, t)
    _, kn, Rn0, tn0 = next(c for c in coarse if c[1] == (k + 2) % 4)
    Rn, tn = _icp(Bs, tree, A, Rn0, tn0, cfg)
    nfit, nmed, _ = _stats(Bs, tree, Rn, tn)
    th = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    resid = (th - 90 * k + 45) % 90 - 45
    return Registration(R, t, k, float(resid), fit, med, rmse, nfit, nmed, fit >= r["min_fitness_5cm"])


# ---- wall segments and widths --------------------------------------------------------------------------------

@dataclass
class Seg:
    axis: int
    offset: float
    s0: float
    s1: float
    sigma: float
    tilt_deg: float = 0.0


def segments(planes) -> list[Seg]:
    return [Seg(p.axis, p.offset, r.s0, r.s1, p.sigma) for p in planes for r in p.runs]


def to_other(seg: Seg, reg: Registration) -> Seg:
    q = reg.apply(np.array([[seg.offset, seg.s0], [seg.offset, seg.s1]]) if seg.axis == 0 else np.array([[seg.s0, seg.offset], [seg.s1, seg.offset]]))
    d = q[1] - q[0]
    if abs(d[0]) >= abs(d[1]):
        axis, off, s = 2, float(q[:, 1].mean()), np.sort(q[:, 0])
        tilt = np.degrees(np.arctan2(abs(d[1]), abs(d[0])))
    else:
        axis, off, s = 0, float(q[:, 0].mean()), np.sort(q[:, 1])
        tilt = np.degrees(np.arctan2(abs(d[0]), abs(d[1])))
    return Seg(axis, off, float(s[0]), float(s[1]), seg.sigma, float(tilt))


def _overlap(a: Seg, b: Seg) -> float:
    return min(a.s1, b.s1) - max(a.s0, b.s0)


def match_segments(SA: list[Seg], SB_in_A: list[Seg], cfg: dict) -> list[tuple[int, int]]:
    """Mutual best matches (index in A, index in B): same axis, offset within tolerance, enough overlap; smallest offset difference wins."""
    r = cfg["register"]

    def best(src: list[Seg], dst: list[Seg]) -> dict[int, int]:
        out = {}
        for i, a in enumerate(src):
            cands = [(abs(a.offset - b.offset), -_overlap(a, b), j) for j, b in enumerate(dst)
                     if a.axis == b.axis and abs(a.offset - b.offset) <= r["plane_match_tol_m"] and _overlap(a, b) >= r["plane_min_overlap_m"]]
            if cands:
                out[i] = min(cands)[2]
        return out

    ab, ba = best(SA, SB_in_A), best(SB_in_A, SA)
    return [(i, j) for i, j in ab.items() if ba.get(j) == i]


def width_pairs(SA: list[Seg], SB: list[Seg], matches: list[tuple[int, int]], cfg: dict) -> list[dict]:
    """Distance between two facing matched planes in A versus in B. Independent of the registration shift."""
    r = cfg["register"]
    out = []
    for x in range(len(matches)):
        for y in range(x + 1, len(matches)):
            (ia, ib), (ja, jb) = matches[x], matches[y]
            a1, a2, b1, b2 = SA[ia], SA[ja], SB[ib], SB[jb]
            if a1.axis != a2.axis:
                continue
            if a1.offset > a2.offset:
                a1, a2, b1, b2 = a2, a1, b2, b1
            wa = a2.offset - a1.offset
            if not (r["baseline_min_m"] <= wa <= r["baseline_max_m"]):
                continue
            if _overlap(a1, a2) < r["plane_min_overlap_m"] or _overlap(b1, b2) < r["plane_min_overlap_m"]:
                continue
            wb = abs(b2.offset - b1.offset)
            out.append(dict(axis=a1.axis, a_lo=a1.offset, a_hi=a2.offset, width_a=wa, width_b=wb, delta=wb - wa,
                            sig_a=float(np.hypot(a1.sigma, a2.sigma)), sig_b=float(np.hypot(b1.sigma, b2.sigma))))
    return out


def summarize_widths(pairs: list[dict], cfg: dict) -> list[dict]:
    r = cfg["register"]
    edges = r["baseline_bins_m"]
    rows = []
    for lo, hi in list(zip(edges[:-1], edges[1:])) + [(edges[0], edges[-1])]:
        sel = [p for p in pairs if lo <= p["width_a"] < hi or (hi == edges[-1] and p["width_a"] == hi)]
        if not sel:
            rows.append(dict(lo=lo, hi=hi, n=0))
            continue
        d = np.array([p["delta"] for p in sel])
        w = np.array([p["width_a"] for p in sel])
        ok_abs, ok_rel = np.abs(d) <= r["gate_abs_m"], np.abs(d) <= r["gate_rel"] * w
        rows.append(dict(lo=lo, hi=hi, n=len(sel), median_abs_cm=float(np.median(np.abs(d)) * 100), p90_abs_cm=float(np.percentile(np.abs(d), 90) * 100),
                         median_signed_pct=float(np.median(d / w) * 100), within_1cm=float(ok_abs.mean()), within_0p5pct=float(ok_rel.mean()),
                         within_either=float((ok_abs | ok_rel).mean())))
    return rows


# ---- rooms and ceilings --------------------------------------------------------------------------------------

def to_other_polygon(poly: Polygon, reg: Registration) -> Polygon:
    R, t = reg.R, reg.t
    return affinity.affine_transform(poly, [R[0, 0], R[0, 1], R[1, 0], R[1, 1], t[0], t[1]])


def fuse_ceilings(plan_b, plan_a, reg: Registration, cfg: dict):
    """Give rooms of capture B that have no ceiling the ceiling of the room of capture A they lie inside (B moved into A's frame).

    A room qualifies only if >= `room_inside_fraction` of it lies inside one room of A that has a ceiling value.
    Returns (new plan, list of what was copied). The copied Measurement says which capture and room it came from.
    """
    frac = cfg["register"]["room_inside_fraction"]
    polys_a = {r.id: Polygon(r.polygon) for r in plan_a.rooms}
    by_id = {r.id: r for r in plan_a.rooms}
    copied, rooms = [], []
    for rb in plan_b.rooms:
        new = rb
        if rb.ceiling_height.value is None:
            tb = to_other_polygon(Polygon(rb.polygon), reg)
            cand = [(tb.intersection(p).area / tb.area, rid) for rid, p in polys_a.items()]
            f, rid = max(cand) if cand else (0.0, None)
            ra = by_id.get(rid)
            if rid and f >= frac and ra.ceiling_height.value is not None:
                m = ra.ceiling_height.model_copy(update=dict(
                    source=f"{plan_a.capture} {rid}",
                    note=f"ceiling taken from capture {plan_a.capture} room {rid}: {100 * f:.0f}% of this room lies inside it (registration fitness {100 * reg.fitness_5cm:.0f}% within 5 cm). Not measured in this capture."))
                new = rb.model_copy(update=dict(ceiling_height=m))
                copied.append(dict(room=rb.id, from_room=rid, inside=f, height=m.value, status=m.status))
        rooms.append(new)
    limits = list(plan_b.limitations)
    if copied:
        limits.append(f"Ceiling heights of {[c['room'] for c in copied]} come from capture {plan_a.capture} via registration, not from this capture.")
    return plan_b.model_copy(update=dict(rooms=rooms, limitations=limits)), copied


def match_rooms(polysA: dict, polysB: dict, reg: Registration) -> list[dict]:
    """Room overlaps (ids -> polygons) between capture A and B moved into A's frame."""
    out = []
    for jb, pb in polysB.items():
        tb = to_other_polygon(pb, reg)
        for ia, pa in polysA.items():
            inter = tb.intersection(pa).area
            if inter > 0.05 * min(tb.area, pa.area):
                out.append(dict(a=ia, b=jb, area_a=pa.area, area_b=tb.area, inter=inter, iou=inter / (pa.area + tb.area - inter),
                                b_in_a=inter / tb.area, a_in_b=inter / pa.area))
    return sorted(out, key=lambda r: (r["b"], -r["iou"]))
