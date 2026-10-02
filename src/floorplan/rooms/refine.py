"""Room-local wall refinement (M7 fix for the repeatability gate).

Diagnosis (JOURNAL M7): a wall's position seen chunk by chunk moves by a robust 3.5 cm (range ~14 cm) over a capture, so a plane fitted to all of
its points is an average of positions from different times, and two facing walls fitted that way inherit the drift between the times they were
seen. Here each room's plane-backed edges are re-measured from ONE visit: the longest run of consecutive chunks whose cameras were inside the room.
Within a visit (tens of seconds) drift is small, so the room's own walls are mutually consistent. Rooms are refined independently, so a wall shared
by two rooms can move by a few cm differently for each (reported as overlap in the stitched plan).
"""
from dataclasses import replace

import numpy as np
import shapely
from shapely.geometry import Polygon

from floorplan.geometry.walls import WallPlane
from floorplan.rooms.polygon import RoomPoly


def _visit(inside: list[bool]) -> list[int]:
    """Indices of the longest run of consecutive True values."""
    best, cur = [], []
    for k, v in enumerate(inside):
        cur = cur + [k] if v else []
        if len(cur) > len(best):
            best = cur
    return best


def _runs(inside: list[bool]) -> list[list[int]]:
    """All runs of consecutive True values (visits), in time order."""
    out, cur = [], []
    for k, v in enumerate(inside):
        if v:
            cur.append(k)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _mode_offset(c: np.ndarray, centre: float, win: float) -> tuple[float, float] | None:
    e = np.arange(centre - win, centre + win + 1e-9, 0.01)
    h = np.convolve(np.histogram(c, e)[0], np.ones(3), mode="same")      # 3 cm smoothing of 1 cm bins
    pk = e[int(np.argmax(h))] + 0.005
    near = c[np.abs(c - pk) < 0.015]
    if len(near) < 10:
        return None
    med = float(np.median(near))
    return med, float(1.4826 * np.median(np.abs(near - med)))


def refine_rooms(polys: list[RoomPoly], chunks: list, fr, corr, cfg: dict) -> tuple[list[RoomPoly], list[WallPlane], list[dict]]:
    """-> (refined room polygons, interval planes for the refined lines (axis, offset, local sigma), per-room report)."""
    rc, w = cfg["refine"], cfg["walls"]
    lo, hi = w["slab_above_floor_m"]
    cams = [corr.apply(c.idx, fr.to_aligned(c.cams)) for c in chunks]
    cache: dict[int, np.ndarray] = {}

    def pts(k: int) -> np.ndarray:
        if k not in cache:
            cache[k] = corr.apply(chunks[k].idx, fr.to_aligned(chunks[k].pts))
        return cache[k]

    out, extra, report = [], [], []
    for rp in polys:
        inside = [float(shapely.contains_xy(rp.polygon, c[:, 0], c[:, 2]).mean()) >= rc["chunk_inside_min"] for c in cams]
        visit = _visit(inside)
        if len(visit) < rc["min_visit_chunks"]:
            out.append(rp)
            report.append(dict(room=f"R{int(rp.id)}", refined=0, reason=f"longest visit {len(visit)} chunk(s) < {rc['min_visit_chunks']}"))
            continue
        # 'longest': walls from the longest visit only. 'median': each wall measured in every visit of >= min_visit_chunks chunks, median taken,
        # so one visit's own error (visits of one room differ by up to ~7 cm, scripts/refine_visits.py) cannot decide a wall alone.
        visits = [visit] if rc.get("visits", "longest") == "longest" else [v for v in _runs(inside) if len(v) >= rc["min_visit_chunks"]]
        clouds = []
        for v in visits:
            P = np.concatenate([pts(k) for k in v])
            clouds.append(P[(P[:, 1] > lo) & (P[:, 1] < hi)])
        maps: dict[int, dict[float, float]] = {0: {}, 2: {}}
        moves = []
        for e in rp.edges:
            if e.source != "plane" or e.offset in maps[e.axis]:
                continue
            along = 2 if e.axis == 0 else 0
            found, n_pts, spreads = [], 0, []
            for P in clouds:
                m = (np.abs(P[:, e.axis] - e.offset) < rc["window_m"]) & (P[:, along] > e.s0 + rc["end_margin_m"]) & (P[:, along] < e.s1 - rc["end_margin_m"])
                if m.sum() < rc["min_points"]:
                    continue
                r = _mode_offset(P[m, e.axis], e.offset, rc["window_m"])
                if r is not None:
                    found.append(r[0]); spreads.append(r[1]); n_pts += int(m.sum())
            if not found:
                continue
            new = float(np.median(found))
            sig = float(max(np.median(spreads), 1.4826 * np.median(np.abs(np.array(found) - new)))) if len(found) > 1 else spreads[0]
            maps[e.axis][e.offset] = new
            extra.append(WallPlane(e.axis, new, n_pts, sigma=sig))
            moves.append(new - e.offset)

        def remap(v: float, axis: int) -> float:
            for old, new in maps[axis].items():
                if abs(v - old) < 1e-9:
                    return new
            return v

        poly = Polygon([(remap(x, 0), remap(z, 2)) for x, z in rp.polygon.exterior.coords])
        if not poly.is_valid or poly.area <= 0:
            out.append(rp)
            report.append(dict(room=f"R{int(rp.id)}", refined=0, reason="refined polygon invalid; kept the original"))
            continue
        edges = [replace(e, offset=remap(e.offset, e.axis), s0=remap(e.s0, 2 if e.axis == 0 else 0), s1=remap(e.s1, 2 if e.axis == 0 else 0)) for e in rp.edges]
        out.append(RoomPoly(rp.id, poly, edges))
        report.append(dict(room=f"R{int(rp.id)}", refined=len(moves), visit_chunks=len(visit), visit=[visit[0], visit[-1]],
                           median_move_cm=round(float(np.median(np.abs(moves))) * 100, 1) if moves else 0.0,
                           max_move_cm=round(float(np.max(np.abs(moves))) * 100, 1) if moves else 0.0))
    return out, extra, report
