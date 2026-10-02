"""Doors and other openings in wall planes, from see-through evidence (aligned frame).

For each wall plane a 2 cm image (along the wall x height) counts wall points and camera rays that crossed the plane and hit
something at least `through_beyond_m` farther. A cell is open only with see-through rays and no wall points; cells with
neither are unknown (an occluded wall is not an opening). Mirrors and glass can fake see-through evidence (known limitation).
"""
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from floorplan.geometry.walls import WallPlane


@dataclass
class Span:
    axis: int
    offset: float
    a0: float
    a1: float


@dataclass
class WallImage:
    span: Span
    wall: np.ndarray      # (cols, rows) wall-point counts
    through: np.ndarray   # (cols, rows) see-through ray counts


@dataclass
class Opening:
    axis: int
    offset: float
    s0: float
    s1: float
    width: float
    interval: tuple[float, float]
    kind: str             # 'door' (open to the floor) or 'raised'
    bottom: float
    top: float
    n_rays: int
    rooms: tuple = ()


def assign_rooms(openings: list[Opening], polygons: dict, side_m: float = 0.4) -> None:
    """Fill `rooms` with the ids of the room polygons found 0.4 m to each side of the opening centre (0 = none)."""
    from shapely.geometry import Point

    for op in openings:
        mid = (op.s0 + op.s1) / 2
        ids = []
        for sgn in (-1, 1):
            p = Point(op.offset + sgn * side_m, mid) if op.axis == 0 else Point(mid, op.offset + sgn * side_m)
            ids.append(next((rid for rid, poly in polygons.items() if poly.contains(p)), 0))
        op.rooms = tuple(ids)


def plane_spans(planes: list[WallPlane], cfg: dict) -> list[Span]:
    gap = cfg["openings"]["max_gap_bridge_m"]
    spans = []
    for p in planes:
        runs = sorted(p.runs, key=lambda r: r.s0)
        a0, a1 = runs[0].s0, runs[0].s1
        for r in runs[1:]:
            if r.s0 - a1 <= gap:
                a1 = max(a1, r.s1)
            else:
                spans.append(Span(p.axis, p.offset, a0, a1))
                a0, a1 = r.s0, r.s1
        spans.append(Span(p.axis, p.offset, a0, a1))
    return spans


def _shape(sp: Span, cfg: dict) -> tuple[int, int]:
    o = cfg["openings"]
    return int(np.ceil((sp.a1 - sp.a0) / o["cell_m"])), int(np.ceil(o["image_height_m"] / o["cell_m"]))


def wall_images(spans: list[Span], pts: np.ndarray, cfg: dict) -> list[WallImage]:
    o = cfg["openings"]
    out = []
    for sp in spans:
        nc, nr = _shape(sp, cfg)
        b = 2 if sp.axis == 0 else 0
        sel = (np.abs(pts[:, sp.axis] - sp.offset) < o["wall_tol_m"]) & (pts[:, b] >= sp.a0) & (pts[:, b] < sp.a1) & (pts[:, 1] >= 0) & (pts[:, 1] < o["image_height_m"])
        wall = np.zeros((nc, nr))
        col = ((pts[sel, b] - sp.a0) / o["cell_m"]).astype(int).clip(0, nc - 1)
        row = (pts[sel, 1] / o["cell_m"]).astype(int).clip(0, nr - 1)
        np.add.at(wall, (col, row), 1)
        out.append(WallImage(sp, wall, np.zeros((nc, nr))))
    return out


def through_one(images: list[WallImage], P: np.ndarray, c: np.ndarray, cfg: dict) -> None:
    """Add one frame's rays that cross each plane (points P and camera c in the aligned frame)."""
    o = cfg["openings"]
    P = P[:: o["through_point_stride"]]
    for im in images:
        sp = im.span
        a, b = sp.axis, (2 if sp.axis == 0 else 0)
        so = c[a] - sp.offset
        if abs(so) < 0.05:
            continue
        sp_ = P[:, a] - sp.offset
        m = (so * sp_ < 0) & (np.abs(sp_) >= o["through_beyond_m"])
        if not m.any():
            continue
        t = so / (so - sp_[m])
        qa = c[b] + t * (P[m, b] - c[b])
        qy = c[1] + t * (P[m, 1] - c[1])
        ok = (qa >= sp.a0) & (qa < sp.a1) & (qy >= 0) & (qy < o["image_height_m"])
        nc, nr = im.through.shape
        col = ((qa[ok] - sp.a0) / o["cell_m"]).astype(int).clip(0, nc - 1)
        row = (qy[ok] / o["cell_m"]).astype(int).clip(0, nr - 1)
        im.through += np.bincount(col * nr + row, minlength=nc * nr).reshape(nc, nr)


def add_through(images: list[WallImage], frames: Iterable[tuple[np.ndarray, np.ndarray]], cfg: dict) -> None:
    """Accumulate rays that cross each plane. `frames` yields (points, camera position), aligned frame."""
    for P, c in frames:
        through_one(images, P, c, cfg)


def _longest_run(col: np.ndarray) -> tuple[int, int]:
    best, start, bs = 0, -1, 0
    for i, v in enumerate(np.append(col, False)):
        if v and start < 0:
            start = i
        elif not v and start >= 0:
            if i - start > best:
                best, bs = i - start, start
            start = -1
    return bs, best


def detect_openings(images: list[WallImage], cfg: dict) -> list[Opening]:
    o = cfg["openings"]
    cell = o["cell_m"]
    out: list[Opening] = []
    for im in images:
        nc, nr = im.wall.shape
        wall = im.wall >= 1
        wall_d = wall.copy()  # 3x3 dilation: the 2 cm cloud fills only ~half of the 2 cm wall cells, so pool a neighbourhood
        wall_d[1:] |= wall[:-1]
        wall_d[:-1] |= wall[1:]
        wall_d[:, 1:] |= wall[:, :-1]
        wall_d[:, :-1] |= wall[:, 1:]
        wall_d[1:, 1:] |= wall[:-1, :-1]
        wall_d[:-1, :-1] |= wall[1:, 1:]
        wall_d[1:, :-1] |= wall[:-1, 1:]
        wall_d[:-1, 1:] |= wall[1:, :-1]
        thr = im.through >= 1
        thr_c = thr.copy()
        thr_c[1:] |= thr[:-1]; thr_c[:-1] |= thr[1:]
        thr_c[:, 1:] |= thr[:, :-1]; thr_c[:, :-1] |= thr[:, 1:]
        # strict: see-through and clear of the dilated wall; loose: see-through and clear of the undilated wall.
        # The opening edge lies between them (the dilation eats ~2 cm per side), so they bracket the width.
        strict_cells, loose_cells = thr_c & ~wall_d, thr_c & ~wall
        runs = [_longest_run(strict_cells[c]) for c in range(nc)]
        loose_runs = [_longest_run(loose_cells[c]) for c in range(nc)]
        min_n = o["min_open_height_m"] / cell
        is_open = np.array([n >= min_n for _, n in runs])
        is_open_loose = np.array([n >= min_n for _, n in loose_runs])
        lo, hi = int(0.1 / cell), int(1.8 / cell)
        wall_col = wall_d[:, lo:hi].mean(1) >= 0.5
        idx = np.flatnonzero(is_open)
        if len(idx) == 0:
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) * cell > o["merge_gap_m"]) + 1)
        for g in groups:
            c0, c1 = int(g[0]), int(g[-1])
            width_strict = (c1 - c0 + 1) * cell
            if width_strict < o["min_opening_m"] or width_strict > o["max_opening_m"]:
                continue
            c0l, c1l = c0, c1
            while c0l > 0 and is_open_loose[c0l - 1]:
                c0l -= 1
            while c1l < nc - 1 and is_open_loose[c1l + 1]:
                c1l += 1
            width_loose = (c1l - c0l + 1) * cell
            left = np.flatnonzero(wall_col[:c0l])
            right = np.flatnonzero(wall_col[c1l + 1:])
            if len(left) == 0 or len(right) == 0:
                continue  # no wall on one side: the end of a wall, not an opening
            gl = (c0l - 1 - left[-1]) * cell   # unknown zone between the last wall column and the see-through start
            gr = (right[0]) * cell
            if gl > o["bounded_max_gap_m"] or gr > o["bounded_max_gap_m"]:
                continue  # wall is too far from the see-through edge to frame it
            lo_w, hi_w = width_strict, width_loose + gl + gr
            bottoms = [runs[c][0] for c in g]
            tops = [runs[c][0] + runs[c][1] for c in g]
            bottom, top = float(np.median(bottoms) * cell), float(np.median(tops) * cell)
            s0 = im.span.a0 + c0l * cell - gl / 2
            s1 = im.span.a0 + (c1l + 1) * cell + gr / 2
            out.append(Opening(im.span.axis, im.span.offset, s0, s1, (lo_w + hi_w) / 2, (lo_w, hi_w),
                               "door" if bottom <= o["door_bottom_max_m"] else "raised", bottom, top, int(im.through[c0:c1 + 1].sum())))
    return out


def merge_wall_faces(openings: list[Opening], cfg: dict) -> list[Opening]:
    """One opening through a wall is seen on both faces of the wall (two parallel planes 10-30 cm apart) and was reported twice. Openings on
    parallel planes closer than a wall thickness (`intervals.adjacency_wall_m`) that overlap by >= half of the narrower one are one opening;
    the one with more see-through rays is kept."""
    tol = cfg["intervals"]["adjacency_wall_m"]
    kept: list[Opening] = []
    for o in sorted(openings, key=lambda o: -o.n_rays):
        dup = any(k.axis == o.axis and abs(k.offset - o.offset) <= tol and min(k.s1, o.s1) - max(k.s0, o.s0) >= 0.5 * min(k.s1 - k.s0, o.s1 - o.s0)
                  for k in kept)
        if not dup:
            kept.append(o)
    return kept
