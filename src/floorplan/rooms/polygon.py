"""Rectilinear room polygons snapped to wall planes (aligned frame).

Lines come from the wall planes (both axes) plus, where a room edge has no plane, a virtual line from the mask edge.
Each grid rectangle is given to the room owning most of its cells; a room's rectangles are unioned into a polygon.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from floorplan.geometry.walls import WallPlane
from floorplan.rooms.freespace import Grid


@dataclass
class Edge:
    axis: int            # 0: edge on the line x' = offset (runs along z'); 2: z' = offset (runs along x')
    offset: float
    s0: float
    s1: float
    source: str          # 'plane' or 'virtual'
    support: float       # fraction of the edge covered by wall-plane runs (0 for virtual)

    @property
    def length(self) -> float:
        return self.s1 - self.s0


@dataclass
class RoomPoly:
    id: int
    polygon: Polygon
    edges: list[Edge] = field(default_factory=list)

    @property
    def area(self) -> float:
        return self.polygon.area


def clean_labels(labels: np.ndarray, grid: Grid, cfg: dict) -> np.ndarray:
    """Remove thin tails (leaks) from every room and keep each room's largest connected piece."""
    r = max(1, int(round(cfg["rooms"]["min_room_width_m"] / 2 / grid.cell)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = np.zeros_like(labels)
    for i in [i for i in np.unique(labels) if i > 0]:
        m = cv2.morphologyEx((labels == i).astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        lab, n = ndimage.label(m)
        if n:
            big = np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1
            out[lab == big] = i
    return out


def _virtual_lines(mask: np.ndarray, grid: Grid, planes: list[WallPlane], cfg: dict) -> list[tuple[int, float]]:
    r = cfg["rooms"]
    min_cells = int(round(r["virtual_line_min_len_m"] / grid.cell))
    tol = r["virtual_line_plane_tol_m"]
    out: list[tuple[int, float]] = []
    left = mask & ~np.pad(mask, ((0, 0), (1, 0)))[:, :-1]
    right = mask & ~np.pad(mask, ((0, 0), (0, 1)))[:, 1:]
    down = mask & ~np.pad(mask, ((1, 0), (0, 0)))[:-1, :]
    up = mask & ~np.pad(mask, ((0, 1), (0, 0)))[1:, :]
    for axis, sets, n, origin in ((0, ((left, 0), (right, 1)), grid.nx, grid.x0), (2, ((down, 0), (up, 1)), grid.nz, grid.z0)):
        existing = np.array([p.offset for p in planes if p.axis == axis])
        for edge_mask, plus in sets:
            idx = np.nonzero(edge_mask)[1 if axis == 0 else 0]
            if len(idx) == 0:
                continue
            counts = np.bincount(idx, minlength=n)
            for c in np.flatnonzero(counts >= min_cells):
                if c > 0 and counts[c - 1] > counts[c]:
                    continue
                if c + 1 < n and counts[c + 1] > counts[c]:
                    continue
                pos = origin + (c + plus) * grid.cell
                if len(existing) == 0 or np.abs(existing - pos).min() > tol:
                    out.append((axis, float(pos)))
    rr, cc = np.nonzero(mask)  # where the mask reaches past the outermost plane there is no grid line to close it: use the mask extent
    for axis, lo, hi, origin in ((0, cc.min(), cc.max() + 1, grid.x0), (2, rr.min(), rr.max() + 1, grid.z0)):
        existing = np.array([p.offset for p in planes if p.axis == axis])
        for k in (lo, hi):
            pos = origin + k * grid.cell
            if len(existing) == 0 or pos < existing.min() - tol or pos > existing.max() + tol:
                out.append((axis, float(pos)))
    return out


def _merge_close(vals: list[float], tol: float) -> list[float]:
    vals = sorted(vals)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] > tol:
            out.append(v)
    return out


def room_polygons(labels: np.ndarray, planes: list[WallPlane], grid: Grid, cfg: dict) -> list[RoomPoly]:
    r = cfg["rooms"]
    ids = [i for i in np.unique(labels) if i > 0]
    plane_x = [p.offset for p in planes if p.axis == 0]
    plane_z = [p.offset for p in planes if p.axis == 2]
    virt: dict[int, list[tuple[int, float]]] = {i: _virtual_lines(labels == i, grid, planes, cfg) for i in ids}
    vx = [v for i in ids for a, v in virt[i] if a == 0]
    vz = [v for i in ids for a, v in virt[i] if a == 2]
    narrow = set()
    if r.get("keep_narrow_rooms", False):   # a corridor needs grid lines along its own edges, or a neighbour's larger rectangle swallows it
        tol = r["virtual_line_plane_tol_m"]
        for i in ids:
            rr, cc = np.nonzero(labels == i)
            x, z = grid.xz(rr, cc)
            wx, wz = np.ptp(x) + grid.cell, np.ptp(z) + grid.cell
            if min(wx, wz) > r["connector_max_width_m"] + 2 * grid.cell or max(wx, wz) < r["connector_min_length_m"]:
                continue
            narrow.add(i)
            for vals, lines, planes_ax in (((x.min() - grid.cell / 2, x.max() + grid.cell / 2), vx, plane_x),
                                           ((z.min() - grid.cell / 2, z.max() + grid.cell / 2), vz, plane_z)):
                for v in vals:
                    if not planes_ax or min(abs(np.array(planes_ax) - v)) > tol:
                        lines.append(float(v))
    xs = sorted(set(plane_x) | set(_merge_close(vx, 0.02) if vx else []))
    zs = sorted(set(plane_z) | set(_merge_close(vz, 0.02) if vz else []))
    plane_set = {0: np.array(plane_x), 2: np.array(plane_z)}

    def cells(lo: float, hi: float, origin: float, n: int) -> slice:
        return slice(max(0, int(np.ceil((lo - origin) / grid.cell - 0.5))), min(n, int(np.floor((hi - origin) / grid.cell - 0.5)) + 1))

    boxes: dict[int, list] = {i: [] for i in ids}
    for a in range(len(xs) - 1):
        cs = cells(xs[a], xs[a + 1], grid.x0, grid.nx)
        for b in range(len(zs) - 1):
            rs = cells(zs[b], zs[b + 1], grid.z0, grid.nz)
            sub = labels[rs, cs]
            if sub.size == 0:
                continue
            cnt = np.bincount(sub.ravel(), minlength=int(labels.max()) + 1)
            cnt[0] = 0
            k = int(np.argmax(cnt))
            if k > 0 and cnt[k] >= r["rect_min_fraction"] * sub.size:
                boxes[k].append(box(xs[a], zs[b], xs[a + 1], zs[b + 1]))

    rooms: list[RoomPoly] = []
    for i in ids:
        if not boxes[i]:
            continue
        poly = unary_union(boxes[i])
        poly = poly.buffer(r["closing_m"], join_style="mitre").buffer(-r["closing_m"], join_style="mitre")
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        holes = [h for h in poly.interiors if Polygon(h).area > r["hole_max_m2"]]
        poly = Polygon(poly.exterior, holes).simplify(0.0)
        if poly.area < (r["connector_min_area_m2"] if i in narrow else r["min_room_area_m2"]):
            continue  # slivers (wall thickness, door gaps) are not rooms; corridors may be smaller than a room
        rooms.append(RoomPoly(i, poly, _edges(poly, planes, plane_set)))
    return rooms


def _edges(poly: Polygon, planes: list[WallPlane], plane_set: dict) -> list[Edge]:
    edges = []
    pts = list(poly.exterior.coords)[:-1]
    for p, q in zip(pts, pts[1:] + pts[:1]):
        if abs(p[0] - q[0]) < 1e-6:
            axis, off, s0, s1 = 0, p[0], min(p[1], q[1]), max(p[1], q[1])
        elif abs(p[1] - q[1]) < 1e-6:
            axis, off, s0, s1 = 2, p[1], min(p[0], q[0]), max(p[0], q[0])
        else:
            continue  # a slanted edge cannot come from axis-aligned rectangles
        match = [pl for pl in planes if pl.axis == axis and abs(pl.offset - off) < 1e-6]
        if match:
            covered = sum(max(0.0, min(s1, r.s1) - max(s0, r.s0)) for pl in match for r in pl.runs)
            edges.append(Edge(axis, off, s0, s1, "plane", min(1.0, covered / (s1 - s0))))
        else:
            edges.append(Edge(axis, off, s0, s1, "virtual", 0.0))
    return edges
