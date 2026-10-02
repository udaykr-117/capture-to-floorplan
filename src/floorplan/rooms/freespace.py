"""Top-down map of observed free floor and its split into rooms (aligned frame)."""
import heapq
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy import ndimage

from floorplan.geometry.walls import WallPlane


@dataclass
class Grid:
    x0: float
    z0: float
    cell: float
    nz: int
    nx: int

    def rc(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.floor((z - self.z0) / self.cell).astype(int), np.floor((x - self.x0) / self.cell).astype(int)

    def xz(self, r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x0 + (c + 0.5) * self.cell, self.z0 + (r + 0.5) * self.cell

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.x0, self.x0 + self.nx * self.cell, self.z0, self.z0 + self.nz * self.cell


def make_grid(Pa: np.ndarray, cell: float, margin: float = 1.0) -> Grid:
    x0, z0 = Pa[:, 0].min() - margin, Pa[:, 2].min() - margin
    nx = int(np.ceil((Pa[:, 0].max() + margin - x0) / cell))
    nz = int(np.ceil((Pa[:, 2].max() + margin - z0) / cell))
    return Grid(float(x0), float(z0), cell, nz, nx)


def _disk(radius_m: float, cell: float) -> np.ndarray:
    r = max(1, int(round(radius_m / cell)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def rasterize_points(grid: Grid, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    img = np.zeros((grid.nz, grid.nx), bool)
    r, c = grid.rc(x, z)
    ok = (r >= 0) & (r < grid.nz) & (c >= 0) & (c < grid.nx)
    img[r[ok], c[ok]] = True
    return img


def rasterize_runs(grid: Grid, planes: list[WallPlane], half_width_m: float) -> np.ndarray:
    img = np.zeros((grid.nz, grid.nx), np.uint8)
    thick = max(1, int(round(2 * half_width_m / grid.cell)))
    for p in planes:
        for run in p.runs:
            if p.axis == 0:
                (r0, c0), (r1, c1) = [(int(a), int(b)) for a, b in zip(*grid.rc(np.array([p.offset, p.offset]), np.array([run.s0, run.s1])))]
            else:
                (r0, c0), (r1, c1) = [(int(a), int(b)) for a, b in zip(*grid.rc(np.array([run.s0, run.s1]), np.array([p.offset, p.offset])))]
            cv2.line(img, (c0, r0), (c1, r1), 1, thick)
    return img.astype(bool)


def carve_one(count: np.ndarray, P: np.ndarray, c: np.ndarray, grid: Grid, cfg: dict) -> None:
    """Add one frame's visibility fan(s) to `count` (points P and camera c in the aligned frame).

    Only hits in the wall slab are used. In each angular bin the second-farthest hit sets how far the camera could see
    (the farthest is dropped as a possible spike); the camera-to-hit fan is free space. Furniture counts as free here
    because rays pass over it: the map is a footprint.
    """
    w, r = cfg["walls"], cfg["rooms"]
    lo, hi = w["slab_above_floor_m"]
    nb = int(round(360 / r["fan_bin_deg"]))
    m = (P[:, 1] > lo) & (P[:, 1] < hi)
    if m.sum() < 50:
        return
    dx, dz = P[m, 0] - c[0], P[m, 2] - c[2]
    rng, ang = np.hypot(dx, dz), np.arctan2(dz, dx)
    b = np.floor((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    order = np.lexsort((rng, b))
    bs, rs = b[order], rng[order]
    ub, start, cnt = np.unique(bs, return_index=True, return_counts=True)
    ok = cnt >= 2
    ub, start, cnt = ub[ok], start[ok], cnt[ok]
    if len(ub) < 3:
        return
    rb = rs[start + cnt - 2]
    th = (ub + 0.5) / nb * 2 * np.pi - np.pi
    canvas = np.zeros((grid.nz, grid.nx), np.uint8)
    for g in np.split(np.arange(len(ub)), np.flatnonzero(np.diff(ub) > r["fan_gap_bins"]) + 1):
        if len(g) < 3:
            continue
        rr, cc = grid.rc(np.r_[c[0], c[0] + rb[g] * np.cos(th[g])], np.r_[c[2], c[2] + rb[g] * np.sin(th[g])])
        cv2.fillPoly(canvas, [np.stack([cc, rr], 1).astype(np.int32)], 1)
    count += canvas


def carve_fans(frames: Iterable[tuple[np.ndarray, np.ndarray]], grid: Grid, cfg: dict) -> np.ndarray:
    """Count, per map cell, the frames whose view saw through it. `frames` yields (points, camera), aligned frame."""
    count = np.zeros((grid.nz, grid.nx), np.int16)
    for P, c in frames:
        carve_one(count, P, c, grid, cfg)
    return count


def free_space(Pa: np.ndarray, Na: np.ndarray, cams_a: np.ndarray, planes: list[WallPlane], fans: np.ndarray, grid: Grid, cfg: dict) -> dict:
    s, r = cfg["scene"], cfg["rooms"]
    floor = (np.abs(Pa[:, 1]) < r["floor_band_m"]) & (np.abs(Na[:, 1]) > s["horizontal_normal_min"])
    floor_obs = rasterize_points(grid, Pa[floor, 0], Pa[floor, 2])
    cam = cv2.dilate(rasterize_points(grid, cams_a[:, 0], cams_a[:, 2]).astype(np.uint8), _disk(r["camera_free_radius_m"], grid.cell)).astype(bool)
    barrier = rasterize_runs(grid, planes, r["barrier_half_width_m"])
    carved = fans >= r["fan_min_frames"]
    free = (floor_obs | cam | carved) & ~barrier
    holes = ndimage.binary_fill_holes(free) & ~free & ~barrier
    lab, n = ndimage.label(holes)
    max_cells = r["fill_hole_max_m2"] / grid.cell**2
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    for i, sz in enumerate(sizes, 1):
        if sz <= max_cells:
            free[lab == i] = True
    return dict(floor_obs=floor_obs, cam=cam, carved=carved, barrier=barrier, free=free)


def segment_rooms(free: np.ndarray, grid: Grid, cfg: dict) -> np.ndarray:
    """Label rooms (1..n, 0 = not free): distance-transform watershed; doorway necks separate rooms."""
    r = cfg["rooms"]
    dist = ndimage.distance_transform_edt(free, sampling=grid.cell)
    seeds, n_seeds = ndimage.label(dist >= r["seed_min_dist_m"])
    comp, n_comp = ndimage.label(free)
    for k in range(1, n_comp + 1):  # free components without a core still get a seed at their deepest cell
        m = comp == k
        if not (seeds[m] > 0).any():
            n_seeds += 1
            seeds[np.unravel_index(np.argmax(np.where(m, dist, -1)), dist.shape)] = n_seeds
    labels = _merge_small(_flood(dist, seeds.astype(np.int32), free), grid, r["min_room_area_m2"])
    return split_connectors(labels, grid, cfg) if r.get("split_connectors", False) else labels


def split_connectors(labels: np.ndarray, grid: Grid, cfg: dict) -> np.ndarray:
    """Give corridors their own label. A corridor narrower than 2 x seed_min_dist never gets a seed, so the flood hands it to whichever
    neighbouring room reaches it first, and that differs between captures of the same home (JOURNAL, second fix loop). Here, inside every room,
    the part that a disk of the connector width cannot reach (narrow), long enough not to be a doorway, and touching ANOTHER room, becomes a
    new room. A strip between furniture and a wall touches only its own room and is left alone."""
    r = cfg["rooms"]
    rad = max(1, int(round(r["connector_max_width_m"] / 2 / grid.cell)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    out = labels.copy()
    nxt = int(labels.max()) + 1
    for i in [i for i in np.unique(labels) if i > 0]:
        m = labels == i
        wide = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        lab, n = ndimage.label(m & ~wide)
        for j in range(1, n + 1):
            c = lab == j
            area = c.sum() * grid.cell**2
            if area < r["connector_min_area_m2"]:
                continue
            pts = np.argwhere(c)[:, ::-1].astype(np.float32)
            (_, _), (w, h), _ = cv2.minAreaRect(pts)
            if max(w, h) * grid.cell < r["connector_min_length_m"]:
                continue
            ring = ndimage.binary_dilation(c, iterations=max(1, int(round(cfg["intervals"]["adjacency_wall_m"] / grid.cell)))) & ~m   # across a wall (same distance as room adjacency)
            if not ((labels[ring] > 0) & (labels[ring] != i)).any():
                continue
            out[c] = nxt
            nxt += 1
    return out


def _flood(dist: np.ndarray, seeds: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Marker-based watershed on -distance: expand seeds over free cells, deepest cells first; never enters non-free cells.

    (scipy's watershed_ift was tried first: its path cost uses intensity differences, so labels leaked across barriers.)
    """
    labels = seeds.copy()
    heap = [(-dist[r, c], int(r), int(c)) for r, c in zip(*np.nonzero(seeds))]
    heapq.heapify(heap)
    nz, nx = free.shape
    while heap:
        _, r, c = heapq.heappop(heap)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < nz and 0 <= cc < nx and free[rr, cc] and labels[rr, cc] == 0:
                labels[rr, cc] = labels[r, c]
                heapq.heappush(heap, (-dist[rr, cc], rr, cc))
    return labels


def _merge_small(labels: np.ndarray, grid: Grid, min_area_m2: float) -> np.ndarray:
    out = labels.copy()
    while True:
        ids = [i for i in np.unique(out) if i > 0]
        areas = {i: (out == i).sum() * grid.cell**2 for i in ids}
        small = [i for i in ids if areas[i] < min_area_m2]
        if not small:
            break
        i = min(small, key=lambda k: areas[k])
        m = out == i
        ring = ndimage.binary_dilation(m, iterations=2) & ~m
        nb = out[ring]
        nb = nb[nb > 0]
        out[m] = np.bincount(nb).argmax() if len(nb) else 0
    ids = {old: new for new, old in enumerate([i for i in np.unique(out) if i > 0], 1)}
    res = np.zeros_like(out)
    for old, new in ids.items():
        res[out == old] = new
    return res
