"""Time chunks of a capture and what each chunk says about its own drift (wall direction, floor height)."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import open3d as o3d

from floorplan.geometry.align import Frame, analysis_cloud, wall_yaw


@dataclass
class Chunk:
    idx: int
    n_frames: int
    pts: np.ndarray   # world frame, voxel-downsampled
    cams: np.ndarray  # world frame, one per sampled frame


def build_chunks(frames: Iterable[tuple[np.ndarray, np.ndarray]], cfg: dict) -> list[Chunk]:
    """Group consecutive sampled frames into chunks of `drift.chunk_frames`; each chunk's points are voxel-downsampled."""
    k, voxel = cfg["drift"]["chunk_frames"], cfg["stray"]["voxel_m"]
    chunks: list[Chunk] = []
    pts, cams = [], []

    def flush() -> None:
        if cams:
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts).astype(np.float64))).voxel_down_sample(voxel)
            chunks.append(Chunk(len(chunks), len(cams), np.asarray(p.points), np.array(cams)))

    for P, c in frames:
        pts.append(P)
        cams.append(np.asarray(c, dtype=float))
        if len(cams) == k:
            flush()
            pts, cams = [], []
    flush()
    return chunks


@dataclass
class ChunkAnalysis:
    idx: int
    Pa: np.ndarray            # analysis cloud (aligned frame)
    Na: np.ndarray
    cams_a: np.ndarray
    yaw_dev_deg: float        # wall direction minus the nearest axis (nan if not trustworthy)
    yaw_resultant: float
    n_vertical: int
    floor_y: float            # median height of floor points in the aligned frame (nan if too few)
    n_floor: int


def analyze_aligned(idx: int, pts_a: np.ndarray, cams_a: np.ndarray, cfg: dict) -> ChunkAnalysis:
    """Wall direction and floor height of one chunk whose points and cameras are already in the aligned frame."""
    d, s, r = cfg["drift"], cfg["scene"], cfg["rooms"]
    Pa, Na = analysis_cloud(pts_a, cfg)  # only voxelises and adds normals
    lo, hi = cfg["walls"]["slab_above_floor_m"]
    slab = (Pa[:, 1] > lo) & (Pa[:, 1] < hi)
    dev, res, nv = float("nan"), 0.0, 0
    if slab.any():
        yaw, ev = wall_yaw(Pa[slab], Na[slab], cfg)
        nv, res = ev["n_vertical"], ev["resultant"]
        if nv >= d["min_vertical_points"] and res >= d["min_resultant"]:
            dev = float((yaw + 45) % 90 - 45)
    fl = (np.abs(Na[:, 1]) > s["horizontal_normal_min"]) & (np.abs(Pa[:, 1]) < r["floor_band_m"])
    fy = float(np.median(Pa[fl, 1])) if fl.sum() >= d["floor_min_points"] else float("nan")
    return ChunkAnalysis(idx, Pa, Na, cams_a, dev, float(res), int(nv), fy, int(fl.sum()))


def analyze_chunk(ch: Chunk, frame: Frame, cfg: dict) -> ChunkAnalysis:
    return analyze_aligned(ch.idx, frame.to_aligned(ch.pts), frame.to_aligned(ch.cams), cfg)
