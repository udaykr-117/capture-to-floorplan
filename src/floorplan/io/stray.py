"""Stray Scanner export -> metric world-frame points.

Convention from kekeblom/StrayVisualizer@195c640 stray_visualize.py:
  L50-55   odometry row `timestamp, frame, x, y, z, qx, qy, qz, qw`; quaternion is xyzw; pose is T_WC
  L67      depth PNG is millimetres
  L79      intrinsics are for 1920x1440 and are scaled to the depth resolution
  L134,144 Open3D pinhole back-projection (x right, y down, z forward), no axis flip: p_world = R @ p_cam + t
  L135-136 depth, confidence and pose are paired by frame index
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation


@dataclass
class Capture:
    dir: Path
    K_rgb: np.ndarray
    odo: pd.DataFrame
    depth_hw: tuple[int, int]
    K_depth: np.ndarray
    rgb_wh: tuple[int, int] = (1920, 1440)


def scale_intrinsics(K_rgb: np.ndarray, rgb_wh: tuple[int, int], depth_hw: tuple[int, int]) -> np.ndarray:
    sx, sy = depth_hw[1] / rgb_wh[0], depth_hw[0] / rgb_wh[1]
    K = K_rgb.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def open_capture(d: str | Path, cfg: dict) -> Capture:
    d = Path(d)
    rgb_wh = (cfg["stray"]["rgb_width"], cfg["stray"]["rgb_height"])
    K_rgb = np.loadtxt(d / "camera_matrix.csv", delimiter=",")
    odo = pd.read_csv(d / "odometry.csv", skipinitialspace=True)
    n = len(odo)
    assert (odo["frame"].astype(int).values == np.arange(n)).all(), "odometry `frame` column is not the row index"
    n_depth = len(list((d / "depth").glob("*.png")))
    n_conf = len(list((d / "confidence").glob("*.png")))
    assert n_depth == n_conf == n, f"counts differ: depth {n_depth}, confidence {n_conf}, odometry {n}"
    cap = cv2.VideoCapture(str(d / "rgb.mp4"))
    video_wh = (int(cap.get(3)), int(cap.get(4)))
    n_video = int(cap.get(7))
    cap.release()
    if min(video_wh) > 0 and video_wh != rgb_wh:   # camera_matrix.csv is at the video resolution; another device may record a different size than the samples
        print(f"note: video is {video_wh[0]}x{video_wh[1]} (samples were {rgb_wh[0]}x{rgb_wh[1]}); intrinsics are taken as given at the video size")
        rgb_wh = video_wh
    if n_video != n:
        print(f"note: video has {n_video} frames, odometry has {n}; depth and poses are used, video frames only for damage detection")
    first = cv2.imread(str(d / "depth" / "000000.png"), -1)
    depth_hw = first.shape
    return Capture(d, K_rgb, odo, depth_hw, scale_intrinsics(K_rgb, rgb_wh, depth_hw), rgb_wh)


def backproject(depth_m: np.ndarray, conf: np.ndarray, K: np.ndarray, conf_min: int, dmin: float, dmax: float) -> np.ndarray:
    """Pinhole back-projection into camera axes (x right, y down, z forward), shape (N, 3)."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    m = (depth_m > dmin) & (depth_m < dmax) & (conf >= conf_min)
    z = depth_m[m]
    return np.stack([(u[m] - K[0, 2]) * z / K[0, 0], (v[m] - K[1, 2]) * z / K[1, 1], z], 1)


def pose(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Camera-to-world rotation (quaternion xyzw) and translation."""
    R = Rotation.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
    return R, np.array([row.x, row.y, row.z])


def iter_frames(cap: Capture, cfg: dict, stride: int | None = None) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (frame index, world points, camera position in world) for every `stride`-th frame."""
    s = cfg["stray"]
    stride = stride or s["frame_stride"]
    for i in range(0, len(cap.odo), stride):
        name = f"{i:06d}.png"
        depth = cv2.imread(str(cap.dir / "depth" / name), -1).astype(np.float32) / 1000.0
        conf = cv2.imread(str(cap.dir / "confidence" / name), -1)
        p_cam = backproject(depth, conf, cap.K_depth, s["confidence_min"], s["depth_min_m"], s["depth_max_m"])
        R, t = pose(cap.odo.iloc[i])
        yield i, p_cam @ R.T + t, t


class StraySource:
    """What the pipeline needs from a capture: a world cloud, camera positions, and a pass over per-frame hits."""

    def __init__(self, capture_dir: str | Path, cfg: dict):
        self.cfg = cfg
        self.cap = open_capture(capture_dir, cfg)
        self.jumps: list[dict] = []
        if cfg["drift"]["enabled"] and cfg["drift"]["jump_correction"]:
            from floorplan.drift.jumps import distribute_jumps

            self.cap.odo, self.jumps = distribute_jumps(self.cap.odo, cfg)
        d = Path(capture_dir).resolve()
        self.name = f"{d.parent.name}/{d.name}"

    def cloud(self) -> np.ndarray:
        return cloud(self.cap, self.cfg)

    def camera_positions(self) -> np.ndarray:
        return self.cap.odo[["x", "y", "z"]].values

    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for _, pts, origin in iter_frames(self.cap, self.cfg):
            yield pts, origin


def cloud(cap: Capture, cfg: dict, stride: int | None = None, voxel: float | None = None) -> np.ndarray:
    """Voxel-downsampled world-frame cloud, shape (N, 3)."""
    s = cfg["stray"]
    voxel = voxel or s["voxel_m"]
    acc = o3d.geometry.PointCloud()
    batch: list[np.ndarray] = []

    def flush() -> None:
        nonlocal acc, batch
        if batch:
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(batch)))
            acc = (acc + p).voxel_down_sample(voxel)
            batch = []

    for _, pts, _ in iter_frames(cap, cfg, stride):
        batch.append(pts.astype(np.float64))
        if len(batch) >= s["flush_every_frames"]:
            flush()
    flush()
    return np.asarray(acc.points)
