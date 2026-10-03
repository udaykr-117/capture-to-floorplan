"""Single-image metric depth (Depth Anything V2 metric indoor, small, CPU) and its alignment to SfM sparse points.

The model's metric scale is unreliable per image (needed scale vs LiDAR 0.53 to 2.66 on 8 test frames). Its SHAPE is usable, so every image's depth is
rescaled to agree with the SfM sparse points of that image: that makes all views consistent with each other in the SfM scale.
"""
from dataclasses import dataclass

import cv2
import numpy as np


class DepthModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg["depth"]
        self._model = None

    def _load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._torch = torch
        self._proc = AutoImageProcessor.from_pretrained(self.cfg["model"], revision=self.cfg["revision"])
        self._model = AutoModelForDepthEstimation.from_pretrained(self.cfg["model"], revision=self.cfg["revision"]).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Metric depth in metres, resized to the image size (H, W)."""
        if self._model is None:
            self._load()
        inp = self._proc(images=rgb, return_tensors="pt")
        with self._torch.no_grad():
            d = self._model(**inp).predicted_depth[0].numpy()
        return cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


@dataclass
class FrameScale:
    a: float        # model metres per SfM unit for this image
    n: int          # sparse points used
    spread: float   # robust relative spread (MAD / median) of the per-point ratios


def align_to_sparse(depth: np.ndarray, xy: np.ndarray, z_sfm: np.ndarray, cfg: dict) -> FrameScale | None:
    """Scale `a` such that depth ~ a * z_sfm at the sparse points (median of ratios, outliers trimmed). None if too few good points."""
    d = cfg["depth"]
    h, w = depth.shape
    ok = (z_sfm > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)
    if ok.sum() < d["min_sparse_points"]:
        return None
    dm = depth[np.round(xy[ok, 1]).astype(int), np.round(xy[ok, 0]).astype(int)]
    ratio = dm / z_sfm[ok]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio) < d["min_sparse_points"]:
        return None
    med = np.median(ratio)
    keep = np.abs(np.log(ratio / med)) < d["ratio_trim_log"]
    if keep.sum() < d["min_sparse_points"]:
        return None
    med = float(np.median(ratio[keep]))
    spread = float(1.4826 * np.median(np.abs(ratio[keep] - med)) / med)
    return FrameScale(med, int(keep.sum()), spread)


def backproject(depth_sfm: np.ndarray, K: tuple, R: np.ndarray, t: np.ndarray, stride: int, edge_rel: float) -> np.ndarray:
    """Dense points in the SfM frame from a depth map in SfM units. K = (f, cx, cy, k): pinhole with one radial term (COLMAP SIMPLE_RADIAL).

    Pixels at depth discontinuities (relative gradient above `edge_rel`) are dropped: they are flying pixels, not surfaces.
    """
    f, cx, cy, k = K
    h, w = depth_sfm.shape
    gy, gx = np.gradient(depth_sfm)
    rel = np.hypot(gx, gy) / np.maximum(depth_sfm, 1e-6)
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_sfm[v, u]
    keep = (rel[v, u] < edge_rel) & np.isfinite(z) & (z > 0)
    xd, yd = (u[keep] - cx) / f, (v[keep] - cy) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(3):  # invert x_d = x_u (1 + k r_u^2)
        r2 = xu**2 + yu**2
        xu, yu = xd / (1 + k * r2), yd / (1 + k * r2)
    zz = z[keep]
    p_cam = np.stack([xu * zz, yu * zz, zz], 1)
    return (p_cam - t) @ R   # world = R^T (p_cam - t)
