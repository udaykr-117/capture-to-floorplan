"""Metric scale for an SfM reconstruction that has none: two independent estimates and their combination.

1. depth model: every frame's depth was aligned to the SfM points with a factor a_i (model metres per SfM unit); the median over frames is
   the global scale. Per-frame values vary a lot (M1: 0.53 to 2.66 against LiDAR), so the spread over frames gives its uncertainty.
2. camera height: a hand-held phone is at chest height; the protocol asks for it. Height of the cameras above the floor in SfM units
   gives scale = prior height / measured height.
They are combined in log space by inverse variance. Both uncertainties come from config (`scale:`), which cites the measurement.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ScaleEstimate:
    metres_per_unit: float
    rel_sigma: float          # one-sigma relative uncertainty of the combined scale
    from_depth: float         # metres per SfM unit from the depth model
    from_height: float | None
    sigma_depth: float
    sigma_height: float | None
    note: str


def depth_scale(a: np.ndarray, cfg: dict) -> tuple[float, float]:
    """Median of the per-frame factors and its relative sigma: the larger of the empirical spread over frames (standard error)
    and the configured floor that covers errors shared by all frames of one capture."""
    s = cfg["scale"]
    a = np.asarray(a, float)
    med = float(np.median(a))
    se = float(1.4826 * np.median(np.abs(np.log(a / med))) / np.sqrt(max(len(a), 1)))
    return med, max(se, s["depth_rel_sigma_floor"])


def height_scale(cam_heights_sfm: np.ndarray, cfg: dict) -> tuple[float, float]:
    s = cfg["scale"]
    h = float(np.median(cam_heights_sfm))
    return s["camera_height_m"] / h, s["camera_height_rel_sigma"]


def combine(a: np.ndarray, cam_heights_sfm: np.ndarray | None, cfg: dict) -> ScaleEstimate:
    sd_val, sd = depth_scale(a, cfg)
    if cam_heights_sfm is None or len(cam_heights_sfm) == 0 or not cfg["scale"]["use_camera_height"]:
        return ScaleEstimate(sd_val, max(sd, cfg["scale"]["min_rel_sigma"]), sd_val, None, sd, None, "depth model only")
    sh_val, sh = height_scale(cam_heights_sfm, cfg)
    w_d, w_h = 1 / sd**2, 1 / sh**2
    ln = (w_d * np.log(sd_val) + w_h * np.log(sh_val)) / (w_d + w_h)
    sigma = float((w_d + w_h) ** -0.5)
    disagreement = abs(np.log(sd_val / sh_val))
    return ScaleEstimate(float(np.exp(ln)), max(sigma, disagreement / 2, cfg["scale"]["min_rel_sigma"]), sd_val, sh_val, sd, sh,
                         f"depth model and camera height combined; they differ by {100 * (np.exp(disagreement) - 1):.0f}%")
