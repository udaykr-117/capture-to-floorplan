"""A source for the pipeline built from images only (video frames or photos): SfM poses + aligned monocular depth.

It gives the pipeline what a LiDAR capture gives it: per-frame world points (metres, Y up) and camera positions.
Steps: depth per image (metric model, rescaled to each image's SfM points) -> dense points in the SfM frame -> provisional metric scale ->
gravity (up = +Y) -> final metric scale (depth model + camera-height prior, see scale.py).
"""
import time
from pathlib import Path

import cv2
import numpy as np

from floorplan import scale as scale_mod
from floorplan.depth import DepthModel, align_to_sparse, backproject
from floorplan.geometry.align import analysis_cloud, estimate_floor
from floorplan.geometry.gravity import camera_up, estimate_up, rotation_to_y
from floorplan.sfm import Sfm


class ImageSource:
    def __init__(self, name: str, frames: list[tuple[np.ndarray, np.ndarray]], cams: np.ndarray, info: dict):
        self.name, self._frames, self._cams, self.info = name, frames, cams, info
        self.jumps: list = []

    def camera_positions(self) -> np.ndarray:
        return self._cams

    def frames(self):
        yield from self._frames


def depth_for_images(sfm: Sfm, image_dir: Path, cfg: dict, model: DepthModel | None = None) -> list[dict]:
    """Per registered image: dense points in the SfM frame (float32), camera centre and up, and the depth alignment factor."""
    d = cfg["depth"]
    model = model or DepthModel(cfg)
    K = (sfm.camera["params"][0], sfm.camera["params"][1], sfm.camera["params"][2], sfm.camera["params"][3])
    out = []
    for im in sfm.images:
        bgr = cv2.imread(str(image_dir / im.name))
        depth = model.predict(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        z_sfm = (im.xyz @ im.R.T + im.t)[:, 2]
        fs = align_to_sparse(depth, im.xy, z_sfm, cfg)
        if fs is None:
            out.append(dict(name=im.name, skipped=True))
            continue
        dsfm = depth / fs.a
        dsfm[(depth < d["min_depth_m"]) | (depth > d["max_depth_m"])] = np.nan
        P = backproject(dsfm, K, im.R, im.t, d["stride"], d["edge_rel"]).astype(np.float32)
        out.append(dict(name=im.name, skipped=False, P=P, center=im.center, up=camera_up(im.R), a=fs.a, n=fs.n, spread=fs.spread))
    return out


def build_image_source(name: str, per_image: list[dict], cfg: dict, n_input: int, n_registered: int, timing: dict | None = None) -> ImageSource:
    used = [p for p in per_image if not p["skipped"]]
    if not used:
        raise ValueError("no image could be aligned to its SfM points")
    a = np.array([p["a"] for p in used])
    s0 = float(np.median(a))                                              # provisional metres per SfM unit
    rng = np.random.default_rng(0)
    sample = np.concatenate([p["P"][rng.choice(len(p["P"]), min(len(p["P"]), 4000), replace=False)] for p in used]) * s0
    P_s, N_s = analysis_cloud(sample.astype(np.float64), cfg)
    up, ginfo = estimate_up(np.array([p["up"] for p in used]), N_s)
    G = rotation_to_y(up)
    cams_prov = np.array([p["center"] for p in used]) * s0 @ G.T
    all_y = (np.concatenate([p["P"][::8] for p in used]) * s0 @ G.T)[:, 1]
    floor = estimate_floor(all_y, cams_prov[:, 1], cfg)
    heights_sfm = (cams_prov[:, 1] - floor.height) / s0                    # camera height above the floor, SfM units
    est = scale_mod.combine(a, heights_sfm, cfg)
    s = est.metres_per_unit
    frames = [((p["P"].astype(np.float64) * s) @ G.T, (p["center"] * s) @ G.T) for p in used]
    cams = np.array([c for _, c in frames])
    info = dict(n_input=n_input, n_registered=n_registered, n_aligned=len(used), per_frame_a_median=s0, per_frame_a_spread=float(np.std(np.log(a))),
                scale=est, gravity=ginfo, camera_height_m=float(np.median(heights_sfm) * s), timing=timing or {})
    return ImageSource(name, frames, cams, info)
