"""Structure from motion (pycolmap, CPU) on a set of images: video frames or photos. Poses and sparse points in an arbitrary scale."""
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SfmImage:
    name: str
    R: np.ndarray       # cam_from_world rotation (3, 3)
    t: np.ndarray       # cam_from_world translation (3,)
    center: np.ndarray  # camera centre in the SfM frame
    xy: np.ndarray      # (n, 2) pixels of the observations that have a 3D point
    xyz: np.ndarray     # (n, 3) those 3D points in the SfM frame


@dataclass
class Sfm:
    images: list[SfmImage]
    camera: dict            # model, width, height, params (SIMPLE_RADIAL: f, cx, cy, k)
    n_input: int
    n_models: int
    n_registered: int
    n_points: int
    reproj_err: float
    timing: dict = field(default_factory=dict)


def run_sfm(image_dir: str | Path, work_dir: str | Path, cfg: dict, matcher: str = "sequential", overrides: dict | None = None) -> Sfm | None:
    """Features, matching and incremental mapping. Returns the largest model, or None if nothing was reconstructed."""
    import pycolmap

    s = {**cfg["sfm"], **(overrides or {})}
    image_dir, work = Path(image_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "db.db"
    if db.exists():
        db.unlink()
    n_input = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    timing = {}

    t0 = time.time()
    reader = pycolmap.ImageReaderOptions()
    reader.default_focal_length_factor = s["focal_factor"]
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.peak_threshold = s["sift_peak_threshold"]
    ext.sift.max_num_features = s["sift_max_features"]
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)
    timing["features"] = time.time() - t0

    t0 = time.time()
    fm = pycolmap.FeatureMatchingOptions()
    fm.guided_matching = bool(s["guided_matching"])
    if matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = int(s["seq_overlap"])
        pycolmap.match_sequential(db, matching_options=fm, pairing_options=pairing)
    else:
        pycolmap.match_exhaustive(db, matching_options=fm)
    timing["matching"] = time.time() - t0

    t0 = time.time()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.mapper.init_min_tri_angle = float(s["init_min_tri_angle_deg"])
    sparse = work / "sparse"
    sparse.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, image_dir, sparse, opts)
    timing["mapping"] = time.time() - t0
    if not recs:
        return None
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    cam = next(iter(rec.cameras.values()))
    images = []
    for im in rec.images.values():
        pts = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in im.points2D if p.has_point3D()]
        xy = np.array([a for a, _ in pts]).reshape(-1, 2)
        xyz = np.array([b for _, b in pts]).reshape(-1, 3)
        c = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        images.append(SfmImage(im.name, np.array(c.rotation.matrix()), np.array(c.translation), np.array(im.projection_center()), xy, xyz))
    images.sort(key=lambda i: i.name)
    return Sfm(images, dict(model=str(cam.model), width=int(cam.width), height=int(cam.height), params=[float(x) for x in cam.params]), n_input, len(recs),
               rec.num_reg_images(), rec.num_points3D(), float(rec.compute_mean_reprojection_error()), timing)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale s, rotation R, translation t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return float(s), R, mu_d - s * R @ mu_s
