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

    return _map(db, image_dir, work, s, n_input, timing)


def _map(db: Path, image_dir: Path, work: Path, s: dict, n_input: int, timing: dict) -> "Sfm | None":
    """Incremental mapping on a database that already holds keypoints and verified matches; returns the largest model."""
    import pycolmap

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


def run_sfm_learned(image_dir: str | Path, work_dir: str | Path, cfg: dict, overrides: dict | None = None) -> "Sfm | None":
    """Same as run_sfm, but keypoints are DISK and matches LightGlue (kornia, CPU), for walls too plain for SIFT. COLMAP still does the
    geometric verification and the mapping. SIFT is run once with very few features only to create the image and camera rows of the database."""
    import cv2
    import kornia as K
    import kornia.feature as KF
    import pycolmap
    import torch

    s = {**cfg["sfm"], **(overrides or {})}
    image_dir, work = Path(image_dir), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "db.db"
    if db.exists():
        db.unlink()
    n_input = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    timing = {}
    reader = pycolmap.ImageReaderOptions()
    reader.default_focal_length_factor = s["focal_factor"]
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = 16
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader, extraction_options=ext)

    t0 = time.time()
    disk = KF.DISK.from_pretrained("depth").eval()
    lg = KF.LightGlueMatcher("disk").eval()
    database = pycolmap.Database.open(db)
    images = sorted(database.read_all_images(), key=lambda im: im.name)
    feats = {}
    database.clear_keypoints()
    database.clear_descriptors()
    for im in images:
        bgr = cv2.imread(str(image_dir / im.name))
        h0, w0 = bgr.shape[:2]
        w = int(s["learned_width_px"])
        h = int(round(h0 * w / w0)) // 16 * 16
        x = K.image_to_tensor(np.ascontiguousarray(cv2.resize(bgr, (w, h))[:, :, ::-1]), False).float() / 255.0
        with torch.inference_mode():
            f = disk(x, n=int(s["learned_max_keypoints"]), pad_if_not_divisible=True)[0]
        kp = f.keypoints.numpy() * np.array([w0 / w, h0 / h])
        database.write_keypoints(im.image_id, (kp + 0.5).astype(np.float32))       # COLMAP pixel centres are at +0.5
        feats[im.image_id] = (f.keypoints, f.descriptors, (h, w))
    timing["features"] = time.time() - t0

    t0 = time.time()
    pairs = []
    for a in range(len(images)):
        for b in range(a + 1, min(len(images), a + 1 + int(s["seq_overlap"]))):
            i, j = images[a].image_id, images[b].image_id
            (k0, d0, s0), (k1, d1, s1) = feats[i], feats[j]
            with torch.inference_mode():
                l0 = KF.laf_from_center_scale_ori(k0[None], torch.ones(1, len(k0), 1, 1))
                l1 = KF.laf_from_center_scale_ori(k1[None], torch.ones(1, len(k1), 1, 1))
                _, idx = lg(d0, d1, l0, l1, hw1=torch.tensor(s0), hw2=torch.tensor(s1))
            if len(idx) >= 15:
                database.write_matches(i, j, idx.numpy().astype(np.uint32))
                pairs.append((images[a].name, images[b].name))
    database.close()
    (work / "pairs.txt").write_text("\n".join(f"{a} {b}" for a, b in pairs))
    pycolmap.verify_matches(db, work / "pairs.txt")
    timing["matching"] = time.time() - t0
    return _map(db, image_dir, work, s, n_input, timing)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale s, rotation R, translation t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return float(s), R, mu_d - s * R @ mu_s
