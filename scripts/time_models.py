"""CPU timing spike for candidate models (M1). Prints seconds per item and, where possible, a sanity comparison to LiDAR.

Usage: uv run python scripts/time_models.py depth|owl|colmap [capture_dir]
Weights are downloaded by Hugging Face on first use (see DISCLOSURE.md); nothing is committed.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = ROOT / "single_room" / "c00a170fe1"
OUT = ROOT / "out"


def read_frames(video: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Sequential decode (seeking in mp4 can land on the wrong frame); returns RGB frames."""
    want, got = set(indices), {}
    cap = cv2.VideoCapture(str(video))
    i = 0
    while i <= max(indices):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return got


def depth_test(d: Path) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    t0 = time.time()
    proc, model = AutoImageProcessor.from_pretrained(name), AutoModelForDepthEstimation.from_pretrained(name).eval()
    print(f"model load {time.time() - t0:.1f}s; params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; torch threads {torch.get_num_threads()}")
    n = len(pd.read_csv(d / "odometry.csv"))
    idx = [int(x) for x in np.linspace(100, n - 100, 8)]
    frames = read_frames(d / "rgb.mp4", idx)
    rows = []
    for k, i in enumerate(idx):
        inp = proc(images=frames[i], return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            pred = model(**inp).predicted_depth[0].numpy()
        dt = time.time() - t0
        lid = cv2.imread(str(d / "depth" / f"{i:06d}.png"), -1).astype(np.float32) / 1000
        conf = cv2.imread(str(d / "confidence" / f"{i:06d}.png"), -1)
        p = cv2.resize(pred, (lid.shape[1], lid.shape[0]), interpolation=cv2.INTER_LINEAR)
        m = (lid > 0.1) & (lid < 5) & (conf >= 2)
        ratio = np.median(lid[m] / p[m])
        absrel = np.mean(np.abs(p[m] - lid[m]) / lid[m])
        absrel_scaled = np.mean(np.abs(p[m] * ratio - lid[m]) / lid[m])
        rows.append((i, dt, ratio, absrel, absrel_scaled, int(m.sum()), inp["pixel_values"].shape[-2:]))
        print(f"frame {i:5d}: {dt:5.2f}s{' (first call, includes warm-up)' if k == 0 else ''}  input {tuple(inp['pixel_values'].shape[-2:])}  "
              f"median(lidar/pred) {ratio:.3f}  absRel as-is {absrel:.3f}  absRel after per-frame scale {absrel_scaled:.3f}  ({int(m.sum()):,} px)")
    ts = [r[1] for r in rows[1:]]
    print(f"seconds per frame (excluding first): mean {np.mean(ts):.2f}, min {np.min(ts):.2f}, max {np.max(ts):.2f}")
    ratios = np.array([r[2] for r in rows])
    print(f"per-frame scale (lidar/pred): mean {ratios.mean():.3f}, std {ratios.std():.3f} ({100 * ratios.std() / ratios.mean():.1f}% of mean)")


def owl_test(d: Path) -> None:
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    name = "google/owlvit-base-patch32"
    t0 = time.time()
    proc, model = OwlViTProcessor.from_pretrained(name), OwlViTForObjectDetection.from_pretrained(name).eval()
    print(f"model load {time.time() - t0:.1f}s; params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    queries = [["a water stain on a wall", "mold", "a crack in a wall", "peeling paint", "a hole in a wall"]]
    n = len(pd.read_csv(d / "odometry.csv"))
    idx = [int(x) for x in np.linspace(100, n - 100, 4)]
    frames = read_frames(d / "rgb.mp4", idx)
    ts = []
    for k, i in enumerate(idx):
        inp = proc(text=queries, images=frames[i], return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            out = model(**inp)
        dt = time.time() - t0
        ts.append(dt)
        print(f"frame {i:5d}: {dt:5.2f}s{' (first call)' if k == 0 else ''}  max score per query {np.round(out.logits[0].sigmoid().max(0).values.numpy(), 3)}")
    print(f"seconds per frame (excluding first): mean {np.mean(ts[1:]):.2f}. Detection quality: untested (no labelled damage).")


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (scale, R, t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def colmap_test(d: Path, n_images: int = 100, width: int = 960) -> None:
    import pycolmap

    odo = pd.read_csv(d / "odometry.csv", skipinitialspace=True)
    idx = [int(x) for x in np.linspace(0, len(odo) - 1, n_images)]
    work = OUT / f"colmap_{d.parent.name}"
    img_dir = work / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    frames = read_frames(d / "rgb.mp4", idx)
    for i, rgb in frames.items():
        h = int(rgb.shape[0] * width / rgb.shape[1])
        cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), cv2.cvtColor(cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))
    print(f"{len(frames)} frames written at {width}px wide in {time.time() - t0:.1f}s (every ~{len(odo) / n_images:.0f}th frame)")

    db = work / "db.db"
    if db.exists():
        db.unlink()
    t0 = time.time()
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE)
    t_feat = time.time() - t0
    t0 = time.time()
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = 10
    pycolmap.match_sequential(db, pairing_options=pairing)
    t_match = time.time() - t0
    t0 = time.time()
    out = work / "sparse"
    out.mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, img_dir, out)
    t_map = time.time() - t0
    print(f"features {t_feat:.1f}s ({t_feat / len(frames):.2f}s/img), sequential matching (overlap 10) {t_match:.1f}s, mapping {t_map:.1f}s")
    if not recs:
        print("mapping produced no reconstruction")
        return
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    print(f"reconstructions {len(recs)}; largest: registered {rec.num_reg_images()}/{len(frames)} images, {rec.num_points3D():,} points, "
          f"mean reprojection error {rec.compute_mean_reprojection_error():.2f}px")
    cams, ark = [], []
    for im in rec.images.values():
        i = int(Path(im.name).stem)
        cams.append(im.projection_center())
        ark.append(odo.loc[i, ["x", "y", "z"]].values.astype(float))
    cams, ark = np.array(cams), np.array(ark)
    s, R, t = umeyama(cams, ark)
    resid = np.linalg.norm(s * cams @ R.T + t - ark, axis=1)
    print(f"Sim(3) fit of COLMAP camera centres to ARKit positions ({len(cams)} images): scale {s:.3f} m per COLMAP unit, "
          f"residual RMSE {100 * np.sqrt((resid ** 2).mean()):.1f} cm, max {100 * resid.max():.1f} cm, ARKit path extent {np.round(np.ptp(ark, 0), 2)} m")


if __name__ == "__main__":
    which = sys.argv[1]
    cap = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CAPTURE
    OUT.mkdir(exist_ok=True)
    print(f"cpu cores {os.cpu_count()}, capture {cap}")
    if which == "colmap":
        colmap_test(cap, n_images=int(sys.argv[3]) if len(sys.argv) > 3 else 100)
    else:
        {"depth": depth_test, "owl": owl_test}[which](cap)
