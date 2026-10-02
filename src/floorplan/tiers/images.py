"""Video and photo tiers: images only (no depth, no poses) -> the same Plan as the LiDAR tier, with intervals widened by the scale uncertainty."""
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import cv2

from floorplan import schema
from floorplan.io.frames import extract_video_frames
from floorplan.io.imagesource import build_image_source, depth_for_images
from floorplan.pipeline import build_plan
from floorplan.sfm import run_sfm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def read_image(p: Path):
    """BGR image, upright. iPhones save HEIC by default: read through pillow-heif (EXIF orientation applied); others through OpenCV
    (which also applies EXIF orientation). None if unreadable."""
    if p.suffix.lower() in (".heic", ".heif"):
        import numpy as np
        import pillow_heif
        from PIL import Image, ImageOps

        pillow_heif.register_heif_opener()
        try:
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return cv2.imread(str(p))


def empty_plan(name: str, tier: str, reason: str, cfg: dict, timing: dict, report: dict) -> schema.Plan:
    """A plan with no rooms and the reason, instead of numbers that are not there."""
    nm = schema.Measurement(value=None, unit="m2", status="unmeasurable", note=reason, interval=schema.Interval(low=None, high=None, method="none: reconstruction failed"))
    return schema.Plan(capture=name, tier=tier, frame=schema.FrameInfo(yaw_deg=0.0, floor_world_y=0.0, tier_report=report), rooms=[], openings=[],
                       stitched=schema.Stitched(footprint_area=nm, n_rooms=0, max_room_overlap_m2=0.0, adjacency=[]), timing_s={k: round(v, 2) for k, v in timing.items()},
                       limitations=[f"{tier} tier produced no plan: {reason}"], config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16])


def run_images(image_dir: Path, work_dir: Path, cfg: dict, matcher: str, name: str, tier: str, n_input: int | None = None) -> tuple[schema.Plan, dict]:
    t: dict[str, float] = {}
    t0 = time.time()
    sfm = run_sfm(image_dir, work_dir / "sfm", cfg, matcher)
    t["sfm"] = time.time() - t0
    n_in = n_input or len(list(image_dir.glob("*.jpg")))
    if sfm is None or sfm.n_registered < cfg["sfm"]["min_registered"]:
        got = 0 if sfm is None else sfm.n_registered
        return empty_plan(name, tier, f"structure from motion registered {got} of {n_in} images (needs at least {cfg['sfm']['min_registered']})", cfg, t,
                          dict(images_in=n_in, images_registered=got)), {}
    t0 = time.time()
    per_image = depth_for_images(sfm, image_dir, cfg)
    t["depth"] = time.time() - t0
    src = build_image_source(name, per_image, cfg, n_in, sfm.n_registered, t)
    cfg2 = copy.deepcopy(cfg)
    cfg2["intervals"]["scale_rel_sigma"] = src.info["scale"].rel_sigma
    try:
        plan, inter = build_plan(src, cfg2, tier=tier)
    except ValueError as e:  # e.g. the partial reconstruction has no consistent rectilinear wall direction
        return empty_plan(name, tier, f"the reconstruction ({sfm.n_registered} of {n_in} images) could not be turned into a plan: {e}", cfg, t,
                          dict(images_in=n_in, images_registered=sfm.n_registered, images_with_depth=src.info["n_aligned"])), dict(sfm=sfm, source=src)
    plan.timing_s.update({k: round(v, 2) for k, v in t.items()})
    return plan, dict(inter, sfm=sfm, source=src)


def run_video(video: Path, work_dir: Path, cfg: dict) -> tuple[schema.Plan, dict]:
    t0 = time.time()
    frames = extract_video_frames(video, work_dir / "frames", cfg["sfm"]["video_fps"], cfg["sfm"]["frame_width_px"], sharpest=True)
    plan, inter = run_images(work_dir / "frames", work_dir, cfg, "sequential", f"{Path(video).parent.name}/{Path(video).name}", "video", len(frames))
    plan.timing_s["frames"] = round(time.time() - t0 - plan.timing_s.get("sfm", 0) - plan.timing_s.get("depth", 0), 2)
    return plan, inter


def run_photos(folder: Path, work_dir: Path, cfg: dict) -> tuple[schema.Plan, dict]:
    """`folder` holds one subfolder of photos per room (or a flat set of photos). All photos go into one SfM with exhaustive matching, so
    photos of a doorway that see the next room can link the rooms."""
    flat = work_dir / "photos"
    if flat.exists():
        shutil.rmtree(flat)
    flat.mkdir(parents=True)
    n, unreadable = 0, []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in IMAGE_SUFFIXES:
            img = read_image(p)
            if img is None:
                unreadable.append(p.name)
                continue
            w = cfg["sfm"]["frame_width_px"]
            room = p.parent.name if p.parent != Path(folder) else "room"
            cv2.imwrite(str(flat / f"{room}__{p.stem}.jpg"), cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA))
            n += 1
    plan, inter = run_images(flat, work_dir, cfg, "exhaustive", f"{Path(folder).name} (photos)", "photo", n)
    if unreadable:
        plan.limitations.append(f"{len(unreadable)} photo file(s) could not be read and were skipped: {unreadable[:10]}")
    return plan, inter
