"""Damage pass for a LiDAR (Stray Scanner) capture: keyframes -> detections -> surface patches -> regions -> flags -> scope."""
import time
from pathlib import Path

import cv2
import numpy as np

from floorplan import schema
from floorplan.damage.detect import Detection
from floorplan.damage.project import merge_patches, project_detection
from floorplan.damage.rules import evaluate, load_rules
from floorplan.io.stray import pose


def room_table(plan: schema.Plan, inter: dict) -> list:
    table = []
    for rp, room in zip(inter["polys"], plan.rooms):
        edges = [(w.id, e.axis, e.offset, e.s0, e.s1) for w, e in zip(room.walls, rp.edges)]
        table.append((room.id, rp.polygon, room.floor_offset_m, edges))
    return table


def run_damage(src, plan: schema.Plan, inter: dict, cfg: dict, out_dir: Path | None = None, detections: list[Detection] | None = None,
               detector=None, rules: dict | None = None) -> schema.DamageSection:
    """`detections` (precomputed, e.g. replayed from a JSON) skips the model; otherwise `detector` (or a fresh Detector) runs live on the keyframes."""
    d, s = cfg["damage"], cfg["stray"]
    assert d["keyframe_every"] % s["frame_stride"] == 0, "keyframes must be sampled frames so the chunk drift correction applies"
    rules = rules or load_rules()
    t0 = time.time()
    cap, fr, corr = src.cap, inter["al"].frame, inter["corr"]
    n_chunks = len(inter["chunks"])
    keyframes = list(range(0, len(cap.odo), d["keyframe_every"]))
    jpegs: dict[int, np.ndarray] = {}
    live = detections is None
    if live:
        detector = detector or __import__("floorplan.damage.detect", fromlist=["Detector"]).Detector(cfg)
        video = cv2.VideoCapture(str(cap.dir / "rgb.mp4"))
        detections = []
        for i in range(len(cap.odo)):
            ok, bgr = video.read()
            if not ok:
                break
            if i % d["keyframe_every"] == 0:
                ds = detector.detect(bgr, i)
                detections += ds
                if ds:
                    jpegs[i] = cv2.imencode(".jpg", cv2.resize(bgr, (960, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1]
        video.release()
    t_detect = time.time() - t0

    rooms = room_table(plan, inter)
    patches, reasons = [], {}
    by_frame: dict[int, list[Detection]] = {}
    for det in detections:
        by_frame.setdefault(det.frame, []).append(det)
    from floorplan.damage.project import box_points

    for i, dets in by_frame.items():
        depth = cv2.imread(str(cap.dir / "depth" / f"{i:06d}.png"), -1).astype(np.float32) / 1000.0
        conf = cv2.imread(str(cap.dir / "confidence" / f"{i:06d}.png"), -1)
        R, t = pose(cap.odo.iloc[i])
        ci = min((i // s["frame_stride"]) // cfg["drift"]["chunk_frames"], n_chunks - 1)

        def to_plan(p, R=R, t=t, ci=ci):
            return corr.apply(ci, fr.to_aligned(p @ R.T + t))

        cam_plan = to_plan(np.zeros((1, 3)))[0]
        for det in dets:
            P = box_points(det, depth, conf, cap.K_depth, (s["rgb_width"], s["rgb_height"]), cfg)
            patch, why = project_detection(det, P, to_plan, cam_plan, cap.K_depth, rooms, cfg)
            if patch is None:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                patches.append(patch)
    regions, dropped = merge_patches(patches, cfg)
    for line in dropped:
        key = "merge: " + ("too large for the class" if "exceeds" in line else "seen in fewer than min_views keyframes")
        reasons[key] = reasons.get(key, 0) + 1

    evidence: dict[int, list[str]] = {}
    if out_dir is not None and jpegs:
        (Path(out_dir) / "damage").mkdir(parents=True, exist_ok=True)
        ordered = sorted(regions, key=lambda r: (r.room_id, r.surface_id, r.cls, r.extent))
        for k, r in enumerate(ordered):
            if r.best.frame in jpegs:
                img = cv2.imdecode(jpegs[r.best.frame], 1)
                x0, y0, x1, y1 = (int(v / 2) for v in r.best.box)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
                cv2.putText(img, f"D{k} {r.cls} {r.best.score:.2f} {r.surface_id}", (x0 + 4, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                name = f"damage/D{k}_{r.cls}_f{r.best.frame}.jpg"
                cv2.imwrite(str(Path(out_dir) / name), img)
                evidence[k] = [name]
    reg, flags, items = evaluate(regions, rules, cfg, evidence)
    rep = dict(model=d["model"], keyframes=len(keyframes), detections=len(detections), patches_on_surfaces=len(patches), regions=len(reg), dropped_reasons=reasons,
               score_min=d["score_min"], min_views=d["min_views"], seconds_detect=round(t_detect, 1), seconds_total=round(time.time() - t0, 1),
               live=live)
    note = ("Zero-shot box detections (OWL-ViT) projected through the LiDAR depth onto room surfaces. Accuracy is untested: no labelled damage exists, "
            "scores are not calibrated, area is an upper bound (box footprint), and the model fires on clean surfaces (see JOURNAL M6). Treat every region as a candidate to check.")
    if not reg:
        note = (f"No damage region above the detector threshold ({d['score_min']}). This is NOT proof of no damage: the threshold was set so that clean captures "
                "report nothing, and recall on real damage is untested.")
    return schema.DamageSection(status="run", note=note, regions=reg, concealed_damage_flags=flags, scope_items=items, report=rep)
