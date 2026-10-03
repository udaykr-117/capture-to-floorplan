"""Capture -> Plan. One pass builds chunk clouds (then alignment and drift correction), a second pass feeds fans and see-through rays."""
import hashlib
import json
import time
from typing import Iterator, Protocol

import numpy as np
import open3d as o3d
from shapely import contains_xy
from shapely.ops import unary_union

from floorplan import intervals, schema
from floorplan.drift.chunks import analyze_chunk, build_chunks
from floorplan.drift.correct import Correction
from floorplan.drift.correct import estimate as estimate_drift
from floorplan.geometry.align import Aligned, analysis_cloud, check_residual, estimate_frame
from floorplan.geometry.walls import find_wall_planes
from floorplan.rooms.ceiling import measure_ceiling
from floorplan.rooms.freespace import carve_one, free_space, make_grid, segment_rooms
from floorplan.rooms.openings import assign_rooms, detect_openings, drop_same_room, merge_wall_faces, plane_spans, through_one, wall_images
from floorplan.rooms.polygon import clean_labels, room_polygons
from floorplan.rooms.refine import refine_rooms


class Source(Protocol):
    name: str

    def camera_positions(self) -> np.ndarray: ...
    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


def _ceiling_measurement(c, cfg: dict) -> schema.Measurement:
    if c.height is None:
        return schema.Measurement(value=None, unit="m", status="unmeasurable", note=c.reason,
                                  interval=schema.Interval(low=None, high=None, method="none: ceiling not captured"))
    w = cfg["intervals"]["sigma_k"] * cfg["intervals"].get("scale_rel_sigma", 0.0) * c.height
    lo, hi = c.interval
    lo, hi = c.height - float(np.hypot(c.height - lo, w)), c.height + float(np.hypot(hi - c.height, w))
    return schema.Measurement(value=round(c.height, 4), unit="m", status=c.status, note=c.reason,
                              interval=schema.Interval(low=round(lo, 4), high=round(hi, 4),
                                                       method=f"{cfg['ceiling']['interval_sigma']:g} sigma of local floor and ceiling spread (provisional)"))


def build_plan(src: Source, cfg: dict, tier: str = "lidar", drift: bool | None = None, chunks: list | None = None) -> tuple[schema.Plan, dict]:
    """`drift` overrides `cfg["drift"]["enabled"]` for the plane-anchored chunk correction (the source decides about odometry jumps).
    `chunks` lets a caller reuse chunk clouds built earlier from the same source (an experiment shortcut)."""
    use_drift = cfg["drift"]["enabled"] if drift is None else drift
    t: dict[str, float] = {}
    t0 = time.time()
    chunks = chunks if chunks is not None else build_chunks(src.frames(), cfg)
    cam = src.camera_positions()
    pts_w = np.concatenate([c.pts for c in chunks])
    t["chunks"] = time.time() - t0

    t0 = time.time()
    fr, floor, ev, _, _ = estimate_frame(pts_w, cam, cfg)
    if use_drift:
        corr = estimate_drift([analyze_chunk(c, fr, cfg) for c in chunks], cfg)
    else:
        corr = Correction.identity(len(chunks))
    pts_a = np.concatenate([corr.apply(c.idx, fr.to_aligned(c.pts)) for c in chunks])
    pts_a = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_a)).voxel_down_sample(cfg["stray"]["voxel_m"]).points)
    Pa, Na = analysis_cloud(pts_a, cfg)
    ev = check_residual(Pa, Na, ev, cfg)
    al = Aligned(fr, floor, ev)
    cams_a = np.concatenate([corr.apply(c.idx, fr.to_aligned(c.cams)) for c in chunks])
    t["align_and_drift"] = time.time() - t0

    t0 = time.time()
    planes = find_wall_planes(Pa, Na, cfg)
    images = wall_images(plane_spans(planes, cfg), pts_a, cfg)
    grid = make_grid(Pa, cfg["rooms"]["cell_m"])
    t["walls"] = time.time() - t0

    t0 = time.time()
    fans = np.zeros((grid.nz, grid.nx), np.int16)
    k_chunk = cfg["drift"]["chunk_frames"]
    for k, (P, c) in enumerate(src.frames()):
        ci = min(k // k_chunk, len(chunks) - 1)
        Pf, cf = corr.apply(ci, fr.to_aligned(P)), corr.apply(ci, fr.to_aligned(np.asarray(c)[None]))[0]
        carve_one(fans, Pf, cf, grid, cfg)
        through_one(images, Pf, cf, cfg)
    t["frame_pass"] = time.time() - t0

    t0 = time.time()
    layers = free_space(Pa, Na, cams_a, planes, fans, grid, cfg)
    labels = clean_labels(segment_rooms(layers["free"], grid, cfg), grid, cfg)
    polys = room_polygons(labels, planes, grid, cfg) if planes else []   # no wall plane found: nothing to build rooms from, do not invent any
    planes_iv, refine_report = planes, []
    if cfg["refine"]["enabled"] and polys:
        polys, extra, refine_report = refine_rooms(polys, chunks, fr, corr, cfg)
        planes_iv = planes + extra
    ops = detect_openings(images, cfg) if planes else []
    if cfg["openings"].get("merge_wall_faces", False):
        ops = merge_wall_faces(ops, cfg)
    assign_rooms(ops, {rp.id: rp.polygon for rp in polys})
    if cfg["openings"].get("drop_same_room", False):
        ops = drop_same_room(ops)
    t["rooms_and_openings"] = time.time() - t0

    t0 = time.time()
    rr, cc = np.meshgrid(np.arange(grid.nz), np.arange(grid.nx), indexing="ij")
    gx, gz = grid.xz(rr, cc)
    rooms, ceilings = [], {}
    for rp in polys:
        rid = f"R{rp.id}"
        inside = contains_xy(rp.polygon, gx.ravel(), gz.ravel()).reshape(gx.shape)
        observed = float((inside & (labels == rp.id)).sum() * grid.cell**2 / max(rp.area, 1e-9))
        ceil = measure_ceiling(rp.polygon, Pa, Na, cfg)
        ceilings[rid] = ceil
        walls = []
        for k, e in enumerate(rp.edges):
            p0, p1 = ((e.offset, e.s0), (e.offset, e.s1)) if e.axis == 0 else ((e.s0, e.offset), (e.s1, e.offset))
            walls.append(schema.Wall(id=f"{rid}.W{k}", p0=p0, p1=p1, length=intervals.wall_length(e, planes_iv, cfg), source=e.source, support=round(e.support, 3)))
        rooms.append(schema.Room(id=rid, polygon=[(round(x, 4), round(z, 4)) for x, z in rp.polygon.exterior.coords[:-1]],
                                 area=intervals.room_area(rp.area, rp.edges, observed, planes_iv, cfg), ceiling_height=_ceiling_measurement(ceil, cfg),
                                 observed_fraction=round(observed, 3), floor_offset_m=round(ceil.floor_offset, 4), walls=walls, openings=[]))
    out_ops = []
    for k, o in enumerate(sorted(ops, key=lambda o: (o.axis, o.offset, o.s0))):
        p0, p1 = ((o.offset, o.s0), (o.offset, o.s1)) if o.axis == 0 else ((o.s0, o.offset), (o.s1, o.offset))
        ids = [f"R{r}" if r else None for r in o.rooms]
        out_ops.append(schema.Opening(
            id=f"O{k}", kind=o.kind, p0=p0, p1=p1, bottom_m=round(o.bottom, 3), top_m=round(o.top, 3), rooms=ids,
            width=schema.Measurement(value=round(o.width, 4), unit="m", interval=schema.Interval(low=round(o.width - float(np.hypot(o.width - o.interval[0], cfg["intervals"]["sigma_k"] * cfg["intervals"].get("scale_rel_sigma", 0.0) * o.width)), 4),
                                                                                                  high=round(o.width + float(np.hypot(o.interval[1] - o.width, cfg["intervals"]["sigma_k"] * cfg["intervals"].get("scale_rel_sigma", 0.0) * o.width)), 4),
                                                                                                  method="bracket: see-through extent to wall-to-wall (provisional)"))))
        for rid in ids:
            for r in rooms:
                if r.id == rid:
                    r.openings.append(f"O{k}")

    geoms = [rp.polygon for rp in polys]
    union = unary_union(geoms) if geoms else None
    overlap = max((a.intersection(b).area for i, a in enumerate(geoms) for b in geoms[i + 1:]), default=0.0)
    adj = []
    for o in out_ops:
        a, b = o.rooms
        if a and b and a != b:
            adj.append(schema.Adjacency(a=a, b=b, via="opening", opening_id=o.id))
    iv = cfg["intervals"]
    for i, ra in enumerate(polys):
        for rb in polys[i + 1:]:
            shared = ra.polygon.buffer(iv["adjacency_wall_m"]).intersection(rb.polygon.boundary).length
            pair = {f"R{ra.id}", f"R{rb.id}"}
            if shared >= iv["adjacency_min_shared_m"] and not any({x.a, x.b} == pair for x in adj):
                adj.append(schema.Adjacency(a=f"R{ra.id}", b=f"R{rb.id}", via="shared_wall"))
    hw = float(np.sqrt(sum(((r.area.interval.high - r.area.interval.low) / 2) ** 2 for r in rooms)))
    fa = union.area if union is not None else 0.0
    stitched = schema.Stitched(
        footprint_area=schema.Measurement(value=round(fa, 3), unit="m2", interval=schema.Interval(low=round(fa - hw, 3), high=round(fa + hw, 3),
                                                                                                    method="root-sum-square of room area half-widths (provisional)")),
        n_rooms=len(rooms), max_room_overlap_m2=round(float(overlap), 4), adjacency=adj)
    t["assemble"] = time.time() - t0

    measured = [r for r in rooms if r.ceiling_height.status != "unmeasurable"]
    limits = [
        "No ground truth: accuracy of every value is untested. Wall-length and room-area intervals are calibrated only against repeat captures of one property (a lower bound); other intervals are provisional.",
        "Damage detection has not been run on this plan (separate pass: `plan run` does it with the models group installed).",
        "Spaces joined by an opening wider than 1.2 m are merged into one room; rooms are assumed rectilinear.",
        "Openings use see-through evidence: mirrors and glass can create phantom openings, windows to the outside are not detected.",
    ]
    if not planes:
        limits.append("No wall plane was found in the reconstruction: no rooms are reported.")
    if not measured:
        limits.append("No ceiling was captured in any room: all ceiling heights are unmeasurable.")
    info = getattr(src, "info", None)
    tier_report: dict = {}
    if info is not None:  # image tiers (video, photos)
        sc = info["scale"]
        tier_report = dict(images_in=info["n_input"], images_registered=info["n_registered"], images_with_depth=info["n_aligned"],
                           registered_fraction=round(info["n_registered"] / max(info["n_input"], 1), 3), scale_metres_per_sfm_unit=round(sc.metres_per_unit, 5),
                           scale_rel_sigma=round(sc.rel_sigma, 3), scale_from_depth=round(sc.from_depth, 5), scale_from_camera_height=round(sc.from_height, 5) if sc.from_height else None,
                           camera_height_m=round(info["camera_height_m"], 2), gravity_refined_deg=round(info["gravity"]["shift_deg"], 1),
                           cloud_extent_m=[round(float(x), 2) for x in np.ptp(Pa, axis=0)], note=sc.note)
        limits.append(f"{tier} tier: no depth sensor and no poses. SfM registered {info['n_registered']} of {info['n_input']} images ({100 * info['n_registered'] / max(info['n_input'], 1):.0f}%); "
                      f"only the registered part of the capture is reconstructed. Metric scale from monocular depth and camera height, relative uncertainty +-{100 * sc.rel_sigma:.0f}% (1 sigma).")
    if cfg["refine"]["enabled"]:
        n_ref = sum(1 for r in refine_report if r["refined"])
        limits.append(f"Room-local refinement: wall positions of {n_ref} of {len(refine_report)} rooms re-measured from that room's longest single visit (drift between visits "
                      "removed from room dimensions); rooms are refined independently, so a shared wall can differ by a few cm between its two rooms.")
    jumps = list(getattr(src, "jumps", []))
    mode = "none"
    report: dict = {}
    if use_drift:
        d = cfg["drift"]
        parts = [name for name, on in (("chunk heading", d["use_yaw"]), ("floor level", d["use_floor"]), ("wall-matching shifts", d["use_shift"]),
                                       ("odometry jump distribution", bool(jumps))) if on]
        mode = " + ".join(parts) or "none"
        dg = corr.diag
        report = dict(components=parts, chunks=len(chunks), chunk_frames=d["chunk_frames"], associations=dg["n_assoc"], residual_rms_cm_before=round(dg["rms_before_cm"], 2),
                      residual_rms_cm_after=round(dg["rms_after_cm"], 2), holdout_rms_cm_before=round(dg["holdout_before_cm"], 2) if "holdout_before_cm" in dg else None,
                      holdout_rms_cm_after=round(dg["holdout_after_cm"], 2) if "holdout_after_cm" in dg else None, max_rotation_deg=round(float(np.max(np.abs(dg["theta_deg"]))), 2),
                      max_shift_cm=round(float(np.max(np.linalg.norm(np.array(dg["shift_cm"]), axis=1))), 1) if d["use_shift"] else 0.0,
                      floor_level_range_cm=[round(float(np.min(dg["dy_cm"])), 1), round(float(np.max(dg["dy_cm"])), 1)] if d["use_floor"] else [0.0, 0.0],
                      odometry_jumps=[dict(frames=j["frames"], jump_cm=round(j["jump_m"] * 100, 1), spread_over_frames=j["spread_over_frames"]) for j in jumps])
        limits.append(f"Drift: corrected per chunk of {d['chunk_frames']} sampled frames by: {', '.join(parts) or 'nothing'}. Assumes rectilinear walls and a flat floor; "
                      "drift without wall evidence is not corrected" + ("" if d["use_shift"] else "; wall-matching shifts are not applied (they did not improve agreement in the ablation)") + ".")
    else:
        limits.append("Drift is NOT corrected (poses used as-is): floor and wall positions can drift by several cm to tens of cm across a capture.")
    plan = schema.Plan(
        capture=src.name, tier=tier, frame=schema.FrameInfo(yaw_deg=round(fr.yaw_deg, 3), floor_world_y=round(fr.floor_y, 4), drift_correction=mode, drift_report={**report, 'room_refinement': refine_report} if refine_report else report, tier_report=tier_report),
        rooms=rooms, openings=out_ops, stitched=stitched, timing_s={k: round(v, 2) for k, v in t.items()}, limitations=limits,
        config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16])
    return plan, dict(Pa=Pa, Na=Na, corr=corr, chunks=chunks, planes=planes, polys=polys, labels=labels, grid=grid, layers=layers, ceilings=ceilings, al=al)
