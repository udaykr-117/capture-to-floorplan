"""Capture -> Plan. One pass to build the cloud and alignment, one pass over the frames for fans and see-through rays."""
import hashlib
import json
import time
from typing import Iterator, Protocol

import numpy as np
from shapely import contains_xy
from shapely.ops import unary_union

from floorplan import intervals, schema
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes
from floorplan.rooms.ceiling import measure_ceiling
from floorplan.rooms.freespace import carve_one, free_space, make_grid, segment_rooms
from floorplan.rooms.openings import assign_rooms, detect_openings, plane_spans, through_one, wall_images
from floorplan.rooms.polygon import clean_labels, room_polygons


class Source(Protocol):
    name: str

    def cloud(self) -> np.ndarray: ...
    def camera_positions(self) -> np.ndarray: ...
    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


def _ceiling_measurement(c, cfg: dict) -> schema.Measurement:
    if c.height is None:
        return schema.Measurement(value=None, unit="m", status="unmeasurable", note=c.reason,
                                  interval=schema.Interval(low=None, high=None, method="none: ceiling not captured"))
    return schema.Measurement(value=round(c.height, 4), unit="m", status=c.status, note=c.reason,
                              interval=schema.Interval(low=round(c.interval[0], 4), high=round(c.interval[1], 4),
                                                       method=f"{cfg['ceiling']['interval_sigma']:g} sigma of local floor and ceiling spread (provisional)"))


def build_plan(src: Source, cfg: dict, tier: str = "lidar") -> tuple[schema.Plan, dict]:
    t: dict[str, float] = {}
    t0 = time.time()
    pts = src.cloud()
    cam = src.camera_positions()
    t["cloud"] = time.time() - t0

    t0 = time.time()
    al, Pa, Na = align(pts, cam, cfg)
    fr = al.frame
    planes = find_wall_planes(Pa, Na, cfg)
    images = wall_images(plane_spans(planes, cfg), fr.to_aligned(pts), cfg)
    grid = make_grid(Pa, cfg["rooms"]["cell_m"])
    t["align_and_walls"] = time.time() - t0

    t0 = time.time()
    fans = np.zeros((grid.nz, grid.nx), np.int16)
    for P, c in src.frames():
        Pf, cf = fr.to_aligned(P), fr.to_aligned(np.asarray(c)[None])[0]
        carve_one(fans, Pf, cf, grid, cfg)
        through_one(images, Pf, cf, cfg)
    t["frame_pass"] = time.time() - t0

    t0 = time.time()
    cams_a = fr.to_aligned(cam)
    layers = free_space(Pa, Na, cams_a, planes, fans, grid, cfg)
    labels = clean_labels(segment_rooms(layers["free"], grid, cfg), grid, cfg)
    polys = room_polygons(labels, planes, grid, cfg)
    ops = detect_openings(images, cfg)
    assign_rooms(ops, {rp.id: rp.polygon for rp in polys})
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
            walls.append(schema.Wall(id=f"{rid}.W{k}", p0=p0, p1=p1, length=intervals.wall_length(e, planes, cfg), source=e.source, support=round(e.support, 3)))
        rooms.append(schema.Room(id=rid, polygon=[(round(x, 4), round(z, 4)) for x, z in rp.polygon.exterior.coords[:-1]],
                                 area=intervals.room_area(rp.area, rp.edges, observed, planes, cfg), ceiling_height=_ceiling_measurement(ceil, cfg),
                                 observed_fraction=round(observed, 3), floor_offset_m=round(ceil.floor_offset, 4), walls=walls, openings=[]))
    out_ops = []
    for k, o in enumerate(sorted(ops, key=lambda o: (o.axis, o.offset, o.s0))):
        p0, p1 = ((o.offset, o.s0), (o.offset, o.s1)) if o.axis == 0 else ((o.s0, o.offset), (o.s1, o.offset))
        ids = [f"R{r}" if r else None for r in o.rooms]
        out_ops.append(schema.Opening(
            id=f"O{k}", kind=o.kind, p0=p0, p1=p1, bottom_m=round(o.bottom, 3), top_m=round(o.top, 3), rooms=ids,
            width=schema.Measurement(value=round(o.width, 4), unit="m", interval=schema.Interval(low=round(o.interval[0], 4), high=round(o.interval[1], 4),
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
        "No ground truth: accuracy of every value is untested; intervals are provisional and uncalibrated.",
        "Poses are used as-is (no drift correction yet); floor and wall positions can drift by several cm across a capture.",
        "Damage regions, concealed-damage flags and scope items are not implemented.",
        "Spaces joined by an opening wider than 1.2 m are merged into one room; rooms are assumed rectilinear.",
        "Openings use see-through evidence: mirrors and glass can create phantom openings, windows to the outside are not detected.",
    ]
    if not measured:
        limits.append("No ceiling was captured in any room: all ceiling heights are unmeasurable.")
    plan = schema.Plan(
        capture=src.name, tier=tier, frame=schema.FrameInfo(yaw_deg=round(fr.yaw_deg, 3), floor_world_y=round(fr.floor_y, 4)),
        rooms=rooms, openings=out_ops, stitched=stitched, timing_s={k: round(v, 2) for k, v in t.items()}, limitations=limits,
        config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16])
    return plan, dict(Pa=Pa, planes=planes, polys=polys, labels=labels, grid=grid, layers=layers, ceilings=ceilings, al=al)
