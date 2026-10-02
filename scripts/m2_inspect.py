"""Stage-by-stage inspection of the LiDAR pipeline on the samples (M2). Prints evidence, saves images to out/.

Usage: uv run python scripts/m2_inspect.py <stage> [name ...]     names: single_room floor_only with_ceiling
The loaded cloud is cached in out/cache keyed by a hash of the capture path and the `stray` config, so reruns are fast.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.io.stray import cloud, open_capture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {
    "single_room": ROOT / "single_room" / "c00a170fe1",
    "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
    "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6",
}
CFG = load_config()


def load_cached(name: str):
    cap = open_capture(SAMPLES[name], CFG)
    key = hashlib.md5((str(SAMPLES[name]) + json.dumps(CFG["stray"], sort_keys=True)).encode()).hexdigest()[:10]
    f = OUT / "cache" / f"{name}_{key}.npy"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        pts = np.load(f)
    else:
        t0 = time.time()
        pts = cloud(cap, CFG)
        np.save(f, pts)
        print(f"  loaded cloud in {time.time() - t0:.0f}s ({len(pts):,} points)")
    return cap, pts


def stage_align(name: str) -> None:
    cap, pts = load_cached(name)
    cam = cap.odo[["x", "y", "z"]].values
    t0 = time.time()
    al, Pa, Na = align(pts, cam, CFG)
    print(f"  align {time.time() - t0:.1f}s; analysis cloud {len(Pa):,} points")
    ev, fl = al.evidence, al.floor
    print(f"  wall yaw {al.frame.yaw_deg:.2f} deg (mod 90); vertical-normal points {ev['n_vertical']:,}, used {ev['n_used']:,}, "
          f"resultant length {ev['resultant']:.3f}, residual after rotating {ev['residual_deg']:.2f} deg")
    print(f"  floor height (world y) {fl.height:.3f} m; peaks (height, count): {[(round(h, 3), c) for h, c in fl.candidates]}; "
          f"chosen-vs-strongest {100 * fl.spread_m:.1f} cm; IQR of points within +-3 cm {100 * fl.iqr_m:.1f} cm")
    ch = cam[:, 1] - fl.height
    print(f"  camera height above floor: min {ch.min():.2f}, median {np.median(ch):.2f}, max {ch.max():.2f} m")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.hist2d(Pa[:, 0], Pa[:, 2], bins=500, norm=matplotlib.colors.LogNorm())
    ax.set_aspect("equal"); ax.set_xlabel("x'"); ax.set_ylabel("z'"); ax.set_title(f"{name}: aligned top-down")
    fig.savefig(OUT / f"m2_align_{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def aligned_scene(name: str):
    cap, pts = load_cached(name)
    al, Pa, Na = align(pts, cap.odo[["x", "y", "z"]].values, CFG)
    return cap, pts, al, Pa, Na


def stage_walls(name: str) -> None:
    from floorplan.geometry.walls import find_wall_planes, structural

    cap, pts, al, Pa, Na = aligned_scene(name)
    t0 = time.time()
    planes = find_wall_planes(Pa, Na, CFG)
    print(f"  wall planes found in {time.time() - t0:.1f}s: {len(planes)} planes, {sum(len(p.runs) for p in planes)} runs; "
          f"structural runs {sum(len(p.runs) for p in structural(planes, CFG))}")
    lines = ["axis offset(m) points runs[(s0, s1, length, coverage)]"]
    for p in sorted(planes, key=lambda p: (p.axis, p.offset)):
        lines.append(f"{'x' if p.axis == 0 else 'z'}' {p.offset:8.3f} {p.n_points:7d}  " +
                     "  ".join(f"({r.s0:.2f},{r.s1:.2f},{r.length:.2f}m,{r.coverage:.2f})" for r in p.runs))
    (OUT / f"m2_walls_{name}.txt").write_text("\n".join(lines))
    print("\n".join(lines[:16]) + (f"\n  ... ({len(lines) - 16} more rows in out/m2_walls_{name}.txt)" if len(lines) > 16 else ""))
    cov = np.array([r.coverage for p in planes for r in p.runs])
    print(f"  run coverage: min {cov.min():.2f}, median {np.median(cov):.2f}; runs below {CFG['walls']['min_height_coverage']}: {(cov < CFG['walls']['min_height_coverage']).sum()}")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist2d(Pa[:, 0], Pa[:, 2], bins=500, norm=matplotlib.colors.LogNorm(), cmap="Greys")
    for p in planes:
        for r in p.runs:
            col = "limegreen" if r.coverage >= CFG["walls"]["min_height_coverage"] else "red"
            xs, zs = ([p.offset, p.offset], [r.s0, r.s1]) if p.axis == 0 else ([r.s0, r.s1], [p.offset, p.offset])
            ax.plot(xs, zs, color=col, lw=1.6, alpha=0.8)
    ax.set_aspect("equal"); ax.set_title(f"{name}: wall plane runs (green structural, red low)")
    fig.savefig(OUT / f"m2_walls_{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


def cached_fans(name, cap, al, grid):
    from floorplan.io.stray import iter_frames
    from floorplan.rooms.freespace import carve_fans

    key = hashlib.md5(json.dumps(dict(stray=CFG["stray"], slab=CFG["walls"]["slab_above_floor_m"],
                                      rooms={k: CFG["rooms"][k] for k in ("cell_m", "fan_bin_deg", "fan_gap_bins")},
                                      yaw=round(al.frame.yaw_deg, 3), floor=round(al.frame.floor_y, 3),
                                      grid=[grid.x0, grid.z0, grid.nz, grid.nx]), sort_keys=True).encode()).hexdigest()[:10]
    f = OUT / "cache" / f"fans_{name}_{key}.npy"
    if f.exists():
        return np.load(f)
    t0 = time.time()
    frames = ((al.frame.to_aligned(p), al.frame.to_aligned(c[None])[0]) for _, p, c in iter_frames(cap, CFG))
    fans = carve_fans(frames, grid, CFG)
    np.save(f, fans)
    print(f"  fans carved in {time.time() - t0:.0f}s")
    return fans


def stage_rooms(name: str) -> None:
    from floorplan.geometry.walls import find_wall_planes
    from floorplan.rooms.freespace import free_space, make_grid, segment_rooms

    cap, pts, al, Pa, Na = aligned_scene(name)
    cams_a = al.frame.to_aligned(cap.odo[["x", "y", "z"]].values)
    planes = find_wall_planes(Pa, Na, CFG)
    grid = make_grid(Pa, CFG["rooms"]["cell_m"])
    fans = cached_fans(name, cap, al, grid)
    t0 = time.time()
    layers = free_space(Pa, Na, cams_a, planes, fans, grid, CFG)
    labels = segment_rooms(layers["free"], grid, CFG)
    c2 = grid.cell ** 2
    print(f"  grid {grid.nz}x{grid.nx} cells; free-space + rooms in {time.time() - t0:.1f}s; layer areas m2: floor seen "
          f"{layers['floor_obs'].sum() * c2:.1f}, carved {layers['carved'].sum() * c2:.1f}, camera {layers['cam'].sum() * c2:.1f}, barrier {layers['barrier'].sum() * c2:.1f}")
    ids = [i for i in np.unique(labels) if i > 0]
    print(f"  free area {layers['free'].sum() * grid.cell ** 2:.1f} m2; rooms found: {len(ids)}")
    for i in ids:
        r, c = np.nonzero(labels == i)
        x, z = grid.xz(r, c)
        print(f"    room {i}: area {len(r) * grid.cell ** 2:6.2f} m2, centroid ({x.mean():.2f}, {z.mean():.2f}), bbox x'[{x.min():.2f},{x.max():.2f}] z'[{z.min():.2f},{z.max():.2f}]")
    ext = grid.extent
    fig, axs = plt.subplots(1, 2, figsize=(18, 9))
    base = np.ones((grid.nz, grid.nx, 3))
    base[layers["carved"]] = (0.80, 0.93, 0.80)
    base[layers["floor_obs"]] = (0.65, 0.65, 0.65)
    base[layers["cam"]] = (0.55, 0.75, 1.0)
    base[layers["barrier"]] = (0.9, 0.1, 0.1)
    axs[0].imshow(base, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]))
    axs[0].set_title(f"{name}: carved (green), floor seen (grey), camera (blue), barrier (red)")
    cmap = plt.get_cmap("tab20")
    col = np.ones((grid.nz, grid.nx, 3))
    for i in ids:
        col[labels == i] = cmap((i - 1) % 20)[:3]
    axs[1].imshow(col, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]))
    for i in ids:
        r, c = np.nonzero(labels == i)
        x, z = grid.xz(r, c)
        axs[1].text(x.mean(), z.mean(), f"{i}\n{len(r) * grid.cell ** 2:.1f}m2", ha="center", va="center", fontsize=8)
    axs[1].set_title(f"{name}: {len(ids)} rooms")
    for a in axs:
        a.set_aspect("equal")
    fig.savefig(OUT / f"m2_rooms_{name}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def build_rooms(name: str):
    from floorplan.geometry.walls import find_wall_planes
    from floorplan.rooms.freespace import free_space, make_grid, segment_rooms
    from floorplan.rooms.polygon import clean_labels, room_polygons

    cap, pts, al, Pa, Na = aligned_scene(name)
    cams_a = al.frame.to_aligned(cap.odo[["x", "y", "z"]].values)
    planes = find_wall_planes(Pa, Na, CFG)
    grid = make_grid(Pa, CFG["rooms"]["cell_m"])
    fans = cached_fans(name, cap, al, grid)
    layers = free_space(Pa, Na, cams_a, planes, fans, grid, CFG)
    labels = clean_labels(segment_rooms(layers["free"], grid, CFG), grid, CFG)
    rooms = room_polygons(labels, planes, grid, CFG)
    return dict(cap=cap, pts=pts, al=al, Pa=Pa, Na=Na, cams_a=cams_a, planes=planes, grid=grid, layers=layers, labels=labels, rooms=rooms)


def stage_polygons(name: str) -> None:
    s = build_rooms(name)
    rooms, grid = s["rooms"], s["grid"]
    print(f"  rooms with polygons: {len(rooms)} (labels after cleaning: {len([i for i in np.unique(s['labels']) if i > 0])})")
    tot = 0.0
    for rp in rooms:
        tot += rp.area
        planes_e = [e for e in rp.edges if e.source == "plane"]
        print(f"  room {rp.id}: area {rp.area:6.2f} m2, {len(rp.edges)} edges ({len(planes_e)} plane, {len(rp.edges) - len(planes_e)} virtual), "
              f"perimeter {rp.polygon.length:.2f} m, mean wall support {np.mean([e.support for e in rp.edges]):.2f}")
        for e in rp.edges:
            print(f"      {'x' if e.axis == 0 else 'z'}'={e.offset:7.3f}  {e.s0:7.2f}..{e.s1:7.2f}  len {e.length:5.2f} m  {e.source:7s} support {e.support:.2f}")
    ov = max((a.polygon.intersection(b.polygon).area for i, a in enumerate(rooms) for b in rooms[i + 1:]), default=0.0)
    print(f"  total room area {tot:.1f} m2; largest pairwise overlap {ov:.3f} m2; mask area {(s['labels'] > 0).sum() * grid.cell ** 2:.1f} m2")
    fig, ax = plt.subplots(figsize=(11, 11))
    Pa = s["Pa"]
    ax.hist2d(Pa[:, 0], Pa[:, 2], bins=500, norm=matplotlib.colors.LogNorm(), cmap="Greys")
    cmap = plt.get_cmap("tab20")
    for rp in rooms:
        x, z = rp.polygon.exterior.xy
        ax.fill(x, z, color=cmap((rp.id - 1) % 20), alpha=0.35)
        ax.plot(x, z, color=cmap((rp.id - 1) % 20), lw=2)
        c = rp.polygon.representative_point()
        ax.text(c.x, c.y, f"{rp.id}\n{rp.area:.1f}m2", ha="center", va="center", fontsize=9, weight="bold")
        for e in rp.edges:
            if e.source == "virtual":
                xs, zs = ([e.offset] * 2, [e.s0, e.s1]) if e.axis == 0 else ([e.s0, e.s1], [e.offset] * 2)
                ax.plot(xs, zs, "k--", lw=1.5)
    ax.set_aspect("equal"); ax.set_title(f"{name}: room polygons (dashed = virtual edge, no wall plane)")
    fig.savefig(OUT / f"m2_polygons_{name}.png", dpi=85, bbox_inches="tight")
    plt.close(fig)


def stage_ceiling(name: str) -> None:
    from floorplan.rooms.ceiling import measure_ceiling

    s = build_rooms(name)
    print(f"  global floor (world y) {s['al'].frame.floor_y:.3f} m")
    for rp in s["rooms"]:
        t0 = time.time()
        c = measure_ceiling(rp.polygon, s["Pa"], s["Na"], CFG)
        if c.height is None:
            print(f"  room {rp.id} ({rp.area:5.1f} m2): {c.status.upper()}: {c.reason}; levels {c.levels[:4]}; local floor offset {100 * c.floor_offset:+.1f} cm")
        else:
            iv = c.interval
            print(f"  room {rp.id} ({rp.area:5.1f} m2): {c.status}: height {c.height:.3f} m, interval [{iv[0]:.3f}, {iv[1]:.3f}] (+-{100 * (iv[1] - iv[0]) / 2:.1f} cm), "
                  f"coverage {100 * c.coverage:.0f}%, local floor offset {100 * c.floor_offset:+.1f} cm; levels (h, cov) {c.levels[:4]}"
                  + (f"; {c.reason}" if c.reason else ""))


def stage_ceiling_conf1(name: str) -> None:
    """Does confidence >= 1 (instead of >= 2) reveal ceilings that the default hides?"""
    import copy

    from floorplan.geometry.align import analysis_cloud
    from floorplan.rooms.ceiling import measure_ceiling

    s = build_rooms(name)
    cfg1 = copy.deepcopy(CFG)
    cfg1["stray"]["confidence_min"] = 1
    f = OUT / "cache" / f"{name}_conf1.npy"
    if f.exists():
        pts1 = np.load(f)
    else:
        t0 = time.time()
        pts1 = cloud(s["cap"], cfg1)
        np.save(f, pts1)
        print(f"  conf>=1 cloud: {len(pts1):,} points (conf>=2: {len(s['pts']):,}), loaded in {time.time() - t0:.0f}s")
    P1, N1 = analysis_cloud(pts1, CFG)
    fr = s["al"].frame
    P1a, N1a = fr.to_aligned(P1), fr.rotate(N1)
    for rp in s["rooms"]:
        c2 = measure_ceiling(rp.polygon, s["Pa"], s["Na"], CFG)
        c1 = measure_ceiling(rp.polygon, P1a, N1a, CFG)
        h = lambda c: "-" if c.height is None else f"{c.height:.3f}"
        print(f"  room {rp.id} ({rp.area:5.1f} m2): conf>=2 {c2.status:12s} h {h(c2):>5s} cov {100 * c2.coverage:3.0f}%   |   conf>=1 {c1.status:12s} h {h(c1):>5s} cov {100 * c1.coverage:3.0f}%")


def cached_openings(name, s):
    import pickle

    from floorplan.io.stray import iter_frames
    from floorplan.rooms.openings import add_through, plane_spans, wall_images

    spans = plane_spans(s["planes"], CFG)
    key = hashlib.md5(json.dumps(dict(stray=CFG["stray"], op=CFG["openings"], spans=[(p.axis, round(p.offset, 3), round(p.a0, 2), round(p.a1, 2)) for p in spans],
                                      yaw=round(s["al"].frame.yaw_deg, 3), floor=round(s["al"].frame.floor_y, 3)), sort_keys=True).encode()).hexdigest()[:10]
    f = OUT / "cache" / f"openings_{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.time()
    fr = s["al"].frame
    images = wall_images(spans, fr.to_aligned(s["pts"]), CFG)
    add_through(images, ((fr.to_aligned(p), fr.to_aligned(c[None])[0]) for _, p, c in iter_frames(s["cap"], CFG)), CFG)
    f.write_bytes(pickle.dumps(images))
    print(f"  wall images + see-through rays for {len(spans)} plane spans in {time.time() - t0:.0f}s")
    return images


def stage_openings(name: str) -> None:
    from floorplan.rooms.openings import assign_rooms, detect_openings

    s = build_rooms(name)
    images = cached_openings(name, s)
    ops = detect_openings(images, CFG)
    assign_rooms(ops, {rp.id: rp.polygon for rp in s["rooms"]})
    print(f"  openings found: {len(ops)} ({sum(o.kind == 'door' for o in ops)} door/passage, {sum(o.kind == 'raised' for o in ops)} raised)")
    for o in sorted(ops, key=lambda o: (o.axis, o.offset, o.s0)):
        print(f"    {o.kind:6s} on {'x' if o.axis == 0 else 'z'}'={o.offset:7.3f} along {o.s0:7.2f}..{o.s1:7.2f}: width {o.width:.2f} m "
              f"(interval {o.interval[0]:.2f}..{o.interval[1]:.2f}), open {o.bottom:.2f}..{o.top:.2f} m, {o.n_rays:,} rays, rooms {o.rooms}")
    fig, ax = plt.subplots(figsize=(11, 11))
    Pa = s["Pa"]
    ax.hist2d(Pa[:, 0], Pa[:, 2], bins=500, norm=matplotlib.colors.LogNorm(), cmap="Greys")
    for rp in s["rooms"]:
        x, z = rp.polygon.exterior.xy
        ax.plot(x, z, color="tab:blue", lw=1.2)
    for o in ops:
        xs, zs = ([o.offset] * 2, [o.s0, o.s1]) if o.axis == 0 else ([o.s0, o.s1], [o.offset] * 2)
        ax.plot(xs, zs, color="red" if o.kind == "door" else "orange", lw=5, solid_capstyle="butt")
    ax.set_aspect("equal"); ax.set_title(f"{name}: openings (red door/passage, orange raised)")
    fig.savefig(OUT / f"m2_openings_{name}.png", dpi=85, bbox_inches="tight")
    plt.close(fig)


STAGES = {"align": stage_align, "walls": stage_walls, "rooms": stage_rooms, "polygons": stage_polygons, "ceiling": stage_ceiling,
          "ceiling_conf1": stage_ceiling_conf1, "openings": stage_openings}

if __name__ == "__main__":
    stage = sys.argv[1]
    for n in (sys.argv[2:] or SAMPLES):
        print(f"== {stage}: {n}")
        STAGES[stage](n)
