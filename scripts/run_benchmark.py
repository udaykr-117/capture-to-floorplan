"""Benchmark: regenerates every M7 number from the raw captures.

  uv run python scripts/run_benchmark.py --variant before    # refine.enabled = false (pipeline as of M6)
  uv run python scripts/run_benchmark.py --variant after     # refine.enabled = true  (the M7 fix)

Per capture: rooms, footprint, openings, ceilings, timing. Repeatability (the gate: two captures agree within 1 cm or 0.5% per wall), measured on
the PLAN OUTPUT: for every pair of captures, rooms are matched after registration (IoU >= 0.5) and each room dimension (distance between two
opposite plane-backed walls of the room) is compared between the captures. The M3 metric (all facing wall-plane pairs) is printed alongside.
There is no ground truth: these are agreement numbers between repeat captures, not accuracy.
Writes bench/results/<variant>/{benchmark.json, <capture>.json}.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Polygon

from floorplan.config import load_config
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import Seg, match_rooms, match_segments, register, segments, to_other, to_other_polygon, wall_points, width_pairs

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "single_room": ROOT / "single_room" / "c00a170fe1"}
PAIRS = [("with_ceiling", "floor_only"), ("with_ceiling", "single_room"), ("floor_only", "single_room")]
OFF = {"keep_narrow_rooms": False, "split_connectors": False}
VARIANTS = {
    "before": {"refine": {"enabled": False}, "rooms": OFF},                                                  # the pipeline before the fixes
    "refined": {"refine": {"enabled": True, "visits": "longest"}, "rooms": OFF},                              # Fix A only
    "final": {"refine": {"enabled": True, "visits": "longest"}, "rooms": {"keep_narrow_rooms": True, "split_connectors": False}},   # Fix A + Fix B: the shipped pipeline
    "corridor_split": {"refine": {"enabled": True}, "rooms": {"keep_narrow_rooms": True, "split_connectors": True}},               # Fix B with corridors cut out of rooms (not shipped)
    "median_visits": {"refine": {"enabled": True, "visits": "median"}, "rooms": {"keep_narrow_rooms": True, "split_connectors": False}},  # tested alternative to Fix A (rejected)
}


def wall_seg(w) -> Seg:
    (x0, z0), (x1, z1) = w.p0, w.p1
    if abs(x0 - x1) < 1e-9:
        return Seg(0, x0, min(z0, z1), max(z0, z1), 0.0)
    return Seg(2, z0, min(x0, x1), max(x0, x1), 0.0)


def room_dimensions(room) -> list[tuple[int, int, float]]:
    """(wall index i, wall index j, distance) for opposite plane-backed walls of one room: same axis, >= 0.5 m overlap, and the
    line between them at the middle of the overlap runs inside the room (so they face each other across the room)."""
    poly = Polygon(room.polygon)
    segs = [(k, wall_seg(w)) for k, w in enumerate(room.walls) if w.source == "plane"]
    out = []
    for a in range(len(segs)):
        for b in range(a + 1, len(segs)):
            (i, s), (j, t) = segs[a], segs[b]
            ov0, ov1 = max(s.s0, t.s0), min(s.s1, t.s1)
            if s.axis != t.axis or ov1 - ov0 < 0.5 or abs(s.offset - t.offset) < 0.5:
                continue
            m = (ov0 + ov1) / 2
            line = LineString([(s.offset, m), (t.offset, m)] if s.axis == 0 else [(m, s.offset), (m, t.offset)])
            if poly.buffer(1e-6).contains(line.interpolate(0.5, normalized=True)) and line.difference(poly.buffer(0.02)).length < 0.05:
                out.append((i, j, abs(t.offset - s.offset)))
    return out


def gate(d: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (np.abs(d) <= 0.01) | (np.abs(d) <= 0.005 * w)


def compare(A: dict, B: dict, cfg: dict) -> dict:
    reg = register(A["W"], B["W"], cfg)
    pa = {r.id: r for r in A["plan"].rooms}
    pb = {r.id: r for r in B["plan"].rooms}
    rm = [r for r in match_rooms({k: Polygon(v.polygon) for k, v in pa.items()}, {k: Polygon(v.polygon) for k, v in pb.items()}, reg) if r["iou"] >= 0.5]
    tol, ov = cfg["register"]["plane_match_tol_m"], cfg["register"]["plane_min_overlap_m"]
    dims = []
    for r in rm:
        ra, rb = pa[r["a"]], pb[r["b"]]
        sb = [to_other(wall_seg(w), reg) if w.source == "plane" else None for w in rb.walls]
        for i, j, wa in room_dimensions(ra):
            match = []
            for k in (i, j):
                s = wall_seg(ra.walls[k])
                c = [(abs(t.offset - s.offset), q) for q, t in enumerate(sb) if t is not None and t.axis == s.axis and abs(t.offset - s.offset) <= tol
                     and min(t.s1, s.s1) - max(t.s0, s.s0) >= ov]
                match.append(min(c)[1] if c else None)
            if None in match or match[0] == match[1]:
                continue
            wb = abs(sb[match[0]].offset - sb[match[1]].offset)
            dims.append(dict(room_a=ra.id, room_b=rb.id, walls_a=[ra.walls[i].id, ra.walls[j].id], walls_b=[rb.walls[match[0]].id, rb.walls[match[1]].id],
                             width_a=round(wa, 4), width_b=round(wb, 4), delta=round(wb - wa, 4)))
    hw = lambda m: (m.interval.high - m.interval.low) / 2
    areas = [dict(room_a=r["a"], room_b=r["b"], area_a=pa[r["a"]].area.value, area_b=pb[r["b"]].area.value, delta=pb[r["b"]].area.value - pa[r["a"]].area.value,
                  halfwidth=float(np.hypot(hw(pa[r["a"]].area), hw(pb[r["b"]].area)))) for r in rm]
    walls = []        # same wall in both captures (plane-backed in both, same axis, offsets within tolerance, >= 50% overlap): length agreement
    for r in rm:
        ra, rb = pa[r["a"]], pb[r["b"]]
        sb = [(w, to_other(wall_seg(w), reg)) for w in rb.walls if w.source == "plane"]
        for w in (w for w in ra.walls if w.source == "plane"):
            s = wall_seg(w)
            c = [(abs(t.offset - s.offset), k) for k, (wb, t) in enumerate(sb) if t.axis == s.axis and abs(t.offset - s.offset) <= tol
                 and min(t.s1, s.s1) - max(t.s0, s.s0) >= 0.5 * min(t.s1 - t.s0, s.s1 - s.s0)]
            if c:
                wb = sb[min(c)[1]][0]
                walls.append(dict(wall_a=w.id, wall_b=wb.id, length_a=w.length.value, length_b=wb.length.value, delta=wb.length.value - w.length.value,
                                  halfwidth=float(np.hypot(hw(w.length), hw(wb.length)))))
    ops = []          # same opening in both captures: centres within 30 cm after registration, same orientation
    for oa in A["plan"].openings:
        ca = np.mean([oa.p0, oa.p1], 0)
        da = np.subtract(oa.p1, oa.p0)
        best = None
        for ob in B["plan"].openings:
            q = reg.apply(np.array([ob.p0, ob.p1], float))
            cb, db = q.mean(0), q[1] - q[0]
            same_dir = abs(abs(np.dot(da, db)) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9) - 1) < 0.05
            dist = float(np.linalg.norm(cb - ca))
            if same_dir and dist <= 0.3 and (best is None or dist < best[0]):
                best = (dist, ob)
        if best:
            ob = best[1]
            ops.append(dict(opening_a=oa.id, opening_b=ob.id, width_a=oa.width.value, width_b=ob.width.value, delta=ob.width.value - oa.width.value,
                            halfwidth=float(np.hypot(hw(oa.width), hw(ob.width))), centre_dist_m=round(best[0], 3)))
    SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
    wp = width_pairs(SA, SB, match_segments(SA, SB, cfg), cfg)
    return dict(registration=dict(fitness_5cm=round(reg.fitness_5cm, 3), median_cm=round(reg.median_cm, 2), accepted=bool(reg.accepted)),
                rooms_matched=[dict(a=r["a"], b=r["b"], iou=round(r["iou"], 2)) for r in rm], room_dims=dims, room_areas=areas, walls=walls, openings=ops,
                n_openings=[len(A["plan"].openings), len(B["plan"].openings)],
                plane_pairs=[dict(width_a=round(p["width_a"], 4), delta=round(p["delta"], 4)) for p in wp])


def stats(d: np.ndarray, w: np.ndarray) -> str:
    if len(d) == 0:
        return "n 0"
    g = gate(d, w)
    return f"n {len(d):2d}  median |d| {np.median(np.abs(d)) * 100:4.1f} cm  p90 {np.percentile(np.abs(d), 90) * 100:4.1f} cm  pass {int(g.sum())}/{len(d)} ({100 * g.mean():.0f}%)"


def main(variant: str):
    cfg = load_config()
    for sec, kv in VARIANTS[variant].items():
        cfg[sec] = {**copy.deepcopy(cfg[sec]), **kv}
    out = ROOT / "bench" / "results" / variant
    out.mkdir(parents=True, exist_ok=True)
    caps = {}
    print(f"=== benchmark, variant '{variant}' (overrides {VARIANTS[variant]}) ===")
    for name, d in SAMPLES.items():
        t0 = time.time()
        plan, inter = build_plan(StraySource(d, cfg), cfg)
        dt = time.time() - t0
        caps[name] = dict(plan=plan, planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], cfg))
        (out / f"{name}.json").write_text(plan.model_dump_json(indent=1))
        ceil = [r.ceiling_height.value for r in plan.rooms if r.ceiling_height.value is not None]
        ref = plan.frame.drift_report.get("room_refinement", [])
        print(f"{name:13s} {len(plan.rooms)} rooms, footprint {plan.stitched.footprint_area.value:.1f} m2, {len(plan.openings)} openings, ceilings {len(ceil)}/{len(plan.rooms)}, "
              f"max room overlap {plan.stitched.max_room_overlap_m2:.3f} m2, {dt:.0f} s" + (f"; refined rooms {sum(1 for r in ref if r['refined'])}/{len(ref)}" if ref else ""))
    res = {}
    all_d, all_w, all_pd, all_pw = [], [], [], []
    print("\nREPEATABILITY on the plan output (room dimensions, opposite plane-backed walls; gate: within 1 cm or 0.5%)")
    for a, b in PAIRS:
        r = compare(caps[a], caps[b], cfg)
        res[f"{b}->{a}"] = r
        d = np.array([x["delta"] for x in r["room_dims"]]); w = np.array([x["width_a"] for x in r["room_dims"]])
        pd_ = np.array([x["delta"] for x in r["plane_pairs"]]); pw = np.array([x["width_a"] for x in r["plane_pairs"]])
        all_d += list(d); all_w += list(w); all_pd += list(pd_); all_pw += list(pw)
        print(f"  {b:>11s} vs {a:<12s} rooms matched {len(r['rooms_matched'])}:  room dims {stats(d, w)}   | M3 plane pairs {stats(pd_, pw)}")
    print(f"  POOLED                                   room dims {stats(np.array(all_d), np.array(all_w))}   | M3 plane pairs {stats(np.array(all_pd), np.array(all_pw))}")
    print("\nWALL LENGTHS (same wall in both captures) and CALIBRATION: does the interval of the difference (root-sum-square of the two half-widths) contain it?")
    pooled = {"walls": [], "room_areas": [], "openings": []}
    for a, b in PAIRS:
        r = res[f"{b}->{a}"]
        line = f"  {b:>11s} vs {a:<12s}"
        for key, wkey, unit in (("walls", "length_a", "m"), ("room_areas", "area_a", "m2"), ("openings", "width_a", "m")):
            x = r[key]
            pooled[key] += x
            if not x:
                line += f" | {key}: none"
                continue
            d = np.array([v["delta"] for v in x]); h = np.array([v["halfwidth"] for v in x]); w = np.array([v[wkey] for v in x])
            g = f", gate {int(gate(d, w).sum())}/{len(d)}" if key == "walls" else ""
            line += f" | {key}: n {len(d)}, median |d| {np.median(np.abs(d)) * (100 if unit == 'm' else 1):.2f} {'cm' if unit == 'm' else 'm2'}{g}, inside interval {int((np.abs(d) <= h).sum())}/{len(d)}"
        print(line + f" | openings detected {r['n_openings'][1]} vs {r['n_openings'][0]}")
    cal = {}
    for key, wkey in (("walls", "length_a"), ("room_areas", "area_a"), ("openings", "width_a")):
        x = pooled[key]
        if x:
            d = np.array([v["delta"] for v in x]); h = np.array([v["halfwidth"] for v in x])
            cal[key] = dict(n=len(x), inside=float((np.abs(d) <= h).mean()), median_abs=float(np.median(np.abs(d))),
                            gate_pass=float(gate(d, np.array([v[wkey] for v in x])).mean()) if key == "walls" else None)
            print(f"  POOLED {key}: n {len(x)}, inside interval {100 * cal[key]['inside']:.0f}% (intervals are ~2 sigma: about 95% expected if calibrated)"
                  + (f", per-wall gate {100 * cal[key]['gate_pass']:.0f}%" if key == "walls" else ""))
    summary = dict(variant=variant, overrides=VARIANTS[variant], pooled_room_dims=dict(n=len(all_d), pass_rate=float(gate(np.array(all_d), np.array(all_w)).mean()) if all_d else None,
                   median_abs_cm=float(np.median(np.abs(all_d)) * 100) if all_d else None),
                   pooled_plane_pairs=dict(n=len(all_pd), pass_rate=float(gate(np.array(all_pd), np.array(all_pw)).mean()) if all_pd else None,
                                           median_abs_cm=float(np.median(np.abs(all_pd)) * 100) if all_pd else None),
                   calibration=cal, pairs=res, timing_s={n: c["plan"].timing_s for n, c in caps.items()})
    (out / "benchmark.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out / 'benchmark.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="final")
    main(ap.parse_args().variant)
