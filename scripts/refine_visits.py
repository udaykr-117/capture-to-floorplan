"""For each room of a capture: every visit (run of consecutive chunks with the cameras inside), and the room's dimensions measured from that
visit alone. Shows whether the 'longest visit' choice of rooms/refine.py picks a visit that disagrees with the others.
Usage: uv run python scripts/refine_visits.py <capture> [room id]"""
import sys
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, SAMPLES
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.rooms.refine import _mode_offset

name = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
plan, inter = build_plan(StraySource(SAMPLES[name], CFG), CFG)
fr, corr, chunks = inter["al"].frame, inter["corr"], inter["chunks"]
rc, w = CFG["refine"], CFG["walls"]
cams = [corr.apply(c.idx, fr.to_aligned(c.cams)) for c in chunks]
pts = [corr.apply(c.idx, fr.to_aligned(c.pts)) for c in chunks]
for rp in inter["polys"]:
    rid = f"R{int(rp.id)}"
    if only and rid != only:
        continue
    inside = [float(shapely.contains_xy(rp.polygon, c[:, 0], c[:, 2]).mean()) >= rc["chunk_inside_min"] for c in cams]
    visits, cur = [], []
    for k, v in enumerate(inside):
        if v:
            cur.append(k)
        elif cur:
            visits.append(cur); cur = []
    if cur:
        visits.append(cur)
    edges = [e for e in rp.edges if e.source == "plane"]
    print(f"\n{name} {rid}: visits {[(v[0], v[-1]) for v in visits]}")
    for v in visits:
        P = np.concatenate([pts[k] for k in v]); P = P[(P[:, 1] > w["slab_above_floor_m"][0]) & (P[:, 1] < w["slab_above_floor_m"][1])]
        offs = {}
        for e in edges:
            along = 2 if e.axis == 0 else 0
            m = (np.abs(P[:, e.axis] - e.offset) < rc["window_m"]) & (P[:, along] > e.s0 + rc["end_margin_m"]) & (P[:, along] < e.s1 - rc["end_margin_m"])
            r = _mode_offset(P[m, e.axis], e.offset, rc["window_m"]) if m.sum() >= rc["min_points"] else None
            offs[(e.axis, round(e.offset, 3))] = None if r is None else r[0]
        dims = []
        for ax in (0, 2):
            got = sorted((o, v2) for (a, o), v2 in offs.items() if a == ax and v2 is not None)
            if len(got) >= 2:
                dims.append(f"{'x' if ax == 0 else 'z'} {got[-1][1] - got[0][1]:.3f} m (global planes {got[-1][0] - got[0][0]:.3f})")
        print(f"  visit chunks {v[0]}-{v[-1]} ({len(v)}): {', '.join(dims) or 'walls not seen'}")
