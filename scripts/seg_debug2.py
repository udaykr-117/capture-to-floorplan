"""Per label: cell area, rectangles won in the polygon step, final polygon area (connectors on). Usage: uv run python scripts/seg_debug2.py <capture>"""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, SAMPLES
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
import floorplan.rooms.polygon as P

cfg = copy.deepcopy(CFG)
cfg["rooms"]["split_connectors"] = True
orig = P.room_polygons
seen = {}


def spy(labels, planes, grid, cfg_):
    seen["labels"], seen["planes"], seen["grid"] = labels, planes, grid
    return orig(labels, planes, grid, cfg_)


import floorplan.pipeline as PL
PL.room_polygons = spy
plan, inter = build_plan(StraySource(SAMPLES[sys.argv[1]], cfg), cfg)
lab, g = seen["labels"], seen["grid"]
cfg2 = copy.deepcopy(cfg); cfg2["rooms"]["min_room_area_m2"] = 0.0
polys = {rp.id: rp.polygon.area for rp in orig(lab, seen["planes"], g, cfg2)}
for i in [i for i in np.unique(lab) if i > 0]:
    rr, cc = np.nonzero(lab == i)
    x, z = g.xz(rr, cc)
    print(f"label {i}: {len(rr) * g.cell ** 2:.2f} m2 of cells, x {x.min():.2f}..{x.max():.2f}, z {z.min():.2f}..{z.max():.2f}; polygon area {polys.get(i, 0):.2f} m2 (min room area {cfg['rooms']['min_room_area_m2']})")
