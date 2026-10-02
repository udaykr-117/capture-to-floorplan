"""Zoom on one block of the property: wall points (0.3-1.6 m slab) of all three captures registered onto with_ceiling, plus room outlines.
Usage: uv run python scripts/seg_zoom.py x0 x1 z0 z1 name"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, OUT, load
from floorplan.register import register, to_other_polygon

ROOT = Path(__file__).resolve().parents[1]
x0, x1, z0, z1 = map(float, sys.argv[1:5])
names = ("with_ceiling", "floor_only", "single_room")
cols = {"with_ceiling": "k", "floor_only": "r", "single_room": "b"}
plans = {n: json.loads((ROOT / "bench" / "results" / "after" / f"{n}.json").read_text()) for n in names}
caps = {n: load(n) for n in names}
fig, axs = plt.subplots(1, 3, figsize=(24, 9))
for ax, n in zip(axs, names):
    reg = None if n == "with_ceiling" else register(caps["with_ceiling"]["W"], caps[n]["W"], CFG)
    W = caps[n]["W"] if reg is None else reg.apply(caps[n]["W"])
    k = (W[:, 0] > x0) & (W[:, 0] < x1) & (W[:, 1] > z0) & (W[:, 1] < z1)
    ax.scatter(W[k, 0], W[k, 1], s=0.6, c="0.5")
    for r in plans[n]["rooms"]:
        p = Polygon(r["polygon"]) if reg is None else to_other_polygon(Polygon(r["polygon"]), reg)
        x, y = p.exterior.xy
        ax.plot(x, y, c=cols[n], lw=2)
        if x0 < p.centroid.x < x1 and z0 < p.centroid.y < z1:
            ax.text(p.centroid.x, p.centroid.y, f"{r['id']} {p.area:.1f}", color=cols[n], fontsize=11)
    for o in plans[n]["openings"]:
        import numpy as np
        q = np.array([o["p0"], o["p1"]], float) if reg is None else reg.apply(np.array([o["p0"], o["p1"]], float))
        ax.plot(q[:, 0], q[:, 1], c="orange", lw=6)
    ax.set_xlim(x0, x1); ax.set_ylim(z0, z1); ax.set_aspect("equal"); ax.set_title(n)
fig.savefig(OUT / f"seg_zoom_{sys.argv[5]}.png", dpi=65, bbox_inches="tight")
print("ok")
