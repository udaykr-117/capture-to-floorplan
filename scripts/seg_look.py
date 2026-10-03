"""Room polygons of floor_only and single_room registered onto with_ceiling, to SEE where rooms are cut differently.
Usage: uv run python scripts/seg_look.py  -> out/seg_look.png (uses the cached plans of m7_diagnose for wall points)"""
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
plans = {n: json.loads((ROOT / "bench" / "results" / (sys.argv[1] if len(sys.argv) > 1 else "refined") / f"{n}.json").read_text()) for n in ("with_ceiling", "floor_only", "single_room")}
caps = {n: load(n) for n in plans}
fig, axs = plt.subplots(1, 2, figsize=(22, 11))
for ax, b in zip(axs, ("floor_only", "single_room")):
    reg = register(caps["with_ceiling"]["W"], caps[b]["W"], CFG)
    W = caps["with_ceiling"]["W"]
    ax.scatter(W[:, 0], W[:, 1], s=0.2, c="0.75")
    for r in plans["with_ceiling"]["rooms"]:
        p = Polygon(r["polygon"]); x, y = p.exterior.xy
        ax.plot(x, y, c="k", lw=2); ax.text(p.centroid.x, p.centroid.y + 0.25, f"wc {r['id']}\n{p.area:.1f}", ha="center", fontsize=9)
    for r in plans[b]["rooms"]:
        p = to_other_polygon(Polygon(r["polygon"]), reg); x, y = p.exterior.xy
        ax.plot(x, y, c="r", lw=1.5, ls="--"); ax.text(p.centroid.x, p.centroid.y - 0.35, f"{b[:2]} {r['id']}\n{p.area:.1f}", ha="center", fontsize=9, color="r")
    for o in plans["with_ceiling"]["openings"]:
        ax.plot([o["p0"][0], o["p1"][0]], [o["p0"][1], o["p1"][1]], c="blue", lw=5)
    ax.set_aspect("equal"); ax.set_title(f"with_ceiling rooms (black, openings blue) vs {b} rooms registered (red dashed)")
fig.savefig(OUT / f"seg_look_{sys.argv[1] if len(sys.argv) > 1 else 'refined'}.png", dpi=60, bbox_inches="tight")
print("wrote out/seg_look.png")
