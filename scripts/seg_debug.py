"""Debug view of the free-space layer and room labels around one area (with_ceiling by default), with connectors on.
Usage: uv run python scripts/seg_debug.py [capture] x0 x1 z0 z1"""
import copy
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, OUT, SAMPLES
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.rooms.freespace import segment_rooms

name = sys.argv[1]
x0, x1, z0, z1 = map(float, sys.argv[2:6])
cfg = copy.deepcopy(CFG)
cfg["rooms"]["split_connectors"] = True
f = OUT / "cache" / f"segdbg_{name}.pkl"
if f.exists():
    inter = pickle.loads(f.read_bytes())
else:
    _, inter = build_plan(StraySource(SAMPLES[name], cfg), cfg)
    inter = {k: inter[k] for k in ("layers", "labels", "grid")}
    f.write_bytes(pickle.dumps(inter))
g, free = inter["grid"], inter["layers"]["free"]
raw = segment_rooms(free, g, cfg)
ext = [g.x0, g.x0 + g.nx * g.cell, g.z0, g.z0 + g.nz * g.cell]
fig, axs = plt.subplots(1, 3, figsize=(24, 8))
for ax, img, t in zip(axs, (free, raw, inter["labels"]), ("free", "segment_rooms (connectors on)", "after clean_labels")):
    ax.imshow(np.ma.masked_equal(img.astype(float), 0) if t != "free" else img, origin="lower", extent=ext, cmap="tab20" if t != "free" else "gray", interpolation="nearest")
    ax.set_xlim(x0, x1); ax.set_ylim(z0, z1); ax.set_title(t)
print("labels in raw:", np.unique(raw).tolist(), "after clean:", np.unique(inter["labels"]).tolist())
fig.savefig(OUT / f"seg_debug_{name}.png", dpi=60, bbox_inches="tight")
