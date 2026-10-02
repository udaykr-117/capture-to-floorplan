"""M7: top-down pictures of the worst width pairs (A grey, B registered red, matched planes as lines) to SEE why two captures disagree.
Usage: uv run python scripts/m7_look.py  -> out/m7_look_<k>.png"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, OUT, load
from floorplan.register import match_segments, register, segments, to_other, width_pairs

caps = {n: load(n) for n in ("with_ceiling", "floor_only")}
A, B = caps["with_ceiling"], caps["floor_only"]
reg = register(A["W"], B["W"], CFG)
SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
m = match_segments(SA, SB, CFG)
ps = sorted(width_pairs(SA, SB, m, CFG), key=lambda p: -abs(p["delta"]))[:6]
Bw = reg.apply(B["W"])
fig, axs = plt.subplots(2, 3, figsize=(18, 12))
for ax, p in zip(axs.ravel(), ps):
    ia = [i for i, s in enumerate(SA) if s.axis == p["axis"] and s.offset in (p["a_lo"], p["a_hi"])]
    pr = [(i, j) for i, j in m if i in ia]
    s0 = max(SA[i].s0 for i, _ in pr); s1 = min(SA[i].s1 for i, _ in pr)
    ax_ = p["axis"]
    if ax_ == 0:
        box = (p["a_lo"] - 0.5, p["a_hi"] + 0.5, s0 - 0.5, s1 + 0.5)
    else:
        box = (s0 - 0.5, s1 + 0.5, p["a_lo"] - 0.5, p["a_hi"] + 0.5)
    for P, c, lab in ((A["W"], "0.4", "with_ceiling"), (Bw, "red", "floor_only (registered)")):
        k = (P[:, 0] > box[0]) & (P[:, 0] < box[1]) & (P[:, 1] > box[2]) & (P[:, 1] < box[3])
        ax.scatter(P[k, 0], P[k, 1], s=0.5, c=c, alpha=0.5, label=lab)
    for i, j in pr:
        for S, idx, c in ((SA, i, "k"), (SB, j, "r")):
            s = S[idx]
            if s.axis == 0:
                ax.plot([s.offset, s.offset], [s.s0, s.s1], c=c, lw=1)
            else:
                ax.plot([s.s0, s.s1], [s.offset, s.offset], c=c, lw=1)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_title(f"width {p['width_a']:.2f} m, floor_only - with_ceiling = {100 * p['delta']:+.1f} cm", fontsize=10)
axs[0, 0].legend(markerscale=10, fontsize=8)
fig.savefig(OUT / "m7_look_worst.png", dpi=70, bbox_inches="tight")
print("wrote out/m7_look_worst.png")
