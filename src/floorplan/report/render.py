"""Render a Plan (from its JSON content only) to an image: rooms, wall lengths, openings, areas, ceilings, with intervals."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

from floorplan.schema import Plan


def _pm(m, fmt="{:.2f}") -> str:
    hw = (m.interval.high - m.interval.low) / 2
    return f"{fmt.format(m.value)} ± {fmt.format(hw)}"


def render_plan(plan: Plan, path: str | Path) -> None:
    cmap = plt.get_cmap("tab20")
    xs = [p[0] for r in plan.rooms for p in r.polygon] or [0, 1]
    zs = [p[1] for r in plan.rooms for p in r.polygon] or [0, 1]
    w, h = max(xs) - min(xs), max(zs) - min(zs)
    fig, ax = plt.subplots(figsize=(10, max(6.0, 10 * h / max(w, 1e-6))))
    for i, r in enumerate(plan.rooms):
        poly = Polygon(r.polygon)
        col = cmap(i % 20)
        x, z = poly.exterior.xy
        ax.fill(x, z, color=col, alpha=0.22, lw=0)
        for wl in r.walls:
            (x0, z0), (x1, z1) = wl.p0, wl.p1
            ax.plot([x0, x1], [z0, z1], color="k" if wl.source == "plane" else "0.45", lw=2.2 if wl.source == "plane" else 1.4,
                    ls="-" if wl.source == "plane" else "--", solid_capstyle="butt")
            L = float(np.hypot(x1 - x0, z1 - z0))
            if L >= 0.6:
                mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
                nx, nz = (z1 - z0) / L, -(x1 - x0) / L
                if not poly.contains(Point(mx + 0.15 * nx, mz + 0.15 * nz)):
                    nx, nz = -nx, -nz
                ax.text(mx + 0.28 * nx, mz + 0.28 * nz, _pm(wl.length), fontsize=6.5, ha="center", va="center", color="0.15",
                        rotation=0 if abs(x1 - x0) > abs(z1 - z0) else 90)
        c = poly.representative_point()
        ceil = f"ceiling {_pm(r.ceiling_height)} m" if r.ceiling_height.value is not None else "ceiling: unmeasurable"
        if r.ceiling_height.status == "ambiguous":
            ceil += " (ambiguous)"
        ax.text(c.x, c.y, f"{r.id}\n{_pm(r.area, '{:.1f}')} m²\n{ceil}", ha="center", va="center", fontsize=8, weight="bold")
    for o in plan.openings:
        ax.plot([o.p0[0], o.p1[0]], [o.p0[1], o.p1[1]], color="red" if o.kind == "door" else "orange", lw=5, solid_capstyle="butt")
        ax.text((o.p0[0] + o.p1[0]) / 2, (o.p0[1] + o.p1[1]) / 2, _pm(o.width), fontsize=6, color="darkred", ha="center", va="bottom")
    fa = plan.stitched.footprint_area
    ax.set_title(f"{plan.capture}  |  tier {plan.tier}  |  {plan.stitched.n_rooms} rooms  |  footprint {_pm(fa, '{:.1f}')} m²", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x' (m)")
    ax.set_ylabel("z' (m)")
    fig.text(0.5, 0.005, "provisional, uncalibrated intervals  |  accuracy untested (no ground truth)  |  damage / scope not implemented  |  dashed = no wall plane",
             ha="center", fontsize=7, color="0.3")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
