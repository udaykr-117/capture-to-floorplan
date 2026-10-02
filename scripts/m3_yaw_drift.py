"""Is the 3 deg yaw between single_room and floor_only (M3) drift inside floor_only?

Register floor_only onto with_ceiling using only the wall points inside the area that single_room covers, and compare the
rotation with the global registration. If floor_only is rotated locally, the local residual differs from the global one.
Usage: uv run python scripts/m3_yaw_drift.py   (uses the M3 cache; run m3_repeatability.py first)
"""
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

import m3_repeatability as m3
from floorplan.register import register, to_other_polygon

CFG = m3.CFG


def main():
    wc, fo, sr = (m3.captured(n) for n in ("with_ceiling", "floor_only", "single_room"))
    g_fo = register(wc["W"], fo["W"], CFG)
    g_sr = register(wc["W"], sr["W"], CFG)
    print(f"global: floor_only -> with_ceiling total yaw {np.degrees(np.arctan2(g_fo.R[1, 0], g_fo.R[0, 0])):.2f} deg (residual {g_fo.yaw_resid_deg:+.2f}); "
          f"single_room -> with_ceiling residual {g_sr.yaw_resid_deg:+.2f}")
    region = unary_union([to_other_polygon(Polygon(r.polygon), g_sr) for r in sr["plan"].rooms]).buffer(0.5)  # single_room's area, in with_ceiling's frame
    A = wc["W"][contains_xy(region, wc["W"][:, 0], wc["W"][:, 1])]
    Bw = g_fo.apply(fo["W"])
    in_region = contains_xy(region, Bw[:, 0], Bw[:, 1])
    print(f"wall points inside single_room's area: with_ceiling {len(A):,}, floor_only {int(in_region.sum()):,}")
    # B in A's frame already; register the local subsets starting from identity (k = 0)
    loc = register(A, Bw[in_region], CFG)
    th = np.degrees(np.arctan2(loc.R[1, 0], loc.R[0, 0]))
    print(f"local registration of floor_only onto with_ceiling inside that area: extra rotation {th:+.2f} deg (rot {90 * loc.rot_k} + residual {loc.yaw_resid_deg:+.2f}), "
          f"shift ({loc.t[0]:+.2f}, {loc.t[1]:+.2f}) m, {100 * loc.fitness_5cm:.0f}% within 5 cm, median {loc.median_cm:.1f} cm (null {100 * loc.null_fitness_5cm:.0f}%)")
    print("expected if floor_only is rotated locally: about the difference of the global residuals (single_room->floor_only -3.18 vs single_room->with_ceiling -0.26 = +2.9 deg)")
    # same for the rest of the house: floor_only points outside single_room's area
    out = ~in_region
    A2 = wc["W"][~contains_xy(region, wc["W"][:, 0], wc["W"][:, 1])]
    rest = register(A2, Bw[out], CFG)
    th2 = np.degrees(np.arctan2(rest.R[1, 0], rest.R[0, 0]))
    print(f"outside that area: extra rotation {th2:+.2f} deg, shift ({rest.t[0]:+.2f}, {rest.t[1]:+.2f}) m, {100 * rest.fitness_5cm:.0f}% within 5 cm, median {rest.median_cm:.1f} cm")


if __name__ == "__main__":
    main()
