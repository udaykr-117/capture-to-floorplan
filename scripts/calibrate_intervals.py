"""Calibrate the LiDAR-tier intervals for wall length and room area from repeat captures (the only reference we have: no ground truth).

For each wall (or room) seen in two captures, the difference d should lie inside the interval of the difference, sqrt(hw_a^2 + hw_b^2), about 95%
of the time. The benchmark shows it does far less often (`run_benchmark.py`, "inside interval"), because room extents differ between captures
(rooms cut at different places), which the plane-spread intervals do not model. This script fits ONE extra per-capture term s, added in
quadrature to every half-width, so that the conformal 95% quantile of the pairs is covered:  |d| <= sqrt(hw_a^2 + hw_b^2 + 2 s^2).
Walls: s in metres. Rooms: s as a fraction of the room area.
Leave-one-capture-pair-out: s is fitted on two pairs and the coverage is measured on the third (out of sample).
The fitted values go to configs/default.yaml (intervals.extent_halfwidth_m, intervals.extent_area_rel) by hand, with this output cited.
Usage: uv run python scripts/calibrate_intervals.py [bench/results/refined/benchmark.json]   (run on a benchmark made WITHOUT these terms, i.e. both set to 0)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVEL = 0.95


def needed(rows, scale_key=None):
    """Per pair, the smallest s that covers it (0 if already covered). For areas, s is divided by the mean area of the two rooms."""
    out = []
    for r in rows:
        s = math.sqrt(max(0.0, (r["delta"] ** 2 - r["halfwidth"] ** 2) / 2))
        if scale_key:
            s /= (r[scale_key[0]] + r[scale_key[1]]) / 2
        out.append(s)
    return np.array(out)


def conformal(s):
    n = len(s)
    k = min(n, math.ceil((n + 1) * LEVEL))
    return float(np.sort(s)[k - 1])


def robust(rows, scale_key=None):
    """2 x robust sigma of the per-capture extent error: per pair, the part of |d| the current interval does not explain, divided by sqrt(2)
    (two captures); sigma = 1.4826 x median (MAD about 0, since the error is symmetric with mean ~0). Gross mismatches (a room cut at a
    different wall) are treated as outliers instead of setting the width for every wall."""
    e = []
    for r in rows:
        x = math.sqrt(max(0.0, r["delta"] ** 2 - r["halfwidth"] ** 2)) / math.sqrt(2)
        if scale_key:
            x /= (r[scale_key[0]] + r[scale_key[1]]) / 2
        e.append(x)
    return 2 * 1.4826 * float(np.median(e))


def covered(rows, s, scale_key=None):
    ok = 0
    for r in rows:
        ss = s * ((r[scale_key[0]] + r[scale_key[1]]) / 2) if scale_key else s
        ok += abs(r["delta"]) <= math.sqrt(r["halfwidth"] ** 2 + 2 * ss ** 2)
    return ok, len(rows)


def main(path):
    b = json.loads(Path(path).read_text())
    for key, label, sk, unit in (("walls", "wall length", None, "m"), ("room_areas", "room area", ("area_a", "area_b"), "x area")):
        per_pair = {p: v[key] for p, v in b["pairs"].items() if v[key]}
        allr = [r for v in per_pair.values() for r in v]
        s_all = conformal(needed(allr, sk))
        print(f"\n{label}: {len(allr)} pairs; inside the current intervals {covered(allr, 0.0, sk)[0]}/{len(allr)}; "
              f"fitted extent term s = {s_all:.3f} {unit} -> in-sample coverage {covered(allr, s_all, sk)[0]}/{len(allr)}")
        for name, fit in (("conformal 95%", lambda rows: conformal(needed(rows, sk))), ("robust 2-sigma", lambda rows: robust(rows, sk))):
            s_fit = fit(allr)
            print(f"  {name:15s}: s = {s_fit:.3f} {unit}; in-sample coverage {covered(allr, s_fit, sk)[0]}/{len(allr)}")
            for held in per_pair:
                train = [r for p, v in per_pair.items() if p != held for r in v]
                if not train:
                    continue
                sh = fit(train)
                ok, n = covered(per_pair[held], sh, sk)
                print(f"     leave out {held:28s}: s = {sh:.3f} {unit}; held-out coverage {ok}/{n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "bench" / "results" / "refined" / "benchmark.json")
