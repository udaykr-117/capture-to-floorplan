"""Wall-plane runs that cross the INSIDE of a room polygon (a partition the room was merged across). Uses the m7 cache (planes) and the
plans in bench/results/after. Usage: uv run python scripts/seg_interior.py"""
import json
import sys
from pathlib import Path

from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import load

ROOT = Path(__file__).resolve().parents[1]
for n in ("with_ceiling", "floor_only", "single_room"):
    plan = json.loads((ROOT / "bench" / "results" / "after" / f"{n}.json").read_text())
    planes = load(n)["planes"]
    print(f"\n{n}:")
    for r in plan["rooms"]:
        poly = Polygon(r["polygon"])
        inner = poly.buffer(-0.25)
        for p in planes:
            for run in p.runs:
                line = LineString([(p.offset, run.s0), (p.offset, run.s1)] if p.axis == 0 else [(run.s0, p.offset), (run.s1, p.offset)])
                L = line.intersection(inner).length
                if L >= 0.5:
                    minx, minz, maxx, maxz = poly.bounds
                    span = (maxz - minz) if p.axis == 0 else (maxx - minx)
                    print(f"  {r['id']} ({poly.area:.1f} m2): plane {'x' if p.axis == 0 else 'z'}={p.offset:.2f}, run {run.s0:.2f}..{run.s1:.2f} "
                          f"(coverage {run.coverage:.2f}), {L:.2f} m inside the room = {100 * L / span:.0f}% of the room's span across")
