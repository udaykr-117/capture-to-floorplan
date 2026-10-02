"""For each pair of adjacent rooms in a plan: the shared boundary, and how much of it is a wall plane (source 'plane', support) vs a virtual cut.
Usage: uv run python scripts/seg_boundaries.py [results dir]"""
import json
import sys
from pathlib import Path

from shapely.geometry import LineString, Polygon

ROOT = Path(__file__).resolve().parents[1]
d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "results" / "after"
for n in ("with_ceiling", "floor_only", "single_room"):
    plan = json.loads((d / f"{n}.json").read_text())
    rooms = plan["rooms"]
    print(f"\n{n}:")
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            pb = Polygon(b["polygon"]).buffer(0.35)
            shared = [(w, LineString([w["p0"], w["p1"]]).intersection(pb).length) for w in a["walls"]]
            shared = [(w, L) for w, L in shared if L > 0.3]
            if not shared:
                continue
            tot = sum(L for _, L in shared)
            plane = sum(L * w["support"] for w, L in shared if w["source"] == "plane")
            ops = [o["id"] for o in plan["openings"] if set(o["rooms"]) == {a["id"], b["id"]}]
            print(f"  {a['id']}-{b['id']}: shared boundary {tot:.2f} m, wall-plane support {plane:.2f} m ({100 * plane / tot:.0f}%), "
                  f"edges {[(w['id'], w['source'], round(w['support'], 2), round(L, 2)) for w, L in shared]}, openings {ops}")
