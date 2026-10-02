"""Openings of all three captures in with_ceiling's frame, each with its nearest same-orientation opening in the other captures.
Usage: uv run python scripts/openings_xcap.py [results dir name]"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from m7_diagnose import CFG, load
from floorplan.register import register

ROOT = Path(__file__).resolve().parents[1]
var = sys.argv[1] if len(sys.argv) > 1 else "after2a"
names = ("with_ceiling", "floor_only", "single_room")
caps = {n: load(n) for n in names}
ops = []
for n in names:
    plan = json.loads((ROOT / "bench" / "results" / var / f"{n}.json").read_text())
    reg = None if n == "with_ceiling" else register(caps["with_ceiling"]["W"], caps[n]["W"], CFG)
    for o in plan["openings"]:
        q = np.array([o["p0"], o["p1"]], float)
        q = q if reg is None else reg.apply(q)
        ops.append(dict(cap=n, id=o["id"], kind=o["kind"], c=q.mean(0), d=q[1] - q[0], w=o["width"]["value"], bot=o["bottom_m"], top=o["top_m"], rooms=o["rooms"]))
for o in ops:
    best = []
    for p in ops:
        if p["cap"] == o["cap"]:
            continue
        same = abs(abs(np.dot(o["d"], p["d"])) / (np.linalg.norm(o["d"]) * np.linalg.norm(p["d"])) - 1) < 0.05
        dist = np.linalg.norm(o["c"] - p["c"])
        if same:
            best.append((dist, p))
    b = min(best, key=lambda t: t[0]) if best else None
    m = f"nearest {b[1]['cap']} {b[1]['id']} at {b[0]:.2f} m (w {b[1]['w']:.2f})" if b else "none"
    print(f"{o['cap']:13s} {o['id']} {o['kind']:6s} centre ({o['c'][0]:5.2f},{o['c'][1]:5.2f}) w {o['w']:.2f} z {o['bot']:.2f}-{o['top']:.2f} rooms {o['rooms']}  | {m}")
