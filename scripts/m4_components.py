"""M4 component ablation: which part of the drift correction helps? Variants are cumulative, each run on all three samples.

off | yaw | yaw+floor | +jump | +shift (= the full correction). Reports footprint, path start-end gap, wall sharpness and cross-capture agreement.
Chunks are cached per (sample, jump on/off) in out/cache; results per variant too. Delete out/cache after code changes.
Usage: uv run python scripts/m4_components.py
"""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from floorplan.config import load_config
from floorplan.drift.chunks import build_chunks
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import match_segments, register, segments, summarize_widths, to_other, wall_points, width_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {"with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "single_room": ROOT / "single_room" / "c00a170fe1"}
BASE = load_config()
# name: (drift.enabled, jump_correction, use_yaw, use_floor, use_shift)
VARIANTS = {
    "off": (False, False, False, False, False),
    "yaw": (True, False, True, False, False),
    "yaw+floor": (True, False, True, True, False),
    "yaw+floor+jump": (True, True, True, True, False),
    "yaw+floor+jump+shift": (True, True, True, True, True),
}
PAIRS = (("floor_only", "with_ceiling"), ("single_room", "with_ceiling"), ("single_room", "floor_only"))


def cfg_for(v):
    e, j, y, f, s = VARIANTS[v]
    c = copy.deepcopy(BASE)
    c["drift"].update(enabled=e, jump_correction=j, use_yaw=y, use_floor=f, use_shift=s)
    return c


def chunks_for(name, jump):
    c = cfg_for("yaw+floor+jump" if jump else "off")
    key = hashlib.md5(json.dumps(dict(s=c["stray"], d=c["drift"]["chunk_frames"], j=jump), sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m4c_chunks_{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    ch = build_chunks(StraySource(SAMPLES[name], c).frames(), c)
    f.write_bytes(pickle.dumps(ch))
    return ch


def run(name, v):
    c = cfg_for(v)
    key = hashlib.md5(json.dumps(c, sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m4c_{name}_{v}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    src = StraySource(SAMPLES[name], c)
    plan, inter = build_plan(src, c, drift=VARIANTS[v][0], chunks=chunks_for(name, VARIANTS[v][1]))
    fr, corr, chunks = inter["al"].frame, inter["corr"], inter["chunks"]
    first = corr.apply(0, fr.to_aligned(chunks[0].cams))[:5].mean(0)
    last = corr.apply(len(chunks) - 1, fr.to_aligned(chunks[-1].cams))[-5:].mean(0)
    d = dict(plan=plan, planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], c), gap=float(np.linalg.norm((first - last)[[0, 2]])),
             sigma=float(np.median([p.sigma * 100 for p in inter["planes"]])))
    f.write_bytes(pickle.dumps(d))
    return d


def cross(A, B, cfg):
    reg = register(A["W"], B["W"], cfg)
    SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
    m = match_segments(SA, SB, cfg)
    pairs = width_pairs(SA, SB, m, cfg)
    return reg, summarize_widths(pairs, cfg)[-1], len(pairs)


def main():
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    res = {(n, v): run(n, v) for v in VARIANTS for n in SAMPLES}
    print("\n=== per capture: footprint m2 / path start-end gap cm / median plane sigma cm ===")
    print(f"{'variant':24s} " + "  ".join(f"{n:>28s}" for n in SAMPLES))
    for v in VARIANTS:
        cells = [f"{res[(n, v)]['plan'].stitched.footprint_area.value:6.1f} / {res[(n, v)]['gap'] * 100:5.1f} / {res[(n, v)]['sigma']:4.2f}" for n in SAMPLES]
        print(f"{v:24s} " + "  ".join(f"{c:>28s}" for c in cells))
    print("\n=== cross-capture: wall pts within 5 cm | yaw residual | widths within 1 cm or 0.5% | median abs diff cm | pairs ===")
    print(f"{'variant':24s} " + "  ".join(f"{b + '->' + a:>40s}" for b, a in PAIRS))
    out = []
    for v in VARIANTS:
        cells = []
        for b, a in PAIRS:
            reg, row, n_pairs = cross(res[(a, v)], res[(b, v)], BASE)
            cells.append(f"{100 * reg.fitness_5cm:4.1f}% | {reg.yaw_resid_deg:+5.2f}d | {100 * row.get('within_either', 0):3.0f}% | {row.get('median_abs_cm', float('nan')):4.1f} | {n_pairs:2d}")
            out.append(dict(variant=v, pair=f"{b}->{a}", fit=reg.fitness_5cm, yaw=reg.yaw_resid_deg, within=row.get("within_either"), med=row.get("median_abs_cm"), pairs=n_pairs))
        print(f"{v:24s} " + "  ".join(f"{c:>40s}" for c in cells))
    (OUT / "m4_components.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
