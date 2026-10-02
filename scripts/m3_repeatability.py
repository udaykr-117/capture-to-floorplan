"""M3: register the three samples onto each other and measure how well they agree.

Prints, per capture pair: the registration (with a wrong-rotation null), wall-segment matches and offset differences,
facing-wall width disagreement by baseline length (shift-independent), room overlaps, and ceilings transferred to rooms
that one capture missed. Results go to out/m3_repeatability.json and out/m3_<pair>.png.
Capture plans are cached in out/cache keyed by the config hash: delete out/cache after changing the pipeline code.
Usage: uv run python scripts/m3_repeatability.py
"""
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from floorplan.config import load_config
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import (fuse_ceilings, match_rooms, match_segments, register, segments, summarize_widths, to_other, wall_points, width_pairs)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {
    "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6",
    "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
    "single_room": ROOT / "single_room" / "c00a170fe1",
}
CFG = load_config()


def captured(name):
    import hashlib

    key = hashlib.md5(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m3_{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    plan, inter = build_plan(StraySource(SAMPLES[name], CFG), CFG)
    d = dict(plan=plan, planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], CFG))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d


def compare(name_a, name_b, A, B):
    print(f"\n=================== {name_b}  ->  {name_a}  (A = {name_a}, B = {name_b}) ===================")
    reg = register(A["W"], B["W"], CFG)
    th = np.degrees(np.arctan2(reg.R[1, 0], reg.R[0, 0]))
    print(f"registration: rotation {90 * reg.rot_k} deg + residual {reg.yaw_resid_deg:+.2f} deg (total {th:.2f}), shift ({reg.t[0]:.2f}, {reg.t[1]:.2f}) m")
    print(f"  wall points of B within 5 cm of A: {100 * reg.fitness_5cm:.1f}% (median distance {reg.median_cm:.1f} cm, rmse of those {reg.rmse_cm:.1f} cm)")
    print(f"  wrong-rotation null ({90 * ((reg.rot_k + 2) % 4)} deg): {100 * reg.null_fitness_5cm:.1f}% within 5 cm, median {reg.null_median_cm:.1f} cm  -> accepted: {reg.accepted}")
    SA = segments(A["planes"])
    SB = [to_other(s, reg) for s in segments(B["planes"])]
    m = match_segments(SA, SB, CFG)
    d = np.array([SB[j].offset - SA[i].offset for i, j in m]) * 100
    tilt = np.array([SB[j].tilt_deg for i, j in m])
    print(f"wall segments: A {len(SA)}, B {len(SB)}, matched {len(m)} ({100 * len(m) / len(SB):.0f}% of B)")
    if len(m):
        print(f"  offset difference after registration (cm): median |d| {np.median(np.abs(d)):.1f}, 90th pct {np.percentile(np.abs(d), 90):.1f}, max {np.abs(d).max():.1f}; "
              f"B segment tilt vs A axes: median {np.median(tilt):.2f} deg, max {tilt.max():.2f} deg")
    pairs = width_pairs(SA, SB, m, CFG)
    rows = summarize_widths(pairs, CFG)
    print(f"facing-wall widths ({len(pairs)} pairs; width in A vs B; independent of the fitted shift):")
    print("   baseline (m)    n   median|d| p90|d| (cm)   median signed (%)   within 1 cm   within 0.5%   either")
    for r in rows:
        label = f"{r['lo']:>4.1f}..{r['hi']:<4.1f}" if r is not rows[-1] else "     all  "
        if r["n"] == 0:
            print(f"   {label}      0")
        else:
            print(f"   {label}   {r['n']:4d}   {r['median_abs_cm']:6.1f} {r['p90_abs_cm']:6.1f}          {r['median_signed_pct']:+6.2f}          "
                  f"{100 * r['within_1cm']:5.0f}%        {100 * r['within_0p5pct']:5.0f}%       {100 * r['within_either']:5.0f}%")
    if pairs:
        k, sp = CFG["intervals"]["sigma_k"], CFG["intervals"]["plane_position_sigma_m"]
        z_old = np.array([abs(p["delta"]) / (k * np.hypot(p["sig_a"], p["sig_b"])) for p in pairs])
        z_new = np.array([abs(p["delta"]) / (k * np.sqrt(np.hypot(p["sig_a"], 2 ** 0.5 * sp) ** 2 + np.hypot(p["sig_b"], 2 ** 0.5 * sp) ** 2)) for p in pairs])
        print(f"interval check on these pairs: fit-spread-only intervals contain the A-vs-B width difference in {100 * (z_old <= 1).mean():.0f}% of pairs; "
              f"with the {100 * sp:.1f} cm systematic term {100 * (z_new <= 1).mean():.0f}% (circular: the term was fitted on these same pairs; ~95% is the target)")
        allrow = rows[-1]
        print(f"REPEATABILITY GATE (within 1 cm or 0.5% per wall): {int(round(allrow['within_either'] * len(pairs)))} of {len(pairs)} facing-wall widths pass ({100 * allrow['within_either']:.0f}%)")
    pa = {r.id: Polygon(r.polygon) for r in A["plan"].rooms}
    pb = {r.id: Polygon(r.polygon) for r in B["plan"].rooms}
    rm = match_rooms(pa, pb, reg)
    print("room overlaps (B room moved into A's frame): B  A   area_B  area_A  inter  IoU   B-in-A  A-in-B")
    for r in rm:
        print(f"      {r['b']:>3s} {r['a']:>3s}   {r['area_b']:6.1f}  {r['area_a']:6.1f}  {r['inter']:5.1f}  {r['iou']:4.2f}   {r['b_in_a']:4.2f}   {r['a_in_b']:4.2f}")
    fused, copied = fuse_ceilings(B["plan"], A["plan"], reg, CFG)
    print(f"ceilings transferred to rooms of {name_b} (qualifying: >= {int(100 * CFG['register']['room_inside_fraction'])}% inside one {name_a} room with a ceiling):")
    for c in copied:
        print(f"      {c['room']} <- {name_a} {c['from_room']}: {c['height']:.3f} m ({c['status']}), {100 * c['inside']:.0f}% inside")
    if not copied:
        print("      none")
    else:
        (OUT / f"m3_fused_{name_b}_from_{name_a}.json").write_text(fused.model_dump_json(indent=2))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(A["W"][:, 0], A["W"][:, 1], s=0.3, c="0.6", label=f"{name_a} walls")
    Bt = reg.apply(B["W"])
    ax.scatter(Bt[:, 0], Bt[:, 1], s=0.3, c="red", alpha=0.4, label=f"{name_b} walls (registered)")
    ax.set_aspect("equal"); ax.legend(markerscale=15); ax.set_title(f"{name_b} on {name_a}: {100 * reg.fitness_5cm:.0f}% within 5 cm")
    fig.savefig(OUT / f"m3_{name_b}_on_{name_a}.png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return dict(pair=f"{name_b}->{name_a}", registration=dict(rot_deg=90 * reg.rot_k, yaw_resid_deg=reg.yaw_resid_deg, fitness_5cm=reg.fitness_5cm, median_cm=reg.median_cm,
                                                              null_fitness_5cm=reg.null_fitness_5cm, accepted=bool(reg.accepted)),
                n_matched=len(m), n_b=len(SB), offset_diff_cm=d.tolist(), width_rows=rows, width_pairs=pairs, room_overlaps=rm, ceilings_copied=copied)


def main():
    OUT.mkdir(exist_ok=True)
    caps = {n: captured(n) for n in SAMPLES}
    results = [compare("with_ceiling", "floor_only", caps["with_ceiling"], caps["floor_only"]),
               compare("with_ceiling", "single_room", caps["with_ceiling"], caps["single_room"]),
               compare("floor_only", "single_room", caps["floor_only"], caps["single_room"])]
    (OUT / "m3_repeatability.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nwrote out/m3_repeatability.json")


if __name__ == "__main__":
    main()
