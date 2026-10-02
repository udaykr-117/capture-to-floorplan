"""M4 ablation: every sample with drift correction OFF and ON, compared on footprint, wall sharpness, path closure and cross-capture agreement.

Usage: uv run python scripts/m4_ablation.py     (results cached in out/cache by config hash + mode; delete out/cache after code changes)
OFF = odometry used as-is. ON = relocalization jumps spread back along the path + per-chunk heading / shift / floor correction.
"""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from floorplan.config import load_config
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import register, segments, summarize_widths, to_other, match_segments, wall_points, width_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {"with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "single_room": ROOT / "single_room" / "c00a170fe1"}
CFG_ON = load_config()
CFG_OFF = copy.deepcopy(CFG_ON)
CFG_OFF["drift"]["enabled"] = False


def run(name, mode):
    cfg = CFG_ON if mode == "on" else CFG_OFF
    key = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m4_{name}_{mode}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    src = StraySource(SAMPLES[name], cfg)
    plan, inter = build_plan(src, cfg, drift=(mode == "on"))
    fr, corr, chunks = inter["al"].frame, inter["corr"], inter["chunks"]
    first = corr.apply(0, fr.to_aligned(chunks[0].cams))[:5].mean(0)
    last = corr.apply(len(chunks) - 1, fr.to_aligned(chunks[-1].cams))[-5:].mean(0)
    d = dict(plan=plan, planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], cfg), start_end_m=float(np.linalg.norm((first - last)[[0, 2]])),
             sigma_cm=[p.sigma * 100 for p in inter["planes"]], n_planes=len(inter["planes"]), diag=corr.diag)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d


def cross(A, B, cfg):
    reg = register(A["W"], B["W"], cfg)
    SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
    m = match_segments(SA, SB, cfg)
    pairs = width_pairs(SA, SB, m, cfg)
    rows = summarize_widths(pairs, cfg)
    allrow = rows[-1]
    return dict(fit=reg.fitness_5cm, med=reg.median_cm, null=reg.null_fitness_5cm, yaw=reg.yaw_resid_deg, matched=len(m), n_b=len(SB), pairs=len(pairs),
                med_abs=allrow.get("median_abs_cm"), p90=allrow.get("p90_abs_cm"), either=allrow.get("within_either"), signed=allrow.get("median_signed_pct"),
                deltas=[p["delta"] for p in pairs])


def main():
    res = {(n, m): run(n, m) for n in SAMPLES for m in ("off", "on")}
    print("\n=========== PER CAPTURE: drift correction OFF vs ON ===========")
    hdr = f"{'capture':14s} {'mode':4s} {'rooms':>5s} {'footprint m2':>13s} {'room areas m2':32s} {'planes':>6s} {'median plane sigma':>18s} {'path start-end gap':>18s}"
    print(hdr)
    for n in SAMPLES:
        for m in ("off", "on"):
            r = res[(n, m)]
            p = r["plan"]
            fa = p.stitched.footprint_area
            print(f"{n:14s} {m:4s} {len(p.rooms):5d} {fa.value:6.1f} +-{(fa.interval.high - fa.interval.low) / 2:4.1f}  {str([round(x.area.value, 1) for x in p.rooms]):32s} "
                  f"{r['n_planes']:6d} {np.median(r['sigma_cm']):15.2f} cm {r['start_end_m'] * 100:15.1f} cm")
    print("\n=========== DRIFT REPORT (ON) ===========")
    for n in SAMPLES:
        dr = res[(n, "on")]["plan"].frame.drift_report
        print(f"{n}: {res[(n, 'on')]['plan'].frame.drift_correction}; chunks {dr['chunks']}, wall associations {dr['associations']}, residual RMS {dr['residual_rms_cm_before']} -> {dr['residual_rms_cm_after']} cm, "
              f"held-out {dr['holdout_rms_cm_before']} -> {dr['holdout_rms_cm_after']} cm, max rotation {dr['max_rotation_deg']} deg, max shift {dr['max_shift_cm']} cm, "
              f"floor level {dr['floor_level_range_cm']} cm, jumps {dr['odometry_jumps']}")
    print("\n=========== CROSS-CAPTURE AGREEMENT (same property), OFF vs ON ===========")
    print(f"{'pair (B -> A)':30s} {'mode':4s} {'fit<5cm':>8s} {'median':>7s} {'yaw':>7s} {'matched':>8s} {'pairs':>6s} {'median|d|':>10s} {'p90|d|':>8s} {'within 1cm/0.5%':>16s} {'signed':>8s}")
    out = []
    for b, a in (("floor_only", "with_ceiling"), ("single_room", "with_ceiling"), ("single_room", "floor_only")):
        for m in ("off", "on"):
            cfg = CFG_ON
            c = cross(res[(a, m)], res[(b, m)], cfg)
            out.append(dict(pair=f"{b}->{a}", mode=m, **{k: v for k, v in c.items() if k != "deltas"}, deltas=c["deltas"]))
            print(f"{b + ' -> ' + a:30s} {m:4s} {100 * c['fit']:7.1f}% {c['med']:5.1f}cm {c['yaw']:+6.2f}d {c['matched']:3d}/{c['n_b']:<3d}  {c['pairs']:5d}  {c['med_abs']:8.1f}cm {c['p90']:6.1f}cm "
                  f"{100 * c['either']:14.0f}%  {c['signed']:+6.2f}%")
    (OUT / "m4_ablation.json").write_text(json.dumps(out, indent=1, default=float))
    print("\nwrote out/m4_ablation.json")


if __name__ == "__main__":
    main()
