"""M7 diagnosis of the worst gate (repeatability). Question: why do facing-wall widths disagree between captures by ~5 cm (gate 1 cm / 0.5%)?

Hypotheses, each with its own test:
  H1 drift blur: a wall's position moves over the time of a capture, so one plane fitted to all its points is an average of several positions.
     Test: per wall plane, the offset seen by each time chunk separately; spread of those offsets across chunks.
  H2 noise of the plane fit: points are scattered (plane sigma); test: does |delta| grow with the fit sigma?
  H3 drift between the two walls: test: width measured only in chunks that see BOTH walls (co-visible, a few seconds apart) vs from the whole capture;
     if H3 holds, co-visible widths agree better across captures.
Usage: uv run python scripts/m7_diagnose.py   -> out/m7_diagnose.txt (printed)
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from floorplan.config import load_config
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan
from floorplan.register import match_segments, register, segments, to_other, wall_points, width_pairs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
SAMPLES = {"with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "single_room": ROOT / "single_room" / "c00a170fe1"}
CFG = load_config()
MIN_PTS = 40       # a chunk "sees" a wall run if >= 40 of its 2 cm voxels lie on it (~0.016 m2)


def load(name):
    key = hashlib.md5(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
    f = OUT / "cache" / f"m7_{name}_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    plan, inter = build_plan(StraySource(SAMPLES[name], CFG), CFG)
    fr, corr = inter["al"].frame, inter["corr"]
    chunk_pts = [corr.apply(c.idx, fr.to_aligned(c.pts)).astype(np.float32) for c in inter["chunks"]]
    d = dict(planes=inter["planes"], W=wall_points(inter["Pa"], inter["Na"], CFG), chunk_pts=chunk_pts)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(d))
    return d


WIN = 0.15        # search window around the global plane: the first version used +-3 cm (plane_tol_m), which clips any displacement > 3 cm
                  # and biases every chunk offset toward the global plane (a first version with +-3 cm hid the effect)


def per_chunk_offsets(seg, chunk_pts):
    """Offset of one wall run as seen by each chunk alone: densest 1 cm position within +-15 cm of the global plane (NaN if not seen)."""
    w = CFG["walls"]
    lo, hi = w["slab_above_floor_m"]
    along = 2 if seg.axis == 0 else 0
    out = np.full(len(chunk_pts), np.nan)
    e = np.arange(seg.offset - WIN, seg.offset + WIN + 1e-9, 0.01)
    for k, P in enumerate(chunk_pts):
        m = (np.abs(P[:, seg.axis] - seg.offset) < WIN) & (P[:, 1] > lo) & (P[:, 1] < hi) & (P[:, along] > seg.s0 + 0.2) & (P[:, along] < seg.s1 - 0.2)
        if m.sum() >= MIN_PTS:
            c = P[m, seg.axis]
            h = np.convolve(np.histogram(c, e)[0], np.ones(3), mode="same")
            pk = e[int(np.argmax(h))] + 0.005
            near = np.abs(c - pk) < 0.015
            if near.sum() >= MIN_PTS // 2:
                out[k] = float(np.median(c[near]))
    return out


def covisible_width(s1, s2, pts):
    o1, o2 = per_chunk_offsets(s1, pts), per_chunk_offsets(s2, pts)
    both = ~np.isnan(o1) & ~np.isnan(o2)
    return (float(np.median(o2[both] - o1[both])), int(both.sum())) if both.any() else (None, 0)


def main():
    caps = {n: load(n) for n in SAMPLES}
    print("H1: spread over time of each wall's position (offset per chunk, walls seen by >= 3 chunks)")
    for n, c in caps.items():
        sp, rng = [], []
        for s in segments(c["planes"]):
            o = per_chunk_offsets(s, c["chunk_pts"])
            o = o[~np.isnan(o)]
            if len(o) >= 3:
                sp.append(1.4826 * np.median(np.abs(o - np.median(o))) * 100)
                rng.append((o.max() - o.min()) * 100)
        print(f"  {n:13s} {len(sp)} wall runs: robust spread of per-chunk offsets median {np.median(sp):.1f} cm (90th pct {np.percentile(sp, 90):.1f}); "
              f"range median {np.median(rng):.1f} cm (90th pct {np.percentile(rng, 90):.1f})")

    print("\nH2/H3 per capture pair: global widths (current) vs co-visible widths (both walls in the same chunks)")
    allg, allc, allsig = [], [], []
    for a, b in (("with_ceiling", "floor_only"), ("with_ceiling", "single_room"), ("floor_only", "single_room")):
        A, B = caps[a], caps[b]
        reg = register(A["W"], B["W"], CFG)
        SA0, SB0 = segments(A["planes"]), segments(B["planes"])
        SB = [to_other(s, reg) for s in SB0]
        m = match_segments(SA0, SB, CFG)
        g, cv, sig = [], [], []
        for p in width_pairs(SA0, SB, m, CFG):
            # recover the segments of this pair (offsets identify them uniquely within an axis)
            ia = [i for i, s in enumerate(SA0) if s.axis == p["axis"] and s.offset in (p["a_lo"], p["a_hi"])]
            pair = [(i, j) for i, j in m if i in ia]
            if len(pair) != 2:
                continue
            (i1, j1), (i2, j2) = sorted(pair, key=lambda ij: SA0[ij[0]].offset)
            wa, na = covisible_width(SA0[i1], SA0[i2], A["chunk_pts"])
            # B in its own frame: B's segments j1/j2 have their own axis there; the width is |offset difference|
            wb, nb = covisible_width(*sorted([SB0[j1], SB0[j2]], key=lambda s: s.offset), B["chunk_pts"])
            g.append((p["width_a"], p["delta"]))
            sig.append(np.hypot(p["sig_a"], p["sig_b"]))
            cv.append((p["width_a"], None if wa is None or wb is None else abs(wb) - abs(wa), na, nb))
        gd = np.array([d for _, d in g])
        ok = [c for c in cv if c[1] is not None]
        cd = np.array([c[1] for c in ok])
        gw = np.array([w for w, _ in g])
        gate = lambda d, w: ((np.abs(d) <= 0.01) | (np.abs(d) <= 0.005 * w)).mean() * 100
        same = np.array([c[1] is not None for c in cv])
        print(f"  {b} vs {a}: {len(g)} width pairs; co-visible in both captures: {len(ok)}")
        if len(g):
            print(f"    global (all pairs):            median |d| {np.median(np.abs(gd)) * 100:.1f} cm, gate pass {gate(gd, gw):.0f}%")
            if len(ok):
                print(f"    global (co-visible subset):    median |d| {np.median(np.abs(gd[same])) * 100:.1f} cm, gate pass {gate(gd[same], gw[same]):.0f}%")
                print(f"    co-visible widths (same subset): median |d| {np.median(np.abs(cd)) * 100:.1f} cm, gate pass {gate(cd, gw[same]):.0f}%; "
                      f"chunks used median A {np.median([c[2] for c in ok]):.0f}, B {np.median([c[3] for c in ok]):.0f}")
            allg += list(np.abs(gd)); allsig += sig
    r = np.corrcoef(allsig, allg)[0, 1] if len(allg) > 2 else float("nan")
    print(f"\nH2: correlation of |delta| with the combined plane-fit sigma over all pairs: r = {r:.2f} (n = {len(allg)})")


if __name__ == "__main__" and __import__("sys").argv[-1] not in ("h45", "h6"):
    main()


# ---- H4 / H5 (added after H1-H3 were rejected) ------------------------------------------------------------------------------------
# H4: a WallPlane collects every point within +-3 cm of one line through the whole house, so near-collinear walls of different rooms share one
#     offset; each run (segment) inherits it. Test: offset of each run from its own points only.
# H5: within +-3 cm of a wall there are other surfaces (skirting, frames, furniture fronts); the median mixes them. Test: the densest 1 cm
#     position (mode) of the run's own points instead of the median.
def run_offsets(seg, P_all, how):
    w = CFG["walls"]
    lo, hi = w["slab_above_floor_m"]
    along = 2 if seg.axis == 0 else 0
    m = (np.abs(P_all[:, seg.axis] - seg.offset) < w["plane_tol_m"]) & (P_all[:, 1] > lo) & (P_all[:, 1] < hi) & (P_all[:, along] > seg.s0) & (P_all[:, along] < seg.s1)
    c = P_all[m, seg.axis].astype(float)
    if len(c) < MIN_PTS:
        return seg.offset
    if how == "median":
        return float(np.median(c))
    e = np.arange(seg.offset - w["plane_tol_m"], seg.offset + w["plane_tol_m"] + 1e-9, 0.005)
    h, _ = np.histogram(c, e)
    h = np.convolve(h, np.ones(3) / 3, mode="same")              # 1.5 cm smoothing
    k = int(np.argmax(h))
    pk = (e[k] + e[k + 1]) / 2
    near = np.abs(c - pk) < 0.01
    return float(np.median(c[near]))


def h4_h5():
    from dataclasses import replace

    caps = {n: load(n) for n in SAMPLES}
    allp = {n: np.concatenate(c["chunk_pts"]) for n, c in caps.items()}
    print("\nH4/H5: run-local offsets. How far does each run's own offset sit from its plane's offset?")
    for n, c in caps.items():
        S = segments(c["planes"])
        d4 = np.array([run_offsets(s, allp[n], "median") - s.offset for s in S]) * 100
        d5 = np.array([run_offsets(s, allp[n], "mode") - s.offset for s in S]) * 100
        print(f"  {n:13s} {len(S)} runs: |run median - plane| median {np.median(np.abs(d4)):.1f} cm, 90th {np.percentile(np.abs(d4), 90):.1f}; "
              f"|run mode - plane| median {np.median(np.abs(d5)):.1f} cm, 90th {np.percentile(np.abs(d5), 90):.1f}")
    print("\nrepeatability gate with each offset estimator (same registration and matching as now):")
    for how in ("plane", "median", "mode"):
        tot, ok, absd = 0, 0, []
        for a, b in (("with_ceiling", "floor_only"), ("with_ceiling", "single_room"), ("floor_only", "single_room")):
            A, B = caps[a], caps[b]
            reg = register(A["W"], B["W"], CFG)
            SA, SB0 = segments(A["planes"]), segments(B["planes"])
            if how != "plane":
                SA = [replace(s, offset=run_offsets(s, allp[a], how)) for s in SA]
                SB0 = [replace(s, offset=run_offsets(s, allp[b], how)) for s in SB0]
            SB = [to_other(s, reg) for s in SB0]
            m = match_segments(SA, SB, CFG)
            ps = width_pairs(SA, SB, m, CFG)
            d = np.array([p["delta"] for p in ps]); w = np.array([p["width_a"] for p in ps])
            g = (np.abs(d) <= 0.01) | (np.abs(d) <= 0.005 * w)
            print(f"  {how:6s} {b:>11s} vs {a:<12s}: {len(ps):2d} pairs, median |d| {np.median(np.abs(d)) * 100:4.1f} cm, pass {g.sum()}/{len(ps)} ({100 * g.mean():.0f}%)")
            tot += len(ps); ok += int(g.sum()); absd += list(np.abs(d))
        print(f"  {how:6s} POOLED: {ok}/{tot} ({100 * ok / tot:.0f}%), median |d| {np.median(absd) * 100:.1f} cm")


if __name__ == "__main__" and __import__("sys").argv[-1] == "h45":
    h4_h5()


# ---- H6: wrong correspondences --------------------------------------------------------------------------------------------------------
# A 1.3 m width differing by 10 cm cannot be drift or fit noise; one of the two "matched" walls may be a different surface (frame, cabinet,
# other face of a partition) within the 15 cm match tolerance. Test: split width pairs by whether all four segments are unambiguous
# (no second same-axis, overlapping segment within the tolerance in the same capture) and by the offset residual of the matches.
def h6():
    caps = {n: load(n) for n in SAMPLES}
    tol = CFG["register"]["plane_match_tol_m"]
    rows = []
    for a, b in (("with_ceiling", "floor_only"), ("with_ceiling", "single_room"), ("floor_only", "single_room")):
        A, B = caps[a], caps[b]
        reg = register(A["W"], B["W"], CFG)
        SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
        m = match_segments(SA, SB, CFG)
        res = {i: abs(SB[j].offset - SA[i].offset) for i, j in m}

        def ambiguous(S, k):
            s = S[k]
            return any(o is not s and o.axis == s.axis and abs(o.offset - s.offset) <= tol and min(o.s1, s.s1) - max(o.s0, s.s0) > 0 for o in S)
        for p in width_pairs(SA, SB, m, CFG):
            ia = [i for i, s in enumerate(SA) if s.axis == p["axis"] and s.offset in (p["a_lo"], p["a_hi"])]
            pr = [(i, j) for i, j in m if i in ia]
            amb = any(ambiguous(SA, i) or ambiguous(SB, j) for i, j in pr)
            rows.append(dict(pair=f"{b}-{a}", d=p["delta"], w=p["width_a"], amb=amb, res=max(res[i] for i, _ in pr)))
    def show(lbl, sel):
        if not sel:
            print(f"  {lbl}: none"); return
        d = np.array([r["d"] for r in sel]); w = np.array([r["w"] for r in sel])
        g = (np.abs(d) <= 0.01) | (np.abs(d) <= 0.005 * w)
        print(f"  {lbl:42s} n {len(sel):2d}  median |d| {np.median(np.abs(d)) * 100:4.1f} cm  pass {100 * g.mean():3.0f}%")
    print("\nH6: width pairs split by match ambiguity and by match residual (pooled over the 3 capture pairs)")
    show("all", rows)
    show("all four walls unambiguous", [r for r in rows if not r["amb"]])
    show("some wall ambiguous", [r for r in rows if r["amb"]])
    for lim in (0.03, 0.05, 0.08):
        show(f"both match residuals <= {100 * lim:.0f} cm", [r for r in rows if r["res"] <= lim])
        show(f"a match residual > {100 * lim:.0f} cm", [r for r in rows if r["res"] > lim])


if __name__ == "__main__" and __import__("sys").argv[-1] == "h6":
    h6()
