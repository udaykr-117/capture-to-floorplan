"""Relocalization jumps in the odometry, spread back along the path.

A handheld phone cannot move at several m/s, so a step that fast is the tracker snapping back to a corrected position (loop closure /
relocalization): the poses before it carried the accumulated drift. The jump is removed by adding it back to the earlier poses in
proportion to the path length travelled since the previous reset (or the start). Rotation is not touched (checked on the sample: the
jump carried no heading change).
"""
import numpy as np
import pandas as pd


def find_jumps(p: np.ndarray, t: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """(first frame, last frame) of every run of steps faster than `drift.jump_speed_mps`."""
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    bad = step / np.maximum(np.diff(t), 1e-3) > cfg["drift"]["jump_speed_mps"]
    out, i = [], 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j + 1 < len(bad) and bad[j + 1]:
                j += 1
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def distribute_jumps(odo: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Return a copy of the odometry with every jump distributed backwards over the preceding path, and a report."""
    p = odo[["x", "y", "z"]].values.astype(float).copy()
    t = odo["timestamp"].values
    events = find_jumps(p, t, cfg)
    if not events:
        return odo.copy(), []
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    for a, b in events:
        step[a:b] = 0.0                       # the jump itself is not travelled distance
    s = np.concatenate([[0.0], np.cumsum(step)])
    report, prev = [], 0
    for a, b in events:
        k = max(a - 10, prev)
        v = (p[a] - p[k]) / max(t[a] - t[k], 1e-3) if a > k else np.zeros(3)
        J = p[b] - p[a] - v * (t[b] - t[a])   # the jump minus the motion that really happened meanwhile
        span = s[a] - s[prev]
        lam = (s[prev:a + 1] - s[prev]) / span if span > 1e-6 else np.ones(a + 1 - prev)
        p[prev:a + 1] += J * lam[:, None]
        report.append(dict(frames=(int(a), int(b)), jump_m=float(np.linalg.norm(J)), vector_m=[float(x) for x in J], spread_over_frames=int(a + 1 - prev),
                           path_m=float(span)))
        prev = b
    out = odo.copy()
    out[["x", "y", "z"]] = p
    return out, report
