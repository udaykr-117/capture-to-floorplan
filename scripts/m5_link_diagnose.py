"""M5: where do consecutive video frames lose their links, and what is different there?

For each pair of consecutive sampled frames: verified SIFT matches (ratio test + RANSAC fundamental matrix), versus the real rotation between
the two frames (from ARKit), the image sharpness and the feature count. Usage: uv run python scripts/m5_link_diagnose.py [capture]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}


def main(name):
    frames = sorted((ROOT / "out" / "m5" / name / "frames").glob("f*.jpg"))
    idx = np.array([int(p.stem[1:]) for p in frames])
    odo = pd.read_csv(SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    q = R.from_quat(odo[["qx", "qy", "qz", "qw"]].values)
    pos = odo[["x", "y", "z"]].values
    sift = cv2.SIFT_create(nfeatures=8000)
    bf = cv2.BFMatcher()
    feats, sharp, nkp = [], [], []
    for p in frames:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        kp, des = sift.detectAndCompute(g, None)
        feats.append((kp, des))
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        nkp.append(len(kp))
    rows = []
    for i in range(len(frames) - 1):
        (k1, d1), (k2, d2) = feats[i], feats[i + 1]
        n_in = 0
        if d1 is not None and d2 is not None and len(k1) > 8 and len(k2) > 8:
            m = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
            if len(m) >= 8:
                p1 = np.float32([k1[x.queryIdx].pt for x in m])
                p2 = np.float32([k2[x.trainIdx].pt for x in m])
                _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 1.5, 0.99)
                n_in = int(mask.sum()) if mask is not None else 0
        a, b = idx[i], idx[i + 1]
        rot = np.degrees((q[a].inv() * q[b]).magnitude())
        dt = odo["timestamp"][b] - odo["timestamp"][a]
        rows.append(dict(i=i, frame=a, inliers=n_in, rot_deg=rot, rot_rate=rot / dt, move_cm=np.linalg.norm(pos[b] - pos[a]) * 100, sharp=min(sharp[i], sharp[i + 1]),
                         kp=min(nkp[i], nkp[i + 1])))
    df = pd.DataFrame(rows)
    weak = df.inliers < 30
    print(f"{name}: {len(frames)} frames, {len(df)} consecutive pairs; pairs with < 30 verified matches: {weak.sum()} ({100 * weak.mean():.0f}%); < 100: {(df.inliers < 100).sum()}")
    print(f"{'':28s} {'weak pairs (<30)':>18s} {'other pairs':>14s}")
    for col, lab in (("rot_deg", "rotation between frames (deg)"), ("rot_rate", "rotation rate (deg/s)"), ("move_cm", "camera movement (cm)"),
                     ("sharp", "sharpness (Laplacian var)"), ("kp", "SIFT keypoints (min of pair)")):
        print(f"{lab:28s} {df[col][weak].median():18.1f} {df[col][~weak].median():14.1f}   (medians)")
    print("\nlongest runs of consecutive strong pairs (>= 30 matches) = stretches SfM can chain:")
    runs, cur = [], 0
    for w in weak:
        if not w:
            cur += 1
        else:
            runs.append(cur)
            cur = 0
    runs.append(cur)
    print("  run lengths in frames:", sorted(runs, reverse=True)[:10], " (a model can only span one run)")
    print("\nthe 12 weakest pairs:")
    print(df.sort_values("inliers").head(12)[["frame", "inliers", "rot_deg", "rot_rate", "move_cm", "sharp", "kp"]].round(1).to_string(index=False))
    print("\ncorrelation of verified matches with: rotation", round(df.inliers.corr(df.rot_deg), 2), ", sharpness", round(df.inliers.corr(df.sharp), 2), ", keypoints", round(df.inliers.corr(df.kp), 2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "single_room")
