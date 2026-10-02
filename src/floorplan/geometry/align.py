"""Align a world-frame cloud to the room: Y up, floor at y' = 0, walls along x' and z'.

World frame is Y up (CLAUDE.md). The yaw is arbitrary in the world frame, so it is estimated from wall normals.
Convention: x' =  c*x + s*z,  z' = -s*x + c*z with (c, s) = (cos yaw, sin yaw), so a wall normal at world angle
atan2(nz, nx) = yaw ends up along x'.
"""
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.signal import find_peaks


@dataclass
class Frame:
    yaw_deg: float
    floor_y: float

    def _cs(self) -> tuple[float, float]:
        return float(np.cos(np.radians(self.yaw_deg))), float(np.sin(np.radians(self.yaw_deg)))

    def rotate(self, v: np.ndarray) -> np.ndarray:
        """Rotate vectors (no translation) world -> aligned."""
        c, s = self._cs()
        out = np.array(v, dtype=float, copy=True)
        out[:, 0] = c * v[:, 0] + s * v[:, 2]
        out[:, 2] = -s * v[:, 0] + c * v[:, 2]
        return out

    def to_aligned(self, p: np.ndarray) -> np.ndarray:
        out = self.rotate(p)
        out[:, 1] = p[:, 1] - self.floor_y
        return out

    def to_world(self, q: np.ndarray) -> np.ndarray:
        c, s = self._cs()
        out = np.array(q, dtype=float, copy=True)
        out[:, 0] = c * q[:, 0] - s * q[:, 2]
        out[:, 2] = s * q[:, 0] + c * q[:, 2]
        out[:, 1] = q[:, 1] + self.floor_y
        return out


@dataclass
class FloorEstimate:
    height: float
    candidates: list[tuple[float, int]]  # (height, count) of every peak considered, lowest first
    spread_m: float                      # height of the highest-count candidate minus the chosen one (0 if same)
    iqr_m: float                         # interquartile range of points within the refine window


def estimate_floor(y: np.ndarray, cam_y: np.ndarray, cfg: dict) -> FloorEstimate:
    a = cfg["align"]
    ymax = cam_y.min() - a["floor_below_camera_m"]
    ys = y[y < ymax]
    bins = np.arange(ys.min() - 5 * a["floor_hist_bin_m"], ymax + a["floor_hist_bin_m"], a["floor_hist_bin_m"])  # margin so the lowest peak is not at the array edge
    h, e = np.histogram(ys, bins=bins)
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    peaks, _ = find_peaks(hs, height=a["floor_peak_min_rel"] * hs.max(), distance=3)
    cands = sorted(((float(e[p] + a["floor_hist_bin_m"] / 2), int(h[p])) for p in peaks), key=lambda c: c[0])
    chosen = cands[0][0]
    win = ys[np.abs(ys - chosen) < a["floor_refine_halfwidth_m"]]
    height = float(np.median(win))
    strongest = max(cands, key=lambda c: c[1])[0]
    return FloorEstimate(height, cands, abs(strongest - chosen), float(np.subtract(*np.percentile(win, [75, 25]))))


def wall_yaw(P: np.ndarray, N: np.ndarray, cfg: dict) -> tuple[float, dict]:
    """Dominant wall-normal direction (degrees, mod 90) from vertical-surface points. Y is up."""
    a = cfg["align"]
    vert = np.abs(N[:, 1]) < cfg["scene"]["vertical_normal_max"]
    ang = np.degrees(np.arctan2(N[vert, 2], N[vert, 0])) % 90
    hist, _ = np.histogram(ang, bins=90, range=(0, 90))
    sm = np.array([hist[np.arange(i - 3, i + 4) % 90].sum() for i in range(90)])
    peak = int(np.argmax(sm)) + 0.5
    d = (ang - peak + 45) % 90 - 45
    near = np.abs(d) < a["yaw_peak_window_deg"]
    z = np.exp(1j * np.radians(4 * ang[near])).mean()
    yaw = float((np.degrees(np.angle(z)) / 4) % 90)
    return yaw, dict(n_vertical=int(vert.sum()), n_used=int(near.sum()), peak_deg=peak, resultant=float(abs(z)))


@dataclass
class Aligned:
    frame: Frame
    floor: FloorEstimate
    evidence: dict = field(default_factory=dict)


def analysis_cloud(pts: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsampled cloud with unoriented normals (|n| = 1)."""
    s = cfg["scene"]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(s["normal_voxel_m"])
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=s["normal_radius_m"], max_nn=s["normal_max_nn"]))
    return np.asarray(pcd.points), np.asarray(pcd.normals)


def estimate_frame(pts: np.ndarray, cam_xyz: np.ndarray, cfg: dict) -> tuple[Frame, FloorEstimate, dict, np.ndarray, np.ndarray]:
    """Floor height and wall yaw of a whole capture. Also returns the world-frame analysis cloud and normals."""
    floor = estimate_floor(pts[:, 1], cam_xyz[:, 1], cfg)
    P, N = analysis_cloud(pts, cfg)
    lo, hi = cfg["checks"]["wall_slab_above_floor_m"]
    slab = (P[:, 1] > floor.height + lo) & (P[:, 1] < floor.height + hi)
    yaw, ev = wall_yaw(P[slab], N[slab], cfg)
    return Frame(yaw, floor.height), floor, ev, P, N


def check_residual(Pa: np.ndarray, Na: np.ndarray, ev: dict, cfg: dict) -> dict:
    """Refuse to continue if the wall direction in the aligned cloud is more than the allowed angle from an axis."""
    lo, hi = cfg["checks"]["wall_slab_above_floor_m"]
    slab_a = (Pa[:, 1] > lo) & (Pa[:, 1] < hi)
    yaw_res, _ = wall_yaw(Pa[slab_a], Na[slab_a], cfg)
    res = min(yaw_res, 90 - yaw_res)
    ev["residual_deg"] = float(res)
    if res > cfg["align"]["yaw_max_residual_deg"]:
        raise ValueError(f"alignment residual {res:.2f} deg exceeds {cfg['align']['yaw_max_residual_deg']} deg")
    return ev


def align(pts: np.ndarray, cam_xyz: np.ndarray, cfg: dict) -> tuple[Aligned, np.ndarray, np.ndarray]:
    """Return the alignment plus the analysis cloud and normals, both already in the aligned frame."""
    frame, floor, ev, P, N = estimate_frame(pts, cam_xyz, cfg)
    Pa, Na = frame.to_aligned(P), frame.rotate(N)
    return Aligned(frame, floor, check_residual(Pa, Na, ev, cfg)), Pa, Na
