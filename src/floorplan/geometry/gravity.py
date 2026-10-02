"""Find 'up' in an SfM frame (arbitrary orientation) and rotate so up = +Y, as the rest of the pipeline expects."""
import numpy as np


def camera_up(R_cam_from_world: np.ndarray) -> np.ndarray:
    """Up direction of a camera in the SfM frame, assuming the phone is held upright (OpenCV axes: x right, y down, z forward)."""
    return -R_cam_from_world[1, :]


def estimate_up(cam_ups: np.ndarray, normals: np.ndarray, max_deg: float = 20.0) -> tuple[np.ndarray, dict]:
    """Mean camera up, refined to the dominant near-vertical surface normal (floor and ceiling dominate those)."""
    up0 = cam_ups.mean(0)
    up0 /= np.linalg.norm(up0)
    c = normals @ up0
    near = np.abs(c) > np.cos(np.radians(max_deg))
    if near.sum() < 100:
        return up0, dict(refined=False, n=int(near.sum()), shift_deg=0.0)
    n = normals[near] * np.sign(c[near])[:, None]
    up = n.mean(0)
    up /= np.linalg.norm(up)
    return up, dict(refined=True, n=int(near.sum()), shift_deg=float(np.degrees(np.arccos(np.clip(up @ up0, -1, 1)))))


def rotation_to_y(up: np.ndarray) -> np.ndarray:
    """Rotation matrix G with G @ up = +Y (Rodrigues); points transform as p' = p @ G.T."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, y)
    s, c = np.linalg.norm(v), float(up @ y)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
