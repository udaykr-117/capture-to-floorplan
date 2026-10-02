import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floorplan.config import load_config
from floorplan.depth import align_to_sparse, backproject
from floorplan.geometry.gravity import camera_up, estimate_up, rotation_to_y

CFG = load_config()


def test_depth_scale_is_recovered_from_noisy_sparse_points_with_outliers():
    rng = np.random.default_rng(0)
    h, w = 720, 960
    depth = 2.0 + 0.002 * np.tile(np.arange(w), (h, 1)) + 0.001 * np.tile(np.arange(h)[:, None], (1, w))  # model depth in metres
    xy = np.c_[rng.uniform(5, w - 5, 400), rng.uniform(5, h - 5, 400)]
    a_true = 3.7                                                                                          # model metres per SfM unit
    z_sfm = depth[np.round(xy[:, 1]).astype(int), np.round(xy[:, 0]).astype(int)] / a_true * rng.normal(1.0, 0.03, 400)
    z_sfm[:80] *= rng.uniform(0.3, 3.0, 80)                                                               # 20% wrong matches
    fs = align_to_sparse(depth, xy, z_sfm, CFG)
    assert fs is not None and fs.a == pytest.approx(a_true, rel=0.03) and fs.n >= 200


def test_too_few_sparse_points_give_no_scale():
    depth = np.full((720, 960), 3.0)
    assert align_to_sparse(depth, np.array([[100.0, 100.0]] * 10), np.ones(10), CFG) is None


def test_backprojection_inverts_the_camera_model():
    f, cx, cy, k = 800.0, 480.0, 360.0, 0.05
    R = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    t = np.array([0.3, -0.2, 1.0])
    depth = np.full((720, 960), 3.0)                                   # a fronto-parallel plane 3 units in front of the camera
    P = backproject(depth, (f, cx, cy, k), R, t, stride=8, edge_rel=0.08)
    pc = P @ R.T + t                                                    # back to camera coordinates
    assert pc[:, 2] == pytest.approx(3.0, abs=1e-9)
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x**2 + y**2
    u, v = f * x * (1 + k * r2) + cx, f * y * (1 + k * r2) + cy         # COLMAP SIMPLE_RADIAL projection
    uu, vv = np.meshgrid(np.arange(0, 960, 8), np.arange(0, 720, 8))
    assert np.sort(u) == pytest.approx(np.sort(uu.ravel().astype(float)), abs=0.01)  # 3 fixed-point steps invert the radial term to ~0.002 px
    assert np.sort(v) == pytest.approx(np.sort(vv.ravel().astype(float)), abs=0.01)


def test_depth_discontinuities_are_dropped():
    depth = np.full((720, 960), 2.0)
    depth[:, 480:] = 5.0                                                # a step edge
    P = backproject(depth, (800.0, 480.0, 360.0, 0.0), np.eye(3), np.zeros(3), stride=1, edge_rel=0.08)
    assert len(P) < depth.size and len(P) > depth.size * 0.99


def test_up_is_found_in_a_randomly_rotated_scene_and_rotated_to_y():
    rng = np.random.default_rng(1)
    Rr = Rotation.random(random_state=3).as_matrix()                     # the SfM frame is arbitrarily oriented
    up_true_world = np.array([0.0, 1.0, 0.0])
    # normals: floor and ceiling (vertical) plus walls (horizontal), noisy
    n_vert = np.tile([0.0, 1.0, 0.0], (4000, 1)) * rng.choice([-1, 1], 4000)[:, None] + rng.normal(0, 0.03, (4000, 3))
    ang = rng.uniform(0, 2 * np.pi, 3000)
    n_wall = np.c_[np.cos(ang), np.zeros(3000), np.sin(ang)] + rng.normal(0, 0.03, (3000, 3))
    normals = np.concatenate([n_vert, n_wall]) @ Rr.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # cameras held roughly upright, tilted up to 15 degrees
    ups = []
    for _ in range(50):
        tilt = Rotation.from_euler("xz", rng.uniform(-15, 15, 2), degrees=True).as_matrix()
        ups.append(Rr @ tilt @ up_true_world)
    up, info = estimate_up(np.array(ups), normals)
    assert np.degrees(np.arccos(np.clip(up @ (Rr @ up_true_world), -1, 1))) < 1.5 and info["refined"]
    G = rotation_to_y(up)
    assert G @ up == pytest.approx([0, 1, 0], abs=1e-9) and np.linalg.det(G) == pytest.approx(1.0)


def test_camera_up_of_an_upright_camera():
    R = np.eye(3)  # camera axes = world axes, y down
    assert camera_up(R) == pytest.approx([0, -1, 0])
