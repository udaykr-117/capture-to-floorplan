from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import pytest

from floorplan.config import load_config
from floorplan.io.stray import backproject, cloud, open_capture, pose, scale_intrinsics

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "single_room": ROOT / "single_room" / "c00a170fe1",
    "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
    "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6",
}
needs_data = pytest.mark.skipif(not all(p.exists() for p in SAMPLES.values()), reason="sample captures not present")
CFG = load_config()
K_RGB = np.array([[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1.0]])


def test_scale_intrinsics_to_depth_resolution():
    K = scale_intrinsics(K_RGB, (1920, 1440), (192, 256))
    assert K[0, 0] == pytest.approx(1600 * 256 / 1920)
    assert K[1, 1] == pytest.approx(1600 * 192 / 1440)
    assert (K[0, 2], K[1, 2]) == pytest.approx((128.0, 96.0))


def test_backproject_principal_point_is_on_optical_axis():
    K = scale_intrinsics(K_RGB, (1920, 1440), (192, 256))
    depth = np.zeros((192, 256), np.float32)
    depth[96, 128] = 2.0
    p = backproject(depth, np.full((192, 256), 2, np.uint8), K, conf_min=2, dmin=0.1, dmax=5.0)
    assert p.shape == (1, 3)
    assert p[0] == pytest.approx([0, 0, 2.0])


def test_backproject_axes_are_x_right_y_down():
    K = scale_intrinsics(K_RGB, (1920, 1440), (192, 256))
    depth = np.zeros((192, 256), np.float32)
    depth[96, 138] = 1.0  # 10 px right of the principal point
    depth[106, 128] = 1.0  # 10 px below it
    p = backproject(depth, np.full((192, 256), 2, np.uint8), K, conf_min=2, dmin=0.1, dmax=5.0)
    right, down = p[p[:, 0] != 0][0], p[p[:, 1] != 0][0]
    assert right[0] > 0 and down[1] > 0


def test_backproject_filters_range_and_confidence():
    K = scale_intrinsics(K_RGB, (1920, 1440), (192, 256))
    depth = np.zeros((192, 256), np.float32)
    depth[10, 10], depth[20, 20], depth[30, 30], depth[40, 40] = 0.05, 6.0, 1.0, 1.0
    conf = np.full((192, 256), 2, np.uint8)
    conf[40, 40] = 1
    p = backproject(depth, conf, K, conf_min=2, dmin=0.1, dmax=5.0)
    assert len(p) == 1 and p[0, 2] == pytest.approx(1.0)


def test_pose_quaternion_is_xyzw():
    s = np.sqrt(0.5)
    ident = pose(pd.Series(dict(qx=0, qy=0, qz=0, qw=1, x=1, y=2, z=3)))
    assert ident[0] == pytest.approx(np.eye(3))
    assert ident[1] == pytest.approx([1, 2, 3])
    rot_y90 = pose(pd.Series(dict(qx=0, qy=s, qz=0, qw=s, x=0, y=0, z=0)))[0]  # +90 deg about y
    assert rot_y90 @ np.array([0, 0, 1.0]) == pytest.approx([1, 0, 0])
    assert rot_y90 @ np.array([1.0, 0, 0]) == pytest.approx([0, 0, -1])  # a wxyz misread passes the line above but not this


@needs_data
@pytest.mark.parametrize("name", SAMPLES)
def test_capture_pairing_and_resolution(name):
    cap = open_capture(SAMPLES[name], CFG)  # asserts counts, frame==row index, 1920x1440, video frames
    assert cap.depth_hw == (192, 256)


@needs_data
@pytest.mark.parametrize("name", SAMPLES)
def test_up_axis_is_y_from_trajectory(name):
    odo = pd.read_csv(SAMPLES[name] / "odometry.csv", skipinitialspace=True)
    ranges = np.ptp(odo[["x", "y", "z"]].values, axis=0)
    assert np.argmin(ranges) == 1, ranges


@needs_data
def test_floor_plane_normal_is_near_y():
    o3d.utility.random.seed(0)
    cap = open_capture(SAMPLES["single_room"], CFG)
    pts = cloud(cap, CFG, stride=20, voxel=0.04)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    model, inliers = pcd.segment_plane(0.02, 3, 2000)
    n = np.array(model[:3]) / np.linalg.norm(model[:3])
    angle = np.degrees(np.arccos(abs(n[1])))
    assert angle < CFG["checks"]["floor_normal_max_deg"], (angle, n, len(inliers))
