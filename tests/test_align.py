import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.geometry.align import Frame, align
from synth import box_room, rotate_yaw

CFG = load_config()


def _scene(yaw, floor_y=-1.4):
    pts = rotate_yaw(box_room(0, 0, 5.0, 4.0, 2.5, floor_y=floor_y), yaw) + np.array([3.0, 0, -2.0])
    cams = rotate_yaw(np.array([[2.0, floor_y + 1.4, 1.5], [3.0, floor_y + 1.4, 2.5]]), yaw) + np.array([3.0, 0, -2.0])
    return pts, cams


@pytest.mark.parametrize("yaw", [0.0, 33.0, 71.5, -20.0])
def test_alignment_recovers_yaw_floor_and_axis_aligned_walls(yaw):
    pts, cams = _scene(yaw)
    al, Pa, Na = align(pts, cams, CFG)
    d = (al.frame.yaw_deg - yaw + 45) % 90 - 45
    assert abs(d) < 0.5, (al.frame.yaw_deg, yaw)
    assert al.floor.height == pytest.approx(-1.4, abs=0.01)
    assert len(al.floor.candidates) == 1  # the ceiling is above the camera and must not be a floor candidate
    walls = Pa[(np.abs(Na[:, 1]) < 0.2) & (Pa[:, 1] > 0.3) & (Pa[:, 1] < 1.6)]
    spans = sorted([np.ptp(walls[:, 0]), np.ptp(walls[:, 2])])
    assert spans == pytest.approx([4.0, 5.0], abs=0.06)
    assert al.evidence["residual_deg"] < 0.5


def test_frame_round_trip():
    f = Frame(yaw_deg=37.0, floor_y=-1.3)
    p = np.random.default_rng(1).normal(size=(50, 3))
    assert f.to_world(f.to_aligned(p)) == pytest.approx(p)
    # a wall normal at world angle == yaw must end up along +x'
    n = np.array([[np.cos(np.radians(37)), 0.0, np.sin(np.radians(37))]])
    assert f.rotate(n)[0] == pytest.approx([1, 0, 0], abs=1e-9)


def test_floor_is_lowest_significant_peak_not_the_largest():
    pts, cams = _scene(0.0)
    table = rotate_yaw(box_room(1.0, 1.0, 3.0, 2.5, 0.0, floor_y=-0.75, step=0.01, ceiling=False, skip=("x0", "x1", "z0", "z1")), 0.0) + np.array([3.0, 0, -2.0])
    al, _, _ = align(np.concatenate([pts, table]), cams, CFG)
    assert al.floor.height == pytest.approx(-1.4, abs=0.01)
    assert len(al.floor.candidates) == 2
