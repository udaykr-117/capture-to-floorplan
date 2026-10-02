import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes, structural
from synth import box_room, rotate_yaw

CFG = load_config()
SHIFT = np.array([3.0, 0, -2.0])


def _world(pts, yaw):
    return rotate_yaw(pts, yaw) + SHIFT


def _cams(points_xz, yaw, floor_y=-1.4):
    c = np.array([[x, floor_y + 1.4, z] for x, z in points_xz])
    return rotate_yaw(c, yaw) + SHIFT


def _planes(pts, cams):
    al, Pa, Na = align(pts, cams, CFG)
    return find_wall_planes(Pa, Na, CFG)


def test_single_room_four_walls_at_known_offsets_and_extents():
    planes = _planes(_world(box_room(0, 0, 5.0, 4.0, 2.5), 33.0), _cams([(2, 1.5), (3, 2.5)], 33.0))
    assert len(planes) == 4
    spans = sorted(max(p.offset for p in planes if p.axis == ax) - min(p.offset for p in planes if p.axis == ax) for ax in (0, 2))
    assert spans == pytest.approx([4.0, 5.0], abs=0.02)
    lengths = sorted(r.length for p in planes for r in p.runs)
    assert lengths == pytest.approx([4.0, 4.0, 5.0, 5.0], abs=0.12)  # 5 cm cells, +-1 cell each end
    assert all(r.coverage > 0.9 for p in planes for r in p.runs)


def test_thin_partition_keeps_both_faces_apart():
    a = box_room(0, 0, 5.0, 4.0, 2.5, seed=1)       # its x1 wall is the face at 5.00
    b = box_room(5.15, 0, 9.0, 4.0, 2.5, seed=2)    # its x0 wall is the face at 5.15
    planes = _planes(_world(np.concatenate([a, b]), 20.0), _cams([(2, 1.5), (7, 2)], 20.0))
    long_axis = max((0, 2), key=lambda ax: max(p.offset for p in planes if p.axis == ax) - min(p.offset for p in planes if p.axis == ax))
    offs = sorted(p.offset for p in planes if p.axis == long_axis)
    assert len(offs) == 4, offs
    assert np.diff(offs) == pytest.approx([5.0, 0.15, 3.85], abs=0.02)


def test_low_furniture_is_detected_but_not_structural():
    room = box_room(0, 0, 5.0, 4.0, 2.5)
    low = box_room(1.0, 1.0, 3.0, 1.05, 1.0, ceiling=False, skip=("x0", "x1"), seed=5)  # 2 m long, 1 m tall, 5 cm thick
    planes = _planes(_world(np.concatenate([room, low]), 0.0), _cams([(2, 2.5), (3, 3)], 0.0))
    kept = structural(planes, CFG)
    assert len(planes) == 5 and len(kept) == 4
    furniture = [r for p in planes for r in p.runs if r.coverage < CFG["walls"]["min_height_coverage"]]
    assert len(furniture) == 1 and furniture[0].length == pytest.approx(2.0, abs=0.12)


def test_the_outermost_walls_are_found_even_with_noise_free_points():
    # all points of a wall in one histogram bin, and the wall is the first / last bin of the range
    planes = _planes(_world(box_room(0, 0, 5.0, 4.0, 2.5, noise=0.0), 33.0), _cams([(2, 1.5), (3, 2.5)], 33.0))
    assert len(planes) == 4
