import numpy as np
import pytest
from shapely.geometry import box

from floorplan.config import load_config
from floorplan.geometry.align import analysis_cloud
from floorplan.rooms.ceiling import measure_ceiling
from synth import box_room

CFG = load_config()


def _aligned(pts, floor_y=-1.4):
    P, N = analysis_cloud(pts, CFG)
    P = P.copy()
    P[:, 1] -= floor_y
    return P, N


def test_measured_ceiling_height_and_interval():
    P, N = _aligned(box_room(0, 0, 5, 4, 2.5))
    c = measure_ceiling(box(0, 0, 5, 4), P, N, CFG)
    assert c.status == "measured"
    assert c.height == pytest.approx(2.5, abs=0.015)
    assert c.interval[0] <= 2.5 <= c.interval[1]
    assert c.coverage > 0.8


def test_missing_ceiling_is_unmeasurable_not_a_number():
    P, N = _aligned(box_room(0, 0, 5, 4, 2.5, ceiling=False))
    c = measure_ceiling(box(0, 0, 5, 4), P, N, CFG)
    assert c.status == "unmeasurable" and c.height is None and c.interval is None and c.reason


def test_two_ceiling_levels_are_ambiguous_and_the_interval_spans_both():
    a = box_room(0, 0, 2.5, 4, 2.4, seed=1, skip=("x1",))
    b = box_room(2.5, 0, 5, 4, 2.9, seed=2, skip=("x0",))
    P, N = _aligned(np.concatenate([a, b]))
    c = measure_ceiling(box(0, 0, 5, 4), P, N, CFG)
    assert c.status == "ambiguous"
    assert c.interval[0] <= 2.4 + 0.02 and c.interval[1] >= 2.9 - 0.02


def test_local_floor_removes_a_floor_offset_of_6_cm():
    # room whose floor is 6 cm higher than the global floor (y' = 0): the ceiling height must still be 2.5 m
    P, N = _aligned(box_room(0, 0, 5, 4, 2.5, floor_y=-1.4 + 0.06))
    c = measure_ceiling(box(0, 0, 5, 4), P, N, CFG)
    assert c.floor_offset == pytest.approx(0.06, abs=0.01) and c.floor_local
    assert c.height == pytest.approx(2.5, abs=0.015)


def test_thin_ceiling_coverage_is_unmeasurable():
    room = box_room(0, 0, 5, 4, 2.5, ceiling=False)
    patch = box_room(0, 0, 1.0, 0.8, 2.5, seed=3, skip=("x0", "x1", "z0", "z1"))  # ceiling over 4% of the room
    P, N = _aligned(np.concatenate([room, patch]))
    c = measure_ceiling(box(0, 0, 5, 4), P, N, CFG)
    assert c.status == "unmeasurable" and "below" in c.reason
