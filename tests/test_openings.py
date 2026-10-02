import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes
from floorplan.rooms.openings import add_through, detect_openings, plane_spans, wall_images
from synth import box_room, cast_rays_3d, rotate_yaw

CFG = load_config()
SHIFT = np.array([3.0, 0, -2.0])
YAW = 25.0
FLOOR_Y = -1.4
H = 2.5
SEGS = [((0, 0), (0, 4)), ((8, 0), (8, 4)), ((0, 0), (8, 0)), ((0, 4), (8, 4)), ((4, 0), (4, 4))]
PATH = [(2, 2), (3, 1), (3, 3), (1.5, 3.4), (2.5, 3.4), (6, 2), (5, 1), (5, 3), (6.5, 3.5), (3.6, 2.0), (4.4, 2.0), (3.5, 3.4), (4.5, 3.4)]


def _run(openings_def, cloud_gaps=(), path=PATH):
    """openings_def: (a0, a1, h0, h1) on the shared wall x = 4 (along = z) that are real openings (points AND rays pass).
    cloud_gaps: same format, wall points removed from the cloud only (a stretch nobody saw): rays still hit the wall."""
    gaps = list(openings_def) + list(cloud_gaps)
    pts = np.concatenate([box_room(0, 0, 4, 4, H, seed=1, openings=[("x1", *g) for g in gaps]),
                          box_room(4, 0, 8, 4, H, seed=2, openings=[("x0", *g) for g in gaps])])
    pts = rotate_yaw(pts, YAW) + SHIFT
    cams = rotate_yaw(np.array([[x, 0.0, z] for x, z in path]), YAW) + SHIFT
    ray_ops = [(4, *g) for g in openings_def]
    hits = [rotate_yaw(cast_rays_3d((x, 0.0, z), SEGS, H, FLOOR_Y, ray_ops), YAW) + SHIFT for x, z in path]
    al, Pa, Na = align(pts, cams, CFG)
    planes = find_wall_planes(Pa, Na, CFG)
    ca = al.frame.to_aligned(cams)
    images = wall_images(plane_spans(planes, CFG), Pa, CFG)
    add_through(images, ((al.frame.to_aligned(h), ca[i]) for i, h in enumerate(hits)), CFG)
    return detect_openings(images, CFG)


def test_door_and_raised_opening_are_found_with_the_right_width_and_height():
    ops = _run([(1.55, 2.45, 0.0, 2.0), (3.0, 3.8, 0.9, 1.9)])
    assert len(ops) == 2, [(o.kind, round(o.width, 3)) for o in ops]
    door = next(o for o in ops if o.kind == "door")
    raised = next(o for o in ops if o.kind == "raised")
    assert door.width == pytest.approx(0.9, abs=0.04) and door.interval[0] - 0.02 <= 0.9 <= door.interval[1] + 0.02
    assert door.top == pytest.approx(2.0, abs=0.06)
    assert raised.width == pytest.approx(0.8, abs=0.08) and raised.interval[0] - 0.02 <= 0.8 <= raised.interval[1] + 0.02
    assert raised.bottom == pytest.approx(0.9, abs=0.06) and raised.top == pytest.approx(1.9, abs=0.06)


def test_a_solid_wall_has_no_phantom_openings():
    assert _run([]) == []


def test_unobserved_stretch_of_wall_is_not_an_opening():
    # 1 m of wall has no points (never seen) but rays hit the wall there, so there is no see-through evidence
    assert _run([], cloud_gaps=[(1.0, 2.0, 0.0, H)]) == []


def test_a_real_door_is_still_found_when_a_neighbouring_stretch_is_unobserved():
    ops = _run([(1.55, 2.45, 0.0, 2.0)], cloud_gaps=[(3.0, 3.8, 0.0, H)])
    assert len(ops) == 1 and ops[0].width == pytest.approx(0.9, abs=0.05)
