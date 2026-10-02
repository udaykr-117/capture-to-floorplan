import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes
from floorplan.rooms.freespace import carve_fans, free_space, make_grid, segment_rooms
from synth import box_room, cast_hits, rotate_yaw

CFG = load_config()
SHIFT = np.array([3.0, 0, -2.0])
YAW = 25.0
FLOOR_Y = -1.4
PATH = [(2, 2), (3, 2), (4, 1.95), (5, 2), (6, 2), (2, 1), (6, 3), (1.5, 3), (7, 1)]


def _segments(door):
    a0, a1 = door
    return [((0, 0), (0, 4)), ((8, 0), (8, 4)), ((0, 0), (8, 0)), ((0, 4), (8, 4)), ((4, 0), (4, a0)), ((4, a1), (4, 4))]


def _scene(door, hole=None):
    a0, a1 = door
    a = box_room(0, 0, 4.0, 4.0, 2.5, seed=1, openings=[("x1", a0, a1, 0.0, 2.0)])
    b = box_room(4.0, 0, 8.0, 4.0, 2.5, seed=2, openings=[("x0", a0, a1, 0.0, 2.0)])
    pts = np.concatenate([a, b])
    if hole is not None:  # drop floor points (e.g. under furniture) in [x0, x1] x [z0, z1]
        x0, z0, x1, z1 = hole
        floor = np.isclose(pts[:, 1], FLOOR_Y, atol=0.03)
        pts = pts[~(floor & (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 2] > z0) & (pts[:, 2] < z1))]
    cams = rotate_yaw(np.array([[x, 0.0, z] for x, z in PATH]), YAW) + SHIFT
    hits = []
    for x, z in PATH:
        h = cast_hits((x, z), _segments(door))
        hits.append(np.concatenate([np.c_[h[:, 0], np.full(len(h), FLOOR_Y + hh), h[:, 1]] for hh in (0.5, 1.0, 1.5)]))
    return rotate_yaw(pts, YAW) + SHIFT, cams, [rotate_yaw(h, YAW) + SHIFT for h in hits]


def _rooms(pts, cams, hits):
    al, Pa, Na = align(pts, cams, CFG)
    planes = find_wall_planes(Pa, Na, CFG)
    grid = make_grid(Pa, CFG["rooms"]["cell_m"])
    ca = al.frame.to_aligned(cams)
    fans = carve_fans(((al.frame.to_aligned(h), ca[i]) for i, h in enumerate(hits)), grid, CFG)
    layers = free_space(Pa, Na, ca, planes, fans, grid, CFG)
    return segment_rooms(layers["free"], grid, CFG), grid, layers, al, ca, fans


def test_door_gap_below_the_bridge_limit_is_closed_in_the_map_so_rooms_split():
    # 0.9 m door < walls.gap_merge_m (1.2): the wall run bridges the gap and the barrier separates the rooms.
    labels, grid, layers, al, ca, fans = _rooms(*_scene((1.55, 2.45)))
    ids = [i for i in np.unique(labels) if i > 0]
    assert len(ids) == 2
    areas = sorted((labels == i).sum() * grid.cell**2 for i in ids)
    assert areas == pytest.approx([16.0, 16.0], rel=0.08)  # barrier is drawn 10 cm wide, so free area is a little under 16


def test_cameras_in_different_rooms_get_different_labels():
    labels, grid, layers, al, ca, fans = _rooms(*_scene((1.55, 2.45)))
    r, c = grid.rc(ca[[0, 4], 0], ca[[0, 4], 2])
    la, lb = labels[r[0], c[0]], labels[r[1], c[1]]
    assert la > 0 and lb > 0 and la != lb


def test_opening_wider_than_the_bridge_limit_stays_one_region_known_limitation():
    labels, grid, layers, al, ca, fans = _rooms(*_scene((1.2, 2.8)))  # 1.6 m opening
    assert len([i for i in np.unique(labels) if i > 0]) == 1


def test_visibility_fans_fill_floor_that_was_never_observed():
    # a 2 m x 2 m patch of floor in room A was never seen (e.g. under a bed); the room must still come out whole
    with_hole = _rooms(*_scene((1.55, 2.45), hole=(0.8, 0.8, 2.8, 2.8)))
    labels, grid = with_hole[0], with_hole[1]
    ids = [i for i in np.unique(labels) if i > 0]
    assert len(ids) == 2
    assert sorted((labels == i).sum() * grid.cell**2 for i in ids) == pytest.approx([16.0, 16.0], rel=0.08)
    fans = with_hole[5]
    assert (fans >= CFG["rooms"]["fan_min_frames"]).sum() * grid.cell**2 > 28.0  # carved space alone already covers most of both rooms
