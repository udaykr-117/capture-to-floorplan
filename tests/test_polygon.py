import numpy as np
import pytest
from shapely.geometry import Point

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes
from floorplan.rooms.freespace import carve_fans, free_space, make_grid, segment_rooms
from floorplan.rooms.polygon import clean_labels, room_polygons
from synth import cast_hits, polygon_room, rotate_yaw
from test_rooms import FLOOR_Y, SHIFT, YAW, _scene

CFG = load_config()


def _polys(pts, cams, hits):
    al, Pa, Na = align(pts, cams, CFG)
    planes = find_wall_planes(Pa, Na, CFG)
    grid = make_grid(Pa, CFG["rooms"]["cell_m"])
    ca = al.frame.to_aligned(cams)
    fans = carve_fans(((al.frame.to_aligned(h), ca[i]) for i, h in enumerate(hits)), grid, CFG)
    layers = free_space(Pa, Na, ca, planes, fans, grid, CFG)
    labels = clean_labels(segment_rooms(layers["free"], grid, CFG), grid, CFG)
    return room_polygons(labels, planes, grid, CFG), al, ca


def test_two_rooms_have_exact_areas_and_do_not_overlap():
    polys, al, ca = _polys(*_scene((1.55, 2.45)))
    assert len(polys) == 2
    assert sorted(p.area for p in polys) == pytest.approx([16.0, 16.0], abs=0.15)  # plane-snapped, so close to the true 4 x 4
    assert polys[0].polygon.intersection(polys[1].polygon).area < 0.01
    for p in polys:
        assert len(p.edges) == 4
        assert sorted(round(e.length, 1) for e in p.edges) == [4.0, 4.0, 4.0, 4.0]
        assert all(e.source == "plane" for e in p.edges)


def test_l_shaped_room_has_six_edges_and_the_right_area():
    verts = [(0, 0), (8, 0), (8, 2), (4, 2), (4, 4), (0, 4)]  # area 8*2 + 4*2 = 24
    pts = polygon_room(verts, 2.5)
    path = [(1, 1), (3, 3), (6, 1), (2, 1), (5, 1.0), (1, 3.5), (7, 1)]
    segs = [(verts[i], verts[(i + 1) % 6]) for i in range(6)]
    hits = []
    for x, z in path:
        h = cast_hits((x, z), segs)
        hits.append(np.concatenate([np.c_[h[:, 0], np.full(len(h), FLOOR_Y + hh), h[:, 1]] for hh in (0.5, 1.0, 1.5)]))
    cams = rotate_yaw(np.array([[x, 0.0, z] for x, z in path]), YAW) + SHIFT
    polys, al, ca = _polys(rotate_yaw(pts, YAW) + SHIFT, cams, [rotate_yaw(h, YAW) + SHIFT for h in hits])
    assert len(polys) == 1
    p = polys[0]
    assert p.area == pytest.approx(24.0, abs=0.35)
    assert len(p.edges) == 6
    assert sorted(round(e.length) for e in p.edges) == [2, 2, 4, 4, 4, 8]
