import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.pipeline import build_plan
from floorplan.report.render import render_plan
from floorplan.schema import Plan
from synth import box_room, cast_rays_3d, rotate_yaw

CFG = load_config()
SHIFT = np.array([3.0, 0, -2.0])
YAW = 25.0
FLOOR_Y = -1.4
SEGS = [((0, 0), (0, 4)), ((8, 0), (8, 4)), ((0, 0), (8, 0)), ((0, 4), (8, 4)), ((4, 0), (4, 4))]
PATH = [(2, 2), (3, 1), (3, 3), (1.5, 3.4), (2.5, 3.4), (1, 1), (6, 2), (5, 1), (5, 3), (6.5, 3.5), (7, 1), (3.6, 2.0), (4.4, 2.0), (3.5, 3.4), (4.5, 3.4)]


class SyntheticSource:
    name = "synthetic/two-rooms"

    def __init__(self):
        door = (1.55, 2.45, 0.0, 2.0)
        a = box_room(0, 0, 4, 4, 2.5, seed=1, openings=[("x1", *door)])
        b = box_room(4, 0, 8, 4, 2.8, seed=2, openings=[("x0", *door)])
        self.pts = rotate_yaw(np.concatenate([a, b]), YAW) + SHIFT
        self.cams = rotate_yaw(np.array([[x, 0.0, z] for x, z in PATH]), YAW) + SHIFT
        self.hits = [rotate_yaw(cast_rays_3d((x, 0.0, z), SEGS, 2.5, FLOOR_Y, [(4, *door)]), YAW) + SHIFT for x, z in PATH]

    def cloud(self):
        return self.pts

    def camera_positions(self):
        return self.cams

    def frames(self):
        for h, c in zip(self.hits, self.cams):
            yield h, c


@pytest.fixture(scope="module")
def plan():
    return build_plan(SyntheticSource(), CFG)[0]


def test_rooms_areas_walls_and_ceilings_match_the_truth(plan):
    assert len(plan.rooms) == 2
    for r in plan.rooms:
        assert r.area.value == pytest.approx(16.0, abs=0.3)
        assert r.area.interval.low <= 16.0 <= r.area.interval.high
        assert len(r.walls) == 4 and all(w.length.value == pytest.approx(4.0, abs=0.1) for w in r.walls)
        assert all(w.length.interval.low <= 4.0 <= w.length.interval.high for w in r.walls)
    heights = sorted(r.ceiling_height.value for r in plan.rooms)
    assert heights == pytest.approx([2.5, 2.8], abs=0.02)
    for r in plan.rooms:
        truth = 2.5 if abs(r.ceiling_height.value - 2.5) < 0.1 else 2.8
        assert r.ceiling_height.status == "measured" and r.ceiling_height.interval.low <= truth <= r.ceiling_height.interval.high


def test_door_and_adjacency(plan):
    assert len(plan.openings) == 1
    o = plan.openings[0]
    assert o.kind == "door" and o.width.value == pytest.approx(0.9, abs=0.05)
    assert o.width.interval.low - 0.02 <= 0.9 <= o.width.interval.high + 0.02
    assert sorted(x for x in o.rooms if x) == sorted(r.id for r in plan.rooms)
    assert any(a.via == "opening" and a.opening_id == o.id for a in plan.stitched.adjacency)


def test_stitched_footprint_has_no_overlap(plan):
    assert plan.stitched.footprint_area.value == pytest.approx(32.0, abs=0.6)
    assert plan.stitched.max_room_overlap_m2 < 0.01


def test_json_round_trip_and_render(plan, tmp_path):
    again = Plan.model_validate_json(plan.model_dump_json())
    assert again == plan
    out = tmp_path / "plan.png"
    render_plan(plan, out)
    assert out.stat().st_size > 10_000
