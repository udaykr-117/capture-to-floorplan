import numpy as np
import pytest
from shapely.geometry import Polygon

from floorplan.config import load_config
from floorplan.pipeline import build_plan
from floorplan.register import fuse_ceilings, register, wall_points
from synth import cast_rays_3d, rotate_yaw

CFG = load_config()
FLOOR_Y = -1.4
# A (x 0..3, ceiling 2.5) and B (x 3..8, ceiling 2.8) with a door in the partition (segment 6)
SEGS = [((0, 0), (0, 4)), ((8, 0), (8, 4)), ((0, 0), (3, 0)), ((3, 0), (8, 0)), ((0, 4), (3, 4)), ((3, 4), (8, 4)), ((3, 0), (3, 4))]
HEIGHTS = [2.5, 2.8, 2.5, 2.8, 2.5, 2.8, 2.8]
CEILS = [(0, 3, 2.5), (3, 8, 2.8)]
DOOR = (1.55, 2.45, 0.0, 2.0)
PATH = [(1.5, 2), (1, 1), (2, 3), (4, 2), (5, 1), (6, 3), (7, 1), (5.5, 3.5), (2.2, 0.8)]


class Source:
    """Frames only; `ceilings` False means that capture saw no ceiling."""

    def __init__(self, yaw, shift, ceilings, seed):
        sh = np.array(shift)
        self.name = f"synthetic/{'with' if ceilings else 'no'}-ceiling"
        self.cams = rotate_yaw(np.array([[x, 0.0, z] for x, z in PATH]), yaw) + sh
        self.hits = [rotate_yaw(cast_rays_3d((x, 0.0, z), SEGS, 2.8, FLOOR_Y, [(6, *DOOR)], ceiling=ceilings, ceilings=CEILS, seg_heights=HEIGHTS), yaw) + sh
                     for x, z in PATH]

    def camera_positions(self):
        return self.cams

    def frames(self):
        yield from zip(self.hits, self.cams)


def test_ceilings_are_transferred_with_a_source_and_only_where_the_rooms_overlap():
    pa, ia = build_plan(Source(20.0, (3.0, 0, -2.0), True, 1), CFG)
    pb, ib = build_plan(Source(20.0 + 90 + 12.0, (-6.0, 0, 4.0), False, 7), CFG)
    assert all(r.ceiling_height.value is None for r in pb.rooms)
    reg = register(wall_points(ia["Pa"], ia["Na"], CFG), wall_points(ib["Pa"], ib["Na"], CFG), CFG)
    assert reg.accepted
    fused, copied = fuse_ceilings(pb, pa, reg, CFG)
    assert len(copied) == len(pb.rooms) == 2
    got = sorted(r.ceiling_height.value for r in fused.rooms)
    assert got == pytest.approx([2.5, 2.8], abs=0.03)
    assert all(r.ceiling_height.source.startswith(pa.capture) for r in fused.rooms)
    assert any("come from capture" in l for l in fused.limitations)
    assert pb.rooms[0].ceiling_height.source is None  # the original plan is not modified


def test_a_room_that_does_not_overlap_gets_nothing():
    pa, ia = build_plan(Source(20.0, (3.0, 0, -2.0), True, 1), CFG)
    pb, ib = build_plan(Source(20.0, (3.0, 0, -2.0), False, 7), CFG)
    reg = register(wall_points(ia["Pa"], ia["Na"], CFG), wall_points(ib["Pa"], ib["Na"], CFG), CFG)
    far = reg.__class__(**{**reg.__dict__, "t": reg.t + np.array([40.0, 40.0])})  # pretend B sits 40 m away from A
    fused, copied = fuse_ceilings(pb, pa, far, CFG)
    assert copied == [] and all(r.ceiling_height.value is None for r in fused.rooms)
