"""Room-local refinement: walls are re-measured from the room's longest visit only."""
import numpy as np
from shapely.geometry import box

from floorplan.config import load_config
from floorplan.drift.chunks import Chunk
from floorplan.drift.correct import Correction
from floorplan.geometry.align import Frame
from floorplan.rooms.polygon import Edge, RoomPoly
from floorplan.rooms.refine import _visit, refine_rooms

CFG = load_config()


def wall_pts(x, rng):
    z, y = np.meshgrid(np.arange(0.3, 3.7, 0.02), np.arange(0.4, 1.5, 0.02))
    return np.stack([np.full(z.size, x) + rng.normal(0, 0.003, z.size), y.ravel(), z.ravel()], 1)


def test_visit_is_longest_consecutive_run():
    assert _visit([True, False, True, True, True, False, True, True]) == [2, 3, 4]
    assert _visit([False, False]) == []


def test_room_dimension_comes_from_one_visit():
    """Room 0..4 m in x. Visit 1 (chunks 0-2) sees walls at x=0.00 and x=4.00; a later visit (chunk 4, drifted) sees them at +0.08 m.
    The global plane (fitted to everything) sits in between; refinement must return the visit-1 positions, width 4.00 m."""
    rng = np.random.default_rng(0)
    inside = np.array([[2.0, 1.4, 2.0]] * 5)
    outside = np.array([[9.0, 1.4, 9.0]] * 5)
    chunks = []
    for k in range(3):
        chunks.append(Chunk(k, 5, np.concatenate([wall_pts(0.0, rng), wall_pts(4.0, rng)]), inside))
    chunks.append(Chunk(3, 5, np.zeros((0, 3)), outside))
    chunks.append(Chunk(4, 5, np.concatenate([wall_pts(0.08, rng), wall_pts(4.08, rng)]), inside))
    g0, g4 = 0.03, 4.03                                                     # where whole-capture planes would land
    rp = RoomPoly(1, box(g0, 0, g4, 4), [Edge(0, g0, 0, 4, "plane", 1.0), Edge(2, 4, g0, g4, "virtual", 0.0),
                                         Edge(0, g4, 0, 4, "plane", 1.0), Edge(2, 0, g0, g4, "virtual", 0.0)])
    out, extra, rep = refine_rooms([rp], chunks, Frame(0.0, 0.0), Correction.identity(5), CFG)
    xs = sorted(e.offset for e in out[0].edges if e.axis == 0)
    assert abs(xs[0] - 0.0) < 0.006 and abs(xs[1] - 4.0) < 0.006, xs
    assert abs(out[0].polygon.bounds[2] - out[0].polygon.bounds[0] - 4.0) < 0.01
    assert rep[0]["refined"] == 2 and rep[0]["visit_chunks"] == 3 and len(extra) == 2
