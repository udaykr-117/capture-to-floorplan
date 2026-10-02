"""An opening seen on both faces of one wall is reported once."""
from floorplan.config import load_config
from floorplan.rooms.openings import Opening, merge_wall_faces

CFG = load_config()


def op(offset, s0, s1, rays, axis=0):
    return Opening(axis, offset, s0, s1, s1 - s0, (s1 - s0 - 0.02, s1 - s0), "door", 0.05, 2.0, rays)


def test_two_faces_of_one_wall_are_one_opening():
    out = merge_wall_faces([op(2.49, 4.34, 5.11, 50), op(2.64, 4.34, 5.13, 80)], CFG)
    assert len(out) == 1 and out[0].n_rays == 80


def test_different_openings_stay():
    ops = [op(2.49, 4.34, 5.11, 50), op(2.49, 6.0, 6.8, 40),        # same wall, different place
           op(3.20, 4.34, 5.11, 30),                                  # parallel wall 71 cm away
           op(4.4, 2.49, 3.2, 20, axis=2)]                            # perpendicular wall
    assert len(merge_wall_faces(ops, CFG)) == 4


def test_same_room_on_both_sides_is_not_an_opening():
    from floorplan.rooms.openings import drop_same_room

    a, b, c = op(1.0, 0, 1, 10), op(2.0, 0, 1, 10), op(3.0, 0, 1, 10)
    a.rooms, b.rooms, c.rooms = ("R1", "R1"), ("R1", "R2"), ("R4", None)
    assert [o.offset for o in drop_same_room([a, b, c])] == [2.0, 3.0]
