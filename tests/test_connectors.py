"""Corridors get their own room; doorways and dead-end alcoves do not (synthetic free-space map, 5 cm cells)."""
import copy

import numpy as np

from floorplan.config import load_config
from floorplan.rooms.freespace import Grid, segment_rooms

CELL = 0.05


def fill(free, x0, x1, z0, z1):
    free[int(round(z0 / CELL)):int(round(z1 / CELL)), int(round(x0 / CELL)):int(round(x1 / CELL))] = True


def scene():
    free = np.zeros((int(7 / CELL), int(11 / CELL)), bool)
    fill(free, 0, 3, 0, 3)          # room A
    fill(free, 5, 8, 0, 3)          # room B
    fill(free, 3, 5, 1.0, 1.9)      # corridor A-B: 0.9 m wide, 2 m long
    fill(free, 0, 3, 3.2, 6.2)      # room C
    fill(free, 1.0, 1.9, 3.0, 3.2)  # doorway A-C through a 0.2 m wall
    fill(free, 8, 10, 0.5, 1.2)     # dead-end alcove off B: 0.7 m wide, 2 m long, touches only B
    return free, Grid(0.0, 0.0, CELL, free.shape[0], free.shape[1])


def at(labels, x, z):
    return labels[int(z / CELL), int(x / CELL)]


def test_corridor_is_its_own_room_doorway_and_alcove_are_not():
    cfg = copy.deepcopy(load_config())
    cfg["rooms"]["split_connectors"] = True
    free, grid = scene()
    lab = segment_rooms(free, grid, cfg)
    a, b, c, corr = at(lab, 1.5, 1.5), at(lab, 6.5, 1.5), at(lab, 1.5, 4.7), at(lab, 4.0, 1.45)
    assert len({a, b, c, corr}) == 4 and 0 not in {a, b, c, corr}
    assert at(lab, 1.45, 3.1) in (a, c)            # the doorway neck stays with a room
    assert at(lab, 9.0, 0.85) == b                  # the alcove stays with room B


def test_switch_off_keeps_the_old_behaviour():
    cfg = copy.deepcopy(load_config())
    cfg["rooms"]["split_connectors"] = False
    free, grid = scene()
    lab = segment_rooms(free, grid, cfg)
    assert at(lab, 4.0, 1.45) in (at(lab, 1.5, 1.5), at(lab, 6.5, 1.5))   # corridor absorbed by a neighbour
