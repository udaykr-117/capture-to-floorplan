import numpy as np
import pytest

from floorplan.config import load_config
from floorplan.geometry.align import align
from floorplan.geometry.walls import find_wall_planes
from floorplan.register import (match_segments, register, segments, summarize_widths, to_other, wall_points, width_pairs)
from synth import box_room, rotate_yaw

CFG = load_config()
FLOOR_Y = -1.4
CORNERS = np.array([[3.0, FLOOR_Y, 0.0], [8.0, FLOOR_Y, 0.0], [8.0, FLOOR_Y, 4.0], [3.0, FLOOR_Y, 4.0], [5.5, FLOOR_Y, 2.0]])


def _capture(points, yaw, shift, seed):
    """A 'capture': world points under a rigid motion; returns aligned wall points, planes and the frame."""
    pts = rotate_yaw(points, yaw) + np.array(shift)
    cams = rotate_yaw(np.array([[5.0, 0.0, 2.0], [6.5, 0.0, 3.0], [3.0, 0.0, 2.0]]), yaw) + np.array(shift)
    al, Pa, Na = align(pts, cams, CFG)
    return dict(W=wall_points(Pa, Na, CFG), planes=find_wall_planes(Pa, Na, CFG), frame=al.frame, yaw=yaw, shift=np.array(shift))


def _scene(scale_x=1.0, seed=0, rooms=((0, 3), (3, 8))):  # 3 m and 5 m wide: not symmetric under a 180-degree turn
    parts = [box_room(x0 * scale_x, 0, x1 * scale_x, 4.0, 2.5, seed=seed + i) for i, (x0, x1) in enumerate(rooms)]
    return np.concatenate(parts)


def _true_map(A, B, pts_world_base):
    """Ground-truth positions of base-scene points in each capture's aligned (x', z') frame."""
    out = []
    for cap in (A, B):
        p = rotate_yaw(pts_world_base, cap["yaw"]) + cap["shift"]
        out.append(cap["frame"].to_aligned(p)[:, [0, 2]])
    return out


def test_registration_recovers_the_known_motion_and_beats_the_wrong_rotation():
    A = _capture(_scene(seed=1), 20.0, (3.0, 0, -2.0), 1)
    B = _capture(_scene(seed=5), 20.0 + 90 + 31.0, (-7.0, 0, 5.0), 2)  # other frame, rotated by 90 + 31 degrees
    reg = register(A["W"], B["W"], CFG)
    assert reg.accepted and reg.fitness_5cm > 0.9 and reg.median_cm < 1.5
    # the outer walls of a rectangle also match under a 180-degree turn; only the off-centre partition breaks that symmetry
    assert reg.null_fitness_5cm < reg.fitness_5cm - 0.1
    a_pts, b_pts = _true_map(A, B, CORNERS)
    assert np.abs(reg.apply(b_pts) - a_pts).max() < 0.03  # corners land within 3 cm of where they really are
    assert abs(reg.yaw_resid_deg) < 0.6


def test_identical_scenes_have_no_width_disagreement():
    A = _capture(_scene(seed=1), 20.0, (3.0, 0, -2.0), 1)
    B = _capture(_scene(seed=7), 20.0 + 180, (1.0, 0, 9.0), 2)
    reg = register(A["W"], B["W"], CFG)
    SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
    m = match_segments(SA, SB, CFG)
    assert len(m) >= 5
    pairs = width_pairs(SA, SB, m, CFG)
    assert len(pairs) >= 3
    assert max(abs(p["delta"]) for p in pairs) < 0.015


def test_a_0p4_percent_scale_difference_is_measured_as_such():
    A = _capture(_scene(seed=1), 20.0, (3.0, 0, -2.0), 1)
    B = _capture(_scene(scale_x=1.004, seed=7), 20.0 + 90, (1.0, 0, 9.0), 2)  # B is 0.4% longer along its x
    reg = register(A["W"], B["W"], CFG)
    SA, SB = segments(A["planes"]), [to_other(s, reg) for s in segments(B["planes"])]
    pairs = width_pairs(SA, SB, match_segments(SA, SB, CFG), CFG)
    stretched = [p for p in pairs if p["width_a"] > 3.0 and abs(p["width_b"] / p["width_a"] - 1.004) < 0.002]
    assert stretched, [(round(p["width_a"], 3), round(p["width_b"], 3)) for p in pairs]
    rows = summarize_widths(pairs, CFG)
    allrow = rows[-1]
    assert allrow["n"] >= 3
    assert allrow["median_signed_pct"] > 0.1  # the stretch shows up as a positive signed difference
