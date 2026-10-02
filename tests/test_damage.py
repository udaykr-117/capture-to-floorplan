"""Damage projection, merging and rules on synthetic surfaces with a known extent (no model needed)."""
import numpy as np
from shapely.geometry import box

from floorplan.config import load_config
from floorplan.damage.detect import Detection, iou, nms
from floorplan.damage.project import merge_patches, project_detection
from floorplan.damage.rules import evaluate, load_rules

CFG = load_config()
ROOM = [("R1", box(0, 0, 4, 4), 0.0, [("R1.W0", 0, 0.0, 0.0, 4.0), ("R1.W1", 2, 4.0, 0.0, 4.0)])]
CAM = np.array([2.0, 1.5, 2.0])
ident = lambda p: p


def wall_patch_points(z0, z1, y0, y1, step=0.01):
    z, y = np.meshgrid(np.arange(z0, z1, step), np.arange(y0, y1, step))
    return np.stack([np.zeros(z.size), y.ravel(), z.ravel()], 1)       # on the plane x' = 0


def ceiling_points(x0, x1, z0, z1, h=2.6, step=0.01):
    x, z = np.meshgrid(np.arange(x0, x1, step), np.arange(z0, z1, step))
    return np.stack([x.ravel(), np.full(x.size, h), z.ravel()], 1)


def det(frame, cls="stain"):
    return Detection(frame, cls, "p", 0.2, (0, 0, 10, 10))


def patches(pts, cls="stain", frames=(0, 25)):
    out = []
    for f in frames:
        p, why = project_detection(det(f, cls), pts, ident, CAM, np.eye(3), ROOM, CFG)
        assert p is not None, why
        out.append(p)
    return out


def test_wall_patch_surface_and_area():
    ps = patches(wall_patch_points(1.0, 2.0, 1.0, 1.5))
    assert ps[0].kind == "wall" and ps[0].surface_id == "R1.W0" and ps[0].room_id == "R1"
    regions, _ = merge_patches(ps, CFG)
    assert len(regions) == 1
    r = regions[0]
    assert abs(r.cells * 0.05**2 - 0.5) < 0.06, r.cells * 0.05**2          # true area 1.0 x 0.5 m
    assert r.cells_low < r.cells < r.cells_high
    u0, u1, v0, v1 = r.extent
    assert abs(u0 - 1.0) < 0.1 and abs(u1 - 2.0) < 0.1 and abs(v0 - 1.0) < 0.1 and abs(v1 - 1.5) < 0.1


def test_single_view_is_dropped():
    regions, dropped = merge_patches(patches(wall_patch_points(1.0, 2.0, 1.0, 1.5), frames=(0,)), CFG)
    assert regions == [] and "1 keyframe" in dropped[0]


def test_two_separate_patches_stay_separate():
    ps = patches(wall_patch_points(0.5, 0.8, 1.0, 1.3)) + patches(wall_patch_points(3.0, 3.3, 1.0, 1.3))
    regions, _ = merge_patches(ps, CFG)
    assert len(regions) == 2


def test_rules_fire_by_location_and_name_the_rule():
    rules = load_rules()
    ceil = patches(ceiling_points(1.0, 1.6, 1.0, 1.6))
    assert ceil[0].kind == "ceiling" and ceil[0].surface_id == "R1.ceiling"
    base = patches(wall_patch_points(1.0, 1.5, 0.0, 0.2))
    mid = patches(wall_patch_points(2.0, 2.5, 0.6, 1.0))
    upper = patches(wall_patch_points(2.0, 2.5, 1.4, 1.9))
    regions, _ = merge_patches(ceil + base + mid + upper, CFG)
    reg, flags, items = evaluate(regions, rules, CFG)
    fired = {(f.surface_id, f.rule_id) for f in flags}
    assert ("R1.ceiling", "CEILING_STAIN") in fired
    assert ("R1.W0", "WALL_BASE_MOISTURE") in fired
    assert ("R1.W0", "WALL_UPPER_STAIN") in fired
    assert len(flags) == 3, fired                   # the mid-height wall stain fires nothing
    assert all(f.rule and f.rule_id for f in flags)
    # every region has a repair item and every flag an inspection item, all keyed to a surface id
    assert len([i for i in items if i.flag_id is None]) == len(reg) == 4
    assert len([i for i in items if i.flag_id]) == 3
    assert all(i.surface_id for i in items)


def test_furniture_height_horizontal_surface_is_rejected():
    table = np.stack([*np.meshgrid(np.arange(1, 1.5, 0.01), np.arange(1, 1.5, 0.01))], -1).reshape(-1, 2)
    pts = np.stack([table[:, 0], np.full(len(table), 0.75), table[:, 1]], 1)    # a tabletop 0.75 m above the floor
    p, why = project_detection(det(0), pts, ident, CAM, np.eye(3), ROOM, CFG)
    assert p is None


def test_floor_is_not_reported_by_default():
    pts = np.stack([*np.meshgrid(np.arange(1, 1.5, 0.01), np.arange(1, 1.5, 0.01))], -1).reshape(-1, 2)
    pts = np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]], 1)
    p, why = project_detection(det(0), pts, ident, CAM, np.eye(3), ROOM, CFG)
    assert p is None and "floor" in why


def test_box_on_two_planes_is_dropped():
    a = wall_patch_points(1.0, 2.0, 1.0, 1.5)
    b = ceiling_points(1.0, 2.0, 1.0, 1.5)
    p, why = project_detection(det(0), np.concatenate([a[:len(a) // 2], b[:len(b) // 2 + 1000]]), ident, CAM, np.eye(3), ROOM, CFG)
    assert p is None and "one plane" in why


def test_nms_keeps_best_box_per_class():
    a, b = Detection(0, "stain", "x", 0.3, (0, 0, 100, 100)), Detection(0, "stain", "y", 0.2, (5, 5, 105, 105))
    c = Detection(0, "mold", "z", 0.15, (0, 0, 100, 100))
    kept = nms([b, a, c])
    assert {(d.cls, d.score) for d in kept} == {("stain", 0.3), ("mold", 0.15)}
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
