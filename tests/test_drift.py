import copy

import numpy as np
import pandas as pd
import pytest

from floorplan.config import load_config
from floorplan.drift.jumps import distribute_jumps, find_jumps
from floorplan.pipeline import build_plan
from synth import cast_rays_3d, rotate_yaw

CFG = load_config()
SHIFT = np.array([3.0, 0, -2.0])
YAW = 25.0
FLOOR_Y = -1.4
# two 4 x 4 m rooms (A: x 0..4, B: x 4..8) with a door at the partition
SEGS = [((0, 0), (0, 4)), ((8, 0), (8, 4)), ((0, 0), (4, 0)), ((4, 0), (8, 0)), ((0, 4), (4, 4)), ((4, 4), (8, 4)), ((4, 0), (4, 4))]
DOOR = (1.55, 2.45, 0.0, 2.0)
LOOP = [(1, 1), (2, 1.5), (3, 2.5), (2, 3.3), (1, 3), (2, 2), (3.4, 2.0), (4.6, 2.0), (5.5, 1), (6.5, 1.5), (7, 2.5), (6, 3.3), (5, 3), (6, 2), (4.6, 2.0), (3.4, 2.0)]


class DriftedSource:
    """Two rooms visited in a loop, three times; the 'tracker' drifts: heading, position and height grow linearly with time."""

    def __init__(self, psi_deg=3.0, shift=(0.25, -0.15), dy=0.05, laps=3, step_az=0.6, step_el=1.2, view_only=False):
        path = LOOP * laps
        n = len(path)
        self.name = "synthetic/drifted"
        self.cams, self.hits = [], []
        p0 = rotate_yaw(np.array([[path[0][0], 0.0, path[0][1]]]), YAW)[0] + SHIFT
        for k, (x, z) in enumerate(path):
            tau = k / (n - 1)
            h = cast_rays_3d((x, 0.0, z), SEGS, 2.5, FLOOR_Y, [(6, *DOOR)], az_step=step_az, el_step=step_el)
            c = np.array([[x, 0.0, z]])
            w = [rotate_yaw(a, YAW) + SHIFT for a in (h, c)]
            if view_only:  # a heading error in what the sensor sees: rotate the hits about the camera itself, the camera stays where it is
                cam = w[1][0]
                q = rotate_yaw(w[0] - cam, psi_deg * tau) + cam
                q[:, 1] -= tau * dy
                c2 = cam.copy()
                c2[1] -= tau * dy
                self.hits.append(q)
                self.cams.append(c2)
            else:
                self.hits.append(self._drift(w[0], p0, tau, psi_deg, shift, dy))
                self.cams.append(self._drift(w[1], p0, tau, psi_deg, shift, dy)[0])
        self.cams = np.array(self.cams)

    @staticmethod
    def _drift(p, p0, tau, psi, shift, dy):
        q = rotate_yaw(p - p0, psi * tau) + p0
        q[:, 0] += tau * shift[0]
        q[:, 2] += tau * shift[1]
        q[:, 1] -= tau * dy
        return q

    def camera_positions(self):
        return self.cams

    def frames(self):
        yield from zip(self.hits, self.cams)


def _cfg():
    cfg = copy.deepcopy(CFG)
    cfg["drift"].update(chunk_frames=8, min_vertical_points=200, use_shift=True)  # the synthetic drift includes a position drift: exercise the shift component
    return cfg


@pytest.fixture(scope="module")
def runs():
    cfg, src = _cfg(), DriftedSource()
    return build_plan(src, cfg, drift=False)[0], build_plan(src, cfg, drift=True)[0], build_plan(DriftedSource(0, (0, 0), 0), cfg, drift=False)[0]


def _err(plan):
    return sum(abs(r.area.value - 16.0) for r in plan.rooms) + abs(plan.stitched.footprint_area.value - 32.0)


def test_the_injected_drift_hurts_and_the_correction_removes_most_of_it(runs):
    off, on, clean = runs
    assert _err(clean) < 0.8                       # no drift: the pipeline is fine
    assert _err(off) > 2 * _err(clean) + 0.5       # drift makes it worse
    assert _err(on) < _err(off) / 2                # the correction removes most of the error
    assert _err(on) < 1.5


def test_corrected_walls_are_close_to_4_m_and_report_what_was_done(runs):
    off, on, _ = runs
    assert "chunk heading" in on.frame.drift_correction and "wall-matching shifts" in on.frame.drift_correction
    assert on.frame.drift_report["associations"] > 10
    assert on.frame.drift_report["max_rotation_deg"] > 0.5  # it found the heading drift (3 deg injected, half is absorbed by the global yaw)
    assert off.frame.drift_correction == "none"
    assert any("NOT corrected" in l for l in off.limitations) and any("Drift:" in l for l in on.limitations)
    for r in on.rooms:
        assert all(abs(w.length.value - 4.0) < 0.25 for w in r.walls if w.source == "plane"), [round(w.length.value, 2) for w in r.walls]


def _odo(positions, dt=0.017):
    n = len(positions)
    return pd.DataFrame(dict(timestamp=np.arange(n) * dt, x=positions[:, 0], y=positions[:, 1], z=positions[:, 2]))


def test_a_relocalization_jump_is_found_and_spread_back_over_the_path():
    n = 400
    t = np.arange(n)
    p = np.c_[0.01 * t, np.zeros(n), np.zeros(n)]               # walking along x at 0.6 m/s
    jump = np.array([-0.30, 0.0, 0.40])                           # tracker snaps back by 50 cm at frame 300
    p[301:] += jump
    odo = _odo(p)
    assert find_jumps(p, odo["timestamp"].values, CFG) == [(300, 301)]
    fixed, rep = distribute_jumps(odo, CFG)
    assert len(rep) == 1 and rep[0]["jump_m"] == pytest.approx(0.5, abs=0.02)
    q = fixed[["x", "y", "z"]].values
    assert np.abs(q[0] - p[0]).max() < 1e-9                      # nothing at the start
    assert np.linalg.norm(q[300] - q[301]) < 0.03                # the jump is gone (only normal motion left)
    assert np.linalg.norm(q[150] - p[150]) == pytest.approx(0.25, abs=0.03)  # half way: half of the jump
    assert np.abs(q[310:] - p[310:]).max() < 1e-9                # after the jump: unchanged
    clean, rep2 = distribute_jumps(_odo(np.c_[0.01 * t, np.zeros(n), np.zeros(n)]), CFG)
    assert rep2 == []


def test_default_components_remove_the_measured_heading_and_floor_error_but_not_position_drift():
    """Sensor-only heading error (4 deg, about the camera) and a 6 cm floor drift. Heading + floor remove what they measure; the position
    offsets a heading error also causes need the optional shift component, so room areas are NOT expected to recover here."""
    from floorplan.drift.chunks import analyze_aligned, analyze_chunk, build_chunks
    from floorplan.drift.correct import estimate
    from floorplan.geometry.align import estimate_frame

    cfg = copy.deepcopy(CFG)
    cfg["drift"].update(chunk_frames=8, min_vertical_points=200)
    assert cfg["drift"]["use_shift"] is False
    src = DriftedSource(psi_deg=4.0, shift=(0.0, 0.0), dy=0.06, view_only=True)
    chunks = build_chunks(src.frames(), cfg)
    frame, *_ = estimate_frame(np.concatenate([c.pts for c in chunks]), src.camera_positions(), cfg)
    before = [analyze_chunk(c, frame, cfg) for c in chunks]
    corr = estimate(before, cfg)
    after = [analyze_aligned(c.idx, corr.apply(c.idx, frame.to_aligned(c.pts)), corr.apply(c.idx, frame.to_aligned(c.cams)), cfg) for c in chunks]
    dev0, dev1 = np.array([a.yaw_dev_deg for a in before]), np.array([a.yaw_dev_deg for a in after])
    assert np.abs(dev0).max() > 1.2 and np.abs(dev1).max() < 0.4          # heading deviation removed (was up to ~1.7 deg)
    f0, f1 = np.array([a.floor_y for a in before]), np.array([a.floor_y for a in after])
    assert np.ptp(f0) > 0.04 and np.ptp(f1) < 0.005                        # floor levelled (was 5 cm apart across the capture)
    assert np.abs(corr.t).max() == 0.0                                     # no shifts applied by default
