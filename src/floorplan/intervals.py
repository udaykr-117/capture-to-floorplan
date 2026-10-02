"""Intervals per measurement (see configs/default.yaml `intervals`): plane spread, systematic and room-extent terms; lengths and areas calibrated on repeat captures only."""
import numpy as np

from floorplan.geometry.walls import WallPlane
from floorplan.rooms.polygon import Edge
from floorplan.schema import Interval, Measurement


def _sigma_lookup(planes: list[WallPlane]) -> dict:
    return {(p.axis, round(p.offset, 6)): p.sigma for p in planes}


def _line_sigma(sig: dict, axis: int, value: float, cfg: dict) -> float:
    s = sig.get((axis, round(value, 6)))
    if s is None:
        return cfg["intervals"]["virtual_end_halfwidth_m"] / cfg["intervals"]["sigma_k"]
    return float(np.hypot(s, cfg["intervals"]["plane_position_sigma_m"]))  # fit spread plus the systematic term from cross-capture agreement


def wall_length(edge: Edge, planes: list[WallPlane], cfg: dict) -> Measurement:
    iv = cfg["intervals"]
    sig = _sigma_lookup(planes)
    other = 2 if edge.axis == 0 else 0  # the wall ends lie on lines of the other axis
    s0, s1 = _line_sigma(sig, other, edge.s0, cfg), _line_sigma(sig, other, edge.s1, cfg)
    hw = max(float(np.hypot(np.hypot(iv["sigma_k"] * s0, iv["sigma_k"] * s1), iv["unsupported_halfwidth_m"] * (1 - edge.support))), iv["min_halfwidth_m"])
    hw = float(np.hypot(hw, iv["sigma_k"] * iv.get("scale_rel_sigma", 0.0) * edge.length))   # image tiers: the metric scale itself is uncertain
    hw = float(np.hypot(hw, iv.get("extent_halfwidth_m", 0.0)))          # where the room is cut differs between repeat captures (calibrated, config)
    return Measurement(value=round(edge.length, 4), unit="m",
                       interval=Interval(low=round(edge.length - hw, 4), high=round(edge.length + hw, 4), method="end-plane spread + 3.5 cm systematic + virtual ends + unsupported length + room-extent term fitted on repeat captures (robust 2 sigma)"))


def room_area(area: float, edges: list[Edge], observed_fraction: float, planes: list[WallPlane], cfg: dict) -> Measurement:
    iv = cfg["intervals"]
    sig = _sigma_lookup(planes)
    var = sum((e.length * _line_sigma(sig, e.axis, e.offset, cfg)) ** 2 for e in edges)
    hw = float(np.hypot(iv["sigma_k"] * np.sqrt(var), iv["unobserved_area_weight"] * (1 - observed_fraction) * area))
    hw = float(np.hypot(hw, iv["sigma_k"] * 2 * iv.get("scale_rel_sigma", 0.0) * area))        # an area scales with the square of a length scale
    hw = float(np.hypot(hw, iv.get("extent_area_rel", 0.0) * area))       # room extent differs between repeat captures (calibrated, config)
    hw = max(hw, iv["min_area_rel"] * area)
    return Measurement(value=round(area, 3), unit="m2",
                       interval=Interval(low=round(area - hw, 3), high=round(area + hw, 3), method="edge-plane spread + 3.5 cm systematic + unobserved share + room-extent term fitted on repeat captures (conformal 95%)"))
