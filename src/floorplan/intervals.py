"""Provisional, uncalibrated intervals (see configs/default.yaml `intervals`). Replace with calibrated ones in M3/M7."""
import numpy as np

from floorplan.geometry.walls import WallPlane
from floorplan.rooms.polygon import Edge
from floorplan.schema import Interval, Measurement


def _sigma_lookup(planes: list[WallPlane]) -> dict:
    return {(p.axis, round(p.offset, 6)): p.sigma for p in planes}


def _line_sigma(sig: dict, axis: int, value: float, cfg: dict) -> float:
    s = sig.get((axis, round(value, 6)))
    return s if s is not None else cfg["intervals"]["virtual_end_halfwidth_m"] / cfg["intervals"]["sigma_k"]


def wall_length(edge: Edge, planes: list[WallPlane], cfg: dict) -> Measurement:
    iv = cfg["intervals"]
    sig = _sigma_lookup(planes)
    other = 2 if edge.axis == 0 else 0  # the wall ends lie on lines of the other axis
    s0, s1 = _line_sigma(sig, other, edge.s0, cfg), _line_sigma(sig, other, edge.s1, cfg)
    hw = max(float(np.hypot(np.hypot(iv["sigma_k"] * s0, iv["sigma_k"] * s1), iv["unsupported_halfwidth_m"] * (1 - edge.support))), iv["min_halfwidth_m"])
    return Measurement(value=round(edge.length, 4), unit="m",
                       interval=Interval(low=round(edge.length - hw, 4), high=round(edge.length + hw, 4), method="end-plane spread + virtual ends + unsupported length (provisional)"))


def room_area(area: float, edges: list[Edge], observed_fraction: float, planes: list[WallPlane], cfg: dict) -> Measurement:
    iv = cfg["intervals"]
    sig = _sigma_lookup(planes)
    var = sum((e.length * _line_sigma(sig, e.axis, e.offset, cfg)) ** 2 for e in edges)
    hw = float(np.hypot(iv["sigma_k"] * np.sqrt(var), iv["unobserved_area_weight"] * (1 - observed_fraction) * area))
    hw = max(hw, iv["min_area_rel"] * area)
    return Measurement(value=round(area, 3), unit="m2",
                       interval=Interval(low=round(area - hw, 3), high=round(area + hw, 3), method="edge-plane spread + unobserved share (provisional)"))
