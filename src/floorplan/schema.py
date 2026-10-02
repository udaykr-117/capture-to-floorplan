"""Output contract (our own schema: no published one was provided). Plan coordinates are metres in the aligned frame
(walls along x and z, Y up dropped); `frame` gives the yaw and floor height to map back to the capture's world frame."""
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "0.1"


class Interval(BaseModel):
    low: float | None
    high: float | None
    method: str
    calibrated: bool = False


class Measurement(BaseModel):
    value: float | None
    unit: str
    interval: Interval
    status: Literal["measured", "ambiguous", "unmeasurable"] = "measured"
    note: str | None = None
    source: str | None = Field(default=None, description="set when the value comes from another capture instead of this one")


class Wall(BaseModel):
    id: str
    p0: tuple[float, float]
    p1: tuple[float, float]
    length: Measurement
    source: Literal["plane", "virtual"]
    support: float = Field(description="fraction of the wall edge covered by wall points (0 to 1)")


class Opening(BaseModel):
    id: str
    kind: Literal["door", "raised"]
    p0: tuple[float, float]
    p1: tuple[float, float]
    width: Measurement
    bottom_m: float
    top_m: float
    rooms: list[str | None] = Field(description="room ids on the two sides of the wall (None = no room there)")


class Room(BaseModel):
    id: str
    polygon: list[tuple[float, float]]
    area: Measurement
    ceiling_height: Measurement
    observed_fraction: float = Field(description="share of the polygon with observed free space; the rest is completed from wall planes")
    floor_offset_m: float = Field(description="this room's local floor minus the capture's global floor")
    walls: list[Wall]
    openings: list[str]


class Adjacency(BaseModel):
    a: str
    b: str
    via: Literal["opening", "shared_wall"]
    opening_id: str | None = None


class Stitched(BaseModel):
    footprint_area: Measurement
    n_rooms: int
    max_room_overlap_m2: float
    adjacency: list[Adjacency]


class DamageSection(BaseModel):
    status: Literal["not_implemented"] = "not_implemented"
    regions: list = []
    concealed_damage_flags: list = []
    scope_items: list = []


class FrameInfo(BaseModel):
    yaw_deg: float
    floor_world_y: float
    drift_correction: str = Field(default="none", description="the components applied, e.g. 'chunk heading + floor level + odometry jump distribution'; 'none' = poses as-is")
    drift_report: dict = {}


class Plan(BaseModel):
    schema_version: str = SCHEMA_VERSION
    capture: str
    tier: Literal["lidar", "video", "photo"]
    frame: FrameInfo
    rooms: list[Room]
    openings: list[Opening]
    stitched: Stitched
    damage: DamageSection = DamageSection()
    timing_s: dict[str, float]
    limitations: list[str]
    config_sha256: str
