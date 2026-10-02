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


class DamageRegion(BaseModel):
    id: str
    cls: Literal["stain", "mold", "crack", "peeling_paint", "hole"]
    room_id: str
    surface_id: str = Field(description="wall edge id (R1.W2), or R1.ceiling / R1.floor")
    surface_kind: Literal["wall", "ceiling", "floor"]
    area: Measurement = Field(description="footprint of the detection box(es) on the surface: an UPPER bound, there is no segmentation model")
    position_xz: tuple[float, float] = Field(description="centre of the patch in the plan frame (m)")
    surface_extent_m: tuple[float, float, float, float] = Field(description="u0, u1, v0, v1 on the surface: wall = along-wall s and height above the room floor; ceiling/floor = x', z'")
    score_max: float = Field(description="best detector score (OWL-ViT zero-shot; not a calibrated probability)")
    n_views: int
    frames: list[int]
    evidence_images: list[str] = []


class ConcealedFlag(BaseModel):
    id: str
    rule_id: str
    rule: str = Field(description="the rule that fired, in words")
    hypothesis: str
    inspect: str
    room_id: str
    surface_id: str
    region_ids: list[str]


class ScopeItem(BaseModel):
    id: str
    surface_id: str
    room_id: str
    region_id: str | None
    flag_id: str | None = None
    action: str
    quantity: Measurement
    note: str | None = None


class DamageSection(BaseModel):
    status: Literal["not_run", "run"] = "not_run"
    note: str = "damage detection was not run"
    regions: list[DamageRegion] = []
    concealed_damage_flags: list[ConcealedFlag] = []
    scope_items: list[ScopeItem] = []
    report: dict = {}


class FrameInfo(BaseModel):
    yaw_deg: float
    floor_world_y: float
    drift_correction: str = Field(default="none", description="the components applied, e.g. 'chunk heading + floor level + odometry jump distribution'; 'none' = poses as-is")
    drift_report: dict = {}
    tier_report: dict = {}


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
