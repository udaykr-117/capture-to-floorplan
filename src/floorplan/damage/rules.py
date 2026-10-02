"""Rule engine: damage regions -> concealed-damage flags (each naming its rule) -> scope items keyed to surface ids."""
from pathlib import Path

import numpy as np
import yaml

from floorplan import schema
from floorplan.damage.project import Region

RULES_PATH = Path(__file__).resolve().parents[3] / "configs" / "rules.yaml"


def load_rules(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or RULES_PATH).read_text(encoding="utf8"))


def _area(r: Region, cfg: dict) -> schema.Measurement:
    c2 = cfg["damage"]["cell_m"] ** 2
    return schema.Measurement(value=round(r.cells * c2, 4), unit="m2", note="upper bound: footprint of the detection box on the surface, no segmentation",
                              interval=schema.Interval(low=round(r.cells_low * c2, 4), high=round(r.cells_high * c2, 4),
                                                       method="raster cells eroded / dilated by one cell (provisional); upper bound on the true damaged area"))


def _matches(w: dict, r: Region) -> bool:
    if r.cls not in w["cls"] or r.kind not in w["surface"]:
        return False
    if "below_m" in w and not (r.kind == "wall" and r.extent[3] <= w["below_m"]):
        return False
    if "above_m" in w and not (r.kind == "wall" and r.extent[2] >= w["above_m"]):
        return False
    return True


def evaluate(regions: list[Region], rules: dict, cfg: dict, evidence: dict[int, list[str]] | None = None):
    """-> (DamageRegions, flags, scope items). Region ids D0.. in order of (room, surface, class)."""
    regions = sorted(regions, key=lambda r: (r.room_id, r.surface_id, r.cls, r.extent))
    out_r, flags, items = [], [], []
    for k, r in enumerate(regions):
        rid = f"D{k}"
        u0, u1, v0, v1 = r.extent
        out_r.append(schema.DamageRegion(id=rid, cls=r.cls, room_id=r.room_id, surface_id=r.surface_id, surface_kind=r.kind, area=_area(r, cfg),
                                         position_xz=(round(r.xz[0], 3), round(r.xz[1], 3)), surface_extent_m=(round(u0, 3), round(u1, 3), round(v0, 3), round(v1, 3)),
                                         score_max=r.score_max, n_views=len(r.frames), frames=r.frames, evidence_images=(evidence or {}).get(k, [])))
        sc = rules["scope"][r.cls]
        c2 = cfg["damage"]["cell_m"] ** 2
        if sc["basis"] == "area":
            q = schema.Measurement(value=round(r.cells * c2, 3), unit=sc["unit"], note="upper bound (box footprint)",
                                   interval=schema.Interval(low=round(r.cells_low * c2, 3), high=round(r.cells_high * c2, 3), method="raster cells eroded / dilated (provisional)"))
        elif sc["basis"] == "diagonal":
            dg = float(np.hypot(u1 - u0, v1 - v0))
            q = schema.Measurement(value=round(dg, 3), unit=sc["unit"], note="upper bound: diagonal of the box extent, crack length itself is not measured",
                                   interval=schema.Interval(low=round(max(max(u1 - u0, v1 - v0) - 2 * cfg["damage"]["cell_m"], 0.0), 3), high=round(dg + 2 * cfg["damage"]["cell_m"], 3),
                                                            method="longest box side to box diagonal, +- one cell (provisional)"))
        else:
            q = schema.Measurement(value=1.0, unit=sc["unit"], interval=schema.Interval(low=1.0, high=1.0, method="count of detected regions"))
        items.append(schema.ScopeItem(id=f"S{len(items)}", surface_id=r.surface_id, room_id=r.room_id, region_id=rid, action=sc["action"], quantity=q))
        for w in rules["rules"]:
            if _matches(w["when"], r):
                fid = f"F{len(flags)}"
                flags.append(schema.ConcealedFlag(id=fid, rule_id=w["id"], rule=w["rule"], hypothesis=w["hypothesis"], inspect=w["inspect"],
                                                  room_id=r.room_id, surface_id=r.surface_id, region_ids=[rid]))
                items.append(schema.ScopeItem(id=f"S{len(items)}", surface_id=r.surface_id, room_id=r.room_id, region_id=rid, flag_id=fid, action=rules["inspection_action"],
                                              quantity=schema.Measurement(value=1.0, unit="each", interval=schema.Interval(low=1.0, high=1.0, method="count")),
                                              note=f"rule {w['id']}"))
    return out_r, flags, items
