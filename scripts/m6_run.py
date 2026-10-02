"""M6 step 2: build the LiDAR plan, replay saved detections (out/m6/<key>/detections.json from m6_detect_explore.py) through projection,
rules and scope, and print every region, flag and scope item.
Usage: uv run python scripts/m6_run.py <capture key>
"""
import json
import sys
from pathlib import Path

from floorplan.config import load_config
from floorplan.damage.detect import Detection
from floorplan.damage.run import run_damage
from floorplan.io.stray import StraySource
from floorplan.pipeline import build_plan

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {"single_room": ROOT / "single_room" / "c00a170fe1", "floor_only": ROOT / "single_scan_floor_only" / "1a8384c3f6",
           "with_ceiling": ROOT / "single_scan_with_ceiling" / "c7d28f72c6"}


def main(key):
    cfg = load_config()
    out = ROOT / "out" / "m6" / key
    dets = [Detection(d["frame"], d["cls"], d["prompt"], d["score"], tuple(d["box"])) for d in json.loads((out / "detections.json").read_text())]
    dets = [d for d in dets if d.score >= cfg["damage"]["score_min"]]
    src = StraySource(SAMPLES[key], cfg)
    plan, inter = build_plan(src, cfg)
    dmg = run_damage(src, plan, inter, cfg, out_dir=None, detections=dets)
    print(f"{key}: {len(dets)} detections -> {dmg.report['patches_on_surfaces']} patches on surfaces -> {len(dmg.regions)} regions")
    print("  dropped:", dmg.report["dropped_reasons"])
    for r in dmg.regions:
        print(f"  {r.id} {r.cls:13s} {r.surface_id:12s} ({r.surface_kind}) area {r.area.value} m2 [{r.area.interval.low}, {r.area.interval.high}] score {r.score_max:.2f} "
              f"views {r.n_views} extent {r.surface_extent_m} frames {r.frames}")
    for f in dmg.concealed_damage_flags:
        print(f"  {f.id} rule {f.rule_id} on {f.surface_id} regions {f.region_ids}")
    for it in dmg.scope_items:
        print(f"  {it.id} {it.surface_id} {it.action}: {it.quantity.value} {it.quantity.unit} [{it.quantity.interval.low}, {it.quantity.interval.high}]")
    (out / "damage_replay.json").write_text(dmg.model_dump_json(indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
