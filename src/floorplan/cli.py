from pathlib import Path

import typer

from floorplan.config import load_config

app = typer.Typer(no_args_is_help=True, help="Phone captures to measured floor plans.")


@app.callback()
def main() -> None:
    """Phone captures to measured floor plans."""


def _detect_tier(path: Path) -> str:
    if path.is_file():
        return "video"
    if (path / "camera_matrix.csv").exists():
        return "lidar"
    return "photo"


@app.command()
def run(
    capture: Path,
    tier: str = typer.Option(None, help="lidar (Stray Scanner folder), video (an mp4/mov file) or photo (a folder of room folders). Default: detected."),
    out: Path = typer.Option(None, help="Output folder (default: out/<capture name>)."),
    config: Path = typer.Option(None, help="Config YAML (default: configs/default.yaml)."),
    damage: bool = typer.Option(True, help="Run damage detection (LiDAR tier; needs `--group models`, about 1 s per keyframe on CPU)."),
) -> None:
    """Run the pipeline on one capture: writes plan.json and plan.png."""
    from floorplan.report.render import render_plan

    cfg = load_config(config)
    tier = tier or _detect_tier(capture)
    name = (capture.parent.name + "_" + capture.stem) if capture.is_file() else capture.name
    out = out or Path("out") / f"{tier}_{name}"
    out.mkdir(parents=True, exist_ok=True)
    if tier == "lidar":
        from floorplan.io.stray import StraySource
        from floorplan.pipeline import build_plan

        src = StraySource(capture, cfg)
        plan, inter = build_plan(src, cfg)
        if damage and plan.rooms:
            try:
                from floorplan.damage.run import run_damage

                plan.damage = run_damage(src, plan, inter, cfg, out_dir=out)
                plan.limitations = [x for x in plan.limitations if not x.startswith("Damage detection has not")]
                plan.limitations.append("Damage: zero-shot box detector, accuracy untested, area is an upper bound; regions and flags are candidates to check (see damage.note).")
            except ImportError as e:
                plan.damage.note = f"damage detection skipped: {e} (run with `uv run --group models plan run ...`)"
    elif tier == "video":
        from floorplan.tiers.images import run_video

        plan, _ = run_video(capture, out / "work", cfg)
    elif tier == "photo":
        from floorplan.tiers.images import run_photos

        plan, _ = run_photos(capture, out / "work", cfg)
    else:
        raise typer.BadParameter("tier must be lidar, video or photo")
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")
    typer.echo(f"{plan.capture} [{plan.tier}]: {plan.stitched.n_rooms} rooms, footprint {plan.stitched.footprint_area.value} m2, {len(plan.openings)} openings; timing {plan.timing_s}")
    for r in plan.rooms:
        c = r.ceiling_height
        typer.echo(f"  {r.id}: area {r.area.value} m2 [{r.area.interval.low}, {r.area.interval.high}], ceiling "
                   + (f"{c.value} m [{c.interval.low}, {c.interval.high}] ({c.status})" if c.value is not None else f"unmeasurable ({c.note})"))
    d = plan.damage
    typer.echo(f"  damage: {d.status}; {len(d.regions)} regions, {len(d.concealed_damage_flags)} concealed-damage flags, {len(d.scope_items)} scope items" + ("" if d.status == "run" else f" ({d.note})"))
    for r in d.regions:
        typer.echo(f"    {r.id} {r.cls} on {r.surface_id}: <= {r.area.value} m2, score {r.score_max:.2f}, {r.n_views} views")
    for f in d.concealed_damage_flags:
        typer.echo(f"    {f.id} [{f.rule_id}] {f.surface_id}: {f.hypothesis}")
    for line in plan.limitations:
        typer.echo(f"  note: {line}")
    typer.echo(f"wrote {out / 'plan.json'} and {out / 'plan.png'}")
