from pathlib import Path

import typer

from floorplan.config import load_config

app = typer.Typer(no_args_is_help=True, help="Phone captures to measured floor plans.")


@app.callback()
def main() -> None:
    """Phone captures to measured floor plans."""


@app.command()
def run(
    capture_dir: Path,
    out: Path = typer.Option(None, help="Output folder (default: out/<capture name>)."),
    config: Path = typer.Option(None, help="Config YAML (default: configs/default.yaml)."),
) -> None:
    """Run the LiDAR pipeline on one Stray Scanner capture folder: writes plan.json and plan.png."""
    from floorplan.io.stray import StraySource
    from floorplan.pipeline import build_plan
    from floorplan.report.render import render_plan

    cfg = load_config(config)
    src = StraySource(capture_dir, cfg)
    out = out or Path("out") / src.name.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)
    plan, _ = build_plan(src, cfg)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    render_plan(plan, out / "plan.png")
    typer.echo(f"{plan.capture}: {plan.stitched.n_rooms} rooms, footprint {plan.stitched.footprint_area.value} m2, {len(plan.openings)} openings; "
               f"timing {plan.timing_s}")
    for r in plan.rooms:
        c = r.ceiling_height
        typer.echo(f"  {r.id}: area {r.area.value} m2 [{r.area.interval.low}, {r.area.interval.high}], ceiling "
                   + (f"{c.value} m [{c.interval.low}, {c.interval.high}] ({c.status})" if c.value is not None else f"unmeasurable ({c.note})"))
    typer.echo(f"wrote {out / 'plan.json'} and {out / 'plan.png'}")
