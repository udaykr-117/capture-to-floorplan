from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Phone captures to measured floor plans.")


@app.callback()
def main() -> None:
    """Phone captures to measured floor plans."""


@app.command()
def run(capture_dir: Path) -> None:
    """Run the pipeline on one capture folder."""
    typer.echo(f"not implemented: {capture_dir}", err=True)
    raise typer.Exit(code=1)
