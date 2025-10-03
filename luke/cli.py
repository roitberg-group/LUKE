"""Command-line interface for LUKE.

Provides subcommands for running the full pipeline and individual steps.
"""

from __future__ import annotations

from pathlib import Path
import typer
from typing import Optional

from .pipeline import run_pipeline
from .logging_utils import get_logger, set_global_log_level

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def pipeline(
    input: Path = typer.Argument(..., exists=True,
                                 help="Input HDF5 or XYZ file"),
    output: Path = typer.Option(
        Path("results"), "--output", "-o", help="Output directory"),
    model: str = typer.Option("ANI2xr", "--model", help="ANI model name"),
    device: Optional[str] = typer.Option(None, "--device", help="cpu or cuda"),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Bad-atom threshold"),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for dataset processing"),
    sanitize: bool = typer.Option(
        False, "--sanitize", help="Attempt to sanitize fragments"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Silence non-error logs"),
):
    """Run the LUKE end-to-end workflow."""
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together.")
    level = "INFO"
    if verbose:
        level = "DEBUG"
    if quiet:
        level = "WARNING"
    set_global_log_level(level)
    logger = get_logger("luke.cli")
    logger.debug("Starting pipeline with params: %s", {
        "input": input,
        "output": output,
        "model": model,
        "device": device,
        "threshold": threshold,
        "batch_size": batch_size,
        "sanitize": sanitize,
    })
    run_pipeline(
        input,
        output,
        model_name=model,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
        sanitize=sanitize,
    )


def main():
    app()


if __name__ == "__main__":
    main()
