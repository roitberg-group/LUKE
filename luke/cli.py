"""Command-line interface for LUKE.

Provides subcommands for running the full pipeline and individual steps.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .logging_utils import get_logger, set_global_log_level
from .pipeline import run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _pipeline_impl(
    input: Path,
    output: Path,
    model: str,
    device: str | None,
    threshold: float,
    batch_size: int,
    sanitize: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    if verbose and quiet:
        raise typer.BadParameter("Cannot use --verbose and --quiet together.")
    level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    set_global_log_level(level)
    logger = get_logger("luke.cli")
    logger.debug(
        "Starting pipeline with params: %s",
        {
            "input": input,
            "output": output,
            "model": model,
            "device": device,
            "threshold": threshold,
            "batch_size": batch_size,
            "sanitize": sanitize,
        },
    )
    run_pipeline(
        input,
        output,
        model_name=model,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
        sanitize=sanitize,
    )


@app.command("pipeline")
def pipeline_command(
    input: Path = typer.Argument(..., exists=True, help="Input HDF5 or XYZ file"),
    output: Path = typer.Option(Path("results"), "--output", "-o", help="Output directory"),
    model: str = typer.Option("ANI2xr", "--model", help="ANI model name"),
    device: str | None = typer.Option(None, "--device", help="cpu or cuda"),
    threshold: float = typer.Option(0.5, "--threshold", help="Bad-atom threshold"),
    batch_size: int = typer.Option(1000, "--batch-size", help="Batch size for dataset processing"),
    sanitize: bool = typer.Option(False, "--sanitize", help="Attempt to sanitize fragments"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Silence non-error logs"),
) -> None:  # noqa: D401
    """Run the LUKE end-to-end workflow."""
    # Delegate to internal implementation to avoid B008 on defaults.
    _pipeline_impl(
        input,
        output,
        model,
        device,
        threshold,
        batch_size,
        sanitize,
        verbose,
        quiet,
    )

def main():
    app()


if __name__ == "__main__":
    main()
