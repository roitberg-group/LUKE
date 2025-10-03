"""Command-line interface for LUKE.

Provides subcommands for running the full pipeline and individual steps.
"""

from __future__ import annotations

from pathlib import Path

import typer

from luke.pipeline import run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def pipeline(
    input: Path = typer.Argument(..., exists=True,
                                 help="Input HDF5 or XYZ file"),
    output: Path = typer.Option(
        Path("results"), "--output", "-o", help="Output directory"),
    model: str = typer.Option("ANI2xr", "--model", help="ANI model name"),
    device: str | None = typer.Option(None, "--device", help="cpu or cuda"),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Bad-atom threshold"),
    batch_size: int = typer.Option(
        1000, "--batch-size", help="Batch size for dataset processing"),
    sanitize: bool = typer.Option(
        False, "--sanitize", help="Attempt to sanitize fragments"),
):
    """Run the LUKE end-to-end workflow."""
    run_pipeline(input, output, model_name=model, device=device,
                 threshold=threshold, batch_size=batch_size, sanitize=sanitize)


def main():
    app()


if __name__ == "__main__":
    main()
