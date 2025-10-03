"""
High-level pipeline orchestration for LUKE.

This module connects force/uncertainty computation, isolation of high-error atoms,
and optional sanitization into a cohesive workflow.
"""

from __future__ import annotations

from pathlib import Path
import typing as tp

import torch
import torchani
from rich.progress import track
from .logging_utils import get_logger

from .ani_forces import ANIForceCalculator
from .isolator import Isolator
from .structure_sanitizer import sanitize_xyz_file


def ensure_dir(path: tp.Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_pipeline(
    input_path: tp.Union[str, Path],
    output_dir: tp.Union[str, Path] = "results",
    model_name: str = "ANI2xr",
    device: tp.Optional[str] = None,
    threshold: float = 0.5,
    batch_size: int = 1000,
    sanitize: bool = False,
) -> Path:
    """
    Run LUKE end-to-end:
    1) compute forces/uncertainty and select structures with high weighted stdev
    2) for each structure, isolate high-error atoms into capped fragments
    3) optionally sanitize the fragments

    Returns the output directory path.
    """
    logger = get_logger("luke.pipeline")
    input_path = Path(input_path)
    out_dir = ensure_dir(output_dir)

    # Forces & uncertainty
    calc = ANIForceCalculator(model_name=model_name,
                              device=device, threshold=threshold)
    species_b, coords_b, bad_b, energy_b, qbc_b, stdev_b = calc.process_dataset(
        str(input_path), batch_size=batch_size
    )

    if species_b.numel() == 0:
        logger.warning(
            "No structures exceeded the weighted_stdev threshold. Nothing to isolate.")
        return out_dir

    # Isolate high-error atoms per structure
    model = getattr(torchani.models, model_name)().to(calc.device)

    for i in track(range(species_b.shape[0]), description="Isolating high-error atoms"):
        species = species_b[i].unsqueeze(0)  # (1, N)
        coords = coords_b[i].unsqueeze(0)    # (1, N, 3)
        bad_mask = bad_b[i]                  # (N,)
        bad_idxs = torch.nonzero(
            bad_mask > 0, as_tuple=False).view(-1).tolist()
        if not bad_idxs:
            logger.debug("Structure %d had no bad atoms above threshold.", i)
            continue

        iso = Isolator(model=model, threshold=threshold)
        iso.species = species
        iso.coordinates = coords.double()
        iso.tensors_to_numpy(conformer_idx=0)
        iso.neighbors = iso.get_neighbors(bad_idxs)

        prefix = out_dir / f"frag_{input_path.stem}_{i}_"
        iso.process_molecule(bad_idxs, output_file_prefix=str(prefix))
        # Optionally sanitize produced files
        if sanitize:
            for frag_file in sorted(out_dir.glob(f"{prefix.name}*.xyz")):
                sanitized = frag_file.with_name(f"sanitized_{frag_file.name}")
                sanitize_xyz_file(frag_file, sanitized)

    if sanitize:
        logger.info("Sanitization complete for generated fragments.")
    logger.info("Pipeline finished. Results in: %s", out_dir)
    return out_dir
