from pathlib import Path

import torch

from luke.pipeline import run_pipeline
from luke.structure_sanitizer import (
    compute_connectivity,
    largest_component_indices,
    sanitize_species_coordinates,
)


def test_compute_connectivity_and_lcc():
    import numpy as np
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ])
    adj = compute_connectivity(coords, threshold=1.6)
    keep = largest_component_indices(adj)
    assert keep == [0, 1]


def test_sanitize_species_coordinates():
    species = torch.tensor([[1, 1, 8]])  # H H O
    coords = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]])
    z_out, xyz_out = sanitize_species_coordinates(
        species, coords, threshold=1.6)
    assert z_out.shape[1] == 2
    assert xyz_out.shape[1] == 2


def test_run_pipeline_sanitize(tmp_path: Path):
    # minimal xyz file
    src = tmp_path / "toy.xyz"
    src.write_text(
        """3
toy
H 0.0 0.0 0.0
H 1.0 0.0 0.0
O 5.0 0.0 0.0
"""
    )
    out_dir = tmp_path / "out"
    # Run without relying on model heavy ops by expecting no fragments; ensures path works
    # This call will attempt model init; if environment lacks torchani datasets, it should still no-op gracefully
    run_pipeline(src, out_dir, device="cpu", batch_size=1, sanitize=True)
    # Directory should exist
    assert out_dir.exists()
