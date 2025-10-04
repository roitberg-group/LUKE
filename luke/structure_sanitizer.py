"""Sanitization utilities: keep the largest connected component by distance cutoff.

Lightweight, no RDKit dependency. Works with luke.io_utils read/write tensors.
"""

from pathlib import Path

import numpy as np
import torch

from .io_utils import read_xyz, write_xyz

ArrayLike = np.ndarray | torch.Tensor


def compute_connectivity(coords: ArrayLike, threshold: float = 1.8) -> np.ndarray:
    """Compute boolean connectivity by Euclidean distance.

    coords: (..., N, 3) or (N, 3) array; operates on the last (N, 3).
    Returns an (N, N) boolean adjacency with diagonal True.
    """
    c = coords.detach().cpu().numpy() if isinstance(
        coords, torch.Tensor) else np.asarray(coords)
    if c.ndim == 3:
        # assume (1, N, 3)
        c = c[0]
    # pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    adj = dist <= float(threshold)
    # ensure diagonal is True
    np.fill_diagonal(adj, True)
    return adj


def largest_component_indices(adj: ArrayLike) -> list[int]:
    """Return indices of the largest connected component of adjacency matrix."""
    A = np.asarray(adj).astype(bool)
    N = A.shape[0]
    visited = np.zeros(N, dtype=bool)
    best: list[int] = []
    for start in range(N):
        if visited[start]:
            continue
        # BFS
        comp: list[int] = []
        q = [start]
        visited[start] = True
        while q:
            u = q.pop(0)
            comp.append(u)
            neighbors = np.where(A[u])[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
        if len(comp) > len(best):
            best = comp
    return sorted(best)


def sanitize_species_coordinates(
    species: torch.Tensor, coordinates: torch.Tensor, threshold: float = 1.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only atoms in the largest connected component; returns tensors with batch dimension preserved (1, K)."""
    # Expect (1, N) and (1, N, 3)
    assert species.dim() == 2 and coordinates.dim(
    ) == 3, "Expect batched tensors shapes (1, N) and (1, N, 3)"
    z = species[0]
    xyz = coordinates[0]
    # drop padded atoms (-1) before connectivity
    mask = z != -1
    z2 = z[mask]
    xyz2 = xyz[mask]
    if z2.numel() == 0:
        return species[:, :0], coordinates[:, :0]
    adj = compute_connectivity(xyz2, threshold=threshold)
    keep = largest_component_indices(adj)
    z_out = z2[keep].unsqueeze(0)
    xyz_out = xyz2[keep].unsqueeze(0)
    return z_out, xyz_out


def sanitize_xyz_file(src: str | Path, dest: str | Path, threshold: float = 1.8) -> Path:
    """Read XYZ, keep the largest connected component, and write to dest."""
    src = Path(src)
    dest = Path(dest)
    species, coordinates, cell, pbc = read_xyz(str(src))[:4]
    species_s, coords_s = sanitize_species_coordinates(
        species, coordinates, threshold=threshold)
    write_xyz(species_s, coords_s, dest, cell=cell)
    return dest
