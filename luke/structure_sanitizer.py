"""
structure_sanitizer.py: Processes molecular structures to ensure chemical viability.

This script takes input from the `isolator.py` script and produces chemically viable substructures.
"""

import numpy as np
from scipy.spatial import distance_matrix
from .io_utils import read_xyz, write_xyz
import rdkit.Chem
from rdkit.Chem import rdmolops
import argparse


def compute_connectivity(coordinates, threshold=1.8):
    """Compute connectivity matrix based on distance threshold."""
    dist_matrix = distance_matrix(coordinates, coordinates)
    return (dist_matrix < threshold).astype(int)


def find_largest_connected_component(connectivity):
    """Find the largest connected component in the connectivity matrix."""
    graph = rdmolops.GetAdjacencyMatrix(connectivity)
    components = rdmolops.GetMolFrags(graph, asMols=False, sanitizeFrags=False)
    largest_component = max(components, key=len)
    return largest_component


def process_molecule(file_path):
    """Process a molecule to ensure chemical viability."""
    coordinates, elements = read_xyz(file_path)
    connectivity = compute_connectivity(coordinates)
    largest_component = find_largest_connected_component(connectivity)
    # Filter coordinates and elements based on the largest component
    filtered_coordinates = [coordinates[i] for i in largest_component]
    filtered_elements = [elements[i] for i in largest_component]
    return filtered_coordinates, filtered_elements


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize molecular structures.")
    parser.add_argument("--input", required=True,
                        help="Path to the input XYZ file.")
    parser.add_argument("--output", required=True,
                        help="Path to the output XYZ file.")
    args = parser.parse_args()

    print("Processing molecule...")
    coordinates, elements = process_molecule(args.input)
    write_xyz(args.output, coordinates, elements)
    print("Sanitization complete. Output saved to", args.output)


if __name__ == "__main__":
    main()
