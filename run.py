"""
run.py: Main script to execute the LUKE pipeline.

This script integrates the various modules in the LUKE project to provide a seamless pipeline for molecular fragmentation and active learning.

Usage:
    python run.py --input <input_file> --output <output_directory>

Arguments:
    --input: Path to the input file (HDF5 or XYZ format).
    --output: Path to the output directory where results will be saved.
"""

import argparse

from luke.ani_forces import run_ani_forces
from luke.io_utils import read_input, write_output
from luke.isolator import isolate_high_error_atoms
from luke.structure_sanitizer import sanitize_structures


def main():
    parser = argparse.ArgumentParser(description="Run the LUKE pipeline.")
    parser.add_argument("--input", required=True,
                        help="Path to the input file (HDF5 or XYZ format).")
    parser.add_argument("--output", required=True,
                        help="Path to the output directory.")
    args = parser.parse_args()

    # Step 1: Read input file
    print("Reading input file...")
    input_data = read_input(args.input)

    # Step 2: Run ANI forces
    print("Running ANI forces...")
    ani_results = run_ani_forces(input_data)

    # Step 3: Isolate high-error atoms
    print("Isolating high-error atoms...")
    isolated_structures = isolate_high_error_atoms(ani_results)

    # Step 4: Sanitize structures
    print("Sanitizing structures...")
    sanitized_structures = sanitize_structures(isolated_structures)

    # Step 5: Write output
    print("Writing output...")
    write_output(sanitized_structures, args.output)

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
