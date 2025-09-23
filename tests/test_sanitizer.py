from luke.structure_sanitizer import compute_connectivity, find_largest_connected_component, process_molecule
from luke.io_utils import read_xyz, write_xyz
import pytest

# Example usage originally written in `structure_sanitizer.py`
# Replace with updated `test_mol.xyz`
input_xyz = "../example_structures/test_mol.xyz"
output_xyz = "test_outputs/test_mol_isolated.xyz"

atoms, coordinates = read_xyz(input_xyz)
connectivity = compute_connectivity(coordinates)
largest_component = find_largest_connected_component(connectivity)

# Filter atoms and coordinates to keep only those in the largest connected component
filtered_atoms = [atom for i, atom in enumerate(
    atoms) if i in largest_component]
filtered_coord = coordinates[list(largest_component)]

write_xyz(filtered_atoms, filtered_coord, output_xyz)


def test_compute_connectivity():
    coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    connectivity = compute_connectivity(coordinates, threshold=1.5)
    expected = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    assert (connectivity == expected).all()


def test_find_largest_connected_component():
    connectivity = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    largest_component = find_largest_connected_component(connectivity)
    assert largest_component == {0, 1}


def test_process_molecule(tmp_path):
    input_xyz = tmp_path / "test_input.xyz"
    output_xyz = tmp_path / "test_output.xyz"

    # Create a temporary XYZ file
    input_xyz.write_text(
        """
        3
        Comment line
        H 0.0 0.0 0.0
        H 1.0 0.0 0.0
        O 3.0 0.0 0.0
        """
    )

    process_molecule(input_xyz, output_xyz)

    # Verify the output
    output_content = output_xyz.read_text()
    assert "2\n" in output_content  # Only 2 atoms should remain
    assert "H 0.0 0.0 0.0" in output_content
    assert "H 1.0 0.0 0.0" in output_content
