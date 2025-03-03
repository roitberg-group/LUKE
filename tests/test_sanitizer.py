from ..io import read_xyz, write_xyz
from ..structure_sanitizer import compute_connectivity, find_largest_connected_component

# Example usage originally written in `structure_sanitizer.py`
input_xyz = "../example_structures/test_mol.xyz"  # Replace with updated `test_mol.xyz`
output_xyz = "test_outputs/test_mol_isolated.xyz"

atoms, coordinates = read_xyz(input_xyz)
connectivity = compute_connectivity(coordinates)
largest_component = find_largest_connected_component(connectivity)

# Filter atoms and coordinates to keep only those in the largest connected component
filtered_atoms = [atom for i, atom in enumerate(atoms) if i in largest_component]
filtered_coord = coordinates[list(largest_component)]

write_xyz(filtered_atoms, filtered_coord, output_xyz)
