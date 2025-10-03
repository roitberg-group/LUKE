from luke.structure_sanitizer import (
    compute_connectivity,
    largest_component_indices,
    sanitize_xyz_file,
)


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


def test_largest_component_indices():
    connectivity = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ]
    largest = largest_component_indices(connectivity)
    assert largest == [0, 1]


def test_sanitize_xyz_file(tmp_path):
    input_xyz = tmp_path / "test_input.xyz"
    output_xyz = tmp_path / "test_output.xyz"

    input_xyz.write_text(
        """3
Comment line
H 0.0 0.0 0.0
H 1.0 0.0 0.0
O 3.0 0.0 0.0
"""
    )

    sanitize_xyz_file(input_xyz, output_xyz, threshold=1.6)
    output_content = output_xyz.read_text()
    lines = output_content.splitlines()
    assert lines[0].strip() == "2"
    # Parse the two atom lines (skip header line index 1)
    atom_lines = lines[2:4]
    elems = []
    coords = []
    for line in atom_lines:
        parts = line.split()
        elems.append(parts[0])
        coords.append(tuple(float(x) for x in parts[1:4]))
    assert elems == ["H", "H"]
    # Distinguish they are ~1.0 Å apart
    import math
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    dz = coords[1][2] - coords[0][2]
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    assert abs(dist - 1.0) < 1e-6
