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
    assert output_content.splitlines()[0].strip() == "2"
    assert "H 0.0 0.0 0.0" in output_content
    assert "H 1.0 0.0 0.0" in output_content
