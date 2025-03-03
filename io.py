# This module contains functions for input/output operations, used in the first and last stages of LUKE

import os
from pathlib import Path
import typing as tp
import shlex
import hashlib

import torch
from torch import Tensor
from torchani.constants import ATOMIC_NUMBER, PERIODIC_TABLE
from torchani.utils import pad_atomic_properties

__all__ = ["read_xyz", "write_xyz", "hash_xyz_coordinates", "write_gaussian_input", "write_slurm"]

class IOErrorLUKE(Exception):
    pass

def read_xyz(
    path: tp.Union[str, Path],
    dtype: tp.Optional[torch.dtype] = None,
    device: tp.Union[torch.device, str, int, None] = None,
    detect_padding: bool = True,
    pad_species_value: int = 100,
    dividing_char: str = ">",
    return_comments: bool = False,
) -> tp.Tuple[Tensor, Tensor, tp.Optional[Tensor], tp.Optional[Tensor]]:
    """
    Read an (extended) XYZ file and return atomic species, coordinates, and (optionally) cell data.

    This function parses the specified XYZ file and gathers:
      - `species`: A tensor holding the atomic numbers for each conformation
      - `coordinates`: A tensor of (x, y, z) coordinates for each atom
      - `cell`: An optional (3 x 3) tensor with lattice vectors, if present
      - `pbc`: An optional boolean tensor of shape (3,) indicating periodic boundary conditions
      - (optionally) a list of comment lines (if `return_comments` is True)

    Multiple “conformations” (i.e., separate frames) in the file are identified via:
      1. An integer line specifying the number of atoms.
      2. A comment line that may contain “Lattice=...”.

    Parameters
    ----------
    path : StrPath
        The path to the XYZ file to read.
    dtype : DType, optional
        The PyTorch data type for the returned tensors. If not specified, defaults are used.
    device : Device, optional
        The PyTorch device for the returned tensors (e.g., CPU or CUDA).
    detect_padding : bool, default=True
        If True, any atom whose species matches `pad_species_value` is converted to -1 and its coordinates
         to (0.0, 0.0, 0.0).
    pad_species_value : int, default=100
        The atomic number used to detect padded (placeholder) atoms when `detect_padding` is True.
    dividing_char : str, default=">"
        A character or line marker that, if encountered on its own line, indicates a division to skip. Helpful for
         ignoring certain separators in multi-structure files.
    return_comments : bool, default=False
        If True, returns an additional list of comment strings (e.g., the lines following the atom count for each
         conformation).

    Returns
    -------
    species : Tensor
        A 2D tensor of shape (n_conformations, n_max_atoms) containing atomic numbers (or -1 for padded atoms if
        `detect_padding` is True).
    coordinates : Tensor
        A 3D tensor of shape (n_conformations, n_max_atoms, 3) containing corresponding (x, y, z) coordinates.
    cell : Tensor or None
        A (3 x 3) lattice tensor if the file includes lattice info; otherwise None.
    pbc : Tensor or None
        A 1D tensor of shape (3,) indicating periodic boundary conditions if `cell` was present; otherwise None.
    comments_list : list of str
        Only returned if `return_comments` is True, capturing all comment lines from the file (one per conformation).

    Raises
    ------
    IOErrorLUKE
        If a cell is found in one conformation but not in the first, or if multiple conformations specify different
         lattice vectors.
    ValueError
        If the file format is invalid or if line parsing fails.

    Notes
    -----
    - The function uses `ATOMIC_NUMBER` to map element symbols to atomic numbers. If a symbol is not found, it is
      parsed as an integer for the atomic number.
    - Conformations are stored in the order encountered.
    - The `pad_atomic_properties` function is called internally to ensure
      tensors for all conformations are padded to the same shape.
    - If no lattice information is detected in any comment line, both `cell`
      and `pbc` are None.

    """
    path = Path(path).resolve()
    cell: tp.Optional[Tensor] = None
    properties: tp.List[tp.Dict[str, Tensor]] = []
    comments_list: tp.List[str] = []
    with open(path, mode="rt", encoding="utf-8") as f:
        lines = iter(f)
        conformation_num = 0
        while True:
            species = []
            coordinates = []
            try:
                molec_num_str = next(lines)
            except StopIteration:
                break
            if dividing_char and molec_num_str.strip() == dividing_char:
                continue
            num = int(molec_num_str)
            comment = next(lines)
            if return_comments:
                comments_list.append(comment)
            if "lattice" in comment.lower():
                if (cell is None) and (conformation_num != 0):
                    raise IOErrorLUKE("If cell is present it should be in the first conformation")
                parts = shlex.split(comment)
                for part in parts:
                    key, value = part.split("=")
                    if key.lower() == "lattice":
                        # cell order is x0 y0 z0 x1 y1 z1 x2 y2 z2 for the 3 unit vectors
                        conformation_cell = torch.tensor(
                            [float(s) for s in value.split()],
                            dtype=dtype,
                            device=device,
                        ).view(3, 3)
                        if cell is None:
                            cell = conformation_cell
                        elif not (cell == conformation_cell).all():
                            raise IOErrorLUKE("Found two conformations with non-matching cells")
            for _ in range(num):
                line = next(lines)
                s, x, y, z = line.split()
                if s in ATOMIC_NUMBER:
                    atomic_num = ATOMIC_NUMBER[s]
                else:
                    atomic_num = int(s)
                if atomic_num == pad_species_value and detect_padding:
                    atomic_num = -1
                    x, y, z = "0.0", "0.0", "0.0"
                species.append(atomic_num)
                coordinates.append([float(x), float(y), float(z)])
            conformation_num += 1
            properties.append(
                {
                    "coordinates": torch.tensor(
                        [coordinates],
                        dtype=dtype,
                        device=device,
                    ),
                    "species": torch.tensor(
                        [species],
                        dtype=torch.long,
                        device=device,
                    ),
                }
            )
    pad_properties = pad_atomic_properties(properties)
    pbc = torch.tensor([True, True, True], device=device) if cell is not None else None
    if return_comments:
        return pad_properties["species"], pad_properties["coordinates"], cell, pbc, comments_list  # type: ignore  # noqa: E501
    return pad_properties["species"], pad_properties["coordinates"], cell, pbc


def write_xyz(
    species: Tensor,
    coordinates: Tensor,
    dest: tp.Union[str, Path],
    cell: tp.Optional[Tensor] = None,
    pad: bool = False,
    pad_coord_value: float = 0.0,
    pad_species_value: int = 100,
) -> None:
    """
    Write molecular coordinates and species data to an extended XYZ file.

    This function expects:
      - A 2D tensor of atomic species (atomic numbers) of shape (M, N), where M is the number of molecule entries and
        N is the maximum number of atoms per molecule (or placeholder entries).
      - A 3D tensor of coordinates of shape (M, N, 3).
      - Optionally, a (3 x 3) tensor for the cell to include lattice information in the extended XYZ header.

    Parameters
    ----------
    species : Tensor
        2D tensor of shape (M, N), containing atomic numbers.
        A value of -1 indicates a placeholder (ignored or replaced, depending on `pad`).
    coordinates : Tensor
        A 3D tensor of shape (M, N, 3) containing the corresponding (x, y, z) coordinates for each atom in `species`.
    dest : StrPath
        File path to write the XYZ output.
    cell : Optional[Tensor], default=None
        A (3 x 3) tensor representing the lattice vectors. If provided, lattice information is written into the
         extended XYZ header.
    pad : bool, default=False
        If False, atoms with `species == -1` are omitted from the output.
        If True, atoms with `species == -1` are replaced by `pad_species_value` and their coordinates replaced by
         `pad_coord_value`.
    pad_coord_value : float, default=0.0
        The coordinate value used when padding atoms. Only relevant if `pad` is True.
    pad_species_value : int, default=100
        The atomic number used to replace a -1 placeholder if `pad` is True.

    Raises
    ------
    ValueError
        If `species` is not 2D or `coordinates` does not have the matching (M, N, 3) shape, or if `cell` is provided
         but not of shape (3, 3). Also raised if padding is requested but the atomic number used for padding
         (`pad_species_value`) already appears in the data.

    Notes
    -----
    - The output is written using the extended XYZ format, including a Properties line that details atom species
      and position.
    - If `cell` is not None, the line `Lattice="..." pbc="T T T"` is included, otherwise `pbc="F F F"` is used.
    - The periodic table mapping is assumed to be available as `PERIODIC_TABLE`, which converts atomic numbers to
      element symbols.

    """
    dest = Path(dest).resolve()
    # Input validation
    if species.dim() != 2:
        raise ValueError("Species should be a 2 dim tensor")
    if coordinates.shape != (species.shape[0], species.shape[1], 3):
        raise ValueError("Coordinates should have shape (molecules, atoms, 3)")
    if cell is not None and cell.shape != (3, 3):
        raise ValueError("Cell should be a tensor of shape (3, 3)")

    with open(dest, mode="wt", encoding="utf-8") as f:
        for j, (znums, coords) in enumerate(zip(species, coordinates)):
            if not pad:
                mask = znums != -1
                coords = coords[mask]
                znums = znums[mask]
            else:
                if (znums == pad_species_value).any():
                    raise ValueError(
                        "Can't pad if there are elements with atomic number 100"
                    )
                mask = znums == -1
                znums[mask] = pad_species_value
                coords[mask] = pad_coord_value
            f.write(f"{len(coords)}\n")
            props = "species:S:1:pos:R:3"
            if cell is not None:
                cell_elements = " ".join(
                    [(f"{e:.10f}" if e != 0.0 else "0.0") for e in cell.view(-1)]
                )
                f.write(f'Lattice="{cell_elements}" Properties={props} pbc="T T T"\n')
            else:
                f.write(f'Properties={props} pbc="F F F"\n')
            for z, atom in zip(znums, coords):
                symbol = PERIODIC_TABLE[z]
                f.write(f"{symbol} {atom[0]:.10f} {atom[1]:.10f} {atom[2]:.10f}\n")


def hash_xyz_coordinates(filepath):
    """Generate MD5 hash for the coordinates in an XYZ file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()[2:-1]  # Skip the first two and the last line -- fix depending on how xyz files are written
            for line in lines:
                hasher.update(line.encode('utf-8'))
        return hasher.hexdigest()
    except IOErrorLUKE as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def remove_duplicate_xyz_files(file_path):
    seen_hashes = set()
    duplicate_count = 0

    for filename in os.listdir(file_path):
        if filename.endswith(".xyz"):
            filepath = os.path.join(file_path, filename)
            file_hash = hash_xyz_coordinates(filepath)
            if file_hash is None:
                continue

            if file_hash in seen_hashes:
                os.remove(filepath)
                duplicate_count += 1
                print(f"Deleted duplicate file: {filename}")
            else:
                seen_hashes.add(file_hash)

    print(f"Total duplicate files deleted: {duplicate_count}")

def write_gaussian_input(symbols, coordinates, file_name, theory='B3LYP', basis_set='6-31G(d)'):
    # TO DO:
        # Input should be species, convert to symbols
        # ???
    header = f"%chk={file_name}.chk\n# {theory}/{basis_set} SP\n\nTitle Card Required\n\n0 1\n"
    molecule_data = "\n".join([f"{symbol} {' '.join(map(str, coord))}" for symbol, coord in zip(symbols, coordinates)])
    footer = "\n\n"

    with open(f"{file_name}.com", "w") as f:
        f.write(header + molecule_data + footer)

# Example usage
# create_gaussian_com_file(['H', 'O', 'H'], [[0, 0, 0], [0, 0, 1], [0, 1, 0]], "water_molecule")

def write_slurm(com_file, script_name):
    script_content = f"""
    #!/bin/bash
    #SBATCH --job-name={com_file}
    #SBATCH --output={com_file}.out
    #SBATCH --error={com_file}.err
    #SBATCH --time=01:00:00
    #SBATCH --partition=your_partition
    #SBATCH --mem=4GB

    module load gaussian
    g09 < {com_file}.com > {com_file}.log
    """
    with open(f"{script_name}.sh", "w") as f:
        f.write(script_content)

# Example usage
# create_slurm_script("water_molecule", "run_water_molecule")


if __name__ == "__main__":
    # Fix this with the addition of an argparser
    directory = input("Enter the directory path containing XYZ files: ").strip()
    remove_duplicate_xyz_files(directory)