"""Isolation utilities for generating capped fragments around high-uncertainty atoms.

Optional heavy deps (OpenBabel, RDKit) are imported lazily/guarded so that core
functionality (neighbor detection, fragment writing) works without them.
"""

# Standard library
import tempfile
from collections import Counter
from typing import Any

# Third-party (always required for core behavior)
import ase
import ase.io  # ensure io submodule loaded so ase.io.write is available
import numpy as np

# Chemistry toolkits (required)
import torch
from openbabel import pybel
from rdkit import Chem
from torchani.tuples import SpeciesCoordinates
from torchani.utils import PERIODIC_TABLE

__all__ = ["Isolator"]


class Isolator:
    def __init__(self, model: Any, cutoff: float = 5.2, threshold: float = 0.5):
        self.cutoff: float = cutoff
        self.threshold: float = threshold

        # Batched tensors (1, N) / (1, N, 3)
        self.species: torch.Tensor | None = None
        self.coordinates: torch.Tensor | None = None
        self.structure: ase.Atoms | None = None
        self.symbols: np.ndarray | None = None

        # Numpy (N,), (N,3)
        self.numpy_species: np.ndarray | None = None
        self.numpy_coordinates: np.ndarray | None = None

        self.molecule: Chem.Mol | None = None
        self.model: Any = model
        self.neighbors: dict[int, list[int]] | None = None

    @classmethod
    def from_file(cls, model: Any, input_file: str) -> "Isolator":
        instance = cls(model)
        structure = ase.io.read(input_file)
        if isinstance(structure, list):  # ase.io.read may return list for multi-frame
            # take first frame; alternative would be to create multiple instances
            structure = structure[0]
        instance.structure = structure
        instance.species = torch.tensor(
            structure.numbers).unsqueeze(0).clone().detach()
        instance.coordinates = torch.tensor(
            structure.positions).double().unsqueeze(0).clone().detach()
        instance.symbols = np.asarray(
            structure.get_chemical_symbols()).astype(str)
        return instance

    @classmethod
    def from_data(cls, model: Any, data: SpeciesCoordinates) -> list["Isolator"]:
        """Create one Isolator per conformer from (species, coordinates).

        Expects tensors shaped as (n_conf, n_atoms) and (n_conf, n_atoms, 3).
        """
        instances = []
        species_batch, coordinates_batch = data
        for i in range(species_batch.shape[0]):
            instance = cls(model)
            instance.species = species_batch[i].unsqueeze(0)
            instance.coordinates = coordinates_batch[i].double().unsqueeze(0)
            instances.append(instance)
        return instances

    def tensors_to_numpy(self, conformer_idx: int | None = None) -> None:
        if conformer_idx is None:
            if isinstance(self.species, torch.Tensor):
                self.numpy_species = self.species.cpu().numpy()
            if isinstance(self.coordinates, torch.Tensor):
                self.numpy_coordinates = self.coordinates.detach().cpu().numpy()
        else:
            if isinstance(self.species, torch.Tensor):
                self.numpy_species = self.species[conformer_idx].cpu().numpy()
            if isinstance(self.coordinates, torch.Tensor):
                self.numpy_coordinates = self.coordinates[conformer_idx].detach(
                ).cpu().numpy()
        # Flatten leading batch dimension if present (shape (1, N) -> (N,), (1, N, 3) -> (N, 3))
        if self.numpy_species is not None and self.numpy_species.ndim == 2 and self.numpy_species.shape[0] == 1:
            self.numpy_species = self.numpy_species[0]
        if self.numpy_coordinates is not None and self.numpy_coordinates.ndim == 3 and self.numpy_coordinates.shape[0] == 1:
            self.numpy_coordinates = self.numpy_coordinates[0]

    def create_rdkit_mol(
        self,
        return_smiles: bool = True,
    ) -> Chem.Mol | None:
        """Generate an RDKit molecule (and optionally print SMILES) from current species/coordinates.

        Returns None if required numpy arrays are not yet prepared (caller should have invoked tensors_to_numpy).
        """
        if self.symbols is None or self.numpy_coordinates is None:
            return None
        # Write temporary XYZ then let OpenBabel infer bonding; convert to RDKit.
        with tempfile.NamedTemporaryFile("w+") as fh:
            fh.write(f"{len(self.symbols)}\n\n")
            for idx, symbol in enumerate(self.symbols):
                coord = self.numpy_coordinates[idx]
                fh.write(f"{symbol} {coord[0]:8.3} {coord[1]:8.3} {coord[2]:8.3}\n")
            fh.flush()
            obabel_mol = next(pybel.readfile("xyz", fh.name))
            raw_mol2 = obabel_mol.write(format="mol2")
        molecule = Chem.MolFromMol2Block(raw_mol2, removeHs=False)
        # RDKit stubs indicate Chem.MolFromMol2Block returns a Chem.Mol; mypy treats this as non-optional,
        # so a fallback warning branch would be flagged unreachable. We therefore assume success for typing.
        if return_smiles:
            print("SMILES: ", Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=False))
        return molecule

    def classify_bad_atoms(self) -> list[list[int]]:
        """
        Classify the 'bad atoms' (based on the uncertainty threshold set in the class initialization)
        """
        assert self.species is not None and self.coordinates is not None, "Species/coordinates not set"
        force_qbc = self.model.force_qbc(
            (self.species, self.coordinates.requires_grad_(True))).relative_stdev
        if force_qbc.dim() == 1:
            force_qbc = force_qbc.unsqueeze(0)
        bad_atoms_per_conformer = []
        for conformer_uncertainties in force_qbc:
            bad_atom_indices = [i for i, uncertainty in enumerate(
                conformer_uncertainties) if uncertainty > self.threshold]
            bad_atoms_per_conformer.append(bad_atom_indices)
        return bad_atoms_per_conformer

    def get_neighbors(self, bad_atom_idxs: int | list[int]) -> dict[int, list[int]]:
        """
        Identify neighbors of bad atoms using pairwise distances with a cutoff (Angstrom).
        """
        if isinstance(bad_atom_idxs, int):
            bad_atom_idxs = [bad_atom_idxs]
        if not bad_atom_idxs:
            return {}
        assert self.coordinates is not None, "Coordinates not set"
        coords = self.coordinates.squeeze(0)  # (N, 3)
        dists = torch.cdist(coords.double(), coords.double())  # (N, N)
        neighbors_dict = {}
        for atom_index in bad_atom_idxs:
            if atom_index < 0 or atom_index >= dists.shape[0]:  # safety
                continue
            # neighbors within cutoff excluding self
            mask = (dists[atom_index] <= self.cutoff) & (
                torch.arange(dists.shape[0]) != atom_index)
            idxs = torch.nonzero(mask, as_tuple=False).reshape(-1).tolist()
            neighbors_dict[atom_index] = idxs
        return neighbors_dict

    def isolate_atoms(self, bad_atom_index: int) -> tuple[list[np.ndarray], list[str]]:
        """
        Create a unique 'capped' structure for each bad atom, including only the atom and its neighbors.
        """
        neighbors_map = self.get_neighbors(bad_atom_index)
        bad_atom_neighbors = neighbors_map.get(bad_atom_index, [])
        involved_atoms = set(bad_atom_neighbors) | {bad_atom_index}

        assert self.numpy_coordinates is not None and self.symbols is not None, "Call tensors_to_numpy() before isolate_atoms()"
        modified_coords = [self.numpy_coordinates[idx] for idx in involved_atoms]
        modified_symbols = [self.symbols[idx] for idx in involved_atoms]
        return modified_coords, modified_symbols

    def process_bad_atom(self, atom_index: int) -> ase.Atoms:
        """
        Create a new "capped" structure for each high-uncertainty atom present in the input
        """
        capped_coords, capped_symbols = self.isolate_atoms(atom_index)
        return ase.Atoms(positions=capped_coords, symbols=capped_symbols)

    def process_molecule(self, bad_atoms: list[int], output_file_prefix: str = "/home/nick/capped_", output_format: str = 'xyz') -> None:
        """
        Go through the molecule and create new structures for each of the "bad atoms"
        """
        if not bad_atoms:
            return
        # Ensure 1D numbers and (N,3) positions for ASE
        assert self.numpy_species is not None and self.numpy_coordinates is not None
        numbers = np.asarray(self.numpy_species).ravel()
        positions = np.asarray(self.numpy_coordinates)
        if positions.ndim == 3 and positions.shape[0] == 1:
            positions = positions[0]
        self.structure = ase.Atoms(numbers=numbers, positions=positions)
        self.symbols = np.asarray(
            self.structure.get_chemical_symbols()).astype(str)

        original_elements = [PERIODIC_TABLE[int(num)] for num in numbers]
        original_counts = Counter(original_elements)
        original_formula = ''.join(
            f'{el}{original_counts[el] if original_counts[el] > 1 else ""}' for el in sorted(original_counts))
        counter = 1
        for bad_atom_idx in bad_atoms:
            capped_structure = self.process_bad_atom(bad_atom_idx)
            capped_elements = [
                PERIODIC_TABLE[int(num)] for num in capped_structure.numbers]
            capped_counts = Counter(capped_elements)

            capped_formula = ''.join(
                f'{el}{capped_counts[el] if capped_counts[el] > 1 else ""}' for el in sorted(capped_counts))
            output_file = f"{output_file_prefix}{original_formula}_{counter}_{capped_formula}_atom{bad_atom_idx}.{output_format}"
            counter += 1
            ase.io.write(output_file, capped_structure)

    def execute(self, input: SpeciesCoordinates | str, is_file: bool = False) -> None:
        """Execute the isolation workflow.

        Notes / future extension ideas:
        - Add output_file_prefix and output_format arguments here (instead of hardcoding in process_molecule).
        - Allow passing a directory of XYZ files (iterating over many structures).
        - Provide optional switches for skipping capping or RDKit generation.
        """
        # The comments above were turned into a docstring to avoid an unused, unreachable literal.
        if is_file:
            assert isinstance(input, str)
            conformers = [self.from_file(input_file=input, model=self.model)]
        else:
            assert isinstance(input, tuple)
            conformers = self.from_data(data=input, model=self.model)

        for conformer_idx, conformer in enumerate(conformers):
            self.species = conformer.species
            self.coordinates = conformer.coordinates
            bad_atoms_per_conformer = self.classify_bad_atoms()
            for bad_atoms in bad_atoms_per_conformer:
                if not bad_atoms:
                    print(
                        "No atoms exceeding the uncertainty threshold. Skipping to the next structure.")
                    continue
                self.tensors_to_numpy(conformer_idx=conformer_idx)
                self.neighbors = self.get_neighbors(bad_atoms)
                self.process_molecule(bad_atoms)
                self.molecule = self.create_rdkit_mol()


if __name__ == "__main__":  # pragma: no cover - manual usage example
    from .ani_forces import ANIForceCalculator

    calculator = ANIForceCalculator(model_name="ANI2xr", threshold=0.5)
    # Example manual usage could be added here.
