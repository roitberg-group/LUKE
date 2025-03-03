import argparse
from pathlib import Path
import typing as tp
import torch
import torchani
from torchani.datasets import ANIDataset
from tqdm import tqdm

# Write this script to:
# - compute the weighted standard deviation for each input molecule
# - find the outlier atoms in terms of max_deviation ([max(F)-mean(F)]/mean(F))
# - return the input structure with a new dim in the tensor with [0,1] to say whether an atom is [good, bad]

class ANIForceCalculator:
    """Compute ANI forces and uncertainty metrics for molecular structures."""

    def __init__(self, model_name: str = "ANI2xr", device: str = "cuda" if torch.cuda.is_available() else "cpu", threshold: float = 0.5):
        """
        Initializes the ANI model.

        Parameters
        ----------
        model_name : str
            Name of the ANI model to load.
        device : str
            Device to run computations on (default: "cuda" if available, else "cpu").
        threshold : float
            Threshold for identifying bad atoms based on max deviation (default: 0.5).
        """
        self.device = torch.device(device)
        self.model = getattr(torchani.models, model_name)().to(self.device)
        self.model.set_enabled('energy_shifter', False)

    def process_structure(self, species: torch.Tensor, coordinates: torch.Tensor) -> tp.Optional[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Computes ANI forces, force uncertainty, and identifies outlier atoms only for structures
        where weighted_stdev exceeds 3.5.

        Parameters
        ----------
        species : torch.Tensor
            Atomic species (N_atoms,).
        coordinates : torch.Tensor
            Atomic coordinates (N_atoms, 3).

        Returns
        -------
        tp.Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            - species: Original species tensor.
            - coordinates: Original coordinate tensor.
            - good_or_bad: Tensor of shape (N_atoms,) with 0 for "good" and 1 for "bad" atoms.
            - energy: Scalar value of ANI-predicted energy for the structure.
            Returns None if weighted_stdev ≤ 3.5.
        """
        self.model.set_enabled('energy_shifter', True)
        ani_input = (species.to(self.device), coordinates.to(self.device))

        _, energies, qbc = self.model.energies_qbcs(ani_input)
        energy_mean = energies.mean().detach().cpu().item()  # Scalar energy

        magnitudes = self.model.force_qbc(ani_input, ensemble_values=True).magnitudes.detach().cpu()

        weights = magnitudes.sum(dim=0)  # Sum of force magnitudes per atom
        weighted_stdev = torch.sqrt((weights * (magnitudes.std(dim=0) ** 2)).sum() / weights.sum()).item()

        # Skip processing if weighted_stdev is below or equal to 3.5
        if weighted_stdev <= 3.5:
            return None

        force_means = magnitudes.mean(dim=0)  # Mean force per atom
        force_max = magnitudes.max(dim=0).values  # Max force per atom
        max_deviation = ((force_max - force_means) / force_means).tolist()

        good_or_bad = (max_deviation > threshold).int()  # 0 = good, 1 = bad

        return species.cpu(), coordinates.cpu(), good_or_bad

    def process_dataset(self, dataset_path: str, batch_size: int = 2500, include_energy: bool = False) -> Tuple:
        """
        Processes an ANI dataset, computing forces and identifying bad atoms only for
        structures with weighted_stdev > 3.5.

        Parameters
        ----------
        dataset_path : str
            Path to the H5 dataset file.
        batch_size : int, optional
            Number of structures processed per batch (default: 2500).
        include_energy : bool, optional
            Whether to include ANI-predicted energies in the output (default: False).

        Returns
        -------
        If include_energy is False:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - species: Batched tensor of species.
            - coordinates: Batched tensor of coordinates.
            - good_or_bad: Batched tensor of 0s and 1s (good vs. bad atoms).

        If include_energy is True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - species, coordinates, good_or_bad as above.
            - energies: Batched tensor of ANI-predicted energies.
        """
        ds = ANIDataset(dataset_path)
        species_list, coordinates_list, good_or_bad_list, energy_list = [], [], [], []

        with ds.keep_open("r") as read_ds:
            total_chunks = read_ds.num_chunks(max_size=batch_size)
            for _, _, conformer in tqdm(read_ds.chunked_items(max_size=batch_size), total=total_chunks,
                                        desc="Processing dataset"):
                species = conformer["species"]
                coordinates = conformer["coordinates"]

                # Compute ANI properties for this batch
                for i in range(species.shape[0]):
                    result = self.process_structure(species[i], coordinates[i])
                    if result is not None:
                        s, c, g, e = result
                        species_list.append(s)
                        coordinates_list.append(c)
                        good_or_bad_list.append(g)
                        energy_list.append(torch.tensor(e))  # Convert energy to tensor

        # If no structures pass the filter, return empty tensors
        if not species_list:
            if include_energy:
                return torch.empty(0), torch.empty(0, 3), torch.empty(0), torch.empty(0)
            return torch.empty(0), torch.empty(0, 3), torch.empty(0)

        # Stack tensors for batch output
        if include_energy:
            return torch.stack(species_list), torch.stack(coordinates_list), torch.stack(good_or_bad_list), torch.stack(
                energy_list)
        return torch.stack(species_list), torch.stack(coordinates_list), torch.stack(good_or_bad_list)

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute ANI energy, forces, and uncertainty.")
    parser.add_argument("dataset", type=str, help="Path to the input H5 dataset or xyz file.")
    parser.add_argument("--model", type=str, default="ANI2xr",
                        help="ANI model to use (default: ANI2xr).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run calculations on (default: CUDA if available).")
    parser.add_argument("--batch_size", type=int, default=2500,
                        help="Batch size for dataset processing.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for marking atoms as 'bad' based on max deviation (default: 0.5).")
    parser.add_argument("--include_energy", action="store_true",
                        help="Include ANI-predicted energies in the output.")
    return parser.parse_args()


def main():
    """Main execution function when running as a script."""
    args = parse_args()

    # Initialize the force calculator with the threshold
    calculator = ANIForceCalculator(model_name=args.model, device=args.device, threshold=args.threshold)

    # Process dataset
    if args.include_energy:
        species, coordinates, good_or_bad, energies = calculator.process_dataset(args.dataset, batch_size=args.batch_size, include_energy=True)
        print("Processing complete!")
        print(f"Processed {species.shape[0]} structures.")
        print(f"Example output tensor shapes: species={species.shape}, coordinates={coordinates.shape}, good_or_bad={good_or_bad.shape}, energies={energies.shape}")
    else:
        species, coordinates, good_or_bad = calculator.process_dataset(args.dataset, batch_size=args.batch_size)
        print("Processing complete!")
        print(f"Processed {species.shape[0]} structures.")
        print(f"Example output tensor shapes: species={species.shape}, coordinates={coordinates.shape}, good_or_bad={good_or_bad.shape}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
