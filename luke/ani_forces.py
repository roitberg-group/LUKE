import argparse
from pathlib import Path
import typing as tp

import numpy as np
import torch
import torchani
from torchani.datasets import ANIDataset
from torchani.units import hartree2kcalpermol
from tqdm import tqdm
from luke.io_utils import read_xyz
import pandas as pd

# Fix the threshold -- 0.5 was determined from kcal/mol*A data


class ANIForceCalculator:
    """Compute ANI forces and uncertainty metrics for molecular structures."""

    def __init__(self, model_name: str = "ANI2xr", device: tp.Optional[str] = None, threshold: float = 0.5):
        """Initialize ANI model and device."""
        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cpu"  # Force CPU if CUDA is unavailable
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = getattr(torchani.models, model_name)()
        self.model.to(self.device)
        self.model.set_enabled("energy_shifter", False)
        self.threshold = threshold

    def process_structure(
        self, species: torch.Tensor, coordinates: torch.Tensor
    ) -> tp.Optional[tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, float]]:
        """
        Compute ANI energies, force uncertainty, and flag outlier atoms for a single structure.
        Returns None if weighted_stdev ≤ 3.5.
        """
        self.model.set_enabled("energy_shifter", True)
        ani_input = (species.to(self.device), coordinates.to(self.device))

        _, energies, qbc = self.model.energies_qbcs(ani_input)
        energy_mean = hartree2kcalpermol(energies.detach().cpu().item())
        energy_qbc = hartree2kcalpermol(qbc.detach().cpu().item())

        force_qbc = self.model.force_qbc(ani_input, ensemble_values=True)
        magnitudes = hartree2kcalpermol(
            force_qbc.magnitudes.detach().cpu())  # (ensemble, N)
        weights = magnitudes.sum(dim=0)
        weighted_stdev = torch.sqrt(
            (weights * (magnitudes.std(dim=0) ** 2)).sum() / weights.sum()).item()

        if weighted_stdev <= 3.5:
            return None

        mean_magnitudes = magnitudes.mean(dim=0)  # (N,)
        deviations = torch.abs(magnitudes - mean_magnitudes)  # (ensemble, N)
        max_deviation_per_atom = deviations.max(dim=0).values  # (N,)
        normalized_max_deviation = max_deviation_per_atom / \
            mean_magnitudes.clamp(min=1e-8)
        good_or_bad = (normalized_max_deviation > self.threshold).int()

        return species.cpu(), coordinates.cpu(), good_or_bad, energy_mean, energy_qbc, weighted_stdev

    def process_dataset(
        self, dataset_path: str, batch_size: int = 2500, include_energy: bool = True
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process an H5 or XYZ dataset and batch valid structures/results."""
        dataset_path = Path(dataset_path)
        species_list: tp.List[torch.Tensor] = []
        coordinates_list: tp.List[torch.Tensor] = []
        good_or_bad_list: tp.List[torch.Tensor] = []
        energy_list: tp.List[float] = []
        qbc_list: tp.List[float] = []
        weighted_stdev_list: tp.List[float] = []

        if dataset_path.suffix == ".h5":
            ds = ANIDataset(str(dataset_path))
            with ds.keep_open("r") as read_ds:
                total_chunks = read_ds.num_chunks(max_size=batch_size)
                for _, _, conformer in tqdm(
                    read_ds.chunked_items(max_size=batch_size), total=total_chunks, desc="Processing dataset"
                ):
                    species_batch = conformer["species"]
                    coordinates_batch = conformer["coordinates"]

                    for i in range(species_batch.shape[0]):
                        species_tensor = species_batch[i].unsqueeze(0)
                        coordinates_tensor = coordinates_batch[i].unsqueeze(0)
                        result = self.process_structure(
                            species_tensor, coordinates_tensor)
                        if result is not None:
                            s, c, g, e, q, w = result
                            species_list.append(s)
                            coordinates_list.append(c)
                            good_or_bad_list.append(g)
                            energy_list.append(e)
                            qbc_list.append(q)
                            weighted_stdev_list.append(w)

        elif dataset_path.suffix == ".xyz":
            species, coordinates, _, _ = read_xyz(str(dataset_path))[:4]
            result = self.process_structure(species, coordinates)
            if result is not None:
                s, c, g, e, q, w = result
                species_list.append(s)
                coordinates_list.append(c)
                good_or_bad_list.append(g)
                energy_list.append(e)
                qbc_list.append(q)
                weighted_stdev_list.append(w)

        if not species_list:
            return (
                torch.empty(0),
                torch.empty(0, 3),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
                torch.empty(0),
            )

        species_tensor = torch.stack(species_list)
        coordinates_tensor = torch.stack(coordinates_list)
        good_or_bad_tensor = torch.stack(good_or_bad_list)
        energy_tensor = torch.tensor(energy_list)
        qbc_tensor = torch.tensor(qbc_list)
        weighted_stdev_tensor = torch.tensor(weighted_stdev_list)

        return (
            species_tensor,
            coordinates_tensor,
            good_or_bad_tensor,
            energy_tensor,
            qbc_tensor,
            weighted_stdev_tensor,
        )


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute ANI energy, forces, and uncertainty.")
    parser.add_argument("dataset", type=str,
                        help="Path to the input H5 dataset or xyz file.")
    parser.add_argument("--model", type=str, default="ANI2xr",
                        help="ANI model to use (default: ANI2xr).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run calculations on (default: CUDA if available).")
    parser.add_argument("--batch_size", type=int, default=2500,
                        help="Batch size for dataset processing.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for marking atoms as 'bad' based on max deviation (default: 0.5).")
    return parser.parse_args()


def main():
    """Main execution function when running as a script."""
    args = parse_args()

    # Initialize the force calculator
    calculator = ANIForceCalculator(
        model_name=args.model, device=args.device, threshold=args.threshold)

    # Create an empty DataFrame to store bad molecules
    df_bad_molecules = pd.DataFrame(
        columns=["species", "coordinates", "good_or_bad", "energy", "energy_qbc"])

    # Process dataset and collect bad molecules
    results = calculator.process_dataset(
        args.dataset, batch_size=args.batch_size)
    species_batch, coordinates_batch, good_or_bad_batch, energy_batch, qbc_batch, stdev_batch = results

    # Iterate over molecules in batch
    for i in range(species_batch.shape[0]):  # Loop over each molecule
        species = species_batch[i]
        coordinates = coordinates_batch[i]
        good_or_bad = good_or_bad_batch[i]
        energy = energy_batch[i]
        qbc = qbc_batch[i]
        weighted_stdev = stdev_batch[i]

        # Check if the molecule is "bad" (i.e., has at least one bad atom)
        if good_or_bad.sum() > 0:
            df_bad_molecules = pd.concat([df_bad_molecules, pd.DataFrame({
                "species": [species.tolist()],
                "coordinates": [coordinates.tolist()],
                "good_or_bad": [good_or_bad.tolist()],
                "energy": [energy.item()],
                "energy_qbc": [qbc.item()],
                "weighted_stdev": [weighted_stdev.item()]
            })], ignore_index=True)

    # Save DataFrame if bad molecules are found
    if not df_bad_molecules.empty:
        output_file = Path(args.dataset).stem + "_bad_molecules.pq"
        df_bad_molecules.to_parquet(output_file, index=False)
        print(
            f"Processed {len(df_bad_molecules)} bad molecules. Saved results to {output_file}.")
    else:
        print("No bad molecules found.")


if __name__ == "__main__":
    main()
