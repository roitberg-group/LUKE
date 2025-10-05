# LUKE: USE the Forces

## NOTE: This is a pre-release, the scripts here do not fully cooperate yet. Each step in the protocol currently exists as standalone scripts, and the pipeline is under construction

## Largest Uncertainty Kaleidoscope Estimator: Uncertainty-driven Sampling of high-Error Forces

Yes, I fit the acronym to the title of the project.

LUKE: USE the Forces is a molecular fragmentation protocol designed to improve active learning in machine-learned interatomic potential models. Built on **TorchANI**, LUKE identifies atomic environments with high force uncertainty and fragments molecules around them, generating smaller molecular systems to enhance training data diversity.

## Overview

LUKE leverages TorchANI to:

- Detect high-uncertainty atomic force predictions
- Fragment molecules around high-error atoms
- Introduce new, diverse molecular structures to the training dataset
- Improve localized understanding of chemical space

## Features

- Automated high-uncertainty detection using force magnitude predictions
- Efficient molecular fragmentation guided by the TorchANI neighbor list
- Designed for active learning workflows in neural network potentials
- Seamless integration with existing TorchANI-based training pipelines

## Installation

LUKE relies on TorchANI as a git submodule (vendored source). All runtime and development
dependencies are declared in `pyproject.toml` (PEP 621). Install in editable mode with the
chemistry and development extras for full functionality.

```bash
git clone --recursive git@github.com:roitberg-group/LUKE.git
cd LUKE
git submodule update --init --recursive
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[chem,dev]
```

### Platform Note (Torch CPU wheels)
The CI pins `torch==2.3.1` (CPU build) via the PyTorch CPU index on Linux. On macOS and Windows the
`+cpu` suffix is not used—just the plain version. For a local environment that mirrors CI, use the
helper script:

```bash
bash ./dev_ci_setup.sh
```

This script:

1. Creates/updates `.venv` with Python 3.11
2. Installs pinned torch (CPU variant where available)
3. Installs editable torchani (vendored submodule) with its dependencies
4. Installs LUKE with chemistry + dev extras
5. Verifies torchani internal tuple import

## Command-line interface

After installation, a console command `luke` is available.

- Run full pipeline:

```bash
luke pipeline example_structures/test_mol.xyz -o results --model ANI2xr --device cpu --threshold 0.5
```

- Or directly via Python module:

```bash
python -m luke.cli pipeline example_structures/test_mol.xyz -o results
```

## Usage

LUKE is designed to be integrated into molecular simulation and machine learning workflows. Below is an example of how to use the pipeline:

### Example

```bash
python run.py --input example_structures/test_mol.xyz --output results/
```

This command will:

1. Read the input XYZ file.
2. Run ANI forces to detect high-uncertainty atomic environments.
3. Fragment molecules around high-error atoms.
4. Sanitize the resulting structures.
5. Save the output to the specified directory.

## Detailed Usage Examples

### Running the Full Pipeline

To execute the LUKE pipeline, use the `run.py` script. Below is an example:

```bash
python run.py --input example_structures/test_mol.xyz --output results/
```

This command will:

1. **Read the Input File**:
   - Parses the molecular structure from the specified XYZ file.

2. **Run ANI Forces**:
   - Computes atomic forces and identifies high-uncertainty atoms using the ANI model.

3. **Isolate High-Error Atoms**:
   - Fragments molecules around high-error atoms to generate smaller substructures.

4. **Sanitize Structures**:
   - Ensures chemical viability of the fragmented structures.

5. **Save Results**:
   - Outputs the sanitized structures and logs to the specified directory.

### Running Individual Modules

Each module in LUKE can be executed independently. Below are examples for running specific modules:

#### 1. **ANI Forces**

```bash
python -m luke.ani_forces --dataset example_structures/test_mol.xyz --model ANI2xr --device cuda --batch_size 1000
```

- **Parameters**:
  - `--dataset`: Path to the input dataset (HDF5 or XYZ format).
  - `--model`: ANI model to use (default: `ANI2xr`).
  - `--device`: Device for computation (`cuda` or `cpu`).
  - `--batch_size`: Number of structures processed per batch.

#### 2. **Structure Sanitizer**

```bash
python -m luke.structure_sanitizer --input results/high_error_atoms.xyz --output results/sanitized_structures.xyz
```

- **Parameters**:
  - `--input`: Path to the input XYZ file.
  - `--output`: Path to save the sanitized XYZ file.

### Example Dataset

An example dataset is provided in the `example_structures/` directory. Use `test_mol.xyz` to test the pipeline:

```bash
python run.py --input example_structures/test_mol.xyz --output results/
```

This will generate:

- Fragmented molecular structures in the `results/` directory.
- Logs and intermediate results for debugging.

## Troubleshooting

If you encounter issues while using LUKE, here are some common problems and their solutions:

- **Problem**: `torchani` submodule is not initialized.
  - **Solution**: Run the following commands to initialize the submodule:

    ```bash
    git submodule update --init --recursive
    ```

- **Problem**: Missing dependencies.
  - **Solution**: Ensure you have installed all dependencies listed in `environment.yaml` or `requirements.txt`.

- **Problem**: CUDA device not available.
  - **Solution**: Check if your system has a compatible GPU and CUDA installed. If not, use the `--device cpu` flag.

## Detailed Input/Output Formats

### Input Formats

1. **HDF5**:
   - Hierarchical data format for large datasets, must be formatted to TorchANI dataset standards.
   - Contains molecular structures, atomic species, and coordinates.

2. **XYZ**:
   - Standard molecular structure format.
   - Example:

     ```xyz
     3
     Comment line
     H 0.0 0.0 0.0
     O 0.0 0.0 1.0
     H 1.0 0.0 0.0
     ```

### Output Formats

1. **Fragmented Structures**:
   - XYZ files containing smaller molecular fragments.

2. **Logs**:
   - Detailed logs for debugging and analysis.

3. **Intermediate Results**:
   - Stored in the specified output directory for further inspection.

## Dependencies

- Python 3.11
- PyTorch
- TorchANI
- ASE
- RDKit
- Other dependencies (see `requirements.txt`)

## Running Tests & Quality Gates

After installation (or via `dev_ci_setup.sh`):

```bash
ruff check luke tests
mypy luke
pytest --disable-warnings --cov=luke
```

To build distribution artifacts locally:

```bash
python -m build --sdist --wheel --outdir dist
twine check dist/*
```

Or run everything through the Makefile target (see below):

```bash
make ci
```

### Local CI Mirror
`dev_ci_setup.sh` + `make ci` closely emulate the GitHub Actions workflow for reproducibility before pushing.

## Makefile Targets

Common developer targets are provided:

```makefile
make ci      # Full lint/type/test/build cycle
make lint    # Ruff lint
make type    # mypy type check
make test    # pytest with coverage
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

### Guidelines

- Follow PEP 8 for Python code.
- Write comprehensive tests for new features.
- Update the documentation as needed.

## Roadmap

- [ ] Integrate all standalone scripts into a cohesive pipeline.
- [ ] Add more example scripts and datasets.
- [ ] Improve test coverage.
- [ ] Optimize performance for large datasets.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
