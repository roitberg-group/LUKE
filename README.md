# LUKE: USE the Forces

## NOTE: This is a pre-release, the scripts here do not fully cooperate yet. Each step in the protocol currently exists as standalone scripts, and the pipeline is under construction

## "Largest Uncertainty Kaleidoscope Estimator: Uncertainty-driven Sampling of high-Error Forces"

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

LUKE relies on TorchANI as an external module, stored as a submodule:

```bash
# Clone the repository with submodules
git clone --recursive git@github.com:roitberg-group/LUKE.git
cd LUKE
git submodule update --init --recursive
pip install -v -e .
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

### Input Formats

- **HDF5**: Hierarchical data format for large datasets.
- **XYZ**: Standard molecular structure format.

### Output

- Fragmented molecular structures in XYZ format.
- Logs and intermediate results for debugging and analysis.

## Dependencies

- Python 3.11
- PyTorch
- TorchANI
- ASE
- RDKit
- Other dependencies (see `requirements.txt`)

## Running Tests

To run the tests, use the following command:

```bash
pytest tests/
```

This will execute all test cases in the `tests/` directory and provide a summary of the results.

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
