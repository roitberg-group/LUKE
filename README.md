# LUKE: USE the Forces

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

LUKE is designed to be integrated into molecular simulation and machine learning workflows. Example scripts will be provided soon for:

- Identifying high-uncertainty atomic environments
- Fragmenting molecules based on detected high-error forces
- Preparing structures for active learning retraining

## Dependencies

- Python 3.11
- PyTorch
- TorchANI
- ASE
- RDKit
- Other dependencies (to be specified in future updates)

## Contributing

Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request.
