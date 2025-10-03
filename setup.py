import subprocess
from pathlib import Path

from setuptools import find_packages, setup

# Ensure the torchani submodule is initialized and updated

def ensure_torchani_submodule():
    try:
        # Check if the submodule directory exists
        submodule_path = Path("external/torchani")
        if not submodule_path.exists():
            print("TorchANI submodule not found. Initializing and updating...")
            subprocess.run(["git", "submodule", "update",
                           "--init", "--recursive"], check=True)
            print("TorchANI submodule initialized successfully.")
    except subprocess.CalledProcessError as e:
        print("Error: Failed to initialize the TorchANI submodule.")
        print(e)
        exit(1)


# Run the submodule check
ensure_torchani_submodule()

setup(
    name="luke",  # Package name
    version="0.2.0",
    author="Nick Terrel",
    author_email="nickterrel4@gmail.com",
    description="LUKE: USE the Forces - Molecular fragmentation for active learning in ANI",
    long_description=Path(__file__).parent.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/roitberg-group/LUKE",
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        "torch>=2.1",
        "torchani>=2.2",
        "numpy>=1.24",
        "pandas>=2.0",
        "tqdm>=4.65",
        "rich>=13",
        "typer>=0.12",
        "ase>=3.22",
        "scipy>=1.10",
    ],
    extras_require={
        "chem": [
            "rdkit-pypi>=2022.9.5",
            "openbabel-wheel>=3.1.1.post1",
        ],
        "dev": [
            "pytest>=8",
            "pytest-cov>=5",
            "ruff>=0.5",
            "mypy>=1.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "luke=luke.__main__:main",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
