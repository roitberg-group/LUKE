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
    name="luke",
    version="0.2.0",
    author="Nick Terrel",
    author_email="nickterrel4@gmail.com",
    description="LUKE: USE the Forces - Molecular fragmentation for active learning in ANI",
    long_description=Path(__file__).parent.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/roitberg-group/LUKE",
    packages=find_packages(),
    # Dependencies, extras, license, classifiers managed by pyproject.toml (PEP 621).
    # Keep this file as a lightweight shim for legacy tooling only.
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "luke=luke.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
