from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="luke",  # Package name
    version="0.1.0",
    author="Nick Terrel",
    author_email="nickterrel4@gmail.com",
    description="LUKE: USE the Forces - Molecular fragmentation for active learning in ANI",
    long_description=Path(__file__).parent.joinpath("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/roitberg-group/LUKE",
    packages=find_packages(),  # Automatically find packages
    install_requires=[
        "torch",
        "torchani",
        "rdkit",
        "numpy",
        "pandas",
        "tqdm"
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
