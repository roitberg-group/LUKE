from setuptools import setup, find_packages

setup(
    name="luke",  # Package name
    version="0.1.0",
    author="Nick Terrel",
    author_email="nterrel@ufl.com",
    description="LUKE: USE the Forces - Molecular fragmentation for active learning in ANI",
    long_description=open("README.md").read(),
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
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

