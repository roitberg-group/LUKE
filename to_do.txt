Overall pipeline:
(Create a `run.py` or some similarly named script to run the protocol)
input h5 or xyz -> ani_forces.py -> isolator.py -> structure_sanitizer + rdkit_fragment_identifier.py -> output files
First and last stages should use functions in io.py


For isolator.py:
* Incorporate `ani_forces.py` module, which is called to run ANI on an input XYZ or HDF5 file


For structure_sanitizer.py:
* Incorporate `isolator.py` which takes inputs from `ani_forces.py` to select high-error atoms
* This script should take some 'nonsense' structures and clean them up to chemically viable species


For io.py:
* Organize functions in logical format, update the `read_xyz` function to reflect newest torchani version
* Clean up the functions to hash xyz coords, write h5 and gaussian/slurm files