# Right now this is kind of nonsense, but it should take an input from the `isolator.py` script
#  and produce a "chemically viable" substructure

# TO DO:
# - add argparser, if __name__ == __main__:, etc
# - get the script actually functioning

import numpy as np
from scipy.spatial import distance_matrix
from .io_utils import read_xyz, write_xyz
import rdkit.Chem

def compute_connectivity(coordinates, threshold=1.8):
    dist_matrix = distance_matrix(coordinates, coordinates)
    connectivity = dist_matrix < threshold
    np.fill_diagonal(connectivity, False)  # Exclude self-connectivity
    return connectivity

def find_largest_connected_component(connectivity):
    # I HATE THIS FUNCTION MAKE SOMETHING DIFFERENT -- MAYBE ONE RDKIT OBJECT PER FRAGMENT THEN RECOMBINE OR SOMETHING JUST DO SOMETHING PLEASE GOD
    # Take inspiration from lammps-ani/cumolfind
    num_atoms = len(connectivity)
    visited = set()
    largest_component = set()

    def dfs(current):
        if current in visited:
            return set()
        visited.add(current)
        component = {current}
        for neighbor in range(num_atoms):
            if connectivity[current, neighbor]:
                component.update(dfs(neighbor))
        return component

    for atom in range(num_atoms):
        component = dfs(atom)
        if len(component) > len(largest_component):
            largest_component = component

    return largest_component

def process_molecule(file_path):
    mol = Chem.MolFromMolFile(file_path)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    for i, fragment in enumerate(fragments):
        frag_with_h = Chem.AddHs(fragment)
        num_conformers = 5
        AllChem.EmbedMultipleConfs(frag_with_h, numConfs=num_conformers)

        for conf in frag_with_h.GetConformers():
            AllChem.MMFFOptimizeMolecule(frag_with_h, conId=conf.GetId())
        for j, conf in enumerate(frag_with_h.GetConformers()):
            xyz_filename = f"{file_path.stem}_frag_{i}_conf_{j}.xyz"
            Chem.MolToXYZFile(frag_with_h, xyz_filename, confId=conf.GetId())
        print(f"Processed Fragment {i+1} of {file_path}")