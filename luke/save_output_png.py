# NOTE: This is not working, in a jupyter notebook it tries to save the png to my local machine
#       Maybe try to use some other functionality to save each of the viewers? 

# EDIT THIS! Should not take a .pkl input, but rather xyz or h5 consistent with input files

#TO DO:
# - add argparser commands for inputs / outputs
# - make it so that the script actually saves things, probably

import pandas as pd
import os
from ase import Atoms
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt

df = pd.read_pickle('/home/nick/filtered_1x_first_w_stdev.pkl')

def count_ones(lst):
    if isinstance(lst, list):
        return sum(1 for val in lst if val == 1)
    return 0

# Count the total number of 1s in the 'relative_range' column
total_ones = df['relative_stdev'].apply(count_ones).sum()
print('There are', total_ones, 'atoms with a relative_stdev > 2 in the filtered 1x_first dataset')

save_directory = '/home/nick/PNGs/'

for formula, row in df.iterrows():
    species = row['species'].squeeze().cpu()
    coordinates = row['coordinates'].squeeze().cpu().detach()
    relative_stdev = row['relative_stdev']
    print('Species:\n',species.tolist())
    print('Bad atoms:\n', relative_stdev)
    atoms = Atoms(symbols=species, positions=coordinates)
    highlight_indices = [i for i, val in enumerate(relative_stdev) if val == 1]

    highlighted_atoms = atoms.copy()
    highlighted_atoms.positions = highlighted_atoms.positions[highlight_indices]

    fig, ax = plt.subplots()
    plot_atoms(atoms, ax, radii=0.3)
    plot_atoms(highlighted_atoms, ax, radii=0.3)
    image_path = os.path.join(save_directory, f'{formula}.png')
    plt.savefig(image_path, format='png', dpi=300)
    plt.close(fig)


