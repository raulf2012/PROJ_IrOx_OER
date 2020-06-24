"""
"""

#| - Import Modules
import os
import sys

import copy

import pickle

import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns

# #########################################################
#__|


def get_df_dft():
    """
    """
    #| - get_df_dft
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/process_bulk_dft",
        "out_data/df_dft.pickle")
    with open(path_i, "rb") as fle:
        df_dft = pickle.load(fle)

    return(df_dft)
    #__|

def symmetrize_atoms(
    write_atoms=False,
    ):
    """
    """
    #| - symmetrize_atoms
    name = argv[1]
    atoms = read(name)

    images = [atoms.copy()]

    spglibdata = get_symmetry_dataset(
        (
            atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers(),
            ),
        symprec=1e-3,
        )

    spacegroup = spglibdata['number']

    wyckoffs = spglibdata['wyckoffs']
    print(spglibdata['number'])
    print(wyckoffs)


    s_name = spglibdata['international']
    #str(spacegroup) + '_' + '_'.join(sorted(wyckoffs))

    std_cell = spglibdata['std_lattice']
    positions = spglibdata['std_positions']
    numbers = spglibdata['std_types']

    atoms = Atoms(
        numbers=numbers,
        cell=std_cell,
        pbc=True,
        )

    atoms.set_scaled_positions(positions)
    atoms.wrap()
    images += [atoms]


    if write_atoms:
        # view(images)
        new_name = name.rstrip('.cif') + '_conventional_' + str(spacegroup) + '.cif'
        #print(new_name)

        atoms.write(new_name)

    return(atoms)
    #__|

def get_structure_coord_df(atoms):
    """
    """
    #| - get_structure_coord_df
    atoms_i = atoms
    structure = AseAtomsAdaptor.get_structure(atoms_i)

    # CrysNN = local_env.VoronoiNN(
    #     tol=0,
    #     targets=None,
    #     cutoff=13.0,
    #     allow_pathological=False,
    #     weight='solid_angle',
    #     extra_nn_info=True,
    #     compute_adj_neighbors=True,
    #     )

    CrysNN = local_env.CrystalNN(
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.01, 0.4),
        x_diff_weight=3.0,
        porous_adjustment=True,
        search_cutoff=7,
        fingerprint_length=None)


    coord_data_dict = dict()
    data_master = []
    for i_cnt, site_i in enumerate(structure.sites):
        site_elem_i = site_i.species_string

        data_dict_i = dict()
        data_dict_i["element"] = site_elem_i
        data_dict_i["structure_index"] = i_cnt

        nn_info_i = CrysNN.get_nn_info(structure, i_cnt)
        data_dict_i["nn_info"] = nn_info_i

        neighbor_list = []
        for neighbor_j in nn_info_i:
            neigh_elem_j = neighbor_j["site"].species_string
            neighbor_list.append(neigh_elem_j)

        neighbor_count_dict = dict()
        for i in neighbor_list:
            neighbor_count_dict[i] = neighbor_count_dict.get(i, 0) + 1

        data_dict_i["neighbor_count"] = neighbor_count_dict
        data_master.append(data_dict_i)

    df_struct_coord_i = pd.DataFrame(data_master)

    # #####################################################
    # #####################################################
    def method(row_i):
        neighbor_count = row_i.neighbor_count
        num_neighbors = 0
        for key, val in neighbor_count.items():
            num_neighbors += val
        return(num_neighbors)

    df_struct_coord_i["num_neighbors"] = df_struct_coord_i.apply(
        method,
        axis=1)

    return(df_struct_coord_i)
    #__|

def remove_atoms(atoms=None, atoms_to_remove=None):
    """Remove atoms that match indices of `atoms_to_remove` and return new ase atoms object
    """
    #| - remove_atoms
    atoms_new = copy.deepcopy(atoms)

    bool_mask = []
    for atom in atoms:

        if atom.index in atoms_to_remove:
            bool_mask.append(False)
        else:
            bool_mask.append(True)

    atoms_new = atoms_new[bool_mask]

    return(atoms_new)
    #__|



#| - __old__
# from spglib import get_symmetry_dataset
# from ase import Atoms
# from ase.io import read
# from sys import argv
# import numpy as np
#
# from ase.visualize import view

# # #############################################################################
# # #############################################################################
# # #############################################################################

# import os
# import sys
# import pickle
#
# import time
# t0 = time.time()
#
# import numpy as np
# import pandas as pd
#
# from pymatgen.analysis.local_env import (
#     NearNeighbors,
#     VoronoiNN, site_is_of_motif_type)
#
#
# # #############################################################################
# from ase_modules.ase_methods import view_in_vesta
#
#
# sys.path.insert(0, os.path.join(os.environ["PROJ_irox"], "data"))
# from proj_data_irox import (
#     unique_ids_path,
#     )
#__|
