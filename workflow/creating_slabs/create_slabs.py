# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Creating slabs from IrOx polymorph dataset
# ---
#
#

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from pathlib import Path
import copy
import pickle

import numpy as np
import pandas as pd

from ase import io

from tqdm import tqdm

# #########################################################
from catkit.gen.surface import SlabGenerator

# #########################################################
from misc_modules.pandas_methods import drop_columns
from misc_modules.misc_methods import GetFriendlyID
from ase_modules.ase_methods import view_in_vesta

# #########################################################
from methods import (
    get_df_dft, symmetrize_atoms,
    get_structure_coord_df, remove_atoms)
from proj_data import metal_atom_symbol

# #########################################################
from local_methods import analyse_local_coord_env, check_if_sys_processed
# -

# # Script Inputs

# +
# Distance from top z-coord of slab that we'll remove atoms from
dz = 4

facets = [
    (1, 0, 0),
    # (0, 1, 0),
    # (0, 0, 1),

    # Weird cuts
    (2, 1, 4)
    ]
# -

# # Read Data

# +
# #########################################################
# DFT dataframe
df_dft = get_df_dft()

# #########################################################
# Previous df_slab dataframe
path_i = os.path.join(
    "out_data",
    "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    print("File exists!")
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
else:
    df_slab_old = pd.DataFrame()

# +
directory = "out_data/final_slabs"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "out_data/slab_progression"
if not os.path.exists(directory):
    os.makedirs(directory)
# -

# Only taking the most stable polymorphs for now

# +
# df_dft_ab2_i = df_dft[df_dft.stoich == "AB2"].sort_values("dH").iloc[0:10]
# df_dft_ab3_i = df_dft[df_dft.stoich == "AB3"].sort_values("dH").iloc[0:10]

df_dft_ab2_i = df_dft[df_dft.stoich == "AB2"].sort_values("dH").iloc[0:3]
df_dft_ab3_i = df_dft[df_dft.stoich == "AB3"].sort_values("dH").iloc[0:2]

df_i = pd.concat([
    df_dft_ab2_i,
    df_dft_ab3_i,
    ], )
# -

# # Creating slabs from bulks

# +
data_dict_list = []
iterator = tqdm(df_i.index.tolist())
for i_cnt, bulk_id in enumerate(iterator):
    row_i = df_dft.loc[bulk_id]

    # #####################################################
    # Row parameters ######################################
    bulk_id_i = row_i.name
    atoms = row_i.atoms
    # #####################################################

    for facet in facets:
        data_dict_i = dict()

        data_dict_i["bulk_id"] = bulk_id_i
    
        facet_str = "".join([str(i) for i in list(facet)])
        data_dict_i["facet"] = facet_str

        sys_processed = check_if_sys_processed(
            bulk_id_i=bulk_id_i,
            facet_str=facet_str,
            df_slab_old=df_slab_old)

        # Only run if not in df_slab_old (already run)
        if not sys_processed:
            slab_id_i = GetFriendlyID(append_random_num=True)
            data_dict_i["slab_id"] = slab_id_i

            SG = SlabGenerator(
                atoms, facet, 10, vacuum=15,
                fixed=None, layer_type='ang',
                attach_graph=True,
                standardize_bulk=True,
                primitive=True, tol=1e-08)

            slab_i = SG.get_slab()
            slab_i.set_pbc([True, True, True])

            # #############################################
            data_dict_list.append(data_dict_i)

            # #############################################
            # Write slab to file ##########################
            file_name_i = bulk_id_i + "_" + slab_id_i + "_" + facet_str + "_0" + ".cif"
            slab_i.write(os.path.join(
                "out_data",
                "slab_progression",
                file_name_i))

            # Rereading the structure file to get it back into ase format (instead of CatKit class)
            slab_i = io.read(os.path.join(
                "out_data",
                "slab_progression",
                file_name_i))
            data_dict_i["slab_0"] = slab_i
            # #############################################


# #########################################################
df_slab = pd.DataFrame(data_dict_list)
num_new_rows = len(data_dict_list)

if num_new_rows > 0:
    df_slab = df_slab.set_index("slab_id")
elif num_new_rows == 0:
    print("There aren't any new rows to process")
    assert False
# -

# # Removing surface Ir atoms that weren't oxygen saturated

# +
data_dict = dict()
iterator = tqdm(df_slab.index.tolist())
for i_cnt, slab_id in enumerate(iterator):
    row_i = df_slab.loc[slab_id]

    # #####################################################
    slab_id_i = row_i.name
    bulk_id_i = row_i.bulk_id
    facet_i = row_i.facet
    slab = row_i.slab_0
    # #####################################################

    # #####################################################
    row_bulk_i = df_dft.loc[bulk_id_i]

    bulk = row_bulk_i.atoms
    # #####################################################

    sys_processed = check_if_sys_processed(
        bulk_id_i=bulk_id_i,
        facet_str=facet_i,
        df_slab_old=df_slab_old)

    # Only run if not in df_slab_old (already run)
    if not sys_processed:

        # out_data_dict = analyse_local_coord_env(atoms=bulk)
        # coor_env_dict_bulk = out_data_dict["coor_env_dict"]


        # #################################################
        # #################################################
        z_positions = slab.positions[:,2]

        z_max = np.max(z_positions)
        z_min = np.min(z_positions)

        # #################################################
        # #################################################
        df_coord_slab_i = get_structure_coord_df(slab)

        # #################################################
        metal_atoms_to_remove = []
        for atom in slab:
            if atom.symbol == metal_atom_symbol:
                z_pos_i = atom.position[2]
                if z_pos_i >= z_max - dz or z_pos_i <= z_min + dz:
                    row_coord = df_coord_slab_i[
                        df_coord_slab_i.structure_index == atom.index].iloc[0]
                    num_o_neighbors = row_coord.neighbor_count["O"]

                    if num_o_neighbors < 6:
                        metal_atoms_to_remove.append(atom.index)

        slab_new = remove_atoms(atoms=slab, atoms_to_remove=metal_atoms_to_remove)

        # #################################################
        # Write slab to file ##############################
        file_name_i = bulk_id_i + "_" + slab_id_i + "_" + facet_i + "_1" + ".cif"
        slab_new.write(os.path.join(
            "out_data",
            "slab_progression",
            file_name_i))

        data_dict[slab_id_i] = slab_new

# #########################################################
df_slab["slab_1"] = df_slab.index.map(data_dict)
# -

# # Removing extra oxygens at surface

# +
data_dict = dict()
# for i_cnt, row_i in df_slab.iterrows():
iterator = tqdm(df_slab.index.tolist())
for i_cnt, slab_id in enumerate(iterator):

    # row_i = df_slab[df_slab.bulk_id == "8p8evt9pcg"]
    # row_i = row_i.iloc[0]
    row_i = df_slab.loc[slab_id]

    # #####################################################
    slab_id_i = row_i.name
    bulk_id_i = row_i.bulk_id

    facet_i = row_i.facet

    slab = row_i.slab_1
    # #####################################################
    
    sys_processed = check_if_sys_processed(
        bulk_id_i=bulk_id_i,
        facet_str=facet_i,
        df_slab_old=df_slab_old)

    # Only run if not in df_slab_old (already run)
    if not sys_processed:

        df_coord_slab_i = get_structure_coord_df(slab)

        df_i = df_coord_slab_i[df_coord_slab_i.element == "O"]
        df_i = df_i[df_i.num_neighbors == 0]

        o_atoms_to_remove = df_i.structure_index.tolist()

        slab_new = remove_atoms(slab, atoms_to_remove=o_atoms_to_remove)

        # #####################################################
        # Write slab to file ##################################
        file_name_i = bulk_id_i + "_" + slab_id_i + "_" + facet_i + "_2" + ".cif"
        slab_new.write(os.path.join(
            "out_data",
            "slab_progression",
            file_name_i))

        data_dict[slab_id_i] = slab_new


# #########################################################
df_slab["slab_2"] = df_slab.index.map(data_dict)
# -

# # Combined new and old `df_slab` dataframes

df_slab = pd.concat([
    df_slab_old,
    df_slab,
    ])


# # Write final slab structures and `df_slabs` to file

# +
def method(row_i):
    slab = row_i.slab_2
    bulk_id_i = row_i.bulk_id
    slab_id_i = row_i.name
    facet_i = row_i.facet

    file_name_i = bulk_id_i + "_" + slab_id_i + "_" + facet_i + "_final" + ".cif"
    slab.write(os.path.join(
        "out_data/final_slabs",
        file_name_i))

tmp = df_slab.apply(
    method,
    axis=1)

# Pickling data ###########################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
    pickle.dump(df_slab, fle)
# #########################################################

# + active=""
#
#
#
#
#

# + jupyter={"source_hidden": true}
# data_dict_list = []
# for i_cnt, row_i in df_i.iterrows():
#     # row_i = df_i.iloc[3]

#     # #####################################################
#     # Row parameters ######################################
#     id_i = row_i.name
#     atoms = row_i.atoms
#     # #####################################################

#     for facet in facets:
#         data_dict_i = dict()

#         data_dict_i["bulk_id"] = id_i
    
#         facet_str = "".join([str(i) for i in list(facet)])
#         data_dict_i["facet"] = facet_str

#         slab_id_i = GetFriendlyID(append_random_num=True)
#         data_dict_i["slab_id"] = slab_id_i

#         SG = SlabGenerator(
#             atoms,
#             facet,
#             10,
#             vacuum=15,
#             fixed=None,
#             layer_type='ang',
#             attach_graph=True,
#             standardize_bulk=True,
#             primitive=True,
#             tol=1e-08,
#             )

#         slab_i = SG.get_slab()
#         slab_i.set_pbc([True, True, True])

#         # #################################################
#         data_dict_list.append(data_dict_i)

#         # #################################################
#         # Write slab to file ##############################        
#         file_name_i = id_i + "_" + slab_id_i + "_" + facet_str + "_0" + ".cif"
#         slab_i.write(os.path.join(
#             "out_data",
#             file_name_i))

#         # Rereading the structure file to get it back into ase format (instead of CatKit class)
#         slab_i = io.read(os.path.join(
#             "out_data",
#             file_name_i))
#         data_dict_i["slab_0"] = slab_i
#         # #################################################


# # #########################################################
# df_slab = pd.DataFrame(data_dict_list)
# df_slab = df_slab.set_index("slab_id")
