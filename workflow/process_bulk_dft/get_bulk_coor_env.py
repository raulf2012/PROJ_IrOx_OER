# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Getting the coordination environment data for each bulk
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

from tqdm.notebook import tqdm

from ase import io

# # #########################################################
from methods import (
    get_df_dft,
    get_structure_coord_df,
    )
# -

# # Read data

df_dft = get_df_dft()

directory = "out_data/df_coord_files"
if not os.path.exists(directory):
    os.makedirs(directory)

data_dict_list = []
iterator = tqdm(df_dft.index.tolist(), desc="1st loop")
for i_cnt, bulk_id_i in enumerate(iterator):
    row_i = df_dft.loc[bulk_id_i]

    data_dict_i = dict()
    # #####################################################
    atoms = row_i.atoms
    bulk_id = row_i.name
    # #####################################################

    df_coord_i = get_structure_coord_df(atoms)
    
    # Pickling data ###########################################
    with open(os.path.join(directory, bulk_id + ".pickle"), "wb") as fle:
        pickle.dump(df_coord_i, fle)
    # #########################################################

# + active=""
#
#
#
#
#
#
#
#
#
#

# + jupyter={"source_hidden": true}

# from pathlib import Path
# import copy

# import json

# import numpy as np
# import pandas as pd


# # from tqdm import tqdm
# from tqdm.notebook import tqdm

# # #########################################################


# # #########################################################
# from misc_modules.pandas_methods import drop_columns
# from misc_modules.misc_methods import GetFriendlyID
# from ase_modules.ase_methods import view_in_vesta

# from proj_data import metal_atom_symbol

# #########################################################
# from local_methods import (
#     analyse_local_coord_env, check_if_sys_processed,
#     remove_nonsaturated_surface_metal_atoms,
#     remove_noncoord_oxygens,
#     create_slab_from_bulk,
#     get_slab_thickness,
#     remove_highest_metal_atoms,
#     remove_all_atoms_above_cutoff,
#     create_final_slab_master,
#     constrain_slab,
#     )

# # from local_methods import calc_surface_area
