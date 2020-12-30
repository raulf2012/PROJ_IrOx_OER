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

# # Create `df_coord` for all atoms objects in `df_init_slabs`
# # Fix `df_coord` for systems in `df_slab`
# ---
#
# TEMP

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path

import numpy as np

# # #########################################################
from methods import (
    get_df_slab,
    get_structure_coord_df,
    get_df_init_slabs,
    get_structure_coord_df,
    )

# # #########################################################
from local_methods import process_sys, df_matches_slab
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read Data

# +
# #########################################################
df_slab = get_df_slab()
df_slab = df_slab.set_index("slab_id")
df_slab_i = df_slab

# #########################################################
df_init_slabs = get_df_init_slabs()
# -

# # Creating `df_coord` for all init slabs

# +
# for name_i, row_i in df_init_slabs.iterrows():

iterator = tqdm(df_init_slabs.index, desc="Loop:")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    row_i = df_init_slabs.loc[index_i]
    # #####################################################
    compenv_i, slab_id_i, ads_i, active_site_i, att_num_i = index_i
    # #####################################################
    init_atoms_i = row_i.init_atoms
    # #####################################################

    if active_site_i == "NaN":
        active_site_str_i = active_site_i
    else:
        active_site_str_i = str(int(active_site_i))

    # #####################################################
    file_name_i = "" + \
        compenv_i + "__" + \
        slab_id_i + "__" + \
        ads_i + "__" +  \
        active_site_str_i + "__" +  \
        str(att_num_i) + \
        ""
        # str(int(active_site_i)) + "__" +  \
    file_name_i += ".pickle"

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs/maintain_df_coord",
        "out_data/df_coord__init_slabs")

    path_i = os.path.join(directory, file_name_i)


    # #####################################################
    my_file = Path(path_i)
    if not my_file.is_file():
        df_coord_i = get_structure_coord_df(
            init_atoms_i,
            porous_adjustment=True,
            )

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path_i, "wb") as fle:
            pickle.dump(df_coord_i, fle)
# -

# # TEMP

# # Main Loop

df_slab_i = df_slab_i[df_slab_i.phase > 1]

if False:
    iterator = tqdm(df_slab_i.index, desc="1st loop")
    for i_cnt, slab_id_i in enumerate(iterator):
        print(
            40 * "*",
            "\n",
            "slab_id: ", slab_id_i,
            sep="")

        # #####################################################
        row_i = df_slab.loc[slab_id_i]
        # #####################################################
        slab_new = row_i.slab_final
        slab_old = row_i.slab_final_old
        # #####################################################


        # #####################################################
        path_pre = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/creating_slabs/out_data/df_coord_files")

        # #####################################################
        out_dict = process_sys(
            slab_id=slab_id_i,
            slab=slab_new,
            path_pre=path_pre,
            mode="new",  # 'new' or 'old'
            )
        df_matches_slab = out_dict["df_matches_slab"]
        df_coord_redone = out_dict["df_coord_redone"]

        # #####################################################
        out_dict = process_sys(
            slab_id=slab_id_i,
            slab=slab_old,
            path_pre=path_pre,
            mode="old",  # 'new' or 'old'
            )
        df_matches_slab = out_dict["df_matches_slab"]
        df_coord_redone = out_dict["df_coord_redone"]

        print("")

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("fix_df_coord.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
