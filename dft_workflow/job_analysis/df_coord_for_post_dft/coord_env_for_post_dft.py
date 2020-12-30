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

# # Analyze neighbor environments for post-DFT optimized slabs
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np

# #########################################################
from methods import get_structure_coord_df, get_df_coord

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    )
# -

# # Script Inputs

# verbose = True
verbose = False

# # Read data

df_jobs_anal = get_df_jobs_anal()
df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# + active=""
#
#
#

# +
# print("TEMP")

# TEMP
# df_jobs_anal_i = df_jobs_anal_i.iloc[0:3]

# df_jobs_anal = df_jobs_anal[df_jobs_anal.job_id_max == "kefehobo_84"]
# -

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/df_coord_for_post_dft",
    "out_data/df_coord_files")
if not os.path.exists(directory):
    os.makedirs(directory)

df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# +
# df_jobs_anal.loc[
#     [('nersc', 'fosurufu_23', 'o', 43, 1)]
#     ]

df_index = df_jobs_anal_i.index.to_frame()

df = df_index
df = df[
    (df["compenv"] == "nersc") &
    (df["slab_id"] == "fosurufu_23") &
    (df["ads"] == "o") &
    [True for i in range(len(df))]
    ]

# df_jobs_anal[
#     df.index
#     ]

df_jobs_anal_i.loc[df.index]
# -

# # Main Loop

# +
# import pandas as pd

# print(180 * "TEMP | ")
# idx = pd.IndexSlice
# df_jobs_anal_i = df_jobs_anal_i.loc[idx["sherlock", "kipatalo_90", "o", "NaN", 2, :], :]
# -

df_jobs_anal_i

# +
index_to_process = []
index_to_not_process = []
for index_i in df_jobs_anal_i.index:
    if index_i in df_atoms_sorted_ind.index:
        index_to_process.append(index_i)
    else:
        index_to_not_process.append(index_i)

df_jobs_anal_i_2 = df_jobs_anal_i.loc[index_to_process]

if len(index_to_not_process) > 0:
    print(
        "These systems don't have the required files locally",
        "Fix with rclone",
        "",
        sep="\n")
    tmp = [print(i) for i in index_to_not_process]
# -

# #########################################################
for name_i, row_i in df_jobs_anal_i_2.iterrows():
    # print(name_i)

    # if verbose:
    #     print(40 * "=")

    if name_i[3] != "NaN":
        active_site_new = int(name_i[3])
    else:
        active_site_new = "NaN"

    name_new_i = (
        name_i[0],
        name_i[1],
        name_i[2],

        active_site_new,
        # int(name_i[3]),

        name_i[4],
        )

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################
    # name_dict_i = dict(zip(list(df_jobs_anal_i.index.names), name_i))
    name_dict_i = dict(zip(list(df_jobs_anal_i_2.index.names), name_i))
    # #####################################################

    # #####################################################
    row_atoms_sorted_i = df_atoms_sorted_ind.loc[name_i]
    # #####################################################
    atoms_sorted_good_i = row_atoms_sorted_i.atoms_sorted_good
    failed_to_sort_i = row_atoms_sorted_i.failed_to_sort
    # #####################################################

    if not failed_to_sort_i:

        df_coord_i = get_df_coord(
            mode="post-dft",  # 'bulk', 'slab', 'post-dft'
            post_dft_name_tuple=name_i,
            )
        if df_coord_i is None:
            if verbose:
                print("No df_coord found, running")
            # #################################################
            # Get df_coord for post-dft, sorted slab
            df_coord_i = get_structure_coord_df(
                atoms_sorted_good_i,
                porous_adjustment=True,
                )

            # Pickling data ###################################
            file_name_i = "_".join([str(i) for i in list(name_new_i)]) + ".pickle"
            file_path_i = os.path.join(directory, file_name_i)
            print(file_path_i)
            with open(file_path_i, "wb") as fle:
                pickle.dump(df_coord_i, fle)
            # #################################################

        if "H" in df_coord_i.element.unique():
        # if True:
            df_coord_porous_adj_False_i = get_df_coord(
                mode="post-dft",  # 'bulk', 'slab', 'post-dft'
                post_dft_name_tuple=name_i,
                porous_adjustment=False,
                )

            if df_coord_porous_adj_False_i is None:
                df_coord_porous_adj_False_i = get_structure_coord_df(
                    atoms_sorted_good_i,
                    porous_adjustment=False,
                    )

                # Pickling data ###################################
                file_name_i = "_".join([str(i) for i in list(name_new_i)]) + "_porous_adj_False"
                file_name_i += ".pickle"
                file_path_i = os.path.join(directory, file_name_i)
                print("*H containing, turning off porous adjustment:", "\n", file_path_i)
                with open(file_path_i, "wb") as fle:
                    pickle.dump(df_coord_porous_adj_False_i, fle)
                #################################################

# + active=""
#
#
#
# -

# # Running through df and reading df_coord to test

# for name_i, row_i in df_jobs_anal_i.iterrows():
for name_i, row_i in df_jobs_anal_i_2.iterrows():
    tmp = get_df_coord(
        slab_id=None,
        bulk_id=None,
        mode="post-dft",  # 'bulk', 'slab', 'post-dft'
        slab=None,
        post_dft_name_tuple=name_i,
        )

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################
