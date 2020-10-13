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

import pickle

# #########################################################
from methods import get_structure_coord_df, get_df_coord

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    )
# -

# # Script Inputs

verbose = True
verbose = False

# # Read data

# df_dft = get_df_dft()
# df_slab = get_df_slab()
# structure_coord_df = get_structure_coord_df()
# df_jobs = get_df_jobs()
# df_jobs_paths = get_df_jobs_paths()
# df_jobs_data = get_df_jobs_data()
# df_jobs_data_clusters = get_df_jobs_data_clusters()
df_jobs_anal = get_df_jobs_anal()
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# df_slab_ids = get_df_slab_ids()
# df_job_ids = get_df_job_ids()
# df_slabs_to_run = get_df_slabs_to_run()
# df_coord = get_df_coord()
# df_active_sites = get_df_active_sites()
# slab_id = get_slab_id()
# job_id = get_job_id()
# slab_thickness = get_slab_thickness()

# + active=""
#
#
#

# +
# TEMP
# df_jobs_anal_i = df_jobs_anal_i.iloc[0:3]

# +
# directory = "out_data/df_coord_files"

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/df_coord_for_post_dft",
    "out_data/df_coord_files")

# /home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/
# dft_workflow/job_analysis/df_coord_for_post_dft

if not os.path.exists(directory):
    os.makedirs(directory)
# -

# # Main Loop

df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

for name_i, row_i in df_jobs_anal_i.iterrows():
    if verbose:
        print(40 * "=")

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################
    name_dict_i = dict(zip(list(df_jobs_anal_i.index.names), name_i))
    # #####################################################

    # #####################################################
    row_atoms_sorted_i = df_atoms_sorted_ind.loc[name_i]
    # #####################################################
    atoms_sorted_good_i = row_atoms_sorted_i.atoms_sorted_good
    # #####################################################



    df_coord_i = get_df_coord(
        mode="post-dft",  # 'bulk', 'slab', 'post-dft'
        post_dft_name_tuple=name_i,
        )
    if df_coord_i is None:
        # #################################################
        # Get df_coord for post-dft, sorted slab
        df_coord_i = get_structure_coord_df(atoms_sorted_good_i)

        # Pickling data ###################################
        file_name_i = "_".join([str(i) for i in list(name_i)]) + ".pickle"
        file_path_i = os.path.join(directory, file_name_i)
        print(file_path_i)
        with open(file_path_i, "wb") as fle:
            pickle.dump(df_coord_i, fle)
        # #################################################

# row_atoms_sorted_i = 
df_atoms_sorted_ind
# .loc[name_i]

# +
# assert False

# + active=""
#
#
#
# -

for name_i, row_i in df_jobs_anal_i.iterrows():

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
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# data_dict_i.extend(name_dict_i)

# + jupyter={"source_hidden": true}
# data_dict_i

# + jupyter={"source_hidden": true}
# df_jobs_anal_i

# + jupyter={"source_hidden": true}
# df_coord_i
