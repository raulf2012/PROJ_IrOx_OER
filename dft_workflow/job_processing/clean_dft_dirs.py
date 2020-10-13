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

# # Clean DFT Jobs
# ---
#
# Run in computer cluster to perform a variety of job clean and processing
#
# Currently the following things are done:
#
# 1. Process large `job.out` files, if `job.out` is larger than `job_out_size_limit` than creates new `job.out.new` file removes middle section of file and leaves behind the beginning and end of the original file
# 1. Rclone copy the job directories to the Stanford Google Drive
#
# ## TODO
# * Remove large files if they are newer revisions (Only time you need large VASP files are when starting a new job and therefore need WAVECAR or charge files)

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import pickle
import shutil
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_paths,
    get_df_jobs_anal,
    )

# #########################################################
from local_methods import (
    cwd, process_large_job_out,
    rclone_sync_job_dir,
    parse_job_state,
    )
# -

# # Script Inputs

# +
verbose = False

job_out_size_limit = 5  # MB

# +
compenv = os.environ.get("COMPENV", None)

# if compenv == "wsl":
#     compenv = "slac"

proj_dir = os.environ.get("PROJ_irox_oer", None)
# -

# # Read Data

# +
# #########################################################
df_jobs = get_df_jobs(exclude_wsl_paths=False)
df_i = df_jobs[df_jobs.compenv == compenv]

# #########################################################
df_jobs_paths = get_df_jobs_paths()
df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv]

# #########################################################
df_jobs_anal = get_df_jobs_anal()

if verbose:
    print(60 * "-")
    print("Directories being parsed")
    tmp = [print(i) for i in df_jobs_paths_i.path_rel_to_proj.tolist()]
    print("")
# -

# # Iterate through rows

# +
# TEMP
# print("TEMP TEMP TEMP ju7y6iuesdfghuhertyui")

# #########################################################
# df_i = df_i.loc[["dunivesu_80"]]
# df_i = df_i.loc[["mitanapo_92"]] 
# df_i = df_i.loc[["tawawobu_24"]] 

# df_i = df_i.loc[["kepigiwu_49"]] 

# +
# print("TEMP")
# assert False

# +
# jobs_processed = []

iterator = tqdm(df_i.index.tolist(), desc="1st loop")
for index_i in iterator:
    # #####################################################
    row_i = df_i.loc[index_i]
    # #####################################################
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    att_num_i = row_i.att_num
    compenv_i = row_i.compenv
    active_site_i = row_i.active_site
    # #####################################################

    if active_site_i == "NaN":
        tmp = 42
    elif np.isnan(active_site_i):
        active_site_i = "NaN"
    

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
    row_jobs_paths_i = df_jobs_paths_i.loc[index_i]
    # #####################################################
    path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
    path_full = row_jobs_paths_i.path_full
    path_rel_to_proj = row_jobs_paths_i.path_rel_to_proj
    # #####################################################

    # #####################################################
    in_index = df_jobs_anal.index.isin(
        [(compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]).any()
    if in_index:
        row_anal_i = df_jobs_anal.loc[
            compenv_i, slab_id_i, ads_i, active_site_i, att_num_i]
        # #################################################
        job_completely_done_i = row_anal_i.job_completely_done
        # #################################################
    else:
        job_completely_done_i = None

    # if job_completely_done_i:
    #     print("job done:", path_full)

    # #####################################################
    path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        path_rel_to_proj)
    print(path_i)

    # Don't run methods if not in remote cluster
    if compenv != "wsl":
        my_file = Path(path_i)
        if my_file.is_dir():

            # Only do these operations on non-running jobs
            job_state_dict = parse_job_state(path_i)
            job_state_i = job_state_dict["job_state"]

            if verbose:
                print("job_state_i:", job_state_i)

            # #########################################
            if job_state_i != "RUNNING":
                process_large_job_out(
                    path_i, job_out_size_limit=job_out_size_limit)

            # #########################################
            rclone_sync_job_dir(
                path_job_root_w_att_rev=path_job_root_w_att_rev,
                path_rel_to_proj=path_rel_to_proj,
                verbose=False,
                )

# +
# assert False
# -

# # Remove systems that are completely done

# +
# iterator = tqdm(df_i.index.tolist(), desc="1st loop")
# for index_i in iterator:
#     # #####################################################
#     row_i = df_i.loc[index_i]
#     # #####################################################
#     slab_id_i = row_i.slab_id
#     ads_i = row_i.ads
#     att_num_i = row_i.att_num
#     compenv_i = row_i.compenv
#     active_site_i = row_i.active_site
#     # #####################################################

#     if active_site_i == "NaN":
#         tmp = 42
#     elif np.isnan(active_site_i):
#         active_site_i = "NaN"

#     # #####################################################
#     df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
#     row_jobs_paths_i = df_jobs_paths_i.loc[index_i]
#     # #####################################################
#     path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
#     path_full = row_jobs_paths_i.path_full
#     path_rel_to_proj = row_jobs_paths_i.path_rel_to_proj
#     # #####################################################

#     # #####################################################
#     in_index = df_jobs_anal.index.isin(
#         [(compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]).any()
#     if in_index:
#         row_anal_i = df_jobs_anal.loc[
#             compenv_i, slab_id_i, ads_i, active_site_i, att_num_i]
#         # #################################################
#         job_completely_done_i = row_anal_i.job_completely_done
#         # #################################################
#     else:
#         continue




#     path_i = os.path.join(os.environ["PROJ_irox_oer"], path_rel_to_proj)

#     # #####################################################
#     if job_completely_done_i:

#         # #####################################################
#         # Check that the directory exists
#         my_file = Path(path_i)
#         dir_exists = False
#         if my_file.is_dir():
#             dir_exists = True

#         # #####################################################
#         # Check if .dft_clean file is present
#         dft_clean_file_path = os.path.join(path_i, ".dft_clean")
#         my_file = Path(dft_clean_file_path)
#         dft_clean_already_exists = False
#         if my_file.is_file():
#             dft_clean_already_exists = True

#         # #####################################################
#         if dir_exists:
#             # Creating .dft_clean file
#             if not dft_clean_already_exists:
#                 with open(dft_clean_file_path, "w") as file:
#                     file.write("")

#         # #####################################################
#         # Remove directory
#         if dir_exists and dft_clean_already_exists:
#             print("Removing: ", path_i, sep="")
#             shutil.rmtree(path_i)

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # row_anal_i = df_jobs_anal.loc[
# #     compenv_i, slab_id_i, ads_i, active_site_i, att_num_i]

# in_index = df_jobs_anal.index.isin(
#     [(compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]).any()
