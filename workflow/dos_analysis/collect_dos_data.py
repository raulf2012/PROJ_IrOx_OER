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

# # Run rapidos and bader scripts in job dirs of DOS calculations
# ---

# ### Import modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from methods_dos import PDOS_Plotting, process_PDOS, calc_band_center

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_jobs_anal,
    get_df_jobs_paths,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read data

# +
df_jobs = get_df_jobs()
df_jobs_i = df_jobs

df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_jobs_paths = get_df_jobs_paths()

# + active=""
#
#
# -

# ### Preprocess data objects
#
# Only include `dos_bader` job types

# +
df_jobs_i = df_jobs_i[df_jobs_i.job_type == "dos_bader"]

df_ind = df_jobs_anal_i.index.to_frame()
df_jobs_anal_i = df_jobs_anal_i.loc[
    df_ind[df_ind.job_type == "dos_bader"].index
    ]
# -

# ### Filtering `df_jobs` by `job_completely_done` being True

# +
job_ids_completely_done = df_jobs_anal_i[
    df_jobs_anal_i.job_completely_done == True].job_id_max.tolist()

df_jobs_i = df_jobs_i.loc[
    job_ids_completely_done
    ]
# -

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/dos_analysis",
    "out_data/pdos_data")
if not os.path.exists(directory):
    os.makedirs(directory)


# ### Methods

def write_data_to_file(job_id_i, df_pdos, df_band_centers, directory=None):

    # #########################################################
    file_path_i = os.path.join(
        directory,
        job_id_i + "__df_pdos" + ".pickle")
    with open(file_path_i, "wb") as fle:
        pickle.dump(df_pdos, fle)

    # #########################################################
    file_path_i = os.path.join(
        directory,
        job_id_i + "__df_band_centers" + ".pickle")
    with open(file_path_i, "wb") as fle:
        pickle.dump(df_band_centers, fle)


# ### Collect into groups and make sure that only 1 revision per system

# +
# #########################################################
job_id_list = []
# #########################################################
group_cols = [
    "job_type", "compenv", "slab_id", "ads", "active_site", "att_num", 
    ]
grouped = df_jobs_i.groupby(group_cols)
# #########################################################
for name_i, group_i in grouped:
    # print(name_i)

    # DOS SLURM jobs only need 1 rev to finish, I think, this will check
    assert group_i.shape[0] == 1, "NOT TRUE ANYMORE RIGHT? | I think that there should only be one revision per system here"

    # #####################################################
    row_i = group_i.iloc[0]
    # #####################################################
    job_id_i = row_i.name
    # #####################################################

    job_id_list.append(job_id_i)

# #########################################################
df_jobs_i = df_jobs_i.loc[job_id_list]
# #########################################################
# -

job_ids_all_done = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True].job_id_max.tolist()

# + jupyter={"outputs_hidden": true}
job_ids_to_process = []
# for job_id_i, row_i in df_jobs_tmp.iterrows():
for job_id_i, row_i in df_jobs_i.iterrows():

    if verbose:
        print(20 * "-")
        print(job_id_i)

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_i]
    # #####################################################
    path_i = row_paths_i.gdrive_path
    # #####################################################

    path_full_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        path_i)
    
    finished_file_path_i = os.path.join(
        path_full_i,
        ".FINISHED.new")

    file_path_0 = os.path.join(directory,
        job_id_i + "__df_pdos" + ".pickle")

    file_path_1 = os.path.join(directory,
        job_id_i + "__df_band_centers" + ".pickle")


    my_file_0 = Path(file_path_0)
    my_file_1 = Path(file_path_1)
    my_file = Path(finished_file_path_i)
    if my_file.is_file() and not my_file_0.is_file() and not my_file_1.is_file():
        print(job_id_i, "processing...")
        job_ids_to_process.append(job_id_i)
    else:
        if not my_file.is_file():
            if verbose:
                print("Not finished")
        elif my_file_0.is_file() and my_file_1.is_file():
            if verbose:
                print("System already processed")
# -

df_jobs_i_2 = df_jobs_i.loc[job_ids_to_process]
for job_id_i, row_i in df_jobs_i_2.iterrows():

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_i]
    # #####################################################
    path_i = row_paths_i.gdrive_path
    # #####################################################

    path_full_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        path_i)

    PDOS_i = PDOS_Plotting(data_file_dir=path_full_i)

    # #####################################################
    pdos_data_dict = process_PDOS(
        PDOS_i=PDOS_i,
        )
    # #####################################################
    df_pdos_i = pdos_data_dict["df_xy"]
    df_band_centers_i = pdos_data_dict["df_band_centers"]
    was_processed = pdos_data_dict["was_processed"]
    # #####################################################

    if was_processed:
        df_band_centers_i.insert(0, "system", job_id_i)

        write_data_to_file(
            job_id_i,
            df_pdos_i,
            df_band_centers_i,
            directory=directory,
            )
    else:
        print("Was not processed, can't write data to file")

# +
from methods import read_pdos_data

df_pdos_i, df_band_centers_i = read_pdos_data(job_id_i)
# -

df_pdos_i.iloc[0:3]

df_band_centers_i.iloc[0:3]

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_dos_data.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
