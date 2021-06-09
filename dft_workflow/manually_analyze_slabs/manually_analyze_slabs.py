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

# # Writing new completed *O slabs to file for analysis
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import shutil

import numpy as np

# # #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_anal,
    get_df_jobs_data,
    )
from methods import get_df_jobs_paths
from methods import get_df_slabs_to_run
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_anal = get_df_jobs_anal()

# #########################################################
df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_run = df_slabs_to_run.set_index(
    ["compenv", "slab_id", "att_num", ], drop=False)

# #########################################################
df_jobs_paths = get_df_jobs_paths()
# -

# ### Filtering down to `oer_adsorbate` jobs

df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal = df_jobs_anal.droplevel(level=0)

# ### Setup

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/manually_analyze_slabs",
    "out_data/o_slabs"
    )

from pathlib import Path
my_file = Path(directory)
if my_file.is_dir():
    try:
        shutil.rmtree(directory)
    except:
        pass

if not os.path.exists(directory):
    os.makedirs(directory)

# +
df_jobs_anal_i = df_jobs_anal

# #########################################################
df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

# #########################################################
var = "o"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

# #########################################################
var = 1
df_jobs_anal_i = df_jobs_anal_i.query('att_num == @var')

# #########################################################
var = "NaN"
df_jobs_anal_i = df_jobs_anal_i.query('active_site == @var')

# #########################################################
df_jobs_anal_i = df_jobs_anal_i.set_index(
    df_jobs_anal_i.index.droplevel(level=[2, 3, ])
    )

# +
not_processed_indices = []
for index_i, row_i in df_jobs_anal_i.iterrows():
    index_man_inspected = index_i in df_slabs_to_run.index
    if not index_man_inspected:
        not_processed_indices.append(index_i)

df_jobs_anal_i = df_jobs_anal_i.loc[
    not_processed_indices
    ]
# -

# ### Writting *O slabs to file

for name_i, row_i in df_jobs_anal_i.iterrows():
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]
    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_max_i]
    # #####################################################
    gdrive_path_i = row_paths_i.gdrive_path
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_max_i]
    # #####################################################
    slab_i = row_data_i.final_atoms
    # #####################################################

    file_name_i = compenv_i + "_" + slab_id_i + "_" + str(att_num_i).zfill(2)
    if verbose:
        print(file_name_i)
        print(
            "$PROJ_irox_oer_gdrive/",
            gdrive_path_i,
            "\n",
            sep="")

    file_name_i = os.path.join(
        directory,
        file_name_i + ".cif")
    slab_i.write(file_name_i) 

        # compenv_i + "_" + slab_id_i + "_" + str(att_num_i).zfill(2) + ".cif")

# ### Write systems not manually processed to file

# +
df_index = df_jobs_anal_i.index.to_frame()


if df_index.shape[0] > 0:
    for i in range(5): print(
        "Number of *O slabs that need to be manually analyzed:",
        df_index.shape[0],
        )

    print("")

path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/manually_analyze_slabs",
    "not_processed.csv")
# print(path_i)
# df_index.to_csv("not_processed.csv", index=False, header=False)
df_index.to_csv(path_i, index=False, header=False)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("manually_analyze_slabs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
