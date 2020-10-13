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

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import json
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd

from ase import io

# # from tqdm import tqdm
from tqdm.notebook import tqdm

# #########################################################
from methods import get_df_slabs_to_run
from methods import (
    get_df_slab,
    get_df_jobs,
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_active_sites,
    )
# -

# # Script Inputs

verbose = True

# # Read Data

# +
# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_anal = get_df_jobs_anal()

# #########################################################
df_slabs_to_run = get_df_slabs_to_run()
# -

# # Setup

# +
directory = "out_data/o_slabs"
if not os.path.exists(directory):
    os.makedirs(directory)

compenv = os.environ["COMPENV"]

# +
# #########################################################
df_jobs_anal_completed = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
var = "o"
df_jobs_anal_completed = df_jobs_anal_completed.query('ads == @var')
# -

# # Writting *O slabs to file

# #########################################################
for name_i, row_i in df_jobs_anal_completed.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    # #####################################################

    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i[df_jobs_i.job_id == job_id_max_i]
    row_jobs_i = row_jobs_i.iloc[0]
    # #####################################################
    att_num_i = row_jobs_i.att_num
    # #####################################################

    # #####################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_i = df_jobs_data_i[df_jobs_data_i.job_id == job_id_max_i]
    row_data_i = row_data_i.iloc[0]
    # #####################################################
    slab_i = row_data_i.final_atoms
    # #####################################################

    file_name_i = os.path.join(
        "out_data/o_slabs",
        compenv_i + "_" + slab_id_i + "_" + str(att_num_i).zfill(2) + ".cif")
    slab_i.write(file_name_i) 

print(df_jobs_anal_completed.shape)
print(df_slabs_to_run.shape)

# +
# #########################################################
df_jobs_anal_completed_index = df_jobs_anal_completed.index.to_frame()

compenv__slab_id__att_num__tuple = tuple(zip(
    df_jobs_anal_completed_index.compenv,
    df_jobs_anal_completed_index.slab_id,
    df_jobs_anal_completed_index.att_num,
    ))

df_jobs_anal_completed["compenv__slab_id__att_num"] = compenv__slab_id__att_num__tuple
df_jobs_anal_completed = df_jobs_anal_completed.set_index("compenv__slab_id__att_num", drop=False)

# #########################################################
compenv__slab_id__att_num__tuple = tuple(zip(
    df_slabs_to_run.compenv,
    df_slabs_to_run.slab_id,
    df_slabs_to_run.att_num,
    ))

df_slabs_to_run["compenv__slab_id__att_num"] = compenv__slab_id__att_num__tuple
df_slabs_to_run = df_slabs_to_run.set_index("compenv__slab_id__att_num", drop=False)

# #########################################################
df_slabs_to_run_i = df_slabs_to_run.drop(columns=["compenv__slab_id__att_num", ])

# +
set_0 = set(df_jobs_anal_completed.index.tolist())
set_1 = set(df_slabs_to_run_i.index.tolist())

print(len(list(set_1 - set_0)))
print(len(list(set_0 - set_1)))

systems_not_manually_processed = list(set_0 - set_1)

systems_not_proc_index = []
for i in systems_not_manually_processed:
    
    compenv_i = i[0]
    slab_id_i = i[1]
    att_num_i = i[2]

    active_site_i = "NaN"
    ads_i = "o"

    index_i = (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)
    systems_not_proc_index.append(index_i)

# #########################################################
df_not_proc = df_jobs_anal.loc[systems_not_proc_index]
# -

# # Write systems not manually processed to file

# +
df_not_proc_index = df_not_proc.index.to_frame()

df_i = df_not_proc_index[["compenv", "slab_id", "att_num", ]]

df_i.to_csv("not_processed.csv", index=False, header=False)
# -

df_not_proc_index

# + active=""
#
#
#
#
# -

assert False

# +
# ('slac', 'nakerafi_91', 'o', 'NaN', 1)

df_slabs_to_run_i[    
    (df_slabs_to_run_i.compenv == "slac") & \
    (df_slabs_to_run_i.slab_id == "nakerafi_91") & \
    [True for i in range(len(df_slabs_to_run_i))]
    ]
# -

df_anal_index = df_jobs_anal.index.to_frame()

# +
# df_i = df_anal_index
# df_i[    
#     (df_i.compenv == "slac") & \
#     (df_i.slab_id == "nakerafi_91") & \
#     [True for i in range(len(df_i))]
#     ].index.tolist()

# df_jobs_anal.loc[
#     ('slac', 'nakerafi_91', 'o', 'NaN', 1)
#     ]
