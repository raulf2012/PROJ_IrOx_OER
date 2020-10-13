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

import copy
import json
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd
from pandas import MultiIndex

from ase import io

# # from tqdm import tqdm
from tqdm.notebook import tqdm

# #########################################################
from methods import (
    get_df_slab,
    get_df_slabs_to_run,
    get_df_jobs,
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_jobs_paths,
    get_df_active_sites,
    get_df_atoms_sorted_ind,
    )

# #########################################################
from dft_workflow_methods import get_job_spec_dft_params
# -

# # Script Inputs

# Slac queue to submit to
slac_sub_queue_i = "suncat2"  # 'suncat', 'suncat2', 'suncat3'

# # Read Data

# + jupyter={"source_hidden": true}
# #########################################################
df_slab = get_df_slab()
df_slab = df_slab.set_index("slab_id")

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_anal = get_df_jobs_anal()

# #########################################################
df_active_sites = get_df_active_sites()

# #########################################################
df_slabs_to_run = get_df_slabs_to_run()

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# -

# # Setup

# +
directory = "out_data/dft_jobs"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "__temp__"
if not os.path.exists(directory):
    os.makedirs(directory)

compenv = os.environ["COMPENV"]

# +
# #########################################################
var = "o"
df_jobs_anal = df_jobs_anal.query('ads == @var')

# #########################################################
df_jobs_anal_completed = df_jobs_anal[df_jobs_anal.job_completely_done == True]
# df_jobs_anal_completed = df_jobs_anal_completed[
#     ["compenv", "slab_id", "job_id_max_i", "att_num", ]]

df_jobs_anal_completed = df_jobs_anal_completed[["job_id_max", ]]

# +
df_index_i = df_jobs_anal_completed.index.to_frame()

compenv__slab_id__att_num__tuple = tuple(zip(
    df_index_i.compenv,
    df_index_i.slab_id,
    df_index_i.att_num,
    ))

# +
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
df_i = pd.concat([
    df_slabs_to_run.status,
    df_jobs_anal_completed,
    ], axis=1)

df_i = df_i.sort_index()

df_i = df_i[
    (df_i.status == "ok")
    ]

ind = MultiIndex.from_tuples(
    df_i.index, sortorder=None,
    names=["compenv", "slab_id", "att_num", ])
df_i = df_i.set_index(ind)

# +
# # #########################################################
# print("TEMP TEMP TEMP")
# # #########################################################

# compenv_i, slab_id_i, att_num_i  = ('slac', 'garituna_73', 1)
# df_i = df_i.loc[
#     [(compenv_i, slab_id_i, att_num_i)]
#     ]

# +
verbose_local = False

# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_i.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i[df_jobs_i.job_id == job_id_max_i]
    row_jobs_i = row_jobs_i.iloc[0]
    # #####################################################
    att_num_i = row_jobs_i.att_num
    bulk_id_i = row_jobs_i.bulk_id
    facet_i = row_jobs_i.facet
    # #####################################################

    # #####################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_i = df_jobs_data_i[df_jobs_data_i.job_id == job_id_max_i]
    row_data_i = row_data_i.iloc[0]
    # #####################################################
    slab_i = row_data_i.final_atoms
    # #####################################################

    # #####################################################
    row_active_site_i = df_active_sites[df_active_sites.slab_id == slab_id_i]
    row_active_site_i = row_active_site_i.iloc[0]
    # #####################################################
    active_sites_unique_i = row_active_site_i.active_sites_unique
    num_active_sites_unique_i = row_active_site_i.num_active_sites_unique
    # #####################################################

    # #####################################################
    # row_atoms_sorted_i = df_atoms_sorted_ind.loc[(compenv_i, slab_id_i, att_num_i)]
    index_atoms_sorted_i = (compenv_i, slab_id_i, "o", "NaN", att_num_i, )
    row_atoms_sorted_i = df_atoms_sorted_ind.loc[index_atoms_sorted_i]
    # #####################################################
    atoms_sorted_i = row_atoms_sorted_i.atoms_sorted_good
    # #####################################################



    for active_site_j in active_sites_unique_i:
        data_dict_i = dict()

        if verbose_local:
            print(40 * "=")
            print("active_site_j:", active_site_j)

        # #####################################################
        rev = 1
        path_i = os.path.join(
            "out_data/dft_jobs", 
            compenv_i, bulk_id_i, facet_i,
            "bare", "active_site__" + str(active_site_j).zfill(2),
            str(att_num_i).zfill(2) + "_attempt",  # Attempt
            "_" + str(rev).zfill(2),  # Revision
            )

        root_frag = "dft_workflow/run_slabs/run_bare_oh_covered"
        path_full = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            root_frag,
            path_i)

        if not os.path.exists(path_full):
            print(name_i, "|", active_site_j, "|", path_full)
            os.makedirs(path_full)

            # #############################################
            # Copy dft script to job folder
            copyfile(
                os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
                os.path.join(path_full, "model.py"),
                )

            # #############################################
            # Removing atom to create 
            atoms_sorted_cpy_i = copy.deepcopy(atoms_sorted_i)
            atoms_sorted_cpy_i.pop(i=active_site_j)
            slab_bare_i = atoms_sorted_cpy_i

            # #############################################
            # Copy atoms object to job folder
            slab_bare_i.write(
                os.path.join(path_full, "init.traj")
                )
            num_atoms_i = slab_bare_i.get_global_number_of_atoms()

            # #############################################
            data_dict_i["compenv"] = compenv_i
            data_dict_i["slab_id"] = slab_id_i
            data_dict_i["bulk_id"] = bulk_id_i
            data_dict_i["att_num"] = att_num_i
            data_dict_i["rev_num"] = rev
            data_dict_i["active_site"] = active_site_j
            data_dict_i["facet"] = facet_i
            data_dict_i["slab_bare"] = slab_bare_i
            data_dict_i["num_atoms"] = num_atoms_i
            data_dict_i["path_i"] = path_i
            data_dict_i["path_full"] = path_full
            # #############################################
            data_dict_list.append(data_dict_i)
            # #############################################


# #########################################################
df_jobs_new = pd.DataFrame(data_dict_list)
# df_jobs_new = df_jobs_new.set_index("slab_id")

# +
# Create empty dataframe with columns if dataframe is empty
if df_jobs_new.shape[0] == 0:
    df_jobs_new = pd.DataFrame(
        columns=["compenv", "slab_id", "att_num", "active_site", ])

# df_jobs_new

# +
df_jobs_new["compenv__slab_id__att_num__active_site"] = list(zip(
    df_jobs_new.compenv,
    df_jobs_new.slab_id,
    df_jobs_new.att_num,
    df_jobs_new.active_site))

df_jobs_new = df_jobs_new.set_index("compenv__slab_id__att_num__active_site", drop=False)

# +
data_dict_list = []
for i_cnt, row_i in df_jobs_new.iterrows():
    data_dict_i = dict()
    # #####################################################
    compenv__slab_id__att_num__active_site_i = row_i.name
    compenv_i, slab_id_i, att_num_i, active_site_i = row_i.name

    compenv_i = row_i.compenv
    num_atoms = row_i.num_atoms
    path_i = row_i.path_i
    path_full = row_i.path_full
    # ####################################################
    dft_params_i = get_job_spec_dft_params(
        compenv=compenv_i,
        slac_sub_queue=slac_sub_queue_i,
        )
    dft_params_i["ispin"] = 2

    # #####################################################
    with open(os.path.join(path_full, "dft-params.json"), "w+") as fle:
        json.dump(dft_params_i, fle, indent=2, skipkeys=True)


    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv__slab_id__att_num__active_site"] = \
        compenv__slab_id__att_num__active_site_i
    data_dict_i["dft_params"] = dft_params_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_dft_params = pd.DataFrame(data_dict_list)

# Create empty dataframe with columns if dataframe is empty
if df_dft_params.shape[0] == 0:
    df_dft_params = pd.DataFrame(columns=["compenv", "slab_id", "att_num", "active_site", ])

keys = ["compenv", "slab_id", "att_num", "active_site"]
df_dft_params = df_dft_params.set_index(keys, drop=False)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("setup_dft.ipynb")
print(20 * "# # ")
# assert False
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# compenv_i

# #     dft_params_i = 
# get_job_spec_dft_params(
#     compenv=compenv_i,
#     slac_sub_queue=slac_sub_queue_i,
#     )
