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

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

sys.path.insert(0, "..")

import copy
import json
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd
from pandas import MultiIndex

from ase import io

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
    get_df_slabs_oh,
    )

# #########################################################
from dft_workflow_methods import get_job_spec_dft_params
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Script Inputs

# Slac queue to submit to
slac_sub_queue_i = "suncat3"  # 'suncat', 'suncat2', 'suncat3'

# # Read Data

# +
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
df_slabs_to_run = df_slabs_to_run.set_index(["compenv", "slab_id", "att_num"], drop=False)

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# #########################################################
df_slabs_oh = get_df_slabs_oh()
# -

# # Setup

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/run_oh_covered",
    "out_data/dft_jobs")
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/run_oh_covered",
    "out_data/__temp__")
if not os.path.exists(directory):
    os.makedirs(directory)

compenv = os.environ["COMPENV"]
# -

# # Filtering `df_jobs_anal`

# +
from methods_run_slabs import get_systems_to_run_bare_and_oh

indices_to_process = get_systems_to_run_bare_and_oh()
df_jobs_anal_i = df_jobs_anal.loc[indices_to_process]


# #########################################################
# Removing systems that were marked to be ignored
from methods import get_systems_to_stop_run_indices
indices_to_stop_running = get_systems_to_stop_run_indices(df_jobs_anal=df_jobs_anal)

indices_to_drop = []
for index_i in df_jobs_anal_i.index:
    if index_i in indices_to_stop_running:
        indices_to_drop.append(index_i)

df_jobs_anal_i = df_jobs_anal_i.drop(index=indices_to_drop)


# #########################################################
# Drop redundent indices (adsorbate and active site)
df_jobs_anal_i = df_jobs_anal_i.set_index(
    df_jobs_anal_i.index.droplevel(level=[2, 3, ])
    )


# #########################################################
idx = np.intersect1d(
    df_jobs_anal_i.index,
    df_slabs_to_run.index,
    )
shared_indices = idx

df_i = pd.concat([
    df_slabs_to_run.loc[shared_indices].status,
    df_jobs_anal_i.loc[shared_indices],
    ], axis=1)
df_i = df_i[df_i.status == "ok"]

# df_i.head()

# +
indices_not_in = []
for i in df_jobs_anal_i.index:
    if i not in df_slabs_to_run.index:
        indices_not_in.append(i)

print(
    "Number of systems that haven't been manually approved:",
    len(indices_not_in),
    )

# +
# assert False

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_i.iterrows():
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]
    # #####################################################

    # if verbose:
    #     print(40 * "=")
    #     print("compenv:", compenv_i, "|", "slab_id:", slab_id_i, "|", "att_num:", att_num_i)

    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i[df_jobs_i.job_id == job_id_max_i]
    row_jobs_i = row_jobs_i.iloc[0]
    # #####################################################
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

    # print(len(active_sites_unique_i))
    # print("TEMP")
    # active_sites_unique_i = [24, ]

    for active_site_j in active_sites_unique_i:
        df_slabs_oh_i = df_slabs_oh.loc[(compenv_i, slab_id_i, "o", active_site_j, att_num_i)]
        for att_num_oh_k, row_k in df_slabs_oh_i.iterrows():
            data_dict_i = dict()

            slab_oh_k = row_k.slab_oh
            num_atoms_k = slab_oh_k.get_global_number_of_atoms()

            # #############################################
            # attempt = 1
            rev = 1

            path_i = os.path.join(

                "out_data/dft_jobs",
                compenv_i, bulk_id_i, facet_i,
                "oh",
                "active_site__" + str(active_site_j).zfill(2),
                str(att_num_oh_k).zfill(2) + "_attempt",  # Attempt
                "_" + str(rev).zfill(2),  # Revision
                )

            path_full_i = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                "dft_workflow/run_slabs/run_oh_covered",
                path_i)

            # if verbose:
            #     print(path_full_i )

            path_exists = False
            if os.path.exists(path_full_i):
                path_exists = True


            # #############################################
            data_dict_i = dict()
            # #############################################
            data_dict_i["compenv"] = compenv_i
            data_dict_i["slab_id"] = slab_id_i
            data_dict_i["att_num"] = att_num_oh_k
            data_dict_i["active_site"] = active_site_j
            data_dict_i["path_short"] = path_i
            data_dict_i["path_full"] = path_full_i
            data_dict_i["path_exists"] = path_exists
            data_dict_i["slab_oh"] = slab_oh_k
            # #############################################
            data_dict_list.append(data_dict_i)
            # #############################################

# #########################################################
df_to_setup = pd.DataFrame(data_dict_list)
df_to_setup = df_to_setup.set_index(
    ["compenv", "slab_id", "att_num", "active_site", ], drop=False)

df_to_setup_i = df_to_setup[df_to_setup.path_exists == False]

# +
# df_to_setup_i

# +
# print(222 * "TEMP | ")
# assert False

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_to_setup_i.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]
    active_site_i = name_i[3]
    # #####################################################
    path_full_i = row_i.path_full
    path_short_i = row_i.path_short
    slab_oh_i = row_i.slab_oh
    # #####################################################

    # print(name_i, "|", active_site_i, "|", path_full_i)
    if verbose:
        print(name_i, "|", active_site_i)
        print(path_full_i)

    os.makedirs(path_full_i)

    # #####################################################
    # Copy dft script to job folder
    copyfile(
        os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
        os.path.join(path_full_i, "model.py"),
        )

    # #####################################################
    # Copy atoms object to job folder
    slab_oh_i.write(
        os.path.join(path_full_i, "init.traj")
        )
    slab_oh_i.write(
        os.path.join(path_full_i, "init.cif")
        )
    num_atoms_i = slab_oh_i.get_global_number_of_atoms()

    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["bulk_id"] = bulk_id_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["rev_num"] = rev
    data_dict_i["active_site"] = active_site_j
    data_dict_i["facet"] = facet_i
    data_dict_i["slab_oh"] = slab_oh_i
    data_dict_i["num_atoms"] = num_atoms_i
    data_dict_i["path_short"] = path_short_i
    data_dict_i["path_full"] = path_full_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_jobs_new = pd.DataFrame(data_dict_list)

# Create empty dataframe with columns if dataframe is empty
if df_jobs_new.shape[0] == 0:
    df_jobs_new = pd.DataFrame(
        columns=["compenv", "slab_id", "att_num", "active_site", ])

# +
# #########################################################
data_dict_list = []
# #########################################################
for i_cnt, row_i in df_jobs_new.iterrows():
    # #####################################################
    compenv_i = row_i.compenv
    num_atoms = row_i.num_atoms
    path_full_i = row_i.path_full
    # ####################################################
    dft_params_i = get_job_spec_dft_params(
        compenv=compenv_i,
        slac_sub_queue=slac_sub_queue_i,
        )
    dft_params_i["ispin"] = 2

    # print(path_full_i)

    # #####################################################
    with open(os.path.join(path_full_i, "dft-params.json"), "w+") as fle:
        json.dump(dft_params_i, fle, indent=2, skipkeys=True)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["dft_params"] = dft_params_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_dft_params = pd.DataFrame(data_dict_list)


# Create empty dataframe with columns if dataframe is empty
if df_dft_params.shape[0] == 0:
    tmp = 42
    # df_jobs_new = pd.DataFrame(
    #     columns=["compenv", "slab_id", "att_num", "active_site", ])
else:
    keys = ["compenv", "slab_id", "att_num", "active_site"]
    df_dft_params = df_dft_params.set_index(keys, drop=False)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("setup_dft_oh.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#

# + jupyter={"source_hidden": true}
# slab_ids_phase_2 = df_slab[df_slab.phase == 2].index.tolist()

# df_index_i = df_to_setup_i.index.to_frame()

# for i in df_index_i.slab_id.unique():
#     if i not in slab_ids_phase_2:
#         print("Not there")

# + jupyter={"source_hidden": true}
# # df_dft_params = 
# pd.DataFrame(data_dict_list)

# # keys = ["compenv", "slab_id", "att_num", "active_site"]
# # df_dft_params = df_dft_params.set_index(keys, drop=False)

# + jupyter={"source_hidden": true}
# df_slabs_oh

# + jupyter={"source_hidden": true}
# (compenv_i, slab_id_i, "o", active_site_j, att_num_i, )

# + jupyter={"source_hidden": true}
# # df_slabs_oh_i

# # df_slabs_oh_i = 
# df_slabs_oh.loc[
#     (compenv_i, slab_id_i, "o", active_site_j, att_num_i, )
#     ]

# + jupyter={"source_hidden": true}
# active_sites_unique_i

# + jupyter={"source_hidden": true}
# df_to_setup_i.path_full.tolist()

# df_to_setup_i

# + jupyter={"source_hidden": true}
# # df_jobs_anal[df_jobs_anal.job_id_max == job_id_j]

# df_ind_i = df_jobs_anal.index.to_frame()

# df = df_ind_i
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "likeniri_51") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]

# df_jobs_anal.loc[
#     df.index
#     ]

# + jupyter={"source_hidden": true}
# print(20 * "TEMP | ")

# df_i = df_i.loc[
#     [("slac", "fodopilu_17", 1)]
#     ]

# + jupyter={"source_hidden": true}
# indices_to_process

# + jupyter={"source_hidden": true}
# job_id_j = "job_id_j"

# df_i[df_i.job_id_max == job_id_j]
