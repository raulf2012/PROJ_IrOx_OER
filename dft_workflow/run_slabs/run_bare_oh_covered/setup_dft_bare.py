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

# # Setup bare (*) slabs after the first *O slab is completed
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
# -

# # Setup

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/run_bare_oh_covered",
    "out_data/dft_jobs")
if not os.path.exists(directory):
    os.makedirs(directory)

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/run_bare_oh_covered",
    "out_data/__temp__")
if not os.path.exists(directory):
    os.makedirs(directory)

compenv = os.environ["COMPENV"]

# +
from methods_run_slabs import get_systems_to_run_bare_and_oh

indices_to_process = get_systems_to_run_bare_and_oh()
df_jobs_anal_i = df_jobs_anal.loc[indices_to_process]

df_jobs_anal_i = df_jobs_anal_i.set_index(
    df_jobs_anal_i.index.droplevel(level=[2, 3, ])
    )
df_jobs_anal_i.head()

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

# +
# print(20 * "TEMP | ")

# compenv_target = "slac"
# slab_id_target = "fodopilu_17"
# att_num_target = 1


# df_ind_i = df_i.index.to_frame()
# df = df_ind_i
# df = df[
#     (df["compenv"] == compenv_target) &
#     (df["slab_id"] == slab_id_target) &
#     (df["att_num"] == att_num_target) &
#     [True for i in range(len(df))]
#     ]

# df_i = df_i.loc[
#     df.index
#     ]

# +
# df_i

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
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    row_jobs_i = df_jobs.loc[job_id_max_i]
    # #####################################################
    bulk_id_i = row_jobs_i.bulk_id
    facet_i = row_jobs_i.facet
    # #####################################################

    # #####################################################
    row_active_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_active_sites_i.active_sites_unique
    # #####################################################


    for active_site_j in active_sites_unique_i:
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

        # print(path_full)
        path_exists = False
        if os.path.exists(path_full):
            path_exists = True

        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["compenv"] = compenv_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["att_num"] = att_num_i
        data_dict_i["active_site"] = active_site_j
        data_dict_i["path_short"] = path_i
        data_dict_i["path_full"] = path_full
        data_dict_i["path_exists"] = path_exists
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_to_setup = pd.DataFrame(data_dict_list)
df_to_setup = df_to_setup.set_index(
    ["compenv", "slab_id", "att_num", "active_site", ], drop=False)

df_to_setup_i = df_to_setup[df_to_setup.path_exists == False]

# +
# df_to_setup_i
# -

print(
    "Number of new bare (*) jobs setup:",
    df_to_setup_i.shape[0]
    )

# +
# assert False

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_to_setup_i.iterrows():
    # #####################################################
    name_dict_i = dict(zip(list(df_to_setup_i.index.names), name_i))
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]
    active_site_i = name_i[3]
    # #####################################################
    path_full_i = row_i.path_full
    # #####################################################

    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[
        (compenv_i, slab_id_i, "o", "NaN", att_num_i, )]
    # #####################################################
    atoms_sorted_i = row_atoms_i.atoms_sorted_good
    # #####################################################

    if verbose:
        print(name_i, "|", active_site_i, "|", path_full_i)
        print(path_full_i)

    os.makedirs(path_full_i)

    # #####################################################
    # Copy dft script to job folder
    copyfile(
        os.path.join(os.environ["PROJ_irox_oer"], "dft_workflow/dft_scripts/slab_dft.py"),
        os.path.join(path_full_i, "model.py"),
        )

    # #####################################################
    # Removing atom to create 
    atoms_sorted_cpy_i = copy.deepcopy(atoms_sorted_i)
    atoms_sorted_cpy_i.pop(i=active_site_i)
    slab_bare_i = atoms_sorted_cpy_i

    # #####################################################
    # Copy atoms object to job folder
    slab_bare_i.write(
        os.path.join(path_full_i, "init.traj"))
    slab_bare_i.write(
        os.path.join(path_full_i, "init.cif"))
    num_atoms_i = slab_bare_i.get_global_number_of_atoms()

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    data_dict_i["bulk_id"] = bulk_id_i
    data_dict_i["rev_num"] = rev
    data_dict_i["facet"] = facet_i
    data_dict_i["slab_bare"] = slab_bare_i
    data_dict_i["num_atoms"] = num_atoms_i
    data_dict_i["path_i"] = path_i
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
# assert False
# -

# # Writing DFT parameters to file

# +
# #########################################################
data_dict_list = []
# #########################################################
for i_cnt, row_i in df_jobs_new.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    compenv_i = row_i.compenv
    num_atoms = row_i.num_atoms
    path_i = row_i.path_i
    path_full_i = row_i.path_full
    # ####################################################

    dft_params_i = get_job_spec_dft_params(
        compenv=compenv_i,
        slac_sub_queue=slac_sub_queue_i,
        )
    dft_params_i["ispin"] = 2

    # #####################################################
    if verbose:
        print(path_full_i)

    with open(os.path.join(path_full_i, "dft-params.json"), "w+") as fle:
        json.dump(dft_params_i, fle, indent=2, skipkeys=True)


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
    df_dft_params = pd.DataFrame(columns=["compenv", "slab_id", "att_num", "active_site", ])

keys = ["compenv", "slab_id", "att_num", "active_site"]
df_dft_params = df_dft_params.set_index(keys, drop=False)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("setup_dft_bare.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # def get_systems_to_run_bare_and_oh():
# """
# Takes df_jobs_anal and filter to:
#   * only *O slabs
#   * slabs that have 'NaN' in the active site (not *O that are run from  *OH, which have an active site value)
#   * Only completed slabs
#   * Only the first att_num, so that you don't start new sets of *OH and bare jobs from rerun *O jobs
# """
# #| - get_systems_to_run_bare_and_oh
# df_jobs_anal = get_df_jobs_anal()


# df_jobs_anal_i = df_jobs_anal

# var = "o"
# df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

# var = "NaN"
# df_jobs_anal_i = df_jobs_anal_i.query('active_site == @var')

# df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]




# # #########################################################
# indices_to_remove = []
# # #########################################################
# group_cols = ["compenv", "slab_id", "ads", ]
# grouped = df_jobs_anal_i.groupby(group_cols)
# for name, group in grouped:

#     num_rows = group.shape[0]
#     if num_rows > 1:
#         # print(name)
#         # print("")
#         # # print(num_rows)
#         # print("COMBAK CHECK THIS")
#         # print("This was made when there was only 1 *O calc, make sure it's not creating new *OH jobs after running more *O calcs")

#         group_index = group.index.to_frame()
#         group_index_i = group_index[group_index.att_num != 1]

#         indices_to_remove.extend(
#             group_index_i.index.tolist()
#             )

# df_jobs_anal_i = df_jobs_anal_i.drop(index=indices_to_remove)

# indices_out = df_jobs_anal_i.index.tolist()

# # return(indices_out)
# #__|

# + jupyter={"source_hidden": true}
# # TEMP
# print(20 * "TEMP | ")
# print("")

# compenv_target = "slac"
# slab_id_target = "fodopilu_17"
# # ads_target = "bare"
# ads_target = "o"
# # active_site_target = 24.
# active_site_target = "NaN"
# att_num_target = 1

# for ind_i in indices_to_process:
# # for ind_i in indices_out:
# # #########################################################
#     compenv_i = ind_i[0]
#     slab_id_i = ind_i[1]
#     ads_i = ind_i[2]
#     active_site_i = ind_i[3]
#     att_num_i = ind_i[4]
#     # #########################################################

#     compenv_comp = compenv_i == compenv_target
#     slab_id_comp = slab_id_i == slab_id_target

#     if compenv_comp and slab_id_comp:
#         print(ind_i)

# + jupyter={"source_hidden": true}
# df_i = df_jobs_anal

# df_ind_i = df_i.index.to_frame()


# compenv_target = "slac"
# slab_id_target = "fodopilu_17"
# # ads_target = "bare"
# ads_target = "o"
# # active_site_target = 24.
# active_site_target = "NaN"
# att_num_target = 1

# df = df_ind_i
# df = df[
#     (df["compenv"] == compenv_target) &
#     (df["slab_id"] == slab_id_target) &
#     (df["ads"] == ads_target) &
#     (df["active_site"] == active_site_target) &
#     # (df["att_num"] == att_num_target) &
#     [True for i in range(len(df))]
#     ]

# # df_ind_i

# df_i.loc[
#     df.index
#     ]
