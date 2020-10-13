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
    get_df_slabs_oh,
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

df_slabs_oh = get_df_slabs_oh()
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
# -

# # Checking for multiple *O calcs, need to make code more robust later

# +
var = "o"
df_jobs_anal_o = df_jobs_anal.query('ads == @var')

# #########################################################
indices_to_remove = []
# #########################################################
group_cols = ["compenv", "slab_id", "ads", ]
grouped = df_jobs_anal_o.groupby(group_cols)
for name, group in grouped:

    num_rows = group.shape[0]
    # print(num_rows)
    if num_rows > 1:
        print(name)
        print("")
        # print(num_rows)
        print("COMBAK CHECK THIS")
        print("This was made when there was only 1 *O calc, make sure it's not creating new *OH jobs after running more *O calcs")
        # assert False

        # group[group.att_num != 1]
        group_index = group.index.to_frame()
        group_index_i = group_index[group_index.att_num != 1]

        indices_to_remove.extend(
            group_index_i.index.tolist()
            )

# +
df_jobs_anal_completed = df_jobs_anal[df_jobs_anal.job_completely_done == True]

df_jobs_anal_completed = df_jobs_anal_completed.drop(labels=indices_to_remove)

print("df_jobs_anal_completed.shape:", df_jobs_anal_completed.shape)
print("df_jobs_anal_completed.index.to_frame().shape:",
      df_jobs_anal_completed.index.to_frame().shape)

df_jobs_anal_completed = pd.concat([
    df_jobs_anal_completed.index.to_frame(),
    df_jobs_anal_completed,
    ], axis=1)

df_jobs_anal_completed = df_jobs_anal_completed[
    ["compenv", "slab_id", "job_id_max", "att_num", ]]

# For the purpose of picking preparing *OH jobs we only need completed 
var = "o"
df_jobs_anal_completed = df_jobs_anal_completed.query('ads == @var')

# +
# #########################################################
compenv__slab_id__att_num__tuple = tuple(zip(
    df_jobs_anal_completed.compenv,
    df_jobs_anal_completed.slab_id,
    df_jobs_anal_completed.att_num,
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
shared_indices = df_jobs_anal_completed.index.intersection(df_slabs_to_run.index)
df_i = pd.concat([
    df_slabs_to_run.loc[shared_indices].status,
    df_jobs_anal_completed.loc[shared_indices],
    ], axis=1)

# +
df_i = df_i[
    (df_i.status == "ok")
    # (df_i.status != "bad")
    ]

ind = MultiIndex.from_tuples(
    df_i.index, sortorder=None,
    names=["compenv", "slab_id", "att_num", ])
df_i = df_i.set_index(ind)

# #########################################################
df_i = df_i.sort_index()

# +
# #########################################################
print("TEMP TEMP TEMP")
print("Remove this to do other *OH systems on NERSC")

# df_i = df_i[
#     (df_i.compenv == "nersc") & \
#     (df_i.slab_id == "vuraruna_65") & \
#     [True for i in range(len(df_i))]
#     ]

# compenv: nersc | slab_id: kalisule_45 | att_num: 1
df_i = df_i[
    (df_i.compenv == "nersc") & \
    (df_i.slab_id == "kalisule_45") & \
    [True for i in range(len(df_i))]
    ]
# #########################################################

# +
data_dict_list = []
for name_i, row_i in df_i.iterrows():

    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    att_num_i = name_i[2]

    print(40 * "=")
    print("compenv:", compenv_i, "|", "slab_id:", slab_id_i, "|", "att_num:", att_num_i)

    # #####################################################
    compenv_i = row_i.compenv
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i[df_jobs_i.job_id == job_id_max_i]
    row_jobs_i = row_jobs_i.iloc[0]
    # #####################################################
    # att_num_i = row_jobs_i.att_num
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

    # print("TEMP manually defining active sites")
    # active_sites_unique_i = [62, 67, 68, 71, 73]
    # active_sites_unique_i = [62, ]

    for active_site_j in active_sites_unique_i:

        df_slabs_oh_i = df_slabs_oh.loc[(compenv_i, slab_id_i, "o", active_site_j, att_num_i)]
        for att_num_oh_k, row_k in df_slabs_oh_i.iterrows():
            data_dict_i = dict()

            slab_oh_k = row_k.slab_oh
            num_atoms_k = slab_oh_k.get_global_number_of_atoms()

            # #####################################################
            # attempt = 1
            rev = 1
            path_i = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                "dft_workflow/run_slabs/run_oh_covered",
                "out_data/dft_jobs",
                compenv_i,
                bulk_id_i, facet_i,
                "oh",
                "active_site__" + str(active_site_j).zfill(2),
                str(att_num_oh_k).zfill(2) + "_attempt",  # Attempt
                "_" + str(rev).zfill(2),  # Revision
                )

            # print("  active_site_j:", active_site_j)
            # print("    att_num:", att_num_oh_k)
            # print(active_site_j, att_num_oh_k)
            # print("path_i:", path_i)
            print(path_i)

            if os.path.exists(path_i):
                tmp = 42
                # print("Already exists")
                # print("DISJFIDSI")

            if not os.path.exists(path_i):
                os.makedirs(path_i)

                # #################################################
                # Copy dft script to job folder
                # #################################################
                copyfile(
                    os.path.join(
                        os.environ["PROJ_irox_oer"],
                        "dft_workflow/dft_scripts/slab_dft.py"),
                    os.path.join(
                        path_i,
                        "model.py"),
                    )


                # #################################################
                # Copy atoms object to job folder
                slab_oh_k.write(
                    os.path.join(path_i, "init.traj")
                    )

                # #################################################
                data_dict_i["compenv"] = compenv_i
                data_dict_i["slab_id"] = slab_id_i
                data_dict_i["bulk_id"] = bulk_id_i
                data_dict_i["att_num"] = att_num_i
                data_dict_i["rev_num"] = rev
                data_dict_i["active_site"] = active_site_j
                data_dict_i["facet"] = facet_i
                data_dict_i["slab_oh"] = slab_oh_k
                data_dict_i["num_atoms"] = num_atoms_k
                data_dict_i["path_i"] = path_i
                # #################################################
                data_dict_list.append(data_dict_i)
                # #################################################


# #########################################################
df_jobs_new = pd.DataFrame(data_dict_list)

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
    # ####################################################
    dft_params_i = get_job_spec_dft_params(
        # compenv=compenv,
        compenv=compenv_i,
        slac_sub_queue=slac_sub_queue_i,
        )
    dft_params_i["ispin"] = 2

    # #####################################################
    with open(os.path.join(path_i, "dft-params.json"), "w+") as fle:
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

# + jupyter={"source_hidden": true}
# df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
# row_jobs_i = df_jobs_i[df_jobs_i.job_id == job_id_max_i]

# # row_jobs_i = 
# # df_jobs_i[df_jobs_i.job_id == job_id_max_i]

# # row_jobs_i = row_jobs_i.iloc[0]
# # row_jobs_i

# # job_id_max_i
# # df_jobs_i

# + jupyter={"source_hidden": true}
# df_jobs_i = 
# df_jobs[df_jobs.compenv == compenv_i]
# compenv_i

# + jupyter={"source_hidden": true}
# # #########################################################
# # Only selecting rows from current compenv, jobs will be continued on same cluster that O* was run in
# if compenv == "wsl":
#     compenv = "slac"

# df_i = df_i.loc[compenv]

# + jupyter={"source_hidden": true}
# print(compenv_i)
# print("")

# # dft_params_i = 
# get_job_spec_dft_params(
#     compenv=compenv_i,
#     slac_sub_queue=slac_sub_queue_i,
#     )

# + jupyter={"source_hidden": true}
# df_dft_params
