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

# # Analyze *OH slab job sets
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import numpy as np
import pandas as pd

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_atoms_sorted_ind,
    get_df_features,
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

# ### Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_jobs_data = get_df_jobs_data()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_features = get_df_features()
# -

df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]

# + active=""
#
#

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
# Dropping rows that failed atoms sort, now it's just one job that blew up 
# job_id = "dubegupi_27"
df_failed_to_sort = df_atoms_sorted_ind[
    df_atoms_sorted_ind.failed_to_sort == True]
df_jobs_anal_i = df_jobs_anal_i.drop(labels=df_failed_to_sort.index)

# #########################################################
df_index_i = df_jobs_anal_i.index.to_frame()

# df_index_i = df_index_i[df_index_i.ads != "o"]
df_index_i = df_index_i[df_index_i.ads == "oh"]

df_jobs_anal_i = df_jobs_anal_i.loc[
    df_index_i.index 
    ]


# +
def method(row_i):
    job_id_max_i = row_i.job_id_max

    # #########################################################
    row_feat_i = df_features[df_features["data"]["job_id_max"] == job_id_max_i]
    if row_feat_i.shape[0] > 0:
        row_feat_i = row_feat_i.iloc[0]
        # #########################################################
        num_missing_Os_i = row_feat_i.data.num_missing_Os
        # #########################################################
    else:
        num_missing_Os_i = None

    return(num_missing_Os_i)

df_jobs_anal_i["num_missing_Os"] = df_jobs_anal_i.apply(method, axis=1)
# -

# ### Main Loop

# +
# #########################################################
data_dict_list = []
# #########################################################
grouped = df_jobs_anal_i.groupby(["compenv", "slab_id", "active_site", ])
for name, group in grouped:

# for i in range(1):
#     name =  ('slac', 'dotivela_46', 26.0, )
#     group = grouped.get_group(name)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    active_site_i = name[2]
    # #####################################################


    # TEMP
    # any_nan_in_missing_O_col = any(group.num_missing_Os.isna())
    # if any_nan_in_missing_O_col:
    #     print("There are NaN in missing_Os col")
    #     print("name:", name)
    #     continue

    # if any_nan_in_missing_O_col:
    #     print("This shouldn't get printed if the prev 'continue' statement is working")

    job_ids_w_missing_Os = group[group.num_missing_Os > 0].job_id_max.tolist()


    # Group of rows that have no missing O bonds
    group_2 = group.drop(
        labels=group[group.num_missing_Os > 0].index
        )

    all_jobs_bad = False
    if group_2.shape[0] == 0:
        all_jobs_bad = True

    # #####################################################
    df_anal_ind = df_jobs_anal.index.to_frame()
    df = df_anal_ind
    df = df[
        (df["compenv"] == compenv_i) &
        (df["slab_id"] == slab_id_i) &
        (df["ads"] == "oh") &
        (df["active_site"] == active_site_i) &
        [True for i in range(len(df))]
        ]
    df_anal_ind_i = df
    # #####################################################
    att_nums_all = df_anal_ind_i.att_num.unique().tolist()
    # #####################################################


    # #####################################################
    # Checking if all *OH slabs are finished, should all be done before making decisions
    group_index_i = group.index.to_frame()
    att_nums_i = group_index_i.att_num.unique()

    all_oh_attempts_done = np.array_equal(att_nums_all, att_nums_i)

    job_ids_sorted_energy = []
    job_id_most_stable = None
    if group_2.shape[0] > 0:
        # #####################################################
        df_jobs_data_i = df_jobs_data.loc[group_2.job_id_max]
        df_jobs_data_i = df_jobs_data_i.sort_values("pot_e")
        # #####################################################
        job_ids_sorted_energy = df_jobs_data_i.job_id.tolist()
        job_id_most_stable = job_ids_sorted_energy[0]
        # #####################################################



    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    # #####################################################
    data_dict_i["all_oh_attempts_done"] = all_oh_attempts_done
    data_dict_i["job_id_most_stable"] = job_id_most_stable
    data_dict_i["all_jobs_bad"] = all_jobs_bad
    data_dict_i["job_ids_sorted_energy"] = job_ids_sorted_energy
    data_dict_i["job_ids_w_missing_Os"] = job_ids_w_missing_Os
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_jobs_oh_anal = pd.DataFrame(data_dict_list)
# df_jobs_oh_anal.iloc[0:2]
# -

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/analyze_oh_jobs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(directory, "df_jobs_oh_anal.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_jobs_oh_anal, fle)
# #########################################################

# +
from methods import get_df_jobs_oh_anal

df_jobs_oh_anal_tmp = get_df_jobs_oh_anal()
df_jobs_oh_anal_tmp.iloc[0:2]
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("anal_oh_slabs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
