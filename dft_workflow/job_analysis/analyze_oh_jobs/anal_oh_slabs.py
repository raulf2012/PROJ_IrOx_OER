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

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import numpy as np
import pandas as pd

# #########################################################
from methods import get_df_jobs
from methods import get_df_jobs_anal
from methods import get_df_jobs_data
# -

# # Read Data

# +
df_jobs = get_df_jobs()

df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

df_index_i = df_jobs_anal_i.index.to_frame()

# df_index_i = df_index_i[df_index_i.ads != "o"]
df_index_i = df_index_i[df_index_i.ads == "oh"]

df_jobs_anal_i = df_jobs_anal_i.loc[
    df_index_i.index 
    ]

# +
# ('slac', 'fagumoha_68', 62.0)

compenv_i = "slac"
slab_id_i = "fagumoha_68"
active_site_i = 62.

# df_jobs_oh_anal_tmp[
#     (df_jobs_oh_anal_tmp.compenv == compenv_i) & \
#     (df_jobs_oh_anal_tmp.slab_id == slab_id_i) & \
#     (df_jobs_oh_anal_tmp.active_site == active_site_i) & \
#     [True for i in range(len(df_jobs_oh_anal_tmp))]
#     ]

# df_jobs_anal_i.loc[(compenv_i, slab_id_i, )]

df_index_i = df_jobs_anal_i.index.to_frame()

df_index_i[
    (df_index_i.compenv == compenv_i) & \
    (df_index_i.slab_id == slab_id_i) & \
    (df_index_i.active_site == active_site_i) & \
    [True for i in range(len(df_index_i))]
    ].index

# #########################################################
# df_index_i = df_jobs_anal.index.to_frame()

# df_index_i[
#     (df_index_i.compenv == compenv_i) & \
#     (df_index_i.slab_id == slab_id_i) & \
# #     (df_index_i.active_site == active_site_i) & \
#     [True for i in range(len(df_index_i))]
#     ]

# +
# print("TEMP")

# # sherlock	kenukami_73	84.0	
# compenv_i =  "sherlock"
# slab_id_i = "kenukami_73"
# active_site_i = 84.0

# df_index_i = df_jobs_anal_i.index.to_frame()
# df_index_i = df_index_i[
#     (df_index_i.compenv == compenv_i) & \
#     (df_index_i.slab_id == slab_id_i) & \
#     (df_index_i.active_site == active_site_i) & \
#     [True for i in range(len(df_index_i))]
#     ]

# df_jobs_anal_i = df_jobs_anal_i.loc[
#     df_index_i.index
#     ]

# +
# #########################################################
data_dict_list = []
# #########################################################
grouped = df_jobs_anal_i.groupby(["compenv", "slab_id", "active_site", ])
for name, group in grouped:
    data_dict_i = dict()
    # print(group.shape)

    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    active_site_i = name[2]
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[
        (df_jobs.compenv == compenv_i) & \
        (df_jobs.slab_id == slab_id_i) & \
        (df_jobs.active_site == active_site_i) & \
        [True for i in range(len(df_jobs))]
        ]
    # #####################################################
    att_nums_all = df_jobs_i.att_num.unique()
    # #####################################################

    # #####################################################
    # Checking if all *OH slabs are finished, should all be done before making decisions
    group_index_i = group.index.to_frame()
    att_nums_i = group_index_i.att_num.unique()
    all_oh_attempts_done = np.array_equal(att_nums_all, att_nums_i)

    # #####################################################
    df_jobs_data_i = df_jobs_data.loc[group.job_id_max]
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
    data_dict_i["job_ids_sorted_energy"] = job_ids_sorted_energy
    data_dict_i["job_id_most_stable"] = job_id_most_stable
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_jobs_oh_anal = pd.DataFrame(data_dict_list)
df_jobs_oh_anal.iloc[0:2]
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

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# all_oh_attempts_done = 
# np.array_equal(att_nums_all, att_nums_i)

# att_nums_i
# att_nums_all

# + jupyter={"source_hidden": true}
# # #########################################################
# import pickle; import os
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "dft_workflow/job_analysis/analyze_oh_jobs",
#     "out_data")
# path_i = os.path.join(directory, "df_jobs_oh_anal.pickle")
# with open(path_i, "rb") as fle:
#     df_jobs_oh_anal = pickle.load(fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# group.job_id_max.tolist()
