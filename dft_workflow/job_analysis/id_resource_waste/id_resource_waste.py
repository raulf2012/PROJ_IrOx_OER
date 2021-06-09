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

# # Identifying job sets that have not progressed and cancel them
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_jobs_anal,
    )
# -

# # Read Data

# +
df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data()

df_jobs_anal = get_df_jobs_anal()

# +
# Removing systems that were marked to be ignored
from methods import get_systems_to_stop_run_indices

indices_to_stop_running = get_systems_to_stop_run_indices(df_jobs_anal=df_jobs_anal)

# df_jobs_anal = df_jobs_anal.drop(index=indices_to_stop_running)
# df_resubmit = df_jobs_anal

# + active=""
#
#
#
# -

# # Main Loop

# +
from misc_modules.pandas_methods import drop_columns

# print(list(df_jobs_anal.columns))

# ['job_id_max', 'timed_out', 'completed', 'brmix_issue', 'job_understandable', 'decision', 'dft_params_new', 'job_completely_done']
cols_to_keep = [
    "job_id_max",
    "job_completely_done",
    ]
df_jobs_anal_i = drop_columns(
    df=df_jobs_anal,
    columns=cols_to_keep,
    keep_or_drop="keep",
    )

# +
# assert False

# +
index_keys = list(df_jobs_anal_i.index.names)

# #########################################################
data_dict_list = []
# #########################################################
for index_i, row_i in df_jobs_anal_i.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    compenv_i, slab_id_i, ads_i, active_site_i, att_num_i = index_i
    # #####################################################
    job_completely_done_i = df_jobs_anal.job_completely_done
    # #####################################################


    index_dict_i = dict(zip(index_keys, index_i))

    df = df_jobs
    df = df[
        (df["compenv"] == compenv_i) &
        (df["slab_id"] == slab_id_i) &
        (df["ads"] == ads_i) &
        (df["active_site"] == active_site_i) &
        (df["att_num"] == att_num_i) &
        [True for i in range(len(df))]
        ]
    df_jobs_i = df

    num_revs_i = df_jobs_i.shape[0]

    # #####################################################
    data_dict_i.update(index_dict_i)
    data_dict_i.update(row_i.to_dict())
    # #####################################################
    data_dict_i["num_revs"] = num_revs_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df = pd.DataFrame(data_dict_list)
df = df.sort_values("num_revs", ascending=False)
df = df.set_index([
    "compenv", "slab_id", "ads", "active_site", "att_num", 
    ], drop=False)
df = df.drop(labels=indices_to_stop_running)
# #########################################################

df.iloc[0:2]

# +
df_i = df[df.job_completely_done == False]
df_i = df_i[df_i.num_revs > 3]

df_i

# +
# df_i.iloc[0:3].index.tolist()

df_i.iloc[0:3].index.tolist()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# keys = list(df_jobs_anal.index.names)
# index_dict_i = dict(zip(keys, index_i))

# + jupyter={"source_hidden": true}
# index_dict_i

# + jupyter={"source_hidden": true}
# data_dict_i

# + jupyter={"source_hidden": true}
# df_jobs_anal.loc[
#     ('sherlock', 'tofiwadi_49', 'oh', 47.0, 1)
#     ]

# + jupyter={"source_hidden": true}
# indices_to_stop_running[0:2]
