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

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_anal,
    get_df_jobs_data,

    get_df_slab,
    )
# -

from methods import get_other_job_ids_in_set

# ### Read Data

# +
df_jobs = get_df_jobs()
df_jobs_anal = get_df_jobs_anal()
df_jobs_data = get_df_jobs_data()

df_slab = get_df_slab()

# +
df_slab_i = df_slab[df_slab.phase == 2]

slab_ids = df_slab_i.index.tolist()

# +
df_jobs_i = df_jobs[
    df_jobs.slab_id.isin(slab_ids)
    ]

df_jobs_anal_i = df_jobs_anal[
    df_jobs_anal.index.to_frame().slab_id.isin(slab_ids)
    ]

# +
df_index = df_jobs_anal_i.index.to_frame()

df_jobs_anal_o = df_jobs_anal_i.loc[
    df_index[df_index.ads == "o"].index
    ]

df_jobs_anal_o_i = df_jobs_anal_o[df_jobs_anal_o.job_completely_done == False]

indices_to_keep = []
for index_i, row_i in df_jobs_anal_o_i.iterrows():
    decision_i = row_i.decision

    if "PENDING" not in decision_i and "RUNNING" not in decision_i:
        indices_to_keep.append(index_i)

df_jobs_anal_o_i.loc[
    indices_to_keep
    ]

# +
# assert False

# +
name_i = ('sherlock', 'begefabi_44', 'o', 'NaN', 1)

df_jobs_anal_o_i = df_jobs_anal_o.loc[[name_i]]
# df_jobs_anal_o_i = df_jobs_anal_o

for index_i, row_i in df_jobs_anal_o_i.iterrows():
    # #####################################################
    compenv_i, slab_id_i, ads_i, active_site_i, att_num_i = index_i
    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################


    df_jobs_oer = get_other_job_ids_in_set(
        job_id_max_i,
        df_jobs=df_jobs,
        oer_set=True,
        )

    bare_present = "bare" in df_jobs_oer.ads.unique()

    if not bare_present:
        print(index_i)
# -

df_jobs_oer

assert False

# +
print(
    "Number of job sets total:",
    df_jobs_anal_i.shape[0],
    )

print(
    "Number of completed job sets:",
    df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True].shape[0]
    )
# -

df_jobs_anal_i[df_jobs_anal_i.job_completely_done == False]

assert False

# +
# Number of job sets total: 253
# Number of completed job sets: 223

# Number of job sets total: 223
# Number of completed job sets: 124

# Number of job sets total: 223
# Number of completed job sets: 120

# +
df_index_i = df_jobs_anal_i.index.to_frame()

df_index_bare_i = df_index_i[df_index_i.ads == "bare"]
# -

print(
    "Number of bare calculations, this will be the number of new data points:",
    "\n",
    df_index_bare_i.shape[0],
    sep="")

for name_i, row_i in df_index_bare_i.iterrows():
    compenv_i, slab_id_i, ads_i, active_site_i, att_num_i = name_i




    idx = pd.IndexSlice
    df_ind_0 = df_index_i.loc[idx[compenv_i, slab_id_i, :, active_site_i, :], :]

    idx = pd.IndexSlice
    df_ind_1 = df_index_i.loc[idx[compenv_i, slab_id_i, "o", "NaN", :], :]

    df_ind = pd.concat([df_ind_0, df_ind_1])

    df_jobs_anal_tmp = df_jobs_anal_i.loc[
        df_ind.index
        ]

    df_jobs_anal_tmp = df_jobs_anal_tmp.drop(
        columns=["timed_out", "completed", "brmix_issue", "dft_params_new", ])


    idx = pd.IndexSlice
    df_bare_i = df_jobs_anal_tmp.loc[idx[:, :, "bare", :, :], :]

    if True not in df_bare_i.job_completely_done.tolist():
        from IPython.display import display
        print(40 * "*")
        display(
            df_jobs_anal_tmp
            )
        print("")
        print("")

df_jobs_anal_tmp

# +
# idx = pd.IndexSlice
# df_bare_i = df_jobs_anal_tmp.loc[idx[:, :, "bare", :, :], :]

# if True not in df_bare_i.job_completely_done.tolist():
#     print("IDJIFDI")
# -

df_jobs_anal[df_jobs_anal.job_id_max == "bitakito_28"]

# +
# ['dahuvisi_85', 'peligiti_14', 'fukudiko_66', 'fipipida_61', 'tibunane_36', 'rehurese_36', 'hutepimu_57', 'bitakito_28']

# +
# df_index_bare_i
