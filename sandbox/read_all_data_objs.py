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

from IPython.display import display

import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 20
# pd.set_option('display.max_rows', None)

# #########################################################
from methods import (
    get_df_jobs_paths,
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_slabs_oh,
    get_df_init_slabs,
    get_df_magmoms,
    )
from methods import (
    get_other_job_ids_in_set,
    )

# + jupyter={"source_hidden": true}
import os
import sys

import copy
import shutil
from pathlib import Path
from contextlib import contextmanager

# import pickle; import os

import pickle
import  json

import pandas as pd
import numpy as np

from ase import io

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns
# -

# # Read data objects with methods

from methods import get_df_ads

df_dft = get_df_dft()
df_job_ids = get_df_job_ids()
df_jobs = get_df_jobs(exclude_wsl_paths=True)
df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_clusters = get_df_jobs_data_clusters()
df_slab = get_df_slab()
df_slab_ids = get_df_slab_ids()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()
df_slabs_oh = get_df_slabs_oh()
df_init_slabs = get_df_init_slabs()
df_magmoms = get_df_magmoms()
df_ads = get_df_ads()


def display_df(df, df_name, display_head=True, num_spaces=3):
    print(40 * "*")
    print(df_name)
    print("df_i.shape:", df_i.shape)
    print(40 * "*")

    if display_head:
        display(df.head())

    print(num_spaces * "\n")


df_list = [
    ("df_dft", df_dft),
    ("df_job_ids", df_job_ids),
    ("df_jobs", df_jobs),
    ("df_jobs_data", df_jobs_data),
    ("df_jobs_data_clusters", df_jobs_data_clusters),
    ("df_slab", df_slab),
    ("df_slab_ids", df_slab_ids),
    ("df_jobs_anal", df_jobs_anal),
    ("df_jobs_paths", df_jobs_paths),
    ("df_slabs_oh", df_slabs_oh),
    ("df_magmoms", df_magmoms),
    ("df_ads", df_ads),
    ]

# +
# for name_i, df_i in df_list:
#     display_df(df_i, name_i)

# +
# print("")
# print("")

# for name_i, df_i in df_list:
#     display_df(
#         df_i,
#         name_i,
#         display_head=False,
#         num_spaces=0)

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

# # TEST TEST TEST TEST

# +
job_id = "ruvaneru_15"

df_jobs_paths.loc[job_id].gdrive_path
# -

df_ads[
    ~df_ads.g_oh.isna()
    ]

assert False

# +
# "rawupuga_30" in df_job_ids.job_id.tolist()

# df_job_ids

df_jobs[
    (df_jobs.compenv == "sherlock") & \
    (df_jobs.slab_id == "kenukami_73") & \
    (df_jobs.ads == "oh") & \
    [True for i in range(len(df_jobs))]
    ]
# -

assert False

# +
# job_id = "gasusupo_45"
job_id = "pewehobe_99"

row_jobs = df_jobs.loc[job_id]
row_paths = df_jobs_paths.loc[job_id]

# row_paths.gdrive_path
row_paths.path_full

# +
df_jobs_i = get_other_job_ids_in_set(job_id, df_jobs=df_jobs)

# df_jobs_paths.loc[
df_jobs_data.loc[
    df_jobs_i.index    
    ]
# -

assert False

# + jupyter={"source_hidden": true}
# def get_other_job_ids_in_set():
#     """
#     """
#     row_jobs = df_jobs.loc[job_id]

#     compenv_i = row_jobs.compenv
#     bulk_id_i = row_jobs.bulk_id
#     slab_id_i = row_jobs.slab_id
#     ads_i = row_jobs.ads
#     att_num_i = row_jobs.att_num

#     df_jobs_i = df_jobs[
#         (df_jobs.compenv == compenv_i) & \
#         (df_jobs.bulk_id == bulk_id_i) & \
#         (df_jobs.slab_id == slab_id_i) & \
#         (df_jobs.ads == ads_i) & \
#         (df_jobs.att_num == att_num_i) & \
#         [True for i in range(len(df_jobs))]
#         ]

#     return(df_jobs_i)

# + jupyter={"source_hidden": true}
# # "b5cgvsb16w/111/oh/active_site__62/03_attempt/_05/out_data"

# df_jobs[
#     (df_jobs.bulk_id == "b5cgvsb16w") & \
#     (df_jobs.ads == "oh") & \
#     (df_jobs.active_site == 62.) & \
#     (df_jobs.att_num == 3) & \
#     [True for i in range(len(df_jobs))]
#     ]

# + jupyter={"source_hidden": true}
# job_id =  "wadifowe_41"

# df_jobs.loc[job_id]

# # df_jobs_paths.loc[job_id].gdrive_path
