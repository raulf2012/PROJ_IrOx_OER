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
from ase.visualize import view

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns

from methods import read_magmom_comp_data

import os
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
    get_df_ads,
    get_df_atoms_sorted_ind,
    get_df_rerun_from_oh,
    get_df_slab_simil,
    get_df_active_sites,
    get_df_features_targets,

    get_other_job_ids_in_set,
    read_magmom_comp_data,

    get_df_coord,
    get_df_slabs_to_run,
    get_df_features,
    )
# + jupyter={"source_hidden": true}
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
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_rerun_from_oh = get_df_rerun_from_oh()
magmom_data_dict = read_magmom_comp_data()
df_slab_simil = get_df_slab_simil()
df_active_sites = get_df_active_sites()
df_features_targets = get_df_features_targets()
df_slabs_to_run = get_df_slabs_to_run()
df_features = get_df_features()


# + jupyter={"source_hidden": true}
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
    ("df_atoms_sorted_ind", df_atoms_sorted_ind),
    ("df_rerun_from_oh", df_rerun_from_oh),
    ("df_slab_simil", df_slab_simil),
    ("df_active_sites", df_active_sites),
    ]

# for name_i, df_i in df_list:
#     display_df(df_i, name_i)

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
# -

print(3 * "\n")

# # TEST TEST TEST TEST

# +
# slab_id_i = "pumusuma_66"
slab_id_i = "romudini_21"

# df_jobs.head()

df_jobs[df_jobs.slab_id == slab_id_i]
# -

df_slab.loc["romudini_21"]

assert False

# +
# df_dft
# df_slab.shape

df_slab_2 = get_df_slab(mode="almost-final")

df_slab_2.shape

# +
# (519, 8)
# (533, 8)
# (544, 8)

# +
# name_i = ('n36axdbw65', '023')
# name_i = ('mkbj6e6e9p', '232')
# name_i = ('v1xpx482ba', '20-21')
# name_i = ('v1xpx482ba', '20-23')

# name_i = ('n36axdbw65', '023')
# <--------------------
# name_i = ('mkbj6e6e9p', '232')
# <--------------------
# name_i = ('v1xpx482ba', '20-21')
# <--------------------
name_i = ('v1xpx482ba', '20-23')
# <--------------------
# -

df = df_slab
df = df[
    (df["bulk_id"] == name_i[0]) &
    (df["facet"] == name_i[1]) &
    [True for i in range(len(df))]
    ]
df

df = df_slab_2
df = df[
    (df["bulk_id"] == name_i[0]) &
    (df["facet"] == name_i[1]) &
    [True for i in range(len(df))]
    ]
df

assert False

# slab_id_to_drop = "dupugulo_25"
# slab_id_to_drop = "rehunuho_26"
# slab_id_to_drop = "wovaseli_71"
slab_id_to_drop = "pipotune_15"

# +
# df_slab = df_slab.drop(slab_id_to_drop)

df_slab_2 = df_slab_2.drop(slab_id_to_drop)

# +
# import os; import pickle
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/creating_slabs",
#     "out_data")
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_slab_final.pickle"), "wb") as fle:
#     pickle.dump(df_slab, fle)
# -

import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
    pickle.dump(df_slab_2, fle)

assert False

# + active=""
#
#

# +
# Getting rid of these rows

# fevofivo_15	n36axdbw65	023

# -

df = df_slab_2
df = df[
    (df["bulk_id"] == "vl9on5zpm1") &
    # (df[""] == "") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

assert False

# + jupyter={"source_hidden": true}
# df_jobs_anal[df_jobs_anal.job_id_max == job_id_j]

df_ind_i = df_jobs_anal.index.to_frame()

df = df_ind_i
df = df[
    (df["compenv"] == "sherlock") &
    (df["slab_id"] == "likeniri_51") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]

df_jobs_anal.loc[
    df.index
    ]

# + jupyter={"source_hidden": true}
df = df_jobs_data
df = df[
    (df["compenv"] == "sherlock") &
    (df["slab_id"] == "likeniri_51") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df.sort_values("rev_num")

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
df_feat_ab2 = df_features_targets[
    df_features_targets["data"]["stoich"] == "AB2"
    ]
df_feat_ab2 = df_feat_ab2.sort_values(("targets", "g_oh", ""), ascending=False)

df_feat_ab3 = df_features_targets[
    df_features_targets["data"]["stoich"] == "AB3"
    ]
df_feat_ab3 = df_feat_ab3.sort_values(("targets", "g_oh", ""), ascending=False)

# + jupyter={"source_hidden": true}
df_feat_ab2

# + jupyter={"source_hidden": true}
df_feat_ab3

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
df_features[
    df_features["features"]["octa_vol"].isna()
    ].shape

# + jupyter={"source_hidden": true}
# ('sherlock', 'telibose_95', 54.0, )

df = df_jobs
df = df[
    (df["compenv"] == "sherlock") &
    (df["slab_id"] == "telibose_95") &
    (df["ads"] == "o") &
    [True for i in range(len(df))]
    ]
df

# + jupyter={"source_hidden": true}
df_jobs_paths.loc["dipekilo_07"].gdrive_path

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
# 122 polymorphs are octahedral and unique
# >>> Removing 12 systems manually because they are not good
# -----
# 110 polymorphs now


# # ###############################################
# 49 are layered materials
# 61 are non-layered materials
# -----
# 61 polymorphs now


# # ###############################################
# 15 polymorphs are above the 0.3 eV/atom above hull cutoff
# -----
# 46 polymorphs now


# # ###############################################
# -----
