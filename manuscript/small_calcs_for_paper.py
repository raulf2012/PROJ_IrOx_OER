# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# ### Import Modules

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

from misc_modules.pandas_methods import reorder_df_columns
# -

# ### Read Data

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

# + active=""
#
#
#
#
#
#
#
# -
# # OER Energetics Quantities

# ## ΔG ranges for *O and *OH

# +
g_o__min = df_features_targets["targets"]["g_o"].min()
g_o__max = df_features_targets["targets"]["g_o"].max()

print(
    "ΔG_O (min): ", g_o__min,
    "\n",

    "ΔG_O (max): ", g_o__max,
    "\n",

    "Range in *O: ", g_o__max - g_o__min,
    sep="")


print(30 * "-")
# #########################################################
g_oh__min = df_features_targets["targets"]["g_oh"].min()
g_oh__max = df_features_targets["targets"]["g_oh"].max()

print(
    "ΔG_OH (min): ", g_oh__min,
    "\n",

    "ΔG_OH (max): ", g_oh__max,
    "\n",

    "Range in *OH: ", g_oh__max - g_oh__min,
    sep="")
# -

# ### Average difference between AB2/3

# +
df_AB2 = df_features_targets[df_features_targets.data.stoich == "AB2"]
df_AB3 = df_features_targets[df_features_targets.data.stoich == "AB3"]

g_o__ab2_ave = df_AB2.targets.g_o.mean()
g_oh__ab2_ave = df_AB2.targets.g_oh.mean()

g_o__ab3_ave = df_AB3.targets.g_o.mean()
g_oh__ab3_ave = df_AB3.targets.g_oh.mean()

# +
print("ave ΔG_O (ab2): ", g_o__ab2_ave, sep="")
print("ave ΔG_O (ab3): ", g_o__ab3_ave, sep="")

print("ave ΔG_OH (ab2): ", g_oh__ab2_ave, sep="")
print("ave ΔG_OH (ab3): ", g_oh__ab3_ave, sep="")
# -

print(

    "ΔG_O-OH (IrO2): ",
    g_o__ab2_ave - g_oh__ab2_ave,
    "\n",

    "ΔG_O-OH (IrO3): ",
    g_o__ab3_ave - g_oh__ab3_ave,
    "\n",

    "Diff AB2/3: ",
    (g_o__ab2_ave - g_oh__ab2_ave) - (g_o__ab3_ave - g_oh__ab3_ave),

    sep="")


# +
2.841 - 1.263

1.5780000000000003
# -

2.043 - 0.578

# +
print(

    "ΔΔG_OH (AB3 - AB2): ",
    g_oh__ab3_ave - g_oh__ab2_ave,
    "\n",

    "ΔΔG_O (AB3 - AB2): ",
    g_o__ab3_ave - g_o__ab2_ave,

    sep="")

print("")

print(
    "Average AB2/3 difference in *O and *OH: ",
    "\n",
    ((g_o__ab3_ave - g_o__ab2_ave) + (g_oh__ab3_ave - g_oh__ab2_ave)) / 2.,
    sep="")
# -

(0.685 + 0.799) / 2.

# +
df_features_targets["targets"]["g_o"].min()
# df_features_targets["targets"]["g_o"].max()

# 1.2617520700000173
# 4.2574460500000155

# -2.995693979999998
# -

df_features_targets.index.to_frame().slab_id.unique().shape

df_jobs.slab_id.unique().shape

row_jobs_o_i = df_jobs.loc['guhenihe_85']
active_site_o_i = row_jobs_o_i.active_site

df = df_jobs
df = df[
    (df["slab_id"] == "momaposi_60") &
    (df["ads"] == "o") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

df_jobs_paths.loc["putabagi_08"].gdrive_path

# +
# df_atoms_sorted_ind.loc[
#     ('oer_adsorbate', 'sherlock', 'momaposi_60', 'o', 50.0, 1)
#     ]
# -

assert False

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# def display_df(df, df_name, display_head=True, num_spaces=3):
#     print(40 * "*")
#     print(df_name)
#     print("df_i.shape:", df_i.shape)
#     print(40 * "*")

#     if display_head:
#         display(df.head())

#     print(num_spaces * "\n")

# df_list = [
#     ("df_dft", df_dft),
#     ("df_job_ids", df_job_ids),
#     ("df_jobs", df_jobs),
#     ("df_jobs_data", df_jobs_data),
#     ("df_jobs_data_clusters", df_jobs_data_clusters),
#     ("df_slab", df_slab),
#     ("df_slab_ids", df_slab_ids),
#     ("df_jobs_anal", df_jobs_anal),
#     ("df_jobs_paths", df_jobs_paths),
#     ("df_slabs_oh", df_slabs_oh),
#     ("df_magmoms", df_magmoms),
#     ("df_ads", df_ads),
#     ("df_atoms_sorted_ind", df_atoms_sorted_ind),
#     ("df_rerun_from_oh", df_rerun_from_oh),
#     ("df_slab_simil", df_slab_simil),
#     ("df_active_sites", df_active_sites),
#     ]

# # for name_i, df_i in df_list:
# #     display_df(df_i, name_i)

# # print("")
# # print("")

# # for name_i, df_i in df_list:
# #     display_df(
# #         df_i,
# #         name_i,
# #         display_head=False,
# #         num_spaces=0)
