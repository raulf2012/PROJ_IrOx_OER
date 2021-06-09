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

# # I'm trying to erase phase 1 from everything so that I can remake those slabs and run them again
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
# +
# pumusuma_66 | 1
# fufalego_15 | 1
# tefenipa_47 | 1
# silovabu_91 | 1
# naronusu_67 | 1
# nofabigo_84 | 1
# kodefivo_37 | 1

# + active=""
#
#
#
# -

df_slab_2 = get_df_slab(mode="almost-final")

# +
df_jobs_paths_i = df_jobs_paths

# df_jobs_paths_i = df_jobs_paths_i.sort_values("gdrive_path")

# +
# assert False

# +
phase_1_slab_ids = df_slab[df_slab.phase == 1].index.tolist()

# TEMP
phase_1_slab_ids = [
    "pumusuma_66",
    "fufalego_15",
    "tefenipa_47",
    "silovabu_91",
    "naronusu_67",
    "nofabigo_84",
    "kodefivo_37",
    ]

df_jobs_i = df_jobs[
    df_jobs.slab_id.isin(phase_1_slab_ids)
    ]

df_paths_i = df_jobs_paths_i.loc[
    df_jobs_i.index
    ]

df_paths_i = df_paths_i.sort_values("gdrive_path")

# +
# "pumusuma_66" in df_jobs.slab_id.unique()

# +
# df_paths_i
# -

df_slab_2.loc[phase_1_slab_ids]

df_slab.loc[phase_1_slab_ids]

assert False

print(df_slab.shape[0])
df_slab = df_slab.drop(phase_1_slab_ids)
print(df_slab.shape[0])

print(df_slab_2.shape[0])
df_slab_2 = df_slab_2.drop(phase_1_slab_ids)
print(df_slab_2.shape[0])

# Pickling data #######################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab_final.pickle"), "wb") as fle:
    pickle.dump(df_slab, fle)
# #####################################################

# +
# # Pickling data #######################################
# import os; import pickle
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/creating_slabs",
#     "out_data")
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
#     pickle.dump(df_slab_2, fle)
# # #####################################################

# + active=""
#
#
#

# +
# tmp = [print(i) for i in df_paths_i.gdrive_path.tolist()]

for i in df_paths_i.gdrive_path.tolist():
    tmp = 42

new_path_i = "old.dft_workflow_phase_1/" + "/".join(i.split("/")[1:])

print(

    "mv ",

    "$PROJ_irox_oer_gdrive/",
    i,

    " ",

    "$PROJ_irox_oer_gdrive/",
    new_path_i,

    sep="")
# + active=""
#
#
#
# -

assert False
