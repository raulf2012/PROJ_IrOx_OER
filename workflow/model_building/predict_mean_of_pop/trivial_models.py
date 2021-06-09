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
# -
# ## Predicting mean for whole dataset

# +
g_o_list = df_features_targets[("targets", "g_o", "", )]

g_o_mean = g_o_list.mean()

mae_i = np.absolute(
    g_o_list - g_o_mean
    ).mean()
print("MAE *O:", mae_i)


g_oh_list = df_features_targets[("targets", "g_oh", "", )]

g_oh_mean = g_oh_list.mean()

mae_i = np.absolute(
    g_oh_list - g_oh_mean
    ).mean()
print("MAE *OH:", mae_i)

# +
df_ab2 = df_features_targets.loc[
    df_features_targets[df_features_targets[("data", "stoich", "")] == "AB2"].index
    ]

df_ab3 = df_features_targets.loc[
    df_features_targets[df_features_targets[("data", "stoich", "")] == "AB3"].index
    ]
# -

# ### Predicting on AB2 only

# +
g_o_list_ab2 = df_ab2[("targets", "g_o", "", )]

g_o_mean_ab2 = g_o_list_ab2.mean()

mae_i = np.absolute(
    g_o_list_ab2 - g_o_mean_ab2
    ).mean()

print("MAE *O:", mae_i)


g_oh_list_ab2 = df_ab2[("targets", "g_oh", "", )]

g_oh_mean_ab2 = g_oh_list_ab2.mean()

mae_i = np.absolute(
    g_oh_list_ab2 - g_oh_mean_ab2
    ).mean()

print("MAE *OH:", mae_i)
# -

# ### Predicting on AB3 only

# +
g_o_list_ab3 = df_ab3[("targets", "g_o", "", )]

g_o_mean_ab3 = g_o_list_ab3.mean()

mae_i = np.absolute(
    g_o_list_ab3 - g_o_mean_ab3
    ).mean()

print("MAE *O:", mae_i)


g_oh_list_ab3 = df_ab3[("targets", "g_oh", "", )]

g_oh_mean_ab3 = g_oh_list_ab3.mean()

mae_i = np.absolute(
    g_oh_list_ab3 - g_oh_mean_ab3
    ).mean()

print("MAE *OH:", mae_i)

# + active=""
#
#
#
#
#
#
#
# -
# The weighted average of the MAE for each eff. ox. state value is equal to the GP MAE for the single feature model with EOS

# +
eff_ox_states = list(np.sort(df_features_targets[("features", "o", "effective_ox_state", )].unique().tolist()))
eff_ox_states = eff_ox_states[:-1]


# df_features_targets.features.columns

for eff_ox_i in eff_ox_states:
    print("eff_ox_i:", np.round(eff_ox_i, 3))

    df_i = df_features_targets[
        df_features_targets[("features", "o", "effective_ox_state")] == eff_ox_i]

    g_o_list_i = df_i[("targets", "g_o", "", )]
    g_o_mean_i = g_o_list_i.mean()

    mae_i = np.absolute(
        g_o_list_i - g_o_mean_i
        ).mean()

    print("df_i.shape:", df_i.shape[0])
#     print(df_i.shape[0])
    print("MAE:", np.round(mae_i, 3))
    print("")

# +
# g_o_list_ab3 = df_ab3[("targets", "g_o", "", )]

# g_o_mean_ab3 = g_o_list_ab3.mean()

# np.absolute(
#     g_o_list_ab3 - g_o_mean_ab3
#     ).mean()
