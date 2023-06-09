# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# + jupyter={"source_hidden": true} tags=[]
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
    get_df_jobs_on_clus__all,
    get_df_octa_info,
    get_df_features_targets_seoin,
    )
from methods import get_df_bader_feat
from methods import get_df_struct_drift

from misc_modules.pandas_methods import reorder_df_columns

# + tags=[] jupyter={"source_hidden": true}
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
df_jobs_max = get_df_jobs(return_max_only=True)
df_jobs_on_clus = get_df_jobs_on_clus__all()
df_bader_feat = get_df_bader_feat()
df_struct_drift = get_df_struct_drift()
df_octa_info = get_df_octa_info()
df_seoin = get_df_features_targets_seoin()


# + jupyter={"source_hidden": true} tags=[]
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
# -
df_features_targets.features.columns.tolist()

assert False

# +
df_features_targets

df = df_features_targets
df = df[
    (df.targets.g_o_m_oh > 0.8) &
    (df.targets.g_o_m_oh < 0.9) &
    [True for i in range(len(df))]
    ]
    # (df[""] == "") &
    # (df[""] == "") &

df
# -

assert False

df_features_targets[
    df_features_targets.data.from_oh__o == False
    ]
    # ].index.tolist()

assert False

# +
# df_features_targets.format.color.norm_sum_norm_abs_magmom_diff.max()


import plotly.graph_objs as go

trace = go.Scatter(
    mode="markers",
    # x=x_array,
    y=np.sort(df_features_targets.format.color.norm_sum_norm_abs_magmom_diff),
    )
data = [trace]

fig = go.Figure(data=data)
fig.show()
# -



df_features_targets.shape

df_features_targets[df_features_targets.format.color.norm_sum_norm_abs_magmom_diff > 0.148725].shape

1 - 15/368

# +
k = 0.94
n = df_features_targets.shape[0]

index = int(np.round(k * n))

np.mean(
    df_features_targets.format.color.norm_sum_norm_abs_magmom_diff.sort_values()[index:index+2]
    )
# -

df_features_targets.format.color.norm_sum_norm_abs_magmom_diff.min()
df_features_targets.format.color.norm_sum_norm_abs_magmom_diff.max()

df_features_targets[
    df_features_targets.data.from_oh__o == False
    ]
    # ].index.tolist()

job_id_x = "labavobe_95"
get_other_job_ids_in_set(
    job_id_x,
    df_jobs=df_jobs,
    oer_set=True,
    only_last_rev=True,
    )

# +
df_features_targets

df = df_features_targets
df = df[
    (df.targets.g_o_m_oh > 2.9) &
    (df.targets.g_o_m_oh < 3.1) &
    [True for i in range(len(df))]
    ]
    # (df[""] == "") &
    # (df[""] == "") &

df
# -



df.index.tolist()

df.targets.g_o_m_oh

assert False

cols_A = [

    ('targets', 'g_o', ''),
    ('data', 'stoich', ''),
    ('features', 'bulk_oxid_state', ''),
    ('features', 'dH_bulk', ''),
    ('features', 'effective_ox_state', ''),
    ('features', 'volume_pa', ''),
    ('features', 'o', 'active_o_metal_dist'),
    ('features', 'o', 'angle_O_Ir_surf_norm'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'ir_o_std'),
    ('features', 'o', 'octa_vol'),
    ('features', 'o', 'degrees_off_of_straight__as_opp'),
    ('features', 'o', 'oxy_opp_as_bl'),

    ]

cols_B = [

    ('targets', 'g_o', ''),
    # ('data', 'job_id_o', ''),
    # ('data', 'job_id_oh', ''),
    # ('data', 'job_id_bare', ''),
    ('data', 'stoich', ''),
    ('features', 'o', 'active_o_metal_dist'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'octa_vol'),
    ('features', 'o', 'oxy_opp_as_bl'),
    ('features', 'o', 'degrees_off_of_straight__as_opp'),
    ('features', 'dH_bulk', ''),
    ('features', 'bulk_oxid_state', ''),
    ('features', 'effective_ox_state', ''),
    ('features_pre_dft', 'active_o_metal_dist__pre', ''),
    ('features_pre_dft', 'ir_o_mean__pre', ''),
    ('features_pre_dft', 'ir_o_std__pre', ''),
    ('features_pre_dft', 'octa_vol__pre', ''),

    ]

# +
print("Shared columns")
print("-------------------")

for col_i in cols_A:
    if col_i in cols_B:
        print(col_i)

# +
print("Cols in A not in B")
print("-------------------")

for col_i in cols_A:
    if col_i not in cols_B:
        print(col_i)

# +
print("Cols in B not in A")
print("-------------------")

for col_i in cols_B:
    if col_i not in cols_A:
        print(col_i)
# -

assert False

# +
df_jobs[df_jobs.bulk_id == "926dnunrxf"]

df = df_jobs
df = df[
    (df["bulk_id"] == "926dnunrxf") &
    (df["facet"] == "010") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

# +
df_ind = df_features_targets.index.to_frame()

df = df_ind
df = df[
    (df["slab_id"] == "tesameli_14") &
    # (df[""] == "") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df.index.tolist()
# -

df_features_targets.loc[[
    ('sherlock', 'tesameli_14', 50.0)
    ]]

cols_0

df_features_targets

assert False

# +
# import plotly.io as pio
# scope = pio.kaleido.scope

# +
# scope
# scope._shutdown_kaleido()

# +
# ('sherlock', 'kagekiha_49', 'o', 96.0, 1, False)
# -

df_jobs_paths.loc["lebuvana_28"].gdrive_path

df = df_jobs
df = df[
    (df["compenv"] == "sherlock") &
    (df["slab_id"] == "kagekiha_49") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

df_jobs_paths.loc["businipu_97"].gdrive_path

# +
df_ind = df_features_targets.index.to_frame()

df = df_ind
df = df[
    (df["compenv"] == "sherlock") &
    # (df["slab_id"] == "kagekiha_49") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df
# -

assert False

# +


for i in df_seoin.columns.to_list():
    print(i)
# -

df_octa_info

df_features_targets.features.o

# +
name_i = ('sherlock', 'kamevuse_75', 49.0)

row = df_features_targets.loc[name_i]

row.data
