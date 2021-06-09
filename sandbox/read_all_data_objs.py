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
    get_df_jobs_on_clus__all,
    get_df_octa_info,
    )
from methods import get_df_bader_feat
from methods import get_df_struct_drift

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
df_jobs_max = get_df_jobs(return_max_only=True)
df_jobs_on_clus = get_df_jobs_on_clus__all()
df_bader_feat = get_df_bader_feat()
df_struct_drift = get_df_struct_drift()
df_octa_info = get_df_octa_info()


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
# +
# slac	dotivela_46	26.0
# -

df_jobs.loc["henasifu_78"]

# +
index_i = ("slac", "dotivela_46", 26.0)
row_i = df_features_targets.loc[[index_i]]

row_i

# +
index_i = ("sherlock", "vipikema_98", 47.0)
row_i = df_features_targets.loc[[index_i]]


row_i
# -

df_jobs.loc["pekukele_64"]

for index_i, row_i in df_features_targets.iterrows():
    slab_id_i = index_i[1]

    job_id_o = row_i[("data", "job_id_o", "", )]

    row_jobs_i = df_jobs.loc[job_id_o]
    bulk_id_i = row_jobs_i["bulk_id"]

    if bulk_id_i == "cqbrnhbacg":
        tmp = 42
        print(index_i)

df_features_targets.loc[[
    ('sherlock', 'kobehubu_94', 52.0),
    ('sherlock', 'kobehubu_94', 60.0),
    ('sherlock', 'vipikema_98', 47.0),
    ('sherlock', 'vipikema_98', 53.0),
    ('sherlock', 'vipikema_98', 60.0),
    ('slac', 'dotivela_46', 26.0),
    ('slac', 'dotivela_46', 32.0),
    ('slac', 'ladarane_77', 15.0),
    ]]



df_features_targets[("data", "norm_sum_norm_abs_magmom_diff", "", )].sort_values(ascending=False).iloc[0:10].index.tolist()

# +
# df_features_targets.describe()

import plotly.graph_objs as go

data = []
trace = go.Scatter(
    mode="markers",
    y=df_features_targets[("data", "norm_sum_norm_abs_magmom_diff", "", )].sort_values(ascending=False),
    # x=np.abs(df_target_pred_i["err_pred"]),
    )
data.append(trace)
# trace = go.Scatter(
#     # mode="markers",
#     y=np.arange(0, 2, 0.1),
#     x=np.arange(0, 2, 0.1),
#     )
# data.append(trace)

# data = [trace]

fig = go.Figure(data=data)
fig.show()
# -

assert False

df_octa_info

assert False

# +
# df_features_targets

# +
# df_features_targets[("data", "job_id_o", "", )]

df_features_targets_i = df_features_targets.loc[[
    ('sherlock', 'kamevuse_75', 49.0)
    ]]

for name_i, row_i in df_features_targets_i.iterrows():
    job_id_o_i = row_i[("data", "job_id_o", "", )]

    df_octa_info_i = df_octa_info[df_octa_info.job_id_max == job_id_o_i]
    row_octa_i = df_octa_info_i.iloc[0]

    if row_octa_i.error:
        print(name_i)
# -

df_octa_info_i

df_features_targets.loc[
    ('sherlock', 'momaposi_60', 50.0)
    ]

# +
# (df[("data", "found_active_Ir__oh", "", )] == True) &

# df_features_targets["data"]
# df_features_targets.columns.tolist()

df_features_targets[
    ('data', 'found_active_Ir', '')
    ].unique()
# -

assert False

# +
job_ids = ['novofide_69', 'solalenu_64', 'huriwara_92', 'bevadofo_80']

df_jobs_paths.loc[
    job_ids
    ]


df_tmp = df_atoms_sorted_ind[df_atoms_sorted_ind.job_id.isin(job_ids)]

for name_i, row_i in df_tmp.iterrows():
    atoms_i = row_i.atoms_sorted_good
    job_id_i = row_i.job_id

    file_path_no_ext_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "sandbox",
        "__temp__/20210530_atoms_write_temp",
        job_id_i)

        # job_id_i + ".traj")
    # atoms_i.write(file_path_i)

    atoms_i.write(file_path_no_ext_i + ".traj")

    atoms_i.write(file_path_no_ext_i + ".cif")
# -

assert False

# + jupyter={"source_hidden": true}
df_octa_info[df_octa_info.missing_oxy_neigh == True]

# df_octa_info.index.tolist()

# + jupyter={"source_hidden": true}
assert False

# + jupyter={"source_hidden": true}
from proj_data import sys_to_ignore__df_features_targets

sys_to_ignore__df_features_targets

df_features_targets = df_features_targets.drop(
    index=sys_to_ignore__df_features_targets)

# + jupyter={"source_hidden": true}
# df_features_targets
# df_features_targets["features"]["o"]["active_o_metal_dist"].sort_values(ascending=False).iloc[0:6].index.tolist()

# df_features_targets["features"]["o"]["p_band_center"].sort_values(ascending=False).iloc[0:6].index.tolist()
df_features_targets["features"]["o"]["p_band_center"].sort_values(ascending=True).iloc[0:12].index.tolist()
