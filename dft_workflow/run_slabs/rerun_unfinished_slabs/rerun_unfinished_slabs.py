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

# # Rerunning jobs with unfinished slabs
# ---
#
# Main issue here is that I found slabs that seems to finish but were not force optimized/minimized for some strange reason

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path

import pandas as pd

import plotly.graph_objs as go

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_jobs_paths,
    get_df_features_targets,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

df_jobs_anal = get_df_jobs_anal()
df_jobs_data = get_df_jobs_data()
df_jobs_paths = get_df_jobs_paths()
df_features_targets = get_df_features_targets()

# ### Filtering `df_jobs_anal` to oer_adsorbate rows

# +
df_ind = df_jobs_anal.index.to_frame()

df_ind = df_ind[df_ind.job_type == "oer_adsorbate"]

df_jobs_anal_i = df_jobs_anal.loc[
    df_ind.index
    ]

df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

# +
# assert False

# +
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(
        df_jobs_anal_i.index.names,
        name_i))
    # #####################################################
    job_id_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_i]
    # #####################################################
    path_i = row_paths_i.gdrive_path
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_i]
    # #####################################################
    force_largest_i = row_data_i["force_largest"]
    force_sum_i = row_data_i["force_sum"]
    force_sum_per_atom_i = row_data_i["force_sum_per_atom"]
    num_scf_cycles_i = row_data_i.num_scf_cycles
    # #####################################################

    
    if force_largest_i is not None:
        if force_largest_i > 0.02:
            print(name_i, "|", num_scf_cycles_i, "|", force_largest_i)
            # print(path_i)


        # #####################################################
        data_dict_i["job_id_max"] = job_id_i
        data_dict_i["force_largest"] = force_largest_i
        data_dict_i["force_sum"] = force_sum_i
        data_dict_i["force_sum_per_atom"] = force_sum_per_atom_i
        # #####################################################
        data_dict_i.update(name_dict_i)
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################

# #########################################################
df = pd.DataFrame(data_dict_list)
df = df.set_index(df_jobs_anal_i.index.names, drop=False)
# #########################################################

# +
# if force_largest_i is not None:
# -

df_to_rerun = df[df.force_largest > 0.02]

# +
# slac	tefovuto_94	16.0
# -

df_to_rerun

# +
# df = df
# df = df[
#     (df["compenv"] == "slac") &
#     (df["slab_id"] == "tefovuto_94") &
#     (df["active_site"] == 16.) &
#     [True for i in range(len(df))]
#     ]
# df

# +
# /mnt/f/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/
# -

# ### Checking how many rows in `df_features_targets` include one of these not-really-finished jobs
#
# There are ~35 OER data points which use one of these not-really-finished jobs
#
# 27 of these seem real, so 8 are jobs that are maybe still being processed or something

for name_i, row_i in df_features_targets.iterrows():
    tmp = 42

    job_id_o_i = row_i[("data", "job_id_o", "")]
    job_id_oh_i = row_i[("data", "job_id_oh", "")]
    job_id_bare_i = row_i[("data", "job_id_bare", "")]

    o_not_good = job_id_o_i in df_to_rerun.job_id_max.tolist()
    oh_not_good = job_id_oh_i in df_to_rerun.job_id_max.tolist()
    bare_not_good = job_id_bare_i in df_to_rerun.job_id_max.tolist()

    if o_not_good or oh_not_good or bare_not_good:
        # print(name_i, o_not_good, oh_not_good, bare_not_good)
        print(o_not_good, oh_not_good, bare_not_good)

# +
# assert False

# +
# Pickling data ###########################################
import os; import pickle

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/run_slabs/rerun_unfinished_slabs",
    "out_data")

if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_to_rerun__not_force_conv.pickle"), "wb") as fle:
    pickle.dump(df_to_rerun, fle)
# #########################################################

# +
trace = go.Scatter(
    y=df.force_largest.sort_values(ascending=False),
    mode="markers"
    )
data = [trace]

fig = go.Figure(data=data)
fig.show()
# -

df

# + active=""
#
#
#
# -

assert False

# + active=""
#
#
#

# + jupyter={"source_hidden": true}

# # df.force_largest.sort_values?

# + jupyter={"source_hidden": true}
# incar_params_i = row_data_i.incar_params

# incar_params_i["NSW"]

# + jupyter={"source_hidden": true}
# ('oer_adsorbate', 'slac', 'vomelawi_63', 'bare', 60.0, 1)

# # slac	wavihanu_77	bare	48.0	1

# df_ind = df_jobs_anal.index.to_frame()

# df = df_ind
# df = df[
#     (df["job_type"] == "oer_adsorbate") &
#     (df["compenv"] == "slac") &
#     (df["slab_id"] == "wavihanu_77") &
#     (df["active_site"] == 48.) &
#     (df["ads"] == "bare") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]
# df_jobs_anal.loc[
#     df.index
#     ]

# + jupyter={"source_hidden": true}
# df_jobs_anal_i.iloc[0:1]
