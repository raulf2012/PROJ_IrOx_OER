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

# # Analyzing job sets (everything within a `02_attempt` dir for example)
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
import dictdiffer
import json
import copy

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 120

from ase import io

# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs_data,
    get_df_jobs,
    get_df_jobs_paths,
    cwd,
    )

from dft_workflow_methods import (
    is_job_understandable,
    job_decision,
    transfer_job_files_from_old_to_new,
    is_job_compl_done,
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

# ### Script Inputs

# +
# TEST_no_file_ops = True  # True if just testing around, False for production mode
# # TEST_no_file_ops = False
# -

# ### Read Data

# +
df_jobs = get_df_jobs()
if verbose:
    print("df_jobs.shape:", 2 * "\t", df_jobs.shape)

df_jobs_data = get_df_jobs_data(drop_cols=False)
if verbose:
    print("df_jobs_data.shape:", 1 * "\t", df_jobs_data.shape)
    
df_jobs_paths = get_df_jobs_paths()

# +
group_cols = ["compenv", "slab_id", "att_num", "ads", "active_site"]
# group_cols = ["compenv", "slab_id", "att_num", ]
grouped = df_jobs.groupby(group_cols)
max_job_row_list = []
data_dict_list = []
for name, group in grouped:
    data_dict_i = dict()

    max_job = group[group.rev_num == group.rev_num.max()]
    max_job_row_list.append(max_job.iloc[0])

    compenv_i = name[0]
    slab_id_i = name[1]
    att_num_i = name[2]

    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["group_key"] = name

    data_dict_list.append(data_dict_i)

df_max_group_keys = pd.DataFrame(data_dict_list)
df_jobs_max = pd.DataFrame(max_job_row_list)

if verbose:
    print("Number of unique jobs :", df_jobs_max.shape)
    print("^^ Only counting hightest rev_num")
# -


# ### Filtering `df_jobs` by rows that are present in `df_jobs_data`

# +
df_jobs_i = df_jobs.loc[
    df_jobs.index.intersection(df_jobs_data.index)
    ]

if verbose:
    print(
        "These job_ids weren't in df_jobs_data:",
        "\n",
        df_jobs.index.difference(df_jobs_data.index).tolist(), sep="")

# +
# # TEMP

# print(222 * "TEST | ")
# # ('dos_bader', 'sherlock', 'momaposi_60', 1, 'o', 54.0)

# df = df_jobs_i
# df = df[
#     (df["job_type"] == "dos_bader") &
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "momaposi_60") &
#     (df["att_num"] == 1) &
#     (df["ads"] == "o") &
#     (df["active_site"] == 54.) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# +
data_dict_list = []
group_cols = ["job_type", "compenv", "slab_id", "att_num", "ads", "active_site"]
grouped = df_jobs_i.groupby(group_cols)
for name, group in grouped:

    data_dict_i = dict()

    if verbose:
        print(40 * "#")
        print("name:", name)

    # #####################################################
    job_type_i = name[0]
    compenv_i = name[1]
    slab_id = name[2]
    att_num = name[3] 
    ads_i = name[4]
    active_site_i = name[5]
    # #####################################################

    # #####################################################
    max_job = group[group.rev_num == group.rev_num.max()]
    assert max_job.shape[0] == 1, "Must only have 1 there"
    row_max_i = max_job.iloc[0]
    # #####################################################
    job_id_max_i = row_max_i.job_id
    submitted_i = row_max_i.submitted
    # #####################################################

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
    row_paths_max_i = df_jobs_paths_i.loc[job_id_max_i]
    # #####################################################
    path_job_root_w_att_rev = row_paths_max_i.path_job_root_w_att_rev
    path_rel_to_proj = row_paths_max_i.path_rel_to_proj
    # #####################################################

    # #####################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_max_i = df_jobs_data_i.loc[job_id_max_i]
    # #####################################################
    timed_out = row_data_max_i.timed_out
    completed = row_data_max_i.completed
    ediff_conv_reached = row_data_max_i.ediff_conv_reached
    brmix_issue = row_data_max_i.brmix_issue
    num_nonconv_scf = row_data_max_i.num_nonconv_scf
    num_conv_scf = row_data_max_i.num_conv_scf
    true_false_ratio = row_data_max_i.true_false_ratio
    frac_true = row_data_max_i.frac_true
    error = row_data_max_i.error
    error_type = row_data_max_i.error_type
    job_state = row_data_max_i.job_state
    incar_params = row_data_max_i.incar_params
    # #####################################################
    if incar_params is not None:
        ispin = incar_params.get("ISPIN", None)
    else:
        ispin = None
    # #####################################################


    job_completely_done = is_job_compl_done(
        ispin=ispin, completed=completed)

    # #####################################################
    job_understandable = is_job_understandable(
        timed_out=timed_out, completed=completed, error=error,
        job_state=job_state, )
    # #####################################################
    job_decision_i = job_decision(
        error=error, error_type=error_type,
        timed_out=timed_out, completed=completed, submitted=submitted_i,
        job_understandable=job_understandable, ediff_conv_reached=ediff_conv_reached,
        incar_params=incar_params, brmix_issue=brmix_issue,
        num_nonconv_scf=num_nonconv_scf, num_conv_scf=num_conv_scf,
        true_false_ratio=true_false_ratio, frac_true=frac_true, job_state=job_state,
        job_completely_done=job_completely_done, )
    decision_i = job_decision_i["decision"]
    dft_params_i = job_decision_i["dft_params"]
    # #####################################################



    # #####################################################
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["job_understandable"] = job_understandable
    data_dict_i["job_type"] = job_type_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id
    data_dict_i["att_num"] = att_num
    data_dict_i["job_id_max"] = job_id_max_i
    data_dict_i["path_rel_to_proj"] = path_rel_to_proj
    data_dict_i["timed_out"] = timed_out
    data_dict_i["completed"] = completed
    data_dict_i["brmix_issue"] = brmix_issue
    data_dict_i["decision"] = decision_i
    data_dict_i["dft_params_new"] = dft_params_i
    data_dict_i["job_completely_done"] = job_completely_done
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_jobs_anal = pd.DataFrame(data_dict_list)
# df_jobs_anal = df_jobs_anal.sort_values(["compenv", "slab_id", "path_rel_to_proj"])
df_jobs_anal = df_jobs_anal.sort_values(
    ["job_type", "compenv", "slab_id", "path_rel_to_proj"])
# #########################################################
# -


# ### Ordering `df_jobs_anal` and setting index

# +
from misc_modules.pandas_methods import reorder_df_columns

col_order_list = [
    "compenv",
    "slab_id",
    "att_num",
    "ads",
    "active_site",
    "job_id_max",

    "path_short",

    "timed_out",
    "completed",
    "brmix_issue",
    "job_understandable",

    "decision",
    "dft_params_new",

    "path_rel_to_proj",
    ]
df_jobs_anal = reorder_df_columns(col_order_list, df_jobs_anal)
df_jobs_anal = df_jobs_anal.drop(columns=["path_rel_to_proj", ])

# #########################################################
# Setting index
# index_keys = ["compenv", "slab_id", "ads", "active_site", "att_num"]
index_keys = ["job_type", "compenv", "slab_id", "ads", "active_site", "att_num"]
df_jobs_anal = df_jobs_anal.set_index(index_keys)
# -

# ### Writing `df_jobs_anal` to file

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing/out_data")
file_name_i = "df_jobs_anal.pickle"
path_i = os.path.join(directory, file_name_i)
if not os.path.exists(directory): os.makedirs(directory)
with open(path_i, "wb") as fle:
    pickle.dump(df_jobs_anal, fle)
# #########################################################

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
# row_paths_max_i = df_jobs_paths_i.loc[job_id_max_i]

# + jupyter={"source_hidden": true}
# compenv_i

# + jupyter={"source_hidden": true}
# df_jobs_paths_i

# + jupyter={"source_hidden": true}
# group

# + jupyter={"source_hidden": true}
# df_jobs_anal[df_jobs_anal.job_id_max == "bisofadi_42"]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# # row_data_max_iNone	False

# df = df_jobs_anal
# df = df[
#     (df["ads"] == "oh") &
#     (df["active_site"] == 69) &
#     (df["att_num"] == 0) &
#     # (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bekusuvu_00") &
#     [True for i in range(len(df))]
#     ]
# df

# + jupyter={"source_hidden": true}
# df_jobs_anal

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# print(40 * "TEMP | ")

# # name_i = ("nafupemu_49", "o", 48., 1)
# # name_i = ("relovalu_12", "o", "NaN", 1)
# # name_i = ('sherlock', 'miforike_08', 1, 'o', 50.0)
# name_i = ('miforike_08', 'o', 50.0, 1, )

# df = df_jobs_i
# df = df[
#     (df["slab_id"] == name_i[0]) &
#     (df["ads"] == name_i[1]) &
#     (df["active_site"] == name_i[2]) &
#     (df["att_num"] == name_i[3]) &
#     [True for i in range(len(df))]
#     ]
# df_jobs_i = df

# df_jobs_i
# -



# + jupyter={"source_hidden": true}
# row_data_max_i

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# group

# + jupyter={"source_hidden": true}
# path_rel_to_proj
