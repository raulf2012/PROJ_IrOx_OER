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

# # Collect/Collate Job Directory Data
# ---
#
# Collect all DFT slab data dataframes from the clusters and collates them together

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# #########################################################
from misc_modules.misc_methods import GetUniqueFriendlyID

# #########################################################
from methods import (
    get_df_slab_ids,
    get_slab_id,
    get_df_job_ids,
    get_job_id,
    )

# #########################################################
from local_methods import get_num_revs_for_group
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read Data

# +
df_job_ids = get_df_job_ids()

compenv_local = os.environ["COMPENV"]
# -


# # Parse `df_jobs_base` files

# +
root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing",
    "out_data")

compenvs = [
    "nersc",
    "sherlock",
    "slac",
    "wsl",
    ]

df_dict = dict()
# df_list = []
for compenv_i in compenvs:
    file_i = "df_jobs_base_" + compenv_i + ".pickle"
    my_file = Path(os.path.join(root_dir, file_i))
    if my_file.is_file():

        # #################################################
        path_i = os.path.join(
            my_file._str)
        with open(path_i, "rb") as fle:
            df_i = pickle.load(fle)
        # #################################################

        df_i["compenv_origin"] = compenv_i

        df_dict[compenv_i] = df_i
        # df_list.append(df_i)

# df_comb = pd.concat(df_list, axis=0)
df_comb = pd.concat(list(df_dict.values()), axis=0)
df_comb = df_comb.reset_index(drop=True)

# Change type of `num_revs` to int
# df_comb.num_revs = df_comb.num_revs.astype("int")

df_jobs = df_comb

# +
# list(df_dict.values())

# +
# assert False

# +
# 1898	slac	mwmg9p7s6o	hivovaru_77 11-20
# 1901

# +
# df_jobs

# +
# df = df_jobs
# df = df[
#     (df["compenv"] == "slac") &
#     # (df["slab_id"] == "mwmg9p7s6o") &
#     (df["bulk_id"] == "mwmg9p7s6o") &
#     (df["ads"] == "bare") &
#     (df["facet"] == "11-20") &
#     (df["active_site"] == 27.) &
#     [True for i in range(len(df))]
#     ]
# #     bare	48	
# df.path_job_root_w_att.tolist()

# +
from misc_modules.pandas_methods import reorder_df_columns

df_jobs = reorder_df_columns(["compenv", "compenv_origin"], df_jobs)

# +
df_slab_ids = get_df_slab_ids()

slab_ids = []
for bulk_id_i, facet_i in zip(df_jobs.bulk_id.tolist(), df_jobs.facet.tolist()):
    slab_id_i = get_slab_id(bulk_id_i, facet_i, df_slab_ids)
    slab_ids.append(slab_id_i)
df_jobs["slab_id"] = slab_ids

#| - Reorder DataFrame columns
from misc_modules.pandas_methods import reorder_df_columns

df_cols = [
    "compenv",
    "bulk_id",
    "slab_id",
    "facet",
    "ads",
    "active_site",

    "num_revs",
    "att_num",
    "rev_num",
    "is_rev_dir",
    "is_attempt_dir",

    # Paths
    "path_job_root",
    "path_job_root_w_att_rev",
    "path_full",
    "path_rel_to_proj",
    "path_job_root_w_att",
    "gdrive_path",
    ]

df_jobs = reorder_df_columns(df_cols, df_jobs)
#__|
# -
# # Remove duplicates from gathering form local and cluster systems

# +
df_jobs["active_site"] = df_jobs.active_site.fillna("NaN")

series_list = []
grouped = df_jobs.groupby([
    "compenv", "bulk_id", "slab_id", "facet",
    "ads", "active_site", "att_num", "rev_num", ])
for name, group in grouped:

    df_i = group
    if df_i.shape[0] > 1:
        df_i_2 = df_i[df_i.compenv == df_i.compenv_origin]

        mess_i = "Hopefully this parses it down to one row"
        assert df_i_2.shape[0] == 1, mess_i

        series_i = df_i_2.iloc[0]
    elif df_i.shape[0] == 1:
        series_i = df_i.iloc[0]
    else:
        print("Not good")

    series_list.append(series_i)

df_jobs = pd.DataFrame(series_list)
# -

df_i.path_full.tolist()

# +
# print("TEMP")
# assert False

# +
# df_i_2
# df_i.path_full.tolist()

# +
# group

# +
# df_jobs

# +
# assert False
# -

# # Creating job ids data

# +
job_ids_list = []
used_ids = set(df_job_ids.job_id.tolist())
data_dict_list = []
df_i = df_jobs[[
    "compenv", "bulk_id", "slab_id",
    "facet", "att_num", "rev_num", "ads", "active_site", ]]
iterator = tqdm(df_i.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    row_i = df_i.loc[index_i]
    # #####################################################
    compenv = row_i.compenv
    bulk_id = row_i.bulk_id
    slab_id = row_i.slab_id
    facet = row_i.facet
    att_num = row_i.att_num
    rev_num = row_i.rev_num
    ads = row_i.ads
    active_site = row_i.active_site
    # #####################################################

    data_dict_i.update(row_i.to_dict())

    job_id_i = get_job_id(
        compenv, bulk_id, slab_id, facet, att_num,
        rev_num, ads, active_site, df_job_ids=df_job_ids)

    if job_id_i is None:
        # print(index_i)
        job_id_i = GetUniqueFriendlyID(used_ids)
        used_ids.add(job_id_i)

    data_dict_i["job_id"] = job_id_i

    data_dict_list.append(data_dict_i)
    job_ids_list.append(job_id_i)


# #########################################################
df_job_ids_new = pd.DataFrame(data_dict_list)

path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing",
    "out_data/job_id_mapping.csv")

from pathlib import Path
my_file = Path(os.environ["PROJ_irox_oer_gdrive"])
if my_file.is_dir():


    from methods import get_df_job_ids
    df_job_ids_old = get_df_job_ids()
    df_job_ids_old = df_job_ids_old.drop_duplicates()

    # #########################################################
    df_job_ids_old = df_job_ids_old.set_index([
        "compenv", "bulk_id", "slab_id",
        "facet", "att_num", "rev_num",
        "ads", "active_site"], drop=False)

    df_job_ids_new = df_job_ids_new.set_index([
        "compenv", "bulk_id", "slab_id",
        "facet", "att_num", "rev_num",
        "ads", "active_site"], drop=False)


    # #########################################################
    drop_from_new_job_ids = []
    for index_i in df_job_ids_new.index:
        already_present = index_i in df_job_ids_old.index

        if already_present:

            row_old_i = df_job_ids_old.loc[index_i]
            job_id_old_i = row_old_i.job_id

            row_new_i = df_job_ids_new.loc[index_i]
            job_id_new_i = row_new_i.job_id

            job_ids_match = job_id_new_i == job_id_old_i

        if already_present and job_ids_match:
            drop_from_new_job_ids.append(index_i)

    df_job_ids_new = df_job_ids_new.drop(labels=drop_from_new_job_ids)


    # #########################################################
    df_job_ids_comb = pd.concat([
        df_job_ids_old,
        df_job_ids_new,
        ], axis=0)

    df_job_ids_comb = df_job_ids_comb.reset_index(drop=True)

    df_job_ids_comb = df_job_ids_comb.drop_duplicates()

    df_job_ids_comb.to_csv(
        path_i,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        na_rep="NULL",
        )

    # df_job_ids_new.to_csv(
    #     path_i,
    #     index=False,
    #     quoting=csv.QUOTE_NONNUMERIC,
    #     na_rep="NULL",
    #     )
else:
    if compenv_local == "wsl":
        for i in range(50): print(
            "The PROJ_irox_oer_gdrive location is not here/accesible",
            "\n",
            "os.environ['PROJ_irox_oer_gdrive']:", os.environ["PROJ_irox_oer_gdrive"],
            "\n",
            "This will mess things up royally I think",
            "\n",
            sep="")

df_jobs["job_id"] = job_ids_list

# +
from misc_modules.pandas_methods import reorder_df_columns

df_jobs = reorder_df_columns(["bulk_id", "slab_id", "job_id", "facet", "compenv", ], df_jobs)

# Set index to `job_id`
df_jobs = df_jobs.set_index("job_id", drop=False)

# +
group_cols = [
    "bulk_id", "slab_id", "facet", "compenv",
    "ads", "active_site", "att_num",
    ]
groups = []
grouped = df_jobs.groupby(group_cols)
for name_i, group_i in grouped:
    num_revs_i = get_num_revs_for_group(group=group_i)
    group_i["num_revs"] = num_revs_i
    groups.append(group_i)

if len(groups) == 0:
    pass
else:
    df_jobs = pd.concat(groups, axis=0)


# +
#| - Adding short path column
def method(row_i):
    """
    """
    path_job_root_w_att_rev = row_i.path_job_root_w_att_rev

    new_path_list = []
    start_adding = False; start_adding_ind = None
    for i_cnt, i in enumerate(path_job_root_w_att_rev.split("/")):
        if i == "dft_jobs":
            start_adding = True
            start_adding_ind = i_cnt
        if start_adding and i_cnt > start_adding_ind:
            new_path_list.append(i)

    path_short_i = "/".join(new_path_list)

    return(path_short_i)

df_i = df_jobs
df_i["path_short"] = df_i.apply(method, axis=1)
df_jobs = df_i
#__|
# -

# # Create `df_jobs_paths`

# +
path_cols = [i for i in df_jobs.columns.tolist() if "path" in i]

cols = ["compenv", "compenv_origin", ] + path_cols
df_jobs_paths = df_jobs[cols]

# +
# #########################################################
df_jobs = df_jobs.drop(
    columns=path_cols,
    )

# #########################################################
cols_to_drop = [
    "is_rev_dir",
    "is_attempt_dir",
    ]

# #########################################################
df_jobs = df_jobs.drop(columns=cols_to_drop)

# #########################################################
# Sorting dataframe
sort_list = ["compenv", "bulk_id", "slab_id", "att_num", "rev_num", "ads", "active_site"]
df_jobs = df_jobs.sort_values(sort_list)
# -

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_jobs_combined.pickle"), "wb") as fle:
    pickle.dump(df_jobs, fle)
# #########################################################

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_jobs_paths.pickle"), "wb") as fle:
    pickle.dump(df_jobs_paths, fle)
# #########################################################

if verbose:
    print("df_jobs.shape:", df_jobs.shape)

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_job_dirs_data.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
