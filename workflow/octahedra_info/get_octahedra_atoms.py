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

# # Obtaining the indices of the atoms that make up the active octahedra
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# # #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_octa_info,
    )

# +
from local_methods import get_octahedra_atoms

# get_octahedra_atoms
# -

from methods import get_df_struct_drift, get_df_jobs, get_df_init_slabs

# +
df_jobs = get_df_jobs()

df_init_slabs = get_df_init_slabs()
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/octahedra_info",
    )

# ### Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

df_octa_info_prev = get_df_octa_info()
# -

df_struct_drift = get_df_struct_drift()

# +
# df_octa_info_prev[df_octa_info_prev.index.duplicated(keep=False)]

# +
# assert df_octa_info_prev.index.is_unique, "SIDFISDI"
# -

# ### Filtering down to `oer_adsorbate` jobs

# +
df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal = df_jobs_anal.droplevel(level=0)


df_ind = df_atoms_sorted_ind.index.to_frame()
df_atoms_sorted_ind = df_atoms_sorted_ind.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_atoms_sorted_ind = df_atoms_sorted_ind.droplevel(level=0)

# + active=""
#
#
#

# +
sys.path.insert(0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering"))

from feature_engineering_methods import get_df_feat_rows
df_feat_rows = get_df_feat_rows(
    df_jobs_anal=df_jobs_anal,
    df_atoms_sorted_ind=df_atoms_sorted_ind,
    df_active_sites=df_active_sites,
    )

df_feat_rows = df_feat_rows.set_index([
    "compenv", "slab_id", "ads",
    # "active_site_orig", "att_num", "from_oh",
    "active_site", "att_num", "from_oh",
    ], drop=False)

# +
# TEMP

# df_feat_rows

# +
# # TEMP
# print(222 * "TEMP | ")

# df = df_feat_rows
# df = df[
#     # (df["compenv"] == "slac") &
#     # (df["slab_id"] == "wonataro_02") &
#     # (df["active_site"] == 56.) &
#     # (df["ads"] == "o") &
#     # (df["from_oh"] == True) &

#     (df["compenv"] == "nersc") &
#     (df["slab_id"] == "buvivore_13") &
#     (df["active_site"] == 38.) &
#     (df["ads"] == "oh") &
#     (df["att_num"] == 3) &
#     (df["from_oh"] == True) &
#     [True for i in range(len(df))]
#     ]
# df_feat_rows = df

# df_feat_rows
# -

# #########################################################
data_dict_list = []
indices_to_process = []
indices_to_not_process = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    from_oh_i = row_i.from_oh
    # #####################################################

    index_i = (compenv_i, slab_id_i, ads_i,
        active_site_i, att_num_i, from_oh_i, )
    if index_i in df_octa_info_prev.index:
        indices_to_not_process.append(index_i)
    else:
        indices_to_process.append(index_i)

# +
# # TEMP
# print(222 * "TEMP | ")

# # # DO NUMBER OF RANDOM SYSTEMS
# # indices_to_process = random.sample(indices_to_not_process, 20)

# # DO EVERYTHING
# indices_to_process = indices_to_not_process

# # # DO SPECIFIC SYSTEMS
# # indices_to_process = [
# #     ('sherlock', 'sifebelo_94', 'o', 63.0, 1, False),
# #     ('sherlock', 'sifebelo_94', 'o', 63.0, 1, True),
# #     ("sherlock", "kapapohe_58", "oh", 29.0, 0, True, ),

# #     ("sherlock", "kamevuse_75", "o", 49.0, 1, False, ),

# #     ]
# -

# ### Main Loop

df_feat_rows_2 = df_feat_rows.loc[
    indices_to_process
    ]

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows_2.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):

    # print(20 * "-")
    # print(index_i)

    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    from_oh_i = row_i.from_oh
    # #####################################################

    # #################################################
    df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_0 == job_id_max_i]
    if df_struct_drift_i.shape[0] == 0:
        df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_1 == job_id_max_i]
    # #################################################
    octahedra_atoms_i = None
    if df_struct_drift_i.shape[0] > 0:
        octahedra_atoms_i = df_struct_drift_i.iloc[0].octahedra_atoms
    # #################################################

    if active_site_orig_i == "NaN":
        from_oh_i = False
    else:
        from_oh_i = True

    # #################################################
    name_i = (
        row_i.compenv, row_i.slab_id, row_i.ads,
        row_i.active_site_orig, row_i.att_num, )
    # #################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_i]
    # #################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    # #################################################


    data_out = get_octahedra_atoms(
        df_jobs=df_jobs,
        df_init_slabs=df_init_slabs,
        atoms_0=atoms_i,
        job_id_0=job_id_max_i,
        active_site=active_site_i,
        compenv=compenv_i,
        slab_id=slab_id_i,
        ads_0=ads_i,
        active_site_0=active_site_orig_i,
        att_num_0=att_num_i,
        )


    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i["job_id_max"] = job_id_max_i
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #################################################
    data_dict_i.update(data_out)
    # #################################################
    data_dict_list.append(data_dict_i)
    # #################################################


# #########################################################
df_octa_info = pd.DataFrame(data_dict_list)

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_octa_info = reorder_df_columns(col_order_list, df_octa_info)

if df_octa_info.shape[0] > 0:
    df_octa_info = df_octa_info.set_index([
        "compenv", "slab_id", "ads",
        # "active_site_orig", "att_num", ],
        # "active_site_orig", "att_num", "from_oh", ],
        "active_site", "att_num", "from_oh", ],
        drop=True)
# #########################################################
# -

# ### Combine previous and current `df_octa_info` to create new one

# +
# # TEMP
# print(111 * "TEMP | ")

# # Set save current version of df_octa_info
# df_octa_info_new = df_octa_info
# -

df_octa_info_new = pd.concat([
    df_octa_info,
    df_octa_info_prev,
    ], axis=0)

# ### Save data to pickle

# #########################################################
# Pickling data ###########################################
directory = os.path.join(
    root_dir, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_octa_info.pickle"), "wb") as fle:
    pickle.dump(df_octa_info_new, fle)
# #########################################################

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("get_octahedra_atoms.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
