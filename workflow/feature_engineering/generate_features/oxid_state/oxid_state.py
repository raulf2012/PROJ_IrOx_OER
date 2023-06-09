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

# # Calculate the formal oxidation state of all relaxed slabs
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

# #########################################################
from proj_data import metal_atom_symbol

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    )

# #########################################################
from local_methods import process_row
from local_methods import get_num_metal_neigh_manually
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

# +
df_jobs_anal = get_df_jobs_anal()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()
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
sys.path.insert(
    0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering",
        ),
    )

from feature_engineering_methods import get_df_feat_rows
df_feat_rows = get_df_feat_rows(
    df_jobs_anal=df_jobs_anal,
    df_atoms_sorted_ind=df_atoms_sorted_ind,
    df_active_sites=df_active_sites,
    )

df_feat_rows = df_feat_rows.set_index(
    # ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ],
    ["compenv", "slab_id", "ads", "active_site_orig", "att_num", ],
    drop=False,
    )
# -

df_feat_rows = df_feat_rows.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ],
    drop=False,
    )

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # print("")
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
    # #####################################################

    # TEMP
    # row_atoms_i = df_atoms_sorted_ind.loc[name_orig_i]
    # atoms_i = row_atoms_i.atoms_sorted_good
    # atoms_i.write("__temp__/atoms.traj")
    # atoms_i.write("__temp__/atoms.cif")

    if active_site_orig_i == "NaN":
        from_oh_i = False
    else:
        from_oh_i = True

    name_orig_i = (
        row_i.compenv, row_i.slab_id, row_i.ads,
        row_i.active_site_orig, row_i.att_num,
        )

    out_dict = process_row(
        name=name_orig_i,
        active_site=active_site_i,
        active_site_original=active_site_orig_i,
        metal_atom_symbol=metal_atom_symbol,
        verbose=verbose)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["job_id_max"] = job_id_max_i
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #####################################################
    data_dict_i.update(out_dict)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_eff_ox = pd.DataFrame(data_dict_list)
# #########################################################
# -

# ### Further processing

df_eff_ox = df_eff_ox.set_index(
    # ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_eff_ox

columns = list(df.columns)

# feature_cols = ["eff_oxid_state", ]
feature_cols = ["effective_ox_state", ]
for feature_i in feature_cols:
    columns.remove(feature_i)

multi_columns_dict = {
    "features": feature_cols,
    "data": columns,
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)

df_eff_ox = df
# -

df_eff_ox = df_eff_ox.reindex(columns = ["data", "features", ], level=0)

# ### Write `df_eff_ox` to file

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/generate_features/oxid_state")

# Pickling data ###########################################
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_eff_ox.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_eff_ox, fle)
# #########################################################

# #########################################################
with open(path_i, "rb") as fle:
    df_eff_ox = pickle.load(fle)
# #########################################################

# +
from methods import get_df_eff_ox

df_eff_ox_tmp = get_df_eff_ox()
df_eff_ox_tmp.head()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("oxid_state.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
