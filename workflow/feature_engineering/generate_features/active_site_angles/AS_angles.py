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

# # Calculating the octahedral volume and other geometric quantities
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy

import numpy as np
import pandas as pd
import math

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from proj_data import metal_atom_symbol
metal_atom_symbol_i = metal_atom_symbol

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_coord,
    )
from methods import unit_vector, angle_between
from methods import get_df_coord, get_df_coord_wrap

# #########################################################
from local_methods import get_angle_between_surf_normal_and_O_Ir
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
df_jobs_anal_i = df_jobs_anal

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

# + active=""
#
#

# +
# #########################################################
data_dict_list = []
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
    # #####################################################

    if active_site_orig_i == "NaN":
        from_oh_i = False
    else:
        from_oh_i = True

    name_i = (
        row_i.compenv, row_i.slab_id, row_i.ads,
        row_i.active_site_orig, row_i.att_num,
        )

    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_i]
    # #####################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    # #####################################################
    # atoms_i.write("out_data/atoms.traj")


    df_coord_i = get_df_coord_wrap(name_i, active_site_i)

    angle_i = get_angle_between_surf_normal_and_O_Ir(
        atoms=atoms_i,
        df_coord=df_coord_i,
        active_site=active_site_i,
        )

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
    data_dict_i["angle_O_Ir_surf_norm"] = angle_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_angles = pd.DataFrame(data_dict_list)

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_angles = reorder_df_columns(col_order_list, df_angles)
# #########################################################
# -

df_angles = df_angles.set_index(
    # ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_angles

multi_columns_dict = {
    "features": ["angle_O_Ir_surf_norm", ],
    "data": ["from_oh", "compenv", "slab_id", "ads", "att_num", "active_site", "job_id_max", ],

    # "features": ["eff_oxid_state", ],
    # "data": ["job_id_max", "from_oh", "compenv", "slab_id", "ads", "att_num", ]
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)

df_angles = df
# -

df_angles = df_angles.reindex(columns = ["data", "features", ], level=0)

# + active=""
#
#
#
#
#
#
#

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/generate_features/active_site_angles")

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_AS_angles.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_angles, fle)
# #########################################################

# +
from methods import get_df_angles

df_angles_tmp = get_df_angles()
df_angles_tmp
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("AS_angles.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_feat_rows = df_feat_rows.sample(n=10)

# + jupyter={"source_hidden": true}
# df = df_feat_rows
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "lufinanu_76") &
#     (df["ads"] == "oh") &
#     (df["active_site"] == 46.) &
#     (df["att_num"] == 0) &
#     [True for i in range(len(df))]
#     ]
# df_feat_rows = df

# + jupyter={"source_hidden": true}
# df_feat_rows = df_feat_rows.loc[[574]]

# + jupyter={"source_hidden": true}
# False	46.0	sherlock	lufinanu_76	o	1	bobatudi_54	

# df_feat_rows

# + jupyter={"source_hidden": true}
# df_angles.head()
# df_angles
