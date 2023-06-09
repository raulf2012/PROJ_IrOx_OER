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

# # Calculating the octahedral volume and other geometric quantities
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from proj_data import metal_atom_symbol
metal_atom_symbol_i = metal_atom_symbol

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_coord,
    get_df_octa_vol,
    get_df_octa_vol_init,
    get_df_jobs_data,
    get_df_jobs,
    get_df_coord,
    )

# #########################################################
# from local_methods import process_row_2 as process_row

from methods import get_df_octa_info
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
    show_plot = True
else:
    from tqdm import tqdm
    verbose = False
    show_plot = False

# ### Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

df_jobs_data = get_df_jobs_data()

df_jobs = get_df_jobs()

df_octa_info = get_df_octa_info()
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

# +
# # TEMP

# print(222 * "TEMP | ")

# df_feat_rows = df_feat_rows.sample(n=10)

# +
unique_slab_id_active_sites = df_feat_rows.set_index(
    ["compenv", "slab_id", "active_site", ]).index.unique().tolist()

df_slab_ids_active_sites = pd.DataFrame(
    unique_slab_id_active_sites,
    columns=["compenv", "slab_id", "active_site", ], )


# print(111 * "TEMP | ")
# df = df_slab_ids_active_sites
# df = df[
#     (df["slab_id"] == "kalisule_45") &
# #     (df[""] == "") &
# #     (df[""] == "") &
# #     (df[""] == "") &
#     [True for i in range(len(df))]
#     ]
# df_slab_ids_active_sites = df


# print(111 * "TEMP | ")
# df_slab_ids_active_sites = df_slab_ids_active_sites.sample(n=20)


# #########################################################
data_dict_list = []
# #########################################################
group_cols = ["compenv", "slab_id", ]
grouped = df_slab_ids_active_sites.groupby(group_cols)
# #########################################################
for (compenv_i, slab_id_i), group in grouped:

    # print(compenv_i, slab_id_i)


# if True:
#     # nersc	legofufi_61
#     print(222 * "TEMP | ")
#     compenv_i = "nersc"
#     slab_id_i = "legofufi_61"
#     group = grouped.get_group((compenv_i, slab_id_i, ))


    df = df_jobs
    df_jobs_i = df[
        (df["job_type"] == "oer_adsorbate") &
        (df["compenv"] == compenv_i) &
        (df["slab_id"] == slab_id_i) &
        (df["ads"] == "o") &
        (df["active_site"] == "NaN") &
        (df["rev_num"] == 1) &
        [True for i in range(len(df))]
        ]

    # assert df_jobs_i.shape[0] == 1, "ISDSJDIFDSS"

    # #########################################################
    row_jobs_i = df_jobs_i.iloc[0]
    # #########################################################
    job_id_i = row_jobs_i.job_id
    compenv_i = row_jobs_i.compenv
    ads_i = row_jobs_i.ads
    att_num_i = row_jobs_i.att_num
    # #########################################################

    # #########################################################
    row_data_i = df_jobs_data.loc[job_id_i]
    # #########################################################
    atoms_init_i = row_data_i.init_atoms
    # #########################################################

    from_oh_i = row_data_i.rerun_from_oh
    if np.isnan(from_oh_i):
        from_oh_i = False




    active_site_orig_i = "NaN"

    df_coord_i = get_df_coord(
        mode="init-slab",  # 'bulk', 'slab', 'post-dft', 'init-slab'
        init_slab_name_tuple=(
            compenv_i, slab_id_i, ads_i,
            active_site_orig_i, att_num_i,
            ),
        verbose=True,
        )

    for active_site_i in group["active_site"].tolist():
    # for active_site_i in [95.0, ]:

        name_tmp_i = (
            "final",
            compenv_i, slab_id_i, ads_i,
            active_site_i, att_num_i, from_oh_i, )
        row_octa_info_i = df_octa_info.loc[name_tmp_i]
        octahedra_atoms_i = row_octa_info_i.octahedra_atoms
        metal_active_site_i = row_octa_info_i.metal_active_site


        name_i = (
            row_jobs_i.compenv, row_jobs_i.slab_id, row_jobs_i.ads,
            "NaN", att_num_i,
            )

        out_dict_i = process_row(
            name=name_i,
            active_site=active_site_i,
            active_site_original=active_site_orig_i,
            atoms=atoms_init_i,
            octahedra_atoms=octahedra_atoms_i,
            df_coord=df_coord_i,
            verbose=verbose,
            metal_active_site=metal_active_site_i
            )


        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["job_id_max"] = job_id_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["compenv"] = compenv_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["ads"] = ads_i
        data_dict_i["active_site_orig"] = active_site_orig_i
        data_dict_i["att_num"] = att_num_i
        # data_dict_i["from_oh"] = from_oh_i
        # #################################################
        data_dict_i.update(out_dict_i)
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_octa_vol_init = pd.DataFrame(data_dict_list)
# #########################################################
# -

assert False

# +
df_octa_vol_init

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_octa_vol_init = reorder_df_columns(col_order_list, df_octa_vol_init)

df_octa_vol_init = df_octa_vol_init.set_index(
        ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    drop=False)



df = df_octa_vol_init

multi_columns_dict = {
    "features": ["active_o_metal_dist", "ir_o_mean", "ir_o_std", "octa_vol", ],
    "data": ["compenv", "slab_id", "ads", "att_num", "active_site", "job_id_max", ],
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)
df_octa_vol_init = df


df_octa_vol_init = df_octa_vol_init.reindex(columns = ["data", "features", ], level=0)

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    from_oh_i = row_i.from_oh
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    # #####################################################

    # #####################################################
    name_tmp_i = (
        "final",
        compenv_i, slab_id_i, ads_i,
        active_site_i, att_num_i, from_oh_i, )
    row_octa_info_i = df_octa_info.loc[name_tmp_i]
    # #####################################################
    octahedra_atoms_i = row_octa_info_i.octahedra_atoms
    metal_active_site_i = row_octa_info_i.metal_active_site
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


    out_dict = process_row(
        name=name_i,
        active_site=active_site_i,
        active_site_original=active_site_orig_i,
        atoms=atoms_i,
        octahedra_atoms=octahedra_atoms_i,
        verbose=verbose,
        metal_active_site=metal_active_site_i,
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
    # #####################################################
    data_dict_i.update(out_dict)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_octa_vol = pd.DataFrame(data_dict_list)

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_octa_vol = reorder_df_columns(col_order_list, df_octa_vol)
# #########################################################

# +
import plotly.express as px
df = px.data.tips()
fig = px.histogram(df_octa_vol, x="octa_vol")

if show_plot:
    fig.show()
# -

df_octa_vol = df_octa_vol.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_octa_vol

multi_columns_dict = {
    "features": ["active_o_metal_dist", "ir_o_mean", "ir_o_std", "octa_vol", ],
    "data": ["from_oh", "compenv", "slab_id", "ads", "att_num", "active_site", "job_id_max", ],
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)
df_octa_vol = df


df_octa_vol = df_octa_vol.reindex(columns = ["data", "features", ], level=0)
# -

df_octa_vol.head()

# + active=""
#
#
#
#
#
#
#
# -

assert False

# ### Save data to file

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/generate_features/octahedra_volume")

directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory):
    os.makedirs(directory)

# +
# Pickling data ###########################################
path_i = os.path.join(root_path_i, "out_data/df_octa_vol.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_octa_vol, fle)
# #########################################################

# Pickling data ###########################################
path_i = os.path.join(root_path_i, "out_data/df_octa_vol_init.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_octa_vol_init, fle)
# #########################################################
# -

df_octa_vol = get_df_octa_vol()
df_octa_vol.head()

df_octa_vol_init = get_df_octa_vol_init()
df_octa_vol_init.head()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("octa_volume.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
