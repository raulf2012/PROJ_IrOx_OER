# -*- coding: utf-8 -*-
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

# # Combining features and adsorption energies into one dataframe
# ---
#
#

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", None)

import plotly.graph_objs as go

# #########################################################
from methods import (
    get_df_dft,
    get_df_job_ids,
    get_df_slab,
    get_df_jobs,
    get_df_jobs_data,

    get_df_ads,
    get_df_features,
    
    get_df_octa_vol_init,
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

target_cols = ["g_o", "g_oh", "e_o", "e_oh", ]

# ### Read Data

# +
df_ads = get_df_ads()
df_ads = df_ads.set_index(["compenv", "slab_id", "active_site", ], drop=False)

df_features = get_df_features()
df_features.index = df_features.index.droplevel(level=5)

df_slab = get_df_slab()

df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data()
df_jobs_data["rerun_from_oh"] = df_jobs_data["rerun_from_oh"].fillna(value=False)

df_dft = get_df_dft()

df_job_ids = get_df_job_ids()
df_job_ids = df_job_ids.set_index("job_id")
df_job_ids = df_job_ids[~df_job_ids.index.duplicated(keep='first')]

df_octa_vol_init = get_df_octa_vol_init()
# -

feature_cols = df_features["features"].columns.tolist()

# + active=""
#
#
# -

# ### Collecting other relevent data columns from various data objects

# +
# #########################################################
data_dict_list = []
# #########################################################
for index_i, row_i in df_ads.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    index_dict_i = dict(zip(
        list(df_ads.index.names), index_i, ))
    # #####################################################
    slab_id_i = row_i.slab_id
    job_id_o = row_i.job_id_o
    # #####################################################

    # #####################################################
    row_ids_i = df_job_ids.loc[job_id_o]
    # #####################################################
    bulk_id_i = row_ids_i.bulk_id
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    stoich_i = row_dft_i.stoich
    # #####################################################

    # #####################################################
    row_slab_i = df_slab.loc[slab_id_i]
    # #####################################################
    phase_i = row_slab_i.phase
    # #####################################################

    # #####################################################
    data_dict_i["phase"] = phase_i
    data_dict_i["stoich"] = stoich_i
    # #####################################################
    data_dict_i.update(index_dict_i)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_extra_data = pd.DataFrame(data_dict_list)
df_extra_data = df_extra_data.set_index(
    ["compenv", "slab_id", "active_site", ], drop=True)

new_columns = []
for col_i in df_extra_data.columns:
    new_columns.append(
        ("data", col_i, "")
        )

idx = pd.MultiIndex.from_tuples(new_columns)
df_extra_data.columns = idx
# #########################################################

# + active=""
#
#
#
# -

# ### Collating features data by looping over `df_ads`

dos_bader_feature_cols = [
    "Ir*O_bader",
    "Ir_bader",
    "O_bader",
    "p_band_center",
    ]

# +
# # TEMP
# print(111 * "TEMP | ")

# df_ads = df_ads.loc[[("sherlock", "lufinanu_76", 46., )]]

# +
# print("df_ads.shape[0]:", df_ads.shape[0])

# #########################################################
o_rows_list = []
o_index_list = []
# #########################################################
oh_rows_list = []
oh_index_list = []
# #########################################################
failed_indices_oh = []
for index_i, row_i in df_ads.iterrows():

    # print("SIDJFIJSDIFJIDSIFJISDIFJSDIOFJIODS867tr86r7t86867876t8t76")

    # #####################################################
    index_dict_i = dict(zip(list(df_ads.index.names), index_i))
    # #####################################################
    job_id_o_i = row_i.job_id_o
    job_id_oh_i = row_i.job_id_oh
    job_id_bare_i = row_i.job_id_bare
    # #####################################################

    

    # #####################################################
    ads_i = "o"

    idx = pd.IndexSlice
    df_feat_i = df_features.loc[idx[
        index_dict_i["compenv"],
        index_dict_i["slab_id"],
        ads_i,
        index_dict_i["active_site"],
        :], :]


    row_feat_i = df_feat_i[df_feat_i.data.job_id_max == job_id_o_i]
    mess_i = "There should only be one row after the previous filtering"
    assert row_feat_i.shape[0] == 1, mess_i
    row_feat_i = row_feat_i.iloc[0]


    tmp = list(row_feat_i["features"][dos_bader_feature_cols].to_dict().values())
    num_nan = len([i for i in tmp if np.isnan(i)])
    if num_nan > 0:

        tmp_dict = dict()
        df_tmp = df_feat_i["features"][dos_bader_feature_cols]
        for i_cnt, (name_i, row_i) in enumerate(df_tmp.iterrows()):
            # print(name_i)
            row_values = list(row_i.to_dict().values())
            num_nan = len([i for i in row_values if np.isnan(i)])
            tmp_dict[i_cnt] = num_nan

        max_key = None
        for key, val in tmp_dict.items():
            if val == np.min(list(tmp_dict.values())):
                max_key = key

        # print("Replaced row_feat_i with the row that has the dos/bader info")
        row_feat_i = df_feat_i.iloc[max_key]

    # elif num_nan == 0:
    #     tmp = 42














    # #####################################################
    o_rows_list.append(row_feat_i)
    o_index_list.append(row_feat_i.name)



    # #####################################################
    ads_i = "oh"

    idx = pd.IndexSlice
    df_feat_i = df_features.loc[idx[
        index_dict_i["compenv"],
        index_dict_i["slab_id"],
        ads_i,
        index_dict_i["active_site"],
        :], :]

    if df_feat_i.shape[0] > 0:
        row_feat_i = df_feat_i[df_feat_i.data.job_id_max == job_id_oh_i]

        if row_feat_i.shape[0] > 0:
            mess_i = "There should only be one row after the previous filtering"
            assert row_feat_i.shape[0] == 1, mess_i
            row_feat_i = row_feat_i.iloc[0]


            # #############################################
            oh_rows_list.append(row_feat_i)
            oh_index_list.append(row_feat_i.name)
        else:
            # failed_indices_oh.append(index_i)
            failed_indices_oh.append(job_id_oh_i)

            




# #########################################################
idx = pd.MultiIndex.from_tuples(o_index_list, names=df_features.index.names)
df_o = pd.DataFrame(o_rows_list, idx)
df_o.index = df_o.index.droplevel(level=[2, 4, ])
# #########################################################
idx = pd.MultiIndex.from_tuples(oh_index_list, names=df_features.index.names)
df_oh = pd.DataFrame(oh_rows_list, idx)
df_oh.index = df_oh.index.droplevel(level=[2, 4, ])
# #########################################################

# +
# df = df_o.index.to_frame()
# df = df[
#     (df["slab_id"] == "lufinanu_76") &
#     (df["active_site"] == 46.) &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]

# df_o.loc[
#     df.index
#     ]

# +
# assert False
# -

# ### Checking failed_indices_oh against systems that couldn't be processed

# +
from methods import get_df_atoms_sorted_ind

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_atoms_sorted_ind_i = df_atoms_sorted_ind[
    df_atoms_sorted_ind.job_id.isin(failed_indices_oh)
    ]

df_tmp_8 = df_atoms_sorted_ind_i[df_atoms_sorted_ind_i.failed_to_sort == False]

if df_tmp_8.shape[0] > 0:
    print("Check out df_tmp_8, there where some *OH rows that weren't processed but maybe should be")

# + active=""
#
#
#
# -

# ### Processing and combining feature data columns

# +
from local_methods import combine_dfs_with_same_cols

df_dict_i = {
    "oh": df_oh[["data"]],
    "o": df_o[["data"]],
    }

df_data_comb = combine_dfs_with_same_cols(
    df_dict=df_dict_i,
    verbose=False,
    )


# Adding another empty level to column index
new_cols = []
for col_i in df_data_comb.columns:
    # new_col_i = ("", col_i[0], col_i[1])
    new_col_i = (col_i[0], col_i[1], "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)
df_data_comb.columns = idx
# -

# ### Creating `df_features_comb` and adding another column level for ads

# +
# #########################################################
ads_i = "o"

df_features_o = df_o[["features"]]
columns_i = df_features_o.columns

new_columns_i = []
for col_i in columns_i:
    new_col_i = (col_i[0], ads_i, col_i[1])
    new_columns_i.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_columns_i)
df_features_o.columns = idx

# #########################################################
ads_i = "oh"

df_features_oh = df_oh[["features"]]
columns_i = df_features_oh.columns

new_columns_i = []
for col_i in columns_i:
    new_col_i = (col_i[0], ads_i, col_i[1])
    new_columns_i.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_columns_i)
df_features_oh.columns = idx

# #########################################################
df_features_comb = pd.concat([
    df_features_o,
    df_features_oh,
    ], axis=1)

# +
eff_ox_state_list = []
for name_i, row_i in df_features_comb.iterrows():
    eff_ox_state_o_i = row_i[("features", "o", "effective_ox_state", )]
    eff_ox_state_oh_i = row_i[("features", "oh", "effective_ox_state", )]

    eff_ox_state_i = eff_ox_state_oh_i
    if eff_ox_state_oh_i != eff_ox_state_oh_i:

        # print(name_i)
        # print(20 * "-")
        # print(eff_ox_state_oh_i)
        # print(eff_ox_state_o_i)
        # print("")


        
        if np.isnan(eff_ox_state_oh_i):
            # print(eff_ox_state_o_i)
            if not np.isnan(eff_ox_state_o_i):
                eff_ox_state_i = eff_ox_state_o_i


        elif np.isnan(eff_ox_state_o_i):
            # print(eff_ox_state_o_i)
            if not np.isnan(eff_ox_state_oh_i):
                eff_ox_state_i = eff_ox_state_oh_i

    # if np.isnan(eff_ox_state_i):
    #     print(eff_ox_state_i)

    eff_ox_state_list.append(
        np.round(eff_ox_state_i, 6),
        # eff_ox_state_i
        )


df_features_comb[("features", "effective_ox_state", "")] = eff_ox_state_list
# -

df_features_comb = df_features_comb.drop(columns=[
        ("features", "o", "effective_ox_state", ),
        ("features", "oh", "effective_ox_state", ),
        ]
    )

non_ads_features = [
    # "effective_ox_state",
    "dH_bulk",
    "volume_pa",
    "bulk_oxid_state",
    ]

# +
cols_to_drop = []

new_cols = []

# for col_i in df_features_comb["features"].columns:
for col_i in df_features_comb.columns:
    # print(col_i)

    if col_i[0] == "features":
        tmp = 42
        # print(col_i)

        if col_i[2] in non_ads_features:
            print(col_i)
            if col_i[1] == "oh":
                cols_to_drop.append(col_i)
                new_cols.append(col_i)
            elif col_i[1] == "o":
                col_new_i = (col_i[0], col_i[2], "", )
                new_cols.append(col_new_i)
        else:
            new_cols.append(col_i)

    else:
        new_cols.append(col_i)
                
# non_ads_features

# +
idx = pd.MultiIndex.from_tuples(new_cols)

df_features_comb.columns = idx

df_features_comb = df_features_comb.drop(columns=cols_to_drop)

# +
oh_features = []
o_features = []
other_features = []
for col_i in df_features_comb.columns:
    if col_i[1] == "oh":
        oh_features.append(col_i)
    elif col_i[1] == "o":
        o_features.append(col_i)
    else:
        other_features.append(col_i)

df_features_comb = df_features_comb[
    oh_features + o_features + other_features
    ]

# +
# Adding more levels to df_ads to combine

new_cols = []
for col_i in df_ads.columns:
    # new_col_i = ("", "", col_i)
    new_col_i = (col_i, "", "", )
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)
df_ads.columns = idx
# -

# ### Combining all dataframes

df_all_comb = pd.concat([
    df_features_comb,
    df_data_comb,
    df_ads,
    df_extra_data,
    ], axis=1)

# ### Removing the p-band center feature for *OH (there are none)

# +
df_all_comb = df_all_comb.drop(columns=[
    ('features', 'oh', 'p_band_center'),
    ])

df_all_comb = df_all_comb.drop(columns=[
    ('features', 'oh', 'Ir_bader'),
    ])

df_all_comb = df_all_comb.drop(columns=[
    ('features', 'oh', 'O_bader'),
    ])

df_all_comb = df_all_comb.drop(columns=[
    ('features', 'oh', 'Ir*O_bader'),
    ])


# -

# ### Create `name_str` column

# +
def method(row_i):
    # #########################################################
    name_i = row_i.name
    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #########################################################
    
    name_i = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i)).zfill(3)

    return(name_i)

df_all_comb["data", "name_str", ""] = df_all_comb.apply(
    method,
    axis=1)

# +
df_ads_columns = [i[0] for i in df_ads.columns.tolist()]

for i in target_cols:
    df_ads_columns.remove(i)

# +
data_columns_all = [i[0] for i in df_all_comb["data"].columns]

df_ads_columns_to_add = []
df_ads_columns_to_drop = []
for col_i in df_ads_columns:
    if col_i not in data_columns_all:
        df_ads_columns_to_add.append(col_i)
    else:
        df_ads_columns_to_drop.append(col_i)


# #########################################################
for col_i in df_ads_columns_to_drop:
    df_all_comb.drop(columns=(col_i, "", ""), inplace=True)

# #########################################################
new_columns = []
for col_i in df_all_comb.columns:
    if col_i[0] in df_ads_columns_to_add:
        new_columns.append(
            ("data", col_i[0], "", )
            )
    elif col_i[0] in target_cols:
        new_columns.append(
            ("targets", col_i[0], "", )
            )
    else:
        new_columns.append(col_i)

idx = pd.MultiIndex.from_tuples(new_columns)
df_all_comb.columns = idx


# -

# ### Adding magmom comparison data

def process_df_magmoms_comp_i(df_magmoms_comp_i):
    """
    """
    def method(row_i):
        new_column_values_dict = dict(
            job_id_0=None,
            job_id_1=None,
            job_id_2=None,
            )

        job_ids_tri = row_i.job_ids_tri

        ids_sorted = list(np.sort(list(job_ids_tri)))

        job_id_0 = ids_sorted[0]
        job_id_1 = ids_sorted[1]
        job_id_2 = ids_sorted[2]

        new_column_values_dict["job_id_0"] = job_id_0
        new_column_values_dict["job_id_1"] = job_id_1
        new_column_values_dict["job_id_2"] = job_id_2

        for key, value in new_column_values_dict.items():
            row_i[key] = value
        return(row_i)

    df_magmoms_comp_i = df_magmoms_comp_i.apply(method, axis=1)
    df_magmoms_comp_i = df_magmoms_comp_i.set_index(["job_id_0", "job_id_1", "job_id_2", ])

    return(df_magmoms_comp_i)


# +
from methods import get_df_magmoms, read_magmom_comp_data

df_magmoms = get_df_magmoms()


data_dict_list = []
for name_i, row_i in df_all_comb.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    index_dict_i = dict(zip(df_all_comb.index.names, name_i))
    # #####################################################

    magmom_data_i = read_magmom_comp_data(name=name_i)
    if magmom_data_i is not None:
        df_magmoms_comp_i = magmom_data_i["df_magmoms_comp"]
        df_magmoms_comp_i = process_df_magmoms_comp_i(df_magmoms_comp_i)

        # tmp = df_magmoms_comp_i.sum_norm_abs_magmom_diff.min()
        # tmp_list.append(tmp)

        job_ids = []
        for ads_j in ["o", "oh", "bare", ]:
            job_id_j = row_i["data"]["job_id_" + ads_j][""]
            if job_id_j is not None:
                job_ids.append(job_id_j)


        sum_norm_abs_magmom_diff_i = None
        if len(job_ids) == 3:
            job_ids = list(np.sort(job_ids))
            job_id_0 = job_ids[0]
            job_id_1 = job_ids[1]
            job_id_2 = job_ids[2]

            row_mags_i = df_magmoms_comp_i.loc[
                (job_id_0, job_id_1, job_id_2, )
                ]
            sum_norm_abs_magmom_diff_i = row_mags_i.sum_norm_abs_magmom_diff
            norm_sum_norm_abs_magmom_diff_i = sum_norm_abs_magmom_diff_i / 3
            
        # #################################################
        data_dict_i.update(index_dict_i)
        # #################################################
        data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff_i
        data_dict_i["norm_sum_norm_abs_magmom_diff"] = norm_sum_norm_abs_magmom_diff_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_tmp = pd.DataFrame(data_dict_list)
df_tmp = df_tmp.set_index(["compenv", "slab_id", "active_site", ])

# #########################################################
new_cols = []
for col_i in df_tmp.columns:
    new_col_i = ("data", col_i, "")
    new_cols.append(new_col_i)
idx = pd.MultiIndex.from_tuples(new_cols)
df_tmp.columns = idx

df_all_comb = pd.concat([df_all_comb, df_tmp], axis=1)
# -

# ### Add OER overpotential data

# #########################################################
import pickle; import os
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis",
    "out_data")
path_i = os.path.join(
    directory,
    "df_overpot.pickle")
with open(path_i, "rb") as fle:
    df_overpot = pickle.load(fle)
# #########################################################

# +
df_overpot = df_overpot.drop(columns="name")

new_cols = []
for col_i in df_overpot.columns:
    new_col_i = ("data", col_i, "", )
    new_cols.append(new_col_i)
df_overpot.columns = pd.MultiIndex.from_tuples(new_cols)
# -

df_all_comb = pd.concat([
    df_all_comb,
    df_overpot,
    ], axis=1)

# ### Adding surface energy data

# +
from methods import get_df_SE
df_SE = get_df_SE()


new_cols = []
for col_i in df_SE.columns:
    new_col_i = ("data", col_i, "", )
    new_cols.append(new_col_i)
df_SE.columns = pd.MultiIndex.from_tuples(new_cols)


cols_to_remove = []
for col_i in df_SE.columns.tolist():
    if col_i in df_all_comb.columns.tolist():
        cols_to_remove.append(col_i)


df_all_comb = pd.concat([
    df_all_comb,
    # df_SE,
    df_SE.drop(columns=cols_to_remove),
    ], axis=1)
# -

# ### Adding plot format properties

# +
from proj_data import stoich_color_dict

# #########################################################
data_dict_list = []
# #########################################################
# for index_i, row_i in df_features_targets.iterrows():
for index_i, row_i in df_all_comb.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    index_dict_i = dict(zip(list(df_all_comb.index.names), index_i))
    # #####################################################
    row_data_i = row_i["data"]
    # #####################################################
    stoich_i = row_data_i["stoich"][""]
    norm_sum_norm_abs_magmom_diff_i = \
        row_data_i["norm_sum_norm_abs_magmom_diff"][""]
    # #####################################################

    if stoich_i == "AB2":
        color__stoich_i = stoich_color_dict["AB2"]
    elif stoich_i == "AB3":
        color__stoich_i = stoich_color_dict["AB3"]
    else:
        color__stoich_i = stoich_color_dict["None"]


    # #####################################################
    data_dict_i[("format", "color", "stoich")] = color__stoich_i
    data_dict_i[("format", "color", "norm_sum_norm_abs_magmom_diff")] = \
        norm_sum_norm_abs_magmom_diff_i
    # #####################################################
    data_dict_i.update(index_dict_i)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_format = pd.DataFrame(data_dict_list)
df_format = df_format.set_index(["compenv", "slab_id", "active_site", ])
df_format.columns = pd.MultiIndex.from_tuples(df_format.columns)
# #########################################################
# -

df_all_comb = pd.concat(
    [
        df_all_comb,
        df_format,
        ],
    axis=1,
    )

# ### Mixing Bader charges with bond lengths

# +
# df_all_comb["features"][""]
# df_all_comb[("features", "o", "Ir*O_bader", )]

df_all_comb[("features", "o", "Ir*O_bader/ir_o_mean", )] = \
    df_all_comb[("features", "o", "Ir*O_bader", )] / df_all_comb[("features", "o", "ir_o_mean", )]
# -

# ### Calculating ΔG_OmOH target column

# +
# Computing ΔG_O-OH
g_o = df_all_comb[("targets", "g_o", "")]
g_oh = df_all_comb[("targets", "g_oh", "")]

df_all_comb[("targets", "g_o_m_oh", "")] = g_o - g_oh

# Computing ΔE_O-OH
e_o = df_all_comb[("targets", "e_o", "")]
e_oh = df_all_comb[("targets", "e_oh", "")]

df_all_comb[("targets", "e_o_m_oh", "")] = e_o - e_oh
# -

# ### Adding in pre-DFT features

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_all_comb.iterrows():
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    job_id_o_i = row_i[("data", "job_id_o", "")]

    name_octa_i = (compenv_i, slab_id_i,
        "o", active_site_i, 1, )
    row_octa_i = df_octa_vol_init.loc[
        name_octa_i
        ]

    row_octa_dict_i = row_octa_i["features"].to_dict()
    
    # #####################################################
    data_dict_i = {}
    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    # #####################################################
    data_dict_i.update(row_octa_dict_i)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################







df_feat_pre = pd.DataFrame(data_dict_list)
df_feat_pre = df_feat_pre.set_index(["compenv", "slab_id", "active_site", ])

new_cols = []
for col_i in df_feat_pre.columns:
    new_col_i = ("features_pre_dft", col_i + "__pre", "")
    new_cols.append(new_col_i)

idx = pd.MultiIndex.from_tuples(new_cols)
df_feat_pre.columns = idx




df_all_comb = pd.concat([
    df_all_comb,
    df_feat_pre,
    ], axis=1)
# -

# ### Reindexing multiindex to get order columns

df_all_comb = df_all_comb.reindex(columns=[
    'targets',
    'data',
    'format',
    'features',
    'features_pre_dft',
    'features_stan',
    ], level=0)

# ### Removing rows that aren't supposed to be processed (bad slabs)

# +
from methods import get_df_slabs_to_run
df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_not_run = df_slabs_to_run[df_slabs_to_run.status == "bad"]

slab_ids_to_not_include = df_slabs_to_not_run.slab_id.tolist()

df_index = df_all_comb.index.to_frame()
df_all_comb = df_all_comb.loc[
    ~df_index.slab_id.isin(slab_ids_to_not_include)
    ]
# -

# ### OLD DEPRECATED | Getting rid of NERSC jobs

# +
# print("Getting rid of NERSC jobs and phase 1 systems")

# Getting rid of NERSC jobs

# indices_to_keep = []
# for i in df_all_comb.index:
#     if i[0] != "nersc":
#         indices_to_keep.append(i)

# df_all_comb = df_all_comb.loc[
#     indices_to_keep
#     ]

df_all_comb = df_all_comb[df_all_comb["data"]["phase"] > 1]
# -

# ### Printing how many `NaN` rows there are for each feature

for col_i in df_all_comb.features.columns:
    if verbose:
        df_tmp_i = df_all_comb[df_all_comb["features"][col_i].isna()]
        print(col_i, ":", df_tmp_i.shape[0])

# ### Write data to pickle

df_features_targets = df_all_comb
# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering",
    "out_data")
file_name_i = "df_features_targets.pickle"
path_i = os.path.join(directory, file_name_i)
if not os.path.exists(directory): os.makedirs(directory)
with open(path_i, "wb") as fle:
    pickle.dump(df_features_targets, fle)
# #########################################################

# +
from methods import get_df_features_targets

df_features_targets_tmp = get_df_features_targets()
df_features_targets_tmp.head()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("combine_features_targets.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
