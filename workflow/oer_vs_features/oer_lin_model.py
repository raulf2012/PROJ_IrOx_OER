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

# # Constructing linear model for OER adsorption energies
# ---
#

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from layout import layout

# #########################################################
from local_methods import create_linear_model_plot
from local_methods import isolate_target_col
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
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

from methods import get_df_slab
df_slab = get_df_slab()

# #########################################################
df_i = df_features_targets

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()

# +
print(
    "Number of rows in df_features_targets:",
    df_i.shape[0],
    )

# 150
# -

# # Dropping phase 1 slabs

# +
df_index = df_i.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

print("Dropping phase 1 slabs")
df_i = df_i.loc[
    df_index_i.index
    ]

# +
# Keeping track of shape, dropping phase 1 points
# 95
# 118
# 126
# 132
# 163
# 176
# 183
# 199
# 214
# 233
# 254
# 267
# 280
# 300

df_i.shape

# +
from proj_data import layout_shared

layout_master = layout_shared.update(layout)
# -

# # -------------------------

# # All single feature models

# ## G_O models

# +
ads_i = "o"
feature_ads_i = "oh"

# if True:
#     feature_col_i = "active_o_metal_dist"

# if True:
if False:
    print(
        list(df_i["features_stan"][ads_i].columns)
        )


    for feature_col_i in df_i["features_stan"][ads_i].columns:
        print(40 * "=")
        print(feature_col_i)
        print("")

        df_j = isolate_target_col(
            df_i,
            target_col="g_o",
            )

        out_dict = create_linear_model_plot(
            df=df_j,
            feature_columns=[feature_col_i, ],
            ads=ads_i,
            feature_ads=feature_ads_i,
            layout=layout_master,
            verbose=verbose,
            )
        fig = out_dict["fig"]
        fig.show()
# -

# ## G_OH models

# +
ads_i = "oh"
feature_ads_i = "o"

# if True:
if False:

    # for feature_col_i in df_i.features_stan.columns:
    for feature_col_i in df_i["features_stan"][ads_i].columns:

        print(40 * "=")
        print(feature_col_i)
        print("")

        df_j = isolate_target_col(
            df_i,
            target_col="g_" + ads_i,
            )

        out_dict = create_linear_model_plot(
            df=df_j,
            feature_columns=[feature_col_i, ],
            ads=ads_i,
            feature_ads=feature_ads_i,
            layout=layout_master,
            verbose=verbose,
            )
        fig = out_dict["fig"]
        fig.show()

# + active=""
#
#
#
#
# -

# # -------------------------

# # G_O Model

# +
ads_i = "o"
feature_ads_i = "oh"

df_j = isolate_target_col(
    df_i,
    target_col="g_o",
    )

feature_cols_all = list(df_j["features_stan"][ads_i].columns)

format_dict_i = {
    "color": "stoich",
    }

out_dict = create_linear_model_plot(
    df=df_j,
    layout=layout_master,
    ads=ads_i,
    feature_ads=feature_ads_i,
    format_dict=format_dict_i,

    # feature_columns=["eff_oxid_state", "octa_vol", "dH_bulk", ],
    # feature_columns=["eff_oxid_state", "octa_vol", "dH_bulk", "bulk_oxid_state", ],
    feature_columns=feature_cols_all,
    verbose=verbose,
    )

fig = out_dict["fig"]

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_vs_features",
        "out_plot/oer_lin_model__G_O_plot.json"))
# -
if show_plot:
    fig.show()

# + active=""
#
#
#
#
# -

# # G_OH Model

# +
ads_i = "oh"
feature_ads_i = "oh"

df_j = isolate_target_col(
    df_i,
    target_col="g_oh",
    )

feature_cols_all = list(df_j["features_stan"][ads_i].columns)

out_dict = create_linear_model_plot(
    df=df_j,
    layout=layout_master,
    feature_ads=feature_ads_i,
    ads=ads_i,
    format_dict=format_dict_i,

    # feature_columns=["eff_oxid_state", "octa_vol", "dH_bulk", ],
    # feature_columns=["eff_oxid_state", "octa_vol", "dH_bulk", "bulk_oxid_state", ],
    # feature_columns=["eff_oxid_state", "octa_vol", "dH_bulk", "bulk_oxid_state", "ir_o_mean", ],
    feature_columns=feature_cols_all,
    verbose=verbose,
    )
fig = out_dict["fig"]

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_vs_features",
        "out_plot/oer_lin_model__G_OH_plot.json"))
# -

if show_plot:
    fig.show()

# + active=""
#
#
#
# -

# ### Get index off of graph with str frag

# +
df_ind = df_features_targets.index.to_frame()

# frag_i = "walogu"
# frag_i = "kese"
frag_i = "vota"
for index_i, row_i in df_ind.iterrows():
    tmp = 42
    name_i = row_i.compenv + "__" + row_i.slab_id
    if frag_i in name_i:
        print(index_i)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("oer_lin_model.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# color__stoich

# + jupyter={"source_hidden": true}
# # stoich_i

# #     stoich_i = 
# row_data_i["stoich"][""]
# 

# + jupyter={"source_hidden": true}
# row_data_i

# + jupyter={"source_hidden": true}
# from proj_data import stoich_color_dict

# # #########################################################
# data_dict_list = []
# # #########################################################
# for index_i, row_i in df_features_targets.iterrows():
#     # #####################################################
#     data_dict_i = dict()
#     # #####################################################
#     index_dict_i = dict(zip(list(df_features_targets.index.names), index_i))
#     # #####################################################
#     row_data_i = row_i["data"]
#     # #####################################################
#     stoich_i = row_data_i["stoich"][""]
#     norm_sum_norm_abs_magmom_diff_i = \
#         row_data_i["norm_sum_norm_abs_magmom_diff"][""]
#     # #####################################################

#     if stoich_i == "AB2":
#         color__stoich_i = stoich_color_dict["AB2"]
#     elif stoich_i == "AB3":
#         color__stoich_i = stoich_color_dict["AB3"]
#     else:
#         color__stoich_i = stoich_color_dict["None"]



#     # #####################################################
#     data_dict_i["color__stoich"] = color__stoich_i
#     data_dict_i["color__norm_sum_norm_abs_magmom_diff_i"] = \
#         norm_sum_norm_abs_magmom_diff_i
#     # #####################################################
#     data_dict_i.update(index_dict_i)
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################


# # #########################################################
# df_format = pd.DataFrame(data_dict_list)
# # #########################################################

# df_format

# + jupyter={"source_hidden": true}
# Script Inputs

# verbose = True
# verbose = False

# + jupyter={"source_hidden": true}
# df_slab[df_slab["slab_id"] == "batipoha_75"]

# df_slab[df_slab["slab_id"] == "bidoripi_03"]

# df_slab[df_slab["slab_id"] == "mj7wbfb5nt"]

# "mj7wbfb5nt" in df_slab["slab_id"].tolist()
# "mj7wbfb5nt" in df_slab["bulk_id"].tolist()
