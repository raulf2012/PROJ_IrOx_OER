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
pd.options.mode.chained_assignment = None  # default='warn'

import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression

# #########################################################
from proj_data import scatter_marker_props, layout_shared

# #########################################################
from local_methods import run_linear_workflow

# +
sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/model_building"
        )
    )

from methods_model_building import (
    simplify_df_features_targets,
    run_kfold_cv_wf,
    process_feature_targets_df,
    process_pca_analysis,
    pca_analysis,
    run_regression_wf,
    )

# + active=""
#
#
#
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

# ### Script Inputs

cols_to_use = [
    'active_o_metal_dist',
    'effective_ox_state',
    'ir_o_mean',
    'ir_o_std',
    'octa_vol',
    'dH_bulk',
    'volume_pa',
    'bulk_oxid_state',
    ]

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
# assert False
# -

# ### Dropping phase 1 slabs

# +
df_index = df_i.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

if verbose:
    print("Dropping phase 1 slabs")
df_i = df_i.loc[
    df_index_i.index
    ]

# +
from proj_data import layout_shared

# layout_master = layout_shared.update(layout)

# + active=""
#
#
#
#
# -

# # -------------------------

# # G_O Model

# +
# target_ads_i = "o"

target_ads_i = "oh"
feature_ads_i = "oh"
# + active=""
#
#
#
#

# +
df_j = simplify_df_features_targets(
    df_i,
    target_ads="o",
    feature_ads="oh",
    )

df_format = df_features_targets[("format", "color", "stoich", )]
# -

df_i

df_j

# +
# assert False
# -

data_dict = dict()
for num_pca_i in range(1, len(cols_to_use) + 1, 1):
# for num_pca_i in range(1, 4, 1):
    if verbose:
        print("")
        print(40 * "*")
        print(num_pca_i)

    out_dict = run_kfold_cv_wf(
        # #################################
        df_features_targets=df_j,
        cols_to_use=cols_to_use,
        df_format=df_format,
        # #################################
        num_pca_comp=num_pca_i,
        k_fold_partition_size=20,
        model_workflow=run_linear_workflow,
        # #################################
        )

    df_target_pred = out_dict["df_target_pred"]
    MAE = out_dict["MAE"]
    R2 = out_dict["R2"]

    if verbose:
        print(
            "MAE: ",
            np.round(MAE, 5),
            " eV",
            sep="")

        print(
            "R2: ",
            np.round(R2, 5),
            sep="")

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["df_target_pred"] = df_target_pred
    data_dict_i["MAE"] = MAE
    data_dict_i["R2"] = R2
    # #####################################################
    data_dict[num_pca_i] = data_dict_i
    # #####################################################

# + active=""
#
#
#

# +
#| - READ WRITE TEMP OBJ
import os
import sys
import pickle

# # Pickling data ###########################################
# import os; import pickle
# path_i = os.path.join(
#     os.environ["HOME"],
#     "__temp__",
#     "temp.pickle")
# with open(path_i, "wb") as fle:
#     pickle.dump(data_dict, fle)
# # #########################################################

# # #########################################################
# import pickle; import os
# path_i = os.path.join(
#     os.environ["HOME"],
#     "__temp__",
#     "temp.pickle")
# with open(path_i, "rb") as fle:
#     data_dict = pickle.load(fle)
# # #########################################################

#__|

# +
data_dict_list = []
for num_pca_i, dict_i in data_dict.items():

    MAE_i = dict_i["MAE"]
    R2_i = dict_i["R2"]

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["num_pca"] = num_pca_i
    data_dict_i["MAE"] = MAE_i
    data_dict_i["R2"] = R2_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df = pd.DataFrame(data_dict_list)
df = df.set_index("num_pca")
# #########################################################

# +
layout_mine = go.Layout(

    showlegend=False,

    yaxis=go.layout.YAxis(
        title=dict(
            text="K-Fold Cross Validated MAE",
            ),
        ),

    xaxis=go.layout.XAxis(
        title=dict(
            text="Num PCA Components",
            ),
        ),

    )


# #########################################################
layout_shared_i = layout_shared.update(layout_mine)
# -

scatter_marker_props

# +
trace_i = go.Scatter(
    x=df.index,
    y=df.MAE,
    mode="markers",
    marker=dict(
        # **scatter_marker_props,
        ),
    )
trace_i.marker.update(
    scatter_marker_props
    )

data = [trace_i, ]

fig = go.Figure(
    data=data,
    layout=layout_shared_i,
    )
if show_plot:
    fig.show()
# -

# ## Plotting the best model (optimal num PCA components)

# num_pca_best = 3
num_pca_best = 6

# +
data_dict_i = data_dict[
    num_pca_best
    ]

df_target_pred = data_dict_i["df_target_pred"]

# +
max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()

dd = 0.1

layout_mine = go.Layout(

    showlegend=False,

    yaxis=go.layout.YAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Simulated ΔG<sub>*{}</sub>".format(feature_ads_i.upper()),
            ),
        ),

    xaxis=go.layout.XAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Predicted ΔG<sub>*{}</sub>".format(feature_ads_i.upper()),
            ),
        ),

    )


# #########################################################
layout_shared_i = layout_shared.update(layout_mine)

# +
trace_parity = go.Scatter(
    y=[min_val - 2 * dd, max_val + 2 * dd],
    x=[min_val - 2 * dd, max_val + 2 * dd],
    mode="lines",
    line_color="black",
    )

trace = go.Scatter(
    y=df_target_pred["y"],
    x=df_target_pred["y_pred"],
    mode="markers",
    opacity=0.8,

    marker=dict(
        # color=df_target_pred["color"],
        # **scatter_marker_props,
        **scatter_marker_props.to_plotly_json(),
        ),

    )

data = [trace_parity, trace, ]

fig = go.Figure(data=data, layout=layout_shared_i)
if show_plot:
    fig.show()

# +
# df_target_pred
# -

# # In-fold model (trained on all data, no test/train split)

# +
out_dict = run_regression_wf(
    df_features_targets=df_j,
    cols_to_use=cols_to_use,
    df_format=df_format,
    num_pca_comp=num_pca_best,
    model_workflow=run_linear_workflow,
    )

df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]
# -

if verbose:
    print("MAE:", MAE)
    print("R2:", R2)

# +
max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()

dd = 0.1

layout_mine = go.Layout(

    showlegend=False,

    yaxis=go.layout.YAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Simulated ΔG<sub>*{}</sub>".format(feature_ads_i.upper()),
            ),
        ),

    xaxis=go.layout.XAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Predicted ΔG<sub>*{}</sub>".format(feature_ads_i.upper()),
            ),
        ),

    )


# #########################################################
layout_shared_i = layout_shared.update(layout_mine)

# +
trace_parity = go.Scatter(
    y=[min_val - 2 * dd, max_val + 2 * dd],
    x=[min_val - 2 * dd, max_val + 2 * dd],
    mode="lines",
    line_color="black",
    )

trace = go.Scatter(
    y=df_target_pred["y"],
    x=df_target_pred["y_pred"],
    mode="markers",
    opacity=0.8,

    marker=dict(
        # color=df_target_pred["color"],
        # **scatter_marker_props,
        **scatter_marker_props.to_plotly_json(),
        ),

    )

data = [trace_parity, trace, ]

fig = go.Figure(data=data, layout=layout_shared_i)
if show_plot:
    fig.show()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("linear_model.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# from local_methods import process_feature_targets_df, process_pca_analysis, run_regression_wf

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_test_targets["y_pred"] = 

# y_pred.shape

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_features_targets=df_j
# cols_to_use=cols_to_use
# df_format=df_format
# run_pca=True
# num_pca_comp=3
# k_fold_partition_size=100
# model_workflow=run_linear_workflow

# # def run_kfold_cv_wf(
# #     df_features_targets=None,
# #     cols_to_use=None,
# #     df_format=None,

# #     run_pca=True,
# #     num_pca_comp=3,
# #     k_fold_partition_size=20,
# #     model_workflow=None,
# #     model_settings=None,
# #     # kdict=None,
# #     ):
# """

# df_features_targets

#     Column levels are as follows:

#     features                         targets
#     feat_0    feat_1    feat_2...    g_oh

# """
# #| -  run_kfold_cv_wf
# # df_j = df_features_targets

# # df_j = process_feature_targets_df(
# #     df_features_targets=df_features_targets,
# #     cols_to_use=cols_to_use,
# #     )

# # df_j = df_j.dropna()

# # # COMBAK New line
# # # print(17 * "TEMP | ")
# # df_j = df_j.dropna(axis=1)

# # # print(df_j)

# # if run_pca:
# #     # #####################################################
# #     # df_pca = process_pca_analysis(
# #     out_dict = process_pca_analysis(
# #         df_features_targets=df_j,
# #         num_pca_comp=num_pca_comp,
# #         )
# #     # #####################################################
# #     df_pca = out_dict["df_pca"]
# #     pca = out_dict["pca"]
# #     # #####################################################

# #     df_data = df_pca
# # else:
# #     df_data = df_j
# #     pca = None


# # #| - Creating k-fold partitions
# # x = df_j.index.tolist()

# # partitions = []
# # for i in range(0, len(x), k_fold_partition_size):
# #     slice_item = slice(i, i + k_fold_partition_size, 1)
# #     partitions.append(x[slice_item])
# # #__|

# # #| - Run k-fold cross-validation
# # df_target_pred_parts = []
# # df_target_pred_parts_2 = []
# # regression_model_list = []
# # for k_cnt, part_k in enumerate(range(len(partitions))):

# #     test_partition = partitions[part_k]

# #     train_partition = partitions[0:part_k] + partitions[part_k + 1:]
# #     train_partition = [item for sublist in train_partition for item in sublist]


# #     # #####################################################
# #     # df_test = df_pca.loc[test_partition]
# #     # df_train = df_pca.loc[train_partition]
# #     df_test = df_data.loc[test_partition]
# #     df_train = df_data.loc[train_partition]
# #     # #####################################################
# #     df_train_features = df_train["features"]
# #     df_train_targets = df_train["targets"]
# #     df_test_features = df_test["features"]
# #     df_test_targets = df_test["targets"]
# #     # #####################################################

# #     # #####################################################
# #     # Using the model on the test set (Normal)
# #     # #####################################################
# #     out_dict = model_workflow(
# #         df_train_features=df_train_features,
# #         df_train_targets=df_train_targets,
# #         df_test_features=df_test_features,
# #         df_test_targets=df_test_targets,
# #         model_settings=model_settings,
# #         )
# #     # #####################################################
# #     df_target_pred = out_dict["df_target_pred"]
# #     min_val = out_dict["min_val"]
# #     max_val = out_dict["max_val"]
# #     RM_1 = out_dict["RegressionModel"]
# #     # #####################################################

# #     regression_model_list.append(RM_1)

# #     # #####################################################
# #     # Using the model on the training set (check for bad model)
# #     # #####################################################
# #     out_dict_2 = model_workflow(
# #         df_train_features=df_train_features,
# #         df_train_targets=df_train_targets,
# #         df_test_features=df_train_features,
# #         df_test_targets=df_train_targets,
# #         model_settings=model_settings,
# #         # kdict=kdict,
# #         )
# #     # #####################################################
# #     df_target_pred_2 = out_dict_2["df_target_pred"]
# #     min_val_2 = out_dict_2["min_val"]
# #     max_val_2 = out_dict_2["max_val"]
# #     RM_2 = out_dict_2["RegressionModel"]
# #     # #####################################################





# #     df_target_pred_parts.append(df_target_pred)
# #     df_target_pred_parts_2.append(df_target_pred_2)

# # # #########################################################
# # df_target_pred_concat = pd.concat(df_target_pred_parts)
# # df_target_pred = df_target_pred_concat

# # # Get format column from main `df_features_targets` dataframe

# # # df_features_targets["format"]["color"]["stoich"]
# # # df_format = df_features_targets[("format", "color", "stoich", )]

# # # df_format.name = "color"
# # #
# # # # Combining model output and target values
# # # df_target_pred = pd.concat([
# # #     df_format,
# # #     df_target_pred,
# # #     ], axis=1)

# # df_target_pred = df_target_pred.dropna()
# # # #########################################################


# # # #########################################################
# # df_target_pred_concat_2 = pd.concat(df_target_pred_parts_2)
# # df_target_pred_2 = df_target_pred_concat_2

# # # Get format column from main `df_features_targets` dataframe

# # # df_features_targets["format"]["color"]["stoich"]
# # # df_format = df_features_targets[("format", "color", "stoich", )]

# # # df_format.name = "color"

# # # # Combining model output and target values
# # # df_target_pred_2 = pd.concat([
# # #     df_format,
# # #     df_target_pred_2,
# # #     ], axis=1)

# # # new_col_list = []
# # # for name_i, row_i in df_target_pred_2.iterrows():
# # #     color_i = df_format.loc[row_i.name]
# # #     new_col_list.append(color_i)
# # #
# # # df_target_pred_2["color"] = new_col_list


# # df_target_pred_2 = df_target_pred_2.dropna()
# # df_target_pred_2 = df_target_pred_2.sort_index()
# # # #########################################################

# # #__|

# # # Calc MAE
# # MAE = df_target_pred["diff_abs"].sum() / df_target_pred["diff"].shape[0]

# # MAE_2 = df_target_pred_2["diff_abs"].sum() / df_target_pred_2["diff"].shape[0]


# # # Calc R2
# # from sklearn.metrics import r2_score
# # coefficient_of_dermination = r2_score(
# #     df_target_pred["y"],
# #     df_target_pred["y_pred"],
# #     )


# # # #####################################################
# # out_dict = dict()
# # # #####################################################
# # out_dict["df_target_pred"] = df_target_pred
# # out_dict["MAE"] = MAE
# # out_dict["R2"] = coefficient_of_dermination
# # out_dict["pca"] = pca
# # out_dict["regression_model_list"] = regression_model_list

# # out_dict["df_target_pred_on_train"] = df_target_pred_2
# # out_dict["MAE_pred_on_train"] = MAE_2
# # out_dict["RM_2"] = RM_2
# # # #####################################################
# # # return(out_dict)
# # # #####################################################
# # # __|

# + jupyter={"source_hidden": true}
# df_features_targets=df_features_targets
# cols_to_use=cols_to_use


# # def process_feature_targets_df(
# #     df_features_targets=None,
# #     cols_to_use=None,
# #     ):
# """
# """
# #| - process_feature_targets_df
# df_j = df_features_targets

# #| - Controlling feature columns to use
# cols_to_keep = []
# cols_to_drop = []
# for col_i in df_j["features"].columns:
#     if col_i in cols_to_use:
#         cols_to_keep.append(("features", col_i))
#     else:
#         cols_to_drop.append(("features", col_i))

# df_j = df_j.drop(columns=cols_to_drop)
# #__|

# # | - Changing target column name to `y`
# new_cols = []
# for col_i in df_j.columns:
#     tmp = 42

#     if col_i[0] == "targets":
#         col_new_i = ("targets", "y")
#         new_cols.append(col_new_i)
#     else:
#         new_cols.append(col_i)

# df_j.columns = pd.MultiIndex.from_tuples(new_cols)
# #__|

# # Splitting dataframe into features and targets dataframe
# df_feat = df_j["features"]


# # Standardizing features
# df_feat = (df_feat - df_feat.mean()) / df_feat.std()
# df_j["features"] = df_feat


# # # #####################################################
# # df_feat_pca = pca_analysis(
# #     df_j["features"],
# #     pca_mode="num_comp",  # 'num_comp' or 'perc'
# #     pca_comp=num_pca_comp,
# #     verbose=False,
# #     )
# #
# # cols_new = []
# # for col_i in df_feat_pca.columns:
# #     col_new_i = ("features", col_i)
# #     cols_new.append(col_new_i)
# # df_feat_pca.columns = pd.MultiIndex.from_tuples(cols_new)
# #
# # df_pca = pd.concat([
# #     df_feat_pca,
# #     df_j[["targets"]],
# #     ], axis=1)

# return(df_j)
# #__|

# + jupyter={"source_hidden": true}
# # df_j["features"]

# df_j.columns

# + jupyter={"source_hidden": true}
# df_j

# + jupyter={"source_hidden": true}
# cols_to_keep = []
# cols_to_drop = []
# for col_i in df_j["features"].columns:
#     if col_i in cols_to_use:
#         cols_to_keep.append(("features", col_i))
#     else:
#         cols_to_drop.append(("features", col_i))

# df_j = df_j.drop(columns=cols_to_drop)

# + jupyter={"source_hidden": true}
# df_j
