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

import plotly.graph_objs as go

from IPython.display import display

# #########################################################
from proj_data import (
    scatter_marker_props,
    layout_shared,
    layout_shared,
    stoich_color_dict,
    font_axis_title_size__pub,
    font_tick_labels_size__pub,
    scatter_shared_props,
    )

# #########################################################
from methods import (
    get_df_features_targets,
    get_df_slab,
    get_df_features_targets_seoin,
    )

# #########################################################
from methods_models import run_gp_workflow

# +
sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/model_building"))

from methods_model_building import (
    simplify_df_features_targets,
    run_kfold_cv_wf,
    process_feature_targets_df,
    process_pca_analysis,
    pca_analysis,
    run_regression_wf,
    )
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

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/model_building/gaussian_process")

# ### Script Inputs

# +
target_ads_i = "o"
# target_ads_i = "oh"

feature_ads_i = "o"
# feature_ads_i = "oh"

use_seoin_data = False

if use_seoin_data:
    feature_ads_i = "o"
# -
# ### Read Data

# +
df_features_targets = get_df_features_targets()
df_i = df_features_targets

# #########################################################
df_slab = get_df_slab()

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()

# #########################################################
df_seoin = get_df_features_targets_seoin()
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

# + active=""
#
#
#
#
# -

# # -------------------------

# + active=""
#
#
#
#
# -

# ### Combining My data with Seoin's

# +
df_i = df_i.reset_index()

df_seoin.index = pd.RangeIndex(
    start=df_i.index.max() + 1,
    stop=df_i.index.max() + df_seoin.shape[0] + 1,
    )

df_i["source"] = "mine"
df_seoin["source"] = "seoin"

if use_seoin_data:
    df_comb = pd.concat([
        df_i,
        df_seoin,
        ], axis=0)
else:
    df_comb = pd.concat([
        df_i,
        # df_seoin,
        ], axis=0)

# +
# # TEMP
# print(222 * "TEMP | ")

# df_comb = pd.concat([
#     # df_i,
#     df_seoin,
#     ], axis=0)

# feature_ads_i = "o"
# -

# ### Count number of NaN in each column

# +
data_dict_list = []
for col_i in df_comb.columns:

    num_nan_i = sum(
        df_comb[col_i].isna())

    ads_i = None
    if col_i[1] in ["o", "oh", "ooh", ]:
        tmp = 42
        ads_i = col_i[1]

    data_dict_i = dict()
    data_dict_i["col"] = col_i
    data_dict_i["num_nan"] = num_nan_i
    data_dict_i["col_type"] = col_i[0]
    data_dict_i["ads"] = ads_i
    data_dict_list.append(data_dict_i)

df_nan = pd.DataFrame(data_dict_list)
df_nan = df_nan[df_nan.col_type == "features"]
df_nan = df_nan[df_nan.ads == "o"]


df_nan.sort_values("num_nan", ascending=False)

# +
# df_comb = df_comb.drop(columns=[
#     # ('features', 'o', 'Ir_bader'),
#     # ('features', 'o', 'O_bader'),

#     # ('features', 'o', 'p_band_center'),

#     ('features', 'o', 'Ir*O_bader/ir_o_mean'),
#     ('features', 'o', 'Ir*O_bader'),
#     # ('features', 'o', 'Ir_magmom'),
#     # ('features', 'o', 'O_magmom'),

#     # ('features', 'o', 'ir_o_std'),
#     # ('features', 'o', 'octa_vol'),
#     # ('features', 'o', 'ir_o_mean'),
#     # ('features', 'o', 'active_o_metal_dist'),
#     # ('features', 'o', 'angle_O_Ir_surf_norm'),

#     # ('dH_bulk', ''),
#     # ('volume_pa', ''),
#     # ('bulk_oxid_state', ''),
#     # ('effective_ox_state', ''),

#     ],

#     errors='ignore',
#     )

# +
# df_comb["features"].columns.tolist()

# df_comb.columns.tolist()

# +
# assert False
# -

df_comb = df_comb[[

    ("data", "stoich", ""),

    # ("compenv", "", ""),
    # ("slab_id", "", ""),
    # ("active_site", "", ""),

    ("targets", "g_o", ""),
    ("targets", "g_oh", ""),
    # ("targets", "e_o", ""),
    # ("targets", "e_oh", ""),
    # ("targets", "g_o_m_oh", ""),
    # ("targets", "e_o_m_oh", ""),

    # ("features", "oh", "O_magmom"),
    # ("features", "oh", "Ir_magmom"),
    # ("features", "oh", "active_o_metal_dist"),
    # ("features", "oh", "angle_O_Ir_surf_norm"),
    # ("features", "oh", "ir_o_mean"),
    # ("features", "oh", "ir_o_std"),
    # ("features", "oh", "octa_vol"),





    ("features", "o", "O_magmom"),
    ("features", "o", "Ir_magmom"),
    ("features", "o", "Ir_bader"),
    ("features", "o", "O_bader"),
    ("features", "o", "active_o_metal_dist"),
    ("features", "o", "angle_O_Ir_surf_norm"),
    ("features", "o", "ir_o_mean"),
    ("features", "o", "ir_o_std"),
    ("features", "o", "octa_vol"),
    ("features", "o", "p_band_center"),

    ("features", "dH_bulk", ""),
    ("features", "volume_pa", ""),
    ("features", "bulk_oxid_state", ""),
    ("features", "effective_ox_state", ""),





    # ("features_pre_dft", "active_o_metal_dist__pre", ""),
    # ("features_pre_dft", "ir_o_mean__pre", ""),
    # ("features_pre_dft", "ir_o_std__pre", ""),
    # ("features_pre_dft", "octa_vol__pre", ""),

    # ("source", "", ""),

    ]]

# +
# # Simple Plotly Plot
# import plotly.graph_objs as go

# x_array = df_comb["features"]["effective_ox_state"].tolist()
# y_array = df_comb["targets"]["g_oh"].tolist()

# trace = go.Scatter(
#     x=x_array,
#     y=y_array,
#     mode="markers",
#     )
# data = [trace]

# fig = go.Figure(data=data)
# fig.show()

# +
df_j = simplify_df_features_targets(
    df_comb,
    target_ads=target_ads_i,
    feature_ads=feature_ads_i,
    )

df_format = df_features_targets[("format", "color", "stoich", )]
# -

# ### Creating pre-DFT features

# +
# pre_dft_features = True
pre_dft_features = False

if pre_dft_features:
    df_j = df_j_2
# -

# if False:
if pre_dft_features:
    if verbose:
        print(
            "Feature columns available:"
            "\n",
            20 * "-",
            sep="")
        tmp = [print(i) for i in list(df_j["features"].columns)]

    cols_to_use = list(df_j["features"].columns)

# ### Single feature models

# +
gp_settings = {
    "noise": 0.02542,
    # "noise": 0.12542,
    }

# noise_default = 0.01  # Regularisation parameter.

# Length scale parameter
# sigma_l_default = 0.8  # Original
sigma_l_default = 1.8  # Length scale parameter

sigma_f_default = 0.2337970892240513  # Scaling parameter.




# alpha_default = 2.04987167  # Alpha parameter.

kdict = [

    # Always works better with just one kernel

    # Guassian Kernel (RBF)
    {
        'type': 'gaussian',
        'dimension': 'single',
        'width': sigma_l_default,
        'scaling': sigma_f_default,
        'scaling_bounds': ((0.0001, 10.),),
        },

    # {
    #     'type': 'gaussian',
    #     'dimension': 'single',
    #     'width': sigma_l_default / 1.5,
    #     'scaling': sigma_f_default / 1.5,
    #     'bounds': ((0.0001, 10.),),
    #     'scaling_bounds': ((0.0001, 10.),),
    #     },

    ]
# -

# ### Random Features

if False:
    # TEMP
    print(20 * "RANDOM FEATURES | ")
    print("")
    print(222 * "TEMP | ")

    num_feats = 10
    df_feats = pd.DataFrame(
        np.random.rand(df_j.shape[0], num_feats),
        index=df_j.index,
        columns=pd.MultiIndex.from_product([["features", ], range(num_feats)]),
        )

    df_j = pd.concat([
        df_feats, df_j[["targets"]],
        ], axis=1)

    cols_to_use = list(range(num_feats))

# +
cols_to_use = df_j["features"].columns.tolist()

if True:
    data_dict = dict()
    for num_pca_i in range(1, len(cols_to_use) + 1, 1):
        if verbose:
            print("")
            print(40 * "*")
            print(num_pca_i)

        # #####################################################
        out_dict = run_kfold_cv_wf(
            df_features_targets=df_j,
            cols_to_use=cols_to_use,
            run_pca=True,
            num_pca_comp=num_pca_i,
            k_fold_partition_size=30,
            # k_fold_partition_size=5,
            model_workflow=run_gp_workflow,
            model_settings=dict(
                gp_settings=gp_settings,
                kdict=kdict,
                ),
            )
        # #####################################################
        df_target_pred = out_dict["df_target_pred"]
        MAE = out_dict["MAE"]
        R2 = out_dict["R2"]
        PCA = out_dict["pca"]
        regression_model_list = out_dict["regression_model_list"]

        df_target_pred_on_train = out_dict["df_target_pred_on_train"]
        MAE_pred_on_train = out_dict["MAE_pred_on_train"]
        RM_2 = out_dict["RM_2"]
        # #####################################################

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

            print(
                "MAE (predicting on train set): ",
                np.round(MAE_pred_on_train, 5),
                sep="")

        # #####################################################
        data_dict_i = dict()
        # #####################################################
        data_dict_i["df_target_pred"] = df_target_pred
        data_dict_i["MAE"] = MAE
        data_dict_i["R2"] = R2
        data_dict_i["PCA"] = PCA
        # #####################################################
        data_dict[num_pca_i] = data_dict_i
        # #####################################################

# +
# regression_model_list[3].gp_model
# -

# ### Plotting in-fold predictions

# +
# data_dict_i = data_dict[
#     num_pca_best
#     ]

# df_target_pred = data_dict_i["df_target_pred"]
df_target_pred = df_target_pred_on_train
# data_dict_i["df_target_pred"]

max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()


# #########################################################
color_list = []
# #########################################################
for ind_i, row_i in df_target_pred.iterrows():
    # #####################################################
    row_data_i = df_comb.loc[ind_i]
    # #####################################################
    stoich_i = row_data_i[("data", "stoich", "", )]
    # #####################################################
    color_i = stoich_color_dict.get(stoich_i, "red")
    color_list.append(color_i)
# #########################################################
df_target_pred["color"] = color_list
# #########################################################



dd = 0.1

trace_parity = go.Scatter(
    y=[min_val - 2 * dd, max_val + 2 * dd],
    x=[min_val - 2 * dd, max_val + 2 * dd],
    mode="lines",
    name="Parity line",
    line_color="black",
    )





trace_i = go.Scatter(
    y=df_target_pred["y"],
    x=df_target_pred["y_pred"],
    mode="markers",
    name="CV Regression",
    # opacity=0.8,
    opacity=1.,

    marker=dict(
        color=df_target_pred["color"],
        **scatter_marker_props.to_plotly_json(),
        ),
    )

# +
max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()

dd = 0.1

layout_mine = go.Layout(

    showlegend=True,

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
layout_shared_i = copy.deepcopy(layout_shared)
layout_shared_i = layout_shared_i.update(layout_mine)

# data = [trace_parity, trace_i, trace_j]
data = [trace_parity, trace_i, ]

fig = go.Figure(data=data, layout=layout_shared_i)
if show_plot:
    fig.show()
# -

# ## Breaking down PCA stats

# +
# PCA = data_dict[len(cols_to_use)]["PCA"]
PCA = data_dict[3]["PCA"]

if verbose:
    print("Explained variance percentage")
    print(40 * "-")
    tmp = [print(100 * i) for i in PCA.explained_variance_ratio_]
    print("")

df_pca_comp = pd.DataFrame(
    abs(PCA.components_),
    # columns=list(df_j["features"].columns),
    columns=cols_to_use,
    )

if verbose:
    display(df_pca_comp)
# -

if verbose:
    for i in range(df_pca_comp.shape[0]):
        print(40 * "-")
        print(i)
        print(40 * "-")

        df_pca_comp_i = df_pca_comp.loc[i].sort_values(ascending=False)

        print(df_pca_comp_i.iloc[0:4].to_string())
        print("")

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

# +
trace_i = go.Scatter(
    x=df.index,
    y=df.MAE,
    mode="markers",

    marker=dict(
        **scatter_marker_props.to_plotly_json(),
        ),

    )

data = [trace_i, ]

fig = go.Figure(
    data=data,
    layout=layout_shared_i,
    )

if show_plot:
    fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    plot_name="MAE_vs_PCA_comp",
    save_dir=root_dir,
    write_html=True,
    write_pdf=True,
    try_orca_write=True,
    )
# -

# ## Plotting the best model (optimal num PCA components)

# +
num_pca_best = 3
# num_pca_best = 1

# num_pca_best = 11

# +
data_dict_i = data_dict[
    num_pca_best
    ]

df_target_pred = data_dict_i["df_target_pred"]

max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()
# -

color_list = []
for ind_i, row_i in df_target_pred.iterrows():
    row_data_i = df_comb.loc[ind_i]
    stoich_i = row_data_i[("data", "stoich", "", )]
    color_i = stoich_color_dict.get(stoich_i, "red")
    color_list.append(color_i)
df_target_pred["color"] = color_list

# +
dd = 0.1

trace_parity = go.Scatter(
    y=[min_val - 2 * dd, max_val + 2 * dd],
    x=[min_val - 2 * dd, max_val + 2 * dd],
    mode="lines",
    name="Parity line",
    line_color="black",
    )
# -

trace_i = go.Scatter(
    y=df_target_pred["y"],
    x=df_target_pred["y_pred"],
    mode="markers",
    name="CV Regression",
    # opacity=0.8,
    opacity=1.,

    marker=dict(
        color=df_target_pred["color"],
        # color="grey",
        **scatter_marker_props.to_plotly_json(),
        ),
    )

# # In-fold model (trained on all data, no test/train split)

# +
# df_j = df_j.dropna()

# +
out_dict = run_regression_wf(
    df_features_targets=df_j,
    cols_to_use=cols_to_use,
    df_format=df_format,

    run_pca=True,

    num_pca_comp=num_pca_best,
    model_workflow=run_gp_workflow,

    # model_settings=None,
    model_settings=dict(
        gp_settings=gp_settings,
        kdict=kdict,
        ),

    )

df_target_pred = out_dict["df_target_pred"]
MAE = out_dict["MAE"]
R2 = out_dict["R2"]

if verbose:
    print("MAE:", MAE)
    print("R2:", R2)

# +
max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()

dd = 0.1

layout_mine = go.Layout(

    showlegend=True,

    yaxis=go.layout.YAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Simulated ΔG<sub>*{}</sub>".format(target_ads_i.upper()),
            ),
        ),

    xaxis=go.layout.XAxis(
        range=[min_val - dd, max_val + dd],
        title=dict(
            text="Predicted ΔG<sub>*{}</sub>".format(target_ads_i.upper()),
            ),
        ),

    )


# #########################################################
layout_shared = layout_shared.update(layout_mine)
# -

color_list = []
for ind_i, row_i in df_target_pred.iterrows():
    row_data_i = df_comb.loc[ind_i]
    stoich_i = row_data_i[("data", "stoich", "", )]
    color_i = stoich_color_dict.get(stoich_i, "red")
    color_list.append(color_i)
df_target_pred["color"] = color_list

# +
trace_j = go.Scatter(
    y=df_target_pred["y"],
    x=df_target_pred["y_pred"],
    mode="markers",
    opacity=0.8,
    name="In-fold Regression",

    marker=dict(
        color=df_target_pred["color"],
        **scatter_marker_props.to_plotly_json(),
        ),

    )

data = [trace_parity, trace_i, trace_j]

fig = go.Figure(data=data, layout=layout_shared)
if show_plot:
    fig.show()

# +
tmp = fig.layout.update(
    go.Layout(

        showlegend=False,

        width=12 * 37.795275591,
        height=12 / 1.61803398875 * 37.795275591,

        margin=go.layout.Margin(
            b=10, l=10,
            r=10, t=10,
            ),

        xaxis=go.layout.XAxis(
            tickfont=go.layout.xaxis.Tickfont(
                size=font_tick_labels_size__pub,
                ),

            title=dict(
                # text="Ir Effective Oxidation State",
                font=dict(
                    size=font_axis_title_size__pub,
                    ),
                )
            ),
        yaxis=go.layout.YAxis(
            tickfont=go.layout.yaxis.Tickfont(
                size=font_tick_labels_size__pub,
                ),

            title=dict(
                # text="ΔG<sub>OH</sub> (eV)",
                font=dict(
                    size=font_axis_title_size__pub,
                    ),
                )
            ),

        )
    )

fig_cpy = copy.deepcopy(fig)


data = [trace_parity, trace_i, ]

fig_2 = go.Figure(data=data, layout=fig.layout)


# fig_2

# +
scatter_shared_props_cpy = copy.deepcopy(scatter_shared_props)

tmp = scatter_shared_props_cpy.update(
    marker=dict(
        size=8,
        )
    )

tmp = fig_2.update_traces(patch=dict(
    scatter_shared_props_cpy.to_plotly_json()
    ))
# -

fig_2

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig_2,
    plot_name="GP_model",
    save_dir=root_dir,
    write_html=True,
    write_pdf=True,
    try_orca_write=True,
    )
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("gaussian_proc.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
