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
import plotly.express as px

from IPython.display import display

# #########################################################
from proj_data import scatter_marker_props, layout_shared

# #########################################################
from methods_models import run_gp_workflow

# +
sys.path.insert(0,  os.path.join(
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

# ### Read Data

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

from methods import get_df_slab
df_slab = get_df_slab()

from methods import get_df_SOAP_AS, get_df_SOAP_MS, get_df_SOAP_ave
df_SOAP_AS = get_df_SOAP_AS()
df_SOAP_MS = get_df_SOAP_MS()
df_SOAP_ave = get_df_SOAP_ave()

# #########################################################
df_i = df_features_targets

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
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

# # G_O Model

# +
target_ads_i = "o"

# target_ads_i = "oh"
feature_ads_i = "o"
# + active=""
#
#
#
#
# -

# Drop all features built into df, going to be using new ones
df_i = df_i.drop(columns=["features", "features_stan", ])

# +
# df_SOAP_i = df_SOAP_AS
df_SOAP_i = df_SOAP_MS
# df_SOAP_i = df_SOAP_ave

# df_SOAP_i

# +
new_cols = []
for col_i in df_SOAP_i.columns:
    new_cols.append(("features", "o", col_i))
df_SOAP_i.columns = pd.MultiIndex.from_tuples(new_cols)

df_tmp = df_i


df_tmp_2 = pd.concat([
    df_tmp,
    df_SOAP_i,
    ], axis=1)

df_j_tmp = df_tmp_2

# +
df_j = simplify_df_features_targets(
    df_j_tmp,
    target_ads="o",
    feature_ads="o",
    )

df_format = df_features_targets[("format", "color", "stoich", )]
# -

# ### Removing columns with no variance

# +
df_j_info = df_j.describe()

tmp = (df_j_info.loc["std"] == 0.)

columns_to_drop = []
for key, val in tmp.to_dict().items():
    if val is True:
        columns_to_drop.append(key)

df_j = df_j.drop(columns=columns_to_drop)

df_j = df_j.dropna()

# +
# if verbose:
#     print(
#         "Feature columns available:"
#         "\n",
#         20 * "-",
#         sep="")
#     tmp = [print(i) for i in list(df_j["features"].columns)]

cols_to_use = list(df_j["features"].columns)

# +

# # 'bounds': 

# (5 * (0.0001, 10.),)

# tuple([(0.0001, 10.) for i in range(5)])

# # (0.0001, 10.)

# +
gp_settings = {
    "noise": 0.02542,
    # "noise": 0.12542,
    }

alpha = 0.01

# sigma_l = 0.1
# sigma_f = 0.1

sigma_l = 1.5
sigma_f = 0.1

kdict = [

    # Guassian Kernel (RBF)
    {
        'type': 'gaussian',
        'dimension': 'single',
        # 'dimension': 'features',
        'width': sigma_l,
        # 'width': 3 * [sigma_l, ],
        'scaling': sigma_f,
        'bounds': ((0.0001, 10.),),
        # 'bounds': (5 * (0.0001, 10.),),
        # 'bounds': tuple([(0.0001, 10.) for i in range(3)]),
        'scaling_bounds': ((0.0001, 10.),),
        },


    ]
# -

# if False:
if True:
    data_dict = dict()
    max_pca_num = 0
    num_pca_list = []
    # for num_pca_i in range(1, len(cols_to_use) + 1, 1):
    # for num_pca_i in range(1, 100, 16):
#     for num_pca_i in range(1, 8, 2):
    # for num_pca_i in [4, 8, 15]:
    for num_pca_i in [1, ]:

        if num_pca_i > max_pca_num:
            max_pca_num = num_pca_i

        num_pca_list.append(num_pca_i)

        if verbose:
            print("")
            print(40 * "*")
            print(num_pca_i)


        # #####################################################
        out_dict = run_kfold_cv_wf(
            df_features_targets=df_j,
            cols_to_use=cols_to_use,
            df_format=df_format,
            run_pca=False,
            num_pca_comp=num_pca_i,
            k_fold_partition_size=40,
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

        data_dict_i["df_target_pred"] = df_target_pred_on_train
        data_dict_i["MAE_2"] = MAE_pred_on_train
        data_dict_i["RM_2"] = RM_2
        # #####################################################
        data_dict[num_pca_i] = data_dict_i
        # #####################################################

for i in regression_model_list:
    print(i.gp_model.kernel_list)

# +
# RM_2.gp_model.N_D
# RM_2.gp_model.theta_opt
# RM_2.gp_model.log_marginal_likelihood
# RM_2.gp_model.kernel_list
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
        # color="grey",
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

# ### Breaking down PCA stats

# +
PCA = data_dict[max_pca_num]["PCA"]

if PCA is not None:
    if verbose:
        print("Explained variance percentage")
        print(40 * "-")
        tmp = [print(100 * i) for i in PCA.explained_variance_ratio_]
        print("")

    df_pca_comp = pd.DataFrame(
        abs(PCA.components_),
        columns=cols_to_use,
        )

    # if verbose:
    if False:
        display(df_pca_comp)
# -

# if verbose:
if False:
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
# -

# ## Plotting the best model (optimal num PCA components)

data_dict.keys()

num_pca_best = 1
# num_pca_best = 1

# +
data_dict_i = data_dict[
    num_pca_best
    ]

df_target_pred = data_dict_i["df_target_pred"]

max_val = df_target_pred[["y", "y_pred"]].max().max()
min_val = df_target_pred[["y", "y_pred"]].min().min()

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
out_dict = run_regression_wf(
    df_features_targets=df_j,
    cols_to_use=cols_to_use,
    df_format=df_format,
    run_pca=False,
    num_pca_comp=num_pca_best,
    model_workflow=run_gp_workflow,
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
layout_shared = layout_shared.update(layout_mine)

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
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("gaussian_proc.ipynb")
print(20 * "# # ")
# #########################################################

# +
# from plotting.my_plotly import my_plotly_plot

# my_plotly_plot(
#     figure=fig,
#     save_dir=None,
#     place_in_out_plot=True,
#     plot_name='TEMP_PLOT_NAME',
#     write_html=True,
#     write_png=False,
#     png_scale=6.0,
#     write_pdf=False,
#     write_svg=False,
#     try_orca_write=True,
#     verbose=False,
#     )
