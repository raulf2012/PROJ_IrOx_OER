# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
    run_kfold_cv_wf__TEMP,
    process_feature_targets_df,
    process_pca_analysis,
    # pca_analysis,
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

using_SOAP = False

# +
# Drop all features built into df, going to be using new ones

if using_SOAP:
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
# df_i

# +
if using_SOAP:
    df_to_use = df_j_tmp
else:
    df_to_use = df_i

df_j = simplify_df_features_targets(
    df_to_use,
    # df_i,
    # df_j_tmp,
    target_ads="o",
    # feature_ads="o",
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
# assert False

# +
# if verbose:
#     print(
#         "Feature columns available:"
#         "\n",
#         20 * "-",
#         sep="")
#     tmp = [print(i) for i in list(df_j["features"].columns)]

cols_to_use = list(df_j["features"].columns)
# -

df_j = df_j.sample(frac=1)

df_j

# if False:
if True:
    data_dict = dict()
    max_pca_num = 0
    num_pca_list = []
    # for num_pca_i in range(1, len(cols_to_use) + 1, 1):
    for num_pca_i in range(1, len(cols_to_use) + 1, 2):
    # for num_pca_i in range(1, 100, 16):
#     for num_pca_i in range(1, 8, 2):
    # for num_pca_i in [4, 8, 15]:
    # for num_pca_i in [1, ]:

        if num_pca_i > max_pca_num:
            max_pca_num = num_pca_i

        num_pca_list.append(num_pca_i)

        if verbose:
            print("")
            print(40 * "*")
            print(num_pca_i)


        # #####################################################
        out_dict = run_kfold_cv_wf__TEMP(
            df_features_targets=df_j,
            cols_to_use=cols_to_use,
            # df_format=df_format,
            run_pca=True,
            num_pca_comp=num_pca_i,
            # k_fold_partition_size=40,
            # k_fold_partition_size=30,
            k_fold_partition_size=60,
            model_workflow=run_gp_workflow,

            # model_settings=None,

            # model_settings=dict(
            #     gp_settings=gp_settings,
            #     kdict=kdict,
            #     ),
            )
        # #####################################################
        df_target_pred = out_dict["df_target_pred"]
        MAE = out_dict["MAE"]
        R2 = out_dict["R2"]
        PCA = out_dict["pca"]
        regression_model_list = out_dict["regression_model_list"]

        df_target_pred_on_train = out_dict["df_target_pred_on_train"]
        MAE_pred_on_train = out_dict["MAE_pred_on_train"]
        # RM_2 = out_dict["RM_2"]
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
        # data_dict_i["RM_2"] = RM_2
        # #####################################################
        data_dict[num_pca_i] = data_dict_i
        # #####################################################

# +
# run_gp_workflow?

# + active=""
#
#
#
#
#
#
#
#
#
#
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
        # color=df_target_pred["color"],
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

num_pca_best = 7
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
        # color=df_target_pred["color"],
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

    # model_settings=dict(
    #     gp_settings=gp_settings,
    #     kdict=kdict,
    #     ),

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
        # color=df_target_pred["color"],
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

# + active=""
#
#
#

# +
# # | - Import  Modules
# import os
# import copy

# import pickle

# import numpy as np
# import pandas as pd

# # SciKitLearn
# from sklearn.decomposition import PCA

# # Catlearn
# from catlearn.regression.gaussian_process import GaussianProcess
# from catlearn.preprocess.clean_data import (
#     clean_infinite,
#     clean_variance,
#     clean_skewness)
# from catlearn.preprocess.scaling import standardize

# from IPython.display import display
# # __|

# +
# from methods_models import run_SVR_workflow

# +
# df_features_targets=df_j
# cols_to_use=cols_to_use
# df_format=df_format
# run_pca=True
# num_pca_comp=3
# k_fold_partition_size=100
# model_workflow=run_SVR_workflow
# model_settings=None,

# # def run_kfold_cv_wf__TEMP(
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

# df_j = process_feature_targets_df(
#     df_features_targets=df_features_targets,
#     cols_to_use=cols_to_use,
#     )

# df_j = df_j.dropna()

# # COMBAK New line
# # print(17 * "TEMP | ")
# df_j = df_j.dropna(axis=1)

# # print(df_j)

# if run_pca:
#     # #####################################################
#     # df_pca = process_pca_analysis(
#     out_dict = process_pca_analysis(
#         df_features_targets=df_j,
#         num_pca_comp=num_pca_comp,
#         )
#     # #####################################################
#     df_pca = out_dict["df_pca"]
#     pca = out_dict["pca"]
#     # #####################################################

#     df_data = df_pca
# else:
#     df_data = df_j
#     pca = None


# #| - Creating k-fold partitions
# x = df_j.index.tolist()

# partitions = []
# for i in range(0, len(x), k_fold_partition_size):
#     slice_item = slice(i, i + k_fold_partition_size, 1)
#     partitions.append(x[slice_item])
# #__|

# #| - Run k-fold cross-validation
# df_target_pred_parts = []
# df_target_pred_parts_2 = []
# regression_model_list = []
# for k_cnt, part_k in enumerate(range(len(partitions))):

#     test_partition = partitions[part_k]

#     train_partition = partitions[0:part_k] + partitions[part_k + 1:]
#     train_partition = [item for sublist in train_partition for item in sublist]


#     # #####################################################
#     # df_test = df_pca.loc[test_partition]
#     # df_train = df_pca.loc[train_partition]
#     df_test = df_data.loc[test_partition]
#     df_train = df_data.loc[train_partition]
#     # #####################################################
#     df_train_features = df_train["features"]
#     df_train_targets = df_train["targets"]
#     df_test_features = df_test["features"]
#     df_test_targets = df_test["targets"]
#     # #####################################################

#     # #####################################################
#     # Using the model on the test set (Normal)
#     # #####################################################
#     out_dict = model_workflow(
#         df_train_features=df_train_features,
#         df_train_targets=df_train_targets,
#         df_test_features=df_test_features,
#         df_test_targets=df_test_targets,
#         model_settings=model_settings,
#         )
#     # #####################################################
#     df_target_pred = out_dict["df_target_pred"]
#     min_val = out_dict["min_val"]
#     max_val = out_dict["max_val"]
#     # RM_1 = out_dict["RegressionModel"]
#     # #####################################################

#     # regression_model_list.append(RM_1)

#     # #####################################################
#     # Using the model on the training set (check for bad model)
#     # #####################################################
#     out_dict_2 = model_workflow(
#         df_train_features=df_train_features,
#         df_train_targets=df_train_targets,
#         df_test_features=df_train_features,
#         df_test_targets=df_train_targets,
#         model_settings=model_settings,
#         # kdict=kdict,
#         )
#     # #####################################################
#     df_target_pred_2 = out_dict_2["df_target_pred"]
#     min_val_2 = out_dict_2["min_val"]
#     max_val_2 = out_dict_2["max_val"]
#     # RM_2 = out_dict_2["RegressionModel"]
#     # #####################################################





#     df_target_pred_parts.append(df_target_pred)
#     df_target_pred_parts_2.append(df_target_pred_2)

# # #########################################################
# df_target_pred_concat = pd.concat(df_target_pred_parts)
# df_target_pred = df_target_pred_concat

# # Get format column from main `df_features_targets` dataframe

# # df_features_targets["format"]["color"]["stoich"]
# # df_format = df_features_targets[("format", "color", "stoich", )]

# df_format.name = "color"

# # Combining model output and target values
# df_target_pred = pd.concat([
#     df_format,
#     df_target_pred,
#     ], axis=1)

# df_target_pred = df_target_pred.dropna()
# # #########################################################


# # #########################################################
# df_target_pred_concat_2 = pd.concat(df_target_pred_parts_2)
# df_target_pred_2 = df_target_pred_concat_2

# # Get format column from main `df_features_targets` dataframe

# # df_features_targets["format"]["color"]["stoich"]
# # df_format = df_features_targets[("format", "color", "stoich", )]

# df_format.name = "color"

# # # Combining model output and target values
# # df_target_pred_2 = pd.concat([
# #     df_format,
# #     df_target_pred_2,
# #     ], axis=1)

# new_col_list = []
# for name_i, row_i in df_target_pred_2.iterrows():
#     color_i = df_format.loc[row_i.name]
#     new_col_list.append(color_i)

# df_target_pred_2["color"] = new_col_list


# df_target_pred_2 = df_target_pred_2.dropna()
# df_target_pred_2 = df_target_pred_2.sort_index()
# # #########################################################

# #__|

# # Calc MAE
# MAE = df_target_pred["diff_abs"].sum() / df_target_pred["diff"].shape[0]

# MAE_2 = df_target_pred_2["diff_abs"].sum() / df_target_pred_2["diff"].shape[0]


# # Calc R2
# from sklearn.metrics import r2_score
# coefficient_of_dermination = r2_score(
#     df_target_pred["y"],
#     df_target_pred["y_pred"],
#     )


# # #####################################################
# out_dict = dict()
# # #####################################################
# out_dict["df_target_pred"] = df_target_pred
# out_dict["MAE"] = MAE
# out_dict["R2"] = coefficient_of_dermination
# out_dict["pca"] = pca
# out_dict["regression_model_list"] = regression_model_list

# out_dict["df_target_pred_on_train"] = df_target_pred_2
# out_dict["MAE_pred_on_train"] = MAE_2
# # out_dict["RM_2"] = RM_2
# # #####################################################
# # return(out_dict)
# # #####################################################
# # __|

# +
# MAE
# MAE_2

# + jupyter={"source_hidden": true}
# df_train_features=df_train_features
# df_train_targets=df_train_targets
# df_test_features=df_test_features
# df_test_targets=df_test_targets
# model_settings=model_settings


# # def run_SVR_workflow(
# #     df_train_features=None,
# #     df_train_targets=None,
# #     df_test_features=None,
# #     df_test_targets=None,
# #     model_settings=None,
# #     # kdict=None,
# #     ):
# """
# """
# # | - run_gp_workflow

# # if model_settings is None:
# #     # GP kernel parameters
# #     gp_settings = {
# #         "noise": 0.02542,
# #         "sigma_l": 2.5,
# #         "sigma_f": 0.8,
# #         "alpha": 0.2,
# #         }
# # else:
# #     gp_settings = model_settings["gp_settings"]

# #| - Setting up Gaussian Regression model

# # Instantiate GP regression model

# # RM = RegressionModel(
# #     df_train=df_train_features,
# #     train_targets=df_train_targets,

# #     df_test=df_test_features,

# #     opt_hyperparameters=True,
# #     gp_settings_dict=gp_settings,
# #     model_settings=model_settings,
# #     uncertainty_type='regular',
# #     verbose=True,
# #     )

# # RM.run_regression()


# # Clean up model dataframe a bit
# # model = RM.model
# # model = model.drop(columns=["acquired"])
# # model.columns = ["y_pred", "err_pred"]

# from sklearn import svm

# # X = [[0, 0], [2, 2]]
# # y = [0.5, 2.5]

# regr = svm.SVR()
# regr.fit(train_x, train_y_standard["y"].to_numpy())

# predicted_y = regr.predict(
#     df_test_features.to_numpy()
#     )


# model_i = pd.DataFrame(
#     predicted_y,
#     columns=["y_pred"],
#     index=df_test_features.index,
#     )


# train_y = df_test_targets["y"]

# y_std = train_y.std()

# if type(y_std) != float and not isinstance(y_std, np.float64):
#     # print("This if is True")
#     y_std = y_std.values[0]

# y_mean = train_y.mean()
# if type(y_mean) != float and not isinstance(y_mean, np.float64):
#     y_mean = y_mean.values[0]

# model_i["y_pred"] = (model_i["y_pred"] * y_std) + y_mean
# # model_i["err"] = (model_i["err"] * y_std)

# # __|

# # Combining model output and target values
# df_target_pred = pd.concat([
#     # df_targets,
#     df_test_targets,
#     model_i,
#     ], axis=1)

# # Drop NaN rows again
# df_target_pred = df_target_pred.dropna()


# # Create columns for y_pred - y and |y_pred - y|
# df_target_pred["diff"] = df_target_pred["y_pred"] - df_target_pred["y"]
# df_target_pred["diff_abs"] = np.abs(df_target_pred["diff"])


# # Get global min/max values (min/max over targets and predictions)
# df_target_pred_i = df_target_pred[["y", "y_pred"]]

# max_val = df_target_pred_i.max().max()
# min_val = df_target_pred_i.min().min()

# max_min_diff = max_val - min_val


# # #####################################################
# out_dict = dict()
# # #####################################################
# out_dict["df_target_pred"] = df_target_pred
# out_dict["min_val"] = min_val
# out_dict["max_val"] = max_val
# # out_dict["RegressionModel"] = RM
# # #####################################################
# # return(out_dict)
# # #####################################################
# # __|

# + jupyter={"source_hidden": true}
# df_target_pred

# + jupyter={"source_hidden": true}
# predicted_y

# model_i

# + jupyter={"source_hidden": true}
# train_y = df_test_targets["y"]

# y_std = train_y.std()

# if type(y_std) != float and not isinstance(y_std, np.float64):
#     # print("This if is True")
#     y_std = y_std.values[0]

# y_mean = train_y.mean()
# if type(y_mean) != float and not isinstance(y_mean, np.float64):
#     y_mean = y_mean.values[0]

# model_i["y_pred"] = (model_i["y_pred"] * y_std) + y_mean
# # model_i["err"] = (model_i["err"] * y_std)

# + jupyter={"source_hidden": true}
# model_i

# + jupyter={"source_hidden": true}
# df_target_pred

# + jupyter={"source_hidden": true}
# train_x = df_train_features.to_numpy()
# # train_y = train_targets
# train_y = df_train_targets
# # TEMP
# # print("train_y.describe():", train_y.describe())
# train_y_standard = (train_y - train_y.mean()) / train_y.std()

# + jupyter={"source_hidden": true}
# from sklearn import svm

# # X = [[0, 0], [2, 2]]
# # y = [0.5, 2.5]

# regr = svm.SVR()
# regr.fit(train_x, train_y_standard["y"].to_numpy())

# predicted_y = regr.predict(
#     df_test_features.to_numpy()
#     )


# model_i = pd.DataFrame(
#     predicted_y,
#     columns=["y_predicted"],
#     index=df_test_features.index,
#     )

# + jupyter={"source_hidden": true}
# df_test_targets

# + jupyter={"source_hidden": true}
# df_test_features.to_numpy()

# + jupyter={"source_hidden": true}

# # 'bounds': 

# (5 * (0.0001, 10.),)

# tuple([(0.0001, 10.) for i in range(5)])

# # (0.0001, 10.)

# + jupyter={"source_hidden": true}
# gp_settings = {
#     "noise": 0.02542,
#     # "noise": 0.12542,
#     }

# alpha = 0.01

# # sigma_l = 0.1
# # sigma_f = 0.1

# sigma_l = 1.5
# sigma_f = 0.1

# kdict = [

#     # Guassian Kernel (RBF)
#     {
#         'type': 'gaussian',
#         'dimension': 'single',
#         # 'dimension': 'features',
#         'width': sigma_l,
#         # 'width': 3 * [sigma_l, ],
#         'scaling': sigma_f,
#         'bounds': ((0.0001, 10.),),
#         # 'bounds': (5 * (0.0001, 10.),),
#         # 'bounds': tuple([(0.0001, 10.) for i in range(3)]),
#         'scaling_bounds': ((0.0001, 10.),),
#         },


#     ]
