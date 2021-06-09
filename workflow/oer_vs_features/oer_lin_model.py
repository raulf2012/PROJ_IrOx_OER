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
# 315
# 325
# 334
# 352 | Sun Jan 31 22:26:52 PST 2021
# 363 | Tue Feb  9 12:43:35 PST 2021
# 374 | Tue Feb 16 15:26:42 PST 2021
# 385 | Sat Feb 20 13:41:31 PST 2021
# 393 | Sat Mar 13 12:13:26 PST 2021
        
df_i.shape
# -

# ### Dropping `p_band_center` for now, very few points

df_i = df_i.drop(columns=[
    ("features", "o", "p_band_center", ),
    # ("features_stan", "o", "p_band_center", ),
    ])

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
filter_cols = [

    ('targets', 'g_o', ''),
    # ('targets', 'g_oh', ''),
    # ('targets', 'g_o_m_oh', ''),


    # ('features', 'oh', 'O_magmom'),
    # ('features', 'oh', 'Ir_magmom'),
    # ('features', 'oh', 'active_o_metal_dist'),
    # ('features', 'oh', 'angle_O_Ir_surf_norm'),
    # ('features', 'oh', 'ir_o_mean'),
    # ('features', 'oh', 'ir_o_std'),
    # ('features', 'oh', 'octa_vol'),

    ('features', 'o', 'O_magmom'),
    ('features', 'o', 'Ir_magmom'),
    ('features', 'o', 'Ir*O_bader'),
    ('features', 'o', 'Ir_bader'),
    ('features', 'o', 'O_bader'),
    ('features', 'o', 'active_o_metal_dist'),
    ('features', 'o', 'angle_O_Ir_surf_norm'),
    ('features', 'o', 'ir_o_mean'),
    ('features', 'o', 'ir_o_std'),
    ('features', 'o', 'octa_vol'),
    ('features', 'o', 'Ir*O_bader/ir_o_mean'),

    ('features', 'dH_bulk', ''),
    ('features', 'volume_pa', ''),
    ('features', 'bulk_oxid_state', ''),
    ('features', 'effective_ox_state', ''),


    # ('features_pre_dft', 'active_o_metal_dist__pre', ''),
    # ('features_pre_dft', 'ir_o_mean__pre', ''),
    # ('features_pre_dft', 'ir_o_std__pre', ''),
    # ('features_pre_dft', 'octa_vol__pre', ''),

    ]



df_i = df_i[filter_cols]

# +
new_cols = []
for col_i in df_i.columns:
    if col_i[0] == "features":
        if col_i[1] in ["o", "oh", "ooh", "bare", ]:
            new_col_i = ("features", col_i[2], )
        elif col_i[2] == "":
            # new_col_i = col_i[1]
            new_col_i = ("features", col_i[1], )
        else:
            print(col_i)
            # new_col_i = "TEMP"
            new_col_i = ("features", "TEMP", )

    elif col_i[0] == "targets":
        # new_col_i = col_i[1]
        new_col_i = ("targets", col_i[1], )

    else:
        print(col_i)
        # new_col_i = "TEMP"
        new_col_i = ("TEMP", "TEMP", )

    new_cols.append(new_col_i)

# new_cols

idx = pd.MultiIndex.from_tuples(new_cols)
df_i.columns = idx

# df_i.columns = new_cols
# -

df_i

# +
ads_i = "o"
feature_ads_i = "oh"

df_j = df_i

# df_j = isolate_target_col(
#     df_i,
#     target_col="g_o",
#     # target_col="g_oh",
#     )

# feature_cols_all = list(df_j["features_stan"][ads_i].columns)
# feature_cols_all = list(df_j["features"][ads_i].columns)

feature_cols_all = df_j["features"].columns.tolist()

format_dict_i = {
    "color": "stoich",
    }

df_j = df_j.dropna()

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
# +
# df_i["features"]["octa_vol"]

# +
# assert False
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

df_j = df_i
df_j = df_j.dropna()

# df_j = isolate_target_col(
#     df_i,
#     target_col="g_oh",
#     )


# feature_cols_all = list(df_j["features_stan"][ads_i].columns)
feature_cols_all = df_j["features"].columns.tolist()


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

frag_i = "vota"
for index_i, row_i in df_ind.iterrows():
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
# df=df_i
# target_col="g_o"

# + jupyter={"source_hidden": true}
# # def isolate_target_col(df, target_col=None):
# """
# """
# #| - isolate_target_col
# df_i = df
# target_col_to_plot = target_col

# cols_tuples = []
# for col_i in list(df_i.columns):
#     if "features_stan" in col_i[0]:
#         cols_tuples.append(col_i)
#     # elif col_i == ("target_cols", target_col_to_plot):
#     elif col_i == ("targets", target_col_to_plot, ""):
#         cols_tuples.append(col_i)
#     elif col_i[0] == "data":
#         cols_tuples.append(col_i)
#     elif col_i[0] == "format":
#         cols_tuples.append(col_i)
#     else:
#         # print("Woops:", col_i)
#         tmp = 42

# df_j = df_i.loc[:, cols_tuples]

# cols_to_check_nan_in = []
# for col_i in df_j.columns:
#     if "features" in col_i[0]:
#         cols_to_check_nan_in.append(col_i)
#     elif "targets" in col_i[0]:
#         cols_to_check_nan_in.append(col_i)


# # df_j = df_j.dropna(subset=cols_to_check_nan_in)  # TEMP

# # df_j = df_j.dropna()

# # return(df_j)
# #__|

# + jupyter={"source_hidden": true}
# df_j

# + jupyter={"source_hidden": true}
# feature_cols_all

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_i.columns.tolist()

# + jupyter={"source_hidden": true}
# df_i.columns.tolist()

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# from sklearn.linear_model import LinearRegression
# import plotly.graph_objs as go
# from proj_data import scatter_marker_size
# from plotting.my_plotly import my_plotly_plot

# + jupyter={"source_hidden": true}
# df = df_j
# # feature_columns = [feature_col_i, ]
# feature_columns = feature_cols_all
# ads = ads_i
# feature_ads = feature_ads_i
# layout = layout
# verbose = True
# save_plot_to_file = True

# # def create_linear_model_plot(
# #     df=None,
# #     feature_columns=None,
# #     ads=None,
# #     feature_ads=None,
# #     format_dict=None,
# #     layout=None,
# #     verbose=True,
# #     save_plot_to_file=False,
# #     ):
# """
# """
# #| - create_linear_model_plot
# # #####################################################
# df_i = df
# features_cols_to_include = feature_columns
# # #####################################################

# #| - Dropping feature columns
# if features_cols_to_include is None or features_cols_to_include == "all":
#     features_cols_to_include = df_i["features_stan"][feature_ads].columns

# cols_to_drop = []
# # for col_i in df_i["features_stan"][feature_ads].columns:
# # for col_i in df_i["features"][feature_ads].columns:
# for col_i in df_i["features"].columns:
#     if col_i not in features_cols_to_include:
#         cols_to_drop.append(col_i)
# df_tmp = copy.deepcopy(df_i)

# for col_i in cols_to_drop:
#     df_i = df_i.drop(columns=[("features_stan", feature_ads, col_i)])

# # feature_cols = list(df_i.features_stan.columns)
# # feature_cols = list(df_i["features_stan"][feature_ads].columns)
# feature_cols = list(df_i["features"].columns)

# # print(feature_cols)


# plot_title = " | ".join(feature_cols)
# plot_title = "Features: " + plot_title
# #__|

# #| - Creating linear model
# # X = df_i["features_stan"][feature_ads].to_numpy()
# # X = X.reshape(-1, len(df_i["features_stan"][feature_ads].columns))

# X = df_i["features"].to_numpy()
# X = X.reshape(-1, len(df_i["features"].columns))

# y = df_i.targets[
#     df_i.targets.columns[0]
#     ]

# model = LinearRegression()
# model.fit(X, y)

# y_predict = model.predict(X)


# #__|


# # | - Put together model output y_pred and y into dataframe
# # y = out_dict["y"]
# # y_predict = out_dict["y_predict"]

# y.name = y.name[0]
# df_model_i = pd.DataFrame(y)

# df_model_i.columns = ["y", ]

# df_model_i["y_predict"] = y_predict


# df_model_i["diff"] = df_model_i["y"] - df_model_i["y_predict"]

# df_model_i["diff_abs"] = np.abs(df_model_i["diff"])
# # __|


# # Calculate Mean Absolute Error (MAE)
# mae = df_model_i["diff_abs"].sum() / df_model_i["diff"].shape[0]



# if verbose:
#     print(20 * "-")
#     print("model.score(X, y):", model.score(X, y))
#     print("Model MAE:", mae)
#     print("")

#     # print(feature_cols)
#     # print(model.coef_)

#     # for i, j in zip(list(df_i["features_stan"][ads].columns), model.coef_):
#     for i, j in zip(list(df_i["features"].columns), model.coef_):
#         print(i, ": ", j, sep="")
#     print(20 * "-")




# #| - Plotting
# data = []


# from methods import get_df_slab
# df_slab = get_df_slab()


# #| - DEPRECATED | Getting colors ready
# # df_slab_tmp = df_slab[["slab_id", "bulk_id"]]
# #
# # bulk_id_slab_id_lists = np.reshape(
# #     df_slab_tmp.to_numpy(),
# #     (
# #         2,
# #         df_slab_tmp.shape[0],
# #         )
# #     )
# #
# # slab_bulk_mapp_dict = dict(zip(
# #     list(bulk_id_slab_id_lists[0]),
# #     list(bulk_id_slab_id_lists[1]),
# #     ))
# #
# #
# # slab_bulk_id_map_dict = dict()
# # for i in df_slab_tmp.to_numpy():
# #     slab_bulk_id_map_dict[i[0]] = i[1]
# #
# # # print("list(bulk_id_slab_id_lists[0]):", list(bulk_id_slab_id_lists[0]))
# # # print("")
# # # print("list(bulk_id_slab_id_lists[1]):", list(bulk_id_slab_id_lists[1]))
# # # print("")
# # # print("slab_bulk_mapp_dict:", slab_bulk_mapp_dict)
# #
# # import random
# # get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# #
# # slab_id_unique_list = df_i.index.to_frame()["slab_id"].unique().tolist()
# #
# # bulk_id_list = []
# # for slab_id_i in slab_id_unique_list:
# #     # bulk_id_i = slab_bulk_mapp_dict[slab_id_i]
# #     bulk_id_i = slab_bulk_id_map_dict[slab_id_i]
# #     bulk_id_list.append(bulk_id_i)
# #
# # color_map_dict = dict(zip(
# #     bulk_id_list,
# #     get_colors(len(slab_id_unique_list)),
# #     ))
# #
# # # Formatting processing
# # color_list = []
# # for name_i, row_i in df_i.iterrows():
# #     # #################################################
# #     slab_id_i = name_i[1]
# #     # #################################################
# #     phase_i = row_i["data"]["phase"][""]
# #     stoich_i = row_i["data"]["stoich"][""]
# #     sum_norm_abs_magmom_diff_i = row_i["data"]["sum_norm_abs_magmom_diff"][""]
# #     norm_sum_norm_abs_magmom_diff_i = row_i["data"]["norm_sum_norm_abs_magmom_diff"][""]
# #     # #################################################
# #
# #     # #################################################
# #     row_slab_i = df_slab.loc[slab_id_i]
# #     # #################################################
# #     bulk_id_i = row_slab_i.bulk_id
# #     # #################################################
# #
# #     bulk_color_i = color_map_dict[bulk_id_i]
# #
# #     if stoich_i == "AB2":
# #         color_list.append("#46cf44")
# #     elif stoich_i == "AB3":
# #         color_list.append("#42e3e3")
# #
# #     # color_list.append(norm_sum_norm_abs_magmom_diff_i)
# #     # color_list.append(bulk_color_i)
# #__|


# #| - Creating parity line
# # x_parity = y_parity = np.linspace(0., 8., num=100, )
# x_parity = y_parity = np.linspace(-2., 8., num=100, )

# trace_i = go.Scatter(
#     x=x_parity,
#     y=y_parity,
#     line=go.scatter.Line(color="black", width=2.),
#     mode="lines")
# data.append(trace_i)
# #__|

# #| - Main Data Trace

# # color_list_i = df_i["format"]["color"][format_dict["color"]]

# trace_i = go.Scatter(
#     y=y,
#     x=y_predict,
#     mode="markers",
#     marker=go.scatter.Marker(
#         # size=12,
#         size=scatter_marker_size,

#         # color=color_list_i,

#         colorscale='Viridis',
#         colorbar=dict(thickness=20),

#         opacity=0.8,

#         ),
#     # text=df_i.name_str,
#     # text=df_i.data.name_str,
#     textposition="bottom center",
#     )
# data.append(trace_i)
# #__|

# #| - Layout
# # y_axis_target_col = df_i.target_cols.columns[0]
# y_axis_target_col = df_i.targets.columns[0]
# y_axis_target_col = y_axis_target_col[0]

# if y_axis_target_col == "g_o":
#     layout.xaxis.title.text = "Predicted ΔG<sub>*O</sub>"
#     layout.yaxis.title.text = "Simulated ΔG<sub>*O</sub>"
# elif y_axis_target_col == "g_oh":
#     layout.xaxis.title.text = "Predicted ΔG<sub>*OH</sub>"
#     layout.yaxis.title.text = "Simulated ΔG<sub>*OH</sub>"
# else:
#     print("Woops isdfsdf8osdfio")

# layout.xaxis.range = [2.5, 5.5]

# layout.showlegend = False

# dd = 0.2
# layout.xaxis.range = [
#     np.min(y_predict) - dd,
#     np.max(y_predict) + dd,
#     ]


# layout.yaxis.range = [
#     np.min(y) - dd,
#     np.max(y) + dd,
#     ]

# layout.title = plot_title
# #__|

# fig = go.Figure(data=data, layout=layout)

# if save_plot_to_file:
#     my_plotly_plot(
#         figure=fig,
#         save_dir=os.path.join(
#             os.environ["PROJ_irox_oer"],
#             "workflow/oer_vs_features",
#             ),
#         plot_name="parity_plot",
#         write_html=True)

# #__|


# # #####################################################
# out_dict = dict()
# # #####################################################
# out_dict["fig"] = fig
# out_dict["df_model_i"] = df_model_i
# out_dict["mae"] = mae

# out_dict["X"] = X
# out_dict["y"] = y
# out_dict["y_predict"] = y_predict
# # #####################################################
# # return(out_dict)
# #__|
