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

# # Creating OER scaling plot from raw data, not my modules
# ---

# ### Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy

import numpy as np

from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# #########################################################
from proj_data import layout_shared as layout_shared_main
from proj_data import scatter_shared_props as scatter_shared_props_main
from proj_data import stoich_color_dict

# #########################################################
from methods import get_df_features_targets

# #########################################################
from layout import layout
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

df_features_targets = get_df_features_targets()

# + active=""
#
#

# +
df_features_targets = df_features_targets.dropna(subset=[
    ("targets", "g_o", ""),
    ("targets", "g_oh", ""),
    ])

# df_targets = df_features_targets["targets"].dropna()
df_targets = df_features_targets["targets"]

x_array = df_targets["g_oh"]
y_array = df_targets["g_o"]

color_array = df_features_targets["format"]["color"]["stoich"]

# +
# print(111 * "TEMP | ")
# print("")

# df_features_targets.columns.tolist()

# df_tmp = df_features_targets.loc[:, 
#     [
#         ('format', 'color', 'stoich'),
#         ('data', 'stoich', ''),
#         ]
#     ]

# for index_i, row_i in df_tmp.iterrows():
#     tmp = 42

#     color_i = row_i["format"]["color"]["stoich"]
#     stoich_i = row_i["data"]["stoich"][""]

#     # print("# ", stoich_i, " '", color_i, "'", sep="")
    
#     if stoich_i == "AB2":
#         if color_i == "#46cf44":
#             tmp = 42
#             # print("AB2 Good")
#         else:
#             print("AB2 Bad")

#     if stoich_i == "AB3":
#         if color_i == "#42e3e3":
#             tmp = 42
#             # print("AB3 Good")
#         else:
#             print("AB3 Bad")
# -

# ### Fitting data

x_poly = np.linspace(x_array.min() - 0.2, x_array.max() + 0.2, num=50)

# +
z_1 = np.polyfit(
    x_array, y_array,
    1,
    )

p_1 = np.poly1d(z_1)

print(
    "Polynomial Fit (1st order): ",
    "\n",
    [np.round(i, 3) for i in list(z_1)],
    sep="")

rmse_i = mean_squared_error(
    y_array,
    [p_1(i) for i in x_array],
    squared=False)

print(
    "RMSE (1st order): ",
    rmse_i,
    sep="")

y_poly_1 = [p_1(i) for i in x_poly]

# +
z_2 = np.polyfit(
    x_array, y_array,
    2,
    )

p_2 = np.poly1d(z_2)

print(
    "Polynomial Fit (2nd order): ",
    "\n",
    [np.round(i, 3) for i in list(z_2)],
    sep="")

rmse_i = mean_squared_error(
    y_array,
    [p_2(i) for i in x_array],
    squared=False)

print(
    "RMSE (2nd order): ",
    rmse_i,
    sep="")

y_poly_2 = [p_2(i) for i in x_poly]
# -

# ### Layout

# +
layout_shared = copy.deepcopy(layout_shared_main)

layout_master = layout_shared.update(
    layout
    )

layout_master["xaxis"]["range"] = [x_array.min() - 0.2, x_array.max() + 0.2]

layout_master["title"] = "*O vs *OH Scaling Plot (1st and 2nd order fits)"
# -

# ### Instantiate scatter plots

# +
trace_poly_1 = go.Scatter(
    x=x_poly, y=y_poly_1,
    mode="lines",
    line_color="grey",
    name="poly_fit (1st order)",
    )

trace_poly_2 = go.Scatter(
    x=x_poly, y=y_poly_2,
    mode="lines",
    line_color="black",
    name="poly_fit (2nd order)",
    )

# +
trace = go.Scatter(
    x=x_array, y=y_array,
    mode="markers",
    # marker_color=color_i,
    marker_color=color_array,
    name="main",
    )

scatter_shared_props = copy.deepcopy(scatter_shared_props_main)

trace = trace.update(
    scatter_shared_props,
    overwrite=False,
    )
# -

# ### Instantiate figure

# +
fig = go.Figure(
    data=[
        trace_poly_1,
        trace_poly_2,
        trace,
        ],
    layout=layout_master,
    )

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis/oer_scaling", 
        "out_plot/oer_scaling__O_vs_OH_plot.json"))
# -

if show_plot:
    fig.show()

# + active=""
# There seems to be some nonlinearities at weak bonding energies

# +
# assert False

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
#
# -

# ## Plotting Histogram

df_ab2 = df_features_targets[df_features_targets["data"]["stoich"] == "AB2"]
df_ab3 = df_features_targets[df_features_targets["data"]["stoich"] == "AB3"]

print(

    # "\n",
    "AB2 ΔG_O Mean: ",
    df_ab2["targets"]["g_o"].mean(),

    "\n",
    "AB3 ΔG_O Mean: ",
    df_ab3["targets"]["g_o"].mean(),


    "\n",
    "diff: ",
    df_ab3["targets"]["g_o"].mean() - df_ab2["targets"]["g_o"].mean(),

    "\n",
    40 * "-",

    "\n",
    "AB2 ΔG_OH Mean: ",
    df_ab2["targets"]["g_oh"].mean(),

    "\n",
    "AB3 ΔG_OH Mean: ",
    df_ab3["targets"]["g_oh"].mean(),

    "\n",
    "diff: ",
    df_ab3["targets"]["g_oh"].mean() - df_ab2["targets"]["g_oh"].mean(),

    sep="")

# +
shared_layout_hist = go.Layout(
    yaxis_title="N",
    barmode="overlay",
    )

shared_trace_hist = dict(
    opacity=0.55,
    nbinsx=15,
    )
# -

# ### Trying to get the number of data in bins to set y-axis range (NOT WORKING SO FAR)

# +
# y_targets_list = [
#     df_ab2.targets.g_oh,
#     # df_ab3.targets.g_oh,
#     # df_ab2.targets.g_o,
#     # df_ab3.targets.g_o,
#     ]

# max_num_data_list = []
# for y_target_i in y_targets_list:
#     width = (y_target_i.max() - y_target_i.min()) / shared_trace_hist["nbinsx"]

#     num_data_in_sliver_list = []
#     for i in np.linspace(y_target_i.min(), y_target_i.max(), 200):

#         i_upper = i + width / 2
#         i_lower = i - width / 2

#         print(i_upper, i_lower)

#         y_in_sliver = y_target_i[
#             (y_target_i < i_upper) & \
#             (y_target_i > i_lower)
#             ]

#         num_data_in_sliver = y_in_sliver.shape[0]
#         #print(num_data_in_sliver)
#         num_data_in_sliver_list.append(num_data_in_sliver)

#     max_num_data_in_sliver_i = np.max(num_data_in_sliver_list)
#     print(max_num_data_in_sliver_i)
#     print("")
#     max_num_data_list.append(max_num_data_in_sliver_i)

# max_max_num_in_sliver = np.max(max_num_data_list)

# max_max_num_in_sliver

# # width = 
# (y_target_i.max() - y_target_i.min()) / shared_trace_hist["nbinsx"]

# # y_targets_list[0]

# # y_in_sliver = 
# y_target_i[
#     (y_target_i < 0.6) & \
#     (y_target_i > 0.4)
#     ]
# -

# ### Instantiate *OH plots

# +
# %%capture

fig_oh = go.Figure()

fig_oh.add_trace(
    go.Histogram(
        x=df_ab2.targets.g_oh,
        marker_color=stoich_color_dict["AB2"],
        name="AB2",
        ).update(dict1=shared_trace_hist)
    )

fig_oh.add_trace(
    go.Histogram(
        x=df_ab3.targets.g_oh,
        marker_color=stoich_color_dict["AB3"],
        name="AB3",
        ).update(dict1=shared_trace_hist)
    )

# #########################################################
# Layout manipulation
layout_shared = copy.deepcopy(layout_shared_main)

layout_shared.update(
    go.Layout(
        # title="TEMP01",
        xaxis=go.layout.XAxis(
            title="ΔG<sub>*OH</sub>",
            ),
        ),
    overwrite=False,
    )

layout_shared.update(shared_layout_hist)
fig_oh.update_layout(dict1=layout_shared)
# -

# ### Instantiate *O plots

# +
# %%capture

fig_o = go.Figure()

fig_o.add_trace(
    go.Histogram(
        x=df_ab2.targets.g_o,
        marker_color=stoich_color_dict["AB2"],
        name="AB2",
        ).update(dict1=shared_trace_hist)
    )

fig_o.add_trace(
    go.Histogram(
        x=df_ab3.targets.g_o,
        marker_color=stoich_color_dict["AB3"],
        name="AB3",
        ).update(dict1=shared_trace_hist)
    )

# #########################################################
# Layout manipulation
layout_shared = copy.deepcopy(layout_shared_main)

layout_shared.update(
    go.Layout(
        # title="",
        xaxis=go.layout.XAxis(
            title="ΔG<sub>*O</sub>",
            ),
        ),
    overwrite=False,
    )

layout_shared.update(shared_layout_hist)
fig_o.update_layout(dict1=layout_shared)
# -

# ### Instantiate subplot

# +
# %%capture

fig = make_subplots(rows=1, cols=2)

for trace_i in fig_o.data:
    fig.add_trace(
        trace_i,
        row=1, col=1,
        )
for trace_i in fig_oh.data:
    fig.add_trace(
        trace_i,
        row=1, col=2,
        )

fig.update_layout(
    height=600,
    width=1000,
    title_text="ΔG<sub>*O</sub> and ΔG<sub>*OH</sub> Histograms (eV)",
    )

fig.update_layout(layout_shared_main)
fig.update_layout(shared_layout_hist)

fig.update_xaxes(
    fig_o.layout["xaxis"],
    row=1, col=1,
    overwrite=False,
    )
fig.update_xaxes(
    fig_oh.layout["xaxis"],
    row=1, col=2,
    overwrite=False,
    )


y_range_ub = 45

fig.update_yaxes(
    fig_o.layout["yaxis"].update(
        range=[0, y_range_ub],
        ),
    row=1, col=1,
    overwrite=False,
    )
fig.update_yaxes(
    fig_oh.layout["yaxis"].update(
        range=[0, y_range_ub],
        ),

    row=1, col=2,
    overwrite=False,
    )
# -

# ### Saving plot to json

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis/oer_scaling", 
        "out_plot/oer_scaling__O_OH_histogram.json"))

if show_plot:
    fig.show()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("oer_scaling.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# stoich_color_dict["AB2"]

# # go.Histogram?

# + jupyter={"source_hidden": true}
# df_features_targets.head()

# df_features_targets.columns.tolist()

# + jupyter={"source_hidden": true}
# color_i

# + jupyter={"source_hidden": true}
# print(len(x_array))
# print(len(y_array))
# print(len(color_i))

# + jupyter={"source_hidden": true}
# df_targets.sort_values("g_oh")
