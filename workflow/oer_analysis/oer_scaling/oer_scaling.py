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

# +
# ALL MODULES NEEDED

import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots


# #########################################################
from plotting.my_plotly import my_plotly_plot

# #########################################################
from proj_data import layout_shared as layout_shared_main
from proj_data import scatter_shared_props as scatter_shared_props_main
from proj_data import (
    stoich_color_dict,
    shared_axis_dict,
    font_tick_labels_size,
    )

# #########################################################
from methods import get_df_features_targets

# #########################################################
from layout import layout
# -

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis/oer_scaling",
    )

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

if ("data", "found_active_Ir__oh", "", ) in df_features_targets.columns:
    # Drop systems were the coordination analysis couldn't find the active Ir
    df = df_features_targets
    df = df[
        (df[("data", "found_active_Ir__oh", "", )] == True) &
        (df[("data", "found_active_Ir__o", "", )] == True) &
        [True for i in range(len(df))]
        ]
    df_features_targets = df

# df_targets = df_features_targets["targets"].dropna()
df_targets = df_features_targets["targets"]

x_array = df_targets["g_oh"]
y_array = df_targets["g_o"]

color_array = df_features_targets["format"]["color"]["stoich"]
# -

# ### Building color scale from numeric magmom data

# +
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# float_color_list = df_features_targets["data"]["norm_sum_norm_abs_magmom_diff"]
float_color_list = df_features_targets[("data", "SE__area_J_m2", "")]

floats = [4.5,5.5]
df = pd.DataFrame({
    'arrays':[(1.2, 3.4, 5.6),(1.7, 4.4, 8.1)],
    'floats': floats,
    })

# colormap = cm.jet
colormap = cm.copper
normalize = mcolors.Normalize(
    vmin=float_color_list.min(),
    vmax=float_color_list.max(),
    )

s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)

# s_map.to_rgba(0.1)
# matplotlib.colors.to_hex([ 0.47, 0.0, 1.0, 0.5 ], keep_alpha=False)

color_list = []
for float_i in float_color_list:
    color_rgba_i = s_map.to_rgba(float_i)
    color_hex_i = matplotlib.colors.to_hex(
        color_rgba_i,
        keep_alpha=False,
        )
    color_list.append(color_hex_i)
# -

float_color_list.min()
float_color_list.max()

# ### Fitting data

x_poly = np.linspace(x_array.min() - 0.2, x_array.max() + 0.2, num=50)

# +
z_1 = np.polyfit(
    x_array, y_array,
    1,
    )

p_1 = np.poly1d(z_1)

if verbose:
    print(
        "Polynomial Fit (1st order): ",
        "\n",
        [np.round(i, 3) for i in list(z_1)],
        sep="")

rmse_i = mean_squared_error(
    y_array,
    [p_1(i) for i in x_array],
    squared=False)

if verbose:
    print(
        "RMSE (1st order): ",
        rmse_i,
        sep="")

y_poly_1 = [p_1(i) for i in x_poly]

# +
# #########################################################
df_m = pd.DataFrame()
# #########################################################
df_m["y"] = y_array
df_m["y_pred"] = [p_1(i) for i in x_array]
df_m["diff"] = df_m["y"] - df_m["y_pred"]
df_m["diff_abs"] = np.abs(df_m["diff"])
# #########################################################

MAE_1 = df_m["diff_abs"].sum() / df_m.shape[0]
R2_1 = r2_score(df_m["y"], df_m["y_pred"])

# +
z_2 = np.polyfit(
    x_array, y_array,
    2,
    )

p_2 = np.poly1d(z_2)

if verbose:
    print(
        "Polynomial Fit (2nd order): ",
        "\n",
        [np.round(i, 3) for i in list(z_2)],
        sep="")

rmse_i = mean_squared_error(
    y_array,
    [p_2(i) for i in x_array],
    squared=False)

if verbose:
    print(
        "RMSE (2nd order): ",
        rmse_i,
        sep="")

y_poly_2 = [p_2(i) for i in x_poly]
# -

# ### Figuring out which systems deviate from scaling the most

# +
data_dict_list = []
for name_i, row_i in df_targets.iterrows():
    name_dict_i = dict(zip(
        list(df_targets.index.names),
        name_i))


    g_o_i = row_i[("g_o", "", )]
    g_oh_i = row_i[("g_oh", "", )]

    g_o_scaling_i = p_1(g_oh_i)

    deviation = g_o_scaling_i - g_o_i

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    data_dict_i["deviation"] = deviation
    data_dict_i["deviation_abs"] = np.abs(deviation)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_scal_dev = pd.DataFrame(data_dict_list)
# #########################################################

# +
df_scal_dev = df_scal_dev.set_index(
    ["compenv", "slab_id", "active_site", ],
    drop=False,
    )

df_scal_dev

# +
df_scal_dev.sort_values("deviation_abs", ascending=False).iloc[0:40]

# df_scal_dev.sort_values("deviation_abs", ascending=False).iloc[0:80].index.tolist()

# +
# #########################################################
df_m = pd.DataFrame()
# #########################################################
df_m["y"] = y_array
df_m["y_pred"] = [p_2(i) for i in x_array]
df_m["diff"] = df_m["y"] - df_m["y_pred"]
df_m["diff_abs"] = np.abs(df_m["diff"])
# #########################################################

MAE_2 = df_m["diff_abs"].sum() / df_m.shape[0]
R2_2 = r2_score(df_m["y"], df_m["y_pred"])
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

# ### Annotations

# +
coeff = [np.round(i, 3) for i in list(z_1)]

linear_fit_eqn_str = "ΔG<sub>O</sub> = {}⋅ΔG<sub>OH</sub> + {}".format(*coeff)
MAE_str = "MAE: {}".format(np.round(MAE_1, 3))
R2_str = "R<sup>2</sup>: {} eV".format(np.round(R2_1, 3))

# +
annotations = [

    {
        "font": {"size": font_tick_labels_size},
        "showarrow": False,
        "text": linear_fit_eqn_str,
        "x": 0.01,
        "xanchor": "left",
        "xref": "paper",
        "y": 0.99,
        "yanchor": "top",
        "yref": "paper",
        "yshift": 0.,
        "bgcolor": "white",
        },

    {
        "font": {"size": font_tick_labels_size},
        "showarrow": False,
        "text": R2_str,
        "x": 0.01,
        "xanchor": "left",
        "xref": "paper",
        "y": 0.89,
        "yanchor": "top",
        "yref": "paper",
        "yshift": 0.,
        "bgcolor": "white",
        },

    {
        "font": {"size": font_tick_labels_size},
        "showarrow": False,
        "text": MAE_str,
        "x": 0.01,
        "xanchor": "left",
        "xref": "paper",
        "y": 0.79,
        "yanchor": "top",
        "yref": "paper",
        "yshift": 0.,
        "bgcolor": "white",
        },

    ]

layout_master.annotations = annotations
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
    # marker_color=color_array,
    # marker_color=color_list,
    marker=go.scatter.Marker(
        # color=color_list,
        # color=color_array,
        color=float_color_list,
        colorscale='Viridis',
        size=14,
        colorbar=dict(
            thickness=20,
            len=0.8,
            y=0.36,
            ),
        ),

    name="main",
    )

scatter_shared_props = copy.deepcopy(scatter_shared_props_main)

trace = trace.update(
    scatter_shared_props,
    overwrite=False,
    )
# -

# Pickling data ###########################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/oer_analysis/oer_scaling", 
    "out_data/trace_poly_1.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(trace_poly_1, fle)
# #########################################################

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

# +
# 2.220651	0.754728	

# + active=""
# There seems to be some nonlinearities at weak bonding energies

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

if verbose:
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

shared_layout_hist_cpy = copy.deepcopy(shared_layout_hist)
shared_layout_hist_cpy.update(dict(yaxis=dict(title=dict(text=""))))

# layout_shared.update(shared_layout_hist)
layout_shared.update(shared_layout_hist_cpy)
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


y_range_ub = 60

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

my_plotly_plot(
    figure=fig,
    save_dir=root_dir,
    place_in_out_plot=True,
    plot_name="oer_histogram_gO_gOH",
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=False,
    write_svg=False,
    try_orca_write=False,
    verbose=False,
    )

if show_plot:
    fig.show()

# ### Creating combined scaling and histogram plot

# +
df_concat = pd.concat([
    df_features_targets[("targets", "g_o", "")],
    df_features_targets[("targets", "g_oh", "")],
    df_features_targets[("data", "stoich", "")],
    ], axis=1)


col_map_dict = {
    ('targets', 'g_oh', ''): "g_oh",
    ('targets', 'g_o', ''): "g_o",
    ('data', 'stoich', ''): "stoich",
    }

new_cols = []
for col_i in df_concat.columns.tolist():
    tmp = 42

    print(col_i)

    new_col_i = col_map_dict[col_i]

    new_cols.append(new_col_i)

new_cols

df_concat.columns = new_cols
# -

df_concat.head()

# +
fig = px.scatter(df_concat,
    x="g_oh",
    y="g_o",
    color="stoich",
    color_discrete_map=stoich_color_dict,
    marginal_x="histogram",
    marginal_y="histogram",
    )


# fig.show()
# -

tmp = fig.layout.update(
    layout_shared_main
    )
tmp = fig.layout.update(
    layout
    )

fig.layout.showlegend = False

# +
layout_keys_0 = list(fig.layout.to_plotly_json().keys())

xaxis_keys = [i for i in layout_keys_0 if "xaxis" in i]
yaxis_keys = [i for i in layout_keys_0 if "yaxis" in i]

for x_axis_i in xaxis_keys:
    x_axis_num = x_axis_i[5:]
    
    if x_axis_num == "":
        x_axis_num_int = 1
    else:
        x_axis_num_int = int(x_axis_num)



    for y_axis_i in yaxis_keys:
        y_axis_num = y_axis_i[5:]

        if y_axis_num == "":
            y_axis_num_int = 1
        else:
            y_axis_num_int = int(y_axis_num)

        # print(x_axis_num_int, y_axis_num_int)
        # print(y_axis_num_int)

        # fig.layout.update(layout_master)

        fig.update_xaxes(
            patch=shared_axis_dict,
            selector=None,
            overwrite=False,
            row=y_axis_num_int,
            col=x_axis_num_int,
            )
        fig.update_yaxes(
            patch=shared_axis_dict,
            selector=None,
            overwrite=False,
            row=y_axis_num_int,
            col=x_axis_num_int,
            )
# -

my_plotly_plot(
    figure=fig,
    save_dir=root_dir,
    place_in_out_plot=True,
    plot_name="oer_scaling_w_histogram",
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=False,
    write_svg=False,
    try_orca_write=False,
    verbose=False,
    )

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
# color_array

# # go.scatter.marker.ColorBar?

# + jupyter={"source_hidden": true}
# df_features_targets[("data", "SE__area_J_m2", "")]

# + jupyter={"source_hidden": true}
# assert False
