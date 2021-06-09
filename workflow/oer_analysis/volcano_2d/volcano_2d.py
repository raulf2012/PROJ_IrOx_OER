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

# # 2D OER Volcano Plot
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

sys.path.insert(
    0, os.path.join(
        os.environ["PROJ_irox"],
        "data"))

sys.path.insert(0, 
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis"))

# #########################################################
# Python Modules
import copy

import numpy as np

import plotly.graph_objs as go

# #########################################################
# My Modules
from oxr_reaction.oxr_plotting_classes.oxr_plot_2d_volcano import Volcano_Plot_2D

from plotting.my_plotly import my_plotly_plot

# #########################################################
# Project Data
from proj_data_irox import (
    smart_format_dict,
    gas_molec_dict,
    scaling_dict_ideal,
    )

from proj_data import (
    stoich_color_dict,
    scatter_marker_size,
    xaxis_layout,
    yaxis_layout,
    font_axis_title_size__pub,
    font_tick_labels_size__pub,
    )
from proj_data import font_tick_labels_size, font_axis_title_size
from proj_data import scaling_dict_mine
from proj_data import scatter_marker_props, scatter_shared_props

# #########################################################

from methods import get_ORR_PLT
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
    "workflow/oer_analysis/volcano_2d")

# ### Script Inputs

save_plot = False
plot_exp_traces = True

# ### Read Data

ORR_PLT = get_ORR_PLT()

# + active=""
#
#
# -

smart_format_dict = [
    [{'stoich': 'AB2'}, {'color2': stoich_color_dict["AB2"]}],
    [{'stoich': 'AB3'}, {'color2': stoich_color_dict["AB3"]}],
    ]

# +
# %%capture

VP = Volcano_Plot_2D(
    ORR_PLT,
    plot_range={
        "x": [+0.0, +3.2],
        "y": [-1.6, +3.2],
        },

    smart_format_dict=smart_format_dict,

    # ooh_oh_scaling_dict={"m": 0.9104, "b": 3.144, },
    ooh_oh_scaling_dict=scaling_dict_mine["ooh"],
    )

data = VP.traces
layout = VP.get_plotly_layout()
# -

font_axis_title_size

scatter_shared_props

# +
# Setting some properties to None, doesn't seem to work with update methods
layout["width"] = None
layout["height"] = None

# Update colorbar font properties
data[0]["colorbar"].update(
    dict(
        titlefont=dict(
            size=font_axis_title_size,
            ),
        tickfont=dict(
            size=font_tick_labels_size,
            ),
        )
    )

layout.update(dict(xaxis=dict(
    dtick=0.2,
    )))

layout["xaxis"].update(xaxis_layout)
layout["yaxis"].update(yaxis_layout)


# #########################################################
for trace_i in data:
    try:
        trace_i.update(scatter_shared_props)


    except:
        pass


layout_override = go.Layout(
    # width=24 * 37.795275591,
    # height=14 * 37.795275591,

    # paper_bgcolor="rgba(255,255,255,1.)",
    paper_bgcolor="rgba(255,255,255,1)",

    # plot_bgcolor="rgba(255,255,255,0.5)",
    showlegend=False,
    xaxis=go.layout.XAxis(
        range=[0.688, 2.268],
        ),
    yaxis=go.layout.YAxis(
        range=[-0.176, 2.311],
        ),
    )

fig = go.Figure(
    data=data,
    layout=layout.update(layout_override))


if show_plot:
    fig.show()
# -

# ### Creating manuscript figure

# +
# #########################################################
# Constants factors
golden_ratio = 1.62
marginInches = 1 / 18
ppi = 96
width_inches = 5
height_inches = width_inches / ( 0.9 * golden_ratio)


fig_cpy = copy.deepcopy(fig)

fig_cpy.layout.width = (width_inches - marginInches) * ppi
fig_cpy.layout.height = (height_inches - marginInches) * ppi


# #########################################################
# Update xaxis/yaxis layout
xaxis_layout_cpy = copy.deepcopy(xaxis_layout)
yaxis_layout_cpy = copy.deepcopy(yaxis_layout)

tmp = xaxis_layout_cpy.update(go.layout.XAxis({
    'tickfont': {'size': font_tick_labels_size__pub},
    'title': {'font': {'size': font_axis_title_size__pub}},
    }))
tmp = yaxis_layout_cpy.update(go.layout.YAxis({
    'tickfont': {'size': font_tick_labels_size__pub},
    'title': {'font': {'size': font_axis_title_size__pub}},
    }))

tmp = fig_cpy.layout.xaxis.update(xaxis_layout_cpy)
tmp = fig_cpy.layout.yaxis.update(yaxis_layout_cpy)


# #########################################################
# Update colorbar font properties
tmp = fig_cpy.data[0]["colorbar"].update(
    dict(
        titlefont=dict(
            size=font_axis_title_size__pub,
            ),
        tickfont=dict(
            size=font_tick_labels_size__pub,
            ),
        )
    )


# #########################################################
scatter_shared_props_cpy = copy.deepcopy(scatter_shared_props)
tmp = scatter_shared_props_cpy.update({"marker": {"size": 6, }})
# for trace_i in data:
for trace_i in fig_cpy.data:
    try:
        trace_i.update(scatter_shared_props_cpy)
    except:
        pass


# #########################################################
my_plotly_plot(
    figure=fig_cpy,
    save_dir=root_dir,
    place_in_out_plot=True,
    plot_name="00_volcano_plot__v",
    write_html=False,
    write_png=False,
    png_scale=6.0,
    write_pdf=True,
    write_svg=False,
    try_orca_write=True,
    verbose=True,
    )
# -

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis/volcano_2d",
        "out_plot/volcano_2d.json"))

# + active=""
#
#

# +
# assert False
# -

# # Replotting with test formatting (SANDBOX)

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

series_g_OmOH = df_features_targets.targets.g_o - df_features_targets.targets.g_oh

# +
traces_markers = []
for index_i, row_i in df_features_targets.iterrows():
    # #####################################################
    g_oh_i = row_i[("targets", "g_oh", "")]
    g_OmOH_i = series_g_OmOH.loc[index_i]
    # #####################################################
    o_from_oh_i = row_i[("data", "from_oh__o", "")]
    # #####################################################

    
    # #####################################################
    # Diagnostic columns, data good or bad
    any_o_w_as_done = row_i["data"]["any_o_w_as_done"][""]
    used_unrelaxed_df_coord__o = row_i["data"]["used_unrelaxed_df_coord__o"][""]
    orig_slab_good__oh = row_i["data"]["orig_slab_good__oh"][""]
    orig_slab_good__o = row_i["data"]["orig_slab_good__o"][""]
    from_oh__o = row_i["data"]["from_oh__o"][""]
    # found_active_Ir__oh = row_i["data"]["found_active_Ir__oh"][""]


    # #####################################################
    lim_step_num_i = row_i["data"]["lim_step_num"][""]


    color_i = "gray"
    opacity_i = 1.
    if True == True and \
       any_o_w_as_done == False or \
       used_unrelaxed_df_coord__o == True or \
       orig_slab_good__oh == False or \
       orig_slab_good__o == False or \
       from_oh__o == False:
       # found_active_Ir__oh == False:
        # ##################################
        # IF STATEMENT | IS SYS A WEIRD ONE?
        color_i = "black"
        # ##################################
    else:
        tmp =  42
        # opacity_i = 0.

    if lim_step_num_i == 1:
        color_i = "#000000"
    elif lim_step_num_i == 2:
        color_i = "#404040"
    elif lim_step_num_i == 3:
        color_i = "#787878"
    elif lim_step_num_i == 4:
        color_i = "#c9c9c9"
    else:
        print(index_i)
        color_i = "red"

    #404040
    #787878
    #c9c9c9


    # if o_from_oh_i:
    #     color_i = "orange"
    # else:
    #     color_i = "purple"









    trace_i = go.Scatter(
        x=[g_OmOH_i],
        y=[g_oh_i],
        mode="markers",
        # marker_color=color_i,
        opacity=opacity_i,
        marker=go.scatter.Marker(
            color=color_i,
            size=14,
            ),
        )
    traces_markers.append(trace_i)



# Use this trace for coloring based on continuous numerical column
trace_tmp = go.Scatter(
    x=series_g_OmOH,
    y=df_features_targets["targets"]["g_oh"],
    mode="markers",

    marker=dict(
        color=df_features_targets["data"]["norm_sum_norm_abs_magmom_diff"],
        colorscale='gray',
        size=14,
        colorbar=go.scatter.marker.ColorBar(
            thickness=20,
            x=1.15,
            ),
        ),
    )






# #########################################################
trace_heatmap = fig.data[0]

traces_new = [trace_heatmap, ]
traces_new.extend(traces_markers)
# traces_new.append(trace_tmp)

fig_2 = go.Figure(data=traces_new, layout=fig.layout)

if show_plot:
    fig_2.show()

# +
# df_features_targets["data"]["lim_step_str"].tolist()

# np.unique(
#     )

# df_features_targets["data"]["lim_step_str"].to_numpy()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("volcano_2d.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
