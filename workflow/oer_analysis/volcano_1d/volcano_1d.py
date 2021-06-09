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

# # OER Volcano Plot
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

sys.path.insert(
    0, os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis"))

# #########################################################
# Python Modules
import numpy as np

import plotly.graph_objs as go

# #########################################################
# My Modules
from oxr_reaction.oxr_plotting_classes.oxr_plot_volcano import Volcano_Plot
from plotting.my_plotly import my_plotly_plot

from proj_data_irox import (
    smart_format_dict,
    gas_molec_dict,
    scaling_dict_ideal,
    )

# #########################################################
# Project Data
from proj_data import (
    stoich_color_dict,
    scatter_marker_size,
    xaxis_layout,
    yaxis_layout,
    scatter_shared_props,
    )
from proj_data import font_tick_labels_size, font_axis_title_size
from proj_data import scaling_dict_mine
from proj_data import font_tick_labels_size, font_axis_title_size, layout_shared

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

# + active=""
#
#

# +
# %%capture

ORR_PLT = get_ORR_PLT()

plot_range = {
    "y": [3.7, 1.2],
    "x": [0.2, 3.3],
    }

smart_format_dict = [
    [{'stoich': 'AB2'}, {'color2': stoich_color_dict["AB2"]}],
    [{'stoich': 'AB3'}, {'color2': stoich_color_dict["AB3"]}],
    ]

VP = Volcano_Plot(
    ORR_PLT,
    x_ax_species="o-oh",  # 'o-oh' or 'oh'
    smart_format_dict=smart_format_dict,
    plot_range=plot_range,
    )

VP.create_volcano_relations_plot()

volcano_legs_data = VP.create_volcano_lines(
    gas_molec_dict=gas_molec_dict,
    scaling_dict=scaling_dict_mine,
    plot_all_legs=False,
    plot_min_max_legs=True,
    trace_priority="bottom",  # 'top' or 'bottom'
    )

data = volcano_legs_data + VP.data_points

layout = VP.get_plotly_layout()
layout = go.Layout(layout)

fig = go.Figure(
    data=data,
    layout=layout,
    )

# +
layout["xaxis"].update(xaxis_layout)
layout["yaxis"].update(yaxis_layout)

layout["xaxis"].update(
    go.layout.XAxis(
        dtick=0.2,
        range=[0.36, 2.5],
        title=go.layout.xaxis.Title(
            text="ΔG<sub>O</sub> - ΔG<sub>OH</sub> (eV)",
            ),
        ),
    )

layout["yaxis"].update(
    go.layout.YAxis(
        # dtick=0.2,

        # range=[1.2, 2.5],
        range=[2.5, 1.2],

        # title=go.layout.xaxis.Title(
        #     text="ΔG<sub>O</sub> - ΔG<sub>OH</sub> (eV)",
        #     ),
        ),
    )

tmp = 42
# -

# #########################################################
for trace_i in data:
    try:
        # trace_i.marker.size = scatter_marker_size
        trace_i.update(scatter_shared_props)
    except:
        pass

# +
layout.update(layout_shared.to_plotly_json())

fig = go.Figure(
    data=data,
    layout=layout,
    )
# -

if show_plot:
    fig.show()

fig.write_json(
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/oer_analysis/volcano_1d",
        "out_plot/volcano_1d.json"))

# + active=""
#
#
# -

# # Replotting with test formatting (SANDBOX)

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()

series_g_OmOH = df_features_targets.targets.g_o - df_features_targets.targets.g_oh

traces_markers = []
for index_i, row_i in df_features_targets.iterrows():
    # #####################################################
    g_oh_i = row_i[("targets", "g_oh", "")]
    g_OmOH_i = series_g_OmOH.loc[index_i]
    # #####################################################
    o_from_oh_i = row_i[("data", "from_oh__o", "")]
    # overpot_i = row_i["data"]["overpot"][""]
    overpot_i = row_i[("data", "overpot", "")]
    # ["data"]["overpot"][""]
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
        # y=[g_oh_i],
        y=[1.23 + overpot_i],
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
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("volcano_1d.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# scaling_dict = scaling_dict_mine

# + jupyter={"source_hidden": true}

# #     any_o_w_as_done = 
# # row_i["data"]["any_o_w_as_done"][""]

# overpot_i = row_i["data"]["overpot"][""]
