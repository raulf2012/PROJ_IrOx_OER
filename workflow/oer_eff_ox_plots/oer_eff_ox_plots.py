# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Misc OER/Feature Figures
# ---

# ### Import Modules

# + tags=[]
import os
print(os.getcwd())
import sys

import copy
import shutil
from pathlib import Path
from contextlib import contextmanager

# import pickle; import os

import pickle
import  json

import pandas as pd
import numpy as np

from ase import io
from ase.visualize import view

import plotly.graph_objects as go

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis import local_env

# #########################################################
from misc_modules.pandas_methods import drop_columns

from methods import read_magmom_comp_data

import os
import sys

from IPython.display import display

import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 20
# pd.set_option('display.max_rows', None)

# #########################################################
from methods import (
    get_df_jobs_paths,
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_slabs_oh,
    get_df_init_slabs,
    get_df_magmoms,
    get_df_ads,
    get_df_atoms_sorted_ind,
    get_df_rerun_from_oh,
    get_df_slab_simil,
    get_df_active_sites,
    get_df_features_targets,

    get_other_job_ids_in_set,
    read_magmom_comp_data,

    get_df_coord,
    get_df_slabs_to_run,
    get_df_features,
    )

from misc_modules.pandas_methods import reorder_df_columns




import plotly.graph_objs as go

from proj_data import layout_shared
from proj_data import stoich_color_dict, scatter_shared_props
from proj_data import font_axis_title_size__pub, font_tick_labels_size__pub

# + jupyter={"source_hidden": true} tags=[]
df_dft = get_df_dft()
df_job_ids = get_df_job_ids()
df_jobs = get_df_jobs(exclude_wsl_paths=True)
df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_clusters = get_df_jobs_data_clusters()
df_slab = get_df_slab()
df_slab_ids = get_df_slab_ids()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()
df_slabs_oh = get_df_slabs_oh()
df_init_slabs = get_df_init_slabs()
df_magmoms = get_df_magmoms()
df_ads = get_df_ads()
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_rerun_from_oh = get_df_rerun_from_oh()
magmom_data_dict = read_magmom_comp_data()
df_slab_simil = get_df_slab_simil()
df_active_sites = get_df_active_sites()
df_features_targets = get_df_features_targets()
df_slabs_to_run = get_df_slabs_to_run()
df_features = get_df_features()


# + jupyter={"source_hidden": true} tags=[]
def display_df(df, df_name, display_head=True, num_spaces=3):
    print(40 * "*")
    print(df_name)
    print("df_i.shape:", df_i.shape)
    print(40 * "*")

    if display_head:
        display(df.head())

    print(num_spaces * "\n")

df_list = [
    ("df_dft", df_dft),
    ("df_job_ids", df_job_ids),
    ("df_jobs", df_jobs),
    ("df_jobs_data", df_jobs_data),
    ("df_jobs_data_clusters", df_jobs_data_clusters),
    ("df_slab", df_slab),
    ("df_slab_ids", df_slab_ids),
    ("df_jobs_anal", df_jobs_anal),
    ("df_jobs_paths", df_jobs_paths),
    ("df_slabs_oh", df_slabs_oh),
    ("df_magmoms", df_magmoms),
    ("df_ads", df_ads),
    ("df_atoms_sorted_ind", df_atoms_sorted_ind),
    ("df_rerun_from_oh", df_rerun_from_oh),
    ("df_slab_simil", df_slab_simil),
    ("df_active_sites", df_active_sites),
    ]

# for name_i, df_i in df_list:
#     display_df(df_i, name_i)

# print("")
# print("")

# for name_i, df_i in df_list:
#     display_df(
#         df_i,
#         name_i,
#         display_head=False,
#         num_spaces=0)
# + active=""
#
#
# -

plot_height = 12 / 1.61803398875 * 37.795275591 - 60
plot_width = (12 / 1.4) * 37.795275591

df_i = df_features_targets[[
    ("targets", "g_o", "", ),
    ("targets", "g_oh", "", ),
    ("targets", "g_o_m_oh", "", ),

    # ("features", "o", "effective_ox_state", ),
    # ("features", "oh", "effective_ox_state", ),

    ("features", "effective_ox_state", "", ),

    ]]

# +
# target = "g_oh"
target = "g_o"

feature_ads = "oh"
# -

# #### Figuring out the mean DG_O at every eff. ox. state

# +
# eff_ox_vals = df_i[("features", "o", "effective_ox_state")].unique()

# eff_ox_vals = df_i[("features", feature_ads, "effective_ox_state")].unique()
eff_ox_vals = df_i[("features", "effective_ox_state", "", )].unique()

eff_ox_vals = eff_ox_vals.tolist()
eff_ox_vals = list(np.sort(eff_ox_vals)[:-1])

eff_ox_vals_uniq = list(np.unique(
    [np.round(i, 10) for i in eff_ox_vals]
    ))


data_dict_list = []
for eff_ox_i in eff_ox_vals_uniq:

    # col_ind = ("features", feature_ads, "effective_ox_state")
    col_ind = ("features", "effective_ox_state", "", )

    df_tmp_0 = df_i[
        (df_i[col_ind] < eff_ox_i + 0.001) & \
        (df_i[col_ind] > eff_ox_i - 0.001)
        ]

    num_points = df_tmp_0.shape[0]

    ave_ads_e_i = df_tmp_0[("targets", target, "")].mean()

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["eff_ox"] = eff_ox_i
    data_dict_i["ads_e_ave"] = ave_ads_e_i
    data_dict_i["num_points"] = num_points
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df_eff_ox_ave = pd.DataFrame(data_dict_list)
# -

df_eff_ox_ave

# + active=""
# These are the bins that don't have a lot of data, probably shouldn't show them

# +
low_pop_bins = df_eff_ox_ave[df_eff_ox_ave.num_points < 5].eff_ox.tolist()


df_eff_ox_ave_2 = df_eff_ox_ave[df_eff_ox_ave.num_points > 5]

df_eff_ox_ave[df_eff_ox_ave.num_points <= 5]
# -

# ### Drop some indiv. rows

# +
rows_to_drop = [
    # ("sherlock", "vipikema_98", 53.0, ),
    ("slac", "dabipilo_28", 59.0, ),
    ("sherlock", "vipikema_98", 47.0, ),
    ]

if False:
    df_i = df_i.drop(index=rows_to_drop)

# +
# df_i[df_i[("features", "oh", "effective_ox_state", )] == 6.0]

# df_i[df_i[("features", "oh", "effective_ox_state", )] == 6.000000000000001]

# df_i[("features", "oh", "effective_ox_state", )].tolist()

# +
df_j = df_i[[
    ("targets", target, ""),
    # ("features", feature_ads, "effective_ox_state"),
    ("features", "effective_ox_state", "", ),
    ]]

df_j = df_j.dropna()
# -

# df_j[df_j.features.effective_ox_state.isin(low_pop_bins)].shape
df_j = df_j[~df_j.features.effective_ox_state.isin(low_pop_bins)]

trace_mean = go.Scatter(
    x=df_eff_ox_ave_2["eff_ox"],
    y=df_eff_ox_ave_2["ads_e_ave"],
    mode="markers+lines",
    marker=go.scatter.Marker(
        size=6,
        ),
    line=go.scatter.Line(
        color="black",
        width=1,
        ),
    )

# +
scatter_shared_props_cpy = copy.deepcopy(scatter_shared_props)

tmp = scatter_shared_props_cpy.update(dict(marker=dict(size=8, )))

# +
layout_cpy = copy.deepcopy(layout_shared)

if target == "g_oh":
    y_axis_title = "ΔG<sub>OH</sub> (eV)"
    dtick_y = 0.5
elif target == "g_o":
    y_axis_title = "ΔG<sub>O</sub> (eV)"
    dtick_y = 1.

dtick_x = 1

# layout_shared
layout_mine = go.Layout(
    # width=20 * 37.795275591,
    # height=20 / 1.61803398875 * 37.795275591,

    # width=14 * 37.795275591,
    # height=14 / 1.61803398875 * 37.795275591,

    # width=14 * 37.795275591,
    # height=14 / 1.61803398875 * 37.795275591,

    # width=12 * 37.795275591,
    width=plot_width,
    # height=12 / 1.61803398875 * 37.795275591,
    height=plot_height,

    margin=go.layout.Margin(
        b=10, l=10,
        r=10, t=10,
        ),

    showlegend=False,

    xaxis=go.layout.XAxis(
        dtick=dtick_x,
        range=[4.8, 10.2],

        tickfont=go.layout.xaxis.Tickfont(
            size=font_tick_labels_size__pub,
            ),

        title=dict(
            text="Ir Effective Oxidation State",
            font=dict(
                size=font_axis_title_size__pub,
                ),
            )
        ),
    yaxis=go.layout.YAxis(
        dtick=dtick_y,

        tickfont=go.layout.yaxis.Tickfont(
            size=font_tick_labels_size__pub,
            ),

        title=dict(
            # text="ΔG<sub>OH</sub> (eV)",
            text=y_axis_title,
            font=dict(
                size=font_axis_title_size__pub,
                ),
            )
        ),
    )
tmp = layout_cpy.update(layout_mine)







ab2_indices = df_features_targets[
    df_features_targets[("data", "stoich", "", )] == "AB2"].index.tolist()
ab3_indices = df_features_targets[
    df_features_targets[("data", "stoich", "", )] == "AB3"].index.tolist()

df_j_ab3 = df_j.loc[
    df_j.index.intersection(ab3_indices)
    ]

df_j_ab2 = df_j.loc[
    df_j.index.intersection(ab2_indices)
    ]





# x_array = df_j_ab2[('features', feature_ads, 'effective_ox_state')]
x_array = df_j_ab2[('features', 'effective_ox_state', "", )]
y_array = df_j_ab2[('targets', target, '')]

trace_ab2 = go.Scatter(
    x=x_array,
    y=y_array,
    mode="markers",
    marker_color=stoich_color_dict["AB2"],
    )
trace_ab2.update(
    scatter_shared_props_cpy,
    )

# x_array = df_j_ab3[('features', feature_ads, 'effective_ox_state')]
x_array = df_j_ab3[('features', 'effective_ox_state', "", )]
y_array = df_j_ab3[('targets', target, '')]

trace_ab3 = go.Scatter(
    x=x_array,
    y=y_array,
    mode="markers",
    marker_color=stoich_color_dict["AB3"],
    )
trace_ab3.update(
    scatter_shared_props_cpy,
    )




data = [trace_mean, trace_ab2, trace_ab3, ]

fig = go.Figure(data=data, layout=layout_cpy)
fig.show()

# +
# df_j_ab3.features.effective_ox_state.tolist()

df_j_ab3.features

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    plot_name="G_ads__vs__eff_ox__" + target,
    write_html=True,
    write_pdf=True,
    try_orca_write=True,
    )
# -

# ## Creating box plots for Eff ox state

# +
new_cols = []
for col_i in df_j_ab2.columns:
    if col_i[0] == "targets":
        new_col_i = "target"
        new_cols.append(new_col_i)
    elif col_i[0] == "features":
        new_col_i = "eff_ox"
        new_cols.append(new_col_i)
    else:
        new_cols.append("TEMP")

# new_cols

df_j_ab2.columns = new_cols

# df_j_ab3

# +
new_cols = []
for col_i in df_j_ab3.columns:
    if col_i[0] == "targets":
        new_col_i = "target"
        new_cols.append(new_col_i)
    elif col_i[0] == "features":
        new_col_i = "eff_ox"
        new_cols.append(new_col_i)
    else:
        new_cols.append("TEMP")

# new_cols

df_j_ab3.columns = new_cols

# df_j_ab3

# +
df_j_2 = pd.concat([
    df_j_ab2,
    df_j_ab3,
    ],
    axis=0)

# df_j_2

# +
import plotly.express as px

# df = px.data.tips()
# df = df_j_ab3
df = df_j_2

fig = px.box(df,
    x="eff_ox",
    y="target",
    # color="red",
    )
# fig.show()
# -

font_axis_title_size__pub

font_tick_labels_size__pub

# +
layout_cpy = copy.deepcopy(layout_shared)

if target == "g_oh":
    y_axis_title = "ΔG<sub>OH</sub> (eV)"
    dtick_y = 0.5
elif target == "g_o":
    y_axis_title = "ΔG<sub>O</sub> (eV)"
    dtick_y = 1.

dtick_x = 1

# layout_shared
layout_mine = go.Layout(
    # width=20 * 37.795275591,
    # height=20 / 1.61803398875 * 37.795275591,

    # width=14 * 37.795275591,
    # height=14 / 1.61803398875 * 37.795275591,

    # width=14 * 37.795275591,
    # height=14 / 1.61803398875 * 37.795275591,

    # width=12 * 37.795275591,
    width=plot_width,
    # height=12 / 1.61803398875 * 37.795275591,
    height=plot_height,

    margin=go.layout.Margin(
        b=10, l=10,
        r=10, t=10,
        ),

    showlegend=False,

    xaxis=go.layout.XAxis(
        dtick=dtick_x,
        range=[4.8, 10.2],

        tickfont=go.layout.xaxis.Tickfont(
            size=font_tick_labels_size__pub,
            ),

        title=dict(
            text="Ir Effective Oxidation State",
            font=dict(
                size=font_axis_title_size__pub,
                ),
            )
        ),
    yaxis=go.layout.YAxis(

        dtick=dtick_y,

        tickfont=go.layout.yaxis.Tickfont(
            size=font_tick_labels_size__pub,
            ),

        title=dict(
            # text="ΔG<sub>OH</sub> (eV)",
            text=y_axis_title,
            font=dict(
                size=font_axis_title_size__pub,
                ),
            )
        ),
    )
tmp = layout_cpy.update(layout_mine)

fig.update_layout(dict1=layout_cpy)

# +
# fig.data[0].marker.color = "#fc035a"
fig.data[0].marker.color = "#1d71d1"

fig

# +
# type(fig.data[0])

# # go.Box?

# +
# # go.box.Line?

# + jupyter={"source_hidden": true} tags=[]
# fig.update_traces()

fig.update_traces(
    line=dict(
        # color="red",
        width=1,
        ),

    marker=dict(
        size=3,
        # color="green"
        ),
    )

fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    save_dir=None,
    place_in_out_plot=True,
    plot_name="box_plot_eff_ox__" + target,
    write_html=True,
    write_png=False,
    png_scale=6.0,
    write_pdf=True,
    write_svg=False,
    try_orca_write=True,
    verbose=False,
    )

# + active=""
#
#
#

# + jupyter={"source_hidden": true} tags=[]
# import plotly.express as px
# df = px.data.tips()
# fig = px.box(df, x="time", y="total_bill")
# fig.show()

# + jupyter={"source_hidden": true}
# # df_j_ab3[("features", "oh", "effective_ox_state", )] = np.round(
# df_j_ab3[("features", "effective_ox_state", "", )] = np.round(
#     # df_j_ab3[("features", "oh", "effective_ox_state", )],
#     df_j_ab3[("features", "effective_ox_state", "", )],
#     5,
#     )


# # df_j_ab2[("features", "oh", "effective_ox_state", )] = np.round(
# df_j_ab2[("features", "effective_ox_state", "", )] = np.round(
#     # df_j_ab2[("features", "oh", "effective_ox_state", )],
#     df_j_ab2[("features", "effective_ox_state", "", )],
#     5,
#     )

# + jupyter={"source_hidden": true}
# df_j_ab3.columns.tolist()

# ('targets', 'g_o', '')
# ('features', 'oh', 'effective_ox_state')

# + jupyter={"source_hidden": true}
# df_j_ab3
