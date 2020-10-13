# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import numpy as np
import pandas as pd

import plotly.graph_objs as go

# #########################################################
from methods import get_df_eff_ox
from methods import get_df_ads

# #########################################################
from layout import layout
# -

# # Script Inputs

verbose = True
# verbose = False

# # Read Data

# +
df_eff_ox = get_df_eff_ox()

df_ads = get_df_ads()

# + active=""
#
#
#

# +
x_array = []
y_array = []

data_dict_list = []
for i_cnt, row_i in df_ads.iterrows():
    data_dict_i = dict()

    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    g_o_i = row_i.g_o
    job_id_o_i = row_i.job_id_o
    # active_site_i = row_i.active_site
    # #####################################################

    # #########################################################
    row_ox_i = df_eff_ox[
        (df_eff_ox.compenv == compenv_i) & \
        (df_eff_ox.slab_id == slab_id_i) & \
        (df_eff_ox.active_site == active_site_i) & \
        [True for i in range(len(df_eff_ox))]
        ]
    row_ox_i = row_ox_i.iloc[0]
    # #########################################################
    eff_oxid_state_i = row_ox_i.eff_oxid_state
    # #########################################################

    # #########################################################
    data_dict_i["eff_oxid_state"] = eff_oxid_state_i
    data_dict_i["g_o"] = g_o_i
    data_dict_i["job_id_o"] = job_id_o_i
    data_dict_i["active_site"] = active_site_i
    # #########################################################
    data_dict_list.append(data_dict_i)
    # #########################################################

    x_array.append(eff_oxid_state_i)
    y_array.append(g_o_i)
# -

df = pd.DataFrame(data_dict_list)
df = df.dropna()

# +
from methods import get_df_atoms_sorted_ind

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id", drop=False)

# df_atoms_sorted_ind


for i_cnt, row_i in df.iterrows():
    tmp = 42

    job_id_o_i = row_i.job_id_o
    active_site_i = row_i.active_site
    # print(job_id_o_i)

    row_atoms_i = df_atoms_sorted_ind.loc[job_id_o_i]

    # job_id_o_i
    atoms_sorted_good_i = row_atoms_i.atoms_sorted_good
    atoms_sorted_good_i.write("out_data/" + job_id_o_i + "_" + str(int(active_site_i)).zfill(3) + ".traj")

# +
# df.job_id_o + str(int(df.active_site))

names_list = []
for i_cnt, i in df.iterrows():
    tmp = 42

    name_i = i.job_id_o + "_" + str(int(i.active_site)).zfill(3)
    names_list.append(name_i)

# +
x_array = df.eff_oxid_state
y_array = df.g_o
# names = df.job_id_o
names = names_list

trace = go.Scatter(
    x=x_array,
    y=y_array,

    mode="markers",
#     mode="markers+text",

    name="Markers and Text",
    text=names,
    textposition="bottom center",
    )
data = [trace]

fig = go.Figure(data=data, layout=layout)
fig.show()

# +
from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    plot_name="G_O__vs__ox_state",
    write_html=True,
    # write_png=False,
    # png_scale=6.0,
    # write_pdf=False,
    # write_svg=False,
    # try_orca_write=False,
    # verbose=False,
    )

# + active=""
#
#
#
#

# +
# import chart_studio.plotly as py
# import plotly.graph_objs as go

# import os

# x_array = [0, 1, 2, 3]
# y_array = [0, 1, 2, 3]


# trace = go.Scatter(
#     x=x_array,
#     y=y_array,
#     mode="markers",
#     opacity=0.8,
#     marker=dict(

#         symbol="circle",
#         color='LightSkyBlue',

#         opacity=0.8,

#         # color=z,
#         colorscale='Viridis',
#         colorbar=dict(thickness=20),

#         size=20,
#         line=dict(
#             color='MediumPurple',
#             width=2
#             )
#         ),

#     line=dict(
#         color="firebrick",
#         width=2,
#         dash="dot",
#         ),

#     error_y={
#         "type": 'data',
#         "array": [0.4, 0.9, 0.3, 1.1],
#         "visible": True,
#         },

#     )

# data = [trace]

# fig = go.Figure(data=data)
# fig.show()

# +
# df_ads.set_index(
#     ["compenv", "slab_id", "active_site"],
#     drop=False
#     )
# -


