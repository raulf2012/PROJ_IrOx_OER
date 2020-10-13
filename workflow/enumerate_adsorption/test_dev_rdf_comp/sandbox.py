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
import numpy as np

import plotly.graph_objs as go

from plotting.my_plotly import my_plotly_plot
# -

# # Read Data

# #########################################################
import pickle; import os
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp.pickle")
with open(path_i, "rb") as fle:
    df_rdf_i, df_rdf_j = pickle.load(fle)
# #########################################################

# # Script Inputs

chunk_to_edit = 0
dx = 0.3

# + active=""
#
#
#

# +
df_rdf_j = df_rdf_j.rename(columns={" g(r)": "g"})

# x-axis spacing of data
dr = df_rdf_j.r.tolist()[1] - df_rdf_j.r.tolist()[0]

df_i = df_rdf_j[df_rdf_j.g > 1e-5]

trace = go.Scatter(
    x=df_i.r, y=df_i.g,
    mode="markers")
data = [trace]

fig = go.Figure(data=data)
my_plotly_plot(
    figure=fig,
    plot_name="temp_rds_distr",
    write_html=True)
# fig.show()

# chunk_coord_list = []

chunk_start_coords = []
chunk_end_coords = []

row_i = df_i.iloc[0]
chunk_start_coords.append(row_i.r)

for i in range(1, df_i.shape[0] - 1):
    # #####################################################
    row_i = df_i.iloc[i]
    row_ip1 = df_i.iloc[i + 1]
    row_im1 = df_i.iloc[i - 1]
    # #####################################################
    r_i = row_i.r
    r_ip1 = row_ip1.r
    r_im1 = row_im1.r
    # #####################################################

    # if i == 0:
    #     chunk_coord_list.append(r_i)

    if r_i - r_im1 > 3 * dr:
        chunk_start_coords.append(r_i)

    if r_ip1 - r_i > 3 * dr:
        chunk_end_coords.append(r_i) 

# #########################################################
row_i = df_i.iloc[-1]
chunk_end_coords.append(row_i.r)

chunk_coord_list = []
for i in range(len(chunk_end_coords)):
    
    start_i = chunk_start_coords[i]
    end_i = chunk_end_coords[i]

    # print(
    #     str(np.round(start_i, 2)).zfill(5),
    #     str(np.round(end_i, 2)).zfill(5),
    #     )

    chunk_coord_list.append([
        start_i, end_i
        ])


df_chunks_list = []
for i_cnt, chunk_i in enumerate(chunk_coord_list):

    if i_cnt == chunk_to_edit:
        dx_tmp = dx
    else:
        dx_tmp = 0

    df_j = df_rdf_j[(df_rdf_j.r >= chunk_i[0]) & (df_rdf_j.r <= chunk_i[1])]
    df_j.r += dx_tmp

    df_chunks_list.append(df_j)

import pandas as pd

df_i = pd.concat(df_chunks_list)

# trace = go.Scatter(
#     x=df_i.r, y=df_i.g,
#     mode="markers")
# data = [trace]

# fig = go.Figure(data=data)
# # my_plotly_plot(
# #     figure=fig,
# #     plot_name="temp_rds_distr",
# #     write_html=True)
# fig.show()
# -

df_i
