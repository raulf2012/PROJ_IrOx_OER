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

# ### Import Modules

# +
import os

import numpy as np
import pandas as pd

import pickle

import plotly.graph_objs as go
# -

# ### Read Data

# +
dir_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data")

# #########################################################
path_i = os.path.join(
    dir_i, "out_data/df_oer_si.pickle")
with open(path_i, "rb") as fle:
    df_oer_si = pickle.load(fle)

# #########################################################
path_i = os.path.join(
    dir_i, "out_data/df_ads_e.pickle")
with open(path_i, "rb") as fle:
    df_ads_e = pickle.load(fle)

# +
df_ads_e_index = df_ads_e.index.tolist()
df_oer_si_index = df_oer_si.index.tolist()

shared_indices = []
idx = pd.MultiIndex.from_tuples(
    df_ads_e_index + df_oer_si_index)
for index_i in idx.unique():

    in_df_0 = index_i in df_ads_e_index
    in_df_1 = index_i in df_oer_si_index

    if in_df_0 and in_df_1:
        # print(index_i)
        shared_indices.append(index_i)

len(shared_indices)
# -

df_ads_e_2 = df_ads_e.loc[shared_indices]
df_oer_si_2 = df_oer_si.loc[shared_indices]

# +
trace = go.Scatter(
    y=df_ads_e_2.g_o - df_oer_si_2.g_o,
    mode="markers",
    # name=df_ads_e_2.index.tolist(),
    )
data = [trace]

fig = go.Figure(data=data)
fig.show()
# -

df_ads_e_2[np.abs(df_ads_e_2.g_o - df_oer_si_2.g_o) < 0.05].index.tolist()

# +
# df_ads_e_2[np.abs(df_ads_e_2.g_o - df_oer_si_2.g_o) > 0.05][["g_o", "g_oh", "g_ooh", ]].round(3)

# +
# df_oer_si_2[np.abs(df_ads_e_2.g_o - df_oer_si_2.g_o) > 0.05][["g_o", "g_oh", "g_ooh", ]]

# +
# df_ads_e_2.iloc[[85]]

# +
# df_oer_si_2.iloc[[85]]

# +
# 2.577122	0.632418	3.620811	2.587122	0.952418	3.930811

# +
# df_oer_si_2

# +
# # np.isclose?

# +
for name_i, row_i in df_ads_e.iterrows():
    if np.isclose(row_i.g_o, 1.76, rtol=5e-03, atol=1e-03):
        print(name_i, row_i.g_o)

print(40 * "-")
for name_i, row_i in df_ads_e.iterrows():
    if np.isclose(row_i.g_oh, 0.33, rtol=5e-03, atol=1e-03):
        print(name_i, row_i.g_oh)

print(40 * "-")
for name_i, row_i in df_ads_e.iterrows():
    if np.isclose(row_i.g_ooh, 3.32, rtol=1je-03, atol=1e-03):
        print(name_i, row_i.g_ooh)
# -

df_ads_e.loc[[
    ('columbite', '120', 'O_covered', 0, 1),
    ('columbite', '120', 'O_covered', 2, 3),
    ]]

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # #########################################################
# directory = os.path.join(
#     os.environ["HOME"],
#     "__temp__")
# path_i = os.path.join(directory, "temp_data.pickle")
# with open(path_i, "rb") as fle:
#     data = pickle.load(fle)
# # #########################################################
# df_ads_e = data["df_ads_e"]
# df_oer_si = data["df_oer_si"]
# # #########################################################
