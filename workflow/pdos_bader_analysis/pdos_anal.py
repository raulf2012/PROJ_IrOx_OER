# -*- coding: utf-8 -*-
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

# # PDOS data analysis and plotting
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import plotly.graph_objs as go

import matplotlib.pyplot as plt
from scipy import stats

# #########################################################
from methods import get_df_features_targets

from proj_data import scatter_marker_props, layout_shared
# -

# ### Read Data

df_features_targets = get_df_features_targets()

# + active=""
#
#

# +
# df_features_targets.columns.tolist()

pband_indices = df_features_targets[[
    (
        'features',
        # 'oh',
        'o',
        'p_band_center',
        )
    ]].dropna().index.tolist()


df_i = df_features_targets.loc[
    pband_indices
    ][[
        ("targets", "g_oh", ""),
        ("targets", "g_o", ""),
        ("targets", "g_o_m_oh", ""),

        ("targets", "e_oh", ""),
        ("targets", "e_o", ""),
        ("targets", "e_o_m_oh", ""),

        ("features", "o", "p_band_center"),
        ]]
# -

# pband_indices = 
df_features_targets[[
    (
        'features',
        # 'oh',
        'o',
        'p_band_center',
        )
    ]]

    # ]].dropna().index.tolist()

# +
# assert False

# +
# df_features_targets.shape

# +
# (288, 7)
# (311, 7)
# (312, 7)
# (316, 7)

# df_i.shape

# +
# assert False

# +
# df_i[""]

df = df_i
df = df[
    (df["features", "o", "p_band_center"] > -3.5) &
    (df["features", "o", "p_band_center"] < -2.) &
    # (df[""] == "") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df_i = df
# +
x = df_i["features", "o", "p_band_center"]
# y = df_i["targets", "g_oh", ""]
# y = df_i["targets", "g_o", ""]
y = df_i["targets", "g_o_m_oh", ""]
# y = df_i["targets", "e_o_m_oh", ""]




res = stats.linregress(x, y)
y_new_fit = res.intercept + res.slope * x


def colin_fit(p_i):
    g_o_m_oh_i = 0.94 * p_i + 3.58
    return(g_o_m_oh_i)

trace_colin_fit = go.Scatter(
    x=[-6, 0],
    y=[colin_fit(-6), colin_fit(0)],
    mode="lines",
    name="Colin fit (G_OmOH = 0.94 * p_i + 3.58)",
    )

trace_my_fit = go.Scatter(
    x=x,
    y=y_new_fit,
    mode="lines",
    name="Colin fit (G_OmOH = 0.94 * p_i + 3.58)",
    )

y_new_fit

trace = go.Scatter(
    x=x, y=y,
    mode="markers",
    name="My DFT data",
    )


# -

x_i = x.to_numpy()
X = x_i.reshape(-1, 1)

# +
import numpy as np
from sklearn.linear_model import LinearRegression
# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
# y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)
reg.score(X, y)

print(
reg.coef_,
reg.intercept_,
)

# reg.predict(np.array([[3, 5]]))
y_pred_mine = reg.predict(
    [[-6], [2]],
    )
# -

trace_my_fit = go.Scatter(
    x=[-6, 2],
    y=y_pred_mine,
    mode="lines",
    name="My fit (G_OmOH = 0.75 * p_i + 3.55)",
    )

# +
data = [trace, trace_colin_fit, trace_my_fit]

# data = [trace, trace_colin_fit, trace_my_fit]

# +
layout_mine = go.Layout(

    showlegend=False,

    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="ε<sub>2p</sub>",
            ),
        range=[-6, 0, ]
        ),

    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="ΔE<sub>O-OH</sub>",
            ),
        range=[-3, 4, ]
        ),

    )


# #########################################################
layout_shared_i = layout_shared.update(layout_mine)
# -

fig = go.Figure(data=data, layout=layout_shared_i)
fig.show()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_i

# + jupyter={"source_hidden": true}
# df_features_targets

# + jupyter={"source_hidden": true}
# (0.94 * 0 + 3.58) - (0.94 * 3 + 3.58)

# + jupyter={"source_hidden": true}
# 0.94 * 0.3

# + jupyter={"source_hidden": true}
# res.intercept

# + jupyter={"source_hidden": true}
# res.slope

# + jupyter={"source_hidden": true}
# layout = go.Layout(

#     xaxis=go.layout.XAxis(
#         title=go.layout.xaxis.Title(
#             text="ε<sub>2p</sub>",
#             ),
#         ),

#     yaxis=go.layout.YAxis(
#         title=go.layout.yaxis.Title(
#             text="ΔE<sub>O-OH</sub>",
#             ),
#         ),

#     )
