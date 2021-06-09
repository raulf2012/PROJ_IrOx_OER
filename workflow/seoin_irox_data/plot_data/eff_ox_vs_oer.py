# -*- coding: utf-8 -*-
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
import sys

import pickle

import numpy as np

import plotly.graph_objs as go

from proj_data import (
    scatter_shared_props,
    shared_axis_dict,
    layout_shared,
    )
# -

# #########################################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data/featurize_data",
    "out_data")
path_i = os.path.join(
    directory,
    "df_features_targets.pickle")
with open(path_i, "rb") as fle:
    df_features_targets = pickle.load(fle)
# #########################################################

# +
layout = go.Layout(

    xaxis=go.layout.XAxis(
        title=dict(
            text="Ir Effective Oxidation State",
            ),
        ),

    yaxis=go.layout.YAxis(
        title=dict(
            text="ΔG<sub>O</sub>",
            ),
        ),

    )

tmp = layout.update(layout_shared)

# +
data = []

trace = go.Scatter(
    x=df_features_targets.effective_ox_state,
    y=df_features_targets.g_o,
    mode="markers",
    )
tmp = trace.update(
    scatter_shared_props
    )
data.append(trace)

fig = go.Figure(data=data, layout=layout)

fig.show()
# -

# ### Violin Plot

# +
import plotly.express as px

fig = px.violin(
    df_features_targets,
    y="g_o",
    x="effective_ox_state",
    box=True,
    points="all",
    )

# +
layout = go.Layout(

    xaxis=go.layout.XAxis(
        title=dict(
            text="Ir Effective Oxidation State",
            ),
        ),

    yaxis=go.layout.YAxis(
        title=dict(
            text="ΔG<sub>O</sub>",
            ),
        ),

    )

tmp = fig.update_layout(layout)

tmp = fig.update_layout(layout_shared)
# -

fig.show()

# ### Computing MAE from trivial Eff. Ox. State model

# +
group_cols = ["effective_ox_state", ]
grouped = df_features_targets.groupby(group_cols)

abs_errors = []
for name, group in grouped:
    abs_errors_i = np.abs(group.g_o - group.g_o.mean()).tolist()
    abs_errors.extend(abs_errors_i)
mae_o = np.mean(abs_errors)

# +
group_cols = ["effective_ox_state", ]
grouped = df_features_targets.groupby(group_cols)

abs_errors = []
for name, group in grouped:
    abs_errors_i = np.abs(group.g_oh - group.g_oh.mean()).tolist()
    abs_errors.extend(abs_errors_i)
mae_oh = np.mean(abs_errors)

# +
group_cols = ["effective_ox_state", ]
grouped = df_features_targets.groupby(group_cols)

abs_errors = []
for name, group in grouped:
    abs_errors_i = np.abs(group.g_ooh - group.g_ooh.mean()).tolist()
    abs_errors.extend(abs_errors_i)
mae_ooh = np.mean(abs_errors)
# -




np.round(
    (mae_o + mae_oh + mae_ooh) / 3,
    3
    )

print(
    "Average MAE: ",
    np.round(
        (mae_o + mae_oh + mae_ooh) / 3, 4
        ),
    sep="")

# ### G_OH vs G_O-OH

# +
layout = go.Layout(

    xaxis=go.layout.XAxis(
        title=dict(
            text="ΔG<sub>O</sub>-ΔG<sub>OH</sub>",
            ),
        ),

    yaxis=go.layout.YAxis(
        title=dict(
            text="ΔG<sub>OH</sub>",
            ),
        ),

    )

tmp = layout.update(layout_shared)

# +
data = []

trace = go.Scatter(
    x=df_features_targets["g_o"] - df_features_targets["g_oh"],
    y=df_features_targets["g_oh"],

    mode="markers",
    )
tmp = trace.update(
    scatter_shared_props
    )
data.append(trace)

fig = go.Figure(data=data, layout=layout)

fig.show()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # Pickling data ###########################################
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/seoin_irox_data/featurize_data",
#     "out_data")
# if not os.path.exists(directory):
#     os.makedirs(directory)

# path_i = os.path.join(directory, "df_features_targets.pickle")
# with open(path_i, "wb") as fle:
#     pickle.dump(df_features_targets, fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# df_features_targets

# + jupyter={"source_hidden": true}
#     scatter_shared_props,
#     shared_axis_dict,
#     layout_shared,
