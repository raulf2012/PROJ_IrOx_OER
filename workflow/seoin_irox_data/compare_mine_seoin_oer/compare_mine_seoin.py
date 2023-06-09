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
# ---

# +
# TEMP

import os
print(os.getcwd())
import sys

import pickle

import numpy as np

import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression

# #########################################################
from methods import get_df_features_targets
# -

# ### Read Data

# +
df_mine = get_df_features_targets()

# #########################################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data/featurize_data",
    "out_data/df_features_targets.pickle")
with open(path_i, "rb") as fle:
    df_seoin = pickle.load(fle)
# #########################################################

# +
data = []

elec_or_gibbs = "g"

trace_mine = go.Scatter(
    x=df_mine["targets"][elec_or_gibbs + "_oh"],
    y=df_mine["targets"][elec_or_gibbs + "_o"],
    mode="markers",
    )
data.append(trace_mine)

trace_seoin = go.Scatter(
    x=df_seoin["targets"][elec_or_gibbs + "_oh"],
    y=df_seoin["targets"][elec_or_gibbs + "_o"],
    mode="markers",
    )
data.append(trace_seoin)
# -

fig = go.Figure(data=data)
fig.show()

# ## Fitting linear fits to each dataset and checking how well they match

X_predict = np.linspace(0, 1, num=50)

# +
df_seoin_i = df_seoin[
    (df_seoin["targets"]["g_oh"] > -1) & \
    (df_seoin["targets"]["g_oh"] < 0.8)
    ]

df_seoin_i = df_seoin_i.dropna(
    subset=[
        ("targets", "e_o", "", ),
        ("targets", "e_oh", "", ),
        ]
    )

X = df_seoin_i["targets"]["e_oh"].to_numpy().reshape(-1, 1)
y = df_seoin_i["targets"]["e_o"].to_numpy()

reg_seoin = LinearRegression().fit(X, y)

reg_seoin.score(X, y)
reg_seoin.coef_
reg_seoin.intercept_

out_pred_seoin = reg_seoin.predict(X_predict.reshape(-1, 1))

# +
df_mine_i = df_mine[
    (df_mine["targets"]["g_oh"] > -1) & \
    (df_mine["targets"]["g_oh"] < 0.8)
    ]

df_mine_i = df_mine_i.dropna(
    subset=[
        ("targets", "e_o", "", ),
        ("targets", "e_oh", "", ),
        ]
    )

X = df_mine_i["targets"]["e_oh"].to_numpy().reshape(-1, 1)
y = df_mine_i["targets"]["e_o"].to_numpy()

reg_mine = LinearRegression().fit(X, y)

reg_mine.score(X, y)
reg_mine.coef_
reg_mine.intercept_

out_pred_mine = reg_mine.predict(X_predict.reshape(-1, 1))
# -

# ### Plotting the relative fits

# +
data = []

trace_mine = go.Scatter(
    x=X_predict,
    y=out_pred_seoin,
    mode="markers",
    )
data.append(trace_mine)

trace_seoin = go.Scatter(
    x=X_predict,
    y=out_pred_mine,
    mode="markers",
    )
data.append(trace_seoin)

trace_diff = go.Scatter(
    x=X_predict,
    y=out_pred_seoin - out_pred_mine,
    mode="markers",
    )
# data.append(trace_diff)

# +
np.sum(out_pred_seoin - out_pred_mine)

# My Gibbs corr, my O_ref, H_ref
# 2.7103

# All Seoin numbers
# -0.5206

# My Gibbs corr, Seoin O_ref, H_ref
# 2.8610

# +
# 3.547067458398018
# 3.340576306583925
# 3.2307764863428408
# 3.0381335180527884


# 2.9043387357517836
# 2.0950456871501713
# -

fig = go.Figure(data=data)
fig.show()

# + active=""
#
#
#

# + jupyter={}
# df_seoin_i.shape
# df_seoin.shape

# + jupyter={}
# reg.predict(np.array([[3, 5]]))

# + jupyter={}
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

# + jupyter={}
# # df_mine_i = 
# # df_seoin[
# #     (df_mine["targets"]["g_oh"] > -1) & \
# #     (df_mine["targets"]["g_oh"] < 0.8)
# #     ]

# (df_mine["targets"]["g_oh"] > -1)
# #     (df_mine["targets"]["g_oh"] < 0.8)

# + jupyter={}
# X
# df_mine_i["targets"]


# df_mine_i
