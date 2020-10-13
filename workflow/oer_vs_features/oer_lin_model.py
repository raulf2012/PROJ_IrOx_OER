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

# # Constructing linear model for OER adsorption energies
# ---
#
# TODO:
#   * Add bulk formation energy as descriptor

# # Import Modules

# + jupyter={}
import os
print(os.getcwd())
import sys

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

import plotly.graph_objs as go

# #########################################################
from methods import (
    get_df_eff_ox,
    get_df_ads,
    get_df_features,
    get_df_jobs,
    )

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

df_features = get_df_features()

df_jobs = get_df_jobs()
# -

feature_cols = df_features.columns.tolist()

# + active=""
#
#

# +
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
    row_jobs_i = df_jobs.loc[job_id_o_i]
    # #########################################################
    att_num_i = row_jobs_i.att_num
    # #########################################################

    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = "o"
    data_dict_i["active_site"] = active_site_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["g_o"] = g_o_i
    data_dict_i["job_id_o"] = job_id_o_i
    data_dict_i["active_site"] = active_site_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df = pd.DataFrame(data_dict_list)
df = df.dropna()
df = df.set_index(["compenv", "slab_id", "ads", "active_site", "att_num"], drop=False)
# -

# compenv	slab_id	ads	active_site	att_num	g_o	job_id_o
df.iloc[0:1]

# # Combine features dataframe with adsorption energy

# +
df_tmp = pd.concat([df, df_features], axis=1)

df_tmp = df_tmp.dropna()

df_tmp = df_tmp.drop(columns=["compenv", "slab_id", "ads", "active_site", "att_num", ])
# -

assert False

for feature_i in feature_cols:
    mean_val = df_tmp[feature_i].mean()

    std_val = df_tmp[feature_i].std()

    df_tmp[feature_i + "_stan"] = (df_tmp[feature_i] - mean_val) / std_val


# +
# for name_i, row_i in df_tmp.iterrows():

#     # #########################################################
#     compenv_i = name_i[0]
#     slab_id_i = name_i[1]
#     ads_i = name_i[2]
#     active_site_i = name_i[3]
#     att_num_i = name_i[4]
#     # #########################################################

#     job_id_o_i = row_i.job_id_o

#     name_i = job_id_o_i + "__" + str(int(active_site_i)).zfill(3)

# +
def method(row_i):
    # #########################################################
    name_i = row_i.name
    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #########################################################
    
    job_id_o_i = row_i.job_id_o

    name_i = job_id_o_i + "__" + str(int(active_site_i)).zfill(3)
    
    return(name_i)

# df_i = df_tmp
df_tmp["name_str"] = df_tmp.apply(
    method,
    axis=1)
# df_tmp = df_i
# -

feature_cols_stan = [i + "_stan" for i in feature_cols]
feature_cols_stan

# +
# X = df_tmp[feature_cols].to_numpy().reshape(-1, len(feature_cols))
X = df_tmp[feature_cols_stan].to_numpy().reshape(-1, len(feature_cols))
y = df_tmp.g_o

model = LinearRegression()
model.fit(X, y)

y_predict = model.predict(X)
# -

# # Plotting

data = []

# +
# x_parity = y_parity = np.linspace(0, 5.5, num=100, )
x_parity = y_parity = np.linspace(0., 8., num=100, )

trace_i = go.Scatter(
    x=x_parity,
    y=y_parity,
    line=go.scatter.Line(color="black", width=2.),
    mode="lines")
data.append(trace_i)
# -

trace_i = go.Scatter(
    y=y,
    x=y_predict,
    mode="markers",
    marker=go.scatter.Marker(
        size=12,
        ),
    # mode="markers+text",
    # name="Markers and Text",
    text=df_tmp.name_str,
    textposition="bottom center",
    )
data.append(trace_i)

# +
# # go.Scatter?

# plotly.graph_objects.scatter.Marker
# # go.scatter.Marker?

# +
layout.xaxis.title.text = "Predicted ΔG<sub>*O</sub>"

layout.yaxis.title.text = "Simulated ΔG<sub>*O</sub>"


layout.xaxis.title.font.size = 25
layout.yaxis.title.font.size = 25

layout.yaxis.tickfont.size = 20
layout.xaxis.tickfont.size = 20

layout.xaxis.range = [2.5, 5.5]

layout.showlegend = False

# layout.xaxis.scaleanchor = "y"

# # go.layout.XAxis?

# +
dd = 0.2
layout.xaxis.range = [
    np.min(y_predict) - dd,
    np.max(y_predict) + dd,
    ]


layout.yaxis.range = [
    np.min(y) - dd,
    np.max(y) + dd,
    ]

# +
# np.min(y)
# np.max(y)

# +
fig = go.Figure(data=data, layout=layout)

from plotting.my_plotly import my_plotly_plot

my_plotly_plot(
    figure=fig,
    plot_name="parity_plot",
    write_html=True)

fig.show()

# +
print("model.score(X, y):", model.score(X, y))
print("")

# print(feature_cols)
# print(model.coef_)

for i, j in zip(feature_cols, model.coef_):
    print(i, ": ", j, sep="")

# +
fragments = [
    "faw",
    ]

for name_i, row_i in df_tmp.iterrows():
    # tmp = 42
    name_str_i = row_i.name_str
    for frag_i in fragments:

        if frag_i in name_str_i:
            print(name_str_i)
            # print("IYUIHUuids")

# name_str_i
# -


