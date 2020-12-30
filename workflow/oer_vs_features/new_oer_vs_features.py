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

# # Plotting OER quantities vs all individual descriptors
# ---
#

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()


import copy

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

import plotly.graph_objs as go
import plotly.express as px

from methods import get_df_features_targets
from methods import get_df_slab

# #########################################################
from layout import layout

# #########################################################
from local_methods import create_linear_model_plot
from local_methods import isolate_target_col
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

# +
# Script Inputs

# verbose = True
# verbose = False
# -

# # Read Data

# +
df_features_targets = get_df_features_targets()
df_m = df_features_targets

df_slab = get_df_slab()
# -

# 150
# 181
print(
    "Number of rows in df_features_targets:",
    df_m.shape[0],
    )

# ### Dropping phase 1 slabs

# +
# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()

df_index = df_m.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

print("Dropping phase 1 slabs")
df_m = df_m.loc[
    df_index_i.index
    ]

# + active=""
#
#
#
# -

# ### Creating separate `df_data` dataframe

# +
df_data = df_m["data"]
df_data = df_data.droplevel(1)

df_data.iloc[0:2]
# -

# ### Filter `df_m` down to features and targets columns

# +
# #########################################################
cols_to_keep = []
target_cols = []
# #########################################################
for col_i in df_m.columns:
    # #####################################################
    lev_0 = col_i[0]
    lev_1 = col_i[1]
    lev_2 = col_i[2]
    # #####################################################

    keep_col = False
    if lev_0 == "features" and lev_1 == "o":
        keep_col = True

    if lev_0 == "targets":
        target_cols.append(col_i)
        keep_col = True

    if keep_col:
        cols_to_keep.append(col_i)

# #########################################################
df_i = df_m[cols_to_keep]
# #########################################################

df_i.iloc[0:2]
# -

# # Plotting everything

df_i["features"]["o"].sort_values("ir_o_mean", ascending=False)

# ### Preparing format dataframe

# +
data_dict_list = []
# for index_i, row_i in df_i_2.iterrows():
for index_i, row_i in df_i.iterrows():
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    index_dict_i = dict(zip(df_i.index.names, index_i))
    # #####################################################

    # #####################################################
    row_data_i = df_features_targets.data.loc[index_i]
    # #####################################################
    stoich_i = row_data_i["stoich"][""]
    # #####################################################

    if stoich_i == "AB2":
        color_i = "orange"
    elif stoich_i == "AB3":
        color_i = "green"
    # #####################################################
    data_dict_i.update(index_dict_i)
    # #####################################################
    # data_dict_i["color"] = color_i
    data_dict_i["stoich"] = stoich_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_format = pd.DataFrame(data_dict_list)
df_format = df_format.set_index(["compenv", "slab_id", "active_site", ])
# #########################################################

# +
feature_ads_i = "o"

for col_i in df_i["features"][feature_ads_i].columns:
    col_tup_i = ("features", feature_ads_i, col_i)

    filter_cols = target_cols + [col_tup_i, ]
    df_i_1 = df_i[filter_cols]

    for target_ads_j in ["o", "oh", ]:
        target_j = "g_" + target_ads_j

        filter_cols = [
            ("targets", target_j, "", ),
            col_tup_i,
            ]
        df_i_2 = df_i_1[filter_cols]
        df_i_2 = df_i_2.dropna()

        
        # #################################################
        # Modifying the columns in preparation of flattening column levels
        new_cols = []
        for col_i in df_i_2.columns:
            if target_j in list(col_i):
                new_col_i = len(col_i) * (target_j, )
                new_cols.append(new_col_i)
            else:
                new_cols.append(col_i)
        idx = pd.MultiIndex.from_tuples(new_cols)
        df_i_2.columns = idx

        # Drop top 2 levels, leaving behind normal column index
        df_i_2.columns = df_i_2.columns.droplevel()
        df_i_2.columns = df_i_2.columns.droplevel()

        df_i_2 = pd.concat([df_i_2, df_format], axis=1)

        # #################################################
        # Plotting
        x_array = df_i_2[col_i[-1]]
        y_array = df_i_2[target_j]

        fig = px.scatter(df_i_2,
            x=col_i[-1],
            y=target_j,
            # color=df_i_2["color"],
            color=df_i_2["stoich"],
            )

        if show_plot:
            fig.show()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("new_oer_vs_features.ipynb")
print(20 * "# # ")
# #########################################################

# +
# px.scatter(df_i_2,
# # px.scatter?

# +
# df_i_2

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# ('features', 'o', )
# 'data'
# 'targets'

# + jupyter={"source_hidden": true}
# x_array = df_i_2["features"][feature_ads_i][col_i]
# y_array = df_i_2["targets"][target_j]

# trace = go.Scatter(
#     x=x_array,
#     y=y_array,
#     mode="markers",
#     )
# data = [trace]

# fig = go.Figure(data=data)
# fig.show()

# + jupyter={"source_hidden": true}
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
