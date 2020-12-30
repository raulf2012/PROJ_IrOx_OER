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

# # TEMP
# ---
#

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import copy

import numpy as np
import pandas as pd
# pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

import plotly.express as px

from sklearn.linear_model import LinearRegression
# -

# # Script Inputs

# +
# target_ads = "o"
target_ads = "oh"

verbose = True
# verbose = False
# -

# # Read Data

# +
from methods import get_df_features_targets
df_features_targets = get_df_features_targets()
df_i = df_features_targets

from methods import get_df_slab
df_slab = get_df_slab()

# Getting phase > 1 slab ids
df_slab_i = df_slab[df_slab.phase > 1]
phase_2_slab_ids = df_slab_i.slab_id.tolist()
# -

# ### Processing dataframe

# +
if target_ads == "o":
    other_ads = "oh"
elif target_ads == "oh":
    other_ads = "o"

df_i = df_i.drop("features_stan", axis=1, level=0)

df_i = df_i.drop(other_ads, axis=1, level=1)
# -

# ### Dropping phase 1 slabs

# +
df_index = df_i.index.to_frame()
df_index_i = df_index[
    df_index.slab_id.isin(phase_2_slab_ids)
    ]

print("Dropping phase 1 slabs")
df_i = df_i.loc[
    df_index_i.index
    ]
# -

df_feat = df_i["features"][target_ads]

# ### Standardize columns

# +
df_features = df_feat

df_features_stan = copy.deepcopy(df_features)
for col_i in df_features_stan.columns:
    max_val = df_features_stan[col_i].max()
    mean_val = df_features_stan[col_i].mean()
    std_val = df_features_stan[col_i].std()
    df_features_stan[col_i] = (df_features_stan[col_i]) / max_val
df_feat_stan = df_features_stan

# + active=""
#
#
#

# +
df = df_feat_stan.cov()

fig = px.imshow(df,
    x=df.columns, y=df.columns,
    )
fig.show()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_cov = df_feat_stan.cov()

# new_cols = []
# for i_cnt, col_i in enumerate(df_cov.columns):
#     col_new_i = (i_cnt, col_i)
#     new_cols.append(col_new_i)

# new_cols

# idx = pd.MultiIndex.from_tuples(new_cols)
# df_cov.columns = idx

# df_cov.index = idx
# df_cov

# + jupyter={"source_hidden": true}
# df_feat_stan.cov

# + jupyter={"source_hidden": true}
# # df.index.set_names("tmp",)
# # # df.index.set_names?
# df.index.name = "inkd"

# + jupyter={"source_hidden": true}
# df.columns.name = "cols"

# + jupyter={"source_hidden": true}
# df

# + jupyter={"source_hidden": true}
# df = px.data.medals_wide(indexed=True)
# df
