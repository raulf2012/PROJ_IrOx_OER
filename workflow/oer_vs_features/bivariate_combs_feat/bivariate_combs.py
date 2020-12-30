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
    # df_features_stan[col_i] = (df_features_stan[col_i] - mean_val) / std_val
    df_features_stan[col_i] = (df_features_stan[col_i]) / max_val
df_feat_stan = df_features_stan

# +
# Create every possible bivariate combination to be tested for feature engineering
from itertools import combinations

column_list = df_feat_stan.columns
interactions = list(combinations(column_list, 2))

interactions


# -

def create_lin_model(
    y=None,
    df_x=None,
    ):
    """
    """
    X_i = df_x_i.to_numpy()
    X_i = X_i.reshape(-1, 1)

    model_i = LinearRegression()
    model_i.fit(X_i, y)

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["model"] = model_i
    out_dict["X"] = X_i
    # #####################################################
    return(out_dict)
    # #####################################################


# +
def local_method_wrap(
    feat_i=None,
    y=None,
    # df_x_i=None,
    df_xy_i=None,
    print_str="",
    ):
    # #####################################################
    # df_x_i = df_xy_i[feat_pair_i[0]]
    df_x_i = df_xy_i[feat_i]
    # #####################################################
    model_dict_i = create_lin_model(y=y, df_x=df_x_i)
    # #####################################################
    model_i = model_dict_i["model"]
    X_i = model_dict_i["X"]
    # #####################################################

    # print("Model 0 Score:", np.round(model_i.score(X_i, y), 3))
    print(print_str + ":", np.round(model_i.score(X_i, y), 3))


for feat_pair_i in interactions:
    print(60 * "=")
    print("feat_pair_i:", feat_pair_i)

    bivar_feat_i = df_feat_stan[feat_pair_i[0]] * df_feat_stan[feat_pair_i[1]]

    feat_name_bivar_i = feat_pair_i[0] + "*" + feat_pair_i[1]
    bivar_feat_i.name = feat_name_bivar_i

    y = df_i["targets"]["g_" + target_ads]

    df_xy_i = pd.concat([
        y,
        bivar_feat_i,
        df_feat_stan[list(feat_pair_i)]
        ], axis=1)
    df_xy_i = df_xy_i.dropna()

    y = df_xy_i["g_" + target_ads]


    df_x_i = df_xy_i[feat_pair_i[0]]
    local_method_wrap(
        feat_i=feat_pair_i[0], y=y, df_xy_i=df_xy_i,
        print_str="Model 0 Score")
    df_x_i = df_xy_i[feat_pair_i[1]]
    local_method_wrap(
        feat_i=feat_pair_i[0], y=y, df_xy_i=df_xy_i,
        print_str="Model 0 Score")
    df_x_i = df_xy_i[feat_name_bivar_i]
    local_method_wrap(
        feat_i=feat_pair_i[0], y=y, df_xy_i=df_xy_i,
        print_str="Model Bivar Score")

    print("")
# -

assert False




# +

# X_bivar = 


# .to_numpy()
# -

df_xy_i

assert False

# + active=""
#
#
# -

# ### Doing same-variable bivariate features
#
# Doesn't seem to do anything for some reason

for feat_i in df_feat_stan.columns:
    print(60 * "=")
    print("feat_i:", feat_i)

    # ^^^^

    bivar_feat_i = df_feat_stan[feat_i] * df_feat_stan[feat_i]

    feat_name_bivar_i = feat_i + "__bivar"
    bivar_feat_i.name = feat_name_bivar_i

    y = df_i["targets"]["g_" + target_ads]

    df_xy_i = pd.concat([
        y,
        bivar_feat_i,
        df_feat_stan[feat_i],
        ], axis=1)
    df_xy_i = df_xy_i.dropna()


    # #########################################################
    X_reg = df_xy_i[feat_i].to_numpy()
    X_reg = X_reg.reshape(-1, 1)

    y = df_xy_i["g_" + target_ads]


    model_reg = LinearRegression()
    model_reg.fit(X_reg, y)

    y_predict = model_reg.predict(X_reg)

    # print(20 * "-")
    print("Model Regular Score:", model_reg.score(X_reg, y))

    # #########################################################
    X_bivar = df_xy_i[feat_name_bivar_i].to_numpy()
    X_bivar = X_bivar.reshape(-1, 1)

    model_bivar = LinearRegression()
    model_bivar.fit(X_bivar, y)

    y_predict = model_bivar.predict(X_bivar)

    # print(20 * "-")
    print("Model Bivar Score:", model_bivar.score(X_bivar, y))

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# print(
#     "Number of rows in df_features_targets:",
#     df_i.shape[0],
#     )

# # 150

# + jupyter={"source_hidden": true}
# # col_i
# # new_col_i
# # col_i + 

# df_features_stan

# + jupyter={"source_hidden": true}
# col_i

# + jupyter={"source_hidden": true}
# df_i["features_stan"]

# + jupyter={"source_hidden": true}
# y

# + jupyter={"source_hidden": true}

# df_i["targets"]

# + jupyter={"source_hidden": true}
#     y = df_i["targets"]["g_" + target_ads]

# df_xy_i = 
# y = df_xy_i["g_" + target_ads]

# + jupyter={"source_hidden": true}
# df_xy_i

# + jupyter={"source_hidden": true}
# model_bivar.get_params()

# model_bivar.coef_

# + jupyter={"source_hidden": true}
# model_reg.coef_

# + jupyter={"source_hidden": true}
# df_xy_i

# + jupyter={"source_hidden": true}

# df_feat_stan
