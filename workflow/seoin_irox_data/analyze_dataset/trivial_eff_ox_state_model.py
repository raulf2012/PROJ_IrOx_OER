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

# # Compute a trivial model of Seoin's data with effective oxidation state
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import numpy as np

# #########################################################
from methods import get_df_features_targets_seoin
# -

# ### Read Data

df_seoin = get_df_features_targets_seoin()

# ### Script Inputs

# +
# feature_ads = "o"

# + active=""
#
#
# -

# ### Main loop

target_ads = "o"

# +
# #########################################################
diff_list = []
# #########################################################
group_cols = [
    ("features", "effective_ox_state", "", ), ]
grouped = df_seoin.groupby(group_cols)
# #########################################################
for name, group in grouped:

    target_col = group["targets"]["g_" + target_ads]

    target_mean = target_col.mean()

    diff = (target_col - target_mean).tolist()

    diff_list.extend(diff)

print(
    "MAE:",
    np.round(
        np.abs(diff_list).mean(),
        4,
        ),
    "eV")
# -

target_ads = "oh"

# +
# #########################################################
diff_list = []
# #########################################################
group_cols = [
    ("features", "effective_ox_state", "", ), ]
grouped = df_seoin.groupby(group_cols)
# #########################################################
for name, group in grouped:

    target_col = group["targets"]["g_" + target_ads]

    target_mean = target_col.mean()

    diff = (target_col - target_mean).tolist()

    diff_list.extend(diff)

print(
    "MAE:",
    np.round(
        np.abs(diff_list).mean(),
        4,
        ),
    "eV")
# -

(0.1522 + 0.1245) / 2

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_seoin.columns

# df_seoin[
#     ("features", "effective_ox_state", "", )
#     ]

# df_seoin
