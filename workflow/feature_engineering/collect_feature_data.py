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

# # Collect feature data into master dataframe
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# #########################################################
from methods import get_df_octa_vol, get_df_eff_ox
# -

# # Read feature dataframes

# +
df_octa_vol = get_df_octa_vol()

df_eff_ox = get_df_eff_ox()
# -

# # Setting proper indices for join

# +
# df_eff_ox = df_eff_ox.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ], drop=False)
df_eff_ox = df_eff_ox.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ], drop=True)

# df_octa_vol = df_octa_vol.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ], drop=False)
df_octa_vol = df_octa_vol.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ], drop=True)
# -

# # Combine dataframes

# +
df_list = [
    df_eff_ox,
    df_octa_vol,
    ]

df_features = pd.concat(df_list, axis=1)
df_features.head()
# -

# # Save data to pickle

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering")

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_features.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_features, fle)
# #########################################################

# #########################################################
import pickle; import os
with open(path_i, "rb") as fle:
    df_features = pickle.load(fle)
# #########################################################
# -

from methods import get_df_features
get_df_features()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # #########################################################
# import pickle; import os
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/feature_engineering",
#     "out_data/df_features.pickle")
# with open(path_i, "rb") as fle:
#     df_features = pickle.load(fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# # Pickling data ###########################################
# import os; import pickle
# directory = "out_data"
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_features.pickle"), "wb") as fle:
#     pickle.dump(df_features, fle)
# # #########################################################
