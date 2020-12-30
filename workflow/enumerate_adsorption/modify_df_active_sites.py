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

# # Manually modify and resave `df_active_sites`
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from methods import (
    get_df_slab,
    get_df_active_sites,
    )

# +
df_slab = get_df_slab()

df_active_sites = get_df_active_sites()

# +
df_slab_i = df_slab[df_slab.phase == 2]

# df_slab_i.head()
print("df_slab_i.shape:", df_slab_i.shape)

df_active_sites_new = df_active_sites.drop(
    index=df_slab_i.index,
    )
# -

df_active_sites_new.shape

assert False

# Pickling data #######################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_active_sites.pickle"), "wb") as fle:
    pickle.dump(df_active_sites_new, fle)
# #####################################################

# + active=""
#
#

# + jupyter={"source_hidden": true}
# df_slab = df_slab.drop(
#     index=df_i.slab_id,
#     )

# df_slab_0 = df_slab_0.drop(
#     index=df_i_0.index,
#     )
