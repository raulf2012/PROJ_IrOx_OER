# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Manually modify and resave `df_slab`
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from methods import get_df_slab

# +
df_slab = get_df_slab()

df_slab_0 = get_df_slab(mode="almost-final")

# +
# assert False

# +
df_i_0 = df_slab_0[df_slab_0.phase == 2]

df_i = df_slab[df_slab.phase == 2]

# +
# assert False

# +
df_slab = df_slab.drop(
    index=df_i.slab_id,
    )

df_slab_0 = df_slab_0.drop(
    index=df_i_0.index,
    )

# +
# Pickling data #######################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab_final.pickle"), "wb") as fle:
    pickle.dump(df_slab, fle)
# #####################################################

# Pickling data #######################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
    pickle.dump(df_slab_0, fle)
# #####################################################
