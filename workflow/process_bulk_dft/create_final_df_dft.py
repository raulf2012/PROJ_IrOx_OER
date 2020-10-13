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

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

# #########################################################
from misc_modules.pandas_methods import drop_columns

# +
# #####################################################
path_i = os.path.join(
    os.environ["PROJ_DATA"],
    "PROJ_IrOx_OER/active_learning_proj_data",
    "df_dft_final_no_dupl.pickle")
with open(path_i, "rb") as fle:
    df_dft = pickle.load(fle)

# #####################################################
df_dft = drop_columns(
    df=df_dft,
    columns=["atoms", "form_e_chris", "id_old", "path", "id", "source", "energy", ],
    keep_or_drop="drop")

df_dft = df_dft.sort_values("dH")
# -

path_i = os.path.join(
    "out_data",
    "atoms_dict.pickle")
with open(path_i, "rb") as fle:
    atoms_dict = pickle.load(fle)

# Create new atoms column from `atoms_dict`

# +
df_dft["atoms"] = df_dft.index.map(atoms_dict)

# df_dft

# +
from misc_modules.pandas_methods import reorder_df_columns

df_dft = reorder_df_columns(
    ['atoms', 'stoich', 'energy_pa', 'dH', 'volume', 'volume_pa', 'num_atoms', ],
    df_dft,
    )
# -

# Pickling data ###########################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_dft.pickle"), "wb") as fle:
    pickle.dump(df_dft, fle)
# #########################################################

# + active=""
#
#
#

# +
# # #####################################################
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/process_bulk_dft",
#     "out_data/df_dft.pickle")
# with open(path_i, "rb") as fle:
#     df_dft = pickle.load(fle)
