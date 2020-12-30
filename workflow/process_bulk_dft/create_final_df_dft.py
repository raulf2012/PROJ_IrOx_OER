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

import pandas as pd

# #########################################################
from misc_modules.pandas_methods import drop_columns
# -

# # Read Data

# +
# #########################################################
path_i = os.path.join(
    os.environ["PROJ_DATA"],
    "PROJ_IrOx_OER/active_learning_proj_data",
    "df_dft_final_no_dupl.pickle")
with open(path_i, "rb") as fle:
    df_dft = pickle.load(fle)

# #########################################################
path_i = os.path.join("out_data", "atoms_dict.pickle")
with open(path_i, "rb") as fle:
    atoms_dict = pickle.load(fle)

# #########################################################
dir_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/process_bulk_dft/standardize_bulks", "out_data")
file_name_i = os.path.join(dir_i, "df_dft_stan.pickle")
with open(file_name_i, "rb") as fle:
    df_dft_stan = pickle.load(fle)

# +
# #####################################################
df_dft = drop_columns(
    df=df_dft,
    columns=["atoms", "form_e_chris", "id_old", "path", "id", "source", "energy", ],
    keep_or_drop="drop")

df_dft = df_dft.sort_values("dH")
# -

# Create new atoms column from `atoms_dict`

df_dft["atoms"] = df_dft.index.map(atoms_dict)

# +
# df_dft.drop(columns=["num_atoms", "", ])

# +
# df_dft_stan

# list(pd.concat([
#     df_dft,
#     df_dft_stan,
#     ], axis=1).columns)

df_dft = pd.concat([
    # df_dft,
    df_dft.drop(columns=["num_atoms", ]),
    df_dft_stan,
    ], axis=1)

# # pd.concat?

# +
from misc_modules.pandas_methods import reorder_df_columns

df_dft = reorder_df_columns(
    [
        "id_unique", "stoich", "energy_pa", "dH", "volume", "volume_pa",
        "num_atoms", "num_atoms_stan", "num_atoms_stan_prim", "num_atoms_red__stan", "num_atoms_red__stan_prim",
        "atoms", "atoms_stan", "atoms_stan_prim",
        ],
    df_dft,
    )

# +
df_dft = df_dft.rename(columns={
    "num_atoms": "na",
    "num_atoms_stan_prim": "na_stan_prim",
    "num_atoms_stan": "na_stan",
    "num_atoms_red__stan": "na_red__stan",
    "num_atoms_red__stan_prim": "na_red__stan_prim",
    })

df_dft.head()

# +
# assert False
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

# + jupyter={"source_hidden": true}
# # #####################################################
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/process_bulk_dft",
#     "out_data/df_dft.pickle")
# with open(path_i, "rb") as fle:
#     df_dft = pickle.load(fle)
