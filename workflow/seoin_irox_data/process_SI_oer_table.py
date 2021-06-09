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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### Processing IrOx Data from Seoin
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

import pandas as pd
# -

# ### Read Data

# #########################################################
path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "in_data/oer_data_seoin.csv")
df_oer_si = pd.read_csv(path_i, dtype={"facet": object})

# +
crystal_rename_dict = {
    "Rutile (P42/nm)": "rutile",
    "Anatase (I41/amd)": "anatase",
    "Pyrite (Pa3)": "pyrite",
    "Brookite (Pbca)": "brookite",
    "Columbite (Pbcn)": "columbite",
    "Amm2": "amm2",
    "Pm3m": "pm-3m",
    "Cmcm": "cmcm",
    }

coverage_rename_dict = {
    "*O": "O_covered",
    "*0": "O_covered",
    "*OH": "OH_covered",
    }

def method(row_i):
    # #####################################################
    new_column_values_dict = {
        "crystal": None,
        "coverage": None,
        }
    # #####################################################
    crystal_i = row_i["crystal"]
    coverage_i = row_i["coverage"]
    # #####################################################

    new_crystal_i = crystal_rename_dict.get(crystal_i, crystal_i)
    new_coverage_i = coverage_rename_dict.get(coverage_i, coverage_i)

    # #####################################################
    new_column_values_dict["crystal"] = new_crystal_i
    new_column_values_dict["coverage"] = new_coverage_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    # #####################################################
    return(row_i)
    # #####################################################



# #########################################################
df_i = df_oer_si
df_oer_si = df_i.apply(
    method,
    axis=1)
# #########################################################
df_oer_si = df_oer_si.set_index(
    [
        "crystal", "facet", "coverage",
        "termination", "active_site",
        ]
    )

df_oer_si = df_oer_si[~df_oer_si.index.duplicated()]
# #########################################################
# -

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_oer_si.pickle"), "wb") as fle:
    pickle.dump(df_oer_si, fle)
# #########################################################

df_oer_si.head()

# + active=""
#
#
#
