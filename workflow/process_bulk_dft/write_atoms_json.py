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
#     display_name: Python [conda env:PROJ_irox]
#     language: python
#     name: conda-env-PROJ_irox-py
# ---

# # Convert atoms objects in `df_dft` to new ase version
# ---
#
# Run this script with the `PROJ_irox` conda environment (used to create df_dft)
#
# The plan of action here will be to convert the atoms objects to json files and then rereading with the new ase version

# # Import Modules

import os
print(os.getcwd())
import sys

from methods import get_df_dft
df_dft = get_df_dft()

# +
# TEMP
# df_dft = df_dft.sample(n=10)
# -

directory = "out_data/json_files"
if not os.path.exists(directory):
    os.makedirs(directory)


# +
# %%capture

def method(row_i):
    atoms = row_i.atoms

    atoms.write(
        os.path.join(
            directory,
            row_i.name + ".json"
            )
        )


df_i = df_dft
df_i.apply(
    method,
    axis=1,
    )
