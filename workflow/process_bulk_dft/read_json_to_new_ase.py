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

# # Read atoms json files with new ase version
# ---
#
# Here we will read the written json atoms files from `write_atoms_json.ipynb` with the `PROJ_irox_oer` environment activated (new ase)

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

import pandas as pd

from ase.io import read

# +
data_dict_list = []
root_dir = "out_data/json_files"
for subdir, dirs, files in os.walk(root_dir):

    if ".ipynb_checkpoints" in subdir.split("/"):
        continue

    for file in files:
        data_dict_i = dict()

        file_path = os.path.join(subdir, file)
        
        atoms_i = read(file_path)
        data_dict_i["atoms"] = atoms_i
        
        id_i = file_path.split("/")[-1].split(".")[0]
        data_dict_i["id"] = id_i

        data_dict_list.append(data_dict_i)

df_atoms = pd.DataFrame(data_dict_list)

# +
atoms_dict = dict(zip(
    df_atoms.id,
    df_atoms.atoms,
    ))

directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "atoms_dict.pickle"), "wb") as fle:
    pickle.dump(atoms_dict, fle)
