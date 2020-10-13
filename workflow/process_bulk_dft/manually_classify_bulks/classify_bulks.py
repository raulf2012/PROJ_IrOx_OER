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

import json
import pickle

import pandas as pd

# #########################################################
from methods import get_df_dft
# -

# # Read Data

# ## Read bulk_ids of octahedral and unique polymorphs

# +
# ########################################################
data_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs/selecting_bulks",
    "out_data/data.json")
with open(data_path, "r") as fle:
    data = json.load(fle)
# ########################################################

bulk_ids__octa_unique = data["bulk_ids__octa_unique"]

# +
df_dft = get_df_dft()

df_dft_i = df_dft[df_dft.index.isin(bulk_ids__octa_unique)]

# +
# df_dft_i.sort_values("num_atoms", ascending=False).iloc[0:15]
# # df_dft_i.sort_values?

# + active=""
#
#

# +
# [print(i) for i in df_dft_i.index.tolist()]

# +
directory = "out_data/all_bulks"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "out_data/layered_bulks"
if not os.path.exists(directory):
    os.makedirs(directory)
# -

for i_cnt, (bulk_id_i, row_i) in enumerate(df_dft_i.iterrows()):
    i_cnt_str = str(i_cnt).zfill(3)

    atoms_i = row_i.atoms

    atoms_i.write("out_data/all_bulks/" + i_cnt_str + "_" + bulk_id_i + ".cif")

# # Reading `bulk_manual_classification.csv`

# +
# df_bulk_class = pd.read_csv("./bulk_manual_classification.csv")
# -

from methods import get_df_bulk_manual_class

df_bulk_class = get_df_bulk_manual_class()

df_bulk_class.head()

# +
print("Total number of bulks being considered:", df_bulk_class.shape[0])

df_bulk_class_layered = df_bulk_class[df_bulk_class.layered == True]
print("Number of layered structures",
    df_bulk_class[df_bulk_class.layered == True].shape)
# -

for i_cnt, (bulk_id_i, row_i) in enumerate(df_bulk_class_layered.iterrows()):
    i_cnt_str = str(i_cnt).zfill(3)
    
    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    atoms_i = row_dft_i.atoms
    # #####################################################

    atoms_i.write("out_data/layered_bulks/" + i_cnt_str + "_" + bulk_id_i + ".cif")

# + active=""
#
#

# + jupyter={"source_hidden": true}
# df_bulk_class = df_bulk_class.fillna(value=False)

# df[['a', 'b']] = df[['a','b']].fillna(value=0)
# # df_bulk_class.fillna?

# + jupyter={"source_hidden": true}
# def read_df_bulk_manual_class():
#     """
#     """
#     # #################################################
#     path_i = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "workflow/process_bulk_dft/manually_classify_bulks",
#         "bulk_manual_classification.csv")
#     df_bulk_class = pd.read_csv(path_i)
#     # df_bulk_class = pd.read_csv("./bulk_manual_classification.csv")
#     # #################################################

#     # Filling empty spots of layerd column with False (if not True)
#     df_bulk_class[["layered"]] = df_bulk_class[["layered"]].fillna(value=False)

#     # Setting index
#     df_bulk_class = df_bulk_class.set_index("bulk_id", drop=False)

#     return(df_bulk_class)
