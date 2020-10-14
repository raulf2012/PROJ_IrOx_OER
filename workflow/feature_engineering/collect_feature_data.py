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
from methods import get_df_dft
from methods import get_df_job_ids
# -

# # Read feature dataframes

# +
df_octa_vol = get_df_octa_vol()

df_eff_ox = get_df_eff_ox()

df_dft = get_df_dft()

df_job_ids = get_df_job_ids()
# -

# # Setting proper indices for join

# +
df_eff_ox = df_eff_ox.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    drop=True)

df_octa_vol = df_octa_vol.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    drop=True)
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

# # Adding in bulk data

# +
def method(row_i):
    new_column_values_dict = {
        "dH_bulk": None,
        "volume_pa": None,
        }

    # #####################################################
    slab_id_i = row_i.name[1]
    # #####################################################
    bulk_ids = df_job_ids[df_job_ids.slab_id == slab_id_i].bulk_id.unique()
    mess_i = "ikjisdjf"
    assert len(bulk_ids) == 1, mess_i
    bulk_id_i = bulk_ids[0]
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    dH_i = row_dft_i.dH
    volume_pa = row_dft_i.volume_pa
    # #####################################################


    # #####################################################
    new_column_values_dict["dH_bulk"] = dH_i
    new_column_values_dict["volume_pa"] = volume_pa
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    return(row_i)

df_features = df_features.apply(method, axis=1)

# +
# assert False
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

# + jupyter={"source_hidden": true}
# row_i = df_features.iloc[0]

# slab_id_i = row_i.name[1]

# bulk_ids = df_job_ids[df_job_ids.slab_id == slab_id_i].bulk_id.unique()
# mess_i = "ikjisdjf"
# assert len(bulk_ids) == 1, mess_i
# bulk_id_i = bulk_ids[0]

# row_dft_i = df_dft.loc[bulk_id_i]

# dH_i = row_dft_i.dH
# volume_pa = row_dft_i.volume_pa

# + jupyter={"source_hidden": true}
# # def method(row_i, argument_0, optional_arg=None):
# def method(row_i):
#     new_column_values_dict = {
#         "TEMP": None,
#         }


#     # #########################################################
#     # row_i = df_dft.iloc[0]
#     # #########################################################
#     bulk_id_i = row_i.name
#     # #########################################################

#     tmp = df_job_ids[df_job_ids.bulk_id == bulk_id_i]
#     if tmp.shape[0] > 0:
#         print(bulk_id_i)
#     # print(tmp.shape[0])

#     # bulk_id_i

#     # #########################################################################
#     for key, value in new_column_values_dict.items():
#         row_i[key] = value
#     return(row_i)

# df_i = df_dft

# # arg1 = "TEMP_0"
# # df_i["column_name"] = df_i.apply(
# df_i = df_i.apply(
#     method,
#     axis=1,
#     # args=(arg1, ),
#     # optional_arg="TEMP_1"
#     )
# df_i

# + jupyter={"source_hidden": true}
# # #########################################################
# # row_i = df_dft.iloc[0]
# row_i = df_dft.loc["b49kx4c19q"]
# # #########################################################
# bulk_id_i = row_i.name
# # #########################################################

# tmp = df_job_ids[df_job_ids.bulk_id == bulk_id_i]
# print(tmp.shape[0])

# # bulk_id_i
# tmp

# + jupyter={"source_hidden": true}
# row_i

# + jupyter={"source_hidden": true}
# lst_0 = list(set(df_job_ids.bulk_id.tolist()))
# lst_1 = df_dft.index.tolist()

# # a = [1,2,3,4,5]
# # b = [1,3,5,6]

# list(set(lst_0) & set(lst_1))
