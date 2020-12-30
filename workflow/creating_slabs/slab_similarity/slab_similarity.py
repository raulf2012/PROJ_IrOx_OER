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

# # Compute similarity of constructed *O IrOx slabs
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

import numpy as np
import pandas as pd

# #########################################################
# from StructurePrototypeAnalysisPackage.ccf import struc2ccf
# from StructurePrototypeAnalysisPackage.ccf import struc2ccf, cal_ccf_d
from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d

# #########################################################
from methods import get_df_slab

from methods import get_ccf
from methods import get_D_ij
from methods import get_identical_slabs

# #########################################################
# from local_methods import get_ccf
# from local_methods import get_D_ij
# from local_methods import get_identical_slabs
# -

# # Script Inputs

# +
verbose = True

r_cut_off = 10
r_vector = np.arange(1, 10, 0.02)
# -

# # Read Data

df_slab = get_df_slab()

# # TEMP | Filtering down `df_slab`

# +
# df_slab = df_slab[df_slab.bulk_id ==  "mjctxrx3zf"]
# df_slab = df_slab[df_slab.bulk_id ==  "64cg6j9any"]
# -

df_slab

df_slab = df_slab.sort_values(["bulk_id", "facet", ])

bulk_ids = [
    "64cg6j9any",
    "9573vicg7f",
    "b19q9p6k72",
    # "",
    ]
# df_slab = df_slab[
#     df_slab.bulk_id.isin(bulk_ids)
#     ]

# +
# assert False

# + active=""
#
#
# -

# # Looping through slabs and computing CCF

grouped = df_slab.groupby(["bulk_id"])
for bulk_id_i, group_i in grouped:
    for slab_id_j, row_j in group_i.iterrows():
        # #####################################################
        slab_final_j = row_j.slab_final
        # #####################################################

        ccf_j = get_ccf(
            slab_id=slab_id_j,
            slab_final=slab_final_j,
            r_cut_off=r_cut_off,
            r_vector=r_vector,
            verbose=False)

# # Constructing D_ij matrix

verbose_local = False
# #########################################################
data_dict_list = []
# #########################################################
grouped = df_slab.groupby(["bulk_id"])
for bulk_id_i, group_i in grouped:
    # #####################################################
    data_dict_i = dict()
    # #####################################################

    if verbose_local:
        print("slab_id:", bulk_id_i)

    D_ij = get_D_ij(group_i, slab_id=bulk_id_i)
    ident_slab_pairs_i = get_identical_slabs(D_ij)

    # print("ident_slab_pairs:", ident_slab_pairs_i)

    ids_to_remove = []
    for ident_pair_i in ident_slab_pairs_i:
        # Checking if any id already added to `id_to_remove` is in a new pair
        for i in ids_to_remove:
            if i in ident_pair_i:
                print("This case needs to be dealt with more carefully")
                break

        ident_pair_2 = np.sort(ident_pair_i)
        ids_to_remove.append(ident_pair_2[0])

    num_ids_to_remove = len(ids_to_remove)

    if verbose_local:
        print("ids_to_remove:", ids_to_remove)

    # #####################################################
    data_dict_i["bulk_id"] = bulk_id_i
    data_dict_i["slab_ids_to_remove"] = ids_to_remove
    data_dict_i["num_ids_to_remove"] = num_ids_to_remove
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# +
df_slab_simil = pd.DataFrame(data_dict_list)

df_slab_simil
# -

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs/slab_similarity",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slab_simil.pickle"), "wb") as fle:
    pickle.dump(df_slab_simil, fle)
# #########################################################

from methods import get_df_slab_simil
df_slab_simil = get_df_slab_simil()

df_slab_simil

assert False

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# ident_slab_pairs_i = [
#     ['bimamuvo_42', 'hidopiha_44'],
#     ['legifipe_18', 'witepote_55'],
#     ]

# ids_to_remove = []
# for ident_pair_i in ident_slab_pairs_i:

#     # Checking if any id already added to `id_to_remove` is in a new pair
#     for i in ids_to_remove:
#         if i in ident_pair_i:
#             print("This case needs to be dealt with more carefully")
#             break

#     ident_pair_2 = np.sort(ident_pair_i)
#     ids_to_remove.append(ident_pair_2[0])

# + jupyter={"source_hidden": true}
# identical_pairs_list = [
#     ["a", "b"],
#     ["b", "a"],

#     ["c", "d"],
#     ]

# # identical_pairs_list_2 = 
# # list(np.unique(
# #     [np.sort(i) for i in identical_pairs_list]
# #     ))

# np.unique(
# [np.sort(i) for i in identical_pairs_list]
# )

# + jupyter={"source_hidden": true}
# import itertools

# lst = identical_pairs_list
# lst.sort()
# lst = [list(np.sort(i)) for i in lst]

# identical_pairs_list_2 = list(lst for lst, _ in itertools.groupby(lst))

# + jupyter={"source_hidden": true}
# def get_identical_slabs(
#     D_ij,

#     min_thresh=1e-5,
#     ):
#     """
#     """
#     #| - get_identical_slabs

#     # #########################################################
#     # min_thresh = 1e-5
#     # #########################################################

#     identical_pairs_list = []
#     for slab_id_i in D_ij.index:
#         for slab_id_j in D_ij.index:
#             if slab_id_i == slab_id_j:
#                 continue
#             if slab_id_i == slab_id_j:
#                 print("Not good if this is printed")

#             d_ij = D_ij.loc[slab_id_i, slab_id_j]
#             if d_ij < min_thresh:
#                 # print(slab_id_i, slab_id_j)
#                 identical_pairs_list.append((slab_id_i, slab_id_j))

#     # #########################################################
#     identical_pairs_list_2 = list(np.unique(
#         [np.sort(i) for i in identical_pairs_list]
#         ))

#     return(identical_pairs_list_2)
#     #__|

# + jupyter={"source_hidden": true}
# # #########################################################
# import pickle; import os
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/creating_slabs/slab_similarity",
#     "out_data/df_slab_simil.pickle")
# with open(path_i, "rb") as fle:
#     df_slab_simil = pickle.load(fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# /home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/workflow/creating_slabs/slab_similarity

# workflow/creating_slabs/slab_similarity
