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

import pandas as pd
import numpy as np

from pymatgen.io.ase import AseAtomsAdaptor

# #########################################################
from methods import get_df_dft

# #########################################################
from local_methods import XRDCalculator
from local_methods import get_top_xrd_facets
from local_methods import compare_facets_for_being_the_same
# -

# # Script Inputs

verbose = True
# verbose = False

# # Read Data

# +
df_dft = get_df_dft()

print("df_dft.shape:", df_dft.shape[0])

# + active=""
#
#
#

# +
# TEMP
# df_dft = df_dft.sample(n=3)

# bulk_id_i = "64cg6j9any"
# bulk_id_i = "zwvqnhbk7f"
# bulk_id_i = "8p8evt9pcg"
bulk_id_i = "b5cgvsb16w"

# df_dft = df_dft.loc[[bulk_id_i]]
# -

# # Main loop

# +
from methods import get_df_xrd
df_xrd_old = get_df_xrd()

print(
    "Number of rows in df_xrd:",
    df_xrd_old.shape[0]
    )

# +
# df_xrd_old.drop_duplicates(

# +
# assert False

# +
for i_cnt, (id_unique_i, row_i) in enumerate(df_dft.iterrows()):
    data_dict_i = dict()
    if verbose:
        print(40 * "=")
        print(str(i_cnt).zfill(3), "id_unique_i:", id_unique_i)

    # #####################################################
    atoms_i = row_i.atoms
    atoms_stan_prim_i = row_i.atoms_stan_prim
    # #####################################################

    from methods import get_df_xrd
    df_xrd_old = get_df_xrd()

    # if not id_unique_i in df_xrd_old.index:
    if id_unique_i in df_xrd_old.index:
        if verbose:
            print("Already computed, skipping")
            
    else:
        # #####################################################
        # df_xrd_i = get_top_xrd_facets(atoms=atoms_stan_prim_i)
        xrd_out_dict = get_top_xrd_facets(atoms=atoms_stan_prim_i)

        df_xrd_all = xrd_out_dict["df_xrd"]
        df_xrd_unique = xrd_out_dict["df_xrd_unique"]

        df_xrd_i = df_xrd_unique

        # Collect all facets into a list
        all_facets = []
        for i in df_xrd_all.facets:
            all_facets.extend(i)
        all_facets = list(set(all_facets))

        # df_xrd_i_1 = df_xrd_i[df_xrd_i.y_norm > 30].iloc[0:10]
        df_xrd_i_1 = df_xrd_i[df_xrd_i.y_norm > 10].iloc[0:15]

        top_facets_i = []
        facet_rank_list = []
        for i_cnt, i in enumerate(df_xrd_i_1.facets.tolist()):
            top_facets_i.extend(i)
            rank_list_i = [i_cnt for i in range(len(i))]
            # print(rank_list_i)
            facet_rank_list.extend(rank_list_i)

        # top_facets_i = facets_list

        num_top_facets = len(top_facets_i)

        if verbose:
            tmp = [len(i) for i in top_facets_i]
            #print(tmp)

        # #################################################
        data_dict_i["id_unique"] = id_unique_i
        data_dict_i["top_facets"] = top_facets_i
        data_dict_i["facet_rank"] = facet_rank_list
        data_dict_i["num_top_facets"] = num_top_facets
        data_dict_i["all_xrd_facets"] = all_facets
        # #################################################


        # #################################################
        # Creating df_xrd with one row and combine it with df_xrd in file
        data_dict_list = []
        data_dict_list.append(data_dict_i)
        df_xrd_row = pd.DataFrame(data_dict_list)
        df_xrd_row = df_xrd_row.set_index("id_unique", drop=False)

        df_xrd_new = pd.concat([
            df_xrd_row,
            df_xrd_old,
            ], axis=0)

        # Pickling data ###################################
        import os; import pickle
        directory = "out_data"
        if not os.path.exists(directory): os.makedirs(directory)
        with open(os.path.join(directory, "df_xrd.pickle"), "wb") as fle:
            pickle.dump(df_xrd_new, fle)
        # #################################################


# # #########################################################
# df_xrd = pd.DataFrame(data_dict_list)
# df_xrd = df_xrd.set_index("id_unique", drop=False)
# # #########################################################
# -

assert False

# +
from methods import get_df_xrd

df_xrd_tmp = get_df_xrd()
df_xrd_tmp

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_xrd_row

# df_xrd_old

# + jupyter={"source_hidden": true}
# Saving data to pickle

# # Pickling data ###########################################
# import os; import pickle
# directory = "out_data"
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_xrd.pickle"), "wb") as fle:
#     pickle.dump(df_xrd, fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# def compare_facets_for_being_the_same(
#     facet_0,
#     facet_1,
#     ):
#     """
#     Checks whether facet_0 and facet_1 differ only by an integer multiplicative.
#     """
#     # #########################################################
#     facet_j_abs = [np.abs(i) for i in facet_j]
#     facet_j_sum = np.sum(facet_j_abs)

#     # #########################################################
#     facet_k_abs = [np.abs(i) for i in facet_k]
#     facet_k_sum = np.sum(facet_k_abs)

#     # #########################################################
#     if facet_j_sum > facet_k_sum:
#         # facet_j_abs / facet_k_abs

#         facet_larger = facet_j_abs
#         facet_small = facet_k_abs
#     else:
#         facet_larger = facet_k_abs
#         facet_small = facet_j_abs

#     # #########################################################
#     facet_frac = np.array(facet_larger) / np.array(facet_small)

#     something_wrong = False
#     all_terms_are_whole_nums = True
#     for i_cnt, i in enumerate(facet_frac):
#         # print(i.is_integer())
#         if np.isnan(i):
#             if facet_j_abs[i_cnt] != 0 or facet_k_abs[i_cnt] != 0:
#                 something_wrong = True
#                 print("Not good, these should both be zero")

#         elif not i.is_integer():
#             all_terms_are_whole_nums = False
#             # print("Not a whole number here")

#     duplicate_found = False
#     if all_terms_are_whole_nums and not something_wrong:
#         duplicate_found = True
#         print("Found a duplicate facet here")


#     return(duplicate_found)

# + jupyter={"source_hidden": true}
# # duplicate_facet_found = \

# facet_j = (1, 0, 1)
# facet_l = (3, 0, 1)

# compare_facets_for_being_the_same(facet_j, facet_l)

# + jupyter={"source_hidden": true}
# facet_0 = (1, 0, 1)
# facet_1 = (3, 0, 1)

# # def compare_facets_for_being_the_same(
# #     facet_0,
# #     facet_1,
# #     ):
# """
# Checks whether facet_0 and facet_1 differ only by an integer multiplicative.
# """
# #| - compare_facets_for_being_the_same
# # #########################################################
# facet_j = facet_0
# facet_k = facet_1

# # #########################################################
# facet_j_abs = [np.abs(i) for i in facet_j]
# facet_j_sum = np.sum(facet_j_abs)

# # #########################################################
# facet_k_abs = [np.abs(i) for i in facet_k]
# facet_k_sum = np.sum(facet_k_abs)

# # #########################################################
# if facet_j_sum > facet_k_sum:
#     # facet_j_abs / facet_k_abs

#     facet_larger = facet_j_abs
#     facet_small = facet_k_abs
# else:
#     facet_larger = facet_k_abs
#     facet_small = facet_j_abs

# # #########################################################
# facet_frac = np.array(facet_larger) / np.array(facet_small)

# # #####################################################
# something_wrong = False
# all_terms_are_whole_nums = True
# # #####################################################
# div_ints =  []
# # #####################################################
# for i_cnt, i in enumerate(facet_frac):
#     # print(i.is_integer())
#     if np.isnan(i):
#         if facet_j_abs[i_cnt] != 0 or facet_k_abs[i_cnt] != 0:
#             something_wrong = True
#             print("Not good, these should both be zero")

#     elif not i.is_integer() or i == 0:
#         all_terms_are_whole_nums = False
#         # print("Not a whole number here")

#     elif i.is_integer():
#         div_ints.append(int(i))

# all_int_factors_are_same = False
# if len(list(set(div_ints))) == 1:
#     all_int_factors_are_same = True

# duplicate_found = False
# if all_terms_are_whole_nums and not something_wrong and all_int_factors_are_same:
#     duplicate_found = True
#     # print("Found a duplicate facet here")

# # return(duplicate_found)
# #__|


# print("duplicate_found:", duplicate_found)

# + jupyter={"source_hidden": true}
# facet_frac

# + jupyter={"source_hidden": true}
# all_terms_are_whole_nums
# something_wrong

# + jupyter={"source_hidden": true}
# # #########################################################
# indices_to_drop = []
# # #########################################################
# for ind_i, row_i in df_xrd_unique.iterrows():

#     # #####################################################
#     facets_i = row_i.facets
#     # #####################################################

#     for facet_j in facets_i:

#         for ind_k, row_k in df_xrd_unique.iterrows():

#             # #############################################
#             facets_k = row_k.facets
#             # #############################################

#             for facet_l in facets_k:

#                 if facet_j == facet_l:
#                     continue
#                 else:
#                     duplicate_facet_found = \
#                         compare_facets_for_being_the_same(facet_j, facet_l)

#                     if duplicate_facet_found:
#                         # print(duplicate_facet_found, facet_j, facet_l)

#                         if np.sum(np.abs(facet_j)) > np.sum(np.abs(facet_l)):
#                             indices_to_drop.append(ind_i)
#                             # print(ind_i)
#                         else:
#                             indices_to_drop.append(ind_k)
#                             # print(ind_k)

# # #########################################################
# indices_to_drop = list(set(indices_to_drop))
# # #########################################################

# df_xrd_unique_1 = df_xrd_unique.drop(index=indices_to_drop)

# + jupyter={"source_hidden": true}
# df_xrd_i_1

# + jupyter={"source_hidden": true}
# top_facets_i = []
# facet_rank_list = []
# for i_cnt, i in enumerate(df_xrd_i_1.facets.tolist()):
#     top_facets_i.extend(i)
#     rank_list_i = [i_cnt for i in range(len(i))]
#     print(rank_list_i)
#     facet_rank_list.extend(rank_list_i)

# + jupyter={"source_hidden": true}
# df_xrd_i_1

# + jupyter={"source_hidden": true}
# facet_rank_list

# + jupyter={"source_hidden": true}
# df_xrd

# df_xrd_all

# + jupyter={"source_hidden": true}
# [0, 10, 22, 29, 30]
# [10, 29, 22, 30]

# + jupyter={"source_hidden": true}
# df_xrd_unique = df_xrd_unique.loc[[2, 19]]
# df_xrd_unique = df_xrd_unique.loc[[2, 22]]

# xrd_out_dict

# df_xrd_unique
