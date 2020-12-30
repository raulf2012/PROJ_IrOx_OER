# """
# """
#
# #| - Import Modules
# import os
# import sys
#
# import pickle
# from pathlib import Path
# import itertools
#
# import numpy as np
# import pandas as pd
#
# from StructurePrototypeAnalysisPackage.ccf import struc2ccf
# from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d
# #__|
#
#
# # get_ccf
# # get_D_ij
# # get_identical_slabs
#
# def get_ccf(
#     slab_id=None,
#     slab_final=None,
#     verbose=True,
#     r_cut_off=None,
#     r_vector=None,
#     ):
#     """
#     """
#     #| - get_ccf_i
#     # #####################################################
#     global os
#     global pickle
#
#     # #####################################################
#     slab_id_i = slab_id
#     slab_final_i = slab_final
#     # #####################################################
#
#     directory = "out_data/ccf_files"
#     name_i = slab_id_i + ".pickle"
#     # print("os:", os)
#     file_path_i = os.path.join(directory, name_i)
#     my_file = Path(file_path_i)
#     if my_file.is_file():
#         if verbose:
#             print("File exists already")
#
#         # #################################################
#         import pickle; import os
#         path_i = os.path.join(
#             os.environ["PROJ_irox_oer"],
#             "workflow/creating_slabs/slab_similarity",
#             file_path_i)
#         with open(path_i, "rb") as fle:
#             ccf_i = pickle.load(fle)
#         # #################################################
#     else:
#         ccf_i = struc2ccf(slab_final_i, r_cut_off, r_vector)
#
#
#         # Pickling data ###################################
#         if not os.path.exists(directory): os.makedirs(directory)
#         with open(file_path_i, "wb") as fle:
#             pickle.dump(ccf_i, fle)
#         # #################################################
#
#     return(ccf_i)
#     #__|
#
# def get_D_ij(group, slab_id=None, verbose=False):
#     """
#     """
#     #| - get_D_ij
#     # #####################################################
#     group_i = group
#     slab_id_i = slab_id
#     # #####################################################
#
#     if slab_id_i == None:
#         print("Need to provide a slab_id so that the D_ij can be saved")
#
#     directory = "out_data/D_ij_files"
#     name_i = slab_id_i + ".pickle"
#     file_path_i = os.path.join(directory, name_i)
#
#
#     my_file = Path(file_path_i)
#     if my_file.is_file():
#         #| - Read data from file
#         if verbose:
#             print("Reading from file")
#
#         with open(file_path_i, "rb") as fle:
#             D_ij = pickle.load(fle)
#         # #########################################################
#         #__|
#     else:
#         #| - Create D_ij from scratch
#         if verbose:
#             print("Creating from scratch")
#
#         num_rows = group_i.shape[0]
#
#         D_ij = np.empty((num_rows, num_rows))
#         D_ij[:] = np.nan
#
#         #| - Looping through group rows
#         slab_ids_j = []
#         for i_cnt, (slab_id_j, row_j) in enumerate(group_i.iterrows()):
#             if verbose:
#                 print(40 * "*")
#             for j_cnt, (slab_id_k, row_k) in enumerate(group_i.iterrows()):
#                 if verbose:
#                     print(30 * "-")
#                 ccf_j = get_ccf(slab_id=slab_id_j, verbose=False)
#                 ccf_k = get_ccf(slab_id=slab_id_k, verbose=False)
#
#                 d_ij = cal_ccf_d(ccf_j, ccf_k)
#
#                 D_ij[i_cnt][j_cnt] = d_ij
#         #__|
#
#         # #####################################################
#         D_ij = pd.DataFrame(
#             D_ij,
#             index=group_i.index,
#             columns=group_i.index,
#             )
#
#         #######################################################
#         # Pickling data #######################################
#         if not os.path.exists(directory): os.makedirs(directory)
#         with open(file_path_i, "wb") as fle:
#             pickle.dump(D_ij, fle)
#         #######################################################
#         #__|
#
#     return(D_ij)
#     #__|
#
# def get_identical_slabs(
#     D_ij,
#
#     min_thresh=1e-5,
#     ):
#     """
#     """
#     #| - get_identical_slabs
#
#     # #########################################################
#     # min_thresh = 1e-5
#     # #########################################################
#
#     identical_pairs_list = []
#     for slab_id_i in D_ij.index:
#         for slab_id_j in D_ij.index:
#             if slab_id_i == slab_id_j:
#                 continue
#             if slab_id_i == slab_id_j:
#                 print("Not good if this is printed")
#
#             d_ij = D_ij.loc[slab_id_i, slab_id_j]
#             if d_ij < min_thresh:
#                 # print(slab_id_i, slab_id_j)
#                 identical_pairs_list.append((slab_id_i, slab_id_j))
#
#     # #########################################################
#     # identical_pairs_list_2 = list(np.unique(
#     #     [np.sort(i) for i in identical_pairs_list]
#     #     ))
#
#     lst = identical_pairs_list
#     lst.sort()
#     lst = [list(np.sort(i)) for i in lst]
#
#     identical_pairs_list_2 = list(lst for lst, _ in itertools.groupby(lst))
#
#     return(identical_pairs_list_2)
#     #__|
