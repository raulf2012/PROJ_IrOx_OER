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
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# +
import itertools

import numpy as np
import pandas as pd

from methods import (
    get_magmom_diff_data,
    # _get_magmom_diff_data,
    )

from methods import get_df_jobs
from methods import CountFrequency
from methods import get_df_atoms_sorted_ind
from methods import get_df_job_ids

# +
# # #########################################################
# df_jobs = get_df_jobs()

# # #########################################################
# df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

# # #########################################################
# df_job_ids = get_df_job_ids()
# df_job_ids = df_job_ids.set_index("job_id")

# +
# #########################################################
import pickle; import os
directory = os.path.join(
    os.environ["HOME"],
    "__temp__")
path_i = os.path.join(directory, "temp_data.pickle")
with open(path_i, "rb") as fle:
    data = pickle.load(fle)
# #########################################################

group_w_o = data

# +
write_atoms_objets = True

from local_methods import process_group_magmom_comp

out_dict = process_group_magmom_comp(
    group=group_w_o,
    # df_jobs=None,
    write_atoms_objects=False,
    verbose=False,
    )
# out_dict

# +
out_dict.keys()

# list(out_dict["pair_wise_magmom_comp_data"].keys())




# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# job_id_0 = 

# df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id

# df_jobs_i[df_jobs_i.ads == "bare"]

# df_jobs_i

# + jupyter={"source_hidden": true}
# magmom_data_out["tot_abs_magmom_diff"]

# magmom_data_out.keys()

# + jupyter={"source_hidden": true}
# if write_atoms_objets:

#     df_i = pd.concat([
#         df_job_ids,
#         df_atoms_sorted_ind.loc[
#             group_w_o.job_id_max.tolist()
#             ]
#         ], axis=1, join="inner")

#     # #########################################################
#     df_index_i = group_w_o.index.to_frame()
#     compenv_i = df_index_i.compenv.unique()[0]
#     slab_id_i = df_index_i.slab_id.unique()[0]

#     active_sites = [i for i in df_index_i.active_site.unique() if i != "NaN"]
#     active_site_i = active_sites[0]

#     folder_name = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i))
#     # #########################################################


#     for job_id_i, row_i in df_i.iterrows():
#         tmp = 42

#         job_id = row_i.name
#         atoms = row_i.atoms_sorted_good
#         ads = row_i.ads

#         file_name = ads + "_" + job_id + ".traj"

#         root_file_path = os.path.join("__temp__", folder_name)
#         if not os.path.exists(root_file_path):
#             os.makedirs(root_file_path)

#         file_path = os.path.join(root_file_path, file_name)

#         atoms.write(file_path)

# + jupyter={"source_hidden": true}
# all_triplet_comb = list(itertools.combinations(
#     group_w_o.job_id_max.tolist(), 3))

# good_triplet_comb = []
# for tri_i in all_triplet_comb:
#     df_jobs_i = df_jobs.loc[list(tri_i)]

#     ads_freq_dict = CountFrequency(df_jobs_i.ads.tolist())

#     tmp_list = list(ads_freq_dict.values())
#     any_repeat_ads = [True if i > 1 else False for i in tmp_list]

#     if not any(any_repeat_ads):
#         good_triplet_comb.append(tri_i)

# # good_triplet_comb

# + jupyter={"source_hidden": true}
# data_dict_list = []
# for tri_i in good_triplet_comb:
#     data_dict_i = dict()

#     # print("tri_i:", tri_i)
#     all_pairs = list(itertools.combinations(tri_i, 2))

#     df_jobs_i = df_jobs.loc[list(tri_i)]
    
#     sum_norm_abs_magmom_diff = 0.
#     for pair_i in all_pairs:

#         row_jobs_0 = df_jobs.loc[pair_i[0]]
#         row_jobs_1 = df_jobs.loc[pair_i[1]]

#         ads_0 = row_jobs_0.ads
#         ads_1 = row_jobs_1.ads

#         # #########################################################
#         if set([ads_0, ads_1]) == set(["o", "oh"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
#         elif set([ads_0, ads_1]) == set(["o", "bare"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
#         elif set([ads_0, ads_1]) == set(["oh", "bare"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
#         else:
#             print("Woops something went wrong here")


#         # #########################################################
#         row_atoms_i = df_atoms_sorted_ind.loc[job_id_0]
#         # #########################################################
#         atoms_0 = row_atoms_i.atoms_sorted_good
#         magmoms_sorted_good_0 = row_atoms_i.magmoms_sorted_good
#         was_sorted_0 = row_atoms_i.was_sorted
#         # #########################################################

#         # #########################################################
#         row_atoms_i = df_atoms_sorted_ind.loc[job_id_1]
#         # #########################################################
#         atoms_1 = row_atoms_i.atoms_sorted_good
#         magmoms_sorted_good_1 = row_atoms_i.magmoms_sorted_good
#         was_sorted_1 = row_atoms_i.was_sorted
#         # #########################################################


#         # #########################################################
#         magmom_data_out = get_magmom_diff_data(
#             ads_atoms=atoms_1,
#             slab_atoms=atoms_0,
#             ads_magmoms=magmoms_sorted_good_1,
#             slab_magmoms=magmoms_sorted_good_0,
#             )

#         # list(magmom_data_out.keys())

#         tot_abs_magmom_diff = magmom_data_out["tot_abs_magmom_diff"]
#         # print("    ", pair_i, ": ", np.round(tot_abs_magmom_diff, 2), sep="")
#         norm_abs_magmom_diff = magmom_data_out["norm_abs_magmom_diff"]
#         print("    ", pair_i, ": ", np.round(norm_abs_magmom_diff, 3), sep="")
        
#         sum_norm_abs_magmom_diff += norm_abs_magmom_diff

#     # #####################################################
#     data_dict_i["job_ids_tri"] = set(tri_i)
#     data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################

#     # print("TEMP")
#     # break

#     print("")

#         # #########################################################

# df_magmoms_i = pd.DataFrame(data_dict_list)
# # df_magmoms_i
