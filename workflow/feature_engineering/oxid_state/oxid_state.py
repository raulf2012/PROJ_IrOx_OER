# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Calculate the formal oxidation state of all relaxed slabs
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd

# #########################################################
from proj_data import metal_atom_symbol

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    )

# #########################################################
from local_methods import process_row
from local_methods import get_num_metal_neigh_manually
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

# + active=""
#
#
#

# +
# df_jobs_anal

# +
sys.path.insert(
    0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering",
        ),
    )

from feature_engineering_methods import get_df_feat_rows
df_feat_rows = get_df_feat_rows(
    df_jobs_anal=df_jobs_anal,
    df_atoms_sorted_ind=df_atoms_sorted_ind,
    df_active_sites=df_active_sites,
    )

df_feat_rows = df_feat_rows.set_index(
    # ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ],
    ["compenv", "slab_id", "ads", "active_site_orig", "att_num", ],
    drop=False,
    )

# +
# names = [
#     ('sherlock', 'filetumi_93', 'o', 'NaN', 1),
#     ('sherlock', 'filetumi_93', 'o', 65.0, 1),
#     ('sherlock', 'filetumi_93', 'o', 67.0, 1),
#     ('sherlock', 'filetumi_93', 'oh', 60.0, 3),
#     ('sherlock', 'ramufalu_44', 'o', 54.0, 1),
#     ('sherlock', 'telibose_95', 'oh', 35.0, 1),
#     ('sherlock', 'vinamepa_43', 'o', 'NaN', 1),
#     ('slac', 'bahusihe_57', 'o', 'NaN', 1),
#     ('slac', 'bofahisa_20', 'o', 'NaN', 1),
#     ('slac', 'ralutiwa_59', 'o', 'NaN', 1),
#     ('slac', 'vovumota_03', 'oh', 31.0, 1),
#     ('slac', 'vuraruna_65', 'oh', 50.0, 2),
#     ]

# df_feat_rows = df_feat_rows.loc[
#     pd.MultiIndex.from_tuples(names).unique().tolist()
#     ]

df_feat_rows = df_feat_rows.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ],
    drop=False,
    )

# +
# print(222 * "TEMP | ")

# df_feat_rows = df_feat_rows.loc[[
#     # ('sherlock', 'filetumi_93', 'o', 60.0, 1, False),
#     # ('sherlock', 'filetumi_93', 'o', 65.0, 1, False),
#     # ('sherlock', 'filetumi_93', 'o', 65.0, 1, True),
#     # ('sherlock', 'filetumi_93', 'o', 67.0, 1, True),
#     # ('slac', 'vuraruna_65', 'oh', 50.0, 2, True),

#     ('sherlock', 'telibose_95', 'oh', 35.0, 1, True)

#     ]]

# +
# df_feat_rows = df_feat_rows.loc[[
#     # ('sherlock', 'kagekiha_49', 'o', 87.0, 1, False),
#     # ('slac', 'dugihabe_70', 'oh', 47.0, 2, True),
#     # ('slac', 'wavihanu_77', 'oh', 48.0, 3, True),

#     ('sherlock', 'kobehubu_94', 'oh', 50.0, 0, True)
#     ]]

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # print("")
    # print(index_i)
    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    # #####################################################

    # TEMP
    # row_atoms_i = df_atoms_sorted_ind.loc[name_orig_i]
    # atoms_i = row_atoms_i.atoms_sorted_good
    # atoms_i.write("__temp__/atoms.traj")
    # atoms_i.write("__temp__/atoms.cif")

    if active_site_orig_i == "NaN":
        from_oh_i = False
    else:
        from_oh_i = True

    name_orig_i = (
        row_i.compenv, row_i.slab_id, row_i.ads,
        row_i.active_site_orig, row_i.att_num,
        )

    out_dict = process_row(
        name=name_orig_i,
        active_site=active_site_i,
        active_site_original=active_site_orig_i,
        metal_atom_symbol=metal_atom_symbol,
        verbose=verbose)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["job_id_max"] = job_id_max_i
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #####################################################
    data_dict_i.update(out_dict)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_eff_ox = pd.DataFrame(data_dict_list)
# #########################################################

# +
# assert False
# -

# # Further processing

df_eff_ox = df_eff_ox.set_index(
    # ["compenv", "slab_id", "ads", "active_site", "att_num", ],
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_eff_ox

columns = list(df.columns)

# feature_cols = ["eff_oxid_state", ]
feature_cols = ["effective_ox_state", ]
for feature_i in feature_cols:
    columns.remove(feature_i)

multi_columns_dict = {
    "features": feature_cols,
    "data": columns,
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)

df_eff_ox = df
# -

df_eff_ox = df_eff_ox.reindex(columns = ["data", "features", ], level=0)

# ### Write `df_eff_ox` to file

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/oxid_state")

# Pickling data ###########################################
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_eff_ox.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_eff_ox, fle)
# #########################################################

# #########################################################
with open(path_i, "rb") as fle:
    df_eff_ox = pickle.load(fle)
# #########################################################

# +
from methods import get_df_eff_ox

df_eff_ox_tmp = get_df_eff_ox()
df_eff_ox_tmp.head()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("oxid_state.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# print(222 * "TEMP | ")
# assert False

# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################
# # #########################################################

# + jupyter={"source_hidden": true}
# out_dict

# + jupyter={"source_hidden": true}
# df_eff_ox

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# #| - Import Modules
# import os
# import sys

# import copy

# import numpy as np
# #  import pandas as pd

# from methods import (
#      get_df_coord,
#      get_df_coord_wrap,
#      )

# from methods_features import find_missing_O_neigh_with_init_df_coord
# from methods_features import original_slab_is_good
# #__|

# + jupyter={"source_hidden": true}
# from local_methods import *

# + jupyter={"source_hidden": true}
# name = name_orig_i
# active_site = active_site_i
# active_site_original = active_site_orig_i
# metal_atom_symbol = metal_atom_symbol
# verbose = verbose

# # def process_row(
# #     name=None,
# #     active_site=None,
# #     active_site_original=None,
# #     metal_atom_symbol="Ir",
# #     verbose=False,
# #     ):
# """
# """
# #| - process_row
# # #####################################################
# name_i = name
# # active_site_j = active_site
# # #####################################################

# df_coord_i = get_df_coord_wrap(name=name_i, active_site=active_site)

# # # #####################################################
# # out_dict = get_effective_ox_state(
# #     name=name_i,
# #     active_site=active_site,
# #     df_coord_i=df_coord_i,
# #     metal_atom_symbol=metal_atom_symbol,
# #     active_site_original=active_site_original,
# #     )
# # # #####################################################
# # effective_ox_state_i = out_dict["effective_ox_state"]
# # used_unrelaxed_df_coord_i = out_dict["used_unrelaxed_df_coord"]
# # num_missing_Os_i = out_dict["num_missing_Os"]
# # orig_slab_good_i = out_dict["orig_slab_good"]
# # # #####################################################

# # # #####################################################
# # data_dict_j = dict()
# # # #####################################################
# # data_dict_j["eff_oxid_state"] = effective_ox_state_i
# # data_dict_j["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord_i
# # data_dict_j["orig_slab_good"] = orig_slab_good_i
# # data_dict_j["num_missing_Os"] = num_missing_Os_i
# # # data_dict_j["active_site"] = active_site
# # # #####################################################
# # # return(data_dict_j)
# # # #####################################################
# #__|

# + jupyter={"source_hidden": true}
# name = name_orig_i
# active_site = active_site
# df_coord_i = df_coord_i
# metal_atom_symbol = metal_atom_symbol
# active_site_original = active_site_original

# # def get_effective_ox_state(
# #     name=None,
# #     active_site=None,
# #     df_coord_i=None,
# #     metal_atom_symbol="Ir",
# #     active_site_original=None,
# #     ):
# """
# """
# #| - get_effective_ox_state
# # #########################################################
# name_i = name
# active_site_j = active_site
# # #########################################################
# compenv_i = name_i[0]
# slab_id_i = name_i[1]
# ads_i = name_i[2]
# active_site_i = name_i[3]
# att_num_i = name_i[4]
# # #########################################################

# # out_dict["effective_ox_state"] = effective_ox_state
# # out_dict["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord
# # out_dict["num_missing_Os"] = num_missing_Os
# # out_dict["orig_slab_good"] = orig_slab_good_i


# # #########################################################
# #| - Processing central Ir atom nn_info
# df_coord_i = df_coord_i.set_index("structure_index", drop=False)


# import os
# import sys
# import pickle



# # row_coord_i = df_coord_i.loc[21]
# row_coord_i = df_coord_i.loc[active_site_j]

# nn_info_i = row_coord_i.nn_info

# neighbor_count_i = row_coord_i.neighbor_count
# num_Ir_neigh = neighbor_count_i.get("Ir", 0)


# mess_i = "For now only deal with active sites that have 1 Ir neighbor"
# assert num_Ir_neigh == 1, mess_i

# for j_cnt, nn_j in enumerate(nn_info_i):
#     site_j = nn_j["site"]
#     elem_j = site_j.as_dict()["species"][0]["element"]

#     if elem_j == metal_atom_symbol:
#         corr_j_cnt = j_cnt

# site_j = nn_info_i[corr_j_cnt]
# metal_index = site_j["site_index"]
# #__|

# # #########################################################
# row_coord_i = df_coord_i.loc[metal_index]

# neighbor_count_i = row_coord_i["neighbor_count"]
# nn_info_i =  row_coord_i.nn_info
# num_neighbors_i = row_coord_i.num_neighbors

# num_O_neigh = neighbor_count_i.get("O", 0)

# six_O_neigh = num_O_neigh == 6
# mess_i = "There should be exactly 6 oxygens about the Ir atom"

# six_neigh = num_neighbors_i == 6
# mess_i = "Only 6 neighbors total is allowed, all oxygens"

# skip_this_sys = False
# if not six_O_neigh or not six_neigh:
#     # print("Skip this sys")
#     skip_this_sys = True


# #| - If missing some O's then go back to slab before DFT and get missing O
# from methods import get_df_coord

# init_slab_name_tuple_i = (
#     compenv_i, slab_id_i, ads_i,
#     active_site_original, att_num_i,
#     )
# df_coord_orig_slab = get_df_coord(
#     mode="init-slab",
#     init_slab_name_tuple=init_slab_name_tuple_i,
#     )
# orig_slab_good_i = original_slab_is_good(
#     nn_info=nn_info_i,
#     metal_index=metal_index,
#     df_coord_orig_slab=df_coord_orig_slab,
#     )


# num_missing_Os = 0
# used_unrelaxed_df_coord = False
# if not six_O_neigh:
#     used_unrelaxed_df_coord = True

#     from methods import get_df_coord
#     init_slab_name_tuple_i = (
#         compenv_i, slab_id_i, ads_i,
#         active_site_original, att_num_i,
#         # active_site_i, att_num_i,
#         )
#     df_coord_orig_slab = get_df_coord(
#         mode="init-slab",
#         init_slab_name_tuple=init_slab_name_tuple_i,
#         )

#     out_dict_0 = find_missing_O_neigh_with_init_df_coord(
#         nn_info=nn_info_i,
#         slab_id=slab_id_i,
#         metal_index=metal_index,
#         df_coord_orig_slab=df_coord_orig_slab,
#         )
#     new_nn_info_i = out_dict_0["nn_info"]
#     num_missing_Os = out_dict_0["num_missing_Os"]
#     orig_slab_good_i = out_dict_0["orig_slab_good"]

#     nn_info_i = new_nn_info_i

#     if new_nn_info_i is not None:
#         skip_this_sys = False
#     else:
#         skip_this_sys = True
# #__|


# # #####################################################
# effective_ox_state = None
# if not skip_this_sys:
#     #| - Iterating through 6 oxygens
#     orig_df_coord_was_used = False

#     second_shell_coord_list = []
#     tmp_list = []
#     for nn_j in nn_info_i:

#         site_index = nn_j["site_index"]

#         #| - Fixing bond number of missing *O
#         # If Ir was missing *O bond, then neigh count for that O will be undercounted
#         # Although sometimes even through the Ir is missing the *O, the *O is not missing the Ir
#         # Happened for this system: ('slac', 'ralutiwa_59', 'o', 31.0, 1)
#         from_orig_df_coord = nn_j.get("from_orig_df_coord", False)
#         active_metal_in_nn_list = False

#         if from_orig_df_coord:
#             orig_df_coord_was_used = True

#             Ir_neigh_adjustment = 1
#             for i in df_coord_i.loc[site_index].nn_info:
#                 if i["site_index"] == metal_index:
#                     active_metal_in_nn_list = True

#             if active_metal_in_nn_list:
#                 Ir_neigh_adjustment = 0

#         else:
#             Ir_neigh_adjustment = 0
#         #__|


#         oxy_ind = site_index
#         num_metal_neigh_2 = get_num_metal_neigh_manually(
#             oxy_ind, df_coord=df_coord_i, metal_atom_symbol=metal_atom_symbol)

#         # #################################################
#         #| - Checking manually the discrepency
#         if False:
#             row_coord_j = df_coord_i.loc[site_index]

#             neighbor_count_j = row_coord_j.neighbor_count

#             # TODO | IMPORTANT
#             # We should check manually the previous structure for Ir neighbors
#             # Also we should check if the 'lost' Ir-O bonds are good or are completely bad
#             num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)
#             num_Ir_neigh_j += Ir_neigh_adjustment

#             if num_Ir_neigh_j != num_metal_neigh_2:
#                 if Ir_neigh_adjustment == 0:

#                     # print("")
#                     print("name:", name)
#                     print(
#                         "oxy_ind:", oxy_ind, "|",
#                         "Original num Ir Neigh: ", num_Ir_neigh_j, "|",
#                         "New num Ir Neigh: ", num_metal_neigh_2, "|",
#                         "Ir adjustment:", Ir_neigh_adjustment, "|",
#                         "orig_df_coord_was_used:", orig_df_coord_was_used, "|",
#                         )

#             # I shouldn't have to do this, but we know that there is at least 1 Ir-O bond (to the active Ir) so we'll just manually set it here
#             if num_Ir_neigh_j == 0:
#                 num_Ir_neigh_j = 1
#         #__|

#         num_metal_neigh_2 += Ir_neigh_adjustment

#         # second_shell_coord_list.append(num_Ir_neigh_j)
#         # tmp_list.append(2 / num_Ir_neigh_j)

#         tmp_list.append(2 / num_metal_neigh_2)
#         second_shell_coord_list.append(num_metal_neigh_2)


#     # second_shell_coord_list
#     effective_ox_state = np.sum(tmp_list)
#     #__|

# # #####################################################
# out_dict = dict()
# # #####################################################
# out_dict["effective_ox_state"] = effective_ox_state
# out_dict["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord
# out_dict["num_missing_Os"] = num_missing_Os
# out_dict["orig_slab_good"] = orig_slab_good_i
# # #####################################################
# return(out_dict)
# #__|

# + jupyter={"source_hidden": true}
# num_Ir_neigh
# -



# +
# feature_cols

# columns
