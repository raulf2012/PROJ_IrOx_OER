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

# import numpy as np
import pandas as pd

# #########################################################
from proj_data import metal_atom_symbol

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_coord,
    )

# #########################################################
from local_methods import get_effective_ox_state
# -

# # Script Inputs

# verbose = True
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
# df_jobs_anal[df_jobs_anal.job_id_max == "vupihona_68"].index.tolist()

# +
# df_jobs_anal = df_jobs_anal.loc[
#     [('nersc', 'galopuba_86', 'o', 'NaN', 1)]
#     ]

# +
df_jobs_anal_i =  df_jobs_anal[df_jobs_anal.job_completely_done == True]

idx = pd.IndexSlice
df_jobs_anal_i = df_jobs_anal_i.loc[idx[:, :, "o", :, :], :]

# #########################################################
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():
    if verbose:
        name_concat_i = "_".join([str(i) for i in list(name_i)])
        print(40 * "=")
        print(name_concat_i)

    # #####################################################
    name_dict_i = dict(zip(
        list(df_jobs_anal_i.index.names),
        list(name_i)))
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################


    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_i]
    # #####################################################
    atoms_sorted_good_i = row_atoms_i.atoms_sorted_good
    # #####################################################
    atoms = atoms_sorted_good_i

    # #####################################################
    row_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_sites_i.active_sites_unique
    # #####################################################

    
    for active_site_j in active_sites_unique_i:
        data_dict_j = dict()
        if verbose:
            print("\t", "active_site_j:", active_site_j)

        # #################################################
        name_i = (
            compenv_i, slab_id_i, ads_i,
            active_site_i, att_num_i)
        df_coord_i = get_df_coord(
            mode="post-dft",
            post_dft_name_tuple=name_i)

        eff_ox_j = get_effective_ox_state(
            df_coord_i=df_coord_i,
            active_site_j=active_site_j,
            metal_atom_symbol=metal_atom_symbol)
        # #################################################

        # #################################################
        data_dict_j.update(name_dict_i)
        data_dict_j["eff_oxid_state"] = eff_ox_j
        data_dict_j["active_site"] = active_site_j
        # data_dict_j[""] = 
        # data_dict_j[""] = 
        # data_dict_j[""] = 
        # data_dict_j[""] = 
        # #################################################
        data_dict_list.append(data_dict_j)
        # #################################################
# -

# eff_oxid_state.tolist()
df_eff_ox =  pd.DataFrame(data_dict_list)
df_eff_ox.head()

# +
# # Pickling data ###########################################
# import os; import pickle
# directory = "out_data"
# if not os.path.exists(directory): os.makedirs(directory)
# with open(os.path.join(directory, "df_eff_ox.pickle"), "wb") as fle:
#     pickle.dump(df_eff_ox, fle)
# # #########################################################

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/oxid_state")

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_eff_ox.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_eff_ox, fle)
# #########################################################

# #########################################################
import pickle; import os
with open(path_i, "rb") as fle:
    df_eff_ox = pickle.load(fle)
# #########################################################

# +
from methods import get_df_eff_ox

df_eff_ox_tmp = get_df_eff_ox()
df_eff_ox_tmp.head()

# +
# # #########################################################
# import pickle; import os
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/feature_engineering",
#     "out_data/df_eff_ox.pickle")
# with open(path_i, "rb") as fle:
#     df_eff_ox = pickle.load(fle)
# # #########################################################
# -

# /home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/workflow/feature_engineering


# + active=""
#
#
#
#

# +
# atoms.write("tmp.traj")
# atoms.write("tmp.cif")

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_jobs_anal_i

# + jupyter={"source_hidden": true}
# def get_effective_ox_state(
#     df_coord_i=None,
#     ):
#     """
#     """
#     #| - get_effective_ox_state
#     df_coord_i = df_coord_i.set_index("structure_index", drop=False)

#     # #########################################################
#     # row_coord_i = df_coord_i.loc[21]
#     row_coord_i = df_coord_i.loc[active_site_j]

#     nn_info_i = row_coord_i.nn_info

#     neighbor_count_i = row_coord_i.neighbor_count
#     num_Ir_neigh = neighbor_count_i.get("Ir", 0)

#     mess_i = "For now only deal with active sites that have 1 Ir neighbor"
#     assert num_Ir_neigh == 1, mess_i

#     for j_cnt, nn_j in enumerate(nn_info_i):
#         site_j = nn_j["site"]
#         elem_j = site_j.as_dict()["species"][0]["element"]

#         if elem_j == metal_atom_symbol:
#             corr_j_cnt = j_cnt

#     site_j = nn_info_i[corr_j_cnt]
#     metal_index = site_j["site_index"]

#     # #########################################################
#     row_coord_i = df_coord_i.loc[metal_index]

#     neighbor_count_i = row_coord_i["neighbor_count"]

#     num_O_neigh = neighbor_count_i.get("O", 0)

#     mess_i = "There should be exactly 6 oxygens about the Ir atom"
#     assert num_O_neigh == 6, mess_i

#     num_neighbors_i = row_coord_i.num_neighbors

#     mess_i = "Only 6 neighbors total is allowed, all oxygens"
#     assert num_neighbors_i == 6, mess_i

#     nn_info_i =  row_coord_i.nn_info


#     # #########################################################
#     second_shell_coord_list = []
#     tmp_list = []
#     for nn_j in nn_info_i:
#         tmp = 42

#         site_index = nn_j["site_index"]

#         row_coord_j = df_coord_i.loc[site_index]

#         neighbor_count_j = row_coord_j.neighbor_count

#         num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)

#         # print("num_Ir_neigh_j:", num_Ir_neigh_j)

#         second_shell_coord_list.append(num_Ir_neigh_j)

#         tmp_list.append(2 / num_Ir_neigh_j)

#     # second_shell_coord_list
#     effective_ox_state = np.sum(tmp_list)
#     #__|

# +
# "_".join([str(i) for i in list(name_i)])

# +
# atoms
# atoms
# row_i
