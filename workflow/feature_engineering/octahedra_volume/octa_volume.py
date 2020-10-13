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

# # Calculating the octahedral volume and other geometric quantities
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

# import numpy as np
import pandas as pd

# pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from proj_data import metal_atom_symbol

from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_coord,
    )

# #########################################################
# from local_methods import get_effective_ox_state
# -

# # Script Inputs

verbose = True
# verbose = False

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

# + active=""
#
#
#

# +
# #########################################################
# #########################################################

# TEMP
# print("TEMP")
# # name_i = ('nersc', 'mubolemu_18', 'o', 'NaN', 1)
# # name_i = ('sherlock', 'kagekiha_49', 'o', 'NaN', 1)
# name_i = ("nersc", "kalisule_45", "o", "NaN", 1)

# df_jobs_anal_i = df_jobs_anal_i.loc[[name_i]]

# +
# # TEMP

# df_index = df_jobs_anal_i.index.to_frame()

# df_index_i = df_index[
#     (df_index.compenv == "nersc") & \
#     (df_index.slab_id == "fosurufu_23") & \
#     [True for i in range(len(df_index))]
#     ]

# df_jobs_anal_i = df_jobs_anal_i.loc[
#     df_index_i.index.tolist()
#     ]

# +
df_jobs_anal_i =  df_jobs_anal_i[df_jobs_anal.job_completely_done == True]

idx = pd.IndexSlice
df_jobs_anal_i = df_jobs_anal_i.loc[idx[:, :, "o", :, :], :]

# +
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
    atoms = row_atoms_i.atoms_sorted_good
    # #####################################################

    # #####################################################
    row_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_sites_i.active_sites_unique
    # #####################################################


    # #####################################################
    # Write atoms to file
    file_name_i = "__".join([str(i) for i in name_i])
    file_name_i += ".cif"

    file_name_j = "__".join([str(i) for i in name_i])
    file_name_j += ".traj"

    # atoms.write(os.path.join("__temp__", file_name_i))
    # atoms.write(os.path.join("__temp__", file_name_j))
    
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


        # #################################################
        from local_methods import get_octa_vol
        vol_i = get_octa_vol(
            df_coord_i=df_coord_i,
            active_site_j=active_site_j,
            verbose=verbose)
        # #################################################
        from local_methods import get_octa_geom
        octa_geom_dict = get_octa_geom(
            df_coord_i=df_coord_i,
            active_site_j=active_site_j,
            atoms=atoms,
            verbose=verbose)
        # #################################################


        # #################################################
        data_dict_j.update(name_dict_i)
        data_dict_j.update(octa_geom_dict)
        data_dict_j["active_site"] = active_site_j
        data_dict_j["octa_vol"] = vol_i
        # #################################################
        data_dict_list.append(data_dict_j)
        # #################################################

# #########################################################
df_octa_vol = pd.DataFrame(data_dict_list)
# df_octa_vol.head()
# -

df_octa_vol

# +
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/octahedra_volume")

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_octa_vol.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_octa_vol, fle)
# #########################################################

# #########################################################
import pickle; import os
with open(path_i, "rb") as fle:
    df_octa_vol = pickle.load(fle)
# #########################################################

# + active=""
#
#
#
# -

# # Saving some data objects to test out elsewhere

# +
out_dict = dict(
    df_coord_i=df_coord_i,
    active_site_j=active_site_j,
    atoms=atoms,
    )

# Pickling data ###########################################
import os; import pickle
directory = "__temp__"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "data.pickle"), "wb") as fle:
    pickle.dump(out_dict, fle)
# #########################################################

# #########################################################
import pickle; import os
path_i = os.path.join(
    # os.environ[""],
    "__temp__",
    "data.pickle")
with open(path_i, "rb") as fle:
    out_data = pickle.load(fle)
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# data_dict_list

# + jupyter={"source_hidden": true}
# name_i = ('nersc', 'mubolemu_18', 'o', 'NaN', 1)

# # for name_i, row_i in df_jobs_anal_i.iterrows():

# df_jobs_anal_i = df_jobs_anal_i.loc[[name_i]]

# + jupyter={"source_hidden": true}
# df_jobs_anal[df_jobs_anal.job_id_max == "vupihona_68"].index.tolist()

# + jupyter={"source_hidden": true}
# df_jobs_anal = df_jobs_anal.loc[
#     [('nersc', 'galopuba_86', 'o', 'NaN', 1)]
#     ]

# + jupyter={"source_hidden": true}
# print("active_site_j:", active_site_j)

# atoms.write("__temp__/tmp.cif"

# + jupyter={"source_hidden": true}
# name_i
# row_i

# + jupyter={"source_hidden": true}
# file_name_i
# active_site_j

# + jupyter={"source_hidden": true}
# df_index = df_jobs_anal_i.index.to_frame()

# df_index_i = df_index[
#     (df_index.compenv == "nersc") & \
#     (df_index.slab_id == "fosurufu_23") & \
#     [True for i in range(len(df_index))]
#     ]

# df_jobs_anal_i = df_jobs_anal_i.loc[
#     df_index_i.index.tolist()
#     ]
