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

# # Compute degree of structural drift across different slabs in OER sets
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

# #########################################################
from proj_data import metal_atom_symbol
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_other_job_ids_in_set,
    nearest_atom_mine,
    get_df_coord,
    get_df_coord_wrap,
    get_df_struct_drift,

    match_atoms,
    )

# #########################################################
from local_methods import (
    # match_atoms,
    get_mean_displacement_octahedra,
    )

from methods import get_df_init_slabs
from methods import get_df_atoms_sorted_ind
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
df_jobs = get_df_jobs()
df_jobs_data = get_df_jobs_data()

df_struct_drift_old = get_df_struct_drift()
df_struct_drift_old = df_struct_drift_old.set_index("pair_str", drop=False)

df_init_slabs = get_df_init_slabs()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# + active=""
#
#

# +
# Removing *O calcs that don't have an active site
# It messes up the groupby
df_jobs_i = df_jobs[df_jobs.active_site != "NaN"]


# Only doing oer_adsorbate calculations
df_jobs_i = df_jobs_i[df_jobs_i.job_type == "oer_adsorbate"]

# +
group_cols = [
    'job_type', 'compenv', 'slab_id',
    'bulk_id', 'active_site', 'facet',
    ]


systems_to_process = []

df_list = []
grouped = df_jobs_i.groupby(group_cols)
iterator = tqdm(grouped, desc="1st loop")
for i_cnt, (name_i, group_i) in enumerate(iterator):

    # #####################################################
    name_dict_i = dict(zip(
        group_cols,
        list(name_i)))
    # #####################################################
    compenv_i = name_dict_i["compenv"]
    slab_id_i = name_dict_i["slab_id"]
    active_site_i = name_dict_i["active_site"]
    # #####################################################


    row_tmp = group_i[group_i.ads != "o"].iloc[0]

    group_i_2 = get_other_job_ids_in_set(
        row_tmp.name,
        df_jobs=df_jobs,
        oer_set=True,
        only_last_rev=True)

    group_i_3 = pd.merge(
        group_i_2,
        df_jobs_data[["final_atoms"]],
        how="left",
        left_index=True,
        right_index=True)
    group_i_3 = group_i_3.dropna(subset=["final_atoms", ])


    all_binary_pairs = list(combinations(
        group_i_3.index.tolist(), 2))

    mean_displacement_dict = dict()

    data_dict_list = []


    # # TEMP
    # print(20 * "TEMP | ")
    # # all_binary_pairs = [('pupofufo_14', 'tabupodu_76'), ]
    # # all_binary_pairs = [('lalanota_37', 'wepewido_07'), ]
    # all_binary_pairs = [('tuwetuta_57', 'firitune_96'), ]

    # #####################################################
    for pair_j in all_binary_pairs:

        # #################################################
        job_id_0 = pair_j[0]
        job_id_1 = pair_j[1]
        # #################################################
        row_jobs_0 = df_jobs.loc[job_id_0]
        row_jobs_1 = df_jobs.loc[job_id_1]
        # #################################################
        ads_0 = row_jobs_0.ads
        ads_1 = row_jobs_1.ads
        att_num_0 = row_jobs_0.att_num
        att_num_1 = row_jobs_1.att_num
        active_site_0 = row_jobs_0.active_site
        active_site_1 = row_jobs_1.active_site
        # #################################################


        # Getting sorted atoms objects
        name_atoms_sorted_0 = (
            "oer_adsorbate", compenv_i, slab_id_i,
            ads_0, active_site_0, att_num_0, )

        name_atoms_sorted_1 = (
            "oer_adsorbate", compenv_i, slab_id_i,
            ads_1, active_site_1, att_num_1, )

        row_atoms_sorted = df_atoms_sorted_ind.loc[name_atoms_sorted_0]
        atoms_0 = row_atoms_sorted.atoms_sorted_good

        row_atoms_sorted = df_atoms_sorted_ind.loc[name_atoms_sorted_1]
        atoms_1 = row_atoms_sorted.atoms_sorted_good


        pair_str_sorted = "__".join(list(np.sort(pair_j)))



        if pair_str_sorted in df_struct_drift_old.index:
            # #############################################
            row_struct_drift_i = df_struct_drift_old.loc[pair_str_sorted]
            # #############################################
            mean_displacement = row_struct_drift_i["mean_displacement"]
            mean_displacement_octahedra = row_struct_drift_i["mean_displacement_octahedra"]
            octahedra_atoms = row_struct_drift_i.octahedra_atoms
            note = row_struct_drift_i["note"]
            error = row_struct_drift_i["error"]
            # #############################################



            # #############################################
            data_dict_i = dict()
            # #############################################
            data_dict_i["pair_str"] = pair_str_sorted
            data_dict_i["job_id_0"] = pair_j[0]
            data_dict_i["job_id_1"] = pair_j[1]
            data_dict_i["job_ids"] = list(pair_j)
            data_dict_i["ads_0"] = ads_0
            data_dict_i["ads_1"] = ads_1
            data_dict_i["att_num_0"] = att_num_0
            data_dict_i["att_num_1"] = att_num_1
            data_dict_i["mean_displacement"] = mean_displacement
            data_dict_i["mean_displacement_octahedra"] = mean_displacement_octahedra
            data_dict_i["octahedra_atoms"] = octahedra_atoms
            data_dict_i["note"] = note
            data_dict_i["error"] = error
            # #############################################
            data_dict_list.append(data_dict_i)
            # #############################################

        else:
            systems_to_process.append(pair_str_sorted)

    df_tmp = pd.DataFrame(data_dict_list)
    df_list.append(df_tmp)


# #########################################################
df_struct_drift__done_prev = pd.concat(df_list, axis=0)
# #########################################################

# +
group_cols = [
    'job_type', 'compenv', 'slab_id',
    'bulk_id', 'active_site', 'facet',
    ]




df_list = []
grouped = df_jobs_i.groupby(group_cols)
iterator = tqdm(grouped, desc="1st loop")
for i_cnt, (name_i, group_i) in enumerate(iterator):


# # TEMP Use for testing
# if True:
#     # name_i = ('oer_adsorbate', 'slac', 'relovalu_12', 'zimixdvdxd', 24.0, '2-1-10')
#     # name_i = ('oer_adsorbate', 'slac', 'vuraruna_65', 'z36lb3bdcq', 50.0, '001')
#     # name_i = ('oer_adsorbate', 'nersc', 'dakoputu_58', 'bpc2nk6qz1', 74.0, '212')
#     # name_i = ('oer_adsorbate', 'nersc', 'hibetede_02', 'mkmsvkcyc5', 32.0, '110')
#     name_i = ('oer_adsorbate', 'slac', 'vuraruna_65', 'z36lb3bdcq', 50.0, '001')
#     group_i = grouped.get_group(name_i)



    # print("")
    # print(20 * "-")
    # print(name_i)


    # # TEMP
    # if i_cnt > 10:
    #     break



    # #####################################################
    name_dict_i = dict(zip(
        group_cols,
        list(name_i)))
    # #####################################################
    compenv_i = name_dict_i["compenv"]
    slab_id_i = name_dict_i["slab_id"]
    active_site_i = name_dict_i["active_site"]
    # #####################################################


    row_tmp = group_i[group_i.ads != "o"].iloc[0]

    group_i_2 = get_other_job_ids_in_set(
        row_tmp.name,
        df_jobs=df_jobs,
        oer_set=True,
        only_last_rev=True)

    group_i_3 = pd.merge(
        group_i_2,
        df_jobs_data[["final_atoms"]],
        how="left",
        left_index=True,
        right_index=True)
    group_i_3 = group_i_3.dropna(subset=["final_atoms", ])


    all_binary_pairs = list(combinations(
        group_i_3.index.tolist(), 2))

    mean_displacement_dict = dict()

    data_dict_list = []


    # # TEMP
    # print(20 * "TEMP | ")
    # # all_binary_pairs = [('pupofufo_14', 'tabupodu_76'), ]
    # # all_binary_pairs = [('lalanota_37', 'wepewido_07'), ]
    # all_binary_pairs = [('tuwetuta_57', 'firitune_96'), ]

    # #####################################################
    for pair_j in all_binary_pairs:

        # #################################################
        job_id_0 = pair_j[0]
        job_id_1 = pair_j[1]
        # #################################################
        row_jobs_0 = df_jobs.loc[job_id_0]
        row_jobs_1 = df_jobs.loc[job_id_1]
        # #################################################
        ads_0 = row_jobs_0.ads
        ads_1 = row_jobs_1.ads
        att_num_0 = row_jobs_0.att_num
        att_num_1 = row_jobs_1.att_num
        active_site_0 = row_jobs_0.active_site
        active_site_1 = row_jobs_1.active_site
        # #################################################


        # Getting sorted atoms objects
        name_atoms_sorted_0 = (
            "oer_adsorbate", compenv_i, slab_id_i,
            ads_0, active_site_0, att_num_0, )

        name_atoms_sorted_1 = (
            "oer_adsorbate", compenv_i, slab_id_i,
            ads_1, active_site_1, att_num_1, )

        row_atoms_sorted = df_atoms_sorted_ind.loc[name_atoms_sorted_0]
        atoms_0 = row_atoms_sorted.atoms_sorted_good

        row_atoms_sorted = df_atoms_sorted_ind.loc[name_atoms_sorted_1]
        atoms_1 = row_atoms_sorted.atoms_sorted_good


        pair_str_sorted = "__".join(list(np.sort(pair_j)))



        # if pair_str_sorted in df_struct_drift_old.index:
        #     # print("IJSIDFISD")
        #     # #############################################
        #     row_struct_drift_i = df_struct_drift_old.loc[pair_str_sorted]
        #     # #############################################
        #     mean_displacement = row_struct_drift_i["mean_displacement"]
        #     mean_displacement_octahedra = row_struct_drift_i["mean_displacement_octahedra"]
        #     octahedra_atoms = row_struct_drift_i.octahedra_atoms
        #     note = row_struct_drift_i["note"]
        #     error = row_struct_drift_i["error"]
        #     # #############################################

        if pair_str_sorted not in df_struct_drift_old.index:

            print(pair_j)

        # else:
        # if True:


            # #############################################
            # Running analysis
            # #############################################
            root_dir = os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/job_analysis/slab_struct_drift",
                "out_data/df_match_files")
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)

            path_i = os.path.join(
                root_dir, pair_str_sorted + ".pickle")
            if Path(path_i).is_file():
                with open(path_i, "rb") as fle:
                    df_match = pickle.load(fle)
            else:
                print("Running:", pair_j)
                df_match = match_atoms(atoms_0, atoms_1)

                pickle_path = os.path.join(
                    root_dir, pair_str_sorted + ".pickle")
                with open(pickle_path, "wb") as fle:
                    pickle.dump(df_match, fle)

            df_match_2 = df_match[df_match.closest_distance > 0.000001]
            mean_displacement = df_match_2["closest_distance"].mean()


            # #############################################
            # Getting mean displacement of the octahedra
            # #############################################

            out_dict_1 = get_mean_displacement_octahedra(
                df_match=df_match,
                df_jobs=df_jobs,
                df_init_slabs=df_init_slabs,
                atoms_0=atoms_0,
                job_id_0=job_id_0,
                active_site=name_dict_i["active_site"],
                compenv=name_dict_i["compenv"],
                slab_id=name_dict_i["slab_id"],
                ads_0=ads_0,
                active_site_0=active_site_0,
                att_num_0=att_num_0,
                )
            mean_displacement_octahedra = out_dict_1["mean_displacement_octahedra"]
            metal_active_site = out_dict_1["metal_active_site"]
            note = out_dict_1["note"]
            error = out_dict_1["error"]
            octahedra_atoms = out_dict_1["octahedra_atoms"]



            # #################################################
            data_dict_i = dict()
            # #################################################
            data_dict_i["pair_str"] = pair_str_sorted
            data_dict_i["job_id_0"] = pair_j[0]
            data_dict_i["job_id_1"] = pair_j[1]
            data_dict_i["job_ids"] = list(pair_j)
            data_dict_i["ads_0"] = ads_0
            data_dict_i["ads_1"] = ads_1
            data_dict_i["att_num_0"] = att_num_0
            data_dict_i["att_num_1"] = att_num_1
            data_dict_i["mean_displacement"] = mean_displacement
            data_dict_i["mean_displacement_octahedra"] = mean_displacement_octahedra
            data_dict_i["octahedra_atoms"] = octahedra_atoms
            data_dict_i["note"] = note
            data_dict_i["error"] = error
            # #################################################
            data_dict_list.append(data_dict_i)
            # #################################################

    df_tmp = pd.DataFrame(data_dict_list)
    df_list.append(df_tmp)


# #########################################################
df_struct_drift = pd.concat(df_list, axis=0)
# #########################################################

# +
# df_struct_drift.shape

# (2242, 13)
# (4956, 13)
# (9895, 13)
# (10256,13)
# (11565, 13)
# -

df_struct_drift_new = pd.concat([
    df_struct_drift,
    df_struct_drift__done_prev,
    ], axis=0)

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/slab_struct_drift",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_struct_drift_new.pickle"), "wb") as fle:
    pickle.dump(df_struct_drift_new, fle)
    # pickle.dump(df_struct_drift, fle)
# #########################################################

# +
from methods import get_df_struct_drift

df_struct_drift_tmp = get_df_struct_drift()
# -

df_struct_drift_tmp.head()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("slab_struct_drift_2.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
