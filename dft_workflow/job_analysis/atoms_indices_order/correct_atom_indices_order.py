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

# # Correct the atom indices of post-DFT atoms with `ase-sort.dat`
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from ase import io

# #########################################################
from IPython.display import display

# #########################################################
from vasp.vasp_methods import read_ase_sort_dat

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_paths,
    are_dicts_the_same,
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_init_slabs,
    )

# #########################################################
from local_methods import (
    get_unique_job_ids_ase_sort,
    all_keys_equal_to_vals,
    get_df_atoms_ind,
    unique_ids_with_no_equal,
    atoms_distance_comparison,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Read data

# +
# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_paths = get_df_jobs_paths()

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_completed = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
df_init_slabs = get_df_init_slabs()
# -

# # Removing rows that don't have the necessary files present locally
#
# Might need to download them with rclone

group_cols = [
    "job_type", "compenv", "slab_id", "ads", "active_site", "att_num", 
    ]

# +
indices_to_process = []
grouped = df_jobs_anal_completed.groupby(
    group_cols
    )
for name, df_jobs_anal_i in grouped:

    # #####################################################
    df_jobs_groups = df_jobs.groupby(group_cols)
    df_jobs_i = df_jobs_groups.get_group(name)
    # #####################################################

    all_ok_to_process_list = []
    for job_id_j in df_jobs_i.index:
        # #################################################
        row_paths_j = df_jobs_paths.loc[job_id_j]
        # #################################################
        gdrive_path_j = row_paths_j.gdrive_path
        # #################################################
        
        job_ok_to_process = False

        # #################################################
        gdrive_path_j = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_j)

        # #############################
        path_tmp_0 = os.path.join(
            gdrive_path_j,
            "CONTCAR")
        contcar_present = False
        if Path(path_tmp_0).is_file():
            contcar_present = True

        # #############################
        path_tmp_1 = os.path.join(
            gdrive_path_j,
            ".SUBMITTED")
        submitted_present = False
        if Path(path_tmp_1).is_file():
            submitted_present = True

        # #############################
        path_j = os.path.join(
            gdrive_path_j,
            "ase-sort.dat")
        file_present = False
        if Path(path_j).is_file():
            file_present = True


        if file_present:
            job_ok_to_process = True
        elif not contcar_present and submitted_present:
            job_ok_to_process = True

        all_ok_to_process_list.append(job_ok_to_process)

    all_ok_to_process = all(all_ok_to_process_list)
    if all_ok_to_process:
        indices_to_process.extend(df_jobs_anal_i.index.tolist())
    else:
        print(name)
        # indices_to_not_process.extend()


# #########################################################
df_jobs_anal_completed_2 = df_jobs_anal_completed.loc[indices_to_process]
# #########################################################

# +
# # df = df_jobs_anal_completed.index.to_frame()
# df = df_jobs_anal.index.to_frame()
# df = df[
#     (df["slab_id"] == "momaposi_60") &
#     (df["ads"] == "o") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]
# df

# +
# grouped.get_group(
#     ('oer_adsorbate', 'sherlock', 'momaposi_60', 'o', 50.0, 1)
#     )

# +
# name

# +
# ('oer_adsorbate', 'sherlock', 'momaposi_60', 'o', 50.0, 1) in indices_to_process

# +
# assert False

# +
not_processed_indices = []
for index_i, row_i in df_jobs_anal_completed.iterrows():
    job_id_max_i = row_i.job_id_max
    if index_i not in df_jobs_anal_completed_2.index:
        not_processed_indices.append([job_id_max_i, index_i])

if len(not_processed_indices) > 0:
    print(
        "These systems don't have the required files locally",
        "Fix with rclone",
        "",
        sep="\n")
    tmp = [print(i[0], "|", i[1]) for i in not_processed_indices]

# + active=""
#
#
#
# -

# # Main Loop

# +
# #########################################################
data_dict_list = []
# #########################################################
grouped = df_jobs_anal_completed_2.groupby(
    ["job_type", "compenv", "slab_id", "ads", "active_site", "att_num", ])
for name, df_jobs_anal_i in grouped:
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    job_type_i = name[0]
    compenv_i = name[1]
    slab_id_i = name[2]
    ads_i = name[3]
    active_site_i = name[4]
    att_num_i = name[5]
    # #####################################################

    # #####################################################
    df_jobs_groups = df_jobs.groupby(group_cols)
    df_jobs_i = df_jobs_groups.get_group(name)
    # #####################################################


    df_atoms_ind_i = get_df_atoms_ind(
        df_jobs_i=df_jobs_i,
        df_jobs_paths=df_jobs_paths,
        )
    df_atoms_ind_i = df_atoms_ind_i.dropna()

    # #####################################################
    job_ids = df_atoms_ind_i.job_id.tolist()
    unique_job_ids = get_unique_job_ids_ase_sort(job_ids, df_atoms_ind_i)

    # #####################################################
    unique_ids_with_no_equal_i = unique_ids_with_no_equal(
        unique_job_ids=unique_job_ids,
        df_atoms_ind_i=df_atoms_ind_i,
        )
    if len(unique_ids_with_no_equal_i) > 1:
        print("Big problem, I think there should only be one unique atoms mapping for any job")

    unique_id = unique_ids_with_no_equal_i[0]

    # #####################################################
    row_i = df_atoms_ind_i.loc[unique_id]
    # #####################################################
    atom_index_mapping_i = row_i.atom_index_mapping
    sort_list_i = row_i.sort_list
    resort_list_i = row_i.resort_list
    # #####################################################


    # #####################################################
    data_dict_i["job_type"] = job_type_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["atom_index_mapping"] = atom_index_mapping_i
    data_dict_i["sort_list"] = sort_list_i
    data_dict_i["resort_list"] = resort_list_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_atoms_index = pd.DataFrame(data_dict_list)

index_cols = [
    "job_type",
    "compenv", "slab_id",
    "ads", "active_site", "att_num"]

df_atoms_index = df_atoms_index.set_index(index_cols)

# + active=""
#
#
#
#
#
# -

# ### Creating atoms objects with correct index order and testing

# +
data_dict_list = []
for name_i, row_i in df_jobs_anal_completed_2.iterrows():
    if verbose:
        print(40 * "=")
        print(name_i)

    data_dict_i = dict()

    # #########################################################
    job_type_i = name_i[0]
    compenv_i = name_i[1]
    slab_id_i = name_i[2]
    ads_i = name_i[3]
    active_site_i = name_i[4]
    att_num_i = name_i[5]
    # #########################################################

    # #########################################################
    job_id_max_i = row_i.job_id_max
    # #########################################################

    # #########################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_i = df_jobs_data_i[df_jobs_data_i.job_id == job_id_max_i].iloc[0]
    # #########################################################
    final_atoms_i = row_data_i.final_atoms
    # #########################################################

    # #####################################################
    row_atoms_index_i = df_atoms_index.loc[name_i]
    # #####################################################
    atom_index_mapping_i = row_atoms_index_i.atom_index_mapping
    sort_list_i = row_atoms_index_i.sort_list
    resort_list_i = row_atoms_index_i.resort_list
    # #####################################################

    # #####################################################
    row_init_slabs_i = df_init_slabs.loc[
        (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    # #####################################################
    init_atoms_i = row_init_slabs_i.init_atoms
    # #####################################################


    # print("final_atoms_i.get_global_number_of_atoms():", final_atoms_i.get_global_number_of_atoms())
    atoms_distance_0 = atoms_distance_comparison(init_atoms_i, final_atoms_i)


    was_sorted = False
    atoms_sorted_good = None
    failed_to_sort = False
    atoms_distance_1 = None
    magmoms_sorted_good = None
    if atoms_distance_0 > 2:

        atoms_sorted = final_atoms_i[resort_list_i]

        magmoms_sorted = final_atoms_i.get_magnetic_moments()
        magmoms_sorted = magmoms_sorted[resort_list_i]

        # atoms_distance_1 = atoms_distance_comparison(slab_final_i, atoms_sorted)
        atoms_distance_1 = atoms_distance_comparison(init_atoms_i, atoms_sorted)
        if atoms_distance_1 < 1.5:

            atoms_sorted_good = atoms_sorted
            magmoms_sorted_good = magmoms_sorted

            atoms_sorted_good.set_initial_magnetic_moments(magmoms_sorted_good)
            was_sorted = True

        else:
            failed_to_sort = True
            if verbose:
                print("The sorted atoms and the initial slab aren't too similar")
                print("Look into this manually")
                print("name_i:", name_i)

    else:
        atoms_sorted_good = final_atoms_i
        atoms_distance_1 = None
        magmoms_sorted_good = None

        # if verbose:
        #     print(atoms_distance_0)
        #     print("Look into this manually if the atoms_distance is less than 2")
        #     print("I currently think that every single atoms object's indices are shuffled after DFT")

    # #####################################################
    data_dict_i["job_type"] = job_type_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["att_num"] = att_num_i

    data_dict_i["job_id"] = job_id_max_i
    data_dict_i["was_sorted"] = was_sorted
    data_dict_i["failed_to_sort"] = failed_to_sort
    data_dict_i["atoms_sorted_good"] = atoms_sorted_good
    data_dict_i["atoms_distance_before_sorting"] = atoms_distance_0
    data_dict_i["atoms_distance_after_sorting"] = atoms_distance_1
    data_dict_i["magmoms_sorted_good"] = magmoms_sorted_good
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_atoms_sorted = pd.DataFrame(data_dict_list)

index_cols = [
    "job_type",
    "compenv", "slab_id",
    "ads", "active_site", "att_num"]
df_atoms_sorted = df_atoms_sorted.set_index(index_cols)
# -

df_comb_i = pd.concat(
    [
        df_atoms_index,
        df_atoms_sorted,
        ],
    axis=1,
    )

# ### Pickling `df_atoms_index`

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/atoms_indices_order",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_atoms_sorted_ind.pickle"), "wb") as fle:
    pickle.dump(df_comb_i, fle)
# #########################################################

# ### Read `df_atoms_index` with Pickle

from methods import get_df_atoms_sorted_ind
df_atoms_sorted_ind_tmp = get_df_atoms_sorted_ind()
df_atoms_sorted_ind_tmp.head()

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("correct_atom_indices_order.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# name_i

# + jupyter={"source_hidden": true}
# df_atoms_index

# + jupyter={"source_hidden": true}
# # row_atoms_index_i = 
# df_atoms_index.loc[name_i]

# + jupyter={"source_hidden": true}
# df_atoms_index.head()

# + jupyter={"source_hidden": true}
# df_atoms_sorted.head()

# + jupyter={"source_hidden": true}
# sort_list_i = df_comb_i.sort_list.iloc[0]
# resort_list_i = df_comb_i.resort_list.iloc[0]
# atom_index_mapping_i = df_comb_i.atom_index_mapping.iloc[0]

# + jupyter={"source_hidden": true}
# df_atoms_index

# + jupyter={"source_hidden": true}
# df_jobs_groups

# + jupyter={"source_hidden": true}
# name

# + jupyter={"source_hidden": true}
# df_jobs_i = 
# df_jobs_groups.get_group(name)

# + jupyter={"source_hidden": true}
# df_jobs_groups.head()

# + jupyter={"source_hidden": true}
# # df_jobs_i = 


# # name
# df_jobs_groups.get_group(name)

# + jupyter={"source_hidden": true}
# name

# + jupyter={"source_hidden": true}
# df_atoms_index

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# # TEMP
# print(222 * "TEMP | ")

# df_jobs_anal_completed_2 = df_jobs_anal_completed_2.loc[[
#     # ('slac', 'ralutiwa_59', 'o', 30.0, 1)
#     ('slac', 'vapopihe_87', 'o', 23.0, 1),
#     ]]
# df_jobs_anal_completed_2

# + jupyter={"source_hidden": true}
# df_jobs_anal_i

# + jupyter={"source_hidden": true}
# df_jobs_anal_completed_2.head()

# + jupyter={"source_hidden": true}
    # ["compenv", "slab_id", "ads", "active_site", "att_num", ])
