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

# # Collect DFT data into *, *O, *OH collections
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_init_slabs,

    get_other_job_ids_in_set,
    )

# #########################################################
from local_methods import (
    read_magmom_comp_data,
    save_magmom_comp_data,
    process_group_magmom_comp,
    )





import random
from methods import get_df_struct_drift, get_df_magmom_drift
from itertools import combinations
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Script Inputs

redo_all_jobs = False
# redo_all_jobs = True

# ### Read Data

# +
# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_jobs_anal = get_df_jobs_anal()

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_atoms_sorted_ind_2 = df_atoms_sorted_ind.set_index("job_id")

# #########################################################
df_init_slabs = get_df_init_slabs()

# #########################################################
magmom_data_dict = read_magmom_comp_data()

# #########################################################
df_struct_drift = get_df_struct_drift()
df_struct_drift = df_struct_drift.set_index("pair_str", drop=False)

# #########################################################
df_magmom_drift_prev = get_df_magmom_drift()
df_magmom_drift_prev = df_magmom_drift_prev.set_index("pair_str", drop=False)
# -
# ### Filter down to `oer_adsorbate` systems

# +
df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal = df_jobs_anal.droplevel(level=0)


df_ind = df_atoms_sorted_ind.index.to_frame()
df_atoms_sorted_ind = df_atoms_sorted_ind.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_atoms_sorted_ind = df_atoms_sorted_ind.droplevel(level=0)

# +
# #########################################################
# Only completed jobs will be considered
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
# Dropping rows that failed atoms sort, now it's just one job that blew up 
# job_id = "dubegupi_27"
df_failed_to_sort = df_atoms_sorted_ind[
    df_atoms_sorted_ind.failed_to_sort == True]

# df_jobs_anal_i = df_jobs_anal_i.drop(labels=df_failed_to_sort.index)
# df_jobs_anal_i = df_jobs_anal_i.drop(labels=df_failed_to_sort.droplevel(level=0).index)
df_jobs_anal_i = df_jobs_anal_i.drop(labels=df_failed_to_sort.index)


# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]

# +
indices_to_keep = []
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal_no_o.groupby(groupby_cols)
# for name_i, group in grouped:

if True:
    name_i = ('nersc', 'hadogato_47', 88.0)
    group = grouped.get_group(name_i)

    group_index = group.index.to_frame()
    ads_list = list(group_index.ads.unique())
    oh_present = "oh" in ads_list
    bare_present = "bare" in ads_list
    all_req_ads_present = oh_present and bare_present
    if all_req_ads_present:
        indices_to_keep.extend(group.index.tolist())

df_jobs_anal_no_o_all_ads_pres = df_jobs_anal_no_o.loc[
    indices_to_keep    
    ]
df_i = df_jobs_anal_no_o_all_ads_pres
# -
# ### Magmom comparison

# +
# #########################################################
groups_to_process = []
# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
# #########################################################
iterator = tqdm(grouped, desc="1st loop")
for i_cnt, (name_i, group) in enumerate(iterator):

#     print(name_i)

# if True:
#     # name_i = ('sherlock', 'batipoha_75', 36.0)
#     # name_i = ('sherlock', 'fogalonu_46', 16.0)
#     # name_i = ('nersc', 'gekawore_16', 86.0)
#     name_i = ('nersc', 'hadogato_47', 88.0)
#     group = grouped.get_group(name_i)



    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    df_index = df_jobs_anal_i.index.to_frame()

    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        [True for i in range(len(df_index))]
        ]

    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index    
        ]

    group_w_o = pd.concat([group, row_o_i, ], axis=0)
    # #####################################################
    job_ids_list = group_w_o.job_id_max.tolist()


    # #####################################################
    # Deciding whether to reprocess the job or not
    # #####################################################
    out_dict_i = magmom_data_dict.get(name_i, None)
    # #####################################################
    if out_dict_i is None:
        run_job = True
    else:
        run_job = False

        job_ids_prev = out_dict_i.get("job_ids", None)
        if job_ids_prev is None:
            run_job = True
        else:
            if list(np.sort(job_ids_prev)) != list(np.sort(job_ids_list)):
                run_job = True

    if redo_all_jobs:
        run_job = True


    # #####################################################
    # Testing whether all entries in df_atoms_sorted_ind exist
    job_ids_list = list(set(group_w_o.job_id_max.tolist()))

    all_pairs_ready_list = []
    for job_id_i in job_ids_list:
        job_id_in_sorted = False
        if job_id_i in df_atoms_sorted_ind.job_id.tolist():
            job_id_in_sorted = True
        all_pairs_ready_list.append(job_id_in_sorted)
    all_pairs_ready = all(all_pairs_ready_list)

    if not all_pairs_ready:
        print("Not all job_ids have row in df_atoms_sorted_ind:", name_i)
        run_job = False



    # #####################################################
    if run_job:
        # print("This is good:", name_i)

        groups_to_process.append(name_i)

        # COMMENT THIS OUT TO RUN PARALLEL!!!!!!!

#         out_dict_i = process_group_magmom_comp(
#             name=name_i,
#             group=group_w_o,
#             write_atoms_objects=False,
#             verbose=False,
#             )

#         save_magmom_comp_data(name_i, out_dict_i)
# -

# ### Running magmom comparison in parallel

# +
def method_wrap(input_dict):
    group_w_o = input_dict["group_w_o"]
    name_i = input_dict["name_i"]

    out_dict_i = process_group_magmom_comp(
        name=name_i,
        group=group_w_o,
        write_atoms_objects=False,
        verbose=False,
        )

    save_magmom_comp_data(name_i, out_dict_i)


input_list = []
for name_i in groups_to_process:
    # #####################################################
    group_i = grouped.get_group(name_i)
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    df_index = df_jobs_anal_i.index.to_frame()

    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        [True for i in range(len(df_index))]
        ]

    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index    
        ]

    group_w_o = pd.concat([group_i, row_o_i, ], axis=0)

    input_dict_i = dict(
        group_w_o=group_w_o,
        name_i=name_i,
        )
    input_list.append(input_dict_i)

variables_dict = dict()
traces_all = Pool().map(
    partial(
        method_wrap,  # METHOD
        **variables_dict,  # KWARGS
        ),
    input_list,
    )

# + active=""
#
#
#
# -

# ### Identifying which slabs have zero magmoms

# +
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():
    # #########################################################
    data_dict_i = dict()
    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################


    # #####################################################
    job_id_i = row_i.job_id_max
    # #####################################################
    
    # #####################################################
    process_row = False
    if name_i in df_atoms_sorted_ind.index:
        process_row = True
        row_atoms_i = df_atoms_sorted_ind.loc[name_i]
        # #################################################
        atoms = row_atoms_i.atoms_sorted_good
        magmoms_i = row_atoms_i.magmoms_sorted_good
        # #################################################

            # (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    # row_atoms_i = df_atoms_sorted_ind.loc[
    #     (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    
    if process_row:
        if atoms.calc != None:
            magmoms_i = atoms.get_magnetic_moments()
        else:
            magmoms_i = magmoms_i

        sum_magmoms_i = np.sum(magmoms_i)
        sum_abs_magmoms = np.sum(np.abs(magmoms_i))

        # #########################################################
        data_dict_i["compenv"] = compenv_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["ads"] = ads_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["att_num"] = att_num_i
        # #########################################################
        data_dict_i["job_id"] = job_id_i
        data_dict_i["sum_magmoms"] = sum_magmoms_i
        data_dict_i["sum_abs_magmoms"] = sum_abs_magmoms
        # #########################################################
        data_dict_list.append(data_dict_i)
        # #########################################################

df_magmoms = pd.DataFrame(data_dict_list)


# -

# ### Normalizing magmoms by number of atoms

# +
def method(row_i):
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_i = row_i.active_site
    att_num_i = row_i.att_num
    sum_magmoms_i = row_i.sum_magmoms
    sum_abs_magmoms_i = row_i.sum_abs_magmoms
    # #####################################################

    # #####################################################
    name_i = (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)
    row_slab_i = df_init_slabs.loc[name_i]
    # #####################################################
    num_atoms_i = row_slab_i.num_atoms
    # #####################################################

    sum_magmoms_pa = sum_magmoms_i / num_atoms_i
    sum_abs_magmoms_pa = sum_abs_magmoms_i / num_atoms_i

    # #####################################################
    row_i["sum_magmoms_pa"] = sum_magmoms_pa
    row_i["sum_abs_magmoms_pa"] = sum_abs_magmoms_pa
    # #####################################################
    return(row_i)
    # #####################################################

df_magmoms = df_magmoms.apply(
    method,
    axis=1)
# -

# ### Further analysis of magmom comparison (collapse into dataframe)

# +
data_dict_list = []
for name_i in magmom_data_dict.keys():
    data_dict_i = dict()

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    magmom_data_dict[name_i].keys()

    from IPython.display import display
    df_magmoms_comp_i = magmom_data_dict[name_i]["df_magmoms_comp"]
    # display(df_magmoms_comp_i)
    min_val_i = df_magmoms_comp_i.sum_norm_abs_magmom_diff.min()

    # #####################################################
    data_dict_i["name"] = name_i
    data_dict_i["min_sum_norm_abs_magmom_diff"] = min_val_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df = pd.DataFrame(data_dict_list)
df.min_sum_norm_abs_magmom_diff.min()

df = df.sort_values("min_sum_norm_abs_magmom_diff")

# + active=""
#
#
#
# -

# ## Magmom Comparison

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

systems_already_processed = []
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

    # mean_displacement_dict = dict()

    # data_dict_list = []

    good_binary_pairs = []
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

        if ads_0 != ads_1:
            good_binary_pairs.append(pair_j)

    # #####################################################
    for pair_j in good_binary_pairs:
        pair_str_sorted = "__".join(list(np.sort(pair_j)))

        if pair_str_sorted in df_magmom_drift_prev.pair_str.unique().tolist():
            systems_already_processed.append(pair_str_sorted)
        else:
            systems_to_process.append(pair_str_sorted)

# +
# # TEMP
# print(111 * "TEMP | ")

# systems_to_process = random.sample(
#     systems_to_process, 10)

# +
# systems_to_process = [
#     'gadahake_36__nenisefu_41',
#     'dosapofu_77__vulopuno_60',
#     'pebelena_58__rerawubo_36',
#     'bibupufo_15__robivufe_82',
#     'bibuhuvi_91__niwatiwo_14',
#     'bitawobo_02__wewukufo_42',
#     'gubipugu_00__himesabi_01',
#     'pesipuho_48__puvafuwe_56',
#     'dubagito_87__valatumu_96',
#     'gotuderi_74__vetabawu_03',
#     ]

# + active=""
#
#

# +
group_cols = [
    'job_type', 'compenv', 'slab_id',
    'bulk_id', 'active_site', 'facet',
    ]

df_list = []
grouped = df_jobs_i.groupby(group_cols)
iterator = tqdm(grouped, desc="1st loop")

for i_cnt, (name_i, group_i) in enumerate(iterator):

    # #####################################################
    name_dict_i = dict(zip(
        group_cols,
        list(name_i)))
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


    good_binary_pairs = []
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

        if ads_0 != ads_1:
            good_binary_pairs.append(pair_j)


    for pair_j in good_binary_pairs:
        # print(pair_j)

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

        # #############################################
        row_atoms_0 = df_atoms_sorted_ind_2.loc[job_id_0]
        row_atoms_1 = df_atoms_sorted_ind_2.loc[job_id_1]
        # #############################################
        atoms_0 = row_atoms_0.atoms_sorted_good
        atoms_1 = row_atoms_1.atoms_sorted_good
        magmoms_sorted_good_0 = row_atoms_0.magmoms_sorted_good
        magmoms_sorted_good_1 = row_atoms_1.magmoms_sorted_good
        # #############################################

        pair_str_sorted = "__".join(list(np.sort(pair_j)))


        # # TEMP
        # systems_to_process = [
        #     'fukohesi_27__vunosepi_77',
        #     ]

        if pair_str_sorted in systems_to_process:

            if pair_str_sorted == 'fukohesi_27__vunosepi_77':
                print("Found fukohesi_27__vunosepi_77")

            # if atoms_0 is None or atoms_1 is None:
            if atoms_0 is not None and atoms_1 is not None:

                if magmoms_sorted_good_0 is None:
                    magmoms_0 = atoms_0.get_magnetic_moments()
                else:
                    magmoms_0 = magmoms_sorted_good_0

                if magmoms_sorted_good_1 is None:
                    magmoms_1 = atoms_1.get_magnetic_moments()
                else:
                    magmoms_1 = magmoms_sorted_good_1



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

                # df_match_2 = df_match[df_match.closest_distance > 0.000001]
                # mean_displacement = df_match_2["closest_distance"].mean()


























                # #########################################
                name_new_i = (name_i[1], name_i[2], name_i[4], )
                if name_new_i not in magmom_data_dict.keys():
                    tmp = 42
                    # if verbose:
                    #     print("name_new_i not in magmom_data_dict")
                    #     print(name_new_i)

                else:

                    magmom_data_i = magmom_data_dict[name_new_i]
                    # #####################################
                    # df_magmoms_comp = magmom_data_i["df_magmoms_comp"]
                    # good_triplet_comb = magmom_data_i["good_triplet_comb"]
                    # job_ids = magmom_data_i["job_ids"]
                    pair_wise_magmom_comp_data = magmom_data_i["pair_wise_magmom_comp_data"]
                    # #####################################
                    # pair_data_i = pair_wise_magmom_comp_data[
                    #     tuple(np.sort(pair_j))]
                    pair_sort = tuple(np.sort(pair_j))
                    pair_rev_sort = tuple(np.sort(pair_j)[::-1])
                    if pair_sort in list(pair_wise_magmom_comp_data.keys()):
                        pair_key = pair_sort
                    elif pair_rev_sort in list(pair_wise_magmom_comp_data.keys()):
                        pair_key = pair_rev_sort
                    else:
                        pair_key = None
                        print("Oh oh issue")

                    pair_data_i = pair_wise_magmom_comp_data[pair_key]
                    # #####################################
                    delta_magmoms_i = pair_data_i["delta_magmoms"]
                    tot_abs_magmom_diff_i = pair_data_i["tot_abs_magmom_diff"]
                    norm_abs_magmom_diff_i = pair_data_i["norm_abs_magmom_diff"]
                    ads_indices_not_used_i = pair_data_i["ads_indices_not_used"]
                    atoms_ave_i = pair_data_i["atoms_ave"]

                    delta_magmoms_unsorted_i = pair_data_i["delta_magmoms_unsorted"]
                    # #####################################

                    row_struct_drift = df_struct_drift.loc[pair_str_sorted]
                    octahedra_atoms = row_struct_drift.octahedra_atoms








                    # #####################################
                    # Get the summed absolute magmom atom-diffs
                    d_magmoms = []
                    for ind_i in list(atoms_0.constraints[0].get_indices()):

                        delta_magmom_i = None
                        for delta_magmom_j in delta_magmoms_i:
                            tmp = 42

                            if delta_magmom_j[0] == ind_i:
                                delta_magmom_i = delta_magmom_j

                        d_magmom_i = delta_magmom_i[1]
                        d_magmoms.append(d_magmom_i)

                    sum_abs_d_magmoms__constrained = np.sum(
                        np.abs(d_magmoms)
                        )    

                    sum_abs_d_magmoms__constrained_pa = sum_abs_d_magmoms__constrained / len(d_magmoms)




                    # #####################################
                    nonoctahedra_indices = []
                    for atom_i in atoms_0:
                        if not atom_i.index in octahedra_atoms:
                            nonoctahedra_indices.append(atom_i.index)

                    d_magmoms = []
                    for ind_i in list(nonoctahedra_indices):
                        delta_magmom_i = None
                        for delta_magmom_j in delta_magmoms_i:
                            if delta_magmom_j[0] == ind_i:
                                delta_magmom_i = delta_magmom_j

                        d_magmom_i = delta_magmom_i[1]
                        d_magmoms.append(d_magmom_i)

                    sum_abs_d_magmoms__nonocta = np.sum(
                        np.abs(d_magmoms)
                        )
                    sum_abs_d_magmoms__nonocta_pa = sum_abs_d_magmoms__nonocta / len(d_magmoms)




                    # #####################################
                    d_magmoms = []
                    for ind_i in octahedra_atoms:
                        delta_magmom_i = None
                        for delta_magmom_j in delta_magmoms_i:
                            if delta_magmom_j[0] == ind_i:
                                delta_magmom_i = delta_magmom_j

                        d_magmom_i = delta_magmom_i[1]
                        d_magmoms.append(d_magmom_i)

                    sum_abs_d_magmoms__octa = np.sum(
                        np.abs(d_magmoms)
                        )
                    sum_abs_d_magmoms__octa_pa = sum_abs_d_magmoms__octa / len(d_magmoms)


                    # #####################################
                    data_dict_i = dict()
                    # #####################################
                    data_dict_i["pair_str"] = pair_str_sorted
                    data_dict_i["job_id_0"] = pair_j[0]
                    data_dict_i["job_id_1"] = pair_j[1]
                    data_dict_i["job_ids"] = list(pair_j)
                    data_dict_i["ads_0"] = ads_0
                    data_dict_i["ads_1"] = ads_1
                    data_dict_i["att_num_0"] = att_num_0
                    data_dict_i["att_num_1"] = att_num_1
                    data_dict_i["sum_abs_d_magmoms__constrained"] = sum_abs_d_magmoms__constrained
                    data_dict_i["sum_abs_d_magmoms__constrained_pa"] = sum_abs_d_magmoms__constrained_pa
                    data_dict_i["sum_abs_d_magmoms__nonocta"] = sum_abs_d_magmoms__nonocta
                    data_dict_i["sum_abs_d_magmoms__nonocta_pa"] = sum_abs_d_magmoms__nonocta_pa
                    data_dict_i["sum_abs_d_magmoms__octa"] = sum_abs_d_magmoms__octa
                    data_dict_i["sum_abs_d_magmoms__octa_pa"] = sum_abs_d_magmoms__octa_pa
                    # #####################################
                    data_dict_list.append(data_dict_i)
                    # #####################################

                df_tmp = pd.DataFrame(data_dict_list)
                df_list.append(df_tmp)



# #########################################################
df_magmom_drift = pd.concat(df_list, axis=0)
# #########################################################

# +
df_magmom_drift_new = pd.concat([
    df_magmom_drift_prev,
    df_magmom_drift,
    ], axis=0)

df_magmom_drift_new = df_magmom_drift_new[
    ~df_magmom_drift_new.index.duplicated(keep="first")]
# -

# ### Save data to pickle

# +
root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/compare_magmoms")
directory = os.path.join(root_dir, "out_data")
if not os.path.exists(directory):
    os.makedirs(directory)

# #########################################################
path_i = os.path.join(directory, "df_magmoms.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_magmoms, fle)

# #########################################################
path_i = os.path.join(directory, "df_magmom_drift.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_magmom_drift_new, fle)

# +
from methods import get_df_magmom_drift

df_magmom_drift_tmp = get_df_magmom_drift()

# +
from methods import get_df_magmoms

df_magmoms_tmp = get_df_magmoms()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("compare_magmoms.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_magmom_drift_new.shape

# # (85207, 14)
# # (85207, 14)

# + jupyter={"source_hidden": true}
# group_i_3.shape
# group_i_2.shape

# group_i_3

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_atoms_sorted_ind

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_magmom_drift_new

# + jupyter={"source_hidden": true}
# df_magmom_drift_new

# + jupyter={"source_hidden": true}
# assert False
