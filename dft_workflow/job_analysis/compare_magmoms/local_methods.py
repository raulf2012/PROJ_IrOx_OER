"""
"""

#| - Import Modules
import os
import sys

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from methods import (
    get_magmom_diff_data,
    get_df_jobs,
    get_df_atoms_sorted_ind,
    get_df_job_ids,
    CountFrequency,
    )
#__|


def process_group_magmom_comp(
    name=None,
    group=None,
    write_atoms_objects=False,
    verbose=False,
    ):
    """
    """
    #| - process_group_magmom_comp
    # #####################################################
    group_w_o = group

    # #####################################################
    out_dict = dict()
    out_dict["df_magmoms_comp"] = None
    out_dict["good_triplet_comb"] = None
    out_dict["job_ids"] = None
    # out_dict[""] =

    job_ids_list = list(set(group.job_id_max.tolist()))


    #| - Reading data
    # #########################################################
    df_jobs = get_df_jobs()

    # #########################################################
    df_atoms_sorted_ind = get_df_atoms_sorted_ind()
    df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

    # #########################################################
    df_job_ids = get_df_job_ids()
    df_job_ids = df_job_ids.set_index("job_id")

    from methods import read_magmom_comp_data

    assert name != None, "Must pass name to read previous data"

    magmom_comp_data_prev = read_magmom_comp_data(name=name)
    if magmom_comp_data_prev is not None:
        pair_wise_magmom_comp_data_prev = \
            magmom_comp_data_prev["pair_wise_magmom_comp_data"]
    #__|

    if write_atoms_objects:
        #| - Write atoms objects
        df_i = pd.concat([
            df_job_ids,
            df_atoms_sorted_ind.loc[
                group_w_o.job_id_max.tolist()
                ]
            ], axis=1, join="inner")

        # #########################################################
        df_index_i = group_w_o.index.to_frame()
        compenv_i = df_index_i.compenv.unique()[0]
        slab_id_i = df_index_i.slab_id.unique()[0]

        active_sites = [i for i in df_index_i.active_site.unique() if i != "NaN"]
        active_site_i = active_sites[0]

        folder_name = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i))
        # #########################################################


        for job_id_i, row_i in df_i.iterrows():
            tmp = 42

            job_id = row_i.name
            atoms = row_i.atoms_sorted_good
            ads = row_i.ads

            file_name = ads + "_" + job_id + ".traj"

            print("Is this saving to the right place d9sf")
            root_file_path = os.path.join("__temp__", folder_name)
            print(os.getcwd(), root_file_path)
            if not os.path.exists(root_file_path):
                os.makedirs(root_file_path)

            file_path = os.path.join(root_file_path, file_name)

            atoms.write(file_path)
        #__|

    # #####################################################
    #| - Getting good triplet combinations
    all_triplet_comb = list(itertools.combinations(
        group_w_o.job_id_max.tolist(), 3))

    good_triplet_comb = []
    for tri_i in all_triplet_comb:
        df_jobs_i = df_jobs.loc[list(tri_i)]

        # Triplet must not contain duplicate ads
        # Must strictly be a *O, *OH, and *bare triplet
        ads_freq_dict = CountFrequency(df_jobs_i.ads.tolist())

        tmp_list = list(ads_freq_dict.values())
        any_repeat_ads = [True if i > 1 else False for i in tmp_list]

        if not any(any_repeat_ads):
            good_triplet_comb.append(tri_i)
    #__|

    # #####################################################
    #| - MAIN LOOP
    if verbose:
        print(
            "Number of viable triplet combinations:",
            len(good_triplet_comb)
            )

    data_dict_list = []
    pair_wise_magmom_comp_data = dict()
    for tri_i in good_triplet_comb:
        #| - Process triplets
        data_dict_i = dict()

        if verbose:
            print("tri_i:", tri_i)

        all_pairs = list(itertools.combinations(tri_i, 2))

        df_jobs_i = df_jobs.loc[list(tri_i)]

        sum_norm_abs_magmom_diff = 0.
        for pair_i in all_pairs:

            # if pair_i in list(pair_wise_magmom_comp_data_prev.keys()):
            if (magmom_comp_data_prev is not None) and \
               (pair_i in list(pair_wise_magmom_comp_data_prev.keys())):
                magmom_data_out = pair_wise_magmom_comp_data_prev[pair_i]
            else:
                # print("Need to run manually")
                # print("pair_i:", pair_i)
                #| - Process pairs
                row_jobs_0 = df_jobs.loc[pair_i[0]]
                row_jobs_1 = df_jobs.loc[pair_i[1]]

                ads_0 = row_jobs_0.ads
                ads_1 = row_jobs_1.ads

                # #############################################
                if set([ads_0, ads_1]) == set(["o", "oh"]):
                    job_id_0 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
                    job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
                elif set([ads_0, ads_1]) == set(["o", "bare"]):
                    job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
                    job_id_1 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
                elif set([ads_0, ads_1]) == set(["oh", "bare"]):
                    job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
                    job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
                else:
                    print("Woops something went wrong here")


                # #############################################
                row_atoms_i = df_atoms_sorted_ind.loc[job_id_0]
                # #############################################
                atoms_0 = row_atoms_i.atoms_sorted_good
                magmoms_sorted_good_0 = row_atoms_i.magmoms_sorted_good
                was_sorted_0 = row_atoms_i.was_sorted
                # #############################################

                # #############################################
                row_atoms_i = df_atoms_sorted_ind.loc[job_id_1]
                # #############################################
                atoms_1 = row_atoms_i.atoms_sorted_good
                magmoms_sorted_good_1 = row_atoms_i.magmoms_sorted_good
                was_sorted_1 = row_atoms_i.was_sorted
                # #############################################


                # #############################################
                magmom_data_out = get_magmom_diff_data(
                    ads_atoms=atoms_1,
                    slab_atoms=atoms_0,
                    ads_magmoms=magmoms_sorted_good_1,
                    slab_magmoms=magmoms_sorted_good_0,
                    )
                #__|

            pair_wise_magmom_comp_data[pair_i] = magmom_data_out

            tot_abs_magmom_diff = magmom_data_out["tot_abs_magmom_diff"]
            norm_abs_magmom_diff = magmom_data_out["norm_abs_magmom_diff"]
            if verbose:
                print("    ", "pair_i: ", pair_i, ": ", np.round(norm_abs_magmom_diff, 3), sep="")

            sum_norm_abs_magmom_diff += norm_abs_magmom_diff

        # #################################################
        data_dict_i["job_ids_tri"] = set(tri_i)
        data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

        #__|

    #__|

    # #####################################################
    df_magmoms_i = pd.DataFrame(data_dict_list)

    # #####################################################
    out_dict["df_magmoms_comp"] = df_magmoms_i
    out_dict["good_triplet_comb"] = good_triplet_comb
    out_dict["pair_wise_magmom_comp_data"] = pair_wise_magmom_comp_data
    out_dict["job_ids"] = job_ids_list
    # #####################################################

    return(out_dict)
    #__|


from methods import read_magmom_comp_data

def save_magmom_comp_data(name_key, data_dict):
    """
    """
    #| - save_magmom_comp_data
    # magmom_data_dict = read_magmom_comp_data()
    # magmom_data_dict[name_key] = data_dict

    name_i = name_key

    name_i_new = []
    for i in name_i:
        if type(i) == float:
            i_new = str(int(i))
            name_i_new.append(i_new)
        else:
            name_i_new.append(i)
    name_str = "__".join(name_i_new)
    file_name_i = name_str + ".pickle"


    import os; import pickle
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/compare_magmoms",
        "out_data/magmom_comparison_data")
    if not os.path.exists(directory): os.makedirs(directory)
    path_i = os.path.join(directory, file_name_i)
    with open(path_i, "wb") as fle:
        pickle.dump(data_dict, fle)
        # pickle.dump(magmom_data_dict, fle)
    #__|


#| - __old__
# def save_magmom_comp_data(name_key, data_dict):
#     """
#     """
#     #| - save_magmom_comp_data
#     magmom_data_dict = read_magmom_comp_data()
#
#     magmom_data_dict[name_key] = data_dict
#
#
#
#     import os; import pickle
#     directory = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "dft_workflow/job_analysis/compare_magmoms",
#         "out_data")
#     if not os.path.exists(directory): os.makedirs(directory)
#     path_i = os.path.join(directory, "magmom_comparison_data.pickle")
#     with open(path_i, "wb") as fle:
#         pickle.dump(magmom_data_dict, fle)
#     #__|


# def save_magmom_comp_data(magmom_data_dict):
#     """
#     """
#     #| - save_magmom_comp_data
#     import os; import pickle
#     directory = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "dft_workflow/job_analysis/compare_magmoms",
#         "out_data")
#     if not os.path.exists(directory): os.makedirs(directory)
#     path_i = os.path.join(directory, "magmom_comparison_data.pickle")
#     with open(path_i, "wb") as fle:
#         pickle.dump(magmom_data_dict, fle)
#     #__|

#__|

# #########################################################
# #########################################################
# #########################################################
# #########################################################


# group = group
# compenv = compenv_i
# slab_id = slab_id_i
# df_jobs_anal = df_jobs_anal

def get_oer_set(
    group=None,
    compenv=None,
    slab_id=None,
    df_jobs_anal=None,
    ):
    """
    """
    #| - get_oer_set
    compenv_i = compenv
    slab_id_i = slab_id


    df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]


    df_index = df_jobs_anal_i.index.to_frame()
    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        (df_index.active_site == "NaN") & \
        [True for i in range(len(df_index))]
        ]

    # df_index_i.shape

    # print(40 * "*")
    # print(df_index_i)
    # print(40 * "*")

    # mess_i = "ISJIdfjisdjij"
    # assert df_index_i.shape[0] == 1, mess_i

    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index
        ]

    # #########################################################
    group_w_o = pd.concat(
        [
            group,
            row_o_i,
            ],
        axis=0)

    return(group_w_o)
    #__|

def analyze_O_in_set(
    data_dict_i,
    group_i,
    df_magmoms,
    magmom_cutoff=None,
    compenv=None,
    slab_id=None,
    active_site=None,
    ):
    """
    """
    #| - analyze_O_in_set
    compenv_i = compenv
    slab_id_i = slab_id
    active_site_i = active_site

    sys_w_not_low_magmoms = False
    sys_w_low_magmoms = False

    # #########################################################
    # Check for *O slabs first
    df_index_i = group_i.index.to_frame()
    df_index_i = df_index_i[df_index_i.ads == "o"]
    # #########################################################
    group_o = group_i.loc[df_index_i.index]
    # #########################################################
    for name_i, row_i in group_o.iterrows():

        # #####################################################
        job_id_i = row_i.job_id_max
        # #####################################################

        # #####################################################
        row_magmoms_i = df_magmoms.loc[job_id_i]
        # #####################################################
        sum_magmoms_i = row_magmoms_i.sum_magmoms
        # #####################################################

        # #####################################################
        row_magmoms_i = df_magmoms.loc[job_id_i]
        # #####################################################
        sum_magmoms_i = row_magmoms_i.sum_magmoms
        sum_abs_magmoms_i = row_magmoms_i.sum_abs_magmoms
        sum_magmoms_pa_i = row_magmoms_i.sum_magmoms_pa
        sum_abs_magmoms_pa = row_magmoms_i.sum_abs_magmoms_pa
        # #####################################################

        if sum_abs_magmoms_pa < magmom_cutoff:
            sys_w_low_magmoms = True
        if sum_abs_magmoms_pa > 0.1:
            sys_w_not_low_magmoms = True


    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    # #####################################################
    data_dict_i["*O_w_low_magmoms"] = sys_w_low_magmoms
    data_dict_i["*O_w_not_low_magmoms"] = sys_w_not_low_magmoms
    # data_dict_i[""] =
    # #####################################################
    # data_dict_list.append(data_dict_i)
    # #####################################################

    return(data_dict_i)
    #__|
