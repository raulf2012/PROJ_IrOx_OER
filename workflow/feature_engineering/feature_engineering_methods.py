"""
"""

#| - Import Modules
# import os
# import sys
# import time; ti = time.time()

import copy

# import numpy as np
import pandas as pd
#
# # #################################################
# from proj_data import metal_atom_symbol
# metal_atom_symbol_i = metal_atom_symbol
#
# from methods import (
#     get_df_jobs_anal,
#     get_df_atoms_sorted_ind,
#     get_df_active_sites,
#     get_df_coord,
#     )
#
# # #################################################
# from local_methods import get_effective_ox_state, process_row
# from local_methods import find_missing_O_neigh_with_init_df_coord
#__|

def get_df_feat_rows(
    df_jobs_anal=None,
    df_atoms_sorted_ind=None,
    df_active_sites=None,
    ):
    """Get rows of df_features, to be used in feature generation.

    The issue here was that in feature generation notebooks, the *O slabs with no active sites needed to be expanded (iterating over active site atoms), which created some complications.
    Here I'll just compile a dataframe that already does the expansion so that in the feature generation notebooks you can simply iterate over this returned object.
    """
    #| - get_df_feat_rows
    df_jobs_anal_i = df_jobs_anal

    #| - Pareparing df_jobs_anal object
    df_jobs_anal_done = df_jobs_anal[df_jobs_anal.job_completely_done == True]

    df_jobs_anal_i =  df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]


    # Selecting *O rows to process
    # dx = pd.IndexSlice
    # df_jobs_anal_i = df_jobs_anal_i.loc[idx[:, :, "o", :, :], :]


    # Selecting *O and *OH systems to process
    df_index = df_jobs_anal_i.index.to_frame()
    df_index_i = df_index[
        df_index.ads.isin(["o", "oh", ])
        # df_index.ads.isin(["oh", ])
        ]
    df_jobs_anal_i = df_jobs_anal_i.loc[
        df_index_i.index
        ]






    # #####################################################
    indices_to_run = []
    # #####################################################
    for name_i, row_i in df_jobs_anal_i.iterrows():
        # #################################################
        run_row = True
        if name_i in df_atoms_sorted_ind.index:
            row_atoms_i = df_atoms_sorted_ind.loc[name_i]
            # #############################################
            failed_to_sort_i = row_atoms_i.failed_to_sort
            # #############################################

            if failed_to_sort_i:
                run_row = False
        else:
            run_row = False

        if run_row:
            indices_to_run.append(name_i)


    # #####################################################
    df_jobs_anal_i = df_jobs_anal_i.loc[
        indices_to_run
        ]
    #__|


    # #####################################################
    data_dict_list = []
    # #####################################################
    # iterator = tqdm(df_jobs_anal_i.index, desc="1st loop")
    # for i_cnt, name_i in enumerate(iterator):
    for i_cnt, name_i in enumerate(df_jobs_anal_i.index):
        # #################################################
        data_dict_i = dict()
        # #################################################
        row_i = df_jobs_anal_i.loc[name_i]
        # #################################################
        slab_id_i = name_i[1]
        active_site_i = name_i[3]
        # #################################################
        job_id_max_i = row_i.job_id_max
        # #################################################
        name_dict_i = dict(zip(
            list(df_jobs_anal_i.index.names), list(name_i)))
        # #################################################


        # #################################################
        row_sites_i = df_active_sites.loc[slab_id_i]
        # #################################################
        active_sites_unique_i = row_sites_i.active_sites_unique
        # #################################################

        data_dict_i["job_id_max"] = job_id_max_i
        data_dict_i["active_site_orig"] = active_site_i
        if active_site_i != "NaN":
            # #############################################
            data_dict_j = dict()
            # #############################################
            data_dict_j["from_oh"] = True
            data_dict_j["active_site"] = active_site_i
            # #############################################
            data_dict_j.update(name_dict_i)
            data_dict_j.update(data_dict_i)
            # #############################################
            data_dict_list.append(data_dict_j)
            # #############################################


        else:
            for active_site_j in active_sites_unique_i:
                name_dict_i_cpy = copy.deepcopy(name_dict_i)
                name_dict_i_cpy.pop("active_site")

                # #########################################
                data_dict_j = dict()
                # #########################################
                data_dict_j["from_oh"] = False
                data_dict_j["active_site"] = active_site_j
                # #########################################
                data_dict_j.update(name_dict_i_cpy)
                data_dict_j.update(data_dict_i)
                # #########################################
                data_dict_list.append(data_dict_j)
                # #########################################


    # #####################################################
    df_feat_rows = pd.DataFrame(data_dict_list)
    # #####################################################

    return(df_feat_rows)
    #__|
