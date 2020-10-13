"""
"""

#| - Import Modules
#  import os
#  print(os.getcwd())
#  import sys

import numpy as np
#  import pandas as pd

# #########################################################
#  from proj_data import metal_atom_symbol

#  from methods import (
#      get_df_jobs_anal,
#      get_df_atoms_sorted_ind,
#      get_df_active_sites,
#      get_df_coord,
#      )

#__|


def get_effective_ox_state(
    df_coord_i=None,
    active_site_j=None,
    metal_atom_symbol="Ir",
    ):
    """
    """
    #| - get_effective_ox_state
    df_coord_i = df_coord_i.set_index("structure_index", drop=False)

    # #########################################################
    #| - Processing central Ir atom nn_info
    # row_coord_i = df_coord_i.loc[21]
    row_coord_i = df_coord_i.loc[active_site_j]

    nn_info_i = row_coord_i.nn_info

    neighbor_count_i = row_coord_i.neighbor_count
    num_Ir_neigh = neighbor_count_i.get("Ir", 0)

    mess_i = "For now only deal with active sites that have 1 Ir neighbor"
    assert num_Ir_neigh == 1, mess_i

    for j_cnt, nn_j in enumerate(nn_info_i):
        site_j = nn_j["site"]
        elem_j = site_j.as_dict()["species"][0]["element"]

        if elem_j == metal_atom_symbol:
            corr_j_cnt = j_cnt

    site_j = nn_info_i[corr_j_cnt]
    metal_index = site_j["site_index"]
    #__|

    # #########################################################
    row_coord_i = df_coord_i.loc[metal_index]

    neighbor_count_i = row_coord_i["neighbor_count"]
    nn_info_i =  row_coord_i.nn_info
    num_neighbors_i = row_coord_i.num_neighbors

    num_O_neigh = neighbor_count_i.get("O", 0)

    six_O_neigh = num_O_neigh == 6
    mess_i = "There should be exactly 6 oxygens about the Ir atom"
    # assert six_O_neigh, mess_i

    six_neigh = num_neighbors_i == 6
    mess_i = "Only 6 neighbors total is allowed, all oxygens"
    # assert six_neigh, mess_i


    # #####################################################
    effective_ox_state = None
    if six_O_neigh and six_neigh:
        #| - Iterating through 6 oxygens
        second_shell_coord_list = []
        tmp_list = []
        for nn_j in nn_info_i:
            tmp = 42

            site_index = nn_j["site_index"]

            row_coord_j = df_coord_i.loc[site_index]

            neighbor_count_j = row_coord_j.neighbor_count

            num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)

            # print("num_Ir_neigh_j:", num_Ir_neigh_j)

            second_shell_coord_list.append(num_Ir_neigh_j)

            tmp_list.append(2 / num_Ir_neigh_j)

        # second_shell_coord_list
        effective_ox_state = np.sum(tmp_list)
        #__|

    return(effective_ox_state)
    #__|
