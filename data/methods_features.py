"""
"""

#| - Import Modules
import os
import sys

import copy

import numpy as np
#  import pandas as pd

from methods import (
     get_df_coord,
     )
#__|


def original_slab_is_good(
    nn_info=None,
    # slab_id=None,
    metal_index=None,
    df_coord_orig_slab=None,
    ):
    """
    """
    #| - original_slab_is_good
    # #####################################################
    # slab_id_i = slab_id
    # #####################################################

    nn_info_i = copy.deepcopy(nn_info)

    df_coord_orig_slab = df_coord_orig_slab.set_index("structure_index")

    row_coord_orig_i = df_coord_orig_slab.loc[
        metal_index
        ]

    nn_info_orig_i = row_coord_orig_i.nn_info

    num_neigh_orig = len(nn_info_orig_i)

    if num_neigh_orig != 6:
        orig_slab_good = False
    else:
        orig_slab_good = True

    return(orig_slab_good)
    #__|

def find_missing_O_neigh_with_init_df_coord(
    nn_info=None,
    slab_id=None,
    metal_index=None,
    df_coord_orig_slab=None,
    verbose=False,
    ):
    """
    """
    #| - find_missing_O_neigh_with_init_df_coord
    # #####################################################
    slab_id_i = slab_id
    # nn_info_i = nn_info
    # #####################################################

    nn_info_i = copy.deepcopy(nn_info)


    # # Getting original unrelaxed slab
    # from methods import get_df_slab
    # df_slab = get_df_slab(mode="final")
    # row_slab_i = df_slab.loc[slab_id_i]
    # slab_i = row_slab_i.slab_final
    # num_atoms_i = slab_i.get_global_number_of_atoms()





    # # atoms_sorted_good_i.write("__temp__/testing_oxid_state/final_atoms_sorted.traj")
    # # atoms_sorted_good_i.write("__temp__/testing_oxid_state/final_atoms_sorted.cif")

    # slab_i.write("__temp__/testing_oxid_state/original_slab.traj")
    # slab_i.write("__temp__/testing_oxid_state/original_slab.cif")

    # atoms.write("__temp__/testing_oxid_state/atoms_I_think_this_one_is_the_one.traj")
    # atoms.write("__temp__/testing_oxid_state/atoms_I_think_this_one_is_the_one.cif")







    # #########################################################
    # df_coord_orig_slab = get_df_coord(
    #     slab_id=slab_id_i,
    #     mode="slab",  # 'bulk', 'slab', 'post-dft'
    #     slab=slab_i,
    #     )


    # from methods import get_df_coord
    # init_slab_name_tuple_i = (
    #     compenv_i, slab_id_i, "o",
    #     active_site_i, att_num_i,
    #     )
    # df_coord_orig_slab = get_df_coord(
    #     mode="init-slab",
    #     init_slab_name_tuple=init_slab_name_tuple_i,
    #     )







    # mess_i = "Must be the same here"
    # assert df_coord_orig_slab.shape[0] == num_atoms_i, mess_i

    df_coord_orig_slab = df_coord_orig_slab.set_index("structure_index")

    row_coord_orig_i = df_coord_orig_slab.loc[
        metal_index
        ]

    nn_info_orig_i = row_coord_orig_i.nn_info

    num_neigh_orig = len(nn_info_orig_i)

    if num_neigh_orig != 6:
        if verbose:
            print("The original slab's active site doesn't havea 6 O's about the active Ir")

        num_missing_Os = 6 - num_neigh_orig

        orig_slab_good = False

        nn_info_i = None

    else:
        orig_slab_good = True

        # #########################################################
        site_indices = []
        for nn_i in nn_info_i:
            site_indices.append(nn_i["site_index"])
        site_indices = list(np.sort(site_indices))

        site_indices_orig = []
        for nn_i in nn_info_orig_i:
            site_indices_orig.append(nn_i["site_index"])
        site_indices_orig = list(np.sort(site_indices_orig))





        # #########################################################
        num_of_orign_nn = len(nn_info_orig_i)

        shared_indices = list(set(site_indices_orig) & set(site_indices))

        nonshared_indices = list(set(site_indices_orig).symmetric_difference(site_indices))

        num_missing_Os = len(nonshared_indices)
        if len(nonshared_indices) > 1:
            if verbose:
                mess_i = "For the time being, I'll only tolerate 1 neighbor O atom missing"
                # assert len(nonshared_indices) == 1, mess_i
                print(mess_i)

            nn_info_i = None

        else:
            num_missing_indices = num_of_orign_nn - len(shared_indices)
            mess_i = "Again just checking that they two df_coord share all but one of the O neighbors"
            assert num_missing_indices == 1, mess_i

            # Append the missing indices from the df_coord with the one found from looking at df_coord_orig
            for ind_i in nonshared_indices:
                nn_info_i.append(
                    dict(
                        site_index=ind_i,
                        from_orig_df_coord=True,
                        )
                    )

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["nn_info"] = nn_info_i
    out_dict["num_missing_Os"] = num_missing_Os
    out_dict["orig_slab_good"] = orig_slab_good
    # #####################################################
    return(out_dict)
    #__|
