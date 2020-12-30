
#| - Import Modules
import os
import sys

import pickle

# # #########################################################
from methods import (
    get_df_slab,
    get_structure_coord_df,
    )
#__|


def df_matches_slab(slab, df_coord, ):
    """
    """
    #| - df_matches_slab
    num_atoms = slab.get_global_number_of_atoms()
    num_rows = df_coord.shape[0]

    same_num = True
    if num_atoms != num_rows:
        same_num = False

    # #####################################################
    all_atoms_indices_present = True
    elems_match_for_all = True
    # #####################################################
    for atom_i in slab:
        index_i = atom_i.index
        sym_i = atom_i.symbol

        all_atoms_indices_present = True
        if index_i not in df_coord.structure_index:
            all_atoms_indices_present = False

        else:

            # #####################################################
            row_coord_i = df_coord[df_coord.structure_index == index_i]
            row_coord_i = row_coord_i.iloc[0]
            # #####################################################
            elem_i = row_coord_i.element
            # #####################################################

            if sym_i != elem_i:
                elems_match_for_all = False

    # print("elems_match_for_all:", elems_match_for_all)
    # print("all_atoms_indices_present:", all_atoms_indices_present)

    df_matches_slab = False
    if all_atoms_indices_present and elems_match_for_all and same_num:
        df_matches_slab = True

    # print("all_atoms_indices_present:", all_atoms_indices_present)
    # print("elems_match_for_all:", elems_match_for_all)
    # print("same_num:", same_num)

    return(df_matches_slab)
    #__|

def process_sys(
    slab_id=None,
    slab=None,
    path_pre=None,
    mode="new",  # 'new' or 'old'
    ):
    """
    """
    #| - process_sys
    if mode == "old":
        str_append = ""
    elif mode == "new":
        str_append = "_after_rep"


    path_i = os.path.join(
        path_pre, slab_id + str_append + ".pickle")
    with open(path_i, "rb") as fle:
        df_coord_i = pickle.load(fle)

    df_matches_slab_i = df_matches_slab(slab, df_coord_i)


    df_coord_redone = None
    if not df_matches_slab_i:

        print("df_coord_" + mode + " is inconsistent with slab")

        df_coord_redone = get_structure_coord_df(slab)
        # print("path_i:", path_i)
        with open(path_i, "wb") as fle:
            # print("Writing file")
            pickle.dump(df_coord_redone, fle)

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["df_matches_slab"] = df_matches_slab_i
    out_dict["df_coord_redone"] = df_coord_redone
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|
