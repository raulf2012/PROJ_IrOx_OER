# | - Import Modules
import os
import sys

import pickle
from pathlib import Path

import numpy as np

# #########################################################
from methods import get_structure_coord_df
from methods_features import (
    get_num_metal_neigh_manually,
    find_missing_O_neigh_with_init_df_coord,
    )
# __|


def get_df_coord_local(
    name=None,
    ads=None,
    atoms=None,
    append_str=None,
    ):
    """
    """
    # | - get_df_coord_local

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/seoin_irox_data/featurize_data",
        "out_data/df_coord")

    if append_str is not None:
        file_name_i = name + "__" + ads + "__" + append_str + ".pickle"
    else:
        file_name_i = name + "__" + ads + ".pickle"

    path_i = os.path.join(directory, file_name_i)

    my_file = Path(path_i)
    if my_file.is_file():

        with open(path_i, "rb") as fle:
            df_coord_i = pickle.load(fle)

    else:
        # Create df_coord

        df_coord_i = get_structure_coord_df(
            atoms,
            porous_adjustment=True,
            )

        directory = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/seoin_irox_data/featurize_data",
            "out_data/df_coord")
        # file_name_i = name + "__o" + ".pickle"

        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, file_name_i), "wb") as fle:
            pickle.dump(df_coord_i, fle)

    return(df_coord_i)
    # __|


# name=None
# active_site=None
# df_coord_i=None
# metal_atom_symbol="Ir"
# active_site_original=None

def get_effective_ox_state(
    name=None,
    active_site=None,
    df_coord_i=None,
    df_coord_init_i=None,
    metal_atom_symbol="Ir",
    active_site_original=None,
    ):
    """
    """
    #| - get_effective_ox_state
    # #########################################################
    name_i = name
    active_site_j = active_site
    # #########################################################

    # compenv_i = name_i[0]
    # slab_id_i = name_i[1]
    # ads_i = name_i[2]
    # active_site_i = name_i[3]
    # att_num_i = name_i[4]

    # #########################################################

    # #########################################################
    #| - Processing central Ir atom nn_info
    df_coord_i = df_coord_i.set_index("structure_index", drop=False)


    import os
    import sys
    import pickle



    # row_coord_i = df_coord_i.loc[21]
    row_coord_i = df_coord_i.loc[active_site_j]

    nn_info_i = row_coord_i.nn_info

    neighbor_count_i = row_coord_i.neighbor_count
    num_Ir_neigh = neighbor_count_i.get("Ir", 0)


    if num_Ir_neigh != 1:
        mess_i = "For now only deal with active sites that have 1 Ir neighbor"
        print(mess_i)
        # assert num_Ir_neigh == 1, mess_i

        # #################################################
        out_dict = dict()
        # #################################################
        out_dict["effective_ox_state"] = None
        out_dict["used_unrelaxed_df_coord"] = None
        out_dict["num_missing_Os"] = None
        out_dict["orig_slab_good"] = None
        out_dict["found_active_Ir"] = False
        # print("s089huyjf8sdyuf8sdf80u9")
        # print(out_dict)
        # #################################################
        return(out_dict)
        # #################################################



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

    six_neigh = num_neighbors_i == 6
    mess_i = "Only 6 neighbors total is allowed, all oxygens"

    skip_this_sys = False
    if not six_O_neigh or not six_neigh:
        # print("Skip this sys")
        skip_this_sys = True


    #| - If missing some O's then go back to slab before DFT and get missing O
    from methods import get_df_coord

    # init_slab_name_tuple_i = (
    #     compenv_i, slab_id_i, ads_i,
    #     active_site_original, att_num_i,
    #     )
    # df_coord_orig_slab = get_df_coord(
    #     mode="init-slab",
    #     init_slab_name_tuple=init_slab_name_tuple_i,
    #     )
    # orig_slab_good_i = original_slab_is_good(
    #     nn_info=nn_info_i,
    #     metal_index=metal_index,
    #     df_coord_orig_slab=df_coord_orig_slab,
    #     )


    num_missing_Os = 0
    used_unrelaxed_df_coord = False
    if not six_O_neigh:
        used_unrelaxed_df_coord = True

        # from methods import get_df_coord
        # init_slab_name_tuple_i = (
        #     compenv_i, slab_id_i, ads_i,
        #     active_site_original, att_num_i,
        #     # active_site_i, att_num_i,
        #     )
        # df_coord_orig_slab = get_df_coord(
        #     mode="init-slab",
        #     init_slab_name_tuple=init_slab_name_tuple_i,
        #     )

        out_dict_0 = find_missing_O_neigh_with_init_df_coord(
            nn_info=nn_info_i,
            # slab_id=slab_id_i,
            metal_index=metal_index,
            # df_coord_orig_slab=df_coord_orig_slab,
            df_coord_orig_slab=df_coord_init_i,
            )
        new_nn_info_i = out_dict_0["nn_info"]
        num_missing_Os = out_dict_0["num_missing_Os"]
        orig_slab_good_i = out_dict_0["orig_slab_good"]

        nn_info_i = new_nn_info_i

        if new_nn_info_i is not None:
            skip_this_sys = False
        else:
            skip_this_sys = True
    #__|


    # #####################################################
    effective_ox_state = None
    if not skip_this_sys:
        #| - Iterating through 6 oxygens
        orig_df_coord_was_used = False

        second_shell_coord_list = []
        tmp_list = []
        for nn_j in nn_info_i:

            site_index = nn_j["site_index"]

            #| - Fixing bond number of missing *O
            # If Ir was missing *O bond, then neigh count for that O will be undercounted
            # Although sometimes even through the Ir is missing the *O, the *O is not missing the Ir
            # Happened for this system: ('slac', 'ralutiwa_59', 'o', 31.0, 1)
            from_orig_df_coord = nn_j.get("from_orig_df_coord", False)
            active_metal_in_nn_list = False

            if from_orig_df_coord:
                orig_df_coord_was_used = True

                Ir_neigh_adjustment = 1
                for i in df_coord_i.loc[site_index].nn_info:
                    if i["site_index"] == metal_index:
                        active_metal_in_nn_list = True

                if active_metal_in_nn_list:
                    Ir_neigh_adjustment = 0

            else:
                Ir_neigh_adjustment = 0
            #__|


            oxy_ind = site_index
            num_metal_neigh_2 = get_num_metal_neigh_manually(
                oxy_ind, df_coord=df_coord_i, metal_atom_symbol=metal_atom_symbol)

            # #################################################
            #| - Checking manually the discrepency
            if False:
                row_coord_j = df_coord_i.loc[site_index]

                neighbor_count_j = row_coord_j.neighbor_count

                # TODO | IMPORTANT
                # We should check manually the previous structure for Ir neighbors
                # Also we should check if the 'lost' Ir-O bonds are good or are completely bad
                num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)
                num_Ir_neigh_j += Ir_neigh_adjustment

                if num_Ir_neigh_j != num_metal_neigh_2:
                    if Ir_neigh_adjustment == 0:

                        # print("")
                        print("name:", name)
                        print(
                            "oxy_ind:", oxy_ind, "|",
                            "Original num Ir Neigh: ", num_Ir_neigh_j, "|",
                            "New num Ir Neigh: ", num_metal_neigh_2, "|",
                            "Ir adjustment:", Ir_neigh_adjustment, "|",
                            "orig_df_coord_was_used:", orig_df_coord_was_used, "|",
                            )

                # I shouldn't have to do this, but we know that there is at least 1 Ir-O bond (to the active Ir) so we'll just manually set it here
                if num_Ir_neigh_j == 0:
                    num_Ir_neigh_j = 1
            #__|

            num_metal_neigh_2 += Ir_neigh_adjustment

            # second_shell_coord_list.append(num_Ir_neigh_j)
            # tmp_list.append(2 / num_Ir_neigh_j)

            tmp_list.append(2 / num_metal_neigh_2)
            second_shell_coord_list.append(num_metal_neigh_2)


        # second_shell_coord_list
        effective_ox_state = np.sum(tmp_list)

        effective_ox_state = np.round(effective_ox_state, decimals=6)

        #__|

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["effective_ox_state"] = effective_ox_state
    out_dict["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord
    out_dict["num_missing_Os"] = num_missing_Os
    # out_dict["orig_slab_good"] = orig_slab_good_i
    out_dict["found_active_Ir"] = True
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|
