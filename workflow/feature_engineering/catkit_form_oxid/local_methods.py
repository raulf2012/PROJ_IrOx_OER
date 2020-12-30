"""
"""

#| - Import Modules
import os
import sys

import numpy as np

# #########################################################
from catkit.gen.utils.connectivity import (
    get_cutoff_neighbors,
    get_voronoi_neighbors,
    )

# #########################################################
from methods import get_df_coord

#| - __old__
# from sys import argv
# import pylab as p
# import ase
# from ase.io import read
# from ase.visualize import view
# # from mendeleev import element
#__|

#__|

from methods_features import original_slab_is_good
from methods_features import find_missing_O_neigh_with_init_df_coord

def get_connectivity(atoms):
    """
    """
    #| - get_connectivity
    matrix = get_cutoff_neighbors(atoms, scale_cov_radii=1.3)
    #matrix = get_voronoi_neighbors(atoms, cutoff=3)
    return matrix
    #__|

def set_formal_oxidation_state(
    atoms,
    charge_O=-2,
    charge_H=1):
    """
    """
    #| - set_formal_oxidation_state
    O_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'O'])
    H_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'H'])
    M_indices = np.array([i for i, a in enumerate(atoms)
                          if not a.symbol in ['O', 'H']])
    non_O_indices = np.array([i for i, a in enumerate(atoms)
                              if not a.symbol in ['O']])

    # Connectivity matrix from CatKit Voronoi
    con_matrix = get_connectivity(atoms)
    oxi_states = np.ones([len(atoms)])
    oxi_states[O_indices] = charge_O

    if H_indices:  # First correct O charge due to H
        oxi_states[H_indices] = charge_H
        for H_i in H_indices:
            H_O_connectivity = con_matrix[H_i][O_indices]
            norm = np.sum(H_O_connectivity)
            O_indices_H = O_indices[np.where(H_O_connectivity)[0]]
            oxi_states[O_indices_H] += charge_H / norm

    for metal_i in M_indices:  # Substract O connectivity
        M_O_connectivity = con_matrix[metal_i][O_indices]
        norm = np.sum(con_matrix[O_indices][:, M_indices], axis=-1)
        oxi_states[metal_i] = sum(
            M_O_connectivity * -oxi_states[O_indices] / norm)

    atoms.set_initial_charges(np.round(oxi_states, 4))

    return atoms
    #__|

def get_catkit_form_oxid_state_wrap(
    atoms=None,
    name=None,
    active_site=None,
    ):
    """
    """
    #| - get_catkit_form_oxid_state_wrap
    # #########################################################
    name_i = name
    active_site_j = active_site
    # #########################################################
    form_oxid_i = None
    # #########################################################

    # #########################################################
    # from local_methods import get_df_coord_wrap
    from methods import get_df_coord_wrap

    df_coord_i = get_df_coord_wrap(
        name=name_i,
        active_site=active_site_j,
        )

    # #########################################################
    from methods import get_metal_index_of_active_site
    metal_index_dict = get_metal_index_of_active_site(
        df_coord=df_coord_i,
        active_site=active_site_j,
        )
    all_good_i = metal_index_dict["all_good"]
    metal_index_i = metal_index_dict["metal_index"]

    atoms_out_i = None
    neigh_dict_i = None
    if all_good_i:
        # #########################################################
        atoms_out_i = set_formal_oxidation_state(
            atoms,
            charge_O=-2,
            charge_H=1,
            )


        neigh_dict_i = get_oxy_coord_dict__catkit(
            atoms,
            metal_index=metal_index_i,
            charge_O=-2,
            charge_H=1,
            )


        # print("atoms_out_i:", atoms_out_i)
        # print("")
        # print("metal_index_i:", metal_index_i)

        # atoms_out_i[active_site_j]
        atom_metal_i = atoms_out_i[metal_index_i]

        form_oxid_i = atom_metal_i.charge

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["form_oxid"] = form_oxid_i
    out_dict["atoms_out"] = atoms_out_i
    out_dict["neigh_dict"] = neigh_dict_i
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|

# name = name_orig_i
# active_site = active_site_i
# df_coord_i = df_coord_i
# metal_atom_symbol = metal_atom_symbol_i
# active_site_original = name_orig_i[3]

def get_effective_ox_state__test(
    name=None,
    active_site=None,
    df_coord_i=None,
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
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
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

    mess_i = "For now only deal with active sites that have 1 Ir neighbor"
    # print("num_Ir_neigh:", num_Ir_neigh)
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

    skip_this_sys = False
    if not six_O_neigh or not six_neigh:
        skip_this_sys = True









    from methods import get_df_coord

    init_slab_name_tuple_i = (
        compenv_i, slab_id_i, ads_i,
        active_site_original, att_num_i,
        )
    # print("init_slab_name_tuple_i:", init_slab_name_tuple_i)
    df_coord_orig_slab = get_df_coord(
        mode="init-slab",
        init_slab_name_tuple=init_slab_name_tuple_i,
        )

    orig_slab_good_i = original_slab_is_good(
        nn_info=nn_info_i,
        # slab_id=None,
        metal_index=metal_index,
        df_coord_orig_slab=df_coord_orig_slab,
        )




    num_missing_Os = 0
    used_unrelaxed_df_coord = False
    if not six_O_neigh:
        used_unrelaxed_df_coord = True

        from methods import get_df_coord
        init_slab_name_tuple_i = (
            compenv_i, slab_id_i, ads_i,
            # active_site_i, att_num_i,
            active_site_original, att_num_i,
            )
        df_coord_orig_slab = get_df_coord(
            mode="init-slab",
            init_slab_name_tuple=init_slab_name_tuple_i,
            )

        out_dict_0 = find_missing_O_neigh_with_init_df_coord(
            nn_info=nn_info_i,
            slab_id=slab_id_i,
            metal_index=metal_index,
            df_coord_orig_slab=df_coord_orig_slab,
            )
        new_nn_info_i = out_dict_0["nn_info"]
        num_missing_Os = out_dict_0["num_missing_Os"]
        orig_slab_good_i = out_dict_0["orig_slab_good"]

        nn_info_i = new_nn_info_i

        if new_nn_info_i is not None:
            skip_this_sys = False
        else:
            skip_this_sys = True

    # #####################################################
    effective_ox_state = None
    # if six_O_neigh and six_neigh:
    if not skip_this_sys:
        #| - Iterating through 6 oxygens
        second_shell_coord_list = []
        tmp_list = []

        # print("nn_info_i:", nn_info_i)

        neigh_dict = dict()
        for nn_j in nn_info_i:

            from_orig_df_coord = nn_j.get("from_orig_df_coord", False)
            if from_orig_df_coord:
                Ir_neigh_adjustment = 1

                active_metal_in_nn_list = False
                for i in df_coord_i.loc[site_index].nn_info:
                    if i["site_index"] == metal_index:
                        active_metal_in_nn_list = True

                if active_metal_in_nn_list:
                    Ir_neigh_adjustment = 0

            else:
                Ir_neigh_adjustment = 0


            site_index = nn_j["site_index"]

            row_coord_j = df_coord_i.loc[site_index]

            neighbor_count_j = row_coord_j.neighbor_count

            num_Ir_neigh_j = neighbor_count_j.get("Ir", 0)

            # print(site_index, "|", num_Ir_neigh_j)

            neigh_dict[site_index] = num_Ir_neigh_j

            # print("num_Ir_neigh_j:", site_index, num_Ir_neigh_j)
            num_Ir_neigh_j += Ir_neigh_adjustment

            # print("num_Ir_neigh_j:", num_Ir_neigh_j)

            second_shell_coord_list.append(num_Ir_neigh_j)

            tmp_list.append(2 / num_Ir_neigh_j)

        # second_shell_coord_list
        effective_ox_state = np.sum(tmp_list)
        #__|


    neigh_keys = list(neigh_dict.keys())

    for i in np.sort(neigh_keys):
        print(
            i,
            "|",
            neigh_dict[i]
            )

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["effective_ox_state"] = effective_ox_state
    out_dict["used_unrelaxed_df_coord"] = used_unrelaxed_df_coord
    out_dict["num_missing_Os"] = num_missing_Os
    out_dict["orig_slab_good"] = orig_slab_good_i
    out_dict["neigh_dict"] = neigh_dict
    # #####################################################
    return(out_dict)
    #__|

# atoms
# charge_O = -2
# charge_H = 1

def get_oxy_coord_dict__catkit(
    atoms,
    metal_index=None,
    charge_O=-2,
    charge_H=1):
    """
    """
    #| - set_formal_oxidation_state
    O_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'O'])
    H_indices = np.array([i for i, a in enumerate(atoms)
                          if a.symbol == 'H'])
    M_indices = np.array([i for i, a in enumerate(atoms)
                          if not a.symbol in ['O', 'H']])
    non_O_indices = np.array([i for i, a in enumerate(atoms)
                              if not a.symbol in ['O']])

    # Connectivity matrix from CatKit Voronoi
    con_matrix = get_connectivity(atoms)
    oxi_states = np.ones([len(atoms)])
    oxi_states[O_indices] = charge_O

    if H_indices:  # First correct O charge due to H
        oxi_states[H_indices] = charge_H
        for H_i in H_indices:
            H_O_connectivity = con_matrix[H_i][O_indices]
            norm = np.sum(H_O_connectivity)
            O_indices_H = O_indices[np.where(H_O_connectivity)[0]]
            oxi_states[O_indices_H] += charge_H / norm

    for metal_i in M_indices:  # Substract O connectivity
        M_O_connectivity = con_matrix[metal_i][O_indices]
        norm = np.sum(con_matrix[O_indices][:, M_indices], axis=-1)
        oxi_states[metal_i] = sum(
            M_O_connectivity * -oxi_states[O_indices] / norm)

    atoms.set_initial_charges(np.round(oxi_states, 4))


    # active_site = 71

    connected_oxy_indices = []
    for ind_i, conn_i in enumerate(con_matrix[metal_index]):
        if conn_i == 1:
            connected_oxy_indices.append(ind_i)

    connected_oxy_indices_2 = []
    for ind_i in connected_oxy_indices:
        if atoms[ind_i].symbol == "O":
            connected_oxy_indices_2.append(ind_i)

    oxy_indices = connected_oxy_indices_2

    neigh_dict = dict()
    for oxy_i in np.sort(oxy_indices):
        num_neigh_i = np.sum(
            con_matrix[oxy_i]
            )
        # print(oxy_i, "|", num_neigh_i)
        neigh_dict[oxy_i] = num_neigh_i

    return(neigh_dict)

    # return atoms
    #__|


# def get_df_coord_wrap(
#     name=None,
#     active_site=None,
#     ):
#     """
#     """
#     #| - get_df_coord_wrap
#     # #####################################################
#     name_i = name
#     active_site_j = active_site
#     # #####################################################
#     compenv_i = name[0]
#     slab_id_i = name[1]
#     ads_i = name[2]
#     active_site_i = name[3]
#     att_num_i = name[4]
#     # #####################################################
#
#
#     # # #############################################
#     # if read_orig_O_df_coord:
#     #     name_i = (
#     #         compenv_i, slab_id_i, ads_i,
#     #         "NaN", att_num_i)
#     # else:
#     #     name_i = (
#     #         compenv_i, slab_id_i, ads_i,
#     #         active_site_i, att_num_i)
#
#     if ads_i == "o":
#         porous_adjustment_i = True
#     elif ads_i == "oh":
#         # porous_adjustment_i = True
#         porous_adjustment_i = False
#
#
#     df_coord_i = get_df_coord(
#         mode="post-dft",
#         post_dft_name_tuple=name_i,
#         porous_adjustment=porous_adjustment_i,
#         )
#
#     df_coord_i = df_coord_i.set_index("structure_index", drop=False)
#     # row_coord_i = df_coord_i.loc[active_site_i]
#     row_coord_i = df_coord_i.loc[active_site_j]
#
#     nn_info_i = row_coord_i.nn_info
#     neighbor_count_i = row_coord_i.neighbor_count
#
#     num_Ir_neigh = neighbor_count_i.get("Ir", 0)
#
#     if num_Ir_neigh != 1 and ads_i == "oh":
#         porous_adjustment_i = True
#         df_coord_i = get_df_coord(
#             mode="post-dft",
#             post_dft_name_tuple=name_i,
#             porous_adjustment=porous_adjustment_i,
#             )
#
#     return(df_coord_i)
#     #__|
