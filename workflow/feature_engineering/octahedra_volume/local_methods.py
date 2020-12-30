"""
"""

#| - Import Modules
import numpy as np
from scipy.spatial import ConvexHull


# #########################################################
from methods import (
    get_df_coord,
    get_df_coord_wrap,
    )
#__|



def process_row_2(
    name=None,
    active_site=None,
    active_site_original=None,
    atoms=None,
    # read_orig_O_df_coord=False,
    verbose=False,
    ):
    """
    """
    #| - process_row
    # #####################################################
    name_i = name
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    df_coord_i = get_df_coord_wrap(name=name_i, active_site=active_site)

    # if df_coord_i is None:
    #     print(10 * "df_coord_i = None \n")
    #     print(name_i)

    # #################################################
    from local_methods import get_octa_vol
    vol_i = get_octa_vol(
        df_coord_i=df_coord_i,
        active_site_j=active_site,
        verbose=verbose)
    # #################################################
    from local_methods import get_octa_geom
    octa_geom_dict = get_octa_geom(
        df_coord_i=df_coord_i,
        active_site_j=active_site,
        atoms=atoms,
        verbose=verbose)
    # #################################################


    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i.update(octa_geom_dict)
    # #################################################
    data_dict_i["active_site"] = active_site
    data_dict_i["octa_vol"] = vol_i
    # #################################################
    return(data_dict_i)
    # #################################################

    #__|



    #| - __old__
    #
    # # if verbose:
    # #     print("\t", "active_site_i:", active_site_i)
    #
    # # #############################################
    # if read_orig_O_df_coord:
    #     name_i = (
    #         compenv_i, slab_id_i, ads_i,
    #         "NaN", att_num_i)
    # else:
    #     # name_i = (
    #     #     compenv_i, slab_id_i, ads_i,
    #     #     active_site_i, att_num_i)
    #     name_i = (
    #         compenv_i, slab_id_i, ads_i,
    #         active_site, att_num_i)
    #
    # # print("name_i:" ,name_i)
    #
    #
    #
    # df_coord_i = get_df_coord(
    #     mode="post-dft",
    #     post_dft_name_tuple=name_i)
    #
    # # #################################################
    # # name_i = (
    # #     compenv_i, slab_id_i, ads_i,
    # #     active_site_i, att_num_i)
    # # df_coord_i = get_df_coord(
    # #     mode="post-dft",
    # #     post_dft_name_tuple=name_i)

    #__|

#| - __old__

# def process_row(
#     name=None,
#     active_site=None,
#     active_site_original=None,
#     atoms=None,
#     # metal_atom_symbol="Ir",
#     read_orig_O_df_coord=False,
#     verbose=False,
#     ):
#     """
#     """
#     #| - process_row
# #     # ('nersc', 'kalisule_45', 'oh', 67.0, 3)
# #
# #     # #####################################################
# #     name_i = name
# #     # #####################################################
# #     compenv_i = name_i[0]
# #     slab_id_i = name_i[1]
# #     ads_i = name_i[2]
# #     active_site_i = name_i[3]
# #     att_num_i = name_i[4]
# #     # #####################################################
# #     # compenv_i = compenv
# #     # slab_id_i = slab_id
# #     # ads_i = ads
# #     # active_site_i = active_site
# #     # att_num_i = att_num
# #     # #####################################################
# #
# #     data_dict_j = dict()
# #
# #     # if verbose:
# #     #     print("\t", "active_site_i:", active_site_i)
# #
# #     # #############################################
# #     if read_orig_O_df_coord:
# #         name_i = (
# #             compenv_i, slab_id_i, ads_i,
# #             "NaN", att_num_i)
# #     else:
# #         # name_i = (
# #         #     compenv_i, slab_id_i, ads_i,
# #         #     active_site_i, att_num_i)
# #         name_i = (
# #             compenv_i, slab_id_i, ads_i,
# #             active_site, att_num_i)
# #
# #     # print("name_i:" ,name_i)
# #
# #
# #
# #     df_coord_i = get_df_coord(
# #         mode="post-dft",
# #         post_dft_name_tuple=name_i)
# #
# #     # #################################################
# #     # name_i = (
# #     #     compenv_i, slab_id_i, ads_i,
# #     #     active_site_i, att_num_i)
# #     # df_coord_i = get_df_coord(
# #     #     mode="post-dft",
# #     #     post_dft_name_tuple=name_i)
# #
# #
# #
# #     if df_coord_i is None:
# #         print(10 * "df_coord_i = None \n")
# #         print(name_i)
# #
# #
# #
# #
# #     # #################################################
# #     from local_methods import get_octa_vol
# #     vol_i = get_octa_vol(
# #         df_coord_i=df_coord_i,
# #         # active_site_j=active_site_i,
# #         active_site_j=active_site,
# #         verbose=verbose)
# #     # #################################################
# #     from local_methods import get_octa_geom
# #     octa_geom_dict = get_octa_geom(
# #         df_coord_i=df_coord_i,
# #         # active_site_j=active_site_i,
# #         active_site_j=active_site,
# #         atoms=atoms,
# #         verbose=verbose)
# #     # #################################################
# #
# #     # vol_i = out_dict[0]
# #     # octa_geom_dict = out_dict[1]
# #
# #     octa_geom_dict["octa_vol"] = vol_i
# #
# #     # #################################################
# #     out_dict = octa_geom_dict
# #     # #################################################
# #     # out_dict["active_site"] = active_site_i
# #     out_dict["active_site"] = active_site
# #     # #################################################
# #     return(out_dict)
# #     # #################################################
# #     #__|

#__|








# df_coord_i = df_coord_i
# active_site_j = active_site_i
# atoms = atoms
# verbose = verbose

def get_octa_geom(
    df_coord_i=None,
    active_site_j=None,
    atoms=None,
    verbose=False,
    ):
    """
    """
    #| - get_octa_geom
    out_dict = dict(
        active_o_metal_dist=None,
        ir_o_mean=None,
        ir_o_std=None,
        )


    process_system = True



    row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
    row_coord_i = row_coord_i.iloc[0]

    nn_info_i = row_coord_i.nn_info

    # ir_nn = nn_info_i[0]

    found_active_Ir = False
    for nn_j in nn_info_i:
        if nn_j["site"].specie.symbol == "Ir":
            ir_nn = nn_j
            found_active_Ir = True
    mess_i = "Didn't find the Ir atom that the active O is bound to"
    # assert found_active_Ir, mess_i

    if not found_active_Ir:
        process_system = False



    num_non_H_neigh = 0
    for nn_j in nn_info_i:
        site_j = nn_j["site"]
        if site_j.specie.name != "H":
            num_non_H_neigh += 1

    if num_non_H_neigh != 1:
        process_system = False


    # if len(nn_info_i) != 1:
    # if len(nn_info_i) != 1:
    # active_o_has_1_neigh = True


        # print("Need to return NaN")
        # active_o_has_1_neigh = False
    #
    # else:

    # if active_o_has_1_neigh:
    if process_system:

        # ir_coord = ir_nn["site"].coords
        ir_site_index = ir_nn["site_index"]
        ir_coord = atoms[ir_site_index].position

        #| - Calculating the distance between the active O atom and Ir
        atom_active_o = atoms[
            int(active_site_j)
            ]

        ir_coord_tmp = ir_nn["site"].coords

        # diff_list = atom_active_o.position - ir_coord
        diff_list = atom_active_o.position - ir_coord_tmp
        dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)

        # ir_nn["site"].coords

        out_dict["active_o_metal_dist"] = dist_i
        #__|

        #| - Getting stats on all 6 Ir-O bonds

        row_coord_j = df_coord_i[df_coord_i.structure_index == ir_site_index]
        row_coord_j = row_coord_j.iloc[0]

        nn_info_ir = row_coord_j.nn_info

        if len(nn_info_ir) != 6:
            tmp = 42
            # print("Pass NaN")
        else:
            ir_o_distances = []
            # print(20 * "TEMP | ")
            # for nn_j in nn_info_ir[0:1]:
            for nn_j in nn_info_ir:
                diff_list = nn_j["site"].coords - ir_coord
                dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)
                ir_o_distances.append(dist_i)

            ir_o_mean = np.mean(ir_o_distances)
            ir_o_std = np.std(ir_o_distances)
            out_dict["ir_o_mean"] = ir_o_mean
            out_dict["ir_o_std"] = ir_o_std
        #__|


    return(out_dict)
    #__|


# df_coord_i = df_coord_i
# active_site_j = active_site_i
# verbose = True

def get_octa_vol(
    df_coord_i=None,
    active_site_j=None,
    verbose=False,
    ):
    """
    """
    #| - get_octa_vol

    from methods import get_metal_index_of_active_site
    metal_index_dict = get_metal_index_of_active_site(
        df_coord=df_coord_i,
        active_site=active_site_j,
        verbose=verbose,
        )
    all_good_i = metal_index_dict["all_good"]
    metal_index_i = metal_index_dict["metal_index"]

    #| - out of sight
    # # #########################################################
    # row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
    # row_coord_i = row_coord_i.iloc[0]
    # # #########################################################
    # nn_info_i = row_coord_i.nn_info
    # # #########################################################
    #
    # num_non_H_neigh = 0
    # nn_info_non_H = []
    # for nn_j in nn_info_i:
    #     site_j = nn_j["site"]
    #     if site_j.specie.name != "H":
    #         num_non_H_neigh += 1
    #         nn_info_non_H.append(nn_j)
    #
    # # Check if the active *O has exactly 1 non-hydrogen neigh
    # process_sys = True
    # if num_non_H_neigh != 1:
    #     process_sys = False
    #     if verbose:
    #         print("The active oxygen has more than 1 NN, this is ambigious")
    #
    # # If good to go, then check that active Ir has 6 *O neigh
    # if process_sys:
    #     nn_info_i = nn_info_non_H[0]
    #     metal_index_i = nn_info_i["site_index"]
    #
    #     row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
    #     row_coord_i = row_coord_i.iloc[0]
    #
    #     nn_info = row_coord_i.nn_info
    #
    #     if len(nn_info) != 6:
    #         process_sys = False
    #         if verbose:
    #             print("This active site Ir doesn't have 6 nearest neighbors")
    #         # return(None)
    #__|

    process_sys = all_good_i

    if process_sys:
        # nn_info_i = nn_info_non_H[0]
        # metal_index_i = nn_info_i["site_index"]

        row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
        row_coord_i = row_coord_i.iloc[0]

        nn_info = row_coord_i.nn_info

        if len(nn_info) != 6:
            process_sys = False
            if verbose:
                print("This active site Ir doesn't have 6 nearest neighbors")
            # return(None)


    #| - Compute volume of octahedra
    volume = None
    if process_sys:
        coord_list = []
        for nn_j in nn_info:
            site_j = nn_j["site"]
            coords_j = site_j.coords
            coord_list.append(coords_j)
        volume = ConvexHull(coord_list).volume
    #__|

    return(volume)
    #__|



#| - __old__
# def get_octa_vol(
#     df_coord_i=None,
#     active_site_j=None,
#     ):
#     """
#     """
#     #| - get_octa_vol
#     # #########################################################
#     row_coord_i = df_coord_i[df_coord_i.structure_index == active_site_j]
#     row_coord_i = row_coord_i.iloc[0]
#     # #########################################################
#     nn_info_i = row_coord_i.nn_info
#     # #########################################################
#
#     mess_i = "Must  only have one row  here"
#     assert len(nn_info_i) == 1, mess_i
#
#     nn_info_i = nn_info_i[0]
#
#     metal_index_i = nn_info_i["site_index"]
#
#
#     # #########################################################
#     # #########################################################
#     # #########################################################
#
#
#     row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
#     row_coord_i = row_coord_i.iloc[0]
#
#     nn_info = row_coord_i.nn_info
#
#     coord_list = []
#     for nn_j in nn_info:
#         site_j = nn_j["site"]
#         coords_j = site_j.coords
#         coord_list.append(coords_j)
#
#     # coord_list
#
#     import numpy as np
#     from scipy.spatial import ConvexHull
#
#     # points = np.array([[....]])
#     volume = ConvexHull(coord_list).volume
#
#     return(volume)
#     #__|
#__|
