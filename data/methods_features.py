"""
"""

#| - Import Modules
import os
import sys

import copy

import numpy as np
#  import pandas as pd

from scipy.spatial import ConvexHull

import math

from methods import unit_vector, angle_between
from methods import (
     get_df_coord,
     )
#__|

from pymatgen import Lattice, Structure, Molecule

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
    # slab_id=None,
    metal_index=None,
    df_coord_orig_slab=None,
    verbose=False,
    ):
    """
    """
    #| - find_missing_O_neigh_with_init_df_coord
    # #####################################################
    # slab_id_i = slab_id
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

def get_num_metal_neigh_manually(
    oxy_ind,
    df_coord=None,
    metal_atom_symbol="Ir",
    ):
    """Checking how many Ir are bound to particular *O the hard way
    """
    #| - get_num_metal_neigh_manually
    df_coord_i = df_coord

    num_metal_neigh = 0
    for ind_i, row_i in df_coord_i[df_coord_i["element"] == metal_atom_symbol].iterrows():
        for j in row_i.nn_info:
            if j["site_index"] == oxy_ind:
                num_metal_neigh += 1

    # print(num_metal_neigh)
    return(num_metal_neigh)
    #__|


# df_coord_i = df_coord_i
# active_site_j = active_site
# atoms = atoms
# verbose = verbose

def get_octa_geom(
    df_coord_i=None,
    active_site_j=None,
    atoms=None,
    octahedra_atoms=None,
    verbose=False,
    ):
    """
    """
    #| - get_octa_geom

    octahedra_atoms_i = octahedra_atoms

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


    # | - __old__
    # if len(nn_info_i) != 1:
    # if len(nn_info_i) != 1:
    # active_o_has_1_neigh = True

        # print("Need to return NaN")
        # active_o_has_1_neigh = False
    #
    # else:
    # if active_o_has_1_neigh:
    # __|

    if process_system:


        if octahedra_atoms_i is not None:

            # | - Use octahedra_atoms_i to get info
            lattice_cell = atoms.cell.tolist()

            # #####################################################
            lattice = Lattice(lattice_cell)


            atom_Ir = atoms[
                ir_nn["site_index"]
                ]

            active_o_metal_dist = None

            ir_o_distances = []
            for O_index_i in octahedra_atoms_i:
                # if atoms[59].symbol == "O":

                atom_O_i = atoms[O_index_i]

                if atom_O_i.symbol == "O":
                    coords = [
                        atom_Ir.position,
                        atom_O_i.position,
                        ]

                    struct = Structure(
                        lattice,
                        [atom_Ir.symbol, atom_O_i.symbol],
                        coords,
                        coords_are_cartesian=True,
                        )

                    dist_i = struct.get_distance(0, 1)
                    # print(dist_i)
                    ir_o_distances.append(dist_i)

                    if O_index_i == int(active_site_j):
                        # print(333 * "TMEP | ")
                        active_o_metal_dist = dist_i




            ir_o_mean = np.mean(ir_o_distances)
            ir_o_std = np.std(ir_o_distances)

            # __|

        else:


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

            # out_dict["active_o_metal_dist"] = dist_i

            active_o_metal_dist = dist_i
            #__|

            #| - Getting stats on all 6 Ir-O bonds
            row_coord_j = df_coord_i[df_coord_i.structure_index == ir_site_index]
            row_coord_j = row_coord_j.iloc[0]

            nn_info_ir = row_coord_j.nn_info

            ir_o_distances = []
            if len(nn_info_ir) != 6:
                tmp = 42
            else:
                for nn_j in nn_info_ir:
                    diff_list = nn_j["site"].coords - ir_coord
                    dist_i = (np.sum([i ** 2 for i in diff_list])) ** (1 / 2)
                    ir_o_distances.append(dist_i)


                # ir_o_mean = np.mean(ir_o_distances)
                # ir_o_std = np.std(ir_o_distances)

                # out_dict["ir_o_mean"] = ir_o_mean
                # out_dict["ir_o_std"] = ir_o_std
            #__|






        ir_o_mean = np.mean(ir_o_distances)
        ir_o_std = np.std(ir_o_distances)

        # #################################################
        out_dict["ir_o_mean"] = ir_o_mean
        out_dict["ir_o_std"] = ir_o_std
        out_dict["active_o_metal_dist"] = active_o_metal_dist
        # #################################################


    return(out_dict)
    #__|


# df_coord_i=df_coord_i
# active_site_j=active_site
# octahedra_atoms=octahedra_atoms
# metal_active_site=metal_active_site
# atoms=atoms
# verbose=verbose

def get_octa_vol(
    df_coord_i=None,
    active_site_j=None,
    octahedra_atoms=None,
    metal_active_site=None,
    atoms=None,
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

    process_sys = all_good_i



    from pymatgen.io.ase import AseAtomsAdaptor
    structure = AseAtomsAdaptor.get_structure(atoms)




    volume = None
    if process_sys and octahedra_atoms is not None:




        metal_active_site = int(metal_active_site)

        if metal_active_site in octahedra_atoms:
            octahedra_atoms.remove(metal_active_site)




        # for atom_index_i in octahedra_atoms:
        #     if atoms[atom_index_i].symbol != "Ir":
        #         pos_i = atoms[atom_index_i].position
        #         coord_list.append(pos_i)





        coord_list = []
        coord_list_orig = []

        # octahedra_atoms = [87, ]
        for atom_index_i in octahedra_atoms:
            atom_index_i = int(atom_index_i)

            if atoms[atom_index_i].symbol != "Ir":

                pos_i = atoms[atom_index_i].position
                # print(11 * "-")
                # print(atom_index_i)
                # print(pos_i)

                coord_list_orig.append(pos_i)

                dist_image_i = structure[metal_active_site].distance_and_image(
                    structure[atom_index_i]
                    )

                dist_i = dist_image_i[0]
                image_i = dist_image_i[1]

                # image_trans = image_i * atoms.cell.tolist()
                # pos_new_i = pos_i
                # for i in image_trans:
                #     pos_new_i = pos_new_i + i
                # print(pos_new_i)

                pos_new_i = pos_i
                for i_cnt, image_j in enumerate(image_i):
                    cell_vect_j = atoms.cell.tolist()[i_cnt]
                    cell_vect_j = np.array(cell_vect_j)
                    trans_j = image_j * cell_vect_j
                    pos_new_i = pos_new_i + trans_j

                coord_list.append(pos_new_i)


        # | - Write atoms of octahedra to file
        if False:
            from ase import Atoms

            symbol_list = ["O" for i in coord_list]
            symbol_list.append("Ir")

            coord_list.append(
                structure[metal_active_site].coords
                )

            atoms_tmp = Atoms(
                symbols=symbol_list,
                positions=coord_list,
                cell=atoms.cell,
                )
            atoms_tmp.write("__temp__/atoms_tmp.traj")


            symbol_list = ["O" for i in coord_list_orig]
            symbol_list.append("Ir")

            coord_list_orig.append(
                structure[metal_active_site].coords
                )

            atoms_tmp = Atoms(
                symbols=symbol_list,
                positions=coord_list_orig,
                cell=atoms.cell,
                )
            atoms_tmp.write("__temp__/atoms_tmp_orig.traj")
        # __|




        volume = ConvexHull(coord_list).volume
        volume




    # if process_sys:
    #     # nn_info_i = nn_info_non_H[0]
    #     # metal_index_i = nn_info_i["site_index"]

    #     row_coord_i = df_coord_i[df_coord_i.structure_index == metal_index_i]
    #     row_coord_i = row_coord_i.iloc[0]

    #     nn_info = row_coord_i.nn_info

    #     if len(nn_info) != 6:
    #         process_sys = False
    #         if verbose:
    #             print("This active site Ir doesn't have 6 nearest neighbors")
    #         # return(None)


    # #| - Compute volume of octahedra
    # volume = None
    # if process_sys:
    #     coord_list = []
    #     for nn_j in nn_info:
    #         site_j = nn_j["site"]
    #         coords_j = site_j.coords
    #         coord_list.append(coords_j)
    #     volume = ConvexHull(coord_list).volume
    # else:
    #     tmp = 42
    # # __|

    return(volume)
    # __|


def get_angle_between_surf_normal_and_O_Ir(
    atoms=None,
    df_coord=None,
    active_site=None,
    ):
    """
    """
    # | - get_angle_between_surf_normal_and_O_Ir
    atoms_i = atoms
    df_coord_i = df_coord
    active_site_i = active_site


    row_coord_i = df_coord_i.loc[active_site_i]

    nn_Ir = None
    for nn_j in row_coord_i.nn_info:
        if nn_j["site"].specie.name == "Ir":
            nn_Ir = nn_j

    if nn_Ir == None:
        return(None)

    # assert nn_Ir != None, "IJDFIDJSIFDS"

    Ir_coord = nn_Ir["site"].coords


    active_O_atom = atoms_i[int(active_site_i)]
    O_coord = active_O_atom.position

    Ir_O_vec = O_coord - Ir_coord

    # Ir_O_vec = [0, 1, -1]

    angle_rad = angle_between(
        unit_vector(Ir_O_vec),
        [0, 0, 1],
        )
    angle_deg = math.degrees(angle_rad)


    return(angle_deg)
    # __|
