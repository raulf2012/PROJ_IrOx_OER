"""
"""

#| - Import Modules
import os
import sys

import random

import numpy as np
# from numpy import dot

from ase import Atoms

from numpy import cross, eye
from scipy.linalg import expm, norm
#__|


def get_neighbor_metal_atom(
    df_coord_i=None,
    site_i=None,
    metal_atom_symbol=None,
    ):
    """
    """
    #| - get_neighbor_metal_atom

    row_coord_i = df_coord_i[df_coord_i.structure_index == site_i]
    row_coord_i = row_coord_i.iloc[0]
    # tmp = [print(i) for i in list(row_coord_i.to_dict().keys())]

    element_i = row_coord_i.element
    structure_index_i = row_coord_i.structure_index
    nn_info_i = row_coord_i.nn_info
    neighbor_count_i = row_coord_i.neighbor_count
    num_neighbors_i = row_coord_i.num_neighbors

    # mess_i = "There should only one neighbor"
    # assert num_neighbors_i == 1, mess_i

    ir_neighbors = neighbor_count_i.get("Ir", 0)
    mess_i = "Must only have 1 Ir neighbor for this to work"
    assert ir_neighbors == 1, mess_i

    for ind_j, nn_j in enumerate(nn_info_i):
        site_j = nn_j["site"]

        elem_j = site_j.specie.as_dict().get("element", None)
        if elem_j == metal_atom_symbol:
            correct_ind_j = ind_j

            # #################################################
            # Checking which image the Ir neighbor is in
            # If it's not in the 0,0,0 image then it may take some extra work to get right
            image_sum = np.sum(list(nn_j["image"]))

            if image_sum > 0:
                tmp = 42
                # Looks like the nn_info_i already has the positions traslated to correct cell repitition

                # print("How to do this when the neighbor is on a different repeated image")
                # assert False


    # #########################################################
    nn_j = nn_info_i[correct_ind_j]
    site_j = nn_j["site"]

    coords_j =  site_j.coords

    return(coords_j)
    #__|


def M(axis, theta):
    """
    """
    return expm(cross(eye(3), axis/norm(axis)*theta))

def get_ads_pos_oh(
    atoms=None,
    site_i=None,
    df_coord_i=None,
    metal_atom_symbol="Ir",
    # #################################
    include_colinear=True,
    verbose=False,
    num_side_ads=4,
    ):
    """Return positions of *H atom to be added to Ir-O ligand to create *OH slabs.


    """
    #| - get_ads_pos_oh
    coords_j = get_neighbor_metal_atom(
        df_coord_i=df_coord_i,
        site_i=site_i,
        metal_atom_symbol=metal_atom_symbol,
        )

    o_position = atoms[site_i].position
    ir_position = coords_j

    if verbose:
        print("ir_position:", ir_position)
        print("o_position: ", o_position)


    atoms_oh_list = []

    # #########################################################
    ir_o_vector = o_position - ir_position

    ir_o_unit_vector = ir_o_vector / np.linalg.norm(ir_o_vector)

    # #########################################################
    h_mol = Atoms(
        [
            "H",
            ],
        positions=[
            o_position + 0.978 * ir_o_unit_vector,
            ]
        )


    atoms_oh_0 = atoms + h_mol

    if include_colinear:
        atoms_oh_list.append(atoms_oh_0)

    # #########################################################
    atoms_oh_tmp = atoms

    random_vect = [
        random.choice([-1., +1.]) * random.random(),
        random.choice([-1., +1.]) * random.random(),
        random.choice([-1., +1.]) * random.random(),
        ]

    arb_vector = np.cross(ir_o_unit_vector, random_vect)

    v, axis, theta = (
        ir_o_unit_vector,
        arb_vector,
        (2. / 4.) * np.pi,
        )
    M0 = M(axis, theta)
    rot_v = np.dot(M0, v)

    # #########################################################
    # for i in range(4):
    for i in range(num_side_ads):
        v, axis, theta = (
            rot_v,
            ir_o_unit_vector,
            #  2 * (i + 1) * (1. / 4.) * np.pi,
            2 * (i + 1) * (1. / num_side_ads) * np.pi,
            )
        M0 = M(axis, theta)

        rot_v_new = np.dot(M0, v)

        h_mol = Atoms(
            ["H", ],
            positions=[
                o_position + 0.978 * rot_v_new]
            )

        atoms_oh_i = atoms_oh_tmp + h_mol
        atoms_oh_list.append(atoms_oh_i)


    # atoms_oh_tmp.write("tmp.cif")
    return(atoms_oh_list)
    #__|
