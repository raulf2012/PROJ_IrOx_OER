"""
"""

#| - Import Modules
import os
import sys

import copy
import pickle
import json
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from StructurePrototypeAnalysisPackage.ccf import struc2ccf
from StructurePrototypeAnalysisPackage.ccf import cal_ccf_d

from pymatgen import Lattice, Structure, Molecule
#__|


# slab = atoms_init
# name_tup = name_i
# init_or_final = "init"
# verbose = True
# r_cut_off = r_cut_off
# r_vector = r_vector


def check_ccf_data_present(
    # slab=None,
    name_tup=None,
    init_or_final=None,
    intact=True,
    verbose=None,
    # r_cut_off=None,
    # r_vector=None,
    ):
    """
    """
    #| - check_ccf_data_present
    # #####################################################
    global os
    global pickle






    name_list = []
    for i in name_tup:
        if type(i) == int or type(i) == float:
            name_list.append(str(int(i)))
        elif type(i) == str:
            name_list.append(i)
        else:
            name_list.append(str(i))
    name_i = "__".join(name_list)
    name_i += "___" + init_or_final + "___" + str(intact) + ".pickle"







    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/slab_struct_drift",
        "out_data/ccf_files")


    file_exists = False
    file_path_i = os.path.join(directory, name_i)
    my_file = Path(file_path_i)
    if my_file.is_file():
        file_exists = True

    #     with open(file_path_i, "rb") as fle:
    #         ccf_i = pickle.load(fle)
    #     # #################################################
    #
    # else:
    #     ccf_i = struc2ccf(slab, r_cut_off, r_vector)
    #
    #
    #     # Pickling data ###################################
    #     if not os.path.exists(directory): os.makedirs(directory)
    #     with open(file_path_i, "wb") as fle:
    #         pickle.dump(ccf_i, fle)
    #     # #################################################

    return(file_exists)
    #__|


def get_ccf(
    slab=None,
    name_tup=None,
    init_or_final=None,
    intact=True,
    verbose=None,
    r_cut_off=None,
    r_vector=None,
    ):
    """
    """
    #| - get_ccf_i
    # #####################################################
    global os
    global pickle



    name_list = []
    for i in name_tup:
        if type(i) == int or type(i) == float:
            name_list.append(str(int(i)))
        elif type(i) == str:
            name_list.append(i)
        else:
            name_list.append(str(i))
    name_i = "__".join(name_list)
    name_i += "___" + init_or_final + "___" + str(intact) + ".pickle"



    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/slab_struct_drift",
        "out_data/ccf_files")


    file_path_i = os.path.join(directory, name_i)
    my_file = Path(file_path_i)
    if my_file.is_file():
        if verbose:
            print("File exists already")

        with open(file_path_i, "rb") as fle:
            ccf_i = pickle.load(fle)
        # #################################################

    else:
        ccf_i = struc2ccf(slab, r_cut_off, r_vector)


        # Pickling data ###################################
        if not os.path.exists(directory): os.makedirs(directory)
        with open(file_path_i, "wb") as fle:
            pickle.dump(ccf_i, fle)
        # #################################################

    return(ccf_i)
    #__|

def remove_constrained_atoms(atoms):
    """
    """
    #| - remove_constrained_atoms

    # atoms_new = copy.deepcopy(atoms_final)
    atoms_new = copy.deepcopy(atoms)

    indices_to_remove = atoms_new.constraints[0].index

    mask = []
    for atom in atoms_new:
        if atom.index in indices_to_remove:
            mask.append(True)
        else:
            mask.append(False)
    del atoms_new[mask]

    return(atoms_new)
    #__|

def get_all_ccf_data(
    atoms_init=None,
    atoms_final=None,
    atoms_init_part=None,
    atoms_final_part=None,
    name_i=None,
    r_cut_off=None,
    r_vector=None,
    ):
    """
    """
    #| - get_all_ccf_data

    ccf_init = get_ccf(
        slab=atoms_init,
        name_tup=name_i,
        init_or_final="init",
        intact=True,
        verbose=False,
        r_cut_off=r_cut_off,
        r_vector=r_vector,
        )

    ccf_init_2 = get_ccf(
        slab=atoms_init_part,
        name_tup=name_i,
        init_or_final="init",
        intact=False,
        verbose=False,
        r_cut_off=r_cut_off,
        r_vector=r_vector,
        )



    ccf_final = get_ccf(
        slab=atoms_final,
        name_tup=name_i,
        init_or_final="final",
        intact=True,
        verbose=False,
        r_cut_off=r_cut_off,
        r_vector=r_vector,
        )

    ccf_final_2 = get_ccf(
        slab=atoms_final_part,
        name_tup=name_i,
        init_or_final="final",
        intact=False,
        verbose=False,
        r_cut_off=r_cut_off,
        r_vector=r_vector,
        )

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["ccf_init"] = ccf_init
    out_dict["ccf_init_2"] = ccf_init_2
    out_dict["ccf_final"] = ccf_final
    out_dict["ccf_final_2"] = ccf_final_2
    # #####################################################
    return(out_dict)
    # #####################################################
    #__|


def get_ave_drift(atoms_init, atoms_final):
    """
    """
    #| - get_ave_drift
    # atoms_init = atoms_init_data
    # atoms_final = atoms_sorted_good
    #
    # atoms_init.write("__temp__/00_atoms_init.traj")
    # atoms_final.write("__temp__/00_atoms_final.traj")


    # #####################################################
    # Check that number of atoms are the same
    num_atoms_init = atoms_init.get_global_number_of_atoms()
    num_atoms_final = atoms_final.get_global_number_of_atoms()

    mess_i = "TEMP isdjfisjd"
    assert num_atoms_init == num_atoms_final, mess_i
    num_atoms = num_atoms_init

    # #####################################################
    # Check that constraints are the same
    constr_init = atoms_init.constraints[0]
    constr_init = list(np.sort(constr_init.index))

    constr_final = atoms_final.constraints[0]
    constr_final = list(np.sort(constr_final.index))

    mess_i = "Constraints need to be equal"
    assert constr_init == constr_final, mess_i
    constr = constr_init

    # #####################################################
    # Check lattice cells are equal
    # lattice_cells_equal = np.all(atoms_init.cell == atoms_final.cell)
    lattice_cells_equal = np.allclose(
        atoms_init.cell,
        atoms_final.cell,
        )
    mess_i = "sijfidsifj"
    assert lattice_cells_equal, mess_i
    lattice_cell = atoms_init.cell.tolist()

    # if True:
    #     ind_i = 39
    #     atom_init = atoms_init[ind_i]
    #     atom_final = atoms_final[ind_i]

    # #####################################################
    data_dict_list = []
    # #####################################################
    for atom_init, atom_final in zip(atoms_init, atoms_final):
        # #####################################################
        data_dict_i = dict()
        # #####################################################

        pos_round_init = np.round(atom_init.position, 2)
        pos_round_final = np.round(atom_final.position, 2)

        # #####################################################
        mess_i = "This should always be true"
        assert atom_init.index == atom_final.index, mess_i
        index_i = atom_init.index

        atom_constr_i = index_i in constr


        # #####################################################
        lattice = Lattice(lattice_cell)

        coords = [
            atom_init.position,
            atom_final.position,
            ]

        struct = Structure(
            lattice,
            [atom_init.symbol, atom_final.symbol],
            coords,
            coords_are_cartesian=True,
            )

        dist_i = struct.get_distance(0, 1)

        # #####################################################
        data_dict_i["index"] = index_i
        data_dict_i["distance"] = dist_i
        data_dict_i["distance_round"] = np.round(dist_i, 2)
        data_dict_i["constrained"] = atom_constr_i
        data_dict_i["pos_round_init"] = pos_round_init
        data_dict_i["pos_round_final"] = pos_round_final
        # #####################################################
        data_dict_list.append(data_dict_i)
        # #####################################################

    # #####################################################3###
    df = pd.DataFrame(data_dict_list)
    df = df.set_index("index", drop=False)
    # #####################################################3###

    ave_dist_pa = df.distance.sum() / num_atoms

    return(ave_dist_pa)
    #__|

def get_ave_drift__wrapper(
    name_i=None,
    atoms_init=None,
    atoms_final=None,
    ):
    """
    """
    #| - get_ave_drift__wrapper
    atoms_final_sorted = atoms_final

    name_tup = name_i
    name_list = []
    for i in name_tup:
        if type(i) == int or type(i) == float:
            name_list.append(str(int(i)))
        elif type(i) == str:
            name_list.append(i)
        else:
            name_list.append(str(i))
    name_out_i = "__".join(name_list)
    # name_i += "___" + init_or_final + "___" + str(intact) + ".pickle"
    name_out_i += ".json"


    data_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/slab_struct_drift",
        "out_data/ave_struct_drift",
        name_out_i)

    my_file = Path(data_path)
    if my_file.is_file():
        # ########################################################
        with open(data_path, "r") as fle:
            ave_dist_pa = json.load(fle)
        # ########################################################
    else:
        ave_dist_pa = get_ave_drift(
            atoms_init,
            atoms_final_sorted,
            )

        with open(data_path, "w") as outfile:
            json.dump(ave_dist_pa, outfile, indent=2)

    return(ave_dist_pa)
    #__|


#| - __old__
# def get_D_ij(group, slab_id=None, verbose=False):
#     """
#     """
#     #| - get_D_ij
#     # #####################################################
#     group_i = group
#     slab_id_i = slab_id
#     # #####################################################
#
#     if slab_id_i == None:
#         print("Need to provide a slab_id so that the D_ij can be saved")
#
#     directory = "out_data/D_ij_files"
#     name_i = slab_id_i + ".pickle"
#     file_path_i = os.path.join(directory, name_i)
#
#
#     my_file = Path(file_path_i)
#     if my_file.is_file():
#         #| - Read data from file
#         if verbose:
#             print("Reading from file")
#
#         with open(file_path_i, "rb") as fle:
#             D_ij = pickle.load(fle)
#         # #####################################################
#         #__|
#     else:
#         #| - Create D_ij from scratch
#         if verbose:
#             print("Creating from scratch")
#
#         num_rows = group_i.shape[0]
#
#         D_ij = np.empty((num_rows, num_rows))
#         D_ij[:] = np.nan
#
#         #| - Looping through group rows
#         slab_ids_j = []
#         for i_cnt, (slab_id_j, row_j) in enumerate(group_i.iterrows()):
#             if verbose:
#                 print(40 * "*")
#             for j_cnt, (slab_id_k, row_k) in enumerate(group_i.iterrows()):
#                 if verbose:
#                     print(30 * "-")
#                 ccf_j = get_ccf(slab_id=slab_id_j, verbose=False)
#                 ccf_k = get_ccf(slab_id=slab_id_k, verbose=False)
#
#                 d_ij = cal_ccf_d(ccf_j, ccf_k)
#
#                 D_ij[i_cnt][j_cnt] = d_ij
#         #__|
#
#         # #####################################################
#         D_ij = pd.DataFrame(
#             D_ij,
#             index=group_i.index,
#             columns=group_i.index,
#             )
#
#         #######################################################
#         # Pickling data #######################################
#         if not os.path.exists(directory): os.makedirs(directory)
#         with open(file_path_i, "wb") as fle:
#             pickle.dump(D_ij, fle)
#         #######################################################
#         #__|
#
#     return(D_ij)
#     #__|
#
# def get_identical_slabs(
#     D_ij,
#
#     min_thresh=1e-5,
#     ):
#     """
#     """
#     #| - get_identical_slabs
#
#     # #####################################################
#     # min_thresh = 1e-5
#     # #####################################################
#
#     identical_pairs_list = []
#     for slab_id_i in D_ij.index:
#         for slab_id_j in D_ij.index:
#             if slab_id_i == slab_id_j:
#                 continue
#             if slab_id_i == slab_id_j:
#                 print("Not good if this is printed")
#
#             d_ij = D_ij.loc[slab_id_i, slab_id_j]
#             if d_ij < min_thresh:
#                 # print(slab_id_i, slab_id_j)
#                 identical_pairs_list.append((slab_id_i, slab_id_j))
#
#     # #####################################################
#     # identical_pairs_list_2 = list(np.unique(
#     #     [np.sort(i) for i in identical_pairs_list]
#     #     ))
#
#     lst = identical_pairs_list
#     lst.sort()
#     lst = [list(np.sort(i)) for i in lst]
#
#     identical_pairs_list_2 = list(lst for lst, _ in itertools.groupby(lst))
#
#     return(identical_pairs_list_2)
#     #__|
#__|
