"""
"""

#| - Import Modules
import os
import sys

import copy
from pathlib import Path

import json

import numpy as np
import pandas as pd

# #############################################################################
from ase import io

from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
    MultiWeightsChemenvStrategy)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.io.ase import AseAtomsAdaptor

from catkit.gen.surface import SlabGenerator

# #############################################################################
from methods import (
    get_df_dft, symmetrize_atoms,
    get_structure_coord_df, remove_atoms)
from proj_data import metal_atom_symbol
#__|

from methods import get_slab_thickness

def analyse_local_coord_env(
    atoms=None,
    ):
    """
    """
    #| - analyse_local_coord_env
    out_dict = dict()

    Ir_indices = []
    for i, s in enumerate(atoms.get_chemical_symbols()):
        if s == "Ir":
            Ir_indices.append(i)

    struct = AseAtomsAdaptor.get_structure(atoms)


    lgf = LocalGeometryFinder()
    lgf.setup_structure(structure=struct)

    se = lgf.compute_structure_environments(
        maximum_distance_factor=1.41,
        only_cations=False)


    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()

    lse = LightStructureEnvironments.from_structure_environments(
        strategy=strategy, structure_environments=se)

    isite = 0
    cor_env = []
    coor_env_dict = dict()
    for isite in Ir_indices:
        c_env = lse.coordination_environments[isite]
        coor_env_dict[isite] = c_env
        cor_env += [c_env[0]['ce_symbol']]

    out_dict["coor_env_dict"] = coor_env_dict

    return(out_dict)
    #__|

def check_if_sys_processed(
    bulk_id_i=None,
    facet_str=None,
    df_slab_old=None,
    ):
    """
    """
    #| - check_if_sys_processed
    sys_processed = False

    num_rows_tot = df_slab_old.shape[0]
    if num_rows_tot > 0:
        df = df_slab_old
        df = df[
            (df["bulk_id"] == bulk_id_i) &
            (df["facet"] == facet_str) &
            [True for i in range(len(df))]
            ]

        num_rows = df.shape[0]
        if num_rows > 0:
            # print("There is a row that already exists")

            sys_processed = True
            if num_rows > 1:
                print("There is more than 1 row for this bulk+facet combination, what to do?")

            row_i = df.iloc[0]

    return(sys_processed)
    #__|

def remove_nonsaturated_surface_metal_atoms(
    atoms=None,
    dz=None,
    ):
    """
    """
    #| - remove_nonsaturated_surface_metal_atoms

    # #################################################
    # #################################################
    z_positions = atoms.positions[:,2]

    z_max = np.max(z_positions)
    z_min = np.min(z_positions)

    # #################################################
    # #################################################
    df_coord_slab_i = get_structure_coord_df(atoms)

    # #################################################
    metal_atoms_to_remove = []
    for atom in atoms:
        if atom.symbol == metal_atom_symbol:
            z_pos_i = atom.position[2]
            if z_pos_i >= z_max - dz or z_pos_i <= z_min + dz:
                row_coord = df_coord_slab_i[
                    df_coord_slab_i.structure_index == atom.index].iloc[0]
                num_o_neighbors = row_coord.neighbor_count.get("O", 0)

                if num_o_neighbors < 6:
                    metal_atoms_to_remove.append(atom.index)

    slab_new = remove_atoms(atoms=atoms, atoms_to_remove=metal_atoms_to_remove)

    return(slab_new)
    # __|

def remove_noncoord_oxygens(
    atoms=None,
    ):
    """ """
    #| - remove_noncoord_oxygens
    df_coord_slab_i = get_structure_coord_df(atoms)

    # #########################################################
    df_i = df_coord_slab_i[df_coord_slab_i.element == "O"]
    df_i = df_i[df_i.num_neighbors == 0]

    o_atoms_to_remove = df_i.structure_index.tolist()

    # #########################################################
    o_atoms_to_remove_1 = []
    df_j = df_coord_slab_i[df_coord_slab_i.element == "O"]
    for j_cnt, row_j in  df_j.iterrows():
        neighbor_count = row_j.neighbor_count

        if neighbor_count.get("Ir", 0) == 0:
            if neighbor_count.get("O", 0) == 1:
                o_atoms_to_remove_1.append(row_j.structure_index)


    o_atoms_to_remove = list(set(o_atoms_to_remove + o_atoms_to_remove_1))

    slab_new = remove_atoms(atoms, atoms_to_remove=o_atoms_to_remove)

    return(slab_new)
    # __|

def create_slab_from_bulk(atoms=None, facet=None, layers=20):
    """Create slab from bulk atoms object and facet.

    Args:
        atoms (ASE atoms object): ASE atoms object of bulk structure.
        facet (str): Facet cut
        layers (int): Number of layers
    """
    #| - create_slab_from_bulk
    SG = SlabGenerator(
        atoms, facet, layers, vacuum=15,
        # atoms, facet, 20, vacuum=15,
        fixed=None, layer_type='ang',
        attach_graph=True,
        standardize_bulk=True,
        primitive=True, tol=1e-03)

    # print("TEMP", "SlabGenerator:", SlabGenerator)

    slab_i = SG.get_slab()
    slab_i.set_pbc([True, True, True])


    # #############################################
    # Write slab to file ##########################
    directory = "out_data"
    if not os.path.exists(directory):
        os.makedirs(directory)

    path_i = os.path.join("out_data", "temp.cif")
    slab_i.write(path_i)

    # Rereading the structure file to get it back into ase format
    slab_i = io.read(path_i)
    # data_dict_i["slab_0"] = slab_i

    return(slab_i)
    #__|

def remove_highest_metal_atoms(
    atoms=None,
    num_atoms_to_remove=None,
    metal_atomic_number=77,
    ):
    """ """
    #| - remove_highest_metal_atom
    slab_m = atoms[atoms.numbers == metal_atomic_number]

    positions_cpy = copy.deepcopy(slab_m.positions)
    positions_cpy_sorted = positions_cpy[positions_cpy[:,2].argsort()]

    indices_to_remove = []
    for coord_i in positions_cpy_sorted[-2:]:
        for i_cnt, atom in enumerate(atoms):
            if all(atom.position == coord_i):
                indices_to_remove.append(i_cnt)

    slab_new = remove_atoms(
        atoms=atoms,
        atoms_to_remove=indices_to_remove,
        )

    return(slab_new)
    #__|

def calc_surface_area(atoms=None):
    """ """
    #| - calc_surface_area
    cell = atoms.cell

    cross_prod = np.cross(cell[0], cell[1])
    area = np.linalg.norm(cross_prod)

    return(area)
    #__|

def remove_all_atoms_above_cutoff(
    atoms=None,
    cutoff_thickness=17,
    ):
    """
    """
    #| - remove_all_atoms_above_cutoff
    positions = atoms.positions

    z_positions = positions[:,2]

    z_max = np.max(z_positions)
    z_min = np.min(z_positions)

    atoms_new = atoms[z_positions < z_min + cutoff_thickness]

    return(atoms_new)
    #__|

def create_final_slab_master(
    atoms=None,
    ):
    """Master method to create final IrOx slab.
    """
    #| - create_final_slab_master
    TEMP = True
    slab_0 = atoms

    ###########################################################
    slab_thickness_i = get_slab_thickness(atoms=slab_0)
    # print("slab_thickness_i:", slab_thickness_i)
    slab_thickness_out = slab_thickness_i


    cutoff_thickness_i = 14
    break_loop = False
    while not break_loop:
        #| - Getting slab pre-ready
        slab = remove_all_atoms_above_cutoff(
            atoms=slab_0,
            cutoff_thickness=cutoff_thickness_i)

        ###########################################################
        slab = remove_nonsaturated_surface_metal_atoms(atoms=slab, dz=4)
        slab = remove_noncoord_oxygens(atoms=slab)

        slab_thickness_i = get_slab_thickness(atoms=slab)
        # print("slab_thickness_i:", slab_thickness_i)

        if slab_thickness_i < 15:
            cutoff_thickness_i = cutoff_thickness_i + 1.
        else:
            break_loop = True
        #__|


    #| - Main loop, chipping off surface atoms
    # print("SIDJFISDIFJISDJFIJSDIJF")
    # ###########################################################
    # i_cnt = 2
    # while slab_thickness_i > 15:
    #     print("slab_thickness_i:", slab_thickness_i)
    #
    #     i_cnt += 1
    #     # print(i_cnt)
    #
    #     # #####################################################
    #     # Figuring out how many surface atoms to remove at one time
    #     # Taken from R-IrO2 (100), which has 8 surface Ir atoms and a surface area of 58 A^2
    #     surf_area_per_surface_metal = 58 / 8
    #     surface_area_i = calc_surface_area(atoms=slab)
    #     ideal_num_surface_atoms = surface_area_i / surf_area_per_surface_metal
    #     num_atoms_to_remove = ideal_num_surface_atoms / 3
    #     num_atoms_to_remove = int(np.round(num_atoms_to_remove))
    #     # #####################################################
    #
    #     slab_new_0 = remove_highest_metal_atoms(
    #         atoms=slab,
    #         num_atoms_to_remove=num_atoms_to_remove,
    #         metal_atomic_number=77)
    #     if TEMP:
    #         slab_new_0.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_0" + ".cif")
    #
    #     slab_new_1 = remove_nonsaturated_surface_metal_atoms(
    #         atoms=slab_new_0,
    #         dz=4)
    #     if TEMP:
    #         slab_new_1.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_1" + ".cif")
    #
    #     slab_new_2 = remove_noncoord_oxygens(atoms=slab_new_1)
    #     if TEMP:
    #         slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_2" + ".cif")
    #
    #
    #     slab_thickness_i = get_slab_thickness(atoms=slab_new_2)
    #     # print("slab_thickness_i:", slab_thickness_i)
    #
    #     if TEMP:
    #         slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + ".cif")
    #
    #     slab = slab_new_2
    #__|

    return(slab)
    #__|

def create_save_dataframe(data_dict_list=None, df_slab_old=None):
    """
    """
    #| - create_save_dataframe
    # #####################################################
    # Create dataframe
    df_slab = pd.DataFrame(data_dict_list)
    num_new_rows = len(data_dict_list)

    if num_new_rows > 0:
        df_slab = df_slab.set_index("slab_id")
    elif num_new_rows == 0:
        print("There aren't any new rows to process")
        assert False

    # #####################################################
    df_slab = pd.concat([
        df_slab_old,
        df_slab,
        ])

    # Pickling data #######################################
    import os; import pickle
    directory = "out_data"
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "df_slab.pickle"), "wb") as fle:
        pickle.dump(df_slab, fle)
    # #####################################################

    df_slab_old = df_slab

    return(df_slab_old)
    #__|

def constrain_slab(
    atoms=None,
    ):
    """Constrain lower portion of slab geometry.

    Has a little bit of built in logic which should provide better slabs.
    """
    #| - constrain_slab
    slab = atoms

    positions = slab.positions

    z_pos = positions[:,2]

    z_max = np.max(z_pos)
    z_min = np.min(z_pos)

    # ang_of_slab_to_constrain = (2 / 4) * (z_max - z_min)
    ang_of_slab_to_constrain = (z_max - z_min) - 6


    # #########################################################
    indices_to_constrain = []
    for atom in slab:
        if atom.symbol == metal_atom_symbol:
            if atom.position[2] < (z_min + ang_of_slab_to_constrain):
                indices_to_constrain.append(atom.index)

    for atom in slab:
        if atom.position[2] < (z_min + ang_of_slab_to_constrain - 2):
            indices_to_constrain.append(atom.index)

    df_coord_slab_i = get_structure_coord_df(slab)

    # #########################################################
    other_atoms_to_constrain = []
    for ind_i in indices_to_constrain:
        row_i = df_coord_slab_i[df_coord_slab_i.structure_index == ind_i]
        row_i = row_i.iloc[0]

        nn_info = row_i.nn_info

        for nn_i in nn_info:
            ind_j = nn_i["site_index"]
            other_atoms_to_constrain.append(ind_j)

    # print(len(indices_to_constrain))

    indices_to_constrain.extend(other_atoms_to_constrain)

    # print(len(indices_to_constrain))

    # #########################################################
    constrain_bool_mask = []
    for atom in slab:
        if atom.index in indices_to_constrain:
            constrain_bool_mask.append(True)
        else:
            constrain_bool_mask.append(False)

    # #########################################################
    slab_cpy = copy.deepcopy(slab)

    from ase.constraints import FixAtoms
    c = FixAtoms(mask=constrain_bool_mask)
    slab_cpy.set_constraint(c)

    # slab_cpy.constraints

    return(slab_cpy)
    #__|

def read_data_json():
    """
    """
    #| - read_data_json
    path_i = os.path.join(
        "out_data", "data.json")
    my_file = Path(path_i)
    if my_file.is_file():
        data_path = os.path.join(
            "out_data/data.json")
        with open(data_path, "r") as fle:
            data = json.load(fle)
    else:
        data = dict()

    return(data)
    #__|

def resize_z_slab(atoms=None, vacuum=15):
    """
    """
    #| - resize_z_slab
    z_pos = atoms.positions[:,2]
    z_max = np.max(z_pos)
    z_min = np.min(z_pos)

    new_z_height = z_max - z_min + vacuum

    cell = atoms.cell

    cell_cpy = cell.copy()
    cell_cpy = cell_cpy.tolist()

    cell_cpy[2][2] = new_z_height

    atoms_cpy = copy.deepcopy(atoms)
    atoms_cpy.set_cell(cell_cpy)

    return(atoms_cpy)
    #__|

def repeat_xy(atoms=None, min_len_x=6, min_len_y=6):
    """
    """
    #| - repeat_xy
    print("min_len_x:", min_len_x, "min_len_y:", min_len_y)

    cell = atoms.cell.array

    x_mag = np.linalg.norm(cell[0])
    y_mag = np.linalg.norm(cell[1])

    import math
    mult_x = 1
    if x_mag < min_len_x:
        mult_x = math.ceil(min_len_x / x_mag)
        #  mult_x = round(min_len_x / x_mag)
        mult_x = int(mult_x)
        # print(mult_x)
    mult_y = 1
    if y_mag < min_len_y:
        mult_y = math.ceil(min_len_y / y_mag)
        #  mult_y = round(min_len_y / y_mag)
        mult_y = int(mult_y)

    repeat_list = (mult_x, mult_y, 1)
    atoms_repeated = atoms.repeat(repeat_list)

    # Check if the atoms were repeated or not
    if mult_x > 1 or mult_y > 1:
        is_repeated = True
    else:
        is_repeated = False

    # Construct final out dict
    out_dict = dict(
        atoms_repeated=atoms_repeated,
        is_repeated=is_repeated,
        repeat_list=repeat_list,
        )
    return(out_dict)
    #__|

