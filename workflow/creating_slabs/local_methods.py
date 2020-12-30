"""
"""

#| - Import Modules
import os
import sys

import copy
import pickle
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
from methods import read_data_json

# import os
# from local_methods import (
#     create_slab_from_bulk,
#     create_final_slab_master,
#     constrain_slab,
#     )


def create_slab(
    atoms=None,
    facet=None,
    slab_thickness=15,
    i_cnt=0,
    ):
    """
    """
    #| - create_slab
    atoms_i = atoms
    facet_i = facet

    slab_0 = create_slab_from_bulk(atoms=atoms_i, facet=facet_i)
    slab_1 = create_final_slab_master(atoms=slab_0, slab_thickness=slab_thickness)
    slab_2 = constrain_slab(atoms=slab_1)



    # df_coord_i = get_structure_coord_df(
    #     slab_2,
    #     porous_adjustment=True,
    #     )
    #
    # from methods import remove_protruding_bottom_Os
    # dz = 0.75
    # angle_thresh = 30
    # slab_3 = remove_protruding_bottom_Os(
    #     atoms=slab_2,
    #     dz=dz,
    #     angle_thresh=angle_thresh,
    #     df_coord=df_coord_i,
    #     )


    slab_final = slab_2


    return(slab_final)
    #__|

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
    # directory = "out_data"
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs",
        "out_data")
    # assert False, "Fix os.makedirs"
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
    mode="all_atoms",  # `all_atoms` or `only_metal`
    ):
    """
    """
    #| - remove_all_atoms_above_cutoff
    # #####################################################
    metal_symbol = "Ir"
    # #####################################################

    positions = atoms.positions

    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    z_min = np.min(z_positions)

    if mode == "all_atoms":
        atoms_new = atoms[z_positions < z_min + cutoff_thickness]
    elif mode == "only_metal":
        #| - only_metal

        # print(
        #     "cutoff_thickness:",
        #     cutoff_thickness
        #     )
        # positions = atoms.positions
        #
        # z_positions = positions[:,2]
        #
        # z_max = np.max(z_positions)
        # z_min = np.min(z_positions)

        # atoms_new = atoms[z_positions < z_min + cutoff_thickness]


        atoms_new = copy.deepcopy(atoms)

        indices_to_remove = []
        for atom in atoms_new:
            z_i = atom.position[2]
            if atom.symbol == metal_symbol:
                if z_i > z_min + cutoff_thickness:
                    indices_to_remove.append(atom.index)

            #     else:
            #         tmp = 42
            #         # indices_to_remove.append(False)
            #         # indices_to_remove.append(atom.index)
            # else:
            #     tmp = 42
            #     # indices_to_remove.append(atom.index)

        # print("indices_to_remove:", indices_to_remove)

        mask = []
        for atom in atoms_new:
            if atom.index in indices_to_remove:
                mask.append(True)
            else:
                mask.append(False)

        del atoms_new[mask]
        #__|

    return(atoms_new)
    #__|




# atoms = slab_0
# slab_thickness = 15

def create_final_slab_master(
    atoms=None,
    slab_thickness=15,
    ):
    """Master method to create final IrOx slab.
    """
    #| - create_final_slab_master
    TEMP = True
    slab_0 = atoms

    ###########################################################
    slab_thickness_i = get_slab_thickness(atoms=slab_0)
    slab_thickness_out = slab_thickness_i


    ###########################################################
    cutoff_thickness_i = slab_thickness - 3
    assert cutoff_thickness_i > 0, "IKDSFIISDJFIJSDIJFISDJIFJuhuuyhuuj"
    break_loop = False
    i_cnt = 0
    ###########################################################
    while not break_loop:
        #| - Getting slab pre-ready
        i_cnt += 1

        slab = remove_all_atoms_above_cutoff(
            atoms=atoms,
            cutoff_thickness=cutoff_thickness_i,
            mode="only_metal",
            )
        num_atoms = slab.get_global_number_of_atoms()

        ###########################################################
        slab = remove_nonsaturated_surface_metal_atoms(atoms=slab, dz=4)
        num_atoms = slab.get_global_number_of_atoms()
        slab_temp = slab

        slab = remove_noncoord_oxygens(atoms=slab)
        num_atoms = slab.get_global_number_of_atoms()

        if num_atoms == 0:
            print("There are 0 atoms in the slab now, probably thickness cutoff was too small")
            print("There are 0 atoms in the slab now, probably thickness cutoff was too small")
            print("There are 0 atoms in the slab now, probably thickness cutoff was too small")
            print("There are 0 atoms in the slab now, probably thickness cutoff was too small")
            print("There are 0 atoms in the slab now, probably thickness cutoff was too small")


        slab_thickness_i = get_slab_thickness(atoms=slab)

        if slab_thickness_i < slab_thickness:
            cutoff_thickness_i = cutoff_thickness_i + 1.
        else:
            break_loop = True
        #__|

    return(slab)
    #__|

def create_save_dataframe(
    data_dict_list=None,
    df_slab_old=None,
    ):
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
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs",
        "out_data")
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

def update_sys_took_too_long(bulk_id, facet):
    """
    """
    #| - update_sys_took_too_long
    bulk_id_i = bulk_id
    facet_i = facet

    data = read_data_json()

    systems_that_took_too_long = data.get("systems_that_took_too_long", [])
    systems_that_took_too_long.append((bulk_id_i, facet_i))

    data["systems_that_took_too_long"] = systems_that_took_too_long

    data_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs",
        "out_data/data.json")
    with open(data_path, "w") as fle:
        json.dump(data, fle, indent=2)
    #__|

def df_dft_for_slab_creation(
    df_dft=None,
    bulk_ids__octa_unique=None,
    bulks_to_not_run=None,
    df_bulk_manual_class=None,
    frac_of_layered_to_include=None,
    verbose=False,
    ):
    """
    """
    #| - df_dft_for_slab_creation

    #| - Take only unique octahedral systems
    df_dft_i = df_dft.loc[bulk_ids__octa_unique]

    if verbose:
        print("df_dft_i.shape:", df_dft_i.shape[0])

    # Drop ids that were manually identified as bad
    ids_i = df_dft_i.index.intersection(bulks_to_not_run)
    df_dft_i = df_dft_i.drop(labels=ids_i)

    if verbose:
        print("df_dft_i.shape:", df_dft_i.shape[0])

    #__|

    #| - Check that all bulks are accounted for in df_bulk_manual_class
    for bulk_id_i, row_i in df_dft_i.iterrows():
        id_avail_i = bulk_id_i in df_bulk_manual_class.index
        if not id_avail_i:
            print("Bulk id nont in `df_bulk_manual_class`", bulk_id_i)
            print("Need to add it in manually")
    #__|

    #| - Removing all layered bulks and adding in just a bit
    df_bulk_manual_class_i = df_bulk_manual_class.loc[
        df_dft_i.index
        ]

    non_layered_ids = df_bulk_manual_class_i[
        df_bulk_manual_class_i.layered == False].index
    num_non_layered = non_layered_ids.shape[0]

    layered_ids = df_bulk_manual_class[
        df_bulk_manual_class.layered == True].index

    num_layered_to_inc = int(frac_of_layered_to_include * num_non_layered)
    layered_ids_to_use = np.random.choice(
        layered_ids, size=num_layered_to_inc, replace=False)

    if verbose:
        print("Number of layered bulks to include: ", len(layered_ids_to_use))
        print("Number of non-layered bulks:", len(non_layered_ids))

    all_ids = non_layered_ids.to_list() + list(layered_ids_to_use)

    df_dft_i = df_dft_i.loc[df_dft_i.index.intersection(all_ids)]
    #__|

    #| - Impose `dH-dH_hull` cutoff of 0.3
    # Actually I'll simply take the top 10 AB2 and AB3 structures

    if verbose:
        print("df_dft_i.shape:", df_dft_i.shape[0])

    # #########################################################
    df_dft_ab2 = df_dft_i[df_dft_i["stoich"] == "AB2"]

    min_dH = df_dft_ab2.dH.min()
    df_dft_ab2.loc[:, "dH_hull"] = df_dft_ab2.dH - min_dH

    # #########################################################
    df_dft_ab3 = df_dft_i[df_dft_i["stoich"] == "AB3"]

    min_dH = df_dft_ab3.dH.min()
    df_dft_ab3.loc[:, "dH_hull"] = df_dft_ab3.dH - min_dH

    # #########################################################
    df_dft_i = pd.concat([
        df_dft_ab2,
        df_dft_ab3,

        # #####################################################
        # df_dft_ab2.iloc[0:5],
        # df_dft_ab3.iloc[0:5],

        # df_dft_ab2.iloc[0:15],
        # df_dft_ab3.iloc[0:15],

        # df_dft_ab2.iloc[0:25],
        # df_dft_ab3.iloc[0:25],
        ])

    # print("df_dft_i.shape:", df_dft_i.shape[0])

    df_dft_i = df_dft_i[df_dft_i.dH_hull < 0.3]

    # print("df_dft_i.shape:", df_dft_i.shape[0])
    #__|


    return(df_dft_i)
    #__|

def create_save_struct_coord_df(
    slab_final=None,
    slab_id=None,
    ):
    """
    """
    #| - create_save_struct_coord_df
    slab_id_i = slab_id

    df_coord_slab_final = get_structure_coord_df(slab_final)

    file_name_i = slab_id_i + ".pickle"
    file_path_i = os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/creating_slabs/out_data/df_coord_files",
        file_name_i)
    my_file = Path(file_path_i)
    if not my_file.is_file():
        # df_coord_slab_final = get_structure_coord_df(slab_final)
        with open(file_path_i, "wb") as fle:
            pickle.dump(df_coord_slab_final, fle)
    #__|
