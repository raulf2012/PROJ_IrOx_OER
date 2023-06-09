#TEMP

# | - Import Modules
import os
# print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import math

from ase import io

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_jobs_data,
    get_df_jobs,
    read_pdos_data,
    )
from methods import get_df_coord
# __|


def get_active_Bader_charges_1(
    path=None,
    df_atoms_sorted_ind=None,
    compenv=None,
    slab_id=None,
    ads=None,
    active_site=None,
    att_num_bader=None,
    verbose=None,
    ):
    """
    """
    #| - get_active_Bader_charges_1
    useful_vars_dict = get_data_for_Bader_methods(
        df_atoms_sorted_ind=df_atoms_sorted_ind,
        compenv=compenv,
        slab_id=slab_id,
        ads=ads,
        active_site=active_site,
        att_num_bader=att_num_bader,
        verbose=verbose,
        )
    Ir_index = useful_vars_dict["active_Ir_index"]
    new_active_site = useful_vars_dict["new_active_site"]
    active_Ir_index_mapped = useful_vars_dict["active_Ir_index_mapped"]



    file_i = "bader_charge.json"
    file_path_i = os.path.join(path, file_i)


    atoms_bader_i = io.read(file_path_i)

    active_O_bader_i = atoms_bader_i[new_active_site].charge


    Ir_bader_i = atoms_bader_i[
        active_Ir_index_mapped
        ].charge

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["active_O_bader"] = active_O_bader_i
    out_dict["Ir_bader"] = Ir_bader_i
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|


























def get_active_Bader_charges_2(
    path=None,
    df_atoms_sorted_ind=None,
    compenv=None,
    slab_id=None,
    ads=None,
    active_site=None,
    att_num_bader=None,
    verbose=None,
    ):
    """
    """
    #| - get_active_Bader_charges_2
    useful_vars_dict = get_data_for_Bader_methods(
        df_atoms_sorted_ind=df_atoms_sorted_ind,
        compenv=compenv,
        slab_id=slab_id,
        ads=ads,
        active_site=active_site,
        att_num_bader=att_num_bader,
        verbose=verbose,
        )
    Ir_index = useful_vars_dict["active_Ir_index"]
    new_active_site = useful_vars_dict["new_active_site"]
    active_Ir_index_mapped = useful_vars_dict["active_Ir_index_mapped"]


    # | - Create Bader DataFrame
    file_path_i = os.path.join(
        path,
        # path_i,
        "bader_charges.txt",
        )

    with open(file_path_i, "r") as f:
        lines = f.read().splitlines()

    data_dict_list = []
    for line_i in lines:
        line_split_i = line_i.split(" ")

        # #################################################
        index_i = line_split_i[1]
        name_i = line_split_i[3]
        charge_i = line_split_i[5]
        # #################################################

        index_i = int(index_i)
        charge_i = float(charge_i)

        # #################################################
        data_dict_i = dict()
        # #################################################
        data_dict_i["atom_index"] = index_i
        data_dict_i["charge"] = charge_i
        data_dict_i["name"] = name_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

    df_bader_i = pd.DataFrame(data_dict_list)
    df_bader_i = df_bader_i.set_index("atom_index", drop=False)
    # __|


    bader_charge_Ir = df_bader_i.loc[active_Ir_index_mapped].charge
    bader_charge_O = df_bader_i.loc[new_active_site].charge


    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["active_O_bader"] = bader_charge_O
    out_dict["Ir_bader"] = bader_charge_Ir
    # #####################################################
    return(out_dict)
    # #####################################################
    # __|





























def get_data_for_Bader_methods(
    path=None,
    df_atoms_sorted_ind=None,
    compenv=None,
    slab_id=None,
    ads=None,
    active_site=None,
    att_num_bader=None,
    verbose=None,
    ):
    """
    This is pre-work for get_active_Bader_charges_1 and get_active_Bader_charges_2
    """
    #| - get_active_Bader_charges_1

    # file_i = "bader_charge.json"
    #
    # file_path_i = os.path.join(path, file_i)
    # # file_path_i = os.path.join(dir_i, file_i)










    # atoms_bader_i = io.read(file_path_i)

    # if False:
    #     atoms_bader_i.write("__temp__/atoms_bader.json")

    # Get the new active site number to use (atoms objects get shuffled around)
    # #############################################
    row_atoms_i = df_atoms_sorted_ind.loc[
        ("dos_bader", compenv, slab_id, ads, active_site, att_num_bader, )]
    # #############################################
    atom_index_mapping_i = row_atoms_i.atom_index_mapping
    # #############################################

    atom_index_mapping_i = {v: k for k, v in atom_index_mapping_i.items()}

    new_active_site = atom_index_mapping_i[active_site]
    # new_active_site = new_active_site + 1

    # active_O_bader_i = atoms_bader_i[new_active_site].charge



    init_slab_tuple = (compenv, slab_id, ads, "NaN", 1, )

    df_coord_i = get_df_coord(
        mode="init-slab",  # 'bulk', 'slab', 'post-dft', 'init-slab'
        init_slab_name_tuple=init_slab_tuple,
        verbose=verbose,
        )

    row_coord_i = df_coord_i.loc[active_site]

    Ir_nn_found = False
    nn_Ir = None
    for nn_i in row_coord_i["nn_info"]:
        symbol_i = nn_i["site"].specie.symbol
        if symbol_i == "Ir":
            nn_Ir = nn_i
            Ir_nn_found = True


    Ir_bader_charge_i = None
    if Ir_nn_found:
        Ir_index = nn_Ir["site_index"]

        # Ir_bader_i = atoms_bader_i[
        #     atom_index_mapping_i[Ir_index]
        #     ].charge
    else:
        print("Ir not found")

    # #####################################################
    out_dict = dict()
    # #####################################################
    out_dict["active_Ir_index"] = Ir_index
    out_dict["active_Ir_index_mapped"] = atom_index_mapping_i[Ir_index]
    out_dict["new_active_site"] = new_active_site
    # #####################################################
    return(out_dict)
    # #####################################################

    # __|
