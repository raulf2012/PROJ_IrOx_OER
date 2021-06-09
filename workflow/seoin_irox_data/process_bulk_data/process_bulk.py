# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle

import pandas as pd

from ase import io
# -

dir_root = os.path.join(
    os.environ["PROJ_DATA"],
    "PROJ_IrOx_OER/seoin_irox_data/bulk",
    )

# +
bulk_systems_dict = {
    
    "iro2": [
        "anatase-fm",
        "brookite-fm",
        "columbite-fm",
        "pyrite-fm",
        "rutile-fm",
        ],

    "iro3": [
        "Amm2",
        "cmcm",
        "pm-3m",
        ],

    }

crystal_rename_dict = {
    'anatase-fm': 'anatase',
    'brookite-fm': 'brookite',
    'columbite-fm': 'columbite',
    'pyrite-fm': 'pyrite',
    'rutile-fm': 'rutile',
    'Amm2': 'amm2',
    'cmcm': 'cmcm',
    'pm-3m': 'pm-3m',
    }


# -

def calc_dH(
    e_per_atom,
    stoich=None,
    num_H_atoms=0,
    ):
    """
    The original method is located in:
    F:\Dropbox\01_norskov\00_git_repos\PROJ_IrOx_Active_Learning_OER\data\proj_data_irox.py

    Based on a E_DFT/atom of -7.047516 for rutile-IrO2

    See the following dir for derivation:
        PROJ_IrOx_Active_Learning_OER/workflow/energy_treatment_deriv/calc_references
    """
    # | - calc_dH
    o_ref = -4.64915959
    ir_metal_fit = -9.32910211636731
    h_ref = -3.20624595

    if stoich == "AB2":
        dH = (2 + 1) * e_per_atom - 2 * o_ref - ir_metal_fit
        dH_per_atom = dH / 3.
    elif stoich == "AB3":
        dH = (3 + 1) * e_per_atom - 3 * o_ref - ir_metal_fit
        dH_per_atom = dH / 4.

    elif stoich == "IrHO3" or stoich == "IrO3H" or stoich == "iro3h" or stoich == "iroh3":
        dH = (3 + 1 + 1) * e_per_atom - 3 * o_ref - ir_metal_fit - h_ref
        dH_per_atom = dH / 5.


    return(dH_per_atom)
    #__|

# +
data_dict_list = []
for bulk_i, polymorphs_i in bulk_systems_dict.items():
    # print(bulk_i)
    
    if bulk_i == "iro2":
        stoich_i = "AB2"
    elif bulk_i == "iro3":
        stoich_i = "AB3"


    for poly_j in polymorphs_i:
        # print("  ", poly_j, sep="")

        # poly_new_j = crystal_rename_dict.get(poly_j, poly_j)
        poly_new_j = crystal_rename_dict.get(poly_j, "TEMP")


        path_rel_j = os.path.join(
            bulk_i,
            poly_j)
        path_j = os.path.join(
            dir_root,
            path_rel_j)


        # #################################################
        # Reading OUTCAR atoms
        atoms_j = io.read(os.path.join(path_j, "OUTCAR"))

        pot_e_j = atoms_j.get_potential_energy()

        volume_j = atoms_j.get_volume()
        num_atoms_j = atoms_j.get_global_number_of_atoms()

        volume_pa_j = volume_j / num_atoms_j

        # print(atoms_j.get_chemical_formula())

        # #################################################
        # Calc formation energy
        dH_i = calc_dH(
            pot_e_j / num_atoms_j,
            stoich=stoich_i,
            num_H_atoms=0,
            )

        # #################################################
        data_dict_j = dict()
        # #################################################
        data_dict_j["stoich"] = stoich_i
        # data_dict_j["crystal"] = poly_j
        data_dict_j["crystal"] = poly_new_j
        data_dict_j["dH"] = dH_i
        data_dict_j["volume"] = volume_j
        data_dict_j["num_atoms"] = num_atoms_j
        data_dict_j["volume_pa"] = volume_pa_j
        data_dict_j["atoms"] = atoms_j
        data_dict_j["path"] = path_rel_j
        # #################################################
        data_dict_list.append(data_dict_j)
        # #################################################

# #########################################################
df = pd.DataFrame(
    data_dict_list
    )
# #########################################################
# -



# +
df_seoin_bulk = df

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/seoin_irox_data/process_bulk_data",
    "out_data")
if not os.path.exists(directory):
    os.makedirs(directory)

with open(os.path.join(directory, "df_seoin_bulk.pickle"), "wb") as fle:
    pickle.dump(df_seoin_bulk, fle)
# #########################################################
# -

df

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
#     'amm2',
#     'anatase',
#     'brookite',
#     'cmcm',
#     'columbite',
#     'pm-3m',
#     'pyrite',
#     'rutile',

# + jupyter={"source_hidden": true}
#     'anatase-fm',
#     'brookite-fm',
#     'columbite-fm',
#     'pyrite-fm',
#     'rutile-fm',
#     'Amm2',
#     'cmcm',
#     'pm-3m',

# + jupyter={"source_hidden": true}
# df.crystal.tolist()
