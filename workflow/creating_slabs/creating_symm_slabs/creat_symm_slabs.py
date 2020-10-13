# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python [conda env:PROJ_irox_oer] *
#     language: python
#     name: conda-env-PROJ_irox_oer-py
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys


# #########################################################
from methods import (
    get_df_dft,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,

    convert_str_facet_to_list,
    )

from local_methods import create_slab_from_bulk_tmp, create_final_slab_master_tmp
# -

# # Read Data

# +
df_dft = get_df_dft()

df_jobs = get_df_jobs()

df_jobs_data = get_df_jobs_data()

df_slab = get_df_slab()

# +
# Only taking systems which are being calculated (or finished) via DFT
# df_slab_i = df_slab.loc[df_jobs.slab_id.unique().tolist()]

slabs_dft_finished = df_jobs_data[df_jobs_data.completed == True].slab_id.tolist()
df_slab_i = df_slab.loc[slabs_dft_finished]
# -

df_slab_i = df_slab_i[df_slab_i.bulk_id == "cq7smr6lvj"]

df_slab_i

for slab_id, row_i in df_slab_i[["bulk_id", "facet", ]].iterrows():
    tmp = 42


    # #####################################################
    bulk_id = row_i.bulk_id
    facet = row_i.facet
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id]
    # #####################################################
    bulk_atoms = row_dft_i.atoms
    # #####################################################

    facet_list = convert_str_facet_to_list(facet)

    slab_0 = create_slab_from_bulk_tmp(
        atoms=bulk_atoms, facet=facet_list)

    # slab_1 = create_final_slab_master(atoms=slab_0)

    # slab_2 = constrain_slab(atoms=slab_1)
    # slab_final = slab_2

    # df_coord_slab_final = get_structure_coord_df(slab_final)

    file_name_i = "temp_" + bulk_id + "_" + slab_id + ".cif"
    file_path_i = os.path.join(
        "out_data",
        file_name_i)
    # slab_0.write("out_data/temp.cif")
    slab_0.write(file_path_i)

# +
# slab_1 = create_final_slab_master_tmp(atoms=slab_0)

# + active=""
#
#
#

# +
# remove_all_atoms_above_cutoff_symm

atoms = slab_0
# -

# # Developing new symmetric surface cutter

# +
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

from local_methods import (
    remove_all_atoms_above_cutoff,
    calc_surface_area,
    remove_highest_metal_atoms,
    get_slab_thickness,
    create_slab_from_bulk_tmp,
    remove_noncoord_oxygens,
    remove_nonsaturated_surface_metal_atoms,
    )
# -

# def create_final_slab_master_tmp(
atoms=slab_0
#     ):
#     """Master method to create final IrOx slab.
#     """

# +
# def create_final_slab_master_tmp(
#     atoms=None,
#     ):
#     """Master method to create final IrOx slab.
#     """
#| - create_final_slab_master
TEMP = True
slab = atoms

###########################################################
slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)


# slab = remove_all_atoms_above_cutoff(atoms=slab, cutoff_thickness=17)
slab = remove_all_atoms_above_cutoff_symm(atoms=slab, cutoff_thickness=17)
if TEMP:
    slab.write("out_data/temp_out/slab_1_0.cif")

slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)


###########################################################
slab = remove_nonsaturated_surface_metal_atoms(
    atoms=slab,
    dz=4)
if TEMP:
    slab.write("out_data/temp_out/slab_1_1.cif")

# slab.write("out_data/temp_2.cif")
slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)

slab = remove_noncoord_oxygens(atoms=slab)
if TEMP:
    slab.write("out_data/temp_out/slab_1_2.cif")

slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)



###########################################################
i_cnt = 2
while slab_thickness_i > 15:
    i_cnt += 1
    # print(i_cnt)

    # #####################################################
    # Figuring out how many surface atoms to remove at one time
    # Taken from R-IrO2 (100), which has 8 surface Ir atoms and a surface area of 58 A^2
    surf_area_per_surface_metal = 58 / 8
    surface_area_i = calc_surface_area(atoms=slab)
    ideal_num_surface_atoms = surface_area_i / surf_area_per_surface_metal
    num_atoms_to_remove = ideal_num_surface_atoms / 3
    num_atoms_to_remove = int(np.round(num_atoms_to_remove))
    # #####################################################

    slab_new_0 = remove_highest_metal_atoms(
        atoms=slab,
        num_atoms_to_remove=num_atoms_to_remove,
        metal_atomic_number=77)
    if TEMP:
        slab_new_0.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_0" + ".cif")

    slab_new_1 = remove_nonsaturated_surface_metal_atoms(
        atoms=slab_new_0,
        dz=4)
    if TEMP:
        slab_new_1.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_1" + ".cif")

    slab_new_2 = remove_noncoord_oxygens(atoms=slab_new_1)
    if TEMP:
        slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + "_2" + ".cif")


    slab_thickness_i = get_slab_thickness(atoms=slab_new_2)
    # print("slab_thickness_i:", slab_thickness_i)

    if TEMP:
        slab_new_2.write("out_data/temp_out/slab_1_" + str(i_cnt) + ".cif")

    slab = slab_new_2

# return(slab)
#__|
# -

slab_1 = slab

# + active=""
#
#
#
# -

slab_1.write("out_data/temp_1.cif")

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# from catkit.gen import symmetry

# # symmetry.g
