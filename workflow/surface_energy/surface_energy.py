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

# # Computing surface energy from OER slabs and bulk formation energy

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import plotly.express as px

# #########################################################
from ase_modules.ase_methods import create_species_element_dict

# #########################################################
from proj_data import metal_atom_symbol
from proj_data import stoich_color_dict

# #########################################################
from methods import (
    get_df_dft,
    get_df_jobs_data,
    get_df_jobs,
    get_df_features_targets,
    get_df_slab,
    )

# #########################################################
# Data from PROJ_irox
sys.path.insert(0, os.path.join(
    os.environ["PROJ_irox"], "data"))
from proj_data_irox import h2_ref, h2o_ref
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
df_features_targets = get_df_features_targets()

df_jobs = get_df_jobs()

df_dft = get_df_dft()

df_jobs_data = get_df_jobs_data()

# +
# # TEMP
# print(222 * "TEMP | ")

# df_features_targets = df_features_targets.sample(n=30)
# -

# ### Preparing oxygen reference energy

G_O = -1 * ((-1.23 * 2) - h2o_ref + h2_ref)

# ### Main loop

# +
# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_features_targets.iterrows():
    # print(name_i)

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(
        df_features_targets.index.names,
        name_i))
    # #####################################################
    job_id_o_i = row_i[("data", "job_id_o", "")]
    stoich_i = row_i[("data", "stoich", "")]
    # #####################################################

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_o_i]
    # #####################################################
    elec_energy_i = row_data_i.pot_e
    atoms_init_i = row_data_i.init_atoms
    # #####################################################

    # #####################################################
    row_jobs_i = df_jobs.loc[job_id_o_i]
    # #####################################################
    bulk_id_i = row_jobs_i.bulk_id
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    bulk_energy_pa_i = row_dft_i.energy_pa
    # #####################################################

    
    # Calculate surface area of slab
    cell = atoms_init_i.cell

    cross_prod_i = np.cross(cell[0], cell[1])
    area_i = np.linalg.norm(cross_prod_i)

    elem_dict_i = create_species_element_dict(
        atoms_init_i,
        include_all_elems=False,
        elems_to_always_include=None,
        )

    stoich_B_i = int(stoich_i[2:])

    num_atoms_in_form_unit = stoich_B_i + 1

    num_metal_atoms = elem_dict_i[metal_atom_symbol]
    N_stoich_units = num_metal_atoms

    num_stoich_O = num_metal_atoms * stoich_B_i

    num_nonstoich_O = elem_dict_i["O"] - num_stoich_O

    assert num_nonstoich_O >= 0, "Must have non-negative number of non-stoich Os"

    surf_energy_i_0 = elec_energy_i - \
    (N_stoich_units * num_atoms_in_form_unit * bulk_energy_pa_i) - \
    (num_nonstoich_O * G_O)

    norm_mode = "area"
    units = "J/m^2"

    if norm_mode == "area":
        norm_term = 2 * area_i
        surf_energy_i_1 = surf_energy_i_0 / norm_term
    else:
        print("NOT GOOD")

    if norm_mode == "area":
        if units == "eV/A^2":
            pass
        elif units == "J/m^2":
            # Convert eV/A^2 to J/m^2
            # (1E10 A/m) ^ 2 * (1.6022E-19 J/eV) = 16.022
            ev_A2__to__J_m2 = 16.022
            surf_energy_i_2 = surf_energy_i_1 * ev_A2__to__J_m2
            surf_energy__area_J_m2 = surf_energy_i_2


    # print(
    #     "SE: ",
    #     # str(np.round(surf_energy_i_2, 3)).zfill(5),
    #     np.round(surf_energy_i_2, 3),
    #     " J/m2",
    #     sep="")


    # #####################################################
    data_dict_i.update(name_dict_i)
    # #####################################################
    data_dict_i["SE__area_J_m2"] = surf_energy__area_J_m2
    data_dict_i["num_nonstoich_O"] = num_nonstoich_O
    data_dict_i["N_stoich_units"] = N_stoich_units
    data_dict_i["stoich"] = stoich_i
    # data_dict_i[""] = 
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_SE = pd.DataFrame(data_dict_list)
df_SE = df_SE.set_index(["compenv", "slab_id", "active_site"])
# #########################################################
# -

df_SE

# ### Plot surface energy data histogram

# +
fig = px.histogram(df_SE,
    x="SE__area_J_m2",
    color="stoich",
    barmode="overlay",
    barnorm="percent",
    color_discrete_map=stoich_color_dict,
    # 'fraction'` or `'percent'`
    )

# 'group'
# 'overlay'
# 'relative'

fig.show()
# -

# ### Writting data to file

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/surface_energy/out_data")
file_name_i = "df_SE.pickle"
path_i = os.path.join(directory, file_name_i)
if not os.path.exists(directory): os.makedirs(directory)
with open(path_i, "wb") as fle:
    pickle.dump(df_SE, fle)
# #########################################################

# +
from methods import get_df_SE

df_SE_tmp = get_df_SE()
df_SE_tmp
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("surface_energy.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# dir(atoms_init_i)

# + jupyter={"source_hidden": true}
# row_i.data

# + jupyter={"source_hidden": true}
# elem_dict_i

# + jupyter={"source_hidden": true}
# data_dict_i

# + jupyter={"source_hidden": true}
# str(np.round(surf_energy_i_2, 3)).zfill(5)

# + jupyter={"source_hidden": true}
# norm_mode
# units

# + jupyter={"source_hidden": true}
# name_dict_i = dict(zip(
#     df_features_targets.index.names,
#     name_i))

# + jupyter={"source_hidden": true}
# norm_mode = "area"
# # units = "eV/A^2"  # 'eV/A^2' or 'J/m^2'
# units = "J/m^2"  # 'eV/A^2' or 'J/m^2'

# + jupyter={"source_hidden": true}
#     surf_energy_i_0 = elec_energy_i - \
#     (N_stoich_units * num_atoms_in_form_unit * bulk_energy_pa_i) - \
#     (num_nonstoich_O * G_O)

# N_stoich_units
# num_nonstoich_O

# + jupyter={"source_hidden": true}
# df_features_targets.head()

# + jupyter={"source_hidden": true}
# def get_df_SE():
#     """
#     The data object is created by the following notebook:

#     $PROJ_irox_oer/workflow/surface_energy/surface_energy.ipynb
#     """
#     #| - get_df_jobs
#     # #####################################################
#     # Reading df_jobs dataframe from pickle
#     import pickle; import os
#     path_i = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "workflow/surface_energy",
#         "out_data/df_SE.pickle")
#     with open(path_i, "rb") as fle:
#         df_SE = pickle.load(fle)

#     return(df_SE)
#     #__|

# + jupyter={"source_hidden": true}
# num_nonstoich_O 

# +

# # px.histogram?


