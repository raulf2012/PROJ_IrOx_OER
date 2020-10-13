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

# +
import os
print(os.getcwd())
import sys

from pathlib import Path
import copy
import pickle

import json

import numpy as np
import pandas as pd

from ase import io

# from tqdm import tqdm
from tqdm.notebook import tqdm

# #########################################################


# #########################################################
from misc_modules.pandas_methods import drop_columns
from misc_modules.misc_methods import GetFriendlyID
from ase_modules.ase_methods import view_in_vesta

# #########################################################
from methods import (
    get_df_dft, symmetrize_atoms,
    get_structure_coord_df, remove_atoms)
from proj_data import metal_atom_symbol

# #########################################################
from local_methods import (
    analyse_local_coord_env, check_if_sys_processed,
    remove_nonsaturated_surface_metal_atoms,
    remove_noncoord_oxygens,
    create_slab_from_bulk,
    get_slab_thickness,
    remove_highest_metal_atoms,
    remove_all_atoms_above_cutoff,
    create_final_slab_master,
    constrain_slab,
    )

from local_methods import calc_surface_area
# -



# +
from methods import get_df_slab, get_slab_thickness

df_slab = get_df_slab()

# +
data_dict_list = []
for slab_id_i, row_i in df_slab.iterrows():
    data_dict_i = dict()

    data_dict_i["slab_id"] = slab_id_i

    # #####################################################
    slab_final = row_i.slab_final
    # #####################################################

    slab_thick_i = get_slab_thickness(atoms=slab_final)
    
    data_dict_i["slab_thick"] = slab_thick_i

    # #####################################################
    data_dict_list.append(data_dict_i)

# #########################################################
df_slab_info = pd.DataFrame(data_dict_list)
df_slab_info = df_slab_info.set_index("slab_id")

df_slab = pd.concat([
    df_slab,
    df_slab_info,
    ], axis=1)
# -

df_slab

assert False

# +
# #########################################################
# DFT dataframe
df_dft = get_df_dft()

# #########################################################
# Previous df_slab dataframe
path_i = os.path.join(
    "out_data",
    # "__old__",
    "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
else:
    df_slab_old = pd.DataFrame()

print("df_slab_old.shape:", df_slab_old.shape)

# #########################################################
# Bulks not to run, manually checked to be erroneous/bad
data_path = os.path.join(
    "in_data/bulks_to_not_run.json")
with open(data_path, "r") as fle:
    bulks_to_not_run = json.load(fle)
# -

df_slab_old = df_slab_old[~df_slab_old.slab_final.isna()]


# +
def method(row_i):
    """
    """
    # row_i = df_slab_old.iloc[0]
    atoms = row_i.slab_final

    num_atoms = atoms.get_number_of_atoms()
    return(num_atoms)

df_i = df_slab_old
df_i["num_atoms"] = df_i.apply(
    method,
    axis=1)

# +
import pandas as pd
import plotly.graph_objs as go

y_array = df_i.iter_time_i / 60
x_array = df_i.num_atoms

trace = go.Scatter(
    x=x_array,
    y=y_array,
    mode="markers",
    )

data = [trace]

fig = go.Figure(data=data)
fig.show()

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
slab_0 = io.read(
    "out_data/final_slabs_2/zimuby8uzj__fagepuha_94__001.cif"
    # "out_data/temp_out/slab_1_2.cif"
    )

slab_0.write("out_data/temp_out/slab_0.cif")
atoms = slab_0

# +

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

# +
# slab_new.write()

slab_new.write("out_data/temp_out/slab_1.cif")

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
slab_0 = io.read(
    "out_data/final_slabs_2/zimuby8uzj__fagepuha_94__001.cif"
    # "out_data/temp_out/slab_1_2.cif"
    )

# slab_0.write("out_data/temp_out_0/slab_0.cif")

# +
# # def remove_highest_metal_atoms(
# atoms=slab_0
# num_atoms_to_remove=num_atoms_to_remove
# metal_atomic_number=77
# # ):
# """
# """
# #| - remove_highest_metal_atom
# slab_m = atoms[atoms.numbers == metal_atomic_number]

# positions_cpy = copy.deepcopy(slab_m.positions)
# positions_cpy_sorted = positions_cpy[positions_cpy[:,2].argsort()]


# # positions_z = positions_cpy[:, 2]
# # positions_z.sort()
# # positions_z = np.flip(positions_z)


# indices_to_remove = []
# for coord_i in positions_cpy_sorted[-2:]:
#     for i_cnt, atom in enumerate(atoms):
#         if all(atom.position == coord_i):
#             print("PSIDFJIDSJi")
#             indices_to_remove.append(i_cnt)

# slab_new = remove_atoms(
#     atoms=atoms,
#     atoms_to_remove=indices_to_remove,
#     )

# # return(slab_new)
# #__|

# +
# indices_to_remove = []
# for coord_i in positions_cpy_sorted[-2:]:
#     for i_cnt, atom in enumerate(atoms):
#         if all(atom.position == coord_i):
#             print("PSIDFJIDSJi")
#             indices_to_remove.append(i_cnt)

# +

# slab_new.write("out_data/temp_out_0/slab_1.cif")

# +
# positions_cpy

# positions_cpy_sorted = positions_cpy[positions_cpy[:,2].argsort()]

# array_tmp = np.array([
#     [1, 2, 3],
#     [1, 2, 8],
#     [1, 2, 1],
#     ]
#     )
# array_tmp[array_tmp[:,2].argsort()]

# positions_cpy

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
# #########################################################
# DFT dataframe
df_dft = get_df_dft()

# #########################################################
# Previous df_slab dataframe
path_i = os.path.join(
    "out_data",
    # "__old__",
    "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
else:
    df_slab_old = pd.DataFrame()

print("df_slab_old.shape:", df_slab_old.shape)

# #######################################################################
# Bulks not to run, manually checked to be erroneous/bad
data_path = os.path.join(
    "in_data/bulks_to_not_run.json")
with open(data_path, "r") as fle:
    bulks_to_not_run = json.load(fle)
# -

df_slab_old

assert False

# +
bad_ids = [
    "fukebife_19",
    "karosepo_32",
    "nepobapa_79",
    "wuhasulu_74",
    "wonasofa_20",
    "fovakevo_63",
    "vasubaba_10",
    "tetekuse_50",
    "tasiluno_60",
    "henivako_70",
    "kivesohe_51",
    "redobodi_26",
    "vokumemi_16",
    "tumolubo_46",
    "lanasahi_54",
    "magevawo_12",
    "sorogane_14",
    "gisasaho_61",
    ]

df_i = df_slab_old.loc[bad_ids][["bulk_id", "facet"]]

data_dict_list = []
for i_cnt, row_i in df_i.iterrows():
    tmp = 42

    data_dict_i = dict()

    bulk_id = row_i.bulk_id
    facet_str = row_i.facet

    facet = (facet_str[0], facet_str[1], facet_str[2])
    facet = tuple([int(i) for i in facet])

    data_dict_i["bulk_id"] = bulk_id
    data_dict_i["facet"] = facet
    data_dict_list.append(data_dict_i)

data_dict_list

# +
# Pickling data ###########################################
out_dict = data_dict_list
# out_dict["TEMP"] = None

import os; import pickle
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(out_dict, fle)
# #########################################################

# #########################################################
import pickle; import os
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp.pickle")
with open(path_i, "rb") as fle:
    out_dict = pickle.load(fle)
# #########################################################

# +
# # #######################################################################
# # Bulks not to run, manually checked to be erroneous/bad
# data_path = os.path.join(
#     "in_data/bulks_to_not_run.json")
# with open(data_path, "r") as fle:
#     bulks_to_not_run = json.load(fle)

# +
# bulks_to_not_run


# root_dir = "out_data"
# for subdir, dirs, files in os.walk(root_dir):
#     if "bulk_structures_temp" in subdir:
#         continue

#     for file in files:
#         # print(os.path.join(subdir, file))
#         file_path = os.path.join(subdir, file)

#         for bulk_i in bulks_to_not_run:
#             if bulk_i in file:
#                 print(file)
#                 # os.remove(file_path)

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
# #########################################################
# DFT dataframe
df_dft = get_df_dft()

# #########################################################
# Previous df_slab dataframe
path_i = os.path.join(
    "out_data",
    "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
else:
    df_slab_old = pd.DataFrame()

print("df_slab_old.shape:", df_slab_old.shape)

# #######################################################################
# Bulks not to run, manually checked to be erroneous/bad
data_path = os.path.join(
    "in_data/bulks_to_not_run.json")
with open(data_path, "r") as fle:
    bulks_to_not_run = json.load(fle)

# +
df_slab_old.shape

# (472, 5)
# -

assert False

# +
df_slab_old

bulks_to_not_run


df_slab_old = df_slab_old[~df_slab_old.bulk_id.isin(bulks_to_not_run)]

# +
df_slab_old

# Pickling data ###########################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "data_new_new.pickle"), "wb") as fle:
    pickle.dump(df_slab_old, fle)
# #########################################################

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# +
# ########################################################
data_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs/selecting_bulks",
    "out_data/data.json")
with open(data_path, "r") as fle:
    data = json.load(fle)
# ########################################################

bulk_ids__octa_unique = data["bulk_ids__octa_unique"]
# -

assert False

# +
path_i = os.path.join("out_data", "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
    print("df_slab_old.shape:", df_slab_old.shape)

    row_i = df_slab_old.iloc[1]
    df_slab_old.sort_values("iter_time_i", ascending=False)
    

# df_slab_old.head()
df_slab_old.sort_values("iter_time_i", ascending=False)
# -

assert False

# +
df_slab = df_slab_old

row_i = df_slab.loc["kupopaba_83"]

atoms = row_i.slab_final


# -

def resize_z_slab(atoms=None, vacuum=15):
    """
    """
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


atoms_new = resize_z_slab(atoms=atoms, vacuum=15)


assert False

# +
row_i = df_slab_old.iloc[-1]
atoms = row_i.slab_final

num_atoms = atoms.get_number_of_atoms()


# +
#COMBAK

# +
def method(row_i):
    """
    """
    # row_i = df_slab_old.iloc[0]
    atoms = row_i.slab_final

    num_atoms = atoms.get_number_of_atoms()
    return(num_atoms)

df_i = df_slab_old
df_i["num_atoms"] = df_i.apply(
    method,
    axis=1)

# +
import pandas as pd
# pd.options.plotting.backend = "plotly"
# df_i[["iter_time_i", "num_atoms"]].plot()

import plotly.graph_objs as go

# x_array = [0, 1, 2, 3]
# y_array = [0, 1, 2, 3]

y_array = df_i.iter_time_i / 60
x_array = df_i.num_atoms

trace = go.Scatter(
    x=x_array,
    y=y_array,
    mode="markers",
    )

data = [trace]

fig = go.Figure(data=data)
fig.show()

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# +
# with open("out_data/out.log", 'a') as the_file:
#     the_file.write("Hello" + "\n")

# +
# import signal
# from time import sleep    # only needed for testing

# timelimit_seconds = 3    # Must be an integer

# # Custom exception for the timeout
# class TimeoutException(Exception):
#     pass

# # Handler function to be called when SIGALRM is received
# def sigalrm_handler(signum, frame):
#     # We get signal!
#     raise TimeoutException()

# # Function that takes too long for bananas and oranges
# def mix(f):
#     if 'n' in f:
#         sleep(20)
#     else:
#         sleep(0.5)

# fruits = ['apple', 'banana', 'grape', 'strawberry', 'orange']
# for f in fruits:
#     # Set up signal handler for SIGALRM, saving previous value
#     old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
#     # Start timer
#     signal.alarm(timelimit_seconds)
#     try:
#         mix(f)
#         print(f, 'was mixed')
#     except TimeoutException:
#         print(f, 'took too long to mix')
#     finally:
#         # Turn off timer
#         signal.alarm(0)
#         # Restore handler to previous value
#         signal.signal(signal.SIGALRM, old_handler)

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
path_i = os.path.join("out_data", "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)

print("df_slab_old.shape:", df_slab_old.shape)

row_i = df_slab_old.iloc[1]

# -

df_slab_old.sort_values("iter_time_i", ascending=False).head()

row_i

# +
slab = row_i.slab_final

slab_new = constrain_slab(atoms=slab)
# -

slab_new.write("temp.traj")

# +
# positions = slab.positions

# z_pos = positions[:,2]

# z_max = np.max(z_pos)
# z_min = np.min(z_pos)

# # ang_of_slab_to_constrain = (2 / 4) * (z_max - z_min)
# ang_of_slab_to_constrain = (z_max - z_min) - 6


# # #########################################################
# indices_to_constrain = []
# for atom in slab:
#     if atom.symbol == metal_atom_symbol:
#         if atom.position[2] < (z_min + ang_of_slab_to_constrain):
#             indices_to_constrain.append(atom.index)

# for atom in slab:
#     if atom.position[2] < (z_min + ang_of_slab_to_constrain - 2):
#         indices_to_constrain.append(atom.index)

# df_coord_slab_i = get_structure_coord_df(slab)

# # #########################################################
# other_atoms_to_constrain = []
# for ind_i in indices_to_constrain:
#     row_i = df_coord_slab_i[df_coord_slab_i.structure_index == ind_i]
#     row_i = row_i.iloc[0]

#     nn_info = row_i.nn_info

#     for nn_i in nn_info:
#         ind_j = nn_i["site_index"]
#         other_atoms_to_constrain.append(ind_j)

# print(len(indices_to_constrain))

# indices_to_constrain.extend(other_atoms_to_constrain)

# print(len(indices_to_constrain))

# # #########################################################
# constrain_bool_mask = []
# for atom in slab:
#     if atom.index in indices_to_constrain:
#         constrain_bool_mask.append(True)
#     else:
#         constrain_bool_mask.append(False)

# # #########################################################
# slab_cpy = copy.deepcopy(slab)

# from ase.constraints import FixAtoms
# c = FixAtoms(mask=constrain_bool_mask)
# slab_cpy.set_constraint(c)

# slab_cpy.constraints

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

assert False

# +
path_i = os.path.join(
    "out_data",
    "df_slab.pickle")
my_file = Path(path_i)
if my_file.is_file():
    print("File exists!")
    with open(path_i, "rb") as fle:
        df_slab_old = pickle.load(fle)
else:
    df_slab_old = pd.DataFrame()

df_slab = df_slab_old

# +
# row_i = df_slab.loc["muniketo_71"]
row_i = df_slab.loc["fipohulu_74"]

# row_i = df_slab.iloc[0]

slab = row_i.slab_2
print(slab.symbols.get_chemical_formula())

slab_thickness_i = get_slab_thickness(atoms=slab)
print("slab_thickness_i:", slab_thickness_i)

slab.write("out_data/temp_0.cif")
# -

slab_final = create_final_slab_master(atoms=slab)

assert False

# +
# ###########################################################
# slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)

# slab = remove_all_atoms_above_cutoff(atoms=slab, cutoff_thickness=17)
# slab.write("out_data/temp_1.cif")

# slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)



# ###########################################################
# slab = remove_nonsaturated_surface_metal_atoms(
#     atoms=slab,
#     dz=4)

# slab.write("out_data/temp_2.cif")
# slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)

# slab = remove_noncoord_oxygens(atoms=slab)

# slab.write("out_data/temp_3.cif")
# slab_thickness_i = get_slab_thickness(atoms=slab)
# print("slab_thickness_i:", slab_thickness_i)



# ###########################################################
# i_cnt = 3
# while slab_thickness_i > 15:
#     i_cnt += 1
#     print(i_cnt)

#     # #####################################################
#     # Figuring out how many surface atoms to remove at one time
#     # Taken from R-IrO2 (100), which has 8 surface Ir atoms and a surface area of 58 A^2
#     surf_area_per_surface_metal = 58 / 8
#     surface_area_i = calc_surface_area(atoms=slab)
#     ideal_num_surface_atoms = surface_area_i / surf_area_per_surface_metal
#     num_atoms_to_remove = ideal_num_surface_atoms / 3
#     num_atoms_to_remove = int(np.round(num_atoms_to_remove))
#     # #####################################################

#     slab_new_0 = remove_highest_metal_atoms(
#         atoms=slab,
#         num_atoms_to_remove=num_atoms_to_remove,
#         metal_atomic_number=77)

#     slab_new_1 = remove_nonsaturated_surface_metal_atoms(
#         atoms=slab_new_0,
#         dz=4)

#     slab_new_2 = remove_noncoord_oxygens(atoms=slab_new_1)


#     slab_thickness_i = get_slab_thickness(atoms=slab_new_2)
#     print("slab_thickness_i:", slab_thickness_i)

#     slab_new_2.write("out_data/temp_" + str(i_cnt) + ".cif")

#     slab = slab_new_2

# +
# surf_area_per_surface_metal = 58 / 8

# +
# slab_new = remove_highest_metal_atoms(
#     atoms=slab,
#     num_atoms_to_remove=3,
#     metal_atomic_number=77,
#     )

# print(slab_new.symbols.get_chemical_formula())
# slab_new.write("out_data/temp_1.cif")

# +
# def remove_highest_metal_atom(
#     atoms=None,
#     metal_atomic_number=77,
#     ):
#     # TODO Make this more robust, don't just code in 77
#     slab_m = slab[slab.numbers == 77]
#     highest_atom_ind = np.argmax(slab_m.positions[:,2])

#     iterator = enumerate(slab.positions == slab_m[highest_atom_ind].position)
#     for i_cnt, bool_list in iterator:
#         if all(bool_list):
#             highest_atom_ind_new = i_cnt

#     slab_new = copy.deepcopy(slab)

#     slab_new.pop(highest_atom_ind_new)

#     return(slab_new)

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# +
# from tqdm.auto import trange
# from time import sleep

# for i in trange(4, desc='1st loop'):
#     for j in trange(5, desc='2nd loop'):
#         for k in trange(50, desc='3rd loop', leave=False):
#             sleep(0.01)

from tqdm.notebook import tqdm
# from tqdm import trange, tqdm
from time import sleep

iterator = tqdm(["a", "b", "c", "d"])
for i_cnt, bulk_id in enumerate(iterator):
    for j in tqdm(range(100), desc='2nd loop'):
        sleep(0.01)
# -

from tqdm import tqdm
for i in tqdm(range(10000000)):
    tmp = 42

assert False

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# +
# # Pickling data ###########################################
# out_dict = dict()
# out_dict["TEMP"] = None

# import os; import pickle
# path_i = os.path.join(
#     os.environ["HOME"],
#     "__temp__",
#     "temp.pickle")
# with open(path_i, "wb") as fle:
#     pickle.dump(out_dict, fle)
# # #########################################################
# -

# #########################################################
import pickle; import os
path_i = os.path.join(
    os.environ["HOME"],
    "__temp__",
    "temp.pickle")
with open(path_i, "rb") as fle:
    out_dict = pickle.load(fle)
# #########################################################

# +
struct = out_dict["struct"]

# dir(struct[0])

site = struct[-1]

# dir(site.species)

# site.oxi_state_guesses()
# site.species.oxi_state_guesses()


# dir(site.specie)
# type(site.specie)
# site.specie

# +
print("type(struct):", "\n", type(struct))

print("type(struct[-1]):", "\n", type(struct[-1]))

print("type(struct[-1].specie):", "\n", type(struct[-1].specie))
# -





# +
from pymatgen import Lattice, Structure, Molecule

coords = [[0, 0, 0], [0.75,0.5,0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120,
                                  beta=90, gamma=60)
struct = Structure(lattice, ["Si", "Si"], coords)

coords = [[0.000000, 0.000000, 0.000000],
          [0.000000, 0.000000, 1.089000],
          [1.026719, 0.000000, -0.363000],
          [-0.513360, -0.889165, -0.363000],
          [-0.513360, 0.889165, -0.363000]]
methane = Molecule(["C", "H", "H", "H", "H"], coords)
