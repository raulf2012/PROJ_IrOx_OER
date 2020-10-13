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

# # Creating slabs from IrOx polymorph dataset
# ---
#
# This notebook is time consuming. Additional processing of the slab (correct vacuum applied, and bulk constraints, etc.) are done in `process_slabs.ipynb`

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import time
import signal
import random
from pathlib import Path

import pickle
import json

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from ase import io

# from tqdm import tqdm
from tqdm.notebook import tqdm

# #########################################################
from misc_modules.pandas_methods import drop_columns
from misc_modules.misc_methods import GetFriendlyID
from ase_modules.ase_methods import view_in_vesta

# #########################################################
from methods import (
    get_df_dft, symmetrize_atoms,
    get_structure_coord_df, remove_atoms)
from methods import TimeoutException, sigalrm_handler

from proj_data import metal_atom_symbol

# #########################################################
from local_methods import (
    analyse_local_coord_env, check_if_sys_processed,
    remove_nonsaturated_surface_metal_atoms,
    remove_noncoord_oxygens,
    create_slab_from_bulk,
    create_final_slab_master,
    create_save_dataframe,
    constrain_slab,
    read_data_json,
    calc_surface_area,
    )


# -

# # Script Inputs

# +
# timelimit_seconds = 0.4 * 60
timelimit_seconds = 10 * 60

max_surf_a = 200

# Distance from top z-coord of slab that we'll remove atoms from
dz = 4

facets = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),

    (1, 1, 1),

    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),

    # (2, 0, 0),
    # (0, 2, 0),
    # (0, 0, 2),

    # Weird cuts
    (3, 1, 4),
    (2, 1, 4),

    (3, 0, 2),
    (3, 0, 3),

    (0, 1, 2),
    (1, 3, 1),

    (3, 3, 1),

    ]

facets = [t for t in (set(tuple(i) for i in facets))]
# -

frac_of_layered_to_include = 0.1

# # Read Data

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
from methods import get_df_xrd

df_xrd = get_df_xrd()
# df_xrd

# +
from methods import get_df_bulk_manual_class

df_bulk_manual_class = get_df_bulk_manual_class()

# df_bulk_manual_class

# +
from methods import get_bulk_selection_data

# #########################################################
bulk_selection_data = get_bulk_selection_data()
# #########################################################
# bulk_selection_data.keys()
bulk_ids__octa_unique = bulk_selection_data["bulk_ids__octa_unique"]
# #########################################################
# -

# # Create needed folders

# +
directory = "out_data/final_slabs"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "out_data/slab_progression"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "out_data/df_coord_files"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = "out_data/temp_out"
if not os.path.exists(directory):
    os.makedirs(directory)
# -

print(
    "Number of bulk structures that are octahedral and unique:",
    "\n",
    len(bulk_ids__octa_unique))

# # Filtering bulk DFT data

# ## Take only unique octahedral systems

# +
df_dft_i = df_dft.loc[bulk_ids__octa_unique]

# Drop ids that were manually identified as bad
ids_i = df_dft_i.index.intersection(bulks_to_not_run)
df_dft_i = df_dft_i.drop(labels=ids_i)

print("df_dft_i.shape:", df_dft_i.shape[0])
# df_dft_i = df_dft_i.sample(n=50)
# df_dft_i = df_dft_i.sample(n=110)
# df_dft_i = df_dft_i.sample(n=20)
# -

# ## Removing all layered bulks and adding in just a bit

# +
non_layered_ids = df_bulk_manual_class[df_bulk_manual_class.layered == False]
non_layered_ids = non_layered_ids.index

layered_ids = df_bulk_manual_class[df_bulk_manual_class.layered == True]
layered_ids = layered_ids.index

print(df_dft_i.index.shape[0])
print(non_layered_ids.shape[0])

non_layered_ids_inter = df_dft_i.index.intersection(non_layered_ids)
# df_dft_i = df_dft_i.loc[non_layered_ids_inter]

num_non_layered = non_layered_ids_inter.shape[0]

num_layered_to_inc = int(frac_of_layered_to_include * num_non_layered)

layered_ids_to_use = np.random.choice(
    layered_ids, size=num_layered_to_inc, replace=False)

print(len(layered_ids_to_use))
print(len(non_layered_ids_inter))

non_layered_ids_inter.join(layered_ids_to_use).shape

all_ids = non_layered_ids_inter.to_list() + list(layered_ids_to_use)

# df_dft_i = df_dft_i.loc[all_ids]

df_dft_i = df_dft_i.loc[df_dft_i.index.intersection(all_ids)]

# +
# all_ids

# df_dft_i = 
# df_dft_i.loc[all_ids]

# all_ids
# -

# # Impose `dH-dH_hull` cutoff of 0.5

# +
# #########################################################
df_dft_ab2 = df_dft_i[df_dft_i["stoich"] == "AB2"]

min_dH = df_dft_ab2.dH.min()
df_dft_ab2.loc[:, "dH_hull"] = df_dft_ab2.dH - min_dH

# #########################################################
df_dft_ab3 = df_dft_i[df_dft_i["stoich"] == "AB3"]

min_dH = df_dft_ab3.dH.min()
df_dft_ab3.loc[:, "dH_hull"] = df_dft_ab3.dH - min_dH

# #########################################################
print(df_dft_ab2.shape)
print(df_dft_ab3.shape)

df_dft_i = pd.concat([
    df_dft_ab2,
    df_dft_ab3,
    ], )

df_dft_i = df_dft_i[df_dft_i.dH_hull < 0.5]

# +
# assert False

# +
# df_dft_i.shape

# +
# TEMP
# df_dft_i = df_dft_i.loc[["xtbocq9o6p"]]

# +
# assert False
# -

# # Creating slabs from bulks

# ## Which systems previously took too long

# +
data = read_data_json()

systems_that_took_too_long = data.get("systems_that_took_too_long", []) 

systems_that_took_too_long_2 = []
for i in systems_that_took_too_long:
    systems_that_took_too_long_2.append(i[0] + "_" + i[1])
# -

# ## Figuring out which systems haven't been run yet

systems_not_processed = []
iterator = df_dft_i.index.tolist()
for i_cnt, bulk_id in enumerate(iterator):
    row_i = df_dft.loc[bulk_id]

    # #####################################################
    # Row parameters ######################################
    bulk_id_i = row_i.name
    atoms = row_i.atoms
    # #####################################################

    # #####################################################
    row_xrd_i = df_xrd.loc[bulk_id]
    # #####################################################
    top_facets_i = row_xrd_i.top_facets
    # #####################################################

    # for facet in facets:
    for facet in top_facets_i:
        data_dict_i = dict()

        data_dict_i["bulk_id"] = bulk_id_i
    
        facet_i = "".join([str(i) for i in list(facet)])
        data_dict_i["facet"] = facet_i

        sys_processed = check_if_sys_processed(
            bulk_id_i=bulk_id_i,
            facet_str=facet_i,
            df_slab_old=df_slab_old)

        if not sys_processed:
            id_comb = bulk_id + "_" + facet_i
            if id_comb not in systems_that_took_too_long_2:
                systems_not_processed.append(dict(bulk_id=bulk_id, facet=facet))

# +
# systems_not_processed = random.sample(systems_not_processed, 20)

# systems_not_processed = random.sample(systems_not_processed, 2)
# -

# # TEMP | Manually creating bulk_id + facet list to process

# +
# #########################################################
# # TEMP
# print("TEMP TEMP TEMP")
# systems_not_processed = [
#     {"bulk_id": "8l919k6s7p", "facet": (1, 0, 0)},
#     {"bulk_id": "8l919k6s7p", "facet": (1, 1, 1)},
#     {"bulk_id": "8l919k6s7p", "facet": (1, 0, 1)},
#     {"bulk_id": "8l919k6s7p", "facet": (0, 0, 1)},
#     {"bulk_id": "8l919k6s7p", "facet": (1, 1, 0)},
#     ]

# #########################################################
# systems_not_processed = [
#     {'bulk_id': 'b19q9p6k72', 'facet': (1, 0, 1)},
#     ]
# -

# # TEMP |  Creating `systems_to_process` from already run jobs

# +
# from methods import get_df_jobs

# df_jobs = get_df_jobs(exclude_wsl_paths=True)

# systems_to_process = []
# for job_id, row_i in df_jobs[["bulk_id", "facet"]].iterrows():
#     bulk_id_i = row_i.bulk_id
#     facet_i = row_i.facet

#     dict_out_i = {
#         "bulk_id": bulk_id_i,
#         "facet": facet_i}
#     systems_to_process.append(dict_out_i)

# +
systems_to_process = systems_not_processed

systems_to_process
# -

assert False

# ## Creating slabs

df_coord_dict = dict()
# iterator = tqdm(systems_not_processed, desc="1st loop")
iterator = tqdm(systems_to_process, desc="1st loop")
for i_cnt, sys_i in enumerate(iterator):

    # #####################################################
    # Set up signal handler for SIGALRM, saving previous value
    old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
    # Start timer
    signal.alarm(int(timelimit_seconds))
    # #####################################################

    data_dict_i = dict()
    t0 = time.time()

    # #####################################################
    bulk_id_i = sys_i["bulk_id"]
    data_dict_i["bulk_id"] = bulk_id_i

    facet = sys_i["facet"]
    facet_i = "".join([str(i) for i in list(facet)])
    data_dict_i["facet"] = facet_i
    # #####################################################
    row_i = df_dft.loc[bulk_id_i]
    atoms = row_i.atoms
    # #####################################################

    # #####################################################
    # Getting or generating id for slab (slab_id)
    from methods import get_df_slab_ids, get_slab_id
    df_slab_ids = get_df_slab_ids()
    slab_id_i = get_slab_id(bulk_id_i, facet_i, df_slab_ids)
    if slab_id_i is None:
        slab_id_i = GetFriendlyID(append_random_num=True)

    data_dict_i["slab_id"] = slab_id_i

    print("bulk_id_i:", bulk_id_i, "facet", facet_i, end="\r", flush=True)

    surf_a = calc_surface_area(atoms=atoms)
    if surf_a > max_surf_a:
        data_dict_i["status"] = "Too large of surface area"
    else:

        try:
            # slab_final = create_slab_from_bulk(
            slab_0 = create_slab_from_bulk(
                atoms=atoms, facet=facet)

            slab_1 = create_final_slab_master(atoms=slab_0)

            slab_2 = constrain_slab(atoms=slab_1)
            slab_final = slab_2

            df_coord_slab_final = get_structure_coord_df(slab_final)
    
            # COMBAK
            # Pickling data ###########################################
            path_i = os.path.join(
                "out_data/df_coord_files", slab_id_i + ".pickle")
            with open(path_i, "wb") as fle:
                pickle.dump(df_coord_slab_final, fle)
            # #########################################################

            df_coord_dict[slab_id_i] = df_coord_slab_final

            file_name_i = bulk_id_i + "_" + slab_id_i + \
                "_" + facet_i + "_final" + ".cif"
            slab_final.write(
                os.path.join("out_data/final_slabs", file_name_i))

            # #####################################################
            data_dict_i["slab_final"] = slab_final

        except TimeoutException:
            data_dict_i["status"] = "Took too long"

            data = read_data_json()

            systems_that_took_too_long = data.get("systems_that_took_too_long", [])
            systems_that_took_too_long.append((bulk_id_i, facet_i))

            data["systems_that_took_too_long"] = systems_that_took_too_long

            data_path = os.path.join(
                "out_data/data.json")
            with open(data_path, "w") as fle:
                json.dump(data, fle, indent=2)


        finally:
            # #################################################
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    # #####################################################
    iter_time_i = time.time() - t0
    data_dict_i["iter_time_i"] = iter_time_i

    data_dict_list = []
    data_dict_list.append(data_dict_i)

    df_slab_old = create_save_dataframe(
        data_dict_list=data_dict_list,
        df_slab_old=df_slab_old)

# +
# df_slab_old

# +
from methods import get_df_slab

df_slab = get_df_slab()

# df_slab.shape: (136, 13)
print("df_slab.shape:", df_slab.shape)
print("")

df_slab[df_slab.bulk_id == "xtbocq9o6p"]

# + active=""
#
#
#
#

# + jupyter={"source_hidden": true}
# def get_bulk_selection_data():
#     """
#     """
#     #| - get_bulk_selection_data

#     # ########################################################
#     data_path = os.path.join(
#         os.environ["PROJ_irox_oer"],
#         "workflow/creating_slabs/selecting_bulks",
#         "out_data/data.json")
#     with open(data_path, "r") as fle:
#         data = json.load(fle)
#     # ########################################################

#     # bulk_ids__octa_unique = data["bulk_ids__octa_unique"]

#     return(data)
#     #__|

# + jupyter={"source_hidden": true}
# # ########################################################
# data_path = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/creating_slabs/selecting_bulks",
#     "out_data/data.json")
# with open(data_path, "r") as fle:
#     data = json.load(fle)
# # ########################################################

# bulk_ids__octa_unique = data["bulk_ids__octa_unique"]

# + jupyter={"source_hidden": true}
# print("bulk_id_i:", bulk_id_i, "facet", facet_i, end="\r", flush=True)

# facet_i

# # create_save_dataframe?

# df_slab_old[df_slab_old.bulk_id == "8l919k6s7p"]

# + jupyter={"source_hidden": true}
# iterator

# for i_cnt, bulk_id in enumerate(iterator):

# bulk_id

# systems_not_processed

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# # #####################################################
# row_xrd_i = df_xrd.loc[bulk_id]
# # #####################################################
# top_facets_i = row_xrd_i.top_facets
# # #####################################################

# top_facets_i
# # facets

# + jupyter={"source_hidden": true}
# top_facets_i
