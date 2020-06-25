# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Creating slabs from IrOx polymorph dataset
# ---
#
# This notebook is time consuming, 

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

from pathlib import Path
import pickle
import time
import signal
import json

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
    # (2, 1, 4),
    # (7, 1, 4),
    ]

facets = [t for t in (set(tuple(i) for i in facets))]
# -

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

# +
df_i = df_dft.loc[bulk_ids__octa_unique]

# Drop ids that were manually identified as bad
ids_i = df_i.index.intersection(bulks_to_not_run)
df_i = df_i.drop(labels=ids_i)

df_i = df_i.sample(n=15)
# -

# # Creating slabs from bulks

# ## Figuring out which systems haven't been run yet

# +
data = read_data_json()

systems_that_took_too_long = data.get("systems_that_took_too_long", []) 

systems_that_took_too_long_2 = []
for i in systems_that_took_too_long:
    systems_that_took_too_long_2.append(i[0] + "_" + i[1])
# -

systems_not_processed = []
iterator = df_i.index.tolist()
for i_cnt, bulk_id in enumerate(iterator):
    row_i = df_dft.loc[bulk_id]

    # #####################################################
    # Row parameters ######################################
    bulk_id_i = row_i.name
    atoms = row_i.atoms
    # #####################################################

    for facet in facets:
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


# ## Creating slabs

# +
# Custom exception for the timeout
class TimeoutException(Exception):
    pass

# Handler function to be called when SIGALRM is received
def sigalrm_handler(signum, frame):
    # We get signal!
    raise TimeoutException()


# -

df_coord_dict = dict()
iterator = tqdm(systems_not_processed, desc="1st loop")
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

# + active=""
#
#
#
#

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}


# def read_data_json():
#     """
#     """
#     #| - read_data_json
#     path_i = os.path.join(
#         "out_data", "data.json")
#     my_file = Path(path_i)
#     if my_file.is_file():
#         data_path = os.path.join(
#             "out_data/data.json")
#         with open(data_path, "r") as fle:
#             data = json.load(fle)
#     else:
#         data = dict()

#     return(data)
#     #__|

# + jupyter={"source_hidden": true}
# # df_dft_ab2_i = df_dft[df_dft.stoich == "AB2"].sort_values("dH").iloc[0:10]
# # df_dft_ab3_i = df_dft[df_dft.stoich == "AB3"].sort_values("dH").iloc[0:10]

# df_dft_ab2_i = df_dft[df_dft.stoich == "AB2"].sort_values("dH").iloc[0:100]
# df_dft_ab3_i = df_dft[df_dft.stoich == "AB3"].sort_values("dH").iloc[0:75]

# df_i = pd.concat([
#     df_dft_ab2_i.sample(n=20),
#     df_dft_ab3_i.sample(n=20),
#     ])
