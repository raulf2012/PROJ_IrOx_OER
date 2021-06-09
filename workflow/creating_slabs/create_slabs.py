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
# This notebook is time consuming. Additional processing of the slab (correct vacuum applied, and bulk constraints, etc.) are done in `process_slabs.ipynb`

# + active=""
# # 565 total polymorphs from first project
#
# # 122 polymorphs are octahedral and unique
# # >>> Removing 12 systems manually because they are not good
# # -----
# # 110 polymorphs now
#
#
# # # ###############################################
# # 49 are layered materials
# # 61 are non-layered materials
# # -----
# # 61 polymorphs now
#
#
# # # ###############################################
# # 15 polymorphs are above the 0.3 eV/atom above hull cutoff
# # -----
# # 46 polymorphs now
# -

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import time
import signal
import random
from pathlib import Path

from IPython.display import display

import pickle
import json

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import ase
from ase import io

# from tqdm import tqdm
from tqdm.notebook import tqdm

# #########################################################
from misc_modules.pandas_methods import drop_columns
from misc_modules.misc_methods import GetFriendlyID
from ase_modules.ase_methods import view_in_vesta

# #########################################################
from proj_data import metal_atom_symbol

from methods import (
    get_df_dft,
    symmetrize_atoms,
    get_structure_coord_df,
    remove_atoms,
    compare_facets_for_being_the_same,
    TimeoutException,
    sigalrm_handler,
    )

# #########################################################
from local_methods import (
    analyse_local_coord_env,
    check_if_sys_processed,
    remove_nonsaturated_surface_metal_atoms,
    remove_noncoord_oxygens,
    create_slab_from_bulk,
    create_final_slab_master,
    create_save_dataframe,
    constrain_slab,
    read_data_json,
    calc_surface_area,
    create_slab,
    update_sys_took_too_long,
    create_save_struct_coord_df,
    )


# -

# # Script Inputs

# +
# timelimit_seconds = 0.4 * 60
# timelimit_seconds = 10 * 60
# timelimit_seconds = 40 * 60
timelimit_seconds = 100 * 60

facets_manual = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),

    (1, 1, 1),

    # (0, 1, 1),
    # (1, 0, 1),
    # (1, 1, 0),

    ]
facets_manual = [t for t in (set(tuple(i) for i in facets_manual))]

frac_of_layered_to_include = 0.0

phase_num = 2

# +
# max_surf_a = 200
# Distance from top z-coord of slab that we'll remove atoms from
# dz = 4
# -

# # Read Data

# +
# #########################################################
df_dft = get_df_dft()

# #########################################################
# Bulks not to run, manually checked to be erroneous/bad
data_path = os.path.join(
    "in_data/bulks_to_not_run.json")
with open(data_path, "r") as fle:
    bulks_to_not_run = json.load(fle)

# #########################################################
from methods import get_df_xrd
df_xrd = get_df_xrd()

# #########################################################
from methods import get_df_bulk_manual_class
df_bulk_manual_class = get_df_bulk_manual_class()

# #########################################################
from methods import get_bulk_selection_data
bulk_selection_data = get_bulk_selection_data()
bulk_ids__octa_unique = bulk_selection_data["bulk_ids__octa_unique"]

# #########################################################
from methods import get_df_slab_ids, get_slab_id
df_slab_ids = get_df_slab_ids()

# #########################################################
from methods import get_df_slab
df_slab_old = get_df_slab(mode="almost-final")

# #########################################################
from local_methods import df_dft_for_slab_creation
df_dft_i = df_dft_for_slab_creation(
    df_dft=df_dft,
    bulk_ids__octa_unique=bulk_ids__octa_unique,
    bulks_to_not_run=bulks_to_not_run,
    df_bulk_manual_class=df_bulk_manual_class,
    frac_of_layered_to_include=frac_of_layered_to_include,
    verbose=False,
    )

# +
# get_bulk_selection_data().keys()

# +
# assert False

# +
# TEMP

# mj7wbfb5nt	011	(0, 1, 1)	

df = df_slab_old
df = df[
    (df["bulk_id"] == "mj7wbfb5nt") &
    (df["facet"] == "011") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

# +
# assert False
# -

# # Create needed folders

# +
root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    )

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

# ### Checking that df_slab_ids are unique, no repeat entries

# +
if not df_slab_ids.index.is_unique:
    print("df_slab_ids isn't unique")
    print("df_slab_ids isn't unique")
    print("df_slab_ids isn't unique")
    print("df_slab_ids isn't unique")
    print("df_slab_ids isn't unique")

print("Duplicate rows here (NOT GOOD!!!)")
display(
    df_slab_ids[df_slab_ids.index.duplicated(keep=False)]
    )

df = df_slab_old
df = df[
    (df["bulk_id"] == "v1xpx482ba") &
    (df["facet"] == "20-21") &
    # (df["facet"] == "20-23") &
    [True for i in range(len(df))]
    ]
df
# -

# ## Removing duplicate rows

# +
# #########################################################
slab_ids_to_drop = []
# #########################################################
group_cols = ["bulk_id", "facet", ]
grouped = df_slab_old.groupby(group_cols)
for name_i, group_i in grouped:
    if group_i.shape[0] > 1:

        # print(name_i)
        # display(group_i)

        # name_i = ('xw9y6rbkxr', '10-12')
        # group_i = grouped.get_group(name_i)

        grp_0 = group_i[group_i.status == "Took too long"]
        grp_1 = group_i[~group_i.slab_final.isna()]

        if grp_1.shape[0] > 0:
            if grp_0.shape[0] > 0:
                slab_ids_to_drop_i = grp_0.index.tolist()
                slab_ids_to_drop.extend(slab_ids_to_drop_i)

# df_slab_old.loc[slab_ids_to_drop]
df_slab_old = df_slab_old.drop(slab_ids_to_drop)

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

print(
    len(systems_that_took_too_long),
    " systems took too long to process and will be ignored",
    sep="")

# +
df_slab_old_tmp = df_slab_old.reset_index(level=0, inplace=False)
df_slab_old_tmp = df_slab_old_tmp.set_index(["bulk_id", "facet", ], drop=False, )

# # df_slab_old.set_index?
# -

print(
    "This was True before, look into it if it's not",
    "\n",

    "\n",
    "df_slab_old_tmp.index.is_unique:",

    "\n",
    df_slab_old_tmp.index.is_unique,

    sep="")

# +
systems_that_took_too_long__new = []
for sys_i in systems_that_took_too_long:
    # print(sys_i)

    atoms_found = False
    name_i = (sys_i[0], sys_i[1])
    if name_i in df_slab_old_tmp.index:
        # #####################################################
        row_i = df_slab_old_tmp.loc[sys_i[0], sys_i[1]]
        # #####################################################
        slab_final_i = row_i.slab_final
        # #####################################################

        if isinstance(slab_final_i, ase.atoms.Atoms):
            atoms_found = True
    else:
        tmp = 42

    keep_sys_in_list = True
    if atoms_found:
        keep_sys_in_list = False

    if keep_sys_in_list:
        systems_that_took_too_long__new.append(sys_i)



# ##########################################################
# ##########################################################
data = read_data_json()
data["systems_that_took_too_long"] = systems_that_took_too_long__new

data_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data/data.json")
with open(data_path, "w") as fle:
    json.dump(data, fle, indent=2)
# -

len(systems_that_took_too_long__new)

len(systems_that_took_too_long)

# +
# assert False

# +
# assert False

# +
# df_slab_old[df_slab_old.bulk_id == "n36axdbw65"]
# -

# ## Figuring out which systems haven't been run yet

# +
# #########################################################
data_dict_list = []
# #########################################################
systems_not_processed = []
# #########################################################
for i_cnt, bulk_id in enumerate(df_dft_i.index.tolist()):

    # #####################################################
    row_i = df_dft.loc[bulk_id]
    # #####################################################
    bulk_id_i = row_i.name
    atoms = row_i.atoms
    # #####################################################

    # #####################################################
    row_xrd_i = df_xrd.loc[bulk_id]
    # #####################################################
    top_facets_i = row_xrd_i.top_facets
    all_xrd_facets_i = row_xrd_i.all_xrd_facets
    facet_rank_i = row_xrd_i.facet_rank
    # #####################################################

    num_of_facets = 5
    # num_of_facets = 8
    top_facets_i = top_facets_i[0:num_of_facets]
    facet_rank_i = facet_rank_i[0:num_of_facets]

    # #####################################################
    # Facet manipulation ##################################
    facets_manual_2 = []
    for i in facets_manual:
        if i not in all_xrd_facets_i:
            facets_manual_2.append(i)

    df_facets_0 = pd.DataFrame()
    df_facets_0["facet"] = top_facets_i
    df_facets_0["facet_rank"] = facet_rank_i
    df_facets_0["source"] = "xrd"

    df_facets_1 = pd.DataFrame()
    df_facets_1["facet"] = facets_manual_2
    df_facets_1["source"] = "manual"

    df_facets = pd.concat([df_facets_0, df_facets_1])
    df_facets = df_facets.reset_index()
    # #####################################################


    # #####################################################
    # Making sure that there are no duplicates in the facets from the manual ones and xrd ones
    # #####################################################
    df_facets_i = df_facets[df_facets.source == "manual"]
    df_facets_j = df_facets[df_facets.source == "xrd"]
    # #####################################################
    indices_to_drop = []
    # #####################################################
    for ind_i, row_i in df_facets_i.iterrows():
        facet_i = row_i.facet
        for ind_j, row_j in df_facets_j.iterrows():
            facet_j = row_j.facet
            facets_same = compare_facets_for_being_the_same(facet_i, facet_j)
            if facets_same:
                indices_to_drop.append(ind_i)
    df_facets = df_facets.drop(index=indices_to_drop)


    for ind_i, row_facet_i in df_facets.iterrows():
        # #################################################
        data_dict_i = dict()
        # #################################################

        # #################################################
        facet = row_facet_i.facet
        source_i = row_facet_i.source
        facet_rank_i = row_facet_i.facet_rank
        # #################################################

        facet_i = "".join([str(i) for i in list(facet)])

        facet_abs_sum_i = np.sum(
            [np.abs(i) for i in facet]
            )

        sys_processed = check_if_sys_processed(
            bulk_id_i=bulk_id_i,
            facet_str=facet_i,
            df_slab_old=df_slab_old)

        id_comb = bulk_id + "_" + facet_i

        took_too_long_prev = False
        if id_comb in systems_that_took_too_long_2:
            took_too_long_prev = True

        # #################################################
        data_dict_i["bulk_id"] = bulk_id_i
        data_dict_i["facet_str"] = facet_i
        data_dict_i["facet"] = facet
        data_dict_i["facet_rank"] = facet_rank_i
        data_dict_i["facet_abs_sum"] = facet_abs_sum_i
        data_dict_i["source"] = source_i
        data_dict_i["sys_processed"] = sys_processed
        data_dict_i["took_too_long_prev"] = took_too_long_prev
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# #########################################################
df_to_run = pd.DataFrame(data_dict_list)
# #########################################################
# -

df_dft_i.loc[
    "v1xpx482ba"
    ]

df_to_run

df = df_to_run
df = df[
    (df["bulk_id"] == "v1xpx482ba") &
    (df["facet_str"] == "20-23") &
    # (df[""] == "") &
    [True for i in range(len(df))]
    ]
df

# +
# assert False
# -

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slabs_to_run.pickle"), "wb") as fle:
    df_slabs_to_run = df_to_run
    pickle.dump(df_slabs_to_run, fle)
# #########################################################

df_to_run_2 = df_to_run[
    (df_to_run.sys_processed == False) & \
    (df_to_run.took_too_long_prev == False) & \
    (df_to_run.facet_abs_sum <= 7)
    ]

# +
# mj7wbfb5nt	011	(0, 1, 1)	

df_to_run_2
# -

assert False

# +
# df_to_run_2 = df_to_run_2.loc[[164]]
# -

assert False

# +
# df_to_run_2 = df_to_run[
#     # (df_to_run.sys_processed == False) & \
#     (df_to_run.took_too_long_prev == True)
#     # (df_to_run.facet_abs_sum < 7)
#     ]

# df_to_run_2 = df_to_run_2.iloc[[0]]

# + active=""
#
#

# +
# df_slabs_to_run

# +
# df_to_run[df_to_run.bulk_id == "b583vr8hvw"]

# +
# b583vr8hvw 110

# +
# df_to_run_2

# +
# 	bulk_id	facet_str	facet	facet_rank	facet_abs_sum	source	sys_processed	took_too_long_prev
# 211	9i6ixublcr	31-3	(3, 1, -3)	1.0	7	xrd	True	True

# +
# assert False

# +
# for i in df_to_run_2.bulk_id.unique().tolist():
#     print(
#         i in df_dft_i.index
#         )
# -

assert False

# # Main Loop | Creating slabs

for i_cnt, (ind_i, row_i) in enumerate(df_to_run_2.iterrows()):
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    bulk_id_i = row_i.bulk_id
    facet = row_i.facet
    facet_rank_i = row_i.facet_rank
    source_i = row_i.source
    # #####################################################

    # #####################################################
    row_dft_i = df_dft.loc[bulk_id_i]
    # #####################################################
    atoms_stan_prim_i = row_dft_i.atoms_stan_prim
    # #####################################################

    # #####################################################
    # Set up signal handler for SIGALRM, saving previous value
    t0 = time.time()
    old_handler = signal.signal(signal.SIGALRM, sigalrm_handler)
    signal.alarm(int(timelimit_seconds))
    # #####################################################

    # #####################################################
    facet_i = "".join([str(i) for i in list(facet)])

    # #####################################################
    # Getting or generating id for slab (slab_id)
    slab_id_i = get_slab_id(bulk_id_i, facet_i, df_slab_ids)
    if slab_id_i is None:
        slab_id_i = GetFriendlyID(append_random_num=True)














    # #####################################################
    data_dict_i["bulk_id"] = bulk_id_i
    data_dict_i["facet"] = facet_i
    data_dict_i["facet_rank"] = facet_rank_i
    data_dict_i["source"] = source_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["phase"] = phase_num
    # #####################################################

    print(
        "bulk_id_i:", bulk_id_i,
        "slab_id_i:", slab_id_i,
        "facet:", facet_i,
        )

    try:
        slab_final = create_slab(
            atoms=atoms_stan_prim_i,
            facet=facet,
            # slab_thickness=15,
            slab_thickness=12,
            i_cnt=i_cnt)

        create_save_struct_coord_df(
            slab_final=slab_final,
            slab_id=slab_id_i)

        data_dict_i["slab_final"] = slab_final

    except TimeoutException:
        print("Took to long skipping")
        data_dict_i["status"] = "Took too long"

        # Updating systems_that_took_too_long if bulk_id+facet combo doesn't finish in time
        update_sys_took_too_long(bulk_id_i, facet_i)

    finally:
        # #################################################
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    # #####################################################
    iter_time_i = time.time() - t0
    data_dict_i["iter_time_i"] = iter_time_i

    df_slab_old = create_save_dataframe(
        data_dict_list=[data_dict_i],
        df_slab_old=df_slab_old)

# +
# slab_id_i = get_slab_id(bulk_id_i, facet_i, df_slab_ids)
# if slab_id_i is None:
#     slab_id_i = GetFriendlyID(append_random_num=True)
# -

slab_id_i

# +
# bulk_id_i: 9i6ixublcr slab_id_i: kogituwu_25 facet: 31-3
# -

data_dict_i

# + active=""
#
#
#
#

# + jupyter={"source_hidden": true}
# import ase

# import pickle
