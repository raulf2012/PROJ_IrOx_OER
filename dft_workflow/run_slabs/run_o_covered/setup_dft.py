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

import json
import pickle
from shutil import copyfile

import numpy as np
import pandas as pd

from ase import io

# # from tqdm import tqdm
from tqdm.notebook import tqdm

# # # #########################################################
from methods import (
    get_df_slab,
    get_df_jobs,
    )

from proj_data import metal_atom_symbol

# # #########################################################
# from local_methods import (
#     # mean_O_metal_coord,
#     # calc_wall_time,
#     )

from dft_workflow_methods import (
    get_job_spec_dft_params,
    get_job_spec_scheduler_params,
    submit_job,
    calc_wall_time)
# -

# # Script Inputs

# +
# Slac queue to submit to
slac_sub_queue = "suncat2"  # 'suncat', 'suncat2', 'suncat3'

# TEMP
dft_batch_size = 3
# -

# # Read Data

# +
# #########################################################
df_slab = get_df_slab()
df_slab = df_slab.set_index("slab_id")

# #########################################################
df_jobs = get_df_jobs()
# -

# # Setup

# +
directory = "out_data/dft_jobs"
if not os.path.exists(directory):
    os.makedirs(directory)

compenv = os.environ["COMPENV"]
# -

# # Selecting Slabs to Run

# +
df_slab = df_slab[~df_slab.index.isin(df_jobs.slab_id.tolist())]

# #########################################################
# Selecting smallest slabs
df_slab = df_slab.loc[
    df_slab.num_atoms.sort_values()[0:dft_batch_size].index
    ]

# #########################################################
# df_slab = df_slab[df_slab.num_atoms < 40]
# # df_slab = df_slab[(df_slab.num_atoms > 50) & (df_slab.num_atoms < 80)]

# #########################################################
# df_slab = df_slab.sample(n=dft_batch_size)
# -

df_slab

# # Setting up the job folders

# +
data_dict_list = []
for i_cnt, row_i in df_slab.iterrows():
    data_dict_i = dict()

    # #####################################################
    slab_id = row_i.name
    bulk_id = row_i.bulk_id
    facet = row_i.facet
    slab_final = row_i.slab_final
    num_atoms = row_i.num_atoms
    loop_time = row_i.loop_time
    iter_time_i = row_i.iter_time_i
    # #####################################################

    attempt = 1
    rev = 1

    path_i = os.path.join(
        "out_data",
        "dft_jobs",
        bulk_id,
        facet,
        str(attempt).zfill(2) + "_attempt",
        "_" + str(rev).zfill(2)
        )
    if not os.path.exists(path_i):
        os.makedirs(path_i)


    # #####################################################
    # Copy dft script to job folder
    # #####################################################
    copyfile(
        os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/dft_scripts/slab_dft.py"
            ),
        os.path.join(
            path_i,
            "model.py",
            ),
        )

    copyfile(
        os.path.join(
            os.environ["PROJ_irox_oer"],
            "dft_workflow/dft_scripts/slab_dft.py"
            ),
        os.path.join(
            path_i,
            "slab_dft.py",
            ),
        )

    # #####################################################
    # Copy atoms object to job folder
    # #####################################################
    slab_final.write(
        os.path.join(path_i, "init.traj")
        )

    data_dict_i["slab_id"] = slab_id
    data_dict_i["bulk_id"] = bulk_id
    data_dict_i["facet"] = facet
    data_dict_i["slab_final"] = slab_final
    data_dict_i["num_atoms"] = num_atoms
    data_dict_i["attempt"] = attempt
    data_dict_i["rev"] = rev
    data_dict_i["path_i"] = path_i


    data_dict_list.append(data_dict_i)

df_jobs_new = pd.DataFrame(data_dict_list)
df_jobs_new = df_jobs_new.set_index("slab_id")

# +
# df_jobs_new

# slab_id
# bulk_id
# facet
# slab_final
# num_atoms
# attempt
# rev
# path_i
# -

# # Assigning job specific DFT parameters

# +
data_dict_list = []
for i_cnt, row_i in df_jobs_new.iterrows():
    data_dict_i = dict()
    # #####################################################
    slab_id = row_i.name
    num_atoms = row_i.num_atoms
    path_i =row_i.path_i
    # #####################################################

    dft_params_dict = get_job_spec_dft_params(
        compenv=compenv,
        slac_sub_queue="suncat2",
        )

    data_dict_i["slab_id"] = slab_id
    data_dict_i["dft_params"] = dft_params_dict

    data_dict_list.append(data_dict_i)

df_dft_params = pd.DataFrame(data_dict_list)
df_dft_params = df_dft_params.set_index("slab_id")



# #########################################################
# Writing DFT params to job directory
for slab_id, row_i in df_dft_params.iterrows():

    # #####################################################
    dft_params = row_i.dft_params
    # #####################################################
    row_slab_i = df_jobs_new.loc[slab_id]
    path_i = row_slab_i.path_i
    # #####################################################

    with open(os.path.join(path_i, "dft-params.json"), "w+") as fle:
        # json.dump(dft_params_dict, fle, indent=2, skipkeys=True)
        json.dump(dft_params, fle, indent=2, skipkeys=True)
# -

# # Setting initial magnetic moments

data_dict_list = []
for i_cnt, row_i in df_jobs_new.iterrows():
    # #####################################################
    atoms = row_i.slab_final
    path_i =row_i.path_i
    # #####################################################

    O_magmom=0.2
    M_magmom=1.2
    magmoms_i = []
    for atom in atoms:
        if atom.symbol == "O":
            magmom_i = O_magmom
        else:
            magmom_i = M_magmom
        magmoms_i.append(magmom_i)

    data_path = os.path.join(path_i, "magmoms.json")
    with open(data_path, "w") as outfile:
        json.dump(magmoms_i, outfile, indent=2)

print("Paths of new jobs:")
tmp = [print(i) for i in df_jobs_new.path_i.tolist()]

# #########################################################
print(20 * "# # ")
print("All done!")
print("setup_dft.ipynb")
print(20 * "# # ")
assert False
# #########################################################

# +
# Some messages for user

# print("")
# print("Manually change if statement to True to submit DFT jobs")
# print("    search for submit_job(")
# print("")

# +
# Submit jobs

# out_dict = get_job_spec_scheduler_params(compenv=compenv)
# wall_time_factor = out_dict["wall_time_factor"]

# for i_cnt, row_i in df_jobs_new.iterrows():
#     # #######################################
#     num_atoms = row_i.num_atoms
#     path_i =row_i.path_i
#     # #######################################

#     if False:
#         submit_job(
#             path_i=path_i,
#             num_atoms=num_atoms,
#             wall_time_factor=wall_time_factor,
#             queue=slac_sub_queue,
#             )
