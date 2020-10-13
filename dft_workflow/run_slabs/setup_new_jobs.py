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

# # Analyzing job sets (everything within a `02_attempt` dir for example)
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import dictdiffer
import json
import copy

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 120

from ase import io

# #########################################################
from methods import (
    get_df_jobs_data,
    get_df_jobs,
    get_df_jobs_paths,
    get_df_jobs_anal,
    cwd,
    )

from dft_workflow_methods import is_job_understandable, job_decision
from dft_workflow_methods import transfer_job_files_from_old_to_new
from dft_workflow_methods import is_job_compl_done
# -

# # Script Inputs

# +
# TEST_no_file_ops = True  # True if just testing around, False for production mode
TEST_no_file_ops = False

# Slac queue to submit to
slac_sub_queue = "suncat2"  # 'suncat', 'suncat2', 'suncat3'
# -

# # Read Data

df_jobs = get_df_jobs()
print("df_jobs.shape:", 2 * "\t", df_jobs.shape)
df_jobs_data = get_df_jobs_data(drop_cols=False)
print("df_jobs_data.shape:", 1 * "\t", df_jobs_data.shape)
df_jobs_paths = get_df_jobs_paths()

# +
df_jobs_anal = get_df_jobs_anal()

df_resubmit = df_jobs_anal
# -

data_root_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/dft_scripts/out_data")
data_path = os.path.join(data_root_path, "conservative_mixing_params.json")
with open(data_path, "r") as fle:
    dft_calc_sett_mix = json.load(fle)


# # Filter `df_resubmit` to only rows that are to be resubmitted

# +
df_resubmit_tmp = copy.deepcopy(df_resubmit)

# #########################################################
mask_list = []
for i in df_resubmit.decision.tolist():
    if "resubmit" in i:
        mask_list.append(True)
    else:
        mask_list.append(False)

df_resubmit = df_resubmit_tmp[mask_list]
df_nosubmit = df_resubmit_tmp[np.invert(mask_list)]

print("df_resubmit.shape:", df_resubmit.shape)
print("df_nosubmit.shape:", df_nosubmit.shape)

# +
df_i = df_nosubmit[df_nosubmit.job_completely_done == False]

# df_i[df_i.decision == []]
index_mask = []
for name_i, row_i in df_i.iterrows():
    decision_i = row_i.decision

    if len(decision_i) == 0:
        index_mask.append(name_i)
df_i = df_i.loc[index_mask]

if df_i.shape[0] > 0:
    print("There are jobs being left idle, nothing to do, fix it")
    print(df_i.job_id_max.tolist())
# -

# # Creating new job directories and initializing

# +
data_dict_list = []
for i_cnt, (name_i, row_i) in enumerate(df_resubmit.iterrows()):
    data_dict_i = dict()
    print(40 * "*")

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    job_id_max_i = row_i.job_id_max
    # compenv_i = row_i.compenv
    dft_params_new = row_i.dft_params_new
    # #####################################################

    # #####################################################
    df_jobs_i = df_jobs[df_jobs.compenv == compenv_i]
    row_jobs_i = df_jobs_i.loc[job_id_max_i]
    # #####################################################
    rev_num = row_jobs_i.rev_num
    # #####################################################

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
    row_paths_max_i = df_jobs_paths_i.loc[job_id_max_i]
    # #####################################################
    gdrive_path = row_paths_max_i.gdrive_path
    # #####################################################

    # #####################################################
    df_jobs_data_i = df_jobs_data[df_jobs_data.compenv == compenv_i]
    row_data_max_i = df_jobs_data_i.loc[job_id_max_i]
    # #####################################################
    num_steps = row_data_max_i.num_steps
    incar_params = row_data_max_i.incar_params
    # #####################################################


    path_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path)

    # #########################################################
    # Copy files to new job dir
    new_path_i = "/".join(path_i.split("/")[0:-1] + ["_" + str(rev_num + 1).zfill(2)])
    print(new_path_i)

    if not TEST_no_file_ops:
        if not os.path.exists(new_path_i):
            os.makedirs(new_path_i)

            
    files_to_transfer_for_new_job = [
        # ["contcar_out.traj", "init.traj"],
        [
            os.path.join(
                os.environ["PROJ_irox_oer"],
                "dft_workflow/dft_scripts/slab_dft.py"),
            "model.py",
            ],
        # "model.py",

        "WAVECAR",
        "dft-params.json",
        ["dir_dft_params/dft-params.json", "dft-params.json"],
        ]

    # #########################################################
    with cwd(path_i):
        if num_steps > 0:
            atoms = io.read("CONTCAR")
            atoms.write("contcar_out.traj")
            files_to_transfer_for_new_job.append(
                ["contcar_out.traj", "init.traj"])


        else:
            atoms = io.read("init.traj")
            files_to_transfer_for_new_job.append(
                "init.traj"
                # ["contcar_out.traj", "init.traj"]
                )

        # If spin-polarized calculation then get magmoms from prev. job and pass to new job
        if incar_params["ISPIN"] == 2:
            if num_steps > 0:
                atoms_outcar = io.read("OUTCAR")
                magmoms_i_tmp = atoms_outcar.get_magnetic_moments()

                data_path = os.path.join("out_data/magmoms_out.json")
                with open(data_path, "w") as outfile:
                    json.dump(magmoms_i_tmp.tolist(), outfile)

                files_to_transfer_for_new_job.append(
                    ["out_data/magmoms_out.json", "magmoms.json"])

        num_atoms = atoms.get_global_number_of_atoms()
        


    # #########################################################
    if not TEST_no_file_ops:
        transfer_job_files_from_old_to_new(
            path_i=path_i,
            new_path_i=new_path_i,
            files_to_transfer_for_new_job=files_to_transfer_for_new_job,
            )

    # #####################################################
    if not TEST_no_file_ops:
        dft_params_path_i = os.path.join(
            new_path_i,
            "dft-params.json")
        with open(dft_params_path_i, "r") as fle:
            dft_params_current = json.load(fle)

        # Update previous DFT parameters with new ones
        dft_params_current.update(dft_params_new)

        with open(dft_params_path_i, "w") as outfile:
            json.dump(dft_params_current, outfile, indent=2)

    # #####################################################
    data_dict_i["path_i"] = new_path_i
    data_dict_i["num_atoms"] = num_atoms
    data_dict_list.append(data_dict_i)

# #########################################################
df_sub = pd.DataFrame(data_dict_list)
# -

assert False

# + jupyter={"source_hidden": true}
# magmoms_i_tmp.to_list()
# magmoms_i_tmp.tolist()

# + jupyter={"source_hidden": true}
# df_resubmit.iloc[0:1]

# + jupyter={"source_hidden": true}
# row_i


# compenv_i = "sherlock"
# slab_id_i = "putarude_21"
# att_num_i = 1

# df_tmp = df_resubmit[
#     (df_resubmit.compenv == compenv_i) & \
#     (df_resubmit.slab_id == slab_id_i) & \
#     (df_resubmit.att_num == att_num_i) & \
#     [True for i in range(len(df_resubmit))]
#     ]

# print("job_id_max_i:", df_tmp.iloc[0].job_id_max)

# # Before when the notebook was breaking the job_id_max for this row was:
# # dunosagi_96

# + jupyter={"source_hidden": true}
# df_jobs_i = 
# df_jobs[df_jobs.compenv == compenv_i].loc[job_id_max_i]
# row_jobs_i = df_jobs_i.loc[job_id_max_i]
