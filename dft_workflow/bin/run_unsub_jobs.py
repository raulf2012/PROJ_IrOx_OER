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

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import random
from pathlib import Path

import pandas as pd

from ase import io

# #########################################################
from dft_workflow_methods import get_path_rel_to_proj
from dft_workflow_methods import get_job_paths_info
from dft_workflow_methods import get_job_spec_dft_params, get_job_spec_scheduler_params
from dft_workflow_methods import submit_job
# -

# # Script Inputs

# +
root_dir = os.getcwd()

compenv = os.environ["COMPENV"]

if compenv == "wsl":
    root_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow/run_slabs/run_o_covered")    

slac_sub_queue = "suncat3"  # 'suncat', 'suncat2', 'suncat3'

# +
# sys.argv = [
#     '/scratch/users/flores12/PROJ_IrOx_OER/dft_workflow/bin/run_unsub_jobs.py',
#     'run',
#     'frac_of_jobs_to_run=1.',
#     'verbose=False',
#     ]


arg_dict = dict()
for i in sys.argv:
    if "=" in i:
        i_split = i.split("=")
        arg_dict[i_split[0]] = i_split[1]

# #########################################################
# Verbosity
verbose_i = arg_dict.get("verbose", True)
if verbose_i == "False":
    verbose = False
elif verbose_i == "True":
    verbose = True
elif verbose_i is True:
    verbose = True

# #########################################################
# Fraction of jobs to submit
frac_of_jobs_to_run_i = arg_dict.get("frac_of_jobs_to_run", 0.)
frac_of_jobs_to_run = float(frac_of_jobs_to_run_i)
# -

import os
last_2_dirs = "/".join(root_dir.split("/")[-2:])
if last_2_dirs == "dft_workflow/bin":
    root_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow")
if verbose:
    print("root_dir:", root_dir)

# +
if verbose:
    print("")
    print("")

    print("Usage:")
    print("  PROJ_irox_oer__comm_jobs_run_unsub_jobs run frac_of_jobs_to_run=0.2")
    print("")

    print("")
    print("sys.argv:", sys.argv)
    print("")

# if sys.argv[-1] == "run":
if "run" in sys.argv:
    run_jobs = True
    if verbose:
        print("running unsubmitted jobs")
else:
    run_jobs = False
    if verbose:
        print("Run script with 'run' flag")
        print("run_unsub_jobs run")

# +
# # TEMP
# print(111 * "TEMP | ")
# print("Not running jobs TEMP")
# run_jobs = False
# -

# # Parse directories

# +
from dft_workflow_methods import parse_job_dirs

df = parse_job_dirs(root_dir=root_dir)

# +
df_not_sub = df[df.is_submitted == False]
df_not_sub = df_not_sub[df_not_sub.is_empty == False]


out_dict = get_job_spec_scheduler_params(compenv=compenv)
wall_time_factor = out_dict["wall_time_factor"]

if verbose:
    print("")
    print("")
    print("Jobs to submit:")
    for path_i in df_not_sub.path_job_root_w_att_rev.tolist():
        print(path_i)
# -

print("")
print("Number of jobs to submit:", df_not_sub.shape[0])
print("")

# +
# assert False
# -

# # Submit jobs

for i_cnt, row_i in df_not_sub.iterrows():
    # #######################################
    path_i = row_i.path_full
    path_job_root_w_att_rev = row_i.path_job_root_w_att_rev
    # #######################################

    atoms_path_i = os.path.join(path_i, "init.traj")
    atoms = io.read(atoms_path_i)
    num_atoms = atoms.get_global_number_of_atoms()

    # #####################################################
    if random.random() <= frac_of_jobs_to_run:
        run_job_i = True
    else:
        run_job_i = False

    if run_jobs and run_job_i:

        print(40 * "*")
        print("Submitting:", path_job_root_w_att_rev)
        submit_job(
            path_i=path_i,
            num_atoms=num_atoms,
            wall_time_factor=wall_time_factor,
            queue=slac_sub_queue,
            )
        print("")

# + active=""
#
#
#
