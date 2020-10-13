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

import os
print(os.getcwd())
import sys

from local_methods import parse_job_err

# +
#| - Import Modules
import os
import sys

import time
import pickle
import subprocess
from pathlib import Path

# from contextlib import contextmanager

import numpy as np

from ase import io

# #########################################################
from vasp.vasp_methods import parse_incar, read_incar
from vasp.parse_oszicar import parse_oszicar
# from vasp.vasp_methods import

# #########################################################
from dft_job_automat.compute_env import ComputerCluster

# #########################################################
from proj_data import compenv
from methods import temp_job_test, cwd
#__|
# -

path_full_i = "/home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/slac/v2blxebixh/100/01_attempt/_01"
compenv_i = "slac"

# +
path = path_full_i
compenv = compenv_i

# def parse_job_err(path, compenv=None):
#     """
#     """
#| - parse_job_err
# print(path)

status_dict = {
    "timed_out": None,
    "error": None,
    "error_type": None,
    "brmix_issue": None,
    }

if compenv is None:
    compenv = os.environ["COMPENV"]


# | - Parsing SLAC job
print("TEMP 00")
if compenv == "slac":
    job_out_file_path = os.path.join(path, "job.out")
    my_file = Path(job_out_file_path)
    if my_file.is_file():
        with open(job_out_file_path, 'r') as f:
            lines = f.readlines()

        # print("This spot here now 0")

        for line in lines:
            if "job killed after reaching LSF run time limit" in line:
                # print("Found following line in job.err")
                # print("job killed after reaching LSF run time limit")
                status_dict["timed_out"] = True
                break
#__|

# | - Parsing error file
job_err_file_path = os.path.join(path, "job.err")
my_file = Path(job_err_file_path)
if my_file.is_file():
    with open(job_err_file_path, 'r') as f:
        lines = f.readlines()

    # else:
    for line in lines:
        if "DUE TO TIME LIMIT" in line:
            status_dict["timed_out"] = True

        if "Traceback (most recent call last):" in line:
            status_dict["error"] = True

        if "ValueError: could not convert string to float" in line:
            status_dict["error"] = True
            status_dict["error_type"] = "calculation blown up"

#__|


# | - Parsing out file

#| - old parser here, keeping for now
if status_dict["error"] is True:
    job_out_file_path = os.path.join(path, "job.out")
    my_file = Path(job_out_file_path)
    if my_file.is_file():
        with open(job_out_file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            err_i = "VERY BAD NEWS! internal error in subroutine SGRCON:"
            if err_i in line:
                status_dict["error_type"] = "Error in SGRCON (symm error)"
                break
#__|


my_file_0 = Path(os.path.join(path, "job.out"))
my_file_1 = Path(os.path.join(path, "job.out.short"))
if my_file_0.is_file():
    job_out_file = my_file_0
elif my_file_1.is_file():
    job_out_file = my_file_1
else:
    job_out_file = None

if job_out_file is not None:
    with open(job_out_file, 'r') as f:
        lines = f.readlines()

    #| - Checking for BRMIX error
    for line in lines:
        err_i = "BRMIX: very serious problems"
        if err_i in line:
            status_dict["brmix_issue"] = True
            status_dict["error"] = True
            # break
    #__|

#__|


# return(status_dict)
#__|
# -

status_dict

# +
# parse_job_err(path_full_i, compenv=compenv_i)
# -

assert False

# + active=""
#
#
#
#
#
#
#

# + jupyter={"source_hidden": true}
from local_methods import temp_job_test

temp_job_test()
