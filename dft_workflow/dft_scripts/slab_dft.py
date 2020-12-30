#!/usr/bin/env python

"""Run VASP job on slab.

Author(s): Michal Badich, Raul A. Flores
"""

# | - Import Modules
import os
print(os.getcwd())
import sys

import json
import subprocess
import time
t0 = time.time()

import numpy as np

import ase.calculators.vasp as vasp_calculator
from ase import io

# My Modules
from ase_modules.ase_methods import clean_up_dft, get_slab_kpts
#__|

# | - Script Inputs
easy_params = False

# set_magmoms_i = False
#__|

#| - Misc setup
directory = "out_data"
if not os.path.exists(directory):
    os.makedirs(directory)
#__|

# | - Read Atoms Object
if os.path.isfile("init.traj"):
    atoms = io.read('init.traj')
#__|

#| - Setting magmoms
from dft_workflow_methods import set_magmoms

set_magmoms(
    atoms=atoms,
    mode=None,  # "set_magmoms_M_O", "set_magmoms_random"
    set_from_file_if_avail=True)

# #########################################################
init_magmoms = atoms.get_initial_magnetic_moments()
data_path = os.path.join("out_data/init_magmoms.json")
with open(data_path, "w") as fle:
    json.dump(list(init_magmoms), fle, indent=2)
# #########################################################
#__|

# | - Calculator

# ########################################################
data_root_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/dft_scripts/out_data")

data_path = os.path.join(data_root_path, "dft_calc_settings.json")
with open(data_path, "r") as fle:
    dft_calc_settings = json.load(fle)

if easy_params:
    easy_data_path = os.path.join(
        data_root_path, "easy_dft_calc_settings.json")
    with open(easy_data_path, "r") as fle:
        easy_dft_calc_settings = json.load(fle)

# ########################################################
kpoints = get_slab_kpts(atoms)

local_dft_settings = dict(
    kpts=kpoints,
    )
dft_calc_settings.update(local_dft_settings)

# ########################################################
# Reading VASP parameters from file and merging with params in script
from ase_modules.dft_params import VASP_Params
VP = VASP_Params(load_defaults=False)
VP.load_params()
dft_calc_settings.update(VP.params)


# ########################################################
if easy_params:
    dft_calc_settings.update(easy_dft_calc_settings)


# ########################################################
calc = vasp_calculator.Vasp(**dft_calc_settings)


# ########################################################
# Writing all calculator parameters to json
with open("out_data/calc_params.json", "w") as outfile:
    json.dump(calc.todict(), outfile, indent=2)

#__|


# Setting magmoms to 0 if not running spin-polarized calculation
if dft_calc_settings.get("ispin", None) == 1:
    print(30 * "*")
    print("Setting magnetic moments to 0 since ispin=1")
    print(30 * "*")
    atoms.set_initial_magnetic_moments(magmoms=None)
    # #####################################################
    init_magmoms = atoms.get_initial_magnetic_moments()
    data_path = os.path.join("out_data/init_magmoms.json")
    with open(data_path, "w") as fle:
        json.dump(list(init_magmoms), fle, indent=2)
    # #####################################################


atoms.set_calculator(calc)

#| - Copy over WAVECAR file if available
cwd = os.getcwd()
rev_num_i = cwd.split("/")[-1]
assert rev_num_i[0] == "_", "ijdfisifjisd"
rev_num_i = int(rev_num_i[1:])

ispin = dft_calc_settings.get("ispin", None)

from pathlib import Path
from vasp.vasp_methods import read_incar

if rev_num_i > 1:
    rev_num_im1 = rev_num_i - 1
    prev_rev_dir = "../_" + str(rev_num_im1).zfill(2)


    my_file = Path(prev_rev_dir)
    if my_file.is_dir():
        # directory exists

        prev_rev_files_list = os.listdir(prev_rev_dir)

        # Checking spin of previous calculation
        path_i = os.path.join(prev_rev_dir, "INCAR")
        my_file = Path(path_i)
        prev_ispin = 1
        if my_file.is_file():
            prev_incar_dict = read_incar(prev_rev_dir, verbose=False)
            prev_ispin = prev_incar_dict.get("ISPIN", None)

        if ispin == 2 and prev_ispin == 1:
            print("Job changed spin, so not copying WAVECAR")
        else:
            if "WAVECAR" in prev_rev_files_list:
                wavecar_dir = os.path.join(prev_rev_dir, "WAVECAR")
                from shutil import copyfile
                copyfile(wavecar_dir, "./WAVECAR")
#__|

#| - Writing pre-DFT objects to file
out_data = dict(
    atoms=atoms,
    dft_calc_settings=dft_calc_settings,
    VP=VP,
    calc=calc,
    local_dft_settings=local_dft_settings,
    t0=t0,
    )

# Pickling data ###########################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "pre_out_data.pickle"), "wb") as fle:
    pickle.dump(out_data, fle)
# #########################################################
#__|


compenv = os.environ["COMPENV"]
if not compenv in ["nersc", "slac", "sherlock", "slac/sdf_cluster", ]:
    print("Not in a environment setup for DFT calculations")
else:
    atoms.get_potential_energy()

    #| - Writing files

    #| - TEST new write
    from ase.io.trajectory import Trajectory
    import subprocess

    traj2=Trajectory('final_with_calculator.traj',  'w')
    traj2.write(atoms)
    subprocess.call('ase convert -f final_with_calculator.traj  final_with_calculator.json', shell=True)
    #__|

    io.write("out.cif", atoms)
    io.write("out.traj", atoms)

    try:
        traj = io.read("OUTCAR", index=":")
        io.write("final_images.traj", traj)
    except:
        print("Couldn't read/write final OUTCAR traj object")

    tf = time.time()
    out_data = dict(
        atoms=atoms,
        dft_calc_settings=dft_calc_settings,
        VP=VP,
        calc=calc,
        local_dft_settings=local_dft_settings,
        run_time=np.abs(tf - t0),
        t0=t0,
        tf=tf,
        )

    # Pickling data #######################################
    import os; import pickle
    directory = "out_data"
    if not os.path.exists(directory): os.makedirs(directory)
    with open(os.path.join(directory, "out_data.pickle"), "wb") as fle:
        pickle.dump(out_data, fle)
    # #####################################################

    #__|

    clean_up_dft()
