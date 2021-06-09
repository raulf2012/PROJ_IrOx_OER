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


import re
import pandas as pd
import glob
import subprocess
from ase.io import write
import os.path

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

kpoints = list(2 * np.array(kpoints))
kpoints[-1] = 1

kpoints = [int(kpoints[0]), int(kpoints[1]), int(kpoints[2]), ]

# kpoints = list(kpoints)

local_dft_settings = dict(
    kpts=kpoints,
    nelm=999,
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

# Setting NSW to 1, just need to create Charge files
dft_calc_settings.update(dict(nsw=1))

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

    # clean_up_dft()
















print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")
print("Running PDOS Worflow Now **********************************88")





# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################


















try:
    from dos__calc_settings import dos_settings_params
except:
    dos_settings_params = dict()

#| - Import Modules
import os
import subprocess
import ase.calculators.vasp as vasp_calculator
from ase import io

# My Modules
from ase_modules.ase_methods import clean_up_dft
#__|

#| - Script Inputs
dipole_corr = True
u_corr = False
algo = "Normal"

ibrion = 2
potim = 0.5
# algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
#__|

#| - Read Atoms Object
if os.path.isfile("init.traj"):
    atoms = io.read('init.traj')
#__|

#| - Copy Previous OUTCAR and moments.traj
subprocess.call('cp -rf OUTCAR OUTCAR_$(date +%s)', shell=True)
subprocess.call('cp -rf moments.traj moments.traj_$(date +%s)', shell=True)
#__|

#| - VASP Calculator

#| - OLD | DFT Parameters
# dft_params = dict(
#     encut=500,
#     xc='PBE',
#     #setups={'O': '_s', 'C': '_s'},
#     gga='PE',
#     #kpts  = (2,2,1),
#     # kpts=(6, 6, 1),
#     # For typical 2x2 cell IrO2 110 with 2 cusp sites and 2 bridge sites
#     kpts=(4, 4, 1),
#     kpar=11,
#     npar=6,
#     gamma=True,  # Gamma-centered (defaults to Monkhorst-Pack)
#     ismear=0,
#
#     #| - Mixing Parameters
#     inimix=0,
#
#     amix=0.2,
#     bmix=0.0001,
#     amix_mag=0.1,
#     bmix_mag=0.0001,
#
#     # Conservative Mixing Paramters
#     # amix=0.05,
#     # bmix=0.00001,
#     # amix_mag=0.05,
#     # bmix_mag=0.00001,
#     #__|
#
#     #nupdown= 0,
#     nelm=250,
#     sigma=0.05,
#     algo=algo,
#     # algo='Very_Fast',  # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
#     # algo='normal',
#     ibrion=ibrion,
#     potim=potim,
#     isif=2,
#     ediffg=-0.02,  # forces
#     ediff=1e-5,  # energy conv.
#     #nedos=2001,
#     prec='Normal',
#     nsw=int(200 / potim),  # don't use the VASP internal relaxation, only use ASE
#     lvtot=False,
#     ispin=2,
#
#     #| - Hubbard U
#     ldau=u_corr,
#     ldautype=2,
#     lreal='auto',
#     lasph=True,
#     ldau_luj={
#         'Ni': {'L': 2, 'U': 6.45, 'J': 0.0},
#         'Co': {'L': 2, 'U': 3.32, 'J': 0.0},
#         'Cr': {'L': 2, 'U': 3.5, 'J': 0.0},
#         'Fe': {'L': 2, 'U': 5.3, 'J': 0.0},
#         'Ce': {'L': 3, 'U': 4.50, 'J': 0.0},
#         'V': {'L': 2, 'U': 3.25, 'J': 0.0},
#         'Mn': {'L': 2, 'U': 3.75, 'J': 0.0},
#         'Ti': {'L': 2, 'U': 3.00, 'J': 0.0},
#         'W': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'O': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'C': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'Au': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'Ir': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'Cu': {'L': -1, 'U': 0.0, 'J': 0.0},
#         'H': {'L': -1, 'U': 0.0, 'J': 0.0},
#         },
#     ldauprint=2,
#     #__|
#
#     # lmaxmix=4,
#     lmaxmix=6,
#
#     lorbit=11,
#
#     idipol=3,
#     dipol=(0, 0, 0.5),
#     ldipol=dipole_corr,
#
#     # ldipol=True,
#
#     # addgrid=True,
#     # isym=0,
#     )
#__|


dft_calc_settings.update(dos_settings_params)

# Setting NSW to 1, just need to create Charge files
dft_calc_settings.update(dict(kpts=kpoints))

calc = vasp_calculator.Vasp(**dft_calc_settings)

atoms.set_calculator(calc)
#__|

atoms.get_potential_energy()

io.write('out.cif', atoms)
io.write('out.traj', atoms)
























print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")
print("RUNNING Bader Script ****************************************")




sys.path.insert(0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/dft_scripts/dos_scripts",
        )
    )


# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################


from bader_get_charge_vasp_py3 import get_bader_charges


print('Argument List:', str(sys.argv))
traj = 'OUTCAR'
Len = len(sys.argv)
if Len > 1:
    for i in range(1, Len):
        if sys.argv[i] == "-t":
            traj = sys.argv[i+1]

atoms_charge=get_bader_charges(traj)
write_charge_traj=True
if write_charge_traj:
    atoms=io.read(traj)
    #atoms_charge.set_initial_magnetic_moments(write_charge)
    atoms.set_initial_charges(atoms_charge)
    io.write('bader_charge.json',atoms)


































print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")
print("Running PDOS Script *****************************************")






# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################
# #########################################################

from rapiDOS import (
    rapiDOS,
    read_dosfile,
    read_posfile,
    write_dos0,
    write_nospin,
    write_spin,
    )


rapiDOS(
    data_folder=None,
    out_folder="rapiDOS_out",
    )


clean_up_dft()
