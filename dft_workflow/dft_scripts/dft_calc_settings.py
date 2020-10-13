"""
"""

#| - Import Modules
import os

import json
#__|


#| - Settings
dipole_corr = True
u_corr = False

algo = "Normal"
#  algo = "Fast"
# algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
# algo='Very_Fast',  # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
# algo='normal',

ibrion = 2
potim = 0.3
#__|

#| - Default calc settings
dft_calc_settings = dict(
    encut=500,
    xc="PBE",
    gga="PE",

    # Description: ISYM determines the way VASP treats symmetry. 
    # isym=1,

    #| - kpoints
    # #kpts  = (2,2,1),
    # # kpts=(6, 6, 1),
    # # For typical 2x2 cell IrO2 110 with 2 cusp sites and 2 bridge sites
    # kpts=(4, 4, 1),

    gamma=True,  # Gamma-centered (defaults to Monkhorst-Pack)
    #__|

    #| - Mixing parameters
    # Description: INIMIX determines the functional form of the initial mixing matrix in the Broyden scheme (IMIX=4)
    inimix=0,

    amix=0.1,
    bmix=0.00005,
    amix_mag=0.1,
    bmix_mag=0.00005,

    # Conservative Mixing Paramters
    # amix=0.05,
    # bmix=0.00001,
    # amix_mag=0.05,
    # bmix_mag=0.00001,
    #__|

    #| - LDA+U Related settings

    # Description: LDAU=.TRUE. switches on the L(S)DA+U.
    ldau=u_corr,

    # Description: LDAUTYPE specifies which type of L(S)DA+U approach will be used
    ldautype=2,

    ldau_luj={
        'Ni': {'L': 2, 'U': 6.45, 'J': 0.0},
        'Co': {'L': 2, 'U': 3.32, 'J': 0.0},
        'Cr': {'L': 2, 'U': 3.5, 'J': 0.0},
        'Fe': {'L': 2, 'U': 5.3, 'J': 0.0},
        'Ce': {'L': 3, 'U': 4.50, 'J': 0.0},
        'V': {'L': 2, 'U': 3.25, 'J': 0.0},
        'Mn': {'L': 2, 'U': 3.75, 'J': 0.0},
        'Ti': {'L': 2, 'U': 3.00, 'J': 0.0},
        'W': {'L': -1, 'U': 0.0, 'J': 0.0},
        'O': {'L': -1, 'U': 0.0, 'J': 0.0},
        'C': {'L': -1, 'U': 0.0, 'J': 0.0},
        'Au': {'L': -1, 'U': 0.0, 'J': 0.0},
        'Ir': {'L': -1, 'U': 0.0, 'J': 0.0},
        'Cu': {'L': -1, 'U': 0.0, 'J': 0.0},
        'H': {'L': -1, 'U': 0.0, 'J': 0.0},
        },

    # Description: LDAUPRINT controls the verbosity of the L(S)DA+U routines
    ldauprint=2,

    # Description: LMAXMIX controls up to which l-quantum number the one-center PAW charge densities are passed through the charge density mixer and written to the CHGCAR file.
    # lmaxmix=4,
    lmaxmix=6,

    #__|

    #| - Dipole Correction
    idipol=3,
    dipol=(0, 0, 0.5),  # (0.5, 0.5, 0.5) Maybe?
    ldipol=dipole_corr,
    #__|

    # #####################################################
    # #####################################################
    #setups={'O': '_s', 'C': '_s'},

    kpar=10,
    npar=4,

    # Description: ISMEAR determines how the partial occupancies fnk are set for each orbital. SIGMA determines the width of the smearing in eV.
    ismear=0,

    #nupdown= 0,

    # Number of allowed SCF iterations per ionic step
    # nelm=350,
    nelm=300,

    sigma=0.05,
    algo=algo,

    ibrion=ibrion,

    potim=potim,

    isif=2,

    ediffg=-0.02,  # forces

    # Description: EDIFF specifies the global break condition for the electronic SC-loop.
    ediff=1e-5,  # energy conv.

    #nedos=2001,

    # Description: PREC specifies the "precision"-mode.
    prec="Normal",  # "Normal" or "Accurate"

    # nsw=int(200 / potim),  # don't use the VASP internal relaxation, only use ASE
    nsw=300,  # don't use the VASP internal relaxation, only use ASE

    # Description: LVTOT determines whether the total local potential is written to the LOCPOT file
    lvtot=False,

    # Description: ISPIN specifies spin polarization.
    ispin=1,

    # Description: LORBIT, together with an appropriate RWIGS, determines whether the PROCAR or PROOUT files are written
    lorbit=11,

    # Description: LREAL determines whether the projection operators are evaluated in real-space or in reciprocal space.
    lreal="auto",

    # Description: include non-spherical contributions related to the gradient of the density in the PAW spheres
    lasph=True,

    )

#__|

#| - Easy DFT settings (Fast convergence)

dft_calc_settings_easy = dict(
    encut=250,
    kpts=[1, 1, 1],
    ldipol=False,
    ediff=5e-04,
    ediffg=-0.2,
    )

#__|

#| - Conservative Mixing Params
cons_mixing_params = dict(

    # Description: INIMIX determines the functional form of the initial mixing matrix in the Broyden scheme (IMIX=4)
    inimix=0,

    # Conservative Mixing Paramters
    amix=0.05,
    bmix=0.00001,
    amix_mag=0.05,
    bmix_mag=0.00001,

    )

#__|


# #########################################################
#| - Write to file
directory = "out_data"
if not os.path.exists(directory):
    os.makedirs(directory)

# Saving main DFT settings
path_i = os.path.join("out_data/dft_calc_settings.json")
with open(path_i, "w") as outfile:
    json.dump(dft_calc_settings, outfile, indent=2)

# Saving easy DFT settings
path_i = os.path.join("out_data/easy_dft_calc_settings.json")
with open(path_i, "w") as outfile:
    json.dump(dft_calc_settings_easy, outfile, indent=2)

# Saving easy DFT settings
path_i = os.path.join("out_data/conservative_mixing_params.json")
with open(path_i, "w") as outfile:
    json.dump(cons_mixing_params, outfile, indent=2)
#__|
# #########################################################
