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

import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor

# #########################################################
from methods import get_df_dft

# #########################################################
sys.path.insert(0, "..")
from local_methods import XRDCalculator
from local_methods import get_top_xrd_facets
# -

# # Script Inputs

# +
verbose = True
# verbose = False

# bulk_id_i = "8ymh8qnl6o"
# bulk_id_i = "8p8evt9pcg"
# bulk_id_i = "8l919k6s7p"
bulk_id_i = "64cg6j9any"
# -

# # Read Data

# +
df_dft = get_df_dft()
print("df_dft.shape:", df_dft.shape[0])

from methods import get_df_xrd
df_xrd = get_df_xrd()

df_xrd = df_xrd.set_index("id_unique", drop=False)

# + active=""
#
#
#

# +
# #########################################################
row_i = df_dft.loc[bulk_id_i]
# #########################################################
atoms_i = row_i.atoms
atoms_stan_prim_i = row_i.atoms_stan_prim
# #########################################################

# Writing bulk facets
atoms_i.write("out_data/bulk.traj")
atoms_i.write("out_data/bulk.cif")

# #########################################################
row_xrd_i = df_xrd.loc[bulk_id_i]
# #########################################################
top_facets_i = row_xrd_i.top_facets
# #########################################################

print(
    "top_facets:",
    top_facets_i
    )

# +
# assert False

# +
# atoms = atoms_i
atoms = atoms_stan_prim_i

AAA = AseAtomsAdaptor()
struct_i = AAA.get_structure(atoms)

XRDCalc = XRDCalculator(
    wavelength='CuKa',
    symprec=0,
    debye_waller_factors=None,
    )

# XRDCalc.get_plot(structure=struct_i)
# # XRDCalc.get_plot?

plt = XRDCalc.plot_structures([struct_i])
# -

# # Saving plot to file

file_name_i = os.path.join(
    "out_plot",
    bulk_id_i + ".png",
    )
plt.savefig(
    file_name_i,
    dpi=1600,
    )
