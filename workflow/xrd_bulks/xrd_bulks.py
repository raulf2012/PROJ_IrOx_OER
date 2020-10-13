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

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import pandas as pd

from pymatgen.io.ase import AseAtomsAdaptor

# #########################################################
from methods import get_df_dft

# #########################################################
from local_methods import XRDCalculator
from local_methods import get_top_xrd_facets
# -

# # Script Inputs

verbose = True
verbose = False

# # Read Data

# +
df_dft = get_df_dft()

print("df_dft.shape:", df_dft.shape[0])

# + active=""
#
#
#
# -

# TEMP
df_dft = df_dft.sample(n=10)

# # Main loop

# +
data_dict_list = []
for i_cnt, (id_unique_i, row_i) in enumerate(df_dft.iterrows()):
    data_dict_i = dict()
    if verbose:
        print(40 * "=")
        print(str(i_cnt).zfill(3), "id_unique_i:", id_unique_i)

    # #####################################################
    atoms_i = row_i.atoms
    # #####################################################

    # #####################################################
    top_facets_i = get_top_xrd_facets(atoms=atoms_i)

    if verbose:
        tmp = [len(i) for i in top_facets_i]
        print(tmp)

    # #####################################################
    data_dict_i["id_unique"] = id_unique_i
    data_dict_i["top_facets"] = top_facets_i
    # #####################################################
    data_dict_list.append(data_dict_i)

# #########################################################
df_xrd = pd.DataFrame(data_dict_list)
df_xrd = df_xrd.set_index("id_unique", drop=False)

# +
# df_xrd
# -

assert False

# # Saving data to pickle

# Pickling data ###########################################
import os; import pickle
directory = "out_data"
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_xrd.pickle"), "wb") as fle:
    pickle.dump(df_xrd, fle)
# #########################################################

# +
from methods import get_df_xrd

df_xrd_tmp = get_df_xrd()

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# # #########################################################
# import pickle; import os
# path_i = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "workflow/xrd_bulks",
#     "out_data/df_xrd.pickle")
# with open(path_i, "rb") as fle:
#     df_xrd = pickle.load(fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# /home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/

# + jupyter={"source_hidden": true}
# This works fine
# tmp = XRDCalc.plot_structures([struct_i])

# + jupyter={"source_hidden": true}
# df_dft = df_dft.sample(n=100)
# df_dft = df_dft.sample(n=10)

# + jupyter={"source_hidden": true}
# df_dft[df_dft.stoich == "AB3"]
