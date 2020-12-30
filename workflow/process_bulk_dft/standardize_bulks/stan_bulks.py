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

# # Standardizing unit cells of bulk polymorphs
# ---
# Hopefully cells will be smaller and more symmetric

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# #########################################################
from catkit.gen.symmetry import get_standardized_cell

# #########################################################
from methods import get_df_dft
# -

# # Script Inputs

# tol = 5e-01
# tol = 1e-01
tol = 1e-03
# tol = 1e-05
# tol = 1e-07
# tol = 1e-09

# # Read Data

df_dft = get_df_dft()

df_dft.head()

# + active=""
#
#

# +
# #########################################################
data_dict_list = []
# #########################################################
for bulk_id_i, row_i in df_dft.iterrows():
    data_dict_i = dict()

    # #####################################################
    atoms = row_i.atoms
    # #####################################################

    num_atoms = atoms.get_global_number_of_atoms()

    atoms_stan_prim = get_standardized_cell(atoms, primitive=True, tol=tol)
    atoms_stan = get_standardized_cell(atoms, primitive=False, tol=tol)

    num_atoms_stan_prim = atoms_stan_prim.get_global_number_of_atoms()
    num_atoms_stan = atoms_stan.get_global_number_of_atoms()

    num_atoms_lost_0 = num_atoms - num_atoms_stan
    num_atoms_lost_1 = num_atoms - num_atoms_stan_prim
    # num_atoms - num_atoms_stan_prim

    # #####################################################
    data_dict_i["id_unique"] = bulk_id_i
    data_dict_i["atoms_stan_prim"] = atoms_stan_prim
    data_dict_i["atoms_stan"] = atoms_stan
    data_dict_i["num_atoms"] = num_atoms
    data_dict_i["num_atoms_stan_prim"] = num_atoms_stan_prim
    data_dict_i["num_atoms_stan"] = num_atoms_stan
    data_dict_i["num_atoms_red__stan"] = num_atoms_lost_0
    data_dict_i["num_atoms_red__stan_prim"] = num_atoms_lost_1
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df = pd.DataFrame(data_dict_list)
df = df.set_index("id_unique", drop=False)
# -

df.head()

# +
print(
    "Num atoms total: ",
    df.num_atoms.sum(),
    "\n",
    "Num atoms reduced: ",
    df.num_atoms_red__stan_prim.sum(),
    sep="")

# df.sort_values("num_atoms", ascending=False)

# +
import plotly.graph_objs as go

x = [5e-1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, ]
y = [4383, 4155, 3141, 1246, 1080, 1080, ]

trace = go.Scatter(
    x=x, y=y)
fig = go.Figure(data=[trace])
fig.update_xaxes(type="log")
print("Number of atoms purged as a function of tolerance")
fig.show()

# +
# assert False
# -

df_dft_stan = df

# Pickling data ###########################################
import os; import pickle
dir_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/process_bulk_dft/standardize_bulks",
    "out_data")
file_name_i = os.path.join(
    dir_i, "df_dft_stan.pickle")
if not os.path.exists(dir_i): os.makedirs(dir_i)
with open(os.path.join(dir_i, "df_dft_stan.pickle"), "wb") as fle:
    pickle.dump(df_dft_stan, fle)
# #########################################################

# Pickling data ###########################################
import os; import pickle
dir_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/process_bulk_dft/standardize_bulks",
    "out_data")
file_name_i = os.path.join(
    dir_i, "df_dft_stan.pickle")
with open(file_name_i, "rb") as fle:
    df_dft_stan = pickle.load(fle)
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# num_atoms_stan_prim
# num_atoms_stan
