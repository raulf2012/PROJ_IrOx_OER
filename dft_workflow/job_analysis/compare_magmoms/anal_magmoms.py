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

# # Analyse magmoms of converged slabs for the purpose of setting initial magmoms in the future
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import numpy as np
import pandas as pd

# #########################################################
from methods import get_df_atoms_sorted_ind
from methods import get_df_jobs
from methods import get_df_jobs_anal
# -

# # Read Data

# +
df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_jobs = get_df_jobs()

df_jobs_anal = get_df_jobs_anal()

# + active=""
#
#
# -

df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# +
df_index_i = df_jobs_anal_i.index.to_frame()
df_index_i = df_index_i[df_index_i.ads == "oh"]

df_index_i.index

# +
df_atoms_sorted_ind_i = df_atoms_sorted_ind.loc[
    df_index_i.index
    ]

# print(10 * "TEMP | ")
# df_atoms_sorted_ind_i = df_atoms_sorted_ind_i.iloc[[0]]

# #########################################################
data_dict_list = []
# #########################################################
for name_i, row_i in df_atoms_sorted_ind_i.iterrows():
    # #####################################################
    was_sorted_i = row_i.was_sorted
    magmoms_sorted_good_i = row_i.magmoms_sorted_good
    atoms_sorted_good_i = row_i.atoms_sorted_good
    # #####################################################

    if magmoms_sorted_good_i is None:
        magmoms_i = atoms_sorted_good_i.get_magnetic_moments()
    else:
        magmoms_i = magmoms_sorted_good_i

    atoms = atoms_sorted_good_i

    # Positions
    z_positions = atoms.positions[:, 2]
    z_max = z_positions.max()


    for atom_j in atoms:
        # #################################################
        data_dict_j = dict()
        # #################################################
        atom_index_j = atom_j.index
        symbol_j = atom_j.symbol
        # #################################################

        magmom_j = magmoms_i[atom_index_j]

        z_pos_j = atom_j.position[2]
        dist_from_top = z_max - z_pos_j

        # #################################################
        data_dict_j["symbol"] = symbol_j
        data_dict_j["magmom"] = magmom_j
        data_dict_j["dist_from_top"] = dist_from_top
        # data_dict_j[""] = 
        # #################################################
        data_dict_list.append(data_dict_j)
        # #################################################


# #########################################################
df = pd.DataFrame(data_dict_list)
df.head()
# -

df["magmom_abs"] = np.abs(df.magmom)

df = df[df.dist_from_top < 4]

# +
# assert False
# -

import plotly.express as px
fig = px.histogram(
    df,
    x="magmom_abs",
    color="symbol",
    marginal="rug", # can be `box`, `violin`
    nbins=100,
    )
fig.show()


