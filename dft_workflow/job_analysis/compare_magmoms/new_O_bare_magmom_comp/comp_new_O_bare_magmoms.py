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

# # Measuring how much better the magmom matching is before and after running additionnal *O and * slabs from *OH
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

# #########################################################
from methods import get_df_magmoms
from methods import read_magmom_comp_data

from methods import get_df_jobs_anal, get_df_jobs_data, get_df_jobs
# -

# # Read Data

df_magmoms = get_df_magmoms()
magmom_data_dict = read_magmom_comp_data()
df_jobs = get_df_jobs()
df_jobs_anal = get_df_jobs_anal()
df_jobs_data = get_df_jobs_data()

# +
# TODO | Don't use df_oer_groups, not needed anymore

from methods import get_df_oer_groups

df_oer_groups = get_df_oer_groups()

# +
# df_oer_groups

# +
for name_i, row_i in df_oer_groups.iterrows():
    tmp = 42

# #########################################################
compenv_i = name_i[0]
slab_id_i = name_i[1]
active_site_i = name_i[2]
# #########################################################
# -

row_i

# +
# # df_jobs

# group_cols = ["compenv", "bulk_id", "slab_id", "", ]
# grouped = df_jobs.groupby(group_cols)
# for name, group in grouped:
#     tmp = 42

# +
# df_magmoms
