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

# # Figuring out which systems are completely ready to be processed by script workflow
# ---
#
# Notebooks commonly broke because of trivial issues, like the files weren't downloaded locally or something like that

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import numpy as np

# #########################################################
from methods import get_df_jobs, get_df_atoms_sorted_ind
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# +
df_jobs = get_df_jobs()
df_jobs_i = df_jobs

df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# -

# #########################################################
group_cols = ["compenv", "slab_id", "ads", "active_site", "att_num", ]
grouped = df_jobs_i.groupby(group_cols)
# #########################################################
num_groups_processed = 0
for i_cnt, (name_i, group_i) in enumerate(grouped):
    tmp = 42

group_i

# +
df_atoms_sorted_ind

name_i
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("sys_fully_ready.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
