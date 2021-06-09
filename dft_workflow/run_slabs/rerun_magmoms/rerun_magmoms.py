# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Rerun jobs to achieve better magmom matching
# ---
#
# Will take most magnetic slab of OER set and apply those magmoms to the other slabs

# ### Import Modules

# +
import os
print(os.getcwd())
import sys

# #########################################################
from methods import get_df_features_targets
from methods import get_df_magmoms
# -

# ### Read Data

# +
df_features_targets = get_df_features_targets()

df_magmoms = get_df_magmoms()
df_magmoms = df_magmoms.set_index("job_id")

# +
for name_i, row_i in df_features_targets.iterrows():
    tmp = 42

# #####################################################
job_id_o_i = row_i[("data", "job_id_o", "", )]
job_id_oh_i = row_i[("data", "job_id_oh", "", )]
job_id_bare_i = row_i[("data", "job_id_bare", "", )]
# #####################################################

job_ids = [job_id_o_i, job_id_oh_i, job_id_bare_i]
# -

df_magmoms.loc[job_ids]
