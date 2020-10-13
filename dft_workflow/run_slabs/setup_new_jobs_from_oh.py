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

# # Setup new jobs to resubmit *O (and possibly *) from *OH to achieve better magmom matching
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pandas as pd

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_anal,
    get_df_oer_groups,
    get_df_jobs_oh_anal,
    get_df_rerun_from_oh,
    get_df_atoms_sorted_ind,
    )

# +
df_jobs = get_df_jobs()

df_oer_groups = get_df_oer_groups()

df_jobs_oh_anal = get_df_jobs_oh_anal()

df_rerun_from_oh = get_df_rerun_from_oh()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# + active=""
#
#

# +
df_rerun_from_oh_i = df_rerun_from_oh[df_rerun_from_oh.rerun_from_oh == True]

# #########################################################
# #########################################################
for i_cnt, row_i in df_rerun_from_oh_i.iterrows():

    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    job_id_most_stable_i = row_i.job_id_most_stable
    # #####################################################

    # #########################################################
    row_jobs_i = df_jobs.loc[job_id_most_stable_i]
    # #########################################################
    att_num_i = row_jobs_i.att_num
    # #########################################################

    # #########################################################
    idx_i = pd.IndexSlice[compenv_i, slab_id_i, "oh", active_site_i, att_num_i]
    row_atoms_i = df_atoms_sorted_ind.loc[idx_i, :]
    # #########################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    magmoms_sorted_good_i = row_atoms_i.magmoms_sorted_good
    # #########################################################

    if atoms_i.calc is None:
        if magmoms_sorted_good_i is not None:
            atoms_i.set_initial_magnetic_moments(magmoms_sorted_good_i)
        else:
            print("Not good there should be something here")
# -

atoms_i.write("__temp__/tmp.traj")

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_atoms_sorted_ind
