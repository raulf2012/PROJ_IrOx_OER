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

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_jobs_anal,
    get_df_jobs_paths,
    )

# #########################################################

# -

# # Script Inputs

# +
# /nfs/slac/g/suncatfs/flores12/PROJ_IrOx_OER/dft_workflow/run_slabs/run_bare_oh_covered/out_data/dft_jobs/b5cgvsb16w/111/bare/active_site__67/01_attempt/_01
# -

# # Read Data

df_jobs = get_df_jobs()
df_jobs_data = get_df_jobs_data()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()

# + active=""
#
#
#
#
# -

df_jobs[df_jobs.ads == "bare"]
