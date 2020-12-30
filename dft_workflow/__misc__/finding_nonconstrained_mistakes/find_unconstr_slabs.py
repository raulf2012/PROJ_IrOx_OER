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
import sys

import pandas as pd
pd.set_option("display.max_columns", None)
# pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

from methods import get_df_jobs_data
# -

# # Read Data

df_jobs_data = get_df_jobs_data()

df_jobs_data_i = df_jobs_data[
    ~df_jobs_data.final_atoms.isna()
    ]

# +
# job_id_i = "sesepado_97"
# df_jobs_data_i = df_jobs_data_i.loc[[job_id_i]]

# +
bad_job_ids = []
for job_id_i, row_i in df_jobs_data_i.iterrows():
    # #####################################################
    final_atoms_i = row_i.final_atoms
    # #####################################################

    # print(job_id_i)

    has_constraints = False
    if len(final_atoms_i.constraints) > 0:
        has_constraints = True

    if not has_constraints:
        # print(job_id_i)
        bad_job_ids.append(job_id_i)

if len(bad_job_ids) > 0:
    print(50 * "ALERT | There are slabs with no constraints!!")

# +
# ['vuvukara_45', 'setumaha_18', 'nububewo_52', 'fowonifu_15']

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# if not has_constraints:
#     print("IDSJSFIDSif")

# + jupyter={"source_hidden": true}
# len(final_atoms_i.constraints)
