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

# # Collect and combine `jobs_mine` create dataframes from clusters
# ---

# ### Import modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# #########################################################
from proj_data import compenvs
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Combine all `df_jobs_o_clus` dataframes and combine into single `df`

# +
df_dir_path = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/cluster_scripts",
    "out_data")

# #########################################################
# Read dataframes generated within all clusters
data_frame_list = []
data_frame_dict = dict()
for compenv_i in compenvs:
    file_path_i = os.path.join(
        df_dir_path,
        "df_jobs_on_clus__%s.pickle" % compenv_i)

    my_file = Path(file_path_i)
    if my_file.is_file():
        with open(file_path_i, "rb") as fle:
            df_i = pickle.load(fle)
            data_frame_list.append(df_i)
            data_frame_dict[compenv_i] = df_i

# #########################################################
df_jobs_on_clus__all = pd.concat(data_frame_list)
# #########################################################
# -

# ### Write dataframe to file

# +
out_data_dir_rel_to_proj = os.path.join(
    "dft_workflow/cluster_scripts",
    "out_data",
    )

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    out_data_dir_rel_to_proj)

if not os.path.exists(directory):
    os.makedirs(directory)

# Pickling data ###########################################
file_path = os.path.join(directory, "df_jobs_on_clus__all.pickle")
with open(file_path, "wb") as fle:
    pickle.dump(df_jobs_on_clus__all, fle)
# #########################################################

# +
from methods import get_df_jobs_on_clus__all

df_jobs_on_clus__all_tmp = get_df_jobs_on_clus__all()
df_jobs_on_clus__all_tmp.iloc[0:2]
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_df_from_clust.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# data_frame_dict["nersc"]

# + jupyter={"source_hidden": true}
# assert False
