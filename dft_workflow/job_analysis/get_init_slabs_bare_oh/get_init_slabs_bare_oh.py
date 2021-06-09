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

# # Extract the initial atoms objects for the bare and *OH slabs
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

from IPython.display import display

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 20
# pd.set_option('display.max_rows', None)

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_active_sites,
    )
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
df_jobs = get_df_jobs(exclude_wsl_paths=True)

df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
# -

# ### Filtering dataframes to only include `oer_adsorbate` job types

df_jobs = df_jobs[df_jobs.job_type == "oer_adsorbate"]

# ### Main loop

# +
# #########################################################
data_dict_list = []
# #########################################################
group_cols = [
    "compenv", "slab_id",
    "ads", "active_site", "att_num"]
grouped = df_jobs.groupby(group_cols)
# #########################################################
for name, group in grouped:

# for i in range(1):
#     name = ('sherlock', 'fogalonu_46', 'o', 16.0, 1)
#     group = grouped.get_group(name)

    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    ads_i = name[2]
    active_site_i = name[3]
    att_num_i = name[4]
    # #####################################################

    group = group.drop(
        # cols_to_drop + ["num_revs", "job_id"],
        group_cols + ["num_revs", "job_id"],
        axis=1)

    # #####################################################
    row_i = group[group.rev_num == 1]
    mess_i = "Must only have one row in a group with rev_num=1"
    assert row_i.shape[0] == 1, mess_i
    row_i = row_i.iloc[0]

    job_id_min_i = row_i.name
    # #####################################################

    if job_id_min_i == "ruhusunu_66":
        print(name)

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_min_i]
    # #####################################################
    init_atoms_i = row_data_i.init_atoms
    # #####################################################


    # #####################################################
    data_dict_i = dict()
    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["att_num"] = att_num_i
    data_dict_i["job_id_min"] = job_id_min_i
    data_dict_i["init_atoms"] = init_atoms_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


# #########################################################
df_init_slabs = pd.DataFrame(data_dict_list)
df_init_slabs = df_init_slabs.set_index(["compenv", "slab_id", "ads", "active_site", "att_num", ])
# #########################################################
# -

# ### Get number of atoms

# +
def method(row_i):
    # #####################################################
    init_atoms_i = row_i.init_atoms
    job_id_min_i = row_i.job_id_min
    # #####################################################

    if init_atoms_i is None:
        print("Couldn't find init_atoms for this job_id")
        print("job_id_min:", job_id_min_i)

    num_atoms_i = init_atoms_i.get_global_number_of_atoms()

    return(num_atoms_i)

df_init_slabs["num_atoms"] = df_init_slabs.apply(
    method,
    axis=1)
# -

# ### Save data to pickle

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/get_init_slabs_bare_oh",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_init_slabs.pickle"), "wb") as fle:
    pickle.dump(df_init_slabs, fle)
# #########################################################

# +
from methods import get_df_init_slabs

df_init_slabs_tmp = get_df_init_slabs()
df_init_slabs_tmp.head()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("get_init_slabs_bare_oh.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_dft = get_df_dft()

# df_job_ids = get_df_job_ids()

# df_jobs_data_clusters = get_df_jobs_data_clusters()

# df_slab = get_df_slab()

# df_slab_ids = get_df_slab_ids()

# df_jobs_anal = get_df_jobs_anal()

# df_active_sites = get_df_active_sites()

# df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# + jupyter={"source_hidden": true}
# df_job_ids
# df_jobs
# df_jobs_data
# df_jobs_data_clusters
# df_jobs_anal

# df_atoms_sorted_in

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# init_atoms_i

# #     row_data_i = 
# df_jobs_data.loc[job_id_min_i]

# + jupyter={"source_hidden": true}
# assert False
