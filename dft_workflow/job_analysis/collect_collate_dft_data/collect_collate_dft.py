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

# # Collect DFT data into OER sets 
# ---

# # Import Modules

# + jupyter={}
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

import numpy as np

# #########################################################
from IPython.display import display

# #########################################################
from methods import get_df_jobs_anal
from methods import get_df_jobs_data
from methods import get_df_slabs_to_run
from methods import get_df_jobs_oh_anal
from methods import get_df_atoms_sorted_ind
from methods import get_df_octa_info
from methods import get_df_struct_drift

# #########################################################
from local_methods import calc_ads_e
from local_methods import get_oer_triplet
from local_methods import get_group_w_all_ads, are_any_ads_done
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
# #########################################################
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_run = df_slabs_to_run.set_index(
    ["compenv", "slab_id", "att_num"], drop=False)

# #########################################################
df_jobs_oh_anal = get_df_jobs_oh_anal()
df_jobs_oh_anal = df_jobs_oh_anal.set_index(["compenv", "slab_id", "active_site"])

# #########################################################
df_atoms_sorted = get_df_atoms_sorted_ind()

# #########################################################
df_octa_info = get_df_octa_info()

# #########################################################
df_struct_drift = get_df_struct_drift()
# -

# ### Filtering `df_jobs_anal` to only `oer_adsorbate` job types

# +
df_ind = df_jobs_anal_i.index.to_frame()

df_jobs_anal_i = df_jobs_anal_i.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal_i = df_jobs_anal_i.droplevel(level=0)
# -

# ### Filter columns in `df_jobs_anal_i`

# +
from misc_modules.pandas_methods import drop_columns

cols_to_keep = [
    'job_id_max',
    # 'timed_out',
    # 'completed',
    # 'brmix_issue',
    # 'job_understandable',
    # 'decision',
    # 'dft_params_new',
    'job_completely_done',
    ]

df_jobs_anal_i = drop_columns(
    df=df_jobs_anal_i,
    columns=cols_to_keep,
    keep_or_drop="keep",
    )


# + active=""
#
#
# -

# ### Grafting `pot_e` and `as_is_nan` to dataframe

# +
def method(row_i):
    # #####################################################
    new_column_values_dict = {
        "pot_e": None,
        "as_is_nan": None,
        }
    # #####################################################
    compenv_i = row_i.name[0]
    slab_id_i = row_i.name[1]
    ads_i = row_i.name[2]
    active_site_i = row_i.name[3]
    att_num_i = row_i.name[4]
    # #####################################################
    job_id_max_i = row_i.job_id_max
    job_completely_done_i = row_i.job_completely_done
    # #####################################################

    as_is_nan = False
    if active_site_i == "NaN":
        as_is_nan = True

    # #####################################################
    row_data_i = df_jobs_data.loc[job_id_max_i]
    # #####################################################
    pot_e_i = row_data_i.pot_e
    rerun_from_oh_i = row_data_i.rerun_from_oh
    # #####################################################


    # #####################################################
    new_column_values_dict["pot_e"] = pot_e_i
    new_column_values_dict["as_is_nan"] = as_is_nan
    new_column_values_dict["rerun_from_oh"] = rerun_from_oh_i
    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    # #####################################################
    return(row_i)

df_jobs_anal_i = df_jobs_anal_i.apply(
    method,
    axis=1,
    )
# -

# ### Removing O slabs from dataframe

# +
# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]
# -

# ### Removing rows whose atoms failed to sort

# +
df_atoms_sorted_i = df_atoms_sorted[df_atoms_sorted.index.to_frame().job_type == "oer_adsorbate"] 
df_atoms_sorted_i = df_atoms_sorted_i.droplevel(level=0)

df_atoms_sorted_i = df_atoms_sorted_i[df_atoms_sorted_i.failed_to_sort == True]


# Dropping rows that have failed to sort atoms objects
df_jobs_anal_no_o = df_jobs_anal_no_o.drop(df_atoms_sorted_i.index)
# -

# ### Main Loop

# +
# print("TEMP TEMP TEMP | heuristic__if_lower_e=True | TEMP TEMP TEMP")

# +
# #########################################################
data_dict_list = []
# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal_no_o.groupby(groupby_cols)
for name_i, group in grouped:



# # #########################################################
# for i in range(1):
#     # name_i = ('sherlock', 'kamevuse_75', 49.0)
#     # name_i = ('nersc', 'kererape_22', 88.0, )
#     name_i = ('nersc', 'dakoputu_58', 74.0)
#     # #####################################################
#     group = grouped.get_group(name_i)
# # #########################################################

    # print(name_i)

    # #####################################################
    ads_e_o_i = None; ads_e_oh_i = None
    job_id_o_i = None; job_id_oh_i  = None; job_id_bare_i = None
    any_bare_done = None; any_oh_done = None; any_o_done = None
    any_o_done_with_active_sites = None; all_jobs_in_group_done = None
    # #####################################################

    # #####################################################
    data_dict_i = dict()
    # #####################################################
    name_dict_i = dict(zip(groupby_cols, name_i))
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################


    out_dict = get_group_w_all_ads(
        name=name_i,
        group=group,
        df_jobs_anal_i=df_jobs_anal_i,
        )
    group_i = out_dict["group_i"]
    any_o_done_with_active_sites = out_dict["any_o_done_with_active_sites"]


    all_jobs_in_group_done = group_i.job_completely_done.all()


    out_dict = are_any_ads_done(
        group=group_i,
        )
    any_o_done = out_dict["any_o_done"]
    any_oh_done = out_dict["any_oh_done"]
    any_bare_done = out_dict["any_bare_done"]


    # Check that potential energy is numerical
    for i in group_i.pot_e.tolist():
        if type(i) != float:
            print("A non-numerical potential energy entered WF: ", name_i)


    # Only consider done jobs from here
    group_done_i = group_i[group_i.job_completely_done == True]

    group_ind_i = group_done_i.index.to_frame()

    necessary_ads_present = True
    if "o" not in group_ind_i.ads.tolist():
        necessary_ads_present = False

    if necessary_ads_present:

        oer_trip_i = get_oer_triplet(
            name=name_i,
            group=group_done_i,
            df_jobs_oh_anal=df_jobs_oh_anal,
            df_octa_info=df_octa_info,
            heuristic__if_lower_e=False,
            # heuristic__if_lower_e=True,
            )




        # #################################################
        # Checking if *O is avail and get job id
        o_avail = "o" in oer_trip_i.index.to_frame()["ads"].tolist()
        if o_avail:
            idx = pd.IndexSlice
            row_o_i = oer_trip_i.loc[idx[:, :, "o", :, :], :].iloc[0]
            job_id_o_i = row_o_i.job_id_max
            low_e_not_from_oh__o = row_o_i.low_e_not_from_oh
        else:
            ads_e_o_i = None
            job_id_o_i = None

        # #################################################
        # Checking if *OH is avail and get job id
        oh_avail = "oh" in oer_trip_i.index.to_frame()["ads"].tolist()
        if oh_avail:
            idx = pd.IndexSlice
            row_oh_i = oer_trip_i.loc[idx[:, :, "oh", :, :], :].iloc[0]
            job_id_oh_i = row_oh_i.job_id_max
        else:
            ads_e_oh_i = None
            job_id_oh_i = None


        # #################################################
        # Can only compute adsorption energies if bare (*) is avail
        bare_avail = "bare" in oer_trip_i.index.to_frame()["ads"].tolist()
        if bare_avail:
            idx = pd.IndexSlice
            row_bare_i = oer_trip_i.loc[idx[:, :, "bare", :, :], :].iloc[0]
            job_id_bare_i = row_bare_i.job_id_max
            low_e_not_from_oh__bare = row_bare_i.low_e_not_from_oh

            df_ads_i = calc_ads_e(oer_trip_i.reset_index())
            df_ads_i = df_ads_i.set_index("ads", drop=False)

            ads_g_o_i = df_ads_i.loc["o"]["ads_e"]
            ads_e_o_i = df_ads_i.loc["o"]["ads_e_elec"]

            if "oh" in df_ads_i.index:
                ads_g_oh_i = df_ads_i.loc["oh"]["ads_e"]
                ads_e_oh_i = df_ads_i.loc["oh"]["ads_e_elec"]
                job_id_oh_i = df_ads_i.loc["oh"]["job_id_max"]
            else:
                ads_g_oh_i = None
                ads_e_oh_i = None
                job_id_oh_i = None

        else:
            job_id_bare_i = None



        # #################################################
        data_dict_i.update(name_dict_i)
        # #################################################
        data_dict_i["g_o"] = ads_g_o_i
        data_dict_i["g_oh"] = ads_g_oh_i
        data_dict_i["e_o"] = ads_e_o_i
        data_dict_i["e_oh"] = ads_e_oh_i
        data_dict_i["job_id_o"] = job_id_o_i
        data_dict_i["job_id_oh"] = job_id_oh_i 
        data_dict_i["job_id_bare"] = job_id_bare_i
        data_dict_i["all_done"] = all_jobs_in_group_done
        data_dict_i["any_bare_done"] = any_bare_done
        data_dict_i["any_oh_done"] = any_oh_done
        data_dict_i["any_o_done"] = any_o_done
        data_dict_i["any_o_w_as_done"] = any_o_done_with_active_sites
        data_dict_i["low_e_not_from_oh__o"] = low_e_not_from_oh__o
        data_dict_i["low_e_not_from_oh__bare"] = low_e_not_from_oh__bare
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################


# #########################################################
df_ads = pd.DataFrame(data_dict_list)
# #########################################################

# + active=""
#
#
#
# -

# ### Writing data to pickle

# Pickling data ###########################################
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/collect_collate_dft_data",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_ads.pickle"), "wb") as fle:
    pickle.dump(df_ads, fle)
# #########################################################

# +
from methods import get_df_ads

df_ads_tmp = get_df_ads()
df_ads_tmp.iloc[0:3]
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("collect_collate_dft.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
