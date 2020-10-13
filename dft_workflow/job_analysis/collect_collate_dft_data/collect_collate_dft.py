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
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# # Collect DFT data into *, *O, *OH collections
# ---

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import pickle

import pandas as pd
# pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100
import numpy as np

# #########################################################
from IPython.display import display

# #########################################################
from methods import get_df_jobs_anal
from methods import get_df_jobs_data

# #########################################################
from local_methods import calc_ads_e
# -

# # Script Inputs

verbose = False
# verbose = True

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

# + active=""
#
#
# -

# # Filtering dataframe for testing

# +
# print("TEMP")

# df_index = df_jobs_anal.index.to_frame()

# df_jobs_anal = df_jobs_anal.loc[
#     df_index[df_index.compenv == "slac"].index
#     ]

# +
# #########################################################
# Only completed jobs will be considered
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]
# -

# # Main Loop

# +
# #########################################################
verbose_local = True
# #########################################################

data_dict_list = []
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal_no_o.groupby(groupby_cols)
for name_i, group in grouped:
# for i in range(1):

    # group = grouped.get_group(
    #     ('slac', 'fagumoha_68', 63.0)
    #     )

    if verbose_local:
        print(40 * "*")
        print(name_i)

    data_dict_i = dict()

    # #####################################################
    name_dict_i = dict(zip(groupby_cols, name_i))
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    # Selecting the relevent *O slab rows and combining with group
    idx = pd.IndexSlice
    df_o_slabs = df_jobs_anal_i.loc[idx[compenv_i, slab_id_i, "o", :, :], :]

    group_i = pd.concat([
        df_o_slabs,
        group
        ])

    data_dict_list_j = []
    for name_j, row_j in group_i.iterrows():
        data_dict_j = dict()

        # #################################################
        name_dict_j = dict(zip(list(group_i.index.names), name_j))
        # #################################################
        job_id_max_j = row_j.job_id_max
        # #################################################

        # #################################################
        row_data_j = df_jobs_data.loc[job_id_max_j]
        # #################################################
        pot_e_j = row_data_j.pot_e
        # #################################################

        # #################################################
        data_dict_j.update(name_dict_j)
        data_dict_j["pot_e"] = pot_e_j
        data_dict_j["job_id_max"] = job_id_max_j
        # #################################################
        data_dict_list_j.append(data_dict_j)
        # #################################################

    df_tmp = pd.DataFrame(data_dict_list_j)
    df_tmp = df_tmp.set_index(
        ["compenv", "slab_id", "ads", "active_site", ],
        drop=False)

    # #####################################################
    df_ads_o = df_tmp[df_tmp.ads == "o"]
    df_ads_oh = df_tmp[df_tmp.ads == "oh"]
    df_ads_bare = df_tmp[df_tmp.ads == "bare"]

    if (df_ads_o.shape[0] > 1) or (df_ads_oh.shape[0] > 1) or (df_ads_bare.shape[0] > 1):
        print("There is more than 1 row per state here, need a better way to select")
    # #####################################################


    # If there isn't a bare * calculation then skip for now
    if df_ads_bare.shape[0] == 0:
        if verbose_local:
            print("No bare slab available")
        continue

    df_ads_oh = df_ads_oh[df_ads_oh.pot_e == df_ads_oh.pot_e.min()]
    df_ads_o = df_ads_o[df_ads_o.pot_e == df_ads_o.pot_e.min()]
    df_ads_bare = df_ads_bare[df_ads_bare.pot_e == df_ads_bare.pot_e.min()]

    job_id_bare_i = df_ads_bare.iloc[0].job_id_max

    df_ads_i = pd.concat([
        df_ads_o,
        df_ads_oh,
        df_ads_bare])
    df_ads_i = calc_ads_e(df_ads_i)



    # #########################################################
    row_oh_i = df_ads_i[df_ads_i.ads == "oh"]
    if row_oh_i.shape[0] == 1:
        row_oh_i = row_oh_i.iloc[0]
        # #####################################################
        ads_e_oh_i = row_oh_i.ads_e
        job_id_oh_i = row_oh_i.job_id_max
        # #####################################################
    else:
        ads_e_oh_i = None
        job_id_oh_i = None

    # #########################################################
    row_o_i = df_ads_i[df_ads_i.ads == "o"]
    if row_o_i.shape[0] == 1:
        row_o_i = row_o_i.iloc[0]
        # #####################################################
        ads_e_o_i = row_o_i.ads_e
        job_id_o_i = row_o_i.job_id_max
        # #####################################################
    else:
        ads_e_o_i = None
        job_id_o_i = None



    print(80 * "#")




    # #####################################################
    data_dict_i.update(name_dict_i)
    data_dict_i["g_o"] = ads_e_o_i
    data_dict_i["g_oh"] = ads_e_oh_i
    data_dict_i["job_id_o"] = job_id_o_i
    data_dict_i["job_id_oh"] = job_id_oh_i 
    data_dict_i["job_id_bare"] = job_id_bare_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################


    # display(df_tmp_1)
    # if df_ads_i.shape[0] == 3:
    #     break


    if verbose_local:
        print("")

# #########################################################
df_ads = pd.DataFrame(data_dict_list)

# df_ads.iloc[0:3]

# +
df_ads_i = df_ads[~df_ads.g_oh.isnull()]
print(df_ads_i.shape)

df_ads_i

# +
# assert False
# -

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

df_ads_tmp.head()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_ads

# assert False
