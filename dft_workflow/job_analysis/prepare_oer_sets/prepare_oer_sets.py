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

# # Preparing OER data sets
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import shutil

import pandas as pd

from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    )
# -

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

# + active=""
#
#

# +
df_jobs_anal_done = df_jobs_anal[df_jobs_anal.job_completely_done == True]

var = "o"
df_jobs_anal_i = df_jobs_anal_done.query('ads != @var')

# #########################################################
data_dict_list = []
# #########################################################
grouped = df_jobs_anal_i.groupby(["compenv", "slab_id", "active_site", ])
for name, group in grouped:
# if True:

    # print("TEMP")
    # name = ('sherlock', 'vuvunira_55', 75.0)
    # group = grouped.get_group(name)

    data_dict_i = dict()

    # #####################################################
    compenv_i = name[0]
    slab_id_i = name[1]
    active_site_i = name[2]
    # #####################################################

    idx = pd.IndexSlice
    df_jobs_anal_o = df_jobs_anal_done.loc[
        idx[compenv_i, slab_id_i, "o", "NaN", :],
        ]

    # #####################################################
    group_wo = pd.concat([
        df_jobs_anal_o,
        group,
        ])

    # display(group_wo)

    # #####################################################
    df_jobs_anal_index = group_wo.index.tolist()


    # #####################################################
    df_index_i = group_wo.index.to_frame()

    ads_list = df_index_i.ads.tolist()
    ads_list_unique = list(set(ads_list))

    num_oh_completed = ads_list.count("oh")

    o_present = "o" in ads_list_unique
    oh_present = "oh" in ads_list_unique
    bare_present = "bare" in ads_list_unique

    all_ads_present = False
    if o_present and oh_present and bare_present:
        all_ads_present = True


    # #####################################################
    # Finding whether *O rerun from *OH exists
    var = "o"
    group_o = group_wo.query('ads == @var')

    df_data_o = df_jobs_data.loc[
        group_o.job_id_max
        ]

    tmp = df_data_o[df_data_o.rerun_from_oh == True]

    # print(name, tmp.shape[0])


    # #####################################################


    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["df_jobs_anal_index"] = df_jobs_anal_index
    data_dict_i["ads_list"] = ads_list
    data_dict_i["ads_list_unique"] = ads_list_unique
    data_dict_i["all_ads_present"] = all_ads_present
    data_dict_i["num_oh_completed"] = num_oh_completed
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_oer_groups = pd.DataFrame(data_dict_list)
df_oer_groups = df_oer_groups.set_index(["compenv", "slab_id", "active_site"], drop=False)
# -

df_oer_groups

assert False

# # Save data to pickle

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/prepare_oer_sets",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(directory, "df_oer_groups.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_oer_groups, fle)
# #########################################################

# +
from methods import get_df_oer_groups

df_oer_groups_tmp = get_df_oer_groups()
# -

df_oer_groups_tmp.head()
# df_oer_groups_tmp

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# "vinamepa_43" in df_oer_groups.slab_id.tolist()

# + jupyter={"source_hidden": true}
# # #########################################################
# import pickle; import os
# directory = os.path.join(
#     os.environ["PROJ_irox_oer"],
#     "dft_workflow/job_analysis/prepare_oer_sets",
#     "out_data")
# path_i = os.path.join(directory, "df_oer_groups.pickle")
# with open(path_i, "rb") as fle:
#     df_oer_groups = pickle.load(fle)
# # #########################################################

# + jupyter={"source_hidden": true}
# vinamepa_43

# df_index_i = df_jobs_anal.index.to_frame()

# "vinamepa_43" in df_index_i.slab_id.tolist()

# + jupyter={"source_hidden": true}
# df_jobs_anal.iloc[0:1]

# + jupyter={"source_hidden": true}
# idx = pd.IndexSlice
# df_jobs_anal.loc[idx[:, "vinamepa_43", :, :, :], :]

# df_jobs_anal.loc[("vinamepa_43", )]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# print(10 * "TEMP ")

# df_index_i = df_jobs_anal.index.to_frame()
# df_index_i = df_index_i[
#     (df_index_i.compenv == "sherlock") & \
#     (df_index_i.slab_id == "kenukami_73") & \
#     (df_index_i.active_site == 84.) & \
#     [True for i in range(len(df_index_i))]
#     ]

# df_jobs_anal = df_jobs_anal.loc[
#     df_index_i.index
#     ]

# + jupyter={"source_hidden": true}
# num_oh_completed = ads_list.count("oh")

# ["oh", "o", "oh"].count("ohd")

# + jupyter={"source_hidden": true}
# # df_jobs_data_i = df_jobs_data.loc[
# #     group_wo.job_id_max
# #     ]
# # df_jobs_data_i

# # group_wo

# var = "o"
# group_o = group_wo.query('ads == @var')

# # group_o

# df_data_o = df_jobs_data.loc[
#     group_o.job_id_max
#     ]

# tmp = df_data_o[df_data_o.rerun_from_oh == True]

# tmp
