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

# # Job accounting attempt 3
# ---

# ### Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import copy
from collections import Counter 

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_slab,
    get_other_job_ids_in_set,
    get_df_active_sites,
    get_df_slabs_to_run,
    get_df_features_targets,
    )
# -

import plotly.graph_objs as go

# ### Read Data

# + jupyter={"source_hidden": true}
df_jobs = get_df_jobs()

df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

df_slab = get_df_slab()

df_active_sites = get_df_active_sites()

df_slabs_to_run = get_df_slabs_to_run()
df_slabs_to_run = df_slabs_to_run.set_index("slab_id")

df_features_targets = get_df_features_targets()

# + active=""
#
#

# +
df_slab_i = df_slab[df_slab.phase == 2]

slab_ids = df_slab_i.index.tolist()

# +
df_jobs_i = df_jobs[
    df_jobs.slab_id.isin(slab_ids)
    ]

df_jobs_anal_i = df_jobs_anal[
    df_jobs_anal.index.to_frame().slab_id.isin(slab_ids)
    ]
# -

# # --------------------------------

# ### Job accounting starting from jobs

print(

    # "\n",
    40 * "#",

    "\n",
    "Job accounting (All slabs)",

    "\n",
    40 * "#",
    sep="")

# +
# Getting number of unique bulks
bulk_ids__i = df_slab_i.bulk_id.unique().tolist()

print(
    "There are a total of ",
    df_slab_i.shape[0],
    " slabs (phase 2)",
    sep="")

df_slab_i_2 = df_slab_i[df_slab_i.num_atoms <= 80]

print(
    "  ",
    len(bulk_ids__i),
    " unique bulks represented",

    "\n",
    "  " + 20 * "-",

    "\n",
    "  ",
    df_slab_i_2.shape[0],
    " of the slabs have 80 atoms or less",
    "\n",
    "  ",
    df_slab_i[df_slab_i.num_atoms > 80].shape[0],
    " slabs are > 80 atoms",
    sep="")


ids_that_have_been_run = []
ids_that_have_not_been_run = []
for slab_id_i, row_i in df_slab_i_2.iterrows():
    df_ind_i = df_jobs_anal_i.index.to_frame()
    df_ind_i = df_ind_i[df_ind_i.slab_id == slab_id_i]

    if df_ind_i.shape[0] == 0:
        ids_that_have_not_been_run.append(slab_id_i)
    else:
        ids_that_have_been_run.append(slab_id_i)

# +
# #########################################################
# Slabs that were run and are good to go
# #########################################################

# Getting number of unique bulks
df_slab__i = df_slab.loc[df_slab_i_2.index.tolist()]
bulk_ids__i = df_slab__i.bulk_id.unique().tolist()

print("")
print(

    "Of the ",
    df_slab_i_2.shape[0],
    " slabs that are 80 atoms or less",

    "\n",
    "  ",
    len(bulk_ids__i),
    " unique bulks represented",

    "\n",
    "  " + 20 * "-",

    "\n",
    "  ",
    len(ids_that_have_been_run),
    " slabs have been run",

    "\n",
    "  ",
    len(ids_that_have_not_been_run),
    " slabs have not been run",
    sep="")

# +
# #########################################################
# Slabs that were run and are good to go
# #########################################################

# #########################################################
ids_run__ok = []
ids_run__bad = []
ids_need_to_man_anal = []
for slab_id_i in ids_that_have_been_run:
    if slab_id_i in df_slabs_to_run.index:
        row_slab_i = df_slabs_to_run.loc[slab_id_i]
        status_i = row_slab_i.status

        if status_i == "ok":
            ids_run__ok.append(slab_id_i)
        elif status_i == "bad":
            ids_run__bad.append(slab_id_i)
        else:
            print("bad bad badf sijdifsd998ijsd")

    else:
        ids_need_to_man_anal.append(slab_id_i)

# Getting number of unique bulks
df_slab__been_run = df_slab.loc[ids_that_have_been_run]
bulk_ids__been_run = df_slab__been_run.bulk_id.unique().tolist()


print("")
print(
    "Of the ", len(ids_that_have_been_run), " slabs that were run:",

    "\n",
    "  ",
    len(bulk_ids__been_run),
    " unique bulks represented",

    "\n",
    "  " + 20 * "-",

    "\n",
    "  ",
    len(ids_run__ok),
    " slabs had good *O relaxed structures",

    "\n",
    "  ",
    len(ids_run__bad),
    " slabs had bad *O relaxed slabs (bad struct. drift)",

    "\n",
    "  ",
    len(ids_need_to_man_anal),
    " slabs haven't finished or been run, or haven't been manually analyzed",
    sep="")

# +
# #########################################################
# Slabs that were run and are good to go
# #########################################################

print("")

df_slab__ok = df_slab.loc[ids_run__ok]
bulk_ids__ok = df_slab__ok.bulk_id.unique().tolist()

print(
    "Of the ",
     len(ids_run__ok),
    " slabs that were good:",
     sep="")

print(
    "  ",
    len(bulk_ids__ok),
    " unique bulks represented",

    "\n",
    "  " + 20 * "-",

    "\n",
    "  There are ",
    df_active_sites.loc[ids_run__ok].num_active_sites_unique.sum(),
    " total active sites",


    "\n",
    "  Each slab has ",
    np.round(
        df_active_sites.loc[ids_run__ok].num_active_sites_unique.mean(),
        3),
    " active sites on average",

    sep="")

# +
df_features_targets_i = df_features_targets[df_features_targets["data"]["phase"] > 1]

print("")
print(
    "There are ",
    df_features_targets_i.shape[0],
    " data points in df_features_targets (phase 2 only)",
    sep="")

df_features_targets_i_2 = df_features_targets_i.dropna(
    axis=0,
    subset=[
        ("targets", "g_o", "", ),
        ("targets", "g_oh", "", ), 
        ]
    )

print(
    "  ",
    df_features_targets_i_2.shape[0],
    " of these points have both G_*O and G_*OH",
    sep="")

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# -

# # --------------------------------

# ### Job accounting using good slabs as starting point

print(
    5  * "\n",

    "\n",
    40 * "#",
    
    "\n",
    "Check progress on 'good' slabs",    

    "\n",
    40 * "#",

    sep="")

df_slab_i = copy.deepcopy(df_slab)

# +
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/creating_slabs",
    "out_data")

# #########################################################
import pickle; import os
path_i = os.path.join(
    directory,
    "df_slabs_to_run.pickle")
with open(path_i, "rb") as fle:
    df_slabs_to_run = pickle.load(fle)
# #########################################################
# -

print(
    "There are ",
    df_slabs_to_run.shape[0],
    " slabs that come from octahedral, non-layered, stable (0.3 eV/atom hull cutoff) polymorphs",
    
    "\n",
    "  ",
    df_slabs_to_run.bulk_id.unique().shape[0],
    " bulk polymorphs make of these slabs",
    
    "\n",
    "  ",
    "Each polymorph makes on average ",
    np.round(
        df_slabs_to_run.shape[0] / df_slabs_to_run.bulk_id.unique().shape[0],
        3),
    " slabs",

    sep="")


# +
from methods import read_data_json

data = read_data_json()
systems_that_took_too_long = data.get("systems_that_took_too_long", []) 

# +
good_slab_ids = []
for slab_id_i, row_i in df_slab_i.iterrows():
    bulk_id_i = row_i.bulk_id
    facet_i = row_i.facet


    took_too_long_i = False
    for i in systems_that_took_too_long:
        if i[0] == bulk_id_i and i[1] == facet_i:
            took_too_long_i = True

    # if took_too_long_i:
    #     print("took_too_long_i")

    df = df_slabs_to_run
    df = df[
        (df["bulk_id"] == bulk_id_i) &
        (df["facet_str"] == facet_i) &
        [True for i in range(len(df))]
        ]
    if df.shape[0] > 0:
        good_slab_ids.append(slab_id_i)
    
    else:
        if not took_too_long_i:
            tmp = 42
            # print("What's up with this one:", slab_id_i)

df_slab_i_2 = df_slab_i.loc[good_slab_ids]

# +
# # TEMP | TEMP | TEMP | TEMP | TEMP | TEMP

# bulk_facet_list = []
# for slab_id_i, row_i in df_slab_i_2.iterrows():
#     tmp = 42

#     tup_i = (
#         row_i.bulk_id,
#         row_i.facet,
#         )
#     bulk_facet_list.append(tup_i)

# idx = pd.MultiIndex.from_tuples(bulk_facet_list)

# len(idx.unique().tolist())


# for i in bulk_facet_list:
#     d = Counter(bulk_facet_list)
#     if d[i] > 1:
#         print(i)

# +
num_took_too_long = 0
for i_cnt, row_i in df_slabs_to_run.iterrows():
    bulk_id_i = row_i.bulk_id
    facet_i = row_i.facet_str

    # #####################################################
    took_too_long_i = False
    for i in systems_that_took_too_long:
        if i[0] == bulk_id_i and i[1] == facet_i:
            took_too_long_i = True


    df = df_slab_i
    df = df[
        (df["bulk_id"] == bulk_id_i) &
        (df["facet"] == facet_i) &
        [True for i in range(len(df))]
        ]

    # if took_too_long_i:
    #     num_took_too_long += 1

    if took_too_long_i and df.shape[0] == 0:
        # print(i_cnt)
        # print("Took too long", bulk_id_i, facet_i)
        num_took_too_long += 1

    if df.shape[0] == 0 and not took_too_long_i:
        tmp = 42
        print(bulk_id_i, facet_i)

#         print("ijisdf")
# -

print(
    "\n",
    df_slab_i_2.shape[0],
    " rows are in df_slab that are from the pristine ",
    df_slabs_to_run.shape[0],
    " set",    

    "\n",
    "  ",
    num_took_too_long,
    " slabs took too long to create and are thus missing",

    "\n",
    "  ",
    df_slabs_to_run.shape[0] - df_slab_i_2.shape[0] - num_took_too_long,
    " are still uncounted for",

    sep="")

# +
# df_slab_i_2.shape

print(

    "\n",
    "Of the 285 pristine slabs:",
    
    "\n",
    "  ",
    df_slab_i_2[df_slab_i_2.num_atoms < 80].shape[0],
    " of them are under 80 atoms",

    "\n",
    "  ",
    df_slab_i_2[df_slab_i_2.num_atoms >= 80].shape[0],
    " of them are over 80 atoms",
    
    sep="")

# +
# df_slab_i_2.num_atoms.max()

# +
cutoff_list = []
num_slabs_list = []
for cutoff_i in range(0, 350, 1):
    cutoff_list.append(cutoff_i)

    num_slabs_i = df_slab_i_2[df_slab_i_2.num_atoms <= cutoff_i].shape[0]
    num_slabs_list.append(num_slabs_i)


x_array = cutoff_list
y_array = num_slabs_list
trace = go.Scatter(
    x=x_array,
    y=y_array,
    )
data = [trace]

fig = go.Figure(data=data)
# fig.show()

# +
# assert False

# +
df_slab_i_3 = df_slab_i_2[df_slab_i_2.num_atoms < 80]

df_ind_i = df_jobs_anal.index.to_frame()

num_have_been_run = 0
num_have_not_been_run = 0
for slab_id_i, row_i in df_slab_i_3.iterrows():

    df = df_ind_i
    df = df[
        (df["slab_id"] == slab_id_i) &
        (df["ads"] == "o") &
        (df["active_site"] == "NaN") &
        [True for i in range(len(df))]
        ]

    
    if df.shape[0] == 0:
        num_have_not_been_run += 1
        print(slab_id_i, "|", row_i.phase)
        # print(df.shape[0])

    elif df.shape[0] > 0:
        num_have_been_run += 1
# -

print(
    "\n",
    "Of the ",
    df_slab_i_3.shape[0],
    " slabs that are under 80 atoms:",

    "\n",
    "  ",
    num_have_been_run,
    " slabs have been run",


    "\n",
    "  ",
    num_have_not_been_run,
    " slabs have not been run",

    sep="")

# +
# #########################################################
# These don't have any jobs run for them, why?
# #########################################################

# pumusuma_66 | 1
# fufalego_15 | 1
# tefenipa_47 | 1
# silovabu_91 | 1
# naronusu_67 | 1
# nofabigo_84 | 1
# kodefivo_37 | 1

# NEW | THESE ARE GOOD NOW I THINK
# romudini_21 | 2
# wafitemi_24 | 2
# kapapohe_58 | 2
# bekusuvu_00 | 2
# pemupehe_18 | 2
# hahesegu_39 | 2
# migidome_55 | 2
# semodave_57 | 2

# + active=""
#
#
#
#
#
# -

# # --------------------------------

# # Doing something here forgot what

# +
# df_features_targets_i_2[
#     df_features_targets_i_2["features"]["oh"]["octa_vol"].isnull()
#     ].index.tolist()

df_features_targets_i_2[
    df_features_targets_i_2["features"]["oh"]["octa_vol"].isnull()
    ]

# +
df_index = df_jobs_anal_i.index.to_frame()

df_jobs_anal_o = df_jobs_anal_i.loc[
    df_index[df_index.ads == "o"].index
    ]

df_jobs_anal_o_i = df_jobs_anal_o[df_jobs_anal_o.job_completely_done == False]

indices_to_keep = []
for index_i, row_i in df_jobs_anal_o_i.iterrows():
    decision_i = row_i.decision

    if "PENDING" not in decision_i and "RUNNING" not in decision_i:
        indices_to_keep.append(index_i)

df_jobs_anal_o_i.loc[
    indices_to_keep
    ]

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_slabs_to_run__ok = df_slabs_to_run[df_slabs_to_run.status == "ok"]
# df_slabs_to_run__bad = df_slabs_to_run[df_slabs_to_run.status == "bad"]

# # df_slabs_to_run__bad

# + jupyter={"source_hidden": true}
# # df_active_sites

# ids_that_have_been_run

# + jupyter={"source_hidden": true}
# # df_features_targets_i["targets"]
# # df_features_targets_i[("targets", "g_o")]
# df_features_targets_i.columns.to_list()

# + jupyter={"source_hidden": true}
#     "\n",
#     "  " + 20 * "-",
# print(
#     "  There are ",
#     df_active_sites.loc[ids_run__ok].num_active_sites_unique.sum(),
#     " total active sites",
#     sep="")
# print(
#     "  Each slab has ",
#     np.round(
#         df_active_sites.loc[ids_run__ok].num_active_sites_unique.mean(),
#         3),
#     " active sites on average",
#     sep="")
# -



# + jupyter={"source_hidden": true}
# len(good_slab_ids)

# np.unique(good_slab_ids).shape

# df_slab_i.shape

# df_slab_i.slab_id.unique().shape

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df = df_slab_i_2
# df = df[
#     (df["bulk_id"] == "n36axdbw65") &
#     (df["facet"] == "023") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]
# df

# + jupyter={"source_hidden": true}
# print('{} has occurred {} times'.format(x, d[x])) 
# l = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] 
# l = bulk_facet_list
# x = i

# + jupyter={"source_hidden": true}
# df_slab_i_2.shape

# + jupyter={"source_hidden": true}
# df_slabs_to_run.loc[[
#     211,
#     217,
#     250,
#     256,
#     263,
#     264,
#     265,
#     267,
#     ]]

# + jupyter={"source_hidden": true}
# b583vr8hvw 110
# b583vr8hvw 310
# b583vr8hvw 200
# b583vr8hvw 111
# b583vr8hvw 001

# + jupyter={"source_hidden": true}
# assert False

# +
# # TEMP
# print(111 * "TEMP | ")
# df_slab_i_3 = df_slab_i_3.loc[[
#     "vapopihe_87"    
#     ]]
