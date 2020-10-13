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

import pandas as pd
# pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
# pd.options.display.max_colwidth = 100
# -

from methods import (
    get_df_jobs_anal,
    get_df_jobs,
    get_df_oer_groups,
    get_df_slabs_to_run,
    get_df_active_sites,
    )

# # Read Data

df_jobs_anal = get_df_jobs_anal()
df_jobs = get_df_jobs()
df_oer_groups = get_df_oer_groups()
df_slabs_to_run = get_df_slabs_to_run()
df_active_sites = get_df_active_sites()

df_slabs_to_run = df_slabs_to_run.set_index(["compenv", "slab_id", "att_num", ], drop=False)

# + active=""
#
#

# +
idx = pd.IndexSlice
df_jobs_anal_o = df_jobs_anal.loc[idx[:, :, "o", :, :], :]

# #########################################################
print(
    "Total *O systems: ",
    df_jobs_anal_o.shape[0],
    sep="",
    )

# #########################################################
print(
    "*O systems done: ",
    df_jobs_anal_o[df_jobs_anal_o.job_completely_done == True].shape[0],
    sep="",
    )

# #########################################################
print(
    "*O systems not done: ",
    df_jobs_anal_o[df_jobs_anal_o.job_completely_done == False].shape[0],
    sep="",
    )

# +
tot_num_active_sites = 0

tot_num_active_sites__bad = 0
tot_num_active_sites__ok = 0
tot_num_active_sites__not_proc_man = 0
for name_i, row_i in df_jobs_anal_o.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    row_active_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_active_sites_i.active_sites_unique
    # #####################################################

    num_active_sites_i = len(active_sites_unique_i)

    # Totalling the number of unique active sites
    tot_num_active_sites += num_active_sites_i

    index_j = (compenv_i, slab_id_i, att_num_i, )
    if index_j in df_slabs_to_run.index:
        row_slabs_to_run_i = df_slabs_to_run.loc[index_j]
        status_i = row_slabs_to_run_i.status
    else:
        status_i = "NaN"

    idx = pd.IndexSlice
    # df_jobs_anal_bare = df_jobs_anal.loc[idx[compenv_i, slab_id_i, :, :, :], :]
    df_jobs_anal_bare = df_jobs_anal.loc[idx[compenv_i, slab_id_i, "bare", :, :], :]

    # from IPython.display import display
    # print(40 * "*")
    # display(df_jobs_anal_bare)
    # print(40 * "*")

    num_systems_i = df_jobs_anal_bare.shape[0]

    # print(
    #     name_i,
    #     "|",
    #     status_i,
    #     "|",
    #     # "df_jobs_anal_bare.shape:",
    #     num_systems_i,
    #     "|",
    #     num_active_sites_i,
    #     )


    if status_i == "bad":
        tot_num_active_sites__bad += num_active_sites_i


    elif status_i == "ok":
        tot_num_active_sites__ok += num_active_sites_i

        if num_systems_i != num_active_sites_i:
            tmp = 42
    else:
        # print(name_i)
        # print("Not bad oor ok!!!")
        tot_num_active_sites__not_proc_man += num_active_sites_i


            # #############################################
            # print(
            #     name_i,
            #     "|",
            #     status_i,
            #     "|",
            #     # "df_jobs_anal_bare.shape:",
            #     num_systems_i,
            #     "|",
            #     num_active_sites_i,
            #     )

print("")
print("")
print("tot_num_active_sites:", tot_num_active_sites)
print("tot_num_active_sites__ok:", tot_num_active_sites__ok)
print("tot_num_active_sites__bad:", tot_num_active_sites__bad)

# +
print(
    "Total number of bare (*) systems: ",
    tot_num_active_sites,
    "\n",
    sep="")

print(
    "Completed bare (*) systems: ",
    tot_num_active_sites__ok,
    sep="")

print(
    "Systems not run b.c. *O was bad: ",
    tot_num_active_sites__bad,
    sep="")

print(
    "Systems whose *O is not done or not manually inspected: ",
    tot_num_active_sites__not_proc_man,
    sep="")

# print("tot_num_active_sites__bad:", tot_num_active_sites__bad)
# -

df_jobs_anal_o.iloc[0:2]

# +
# df_slabs_to_run.iloc[0:2]

# df_index_i = df_jobs_anal_o.index.to_frame()


good_indices_to_keep = []
for name_i, row_i in df_jobs_anal_o.iterrows():

    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #########################################################


    key_i = (compenv_i, slab_id_i, att_num_i, )
    if key_i in df_slabs_to_run.index:
        row_slabs_i = df_slabs_to_run.loc[key_i]
        status_i = row_slabs_i.status
    else:
        status_i = None

    if status_i == "ok":
        good_indices_to_keep.append(name_i)

df_jobs_anal_o_ok = df_jobs_anal_o.loc[
    good_indices_to_keep    
    ]

# +
# df_jobs_anal_o_ok__slac = df_jobs_anal_o_ok[df_jobs_anal_o_ok.compenv == "slac"]
# df_jobs_anal_o_ok

idx = pd.IndexSlice
df_jobs_anal_o_ok__slac = df_jobs_anal_o_ok.loc[idx["slac", :, :, :, :], :]

idx = pd.IndexSlice
df_jobs_anal_o_ok__sher = df_jobs_anal_o_ok.loc[idx["sherlock", :, :, :, :], :]

idx = pd.IndexSlice
df_jobs_anal_o_ok__nersc = df_jobs_anal_o_ok.loc[idx["nersc", :, :, :, :], :]
# -

print(40 * "*")
print(40 * "*")
print(40 * "*")
print("Slac")

# + jupyter={"outputs_hidden": true}
tot_num_oh_jobs = 0
tot_num_oh_jobs_2 = 0
tot_num_completed_oh_jobs = 0

tot_num_oh_job_sets = 0
tot_num_job_sets_with_compl_oh = 0
# for name_i, row_i in df_jobs_anal_o_ok.iterrows():
for name_i, row_i in df_jobs_anal_o_ok__slac.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    row_active_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_active_sites_i.active_sites_unique
    # #####################################################

    num_active_sites_i = len(active_sites_unique_i)

    # Totalling the number of unique active sites
    tot_num_active_sites += num_active_sites_i

    for active_site_j in active_sites_unique_i:
        tot_num_oh_job_sets += 1

        df_jobs_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            (df_jobs.ads == "oh") & \
            (df_jobs.active_site == active_site_j) & \
            [True for i in range(len(df_jobs))]
            ]

        num_oh_calcs = df_jobs_i.att_num.unique().shape[0]
        tot_num_oh_jobs += num_oh_calcs
        tot_num_oh_jobs_2 += 4

        # #################################################
        df_index = df_jobs_anal.index.to_frame()
        df_index_j = df_index[
            (df_index.compenv == compenv_i) & \
            (df_index.slab_id == slab_id_i) & \
            (df_index.ads == "oh") & \
            (df_index.active_site == active_site_j) & \
            [True for i in range(len(df_index))]
            ]

        df_jobs_anal_oh_j = df_jobs_anal.loc[
            df_index_j.index    
            ]

        df_i = df_jobs_anal_oh_j[df_jobs_anal_oh_j.job_completely_done == True]
        completed_oh_i = df_i.shape[0]

        tot_num_completed_oh_jobs += completed_oh_i

        if completed_oh_i > 0:
            tot_num_job_sets_with_compl_oh += 1
        else:
            tmp = 42
#             from IPython.display import display
#             print(40 * "*")
#             print(name_i, active_site_j)
#             display(df_jobs_anal_oh_j)

print("tot_num_oh_jobs:", tot_num_oh_jobs)
print("tot_num_oh_jobs_2:", tot_num_oh_jobs_2)
print("tot_num_completed_oh_jobs:", tot_num_completed_oh_jobs)
print("tot_num_oh_job_sets:", tot_num_oh_job_sets)
print("tot_num_job_sets_with_compl_oh:", tot_num_job_sets_with_compl_oh)
# -

print(40 * "*")
print(40 * "*")
print(40 * "*")
print("Sherlock")

# +
tot_num_oh_jobs = 0
tot_num_oh_jobs_2 = 0
tot_num_completed_oh_jobs = 0

tot_num_oh_job_sets = 0
tot_num_job_sets_with_compl_oh = 0
# for name_i, row_i in df_jobs_anal_o_ok.iterrows():
for name_i, row_i in df_jobs_anal_o_ok__sher.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    row_active_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_active_sites_i.active_sites_unique
    # #####################################################

    num_active_sites_i = len(active_sites_unique_i)

    # Totalling the number of unique active sites
    tot_num_active_sites += num_active_sites_i

    for active_site_j in active_sites_unique_i:
        tot_num_oh_job_sets += 1

        df_jobs_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            (df_jobs.ads == "oh") & \
            (df_jobs.active_site == active_site_j) & \
            [True for i in range(len(df_jobs))]
            ]

        num_oh_calcs = df_jobs_i.att_num.unique().shape[0]
        tot_num_oh_jobs += num_oh_calcs
        tot_num_oh_jobs_2 += 4

        # #################################################
        df_index = df_jobs_anal.index.to_frame()
        df_index_j = df_index[
            (df_index.compenv == compenv_i) & \
            (df_index.slab_id == slab_id_i) & \
            (df_index.ads == "oh") & \
            (df_index.active_site == active_site_j) & \
            [True for i in range(len(df_index))]
            ]

        df_jobs_anal_oh_j = df_jobs_anal.loc[
            df_index_j.index    
            ]

        df_i = df_jobs_anal_oh_j[df_jobs_anal_oh_j.job_completely_done == True]
        completed_oh_i = df_i.shape[0]

        tot_num_completed_oh_jobs += completed_oh_i

        if completed_oh_i > 0:
            tot_num_job_sets_with_compl_oh += 1
        else:
            tmp = 42
#             from IPython.display import display
#             print(40 * "*")
#             print(name_i, active_site_j)
#             display(df_jobs_anal_oh_j)

print("tot_num_oh_jobs:", tot_num_oh_jobs)
print("tot_num_oh_jobs_2:", tot_num_oh_jobs_2)
print("tot_num_completed_oh_jobs:", tot_num_completed_oh_jobs)
print("tot_num_oh_job_sets:", tot_num_oh_job_sets)
print("tot_num_job_sets_with_compl_oh:", tot_num_job_sets_with_compl_oh)
# -

print(40 * "*")
print(40 * "*")
print(40 * "*")
print("NERSC")

# +
tot_num_oh_jobs = 0
tot_num_oh_jobs_2 = 0
tot_num_completed_oh_jobs = 0

tot_num_oh_job_sets = 0
tot_num_job_sets_with_compl_oh = 0
# for name_i, row_i in df_jobs_anal_o_ok.iterrows():
# for name_i, row_i in df_jobs_anal_o_ok__sher.iterrows():
for name_i, row_i in df_jobs_anal_o_ok__nersc.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    row_active_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_active_sites_i.active_sites_unique
    # #####################################################

    num_active_sites_i = len(active_sites_unique_i)

    # Totalling the number of unique active sites
    tot_num_active_sites += num_active_sites_i

    for active_site_j in active_sites_unique_i:
        tot_num_oh_job_sets += 1

        df_jobs_i = df_jobs[
            (df_jobs.compenv == compenv_i) & \
            (df_jobs.slab_id == slab_id_i) & \
            (df_jobs.ads == "oh") & \
            (df_jobs.active_site == active_site_j) & \
            [True for i in range(len(df_jobs))]
            ]

        num_oh_calcs = df_jobs_i.att_num.unique().shape[0]
        tot_num_oh_jobs += num_oh_calcs
        tot_num_oh_jobs_2 += 4

        # #################################################
        df_index = df_jobs_anal.index.to_frame()
        df_index_j = df_index[
            (df_index.compenv == compenv_i) & \
            (df_index.slab_id == slab_id_i) & \
            (df_index.ads == "oh") & \
            (df_index.active_site == active_site_j) & \
            [True for i in range(len(df_index))]
            ]

        df_jobs_anal_oh_j = df_jobs_anal.loc[
            df_index_j.index    
            ]

        df_i = df_jobs_anal_oh_j[df_jobs_anal_oh_j.job_completely_done == True]
        completed_oh_i = df_i.shape[0]

        tot_num_completed_oh_jobs += completed_oh_i

        if completed_oh_i > 0:
            tot_num_job_sets_with_compl_oh += 1
        else:
            tmp = 42
#             from IPython.display import display
#             print(40 * "*")
#             print(name_i, active_site_j)
#             display(df_jobs_anal_oh_j)

print("tot_num_oh_jobs:", tot_num_oh_jobs)
print("tot_num_oh_jobs_2:", tot_num_oh_jobs_2)
print("tot_num_completed_oh_jobs:", tot_num_completed_oh_jobs)
print("tot_num_oh_job_sets:", tot_num_oh_job_sets)
print("tot_num_job_sets_with_compl_oh:", tot_num_job_sets_with_compl_oh)

# +
# df_jobs_anal.iloc[0:2]

# active_site_j
# -

assert False

# + jupyter={"source_hidden": true}
    # num_systems_i = df_jobs_anal_bare.shape[0]


    # if status_i == "bad":
    #     tot_num_active_sites__bad += num_active_sites_i

    # elif status_i == "ok":
    #     tot_num_active_sites__ok += num_active_sites_i
    #     if num_systems_i != num_active_sites_i:
    #         tmp = 42
    # else:
    #     tot_num_active_sites__not_proc_man += num_active_sites_i

# print("")
# print("")
# print("tot_num_active_sites:", tot_num_active_sites)
# print("tot_num_active_sites__ok:", tot_num_active_sites__ok)
# print("tot_num_active_sites__bad:", tot_num_active_sites__bad)

# + jupyter={"source_hidden": true}
# df_jobs_i = df_jobs[
#     (df_jobs.compenv == compenv_i) & \
#     (df_jobs.slab_id == slab_id_i) & \
#     (df_jobs.ads == "oh") & \
#     (df_jobs.active_site == active_site_j) & \
#     [True for i in range(len(df_jobs))]
#     ]

# num_oh_calcs = df_jobs_i.att_num.unique().shape[0]
# -

45 + 54

df_jobs_anal_bare

assert False

# +
# index_j = (compenv_i, slab_id_i, att_num_i, )
index_j = ("slac", "garituna_73", )

df_jobs_anal.loc[index_j]
# -

df_jobs[
    (df_jobs.compenv == "slac") & \
    (df_jobs.slab_id == "garituna_73") & \
    [True for i in range(len(df_jobs))]
    ]

assert False

group_cols = ["compenv", "bulk_id", ]
grouped = df_jobs.groupby(group_cols)
for name, group in grouped:
    tmp = 42

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_slabs_to_run

# df_jobs

# df_jobs_anal

# df_active_sites
