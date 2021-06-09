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

# # Measuring the amount of computational resources used
# ---
#
# TODO:
#   * I haven't incorporated the *O slabs because they don't cut cleany across OER groups

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from pathlib import Path

import pandas as pd
import numpy as np

# #########################################################
from local_methods import calculate_loop_time_outcar
# -

# # Script Inputs

verbose = False

# # Read Data

# +
from methods import get_df_jobs_paths, get_df_jobs
from methods import get_df_ads

df_jobs_paths = get_df_jobs_paths()
df_jobs = get_df_jobs()
df_ads = get_df_ads()
# -

df_ads_i = df_ads[~df_ads.g_oh.isnull()]
# df_ads_i

data_dict_list = []
total_time = 0.
for i_cnt, row_i in df_ads_i.iterrows():

    # #########################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    job_id_o_i = row_i.job_id_o
    job_id_oh_i = row_i.job_id_oh
    job_id_bare_i = row_i.job_id_bare
    # #########################################################


    df_jobs_i = df_jobs[
        (df_jobs.compenv == compenv_i) & \
        (df_jobs.slab_id == slab_id_i) & \
        (df_jobs.active_site == active_site_i) & \
        [True for i in range(len(df_jobs))]
        ]

    for job_id_j, row_j in df_jobs_i.iterrows():
        data_dict_i = dict()

        # #####################################################
        row_paths_i = df_jobs_paths.loc[job_id_j]
        # #####################################################
        gdrive_path_i = row_paths_i.gdrive_path
        # #####################################################

        # print(gdrive_path_i)
        # gdrive_path_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/sherlock/9573vicg7f/110/oh/active_site__98/00_attempt/_02"

        full_path_i = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i)

        outcar_path_i =  os.path.join(
            full_path_i,
            "OUTCAR")

        my_file = Path(outcar_path_i)
        if my_file.is_file():
            loop_time_dict_i = calculate_loop_time_outcar(outcar_path_i)
            # list(loop_time_dict_i.keys())
            total_loop_time = loop_time_dict_i["total_loop_time"]
            loop_time__hr = total_loop_time["hr"]
            num_cores = loop_time_dict_i["num_cores"]
        else:
            print("er567ytgfyuyghuieww3trfyoi")
            loop_time__hr = 0.
        
        total_time += loop_time__hr

        # #################################################
        data_dict_i["job_id"] = job_id_j
        data_dict_i["loop_time__hr"] = loop_time__hr
        data_dict_i["num_cores"] = num_cores
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

# +
# df_jobs.loc[
#     list(set(df_ads_i.job_id_o.tolist() + df_ads_i.job_id_oh.tolist() + df_ads_i.job_id_bare.tolist()))    
#     ]

# +
df_jobs_time = pd.DataFrame(data_dict_list)
df_jobs_time = df_jobs_time.set_index("job_id")

df_jobs_time.loc[df_jobs_i.index].loop_time__hr.sum()

cpu_hours = df_jobs_time["loop_time__hr"] * df_jobs_time["num_cores"]
df_jobs_time["cpu_hours"] = cpu_hours

# df_jobs_time
# -

data_dict_list = []
for i_cnt, row_i in df_ads_i.iterrows():
    data_dict_i = dict()

    # #########################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    # job_id_o_i = row_i.job_id_o
    # job_id_oh_i = row_i.job_id_oh
    # job_id_bare_i = row_i.job_id_bare
    # #########################################################

    df_jobs_i = df_jobs[
        (df_jobs.compenv == compenv_i) & \
        (df_jobs.slab_id == slab_id_i) & \
        (df_jobs.active_site == active_site_i) & \
        [True for i in range(len(df_jobs))]
        ]


    df_jobs_time_i = df_jobs_time.loc[df_jobs_i.index]

    cpu_hours_cum = df_jobs_time_i.cpu_hours.sum()
    # print(cpu_hours_cum)

    # #########################################################
    data_dict_i["cpu_hours_cum"] = cpu_hours_cum
    # ######################################################### 
    data_dict_list.append(data_dict_i)
    # #########################################################

# +
df_cpu_hrs = pd.DataFrame(data_dict_list)

df_cpu_hrs.cpu_hours_cum.mean()

# +
df_cpu_hrs.cpu_hours_cum.sum()

1815590 / 1e6
# -

df_cpu_hrs

# +
3000 * 1000

# 3,000,000
# 3,000,000

# +
# 3e6
# -

assert False

# +
df_jobs_time_i = df_jobs_time.loc[df_jobs_i.index]

df_jobs_time_i["cpu_hours"] = df_jobs_time_i["loop_time__hr"] * df_jobs_time_i["num_cores"]

df_jobs_time_i.cpu_hours.sum()
# -

2443 / 24

# +
df_jobs_time.num_cores.unique()

df_jobs_time

# +
# line_i = " running on   16 total cores"

# split_0 = line_i.split("  ")[-1]

# num_cores = None
# for i in split_0.split(" "):
#     # print(i)
#     if i.isnumeric():
#         # print(i)
#         num_cores = int(i)
#         break

# num_cores

# +
# df_jobs_i

# +
# total_time / 24

# +
# line_i = '     LOOP+:  cpu time18628.7941: real time18676.4463'

# frag_j = "real time"
# ind_j = line_i.find(frag_j)

# line_frag_j = line_i[ind_j + len(frag_j):]

# try:
#     loop_time_j = float(line_frag_j)
# except:
#     loop_time_j = 0.
#     print("Uh oh, no good")
# -

assert False

df_jobs_wo_o = df_jobs[df_jobs.ads != "o"]

# +
# df_jobs_paths
# df_jobs

grouped = df_jobs_wo_o.groupby(["bulk_id", "slab_id", "facet", "compenv", "active_site"])
for name, group in grouped:
    if verbose:
        print(name)
        print(group.shape[0])
        print("")

# +
# grouped.get_group(
#     ('81meck64ba', 'fagumoha_68', '110', 'slac', 63.0)
#     )
# -

group

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# def calculate_loop_time_outcar():
#     """
#     """
#     #| - calculate_loop_time_outcar
#     out_dict = dict()

#     with open(outcar_path_i, "r") as f:
#         lines = f.read().splitlines()

#     lines_2 = []
#     for line_i in lines:
#         frag_i = "LOOP+"
#         if frag_i in line_i:
#             lines_2.append(line_i)

#     loop_time_list = []
#     frag_j = "real time"
#     for line_i in lines_2:
#         tmp = 42

#         # line_i = '     LOOP+:  cpu time18628.7941: real time18676.4463'

#         ind_j = line_i.find(frag_j)

#         line_frag_j = line_i[ind_j + len(frag_j):]

#         try:
#             loop_time_j = float(line_frag_j)
#         except:
#             loop_time_j = 0.
#             print("Uh oh, no good")

#         loop_time_list.append(loop_time_j)

#     loop_time_sum__sec = np.sum(loop_time_list)
#     loop_time_sum__min = loop_time_sum__sec / 60
#     loop_time_sum__hr = loop_time_sum__min / 60

#     loop_time_dict = dict()
#     loop_time_dict["sec"] = loop_time_sum__sec
#     loop_time_dict["min"] = loop_time_sum__min
#     loop_time_dict["hr"] = loop_time_sum__hr

#     # #####################################################
#     out_dict["total_loop_time"] = loop_time_dict
#     out_dict["loop_time_list"] = loop_time_list
#     # #####################################################

#     return(out_dict)
#     #__|
