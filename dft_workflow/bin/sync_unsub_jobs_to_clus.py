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

from pathlib import Path

import pandas as pd

from ase import io

# #########################################################
from dft_workflow_methods import get_path_rel_to_proj
from dft_workflow_methods import get_job_paths_info
from dft_workflow_methods import get_job_spec_dft_params, get_job_spec_scheduler_params
from dft_workflow_methods import submit_job
# -

# # Script Inputs

# +
root_dir = os.getcwd()

compenv = os.environ["COMPENV"]

if compenv == "wsl":
    root_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow")    
        # "dft_workflow/run_slabs/run_o_covered")    
        # "dft_workflow/run_slabs/run_o_covered/out_data.old/dft_jobs/slac")    


# slac_sub_queue = "suncat"  # 'suncat', 'suncat2', 'suncat3'

# +
# print("sys.argv:", "\n", sys.argv)

# print("")
# print("sys.argv:")
# tmp = [print(i) for i in sys.argv]
# print("")
# print("What is this one?", sys.argv[-1])
# print("")


# if sys.argv[-1] == "run":
#     run_jobs = True
#     print("running job isdjifjsiduf89usd089ufg089sady890gyasd9p8yf978asdy89fyasd89yf890asd7890f7890asd7f89sd")
# else:
#     run_jobs = False
# -

# # Parse directories

# +
from dft_workflow_methods import parse_job_dirs

df = parse_job_dirs(root_dir=root_dir)


# +
# def method(row_i, argument_0, optional_arg=None):
def method(row_i):
    new_column_values_dict = {
        "compenv": None,
        }

    cand_clusters = []
    clusters_list = ["nersc", "sherlock", "slac", ]
    for i in row_i.path_job_root.split("/"):
        if i in clusters_list:
            cand_clusters.append(i)

    if len(cand_clusters) == 1:
        cluster_i = cand_clusters[0]
        new_column_values_dict["compenv"] = cluster_i
    else:
        print("Couldn't parse cluster from path")
        print(cand_clusters)

    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value
    return(row_i)

df_i = df
df_i = df_i.apply(
    method,
    axis=1)
df = df_i

# +
from misc_modules.pandas_methods import reorder_df_columns

col_order = [
    "compenv",
    "is_submitted",
    "att_num",
    "rev_num",
    "is_rev_dir",
    "is_attempt_dir",

    "path_full",
    "path_rel_to_proj",
    "path_job_root",
    "path_job_root_w_att_rev",
    "path_job_root_w_att",
    "gdrive_path",
    ]
df = reorder_df_columns(col_order, df)

# +
df_i = df[df.is_submitted == False]

grouped = df_i.groupby(["compenv", ])
for name, group in grouped:
    print(40 * "=")
    print(name)
    print(40 * "=")

    for name_i, row_i in group.iterrows():

        # #########################################################
        path_job_root_w_att_rev = row_i.path_job_root_w_att_rev
        # #########################################################

        # #########################################################
        # Constructing path on cluster (remove cluster from path)
        clust_path_list = []
        for i in path_job_root_w_att_rev.split("/"):
            clusters_list = ["nersc", "sherlock", "slac", ]

            if i not in clusters_list:
                clust_path_list.append(i)

        clust_path = "/".join(clust_path_list)

        # #########################################################
        # Constructing Rclone command
        rclone_comm = "" + \
            "rclone copy " + \
            " \\" + \
            "\n" + \
            "$rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/" + \
            path_job_root_w_att_rev + \
            " \\" + \
            "\n" + \
            "$PROJ_irox_oer/" + \
            clust_path + \
            ""

            # " \\" + \

        print(rclone_comm)
    print("")

# +
# "rclone copy $rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/nersc/b19q9p6k72/101/01_attempt/_02 $PROJ_irox_oer/dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/b19q9p6k72/101/01_attempt/_02"

# # $rclone_gdrive_stanford
# # raul_gdrive_stanford

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_i.iloc[0].to_dict()

# + jupyter={"source_hidden": true}
# row_i = df.iloc[0]

# cand_clusters = []
# clusters_list = ["nersc", "sherlock", "slac", ]
# for i in row_i.path_job_root.split("/"):
#     if i in clusters_list:
#         cand_clusters.append(i)

# if len(cand_clusters) == 1:
#     cluster_i = cand_clusters[0]
# else:
#     print("Couldn't parse cluster from path")
#     print(cand_clusters)

# cluster_i
