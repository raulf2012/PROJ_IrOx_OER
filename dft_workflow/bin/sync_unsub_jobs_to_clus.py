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

# # Sync recently created, unsubmitted jobs to GDrive
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

from pathlib import Path
from multiprocessing import Pool
from functools import partial
import subprocess

import numpy as np
import pandas as pd

from ase import io

# #########################################################
from dft_workflow_methods import (
    get_path_rel_to_proj,
    get_job_paths_info,
    get_job_spec_dft_params,
    get_job_spec_scheduler_params,
    submit_job,
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

# +
root_dir = os.getcwd()

compenv = os.environ["COMPENV"]

if compenv == "wsl":
    root_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow")    
# -

# # Run sync script if on cluster

if compenv != "wsl":
    bash_file_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "scripts/rclone_commands/rclone_proj_repo.sh")

    result = subprocess.run(
        [bash_file_path],
        stdout=subprocess.PIPE)


    out_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/bin")
    bash_script_path = os.path.join(
        out_path,
        "out_data/bash_sync_out.sh")

    os.chmod(bash_script_path, 0o777)

    result = subprocess.run(
        [bash_script_path, ],
        shell=True,
        stdout=subprocess.PIPE,
        )

# +
# assert False
# -

# # Parse directories

# +
from dft_workflow_methods import parse_job_dirs

df = parse_job_dirs(root_dir=root_dir)


# +
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
        if os.environ["COMPENV"] == "wsl":
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
# -

# # Preparing rclone commands to run on the cluster

# +
df_i = df[df.is_submitted == False]

if compenv == "wsl":
    bash_comm_files_line_list = []
    grouped = df_i.groupby(["compenv", ])
    for i_cnt, (name, group) in enumerate(grouped):
        if verbose:
            print(40 * "=")
            print(name)
            print(40 * "=")

        if i_cnt == 0:
            bash_if_statement = 'if [[ "$COMPENV" == "' + name + '" ]]; then'
        else:
            bash_if_statement = 'elif [[ "$COMPENV" == "' + name + '" ]]; then'

        bash_comm_files_line_list.append(bash_if_statement)

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

            if verbose is False:
                quiet_prt = "--quiet "
            else:
                quiet_prt = ""

            # #########################################################
            # Constructing Rclone command
            rclone_comm = "" + \
                "rclone copy " + \
                quiet_prt + \
                " \\" + \
                "\n" + \
                "$rclone_gdrive_stanford:norskov_research_storage/00_projects/PROJ_irox_oer/" + \
                path_job_root_w_att_rev + \
                " \\" + \
                "\n" + \
                "$PROJ_irox_oer/" + \
                clust_path + \
                ""


            # rclone_comm += "--quiet"

            if verbose:
                print(rclone_comm)
    
            tmp = "$PROJ_irox_oer/" + clust_path
            if verbose:
                print(tmp)

                # " \\" + \

            # bash_comm_files_line_list.append("    " + rclone_comm)
            bash_comm_files_line_list.append(rclone_comm)

            # print(rclone_comm)
        if verbose:
            print("")






    # #####################################################
    bash_comm_files_line_list.append("fi")

    my_list = bash_comm_files_line_list
    out_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/bin")
    out_file = os.path.join(
        out_path,
        "out_data/bash_sync_out.sh")
    with open(out_file, "w") as fle:
        for item in my_list:
            fle.write("%s\n" % item)
    # os.chmod(out_file, 777)
    os.chmod(out_file, 0o777)

# +
# assert False

# + active=""
#
#
#
#
# -

# # Rclone local dirs to gdrive

# +
# compenv

# gdrive_daemon

# +
# gdrive_daemon = os.environ.get("GDRIVE_DAEMON", False)

gdrive_daemon = os.environ.get("GDRIVE_DAEMON", False)
if gdrive_daemon == "True":
    gdrive_daemon = True
elif gdrive_daemon == "False":
    gdrive_daemon = False

if compenv == "wsl" and not gdrive_daemon:
    print("Syncing new files to GDrive using rclone")
    variables_dict = dict(kwarg_0="kwarg_0")

    def method_wrap(
        input_dict,
        kwarg_0=None,
        ):
        bash_comm_i = input_dict["bash_comm"]
        result = subprocess.run(
            bash_comm_i.split(" "),
            stdout=subprocess.PIPE)

    input_list = []
    for ind_i, row_i in df_i.iterrows():
        path_job_root_w_att_rev = row_i.path_job_root_w_att_rev
        if verbose:
            print(path_job_root_w_att_rev)

        rclone_comm_flat = "" + \
            "rclone copy" + \
            " " + \
            os.environ["PROJ_irox_oer_gdrive"] + "/" + \
            path_job_root_w_att_rev + \
            " " + \
            os.environ["rclone_gdrive_stanford"] + ":norskov_research_storage/00_projects/PROJ_irox_oer/" + \
            path_job_root_w_att_rev + \
            ""

        input_dict_i = dict(bash_comm=rclone_comm_flat)
        input_list.append(input_dict_i)


    traces_all = Pool().map(
        partial(
            method_wrap,  # METHOD
            **variables_dict,  # KWARGS
            ),
        input_list,
        )
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("sync_unsub_jobs_to_clus.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
