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

# # Clean DFT Jobs
# ---
#
# Run in computer cluster to perform a variety of job clean and processing
#
# Currently the following things are done:
#
# 1. Process large `job.out` files, if `job.out` is larger than `job_out_size_limit` than creates new `job.out.new` file removes middle section of file and leaves behind the beginning and end of the original file
# 1. Rclone copy the job directories to the Stanford Google Drive
#
# ## TODO
# * Remove large files if they are newer revisions (Only time you need large VASP files are when starting a new job and therefore need WAVECAR or charge files)

# # Import Modules

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import copy
import shutil
from pathlib import Path
import subprocess
import pickle

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_paths,
    get_df_jobs_anal,
    )

# #########################################################
from local_methods import (
    cwd, process_large_job_out,
    rclone_sync_job_dir,
    parse_job_state,
    local_dir_matches_remote,
    )
# -

from methods import (
    get_other_job_ids_in_set,
    )

# # Script Inputs

# +
verbose = False

job_out_size_limit = 5  # MB

# +
compenv = os.environ.get("COMPENV", None)

proj_dir = os.environ.get("PROJ_irox_oer", None)
# -

# # Read Data

# +
# #########################################################
df_jobs = get_df_jobs(exclude_wsl_paths=False)

if compenv != "wsl":
    df_i = df_jobs[df_jobs.compenv == compenv]
else:
    df_i = df_jobs

# #########################################################
df_jobs_paths = get_df_jobs_paths()
df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv]

# #########################################################
df_jobs_anal = get_df_jobs_anal()

if verbose:
    print(60 * "-")
    print("Directories being parsed")
    tmp = [print(i) for i in df_jobs_paths_i.path_rel_to_proj.tolist()]
    print("")
# -

# # Iterate through rows

if compenv != "wsl":

    iterator = tqdm(df_i.index.tolist(), desc="1st loop")
    for index_i in iterator:
        # #####################################################
        row_i = df_i.loc[index_i]
        # #####################################################
        slab_id_i = row_i.slab_id
        ads_i = row_i.ads
        att_num_i = row_i.att_num
        compenv_i = row_i.compenv
        active_site_i = row_i.active_site
        # #####################################################

        if active_site_i == "NaN":
            tmp = 42
        elif np.isnan(active_site_i):
            active_site_i = "NaN"

        # #####################################################
        df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
        row_jobs_paths_i = df_jobs_paths_i.loc[index_i]
        # #####################################################
        path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
        path_full = row_jobs_paths_i.path_full
        path_rel_to_proj = row_jobs_paths_i.path_rel_to_proj
        gdrive_path_i = row_jobs_paths_i.gdrive_path
        # #####################################################

        # #####################################################
        in_index = df_jobs_anal.index.isin(
            [(compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]).any()
        if in_index:
            row_anal_i = df_jobs_anal.loc[
                compenv_i, slab_id_i, ads_i, active_site_i, att_num_i]
            # #################################################
            job_completely_done_i = row_anal_i.job_completely_done
            # #################################################
        else:
            job_completely_done_i = None

        # if job_completely_done_i:
        #     print("job done:", path_full)

        # #####################################################
        if compenv != "wsl":

            from proj_data import compenvs
            compenv_in_path = None
            for compenv_j in compenvs:
                if compenv_j in path_rel_to_proj:
                    compenv_in_path = compenv_j

            if compenv_in_path is not None:
                new_path_list = []
                for i in path_rel_to_proj.split("/"):
                    if i != compenv_in_path:
                        new_path_list.append(i)
                path_rel_to_proj_new = "/".join(new_path_list)
                path_rel_to_proj = path_rel_to_proj_new


            path_i = os.path.join(
                os.environ["PROJ_irox_oer"],
                path_rel_to_proj)
        else:
            path_i = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                gdrive_path_i)



        # print("path_i:", path_i)

        my_file = Path(path_i)
        if my_file.is_dir():

            # Only do these operations on non-running jobs
            job_state_dict = parse_job_state(path_i)
            job_state_i = job_state_dict["job_state"]

            if verbose:
                print("job_state_i:", job_state_i)

            # #########################################
            if job_state_i != "RUNNING":
                # print("Doing large job processing")
                process_large_job_out(
                    path_i, job_out_size_limit=job_out_size_limit)

            # #########################################
            rclone_sync_job_dir(
                path_job_root_w_att_rev=path_job_root_w_att_rev,
                path_rel_to_proj=path_rel_to_proj,
                verbose=False,
                )

# # Remove left over large job.out files
# For some reason some are left over

if compenv == "wsl":
    iterator = tqdm(df_i.index.tolist(), desc="1st loop")
    for index_i in iterator:
        # #####################################################
        row_i = df_i.loc[index_i]
        # #####################################################
        slab_id_i = row_i.slab_id
        ads_i = row_i.ads
        att_num_i = row_i.att_num
        compenv_i = row_i.compenv
        active_site_i = row_i.active_site
        # #####################################################

        # #####################################################
        df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
        row_jobs_paths_i = df_jobs_paths_i.loc[index_i]
        # #####################################################
        gdrive_path_i = row_jobs_paths_i.gdrive_path
        # #####################################################

        path_i = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i)
        if Path(path_i).is_dir():

            # #############################################
            path_job_short = os.path.join(path_i, "job.out.short")
            if Path(path_job_short).is_file():
                path_job = os.path.join(path_i, "job.out")
                if Path(path_job).is_file():
                    print("Removing job.out", path_i)
                    os.remove(path_job)

            # #############################################
            path_job = os.path.join(path_i, "job.out")
            if Path(path_job).is_file():
                if not Path(path_job_short).is_file():
                    file_size = os.path.getsize(path_job)
                    file_size_mb = file_size / 1000 / 1000
                    
                    if file_size_mb > job_out_size_limit:
                        print("Large job.out, but no job.out.short", path_i)
                        process_large_job_out(
                            path_i, job_out_size_limit=job_out_size_limit)

# # Remove systems that are completely done

print(5 * "\n")
print(80 * "*")
print(80 * "*")
print(80 * "*")
print(80 * "*")
print("Removing job folders/data that are no longer needed")
print("Removing job folders/data that are no longer needed")
print("Removing job folders/data that are no longer needed")
print("Removing job folders/data that are no longer needed")
print("Removing job folders/data that are no longer needed")
print("Removing job folders/data that are no longer needed")
print(2 * "\n")

iterator = tqdm(df_i.index.tolist(), desc="1st loop")
for job_id_i in iterator:
    # #####################################################
    row_i = df_i.loc[job_id_i]
    # #####################################################
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    att_num_i = row_i.att_num
    compenv_i = row_i.compenv
    active_site_i = row_i.active_site
    # #####################################################

    if active_site_i == "NaN":
        tmp = 42
    elif np.isnan(active_site_i):
        active_site_i = "NaN"

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[df_jobs_paths.compenv == compenv_i]
    row_jobs_paths_i = df_jobs_paths_i.loc[job_id_i]
    # #####################################################
    path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
    path_full = row_jobs_paths_i.path_full
    path_rel_to_proj = row_jobs_paths_i.path_rel_to_proj
    gdrive_path_i = row_jobs_paths_i.gdrive_path
    # #####################################################

    # #####################################################
    in_index = df_jobs_anal.index.isin(
        [(compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]).any()
    if in_index:
        row_anal_i = df_jobs_anal.loc[
            compenv_i, slab_id_i, ads_i, active_site_i, att_num_i]
        # #################################################
        job_completely_done_i = row_anal_i.job_completely_done
        # #################################################
    else:
        continue




    path_i = os.path.join(os.environ["PROJ_irox_oer"], path_rel_to_proj)



    delete_job = False

    if not job_completely_done_i:
        df_job_set_i = get_other_job_ids_in_set(job_id_i, df_jobs=df_jobs)

        num_revs_list = df_job_set_i.num_revs.unique()
        assert len(num_revs_list) == 1, "kisfiisdjf"
        num_revs = num_revs_list[0]

        df_jobs_to_delete = df_job_set_i[df_job_set_i.rev_num < num_revs - 1]

        if job_id_i in df_jobs_to_delete.index.tolist():
            delete_job = True

    # #####################################################
    if job_completely_done_i:
        delete_job = True

    if delete_job:

        # #####################################################
        # Check that the directory exists
        my_file = Path(path_i)
        dir_exists = False
        if my_file.is_dir():
            dir_exists = True

        # #####################################################
        # Check if .dft_clean file is present
        dft_clean_file_path = os.path.join(path_i, ".dft_clean")
        my_file = Path(dft_clean_file_path)
        dft_clean_already_exists = False
        if my_file.is_file():
            dft_clean_already_exists = True

        # #####################################################
        if dir_exists:
            # Creating .dft_clean file
            if not dft_clean_already_exists:
                if compenv != "wsl":
                    with open(dft_clean_file_path, "w") as file:
                        file.write("")

        # #####################################################
        # Remove directory
        if dir_exists and dft_clean_already_exists and compenv != "wsl":
            local_dir_matches_remote_i = local_dir_matches_remote(
                path_i=path_i,
                gdrive_path_i=gdrive_path_i,
                )
            print(40 * "*")
            print(path_i)
            if local_dir_matches_remote_i:
                print("Removing:")
                shutil.rmtree(path_i)
            else:
                print("Gdrive doesn't match local")
            print("")

# + active=""
#
#
#
