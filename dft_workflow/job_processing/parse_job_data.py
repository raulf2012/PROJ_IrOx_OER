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

# # Pares Job Data
# ---
#
# Applied job analaysis scripts to job directories and compiles.

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("display.max_columns", None)
pd.options.display.max_colwidth = 100

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns
from vasp.vasp_methods import read_incar, get_irr_kpts_from_outcar

# #########################################################
from methods import (
    get_df_jobs,
    get_df_jobs_data,
    get_df_jobs_paths,
    get_df_jobs_data_clusters,
    )
from methods import get_df_jobs_data

from local_methods import (
    parse_job_err,
    parse_finished_file,
    parse_job_state,
    is_job_submitted,
    get_isif_from_incar,
    get_number_of_ionic_steps,
    analyze_oszicar,
    read_data_pickle,
    get_final_atoms,
    get_init_atoms,
    get_magmoms_from_job,
    get_ads_from_path,
    )
from local_methods import is_job_started
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# # Script Inputs

# Rerun job parsing on all existing jobs, needed if job parsing methods are updated
rerun_all_jobs = False
# rerun_all_jobs = True

# +
compenv = os.environ.get("COMPENV", None)
if compenv != "wsl":
    rerun_all_jobs = True

if rerun_all_jobs:
    print("rerun_all_jobs=True")
    # print("Remember to turn off this flag under normal operation")

PROJ_irox_oer_gdrive = os.environ["PROJ_irox_oer_gdrive"]
# -

# # Read Data

# +
# #########################################################
df_jobs_paths = get_df_jobs_paths()

# #########################################################
df_jobs = get_df_jobs(exclude_wsl_paths=True)

# #########################################################
df_jobs_data_clusters = get_df_jobs_data_clusters()

# #########################################################
df_jobs_data_old = get_df_jobs_data(exclude_wsl_paths=True, drop_cols=False)

# #########################################################
# Checking if in local env
if compenv == "wsl":
    df_jobs_i = df_jobs
else:
    df_jobs_i = df_jobs[df_jobs.compenv == compenv]
# -

# # Getting job state loop

# +
data_dict_list = []
for job_id_i, row_i in df_jobs_i.iterrows():
    data_dict_i = dict()

    # #####################################################
    compenv_i = row_i.compenv
    # #####################################################

    # #####################################################
    job_id = row_i.job_id
    att_num = row_i.att_num
    # #####################################################

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[
        df_jobs_paths.compenv == compenv_i]
    row_jobs_paths_i = df_jobs_paths_i.loc[job_id_i]
    # #####################################################
    gdrive_path = row_jobs_paths_i.gdrive_path
    path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
    # #####################################################

    data_dict_i["job_id"] = job_id
    data_dict_i["compenv"] = compenv_i
    data_dict_i["att_num"] = att_num

    if compenv == "wsl":
        path_full_i = os.path.join(
            PROJ_irox_oer_gdrive,
            gdrive_path)
    else:
        path_full_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            path_job_root_w_att_rev)

    # #################################################
    job_state_i = parse_job_state(path_full_i)
    data_dict_i.update(job_state_i)
    data_dict_list.append(data_dict_i)
    # #################################################


df_jobs_state = pd.DataFrame(data_dict_list)


# -

def read_dir_data_dict(path_i):
    """
    """
    # ########################################################
    import os
    import json
    from pathlib import Path

    data_path = os.path.join(
        path_i, "data_dict.json")
    my_file = Path(data_path)
    if my_file.is_file():
        with open(data_path, "r") as fle:
            data_dict_i = json.load(fle)
    else:
        data_dict_i = dict()
    # ########################################################
    
    return(data_dict_i)


# # Main Loop

# +
# # TEMP
# print("TEMP filtering df for testing")
# print(222 * "TEMP | ")

# # df_jobs_i = df_jobs_i[df_jobs_i.bulk_id == "64cg6j9any"]

# # df_jobs_i = df_jobs_i.loc[["toberotu_75"]]
# df_jobs_i = df_jobs_i.loc[["seladuri_58"]]

# df_jobs_i
# -

print("Starting the main loop on parse_job_data.py")

# +
# verbose

# +
rows_from_clusters = []
rows_from_prev_df = []
data_dict_list = []
for job_id_i, row_i in df_jobs_i.iterrows():
    # print(job_id_i)
    # #####################################################
    data_dict_i = dict()
    # #####################################################
    bulk_id = row_i.bulk_id
    slab_id = row_i.slab_id
    job_id = row_i.job_id
    facet = row_i.facet
    ads = row_i.ads
    compenv_i = row_i.compenv
    active_site_i = row_i.active_site
    att_num = row_i.att_num
    rev_num = row_i.rev_num
    # #####################################################

    # #####################################################
    row_jobs_paths_i = df_jobs_paths.loc[job_id_i]
    # #####################################################
    path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
    gdrive_path = row_jobs_paths_i.gdrive_path
    # #####################################################

    # #####################################################
    df_jobs_data_clusters_i = df_jobs_data_clusters[
        df_jobs_data_clusters.compenv == compenv_i]
    # #####################################################













    # #####################################################
    # #####################################################
    # #####################################################
    # Deciding to run job or grabbing it from elsewhere
    # #####################################################
    # #####################################################
    # #####################################################
    run_job_i = True
    # job_grabbed_from_clusters = False
    job_grabbed_from_prev_df = False

    if rerun_all_jobs:
        run_job_i = True
    else:

    # if not rerun_all_jobs:

        if job_id_i in df_jobs_data_clusters_i.index:

            run_job_i = False
            job_grabbed_from_clusters = True

            # #############################################
            row_cluster_i = df_jobs_data_clusters_i.loc[job_id_i]
            # #############################################

            completed_i = row_cluster_i.completed

            gdrive_path_i = df_jobs_paths.loc[job_id_i].gdrive_path
            finished_path = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                gdrive_path_i,
                ".FINISHED")

            job_finished = False
            my_file = Path(finished_path)
            if my_file.is_file():
                job_finished = True
                # print("Finished is there")

            if not completed_i and job_finished:
                run_job_i = True    
                job_grabbed_from_clusters = False

            if not run_job_i and job_grabbed_from_clusters:
                if verbose:
                    print(job_id_i, "Grabbing from df_jobs_data_clusters")
                rows_from_clusters.append(row_cluster_i)

        # if not job_grabbed_from_clusters and job_id_i in df_jobs_data_old.index:
        elif job_id_i in df_jobs_data_old.index:

            run_job_i = False
            # job_grabbed_from_clusters = True

            row_from_prev_df = df_jobs_data_old.loc[job_id_i]

            # #############################################
            gdrive_path_i = df_jobs_paths.loc[job_id_i].gdrive_path
            incar_path = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                gdrive_path_i,
                "INCAR")

            # #############################################
            # If the prev INCAR params is None but the incar file is there then rerun
            incar_params_i = row_from_prev_df.incar_params
            incar_file_and_df_dont_match = False
            if incar_params_i is None:
                my_file = Path(incar_path)
                if my_file.is_file():
                    incar_file_and_df_dont_match = True
                    run_job_i = True

            # #############################################

            if not incar_file_and_df_dont_match:
                if verbose:
                    print(job_id_i, "Grabbing from prev df_jobs_data")
                rows_from_prev_df.append(row_from_prev_df)


        else:
            if verbose:
                print(job_id_i, "Failed to grab job data from elsewhere")

    # #####################################################
    # #####################################################
    # #####################################################
    # Deciding to run job or grabbing it from elsewhere
    # #####################################################
    # #####################################################
    # #####################################################










    if compenv == "wsl":
        path_full_i = os.path.join(
            PROJ_irox_oer_gdrive,
            gdrive_path)
    else:
        path_full_i = os.path.join(
            os.environ["PROJ_irox_oer"],
            path_job_root_w_att_rev)

    path_exists = False
    my_file = Path(path_full_i)
    if my_file.is_dir():
        path_exists = True        

    if run_job_i and path_exists:

        print(path_full_i)

        if verbose:
            print("running job")


        # #################################################
        job_err_out_i = parse_job_err(path_full_i, compenv=compenv_i)
        finished_i = parse_finished_file(path_full_i)
        job_state_i = parse_job_state(path_full_i)
        job_submitted_i = is_job_submitted(path_full_i)
        job_started_i = is_job_started(path_full_i)
        isif_i = get_isif_from_incar(path_full_i)
        num_steps = get_number_of_ionic_steps(path_full_i)
        oszicar_anal = analyze_oszicar(path_full_i)
        incar_params = read_incar(path_full_i, verbose=verbose)
        irr_kpts = get_irr_kpts_from_outcar(path_full_i)
        pickle_data = read_data_pickle(path_full_i)
        init_atoms = get_init_atoms(path_full_i)
        final_atoms = get_final_atoms(path_full_i)
        magmoms_i = get_magmoms_from_job(path_full_i)
        data_dict_out_i = read_dir_data_dict(path_full_i)
        # #################################################


        # #################################################
        data_dict_i.update(job_err_out_i)
        data_dict_i.update(finished_i)
        data_dict_i.update(job_state_i)
        data_dict_i.update(job_submitted_i)
        data_dict_i.update(job_started_i)
        data_dict_i.update(isif_i)
        data_dict_i.update(num_steps)
        data_dict_i.update(oszicar_anal)
        data_dict_i.update(pickle_data)
        data_dict_i.update(data_dict_out_i)
        # #################################################
        data_dict_i["facet"] = facet
        data_dict_i["bulk_id"] = bulk_id
        data_dict_i["slab_id"] = slab_id
        data_dict_i["ads"] = ads
        data_dict_i["job_id"] = job_id
        data_dict_i["compenv"] = compenv_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["att_num"] = att_num
        data_dict_i["rev_num"] = rev_num
        data_dict_i["incar_params"] = incar_params
        data_dict_i["irr_kpts"] = irr_kpts
        data_dict_i["init_atoms"] = init_atoms
        data_dict_i["final_atoms"] = final_atoms
        data_dict_i["magmoms"] = magmoms_i
        # #################################################
        data_dict_list.append(data_dict_i)
        # #################################################

    elif run_job_i and not path_exists and compenv == "wsl":
        print("A job needed to be processed but couldn't be found locally, or wasn't processed on the cluster")
        print(job_id_i, "|", gdrive_path)
    # else:
    #     print("Uhhh something didn't go through properly, check out")
        
# #########################################################
df_jobs_data = pd.DataFrame(data_dict_list)
df_jobs_data_clusters_tmp = pd.DataFrame(rows_from_clusters)
df_jobs_data_from_prev = pd.DataFrame(rows_from_prev_df)
# #########################################################

# +
# assert False
# -

if verbose:
    print("df_jobs_data.shape:", df_jobs_data.shape[0])
    print("df_jobs_data_clusters_tmp.shape:", df_jobs_data_clusters_tmp.shape[0])
    print("df_jobs_data_from_prev.shape:", df_jobs_data_from_prev.shape[0])

# # Process dataframe

# +
if df_jobs_data.shape[0] > 0:
    df_jobs_data = reorder_df_columns(["bulk_id", "slab_id", "job_id", "facet", ], df_jobs_data)

    # Set index to job_id
    df_jobs_data = df_jobs_data.set_index("job_id", drop=False)


df_jobs_data_0 = df_jobs_data

# Combine rows processed here with those already processed in the cluster
df_jobs_data = pd.concat([
    df_jobs_data_clusters_tmp,
    df_jobs_data_0,
    df_jobs_data_from_prev,
    ])
# -

# # Grabbing `job_state` column from `df_jobs_data_clusters`

# +
# #########################################################
df_i = df_jobs_data
df_i["unique_key"] = list(zip(df_i["compenv"], df_i["job_id"], df_i["att_num"], ))
df_i = df_i.set_index("unique_key", drop=False)
df_jobs_data = df_i

# #########################################################
df_i = df_jobs_data_clusters
df_i["unique_key"] = list(zip(df_i["compenv"], df_i["job_id"], df_i["att_num"], ))
df_i = df_i.set_index("unique_key", drop=False)
df_jobs_data_clusters = df_i

# #########################################################
df_i = df_jobs_state
df_i["unique_key"] = list(zip(df_i["compenv"], df_i["job_id"], df_i["att_num"], ))
df_i = df_i.set_index("unique_key", drop=False)
df_jobs_state = df_i

df_jobs_state_i = df_jobs_state.drop(columns=["compenv", "job_id", "att_num"])


# #########################################################
if compenv != "wsl":
    df1 = df_jobs_data.drop(columns=["job_state"])
    df2 = df_jobs_state_i.job_state

    df_jobs_data = pd.merge(df1, df2, left_index=True, right_index=True)

    df_jobs_data = df_jobs_data.set_index("job_id", drop=False)

# #########################################################
if compenv == "wsl":

    tmp = df_jobs_data.index.difference(df_jobs_data_clusters.index)
    mess_i = "Must be no differencec between df_jobs_data and df_jobs_data_clusters"
    mess_i += "\n" + "Usually this means you must rerun scripts on cluster"
    # assert len(tmp) == 0, mess_i

    mess_i = "Must be equal"
    # assert df_jobs_data.shape[0] == df_jobs_data_clusters.shape[0], mess_i

    # #########################################################
    df1 = df_jobs_data.drop(columns=["job_state"])
    df2 = df_jobs_data_clusters.job_state

    df_jobs_data = pd.concat([
        df1,
        df2.loc[df2.index.intersection(df1.index)]
        ], axis=1, )

    df_jobs_data = df_jobs_data.set_index("job_id", drop=False)

# +
if verbose:
    print("df1.shape:", df1.shape[0])
    print("df2.shape:", df2.shape[0])
    print("")

keys_missing_from_df2 = []
for key_i in df1.index:
    key_in_df = key_i in df2.index
    if not key_in_df:
        keys_missing_from_df2.append(key_i)
if verbose:
    print("len(keys_missing_from_df2)", len(keys_missing_from_df2))


keys_missing_from_df1 = []
for key_i in df2.index:
    key_in_df = key_i in df1.index
    if not key_in_df:
        keys_missing_from_df1.append(key_i)
if verbose:
    print("len(keys_missing_from_df1)", len(keys_missing_from_df1))


# -

# # Map current `df_job_ids` to `df_jobs_data_clusters`

# # Getting DFT electronic energy

# +
def method(row_i):
    """
    """
    # #####################################################
    final_atoms_i = row_i.final_atoms
    # #####################################################

    if final_atoms_i is not None:
        pot_e_i = final_atoms_i.get_potential_energy()
    else:
        pot_e_i = None

    return(pot_e_i)

df_jobs_data["pot_e"] = df_jobs_data.apply(method, axis=1)
# -

# # Write `df_jobs_data` to file

from pathlib import Path
my_file = Path(os.environ["PROJ_irox_oer_gdrive"])
if my_file.is_dir() or compenv != "wsl":
    # Pickling data ###########################################
    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing",
        "out_data")

    pre_path = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_processing")

    if compenv == "wsl":
        file_name_i = "df_jobs_data.pickle"
        path_i = os.path.join(pre_path, directory, file_name_i)
    else:
        file_name_i = "df_jobs_data_" + compenv + ".pickle"
        path_i = os.path.join(pre_path, directory, file_name_i)

    if not os.path.exists(directory): os.makedirs(directory)
    with open(path_i, "wb") as fle:
        pickle.dump(df_jobs_data, fle)
    # #########################################################

    file_path_i = path_i

    db_path = os.path.join(
        "01_norskov/00_git_repos/PROJ_IrOx_OER",
        "dft_workflow/job_processing/out_data" ,
        file_name_i)

    rclone_remote = os.environ.get("rclone_dropbox", "raul_dropbox")
    bash_comm = "rclone copyto " + file_path_i + " " + rclone_remote + ":" + db_path

    if compenv != "wsl":
        if verbose:
            print("Running rclone command")
            print("bash_comm:", bash_comm)
        os.system(bash_comm)

# # Printing dataframe, rereading from method

# +
from methods import get_df_jobs_data

df_jobs_data_new = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_new.iloc[0:2]
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("parse_job_data.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_jobs_data.shape

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# Run rclone commnad (if in cluster) to sync `df_jobs_data` to Dropbox

# file_path_i = path_i

# db_path = os.path.join(
#     "01_norskov/00_git_repos/PROJ_IrOx_OER",
#     "dft_workflow/job_processing/out_data" ,
#     file_name_i)

# rclone_remote = os.environ.get("rclone_dropbox", "raul_dropbox")
# bash_comm = "rclone copyto " + file_path_i + " " + rclone_remote + ":" + db_path

# if compenv != "wsl":
#     if verbose:
#         print("Running rclone command")
#         print("bash_comm:", bash_comm)
#     os.system(bash_comm)

# + jupyter={"source_hidden": true}
# from pathlib import Path
# incar_file_and_df_dont_match
# incar_params_i
# df_jobs_data

# + jupyter={"source_hidden": true}
# completed_i = row_cluster_i.completed

# gdrive_path_i = df_jobs_paths.loc[job_id_i].gdrive_path
# finished_path = os.path.join(
#     os.environ["PROJ_irox_oer_gdrive"],
#     gdrive_path_i,
#     ".FINISHED")

# job_finished = False
# my_file = Path(finished_path)
# if my_file.is_file():
#     job_finished = True
#     # print("Finished is there")

# if not completed_i and job_finished:
#     run_job_i = True    

# + jupyter={"source_hidden": true}
# row_from_prev_df.completed

# + jupyter={"source_hidden": true}
# completed

# + jupyter={"source_hidden": true}
# # df_jobs_data
# df_jobs_data_clusters_tmp
# # df_jobs_data_from_prev

# + jupyter={"source_hidden": true}
# run_job_i 
# path_exists
