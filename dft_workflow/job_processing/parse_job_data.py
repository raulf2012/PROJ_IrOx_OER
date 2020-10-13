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

# # Pares Job Data
# ---
#
# Applied job analaysis scripts to job directories and compiles.

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import pickle
from pathlib import Path

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
# -

# # Script Inputs

# +
# Rerun job parsing on all existing jobs, needed if job parsing methods are updated
# Check if error is caused by turning this on
# rerun_all_jobs = True
rerun_all_jobs = False

verbose = False
# verbose = True

# +
if rerun_all_jobs:
    print("rerun_all_jobs=True")
    # print("Remember to turn off this flag under normal operation")

compenv = os.environ.get("COMPENV", None)

PROJ_irox_oer_gdrive = os.environ["PROJ_irox_oer_gdrive"]


# -

if compenv != "wsl":
    rerun_all_jobs = True

# # Read Data

# +
# #########################################################
df_jobs_paths = get_df_jobs_paths()

# #########################################################
df_jobs = get_df_jobs(exclude_wsl_paths=True)

# #########################################################
df_jobs_data_clusters = get_df_jobs_data_clusters()


from methods import get_df_jobs_data
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

# # Main Loop

# +
# TEMP
# print("REMOVE THIS, THIS SHOULD NOT STAY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# df_jobs_i = df_jobs_i.loc[["guganono_69"]]

# df_jobs_i = df_jobs_i.iloc[0:10]
# df_jobs_i = df_jobs_i.iloc[0:4]

# df_jobs_i = df_jobs_i.loc[["voburula_03"]]

# +
# TEMP
# print("TEMP filtering df for testing")
# df_jobs_i = df_jobs_i.loc[["vurabamu_02"]]

# TEMP
# print("TEMP | Filtering data")
# # df_jobs_i = df_jobs_i.loc[["fusoreva_23"]]
# df_jobs_i = df_jobs_i.loc[["wipuhite_59"]]

# df_jobs_i

# +
# indices_from_clusters = []
rows_from_clusters = []
rows_from_prev_df = []
data_dict_list = []
for job_id_i, row_i in df_jobs_i.iterrows():
    data_dict_i = dict()

    # #####################################################
    compenv_i = row_i.compenv
    # #####################################################

    # #####################################################
    bulk_id = row_i.bulk_id
    slab_id = row_i.slab_id
    job_id = row_i.job_id
    facet = row_i.facet
    ads = row_i.ads
    compenv_i = row_i.compenv
    compenv_i = row_i.compenv
    att_num = row_i.att_num
    rev_num = row_i.rev_num
    # #####################################################

    # #####################################################
    df_jobs_paths_i = df_jobs_paths[
        df_jobs_paths.compenv == compenv_i]
    row_jobs_paths_i = df_jobs_paths_i.loc[job_id_i]
    # #####################################################
    path_job_root_w_att_rev = row_jobs_paths_i.path_job_root_w_att_rev
    gdrive_path = row_jobs_paths_i.gdrive_path
    # #####################################################

    # #####################################################
    df_jobs_data_clusters_i = df_jobs_data_clusters[
        df_jobs_data_clusters.compenv == compenv_i]
    # #####################################################

    # run_job_i = True
    # job_grabbed_from_clusters = False
    # if job_id_i in df_jobs_data_clusters_i.index:
    #     row_cluster_i = df_jobs_data_clusters_i.loc[job_id_i]

    #     if not rerun_all_jobs:
    #         run_job_i = False
    #         job_grabbed_from_clusters = True
    #         # indices_from_clusters.append(job_id_i)
    #         rows_from_clusters.append(row_cluster_i)

    run_job_i = True
    job_grabbed_from_clusters = False
    job_grabbed_from_prev_df = False
    if not rerun_all_jobs:

        if job_id_i in df_jobs_data_clusters_i.index:
            if verbose:
                print(job_id_i, "Grabbing from df_jobs_data_clusters")
            run_job_i = False
            job_grabbed_from_clusters = True

            row_cluster_i = df_jobs_data_clusters_i.loc[job_id_i]
            rows_from_clusters.append(row_cluster_i)
            # indices_from_clusters.append(job_id_i)

        # if not job_grabbed_from_clusters and job_id_i in df_jobs_data_old.index:
        elif job_id_i in df_jobs_data_old.index:
            if verbose:
                print(job_id_i, "Grabbing from prev df_jobs_data")
            run_job_i = False
            job_grabbed_from_clusters = True

            row_from_prev_df = df_jobs_data_old.loc[job_id_i]
            rows_from_prev_df.append(row_from_prev_df)
        else:
            if verbose:
                print(job_id_i, "Failed to grab job data from elsewhere")



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
        if verbose:
            print("running job")

        data_dict_i["facet"] = facet
        data_dict_i["bulk_id"] = bulk_id
        data_dict_i["slab_id"] = slab_id
        data_dict_i["ads"] = ads
        data_dict_i["job_id"] = job_id
        data_dict_i["compenv"] = compenv_i
        data_dict_i["att_num"] = att_num
        data_dict_i["rev_num"] = rev_num

        # #################################################
        job_err_out_i = parse_job_err(path_full_i, compenv=compenv_i)
        finished_i = parse_finished_file(path_full_i)
        job_state_i = parse_job_state(path_full_i)
        job_submitted_i = is_job_submitted(path_full_i)
        isif_i = get_isif_from_incar(path_full_i)
        num_steps = get_number_of_ionic_steps(path_full_i)
        oszicar_anal = analyze_oszicar(path_full_i)
        incar_params = read_incar(path_full_i, verbose=verbose)
        irr_kpts = get_irr_kpts_from_outcar(path_full_i)
        pickle_data = read_data_pickle(path_full_i)
        init_atoms = get_init_atoms(path_full_i)
        final_atoms = get_final_atoms(path_full_i)
        magmoms_i = get_magmoms_from_job(path_full_i)
        # #################################################

        # path_rel_to_proj_i = row_jobs_paths_i.path_rel_to_proj
        # ads_i = get_ads_from_path(path_job_root_w_att_rev)

        # #################################################
        data_dict_i.update(job_err_out_i)
        data_dict_i.update(finished_i)
        data_dict_i.update(job_state_i)
        data_dict_i.update(job_submitted_i)
        data_dict_i.update(isif_i)
        data_dict_i.update(num_steps)
        data_dict_i.update(oszicar_anal)
        data_dict_i.update(pickle_data)
        data_dict_i["incar_params"] = incar_params
        data_dict_i["irr_kpts"] = irr_kpts
        data_dict_i["init_atoms"] = init_atoms
        data_dict_i["final_atoms"] = final_atoms
        data_dict_i["magmoms"] = magmoms_i
        # data_dict_i["ads_i"] = ads_i
        # #################################################
        data_dict_list.append(data_dict_i)

        # if verbose:
        #     print("")

    elif run_job_i and not path_exists:
        print("A job needed to be processed but couldn't be found locally, or wasn't processed on the cluster")
        print(gdrive_path)
        print(job_id_i)

        

df_jobs_data = pd.DataFrame(data_dict_list)
df_jobs_data_clusters_tmp = pd.DataFrame(rows_from_clusters)
df_jobs_data_from_prev = pd.DataFrame(rows_from_prev_df)
# -

if verbose:
    print("df_jobs_data.shape:", df_jobs_data.shape[0])
    print("df_jobs_data_clusters_tmp.shape:", df_jobs_data_clusters_tmp.shape[0])
    print("df_jobs_data_from_prev.shape:", df_jobs_data_from_prev.shape[0])

# +
# df_jobs_data_clusters_tmp

# +
# assert False
# -

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

# print("df_jobs_data.shape:", df_jobs_data.shape)

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
#     assert len(tmp) == 0, mess_i

    mess_i = "Must be equal"
#     assert df_jobs_data.shape[0] == df_jobs_data_clusters.shape[0], mess_i

    # #########################################################
    df1 = df_jobs_data.drop(columns=["job_state"])
    df2 = df_jobs_data_clusters.job_state

    # df_jobs_data = pd.merge(df1, df2, left_index=True, right_index=True)
    # df_jobs_data = pd.concat([df1, df2], axis=1, join="inner")
    # df_jobs_data = pd.concat([df1, df2], axis=1, )

    df_jobs_data = pd.concat([
        df1,
        df2.loc[df2.index.intersection(df1.index)]
        # df2,
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
# def method(row_i, argument_0, optional_arg=None):
def method(row_i):
    """
    """
    # #####################################################
    final_atoms_i = row_i.final_atoms
    # #####################################################
    # print(row_i)

    if final_atoms_i is not None:
        pot_e_i = final_atoms_i.get_potential_energy()
    else:
        pot_e_i = None

    return(pot_e_i)

df_jobs_data["pot_e"] = df_jobs_data.apply(method, axis=1)
# df_jobs_data.head()
# -

# # Write `df_jobs_data` to file

# +
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
# -

# # Run rclone commnad (if in cluster) to sync `df_jobs_data` to Dropbox

# +
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
# -

# # Printing dataframe, rereading from method

# +
from methods import get_df_jobs_data

df_jobs_data_new = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_new.iloc[0:2]
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("parse_job_data.ipynb")
print(20 * "# # ")
# assert False
# #########################################################

# + active=""
#
#
#

# +
# job_ids =  [
#     'wulumoha_81',
#     'kefasowu_80',
#     'supibepa_48',
#     'kefadusu_22',
#     'butapime_67',
#     'lerosove_43',
#     'kosivele_15',
#     'bomowoto_88',
#     'buwinalo_86',
#     'tonipita_82',
#     'lunosahi_89',
#     'dirigufo_28',
#     'dofusoba_54',
#     'pufanime_49',
#     'sebitubi_78',
#     'betarara_84',
#     'novadesu_38',
#     'tuvavali_48',
#     'mekififo_83',
#     'weratovu_37',
#     'mowavuna_20',
#     'ronafaki_75',
#     'higewedo_88',
#     'wodurowa_29',
#     'kilalawi_94',
#     'kuvapabo_66',
#     'dinibone_46',
#     'kavuvuhe_72',
#     'kusewage_44',
#     'kufuwudi_46',
#     ]

# df_jobs_data.loc[job_ids]

# + jupyter={"source_hidden": true}
# final_atoms_list = df_jobs_data.final_atoms.tolist()

# final_atoms_list = [i for i in final_atoms_list if i != None]
# for final_atoms_i in final_atoms_list:
#     tmp = 42


# final_atoms_i.get_potential_energy()

# + jupyter={"source_hidden": true}
# df_jobs_data

# + jupyter={"source_hidden": true}
# df_jobs_data

# + jupyter={"source_hidden": true}
# # get_df_jobs_data?

# + jupyter={"source_hidden": true}
# df_jobs_i = df_jobs_i[df_jobs_i.compenv != "nersc"]


# + jupyter={"source_hidden": true}
# df_jobs_data_clusters.loc[
#     (df_jobs_data_clusters.compenv == "sherlock") & \
#     (df_jobs_data_clusters.ads == "oh") & \
#     (df_jobs_data_clusters.facet == "010") & \
#     (df_jobs_data_clusters.slab_id == "fogalonu_46") & \
#     [True for i in range(len(df_jobs_data_clusters))]
#     ]

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters_tmp.loc["fusoreva_23"]
# df_jobs_data.loc["fusoreva_23"]
# df_jobs_data_from_prev.loc["fusoreva_23"]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# job_id_i = "vurabamu_02"

# job_id_i in df_jobs_data_clusters_i.index
# job_id_i in df_jobs_data_old.index

# df_jobs_data_clusters_i.loc["vurabamu_02"]

# df_jobs_data_from_prev

# + jupyter={"source_hidden": true}
# df_jobs_data

# + jupyter={"source_hidden": true}
# df_jobs_data

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df1

# + jupyter={"source_hidden": true}
# df_jobs_data_clusters.loc[
#     [('nersc', 'bokedolu_84', 1)]
#     ]


# # bokedolu_84
# # df_jobs.loc["bokedolu_84"]

# + jupyter={"source_hidden": true}
# keys_missing_from_df1

# + jupyter={"source_hidden": true}
# # df_jobs_data
# # df2
# # df1.index




# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df2

# + jupyter={"source_hidden": true}
# # pd.concat?

# + jupyter={"source_hidden": true}
# df_jobs_data
