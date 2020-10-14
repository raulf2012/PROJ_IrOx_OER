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

# # Parse Job Directories
# ---
#
# Meant to be run within one of the computer clusters on which jobs are run (Nersc, Sherlock, Slac). Will `os.walk` through `jobs_root_dir` and cobble together all the job directories and then upload the data to Dropbox.
# This script is meant primarily to get simple job information, for more detailed info run the `parse_job_data.ipynb` notebook.

# # Import Modules

# +
import os
print(os.getcwd())
import sys

from pathlib import Path

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from local_methods import (
    is_attempt_dir,
    is_rev_dir,
    get_job_paths_info,
    )
# -

# # Script Inputs

verbose = False
# verbose = True

# +
compenv = os.environ.get("COMPENV", "wsl")
if verbose:
    print("compenv:", compenv)

if compenv == "wsl":
    # This is a test compenv
    # jobs_root_dir = os.path.join(
    #     os.environ["PROJ_irox_oer"],
    #     "__test__/dft_workflow")
    jobs_root_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        "dft_workflow")

elif compenv == "nersc" or compenv == "sherlock" or compenv == "slac":
    jobs_root_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow")


# -

# # Gathering prelim info, get all base job dirs

def get_path_rel_to_proj(full_path):
    """
    """
    #| - get_path_rel_to_proj
    subdir = full_path

    PROJ_dir = os.environ["PROJ_irox_oer"]

    search_term = PROJ_dir.split("/")[-1]
    ind_tmp = subdir.find(search_term)
    if ind_tmp == -1:
        search_term = "PROJ_irox_oer"
        ind_tmp = subdir.find(search_term)

    path_rel_to_proj = subdir[ind_tmp:]
    path_rel_to_proj = "/".join(path_rel_to_proj.split("/")[1:])



    # subdir = full_path
    # PROJ_dir = os.environ["PROJ_irox_oer"]
    # ind_tmp = subdir.find(PROJ_dir.split("/")[-1])
    # path_rel_to_proj = subdir[ind_tmp:]
    # path_rel_to_proj = "/".join(path_rel_to_proj.split("/")[1:])

    return(path_rel_to_proj)
    #__|

# +
# jobs_root_dir = "/home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/__test__/dft_workflow/run_slabs/run_oh_covered/"

# +
data_dict_list = []
for subdir, dirs, files in os.walk(jobs_root_dir):

    data_dict_i = dict()
    data_dict_i["path_full"] = subdir

    last_dir = jobs_root_dir.split("/")[-1]
    path_i = os.path.join(last_dir, subdir[len(jobs_root_dir) + 1:])
    # path_i = subdir[len(jobs_root_dir) + 1:]


    if "dft_jobs" not in subdir:
        continue
    if ".old" in subdir:
        continue
    if path_i == "":
        continue

    if verbose:
        print(path_i)



    path_rel_to_proj = get_path_rel_to_proj(subdir)
    data_dict_i["path_rel_to_proj"] = path_rel_to_proj

    # # #####################################################
    # # TEMP
    # if not "v59s9lxdxr/131/oh/active_site__86/03_attempt" in path_i:
    #     continue
    # print(path_i)

    
    out_dict = get_job_paths_info(path_i)
    data_dict_i.update(out_dict)

    # Only add job directory if it's been submitted
    my_file = Path(os.path.join(subdir, ".SUBMITTED"))
    if my_file.is_file():
        data_dict_list.append(data_dict_i)
    # #####################################################

if len(data_dict_list) == 0:
    df_cols = [
        "path_full",
        "path_rel_to_proj",
        "path_job_root",
        "path_job_root_w_att_rev",
        "att_num",
        "rev_num",
        "is_rev_dir",
        "is_attempt_dir",
        "path_job_root_w_att",
        "gdrive_path",
        ]

    df = pd.DataFrame(columns=df_cols)
else:
    df = pd.DataFrame(data_dict_list)
    df = df[~df.path_job_root_w_att_rev.isna()]
    df = df.drop_duplicates(subset=["path_job_root_w_att_rev", ], keep="first")
    df = df.reset_index(drop=True)

assert df.index.is_unique, "Index must be unique here"

# +
# path_i = "/home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/sherlock/v59s9lxdxr/131/oh/active_site__86/03_attempt/_01"
# path_i = "/home/raulf2012/rclone_temp/PROJ_irox_oer/dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/sherlock/v59s9lxdxr/131/oh/active_site__86/03_attempt/_01"
# path_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/sherlock/v59s9lxdxr/131/oh/active_site__86/03_attempt/_01"
# path_i = "dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/v59s9lxdxr/131/oh/active_site__86/03_attempt"

# out_dict = 
# get_job_paths_info(path_i)

# +
# assert False

# +
# get_gdrive_job_path

# +
# from local_methods import get_gdrive_job_path

# +
# # def get_job_paths_info(path_i):
# #     """
# #     """
# #| - get_job_paths_info
# out_dict = dict()

# # #####################################################
# start_ind_to_remove = None
# rev_num_i = None
# att_num_i = None
# path_job_root_i = None
# path_job_root_w_att_rev = None
# is_rev_dir_i_i = None
# is_attempt_dir_i = None

# path_job_root_w_att = None
# gdrive_path = None
# # #####################################################


# # #########################################################
# #  Getting the compenv
# compenvs = ["slac", "sherlock", "nersc", ]
# compenv_out = None
# got_compenv_from_path = False
# for compenv_i in compenvs:
#     compenv_in_path = compenv_i in path_i
#     if compenv_in_path:
#         compenv_out = compenv_i
#         got_compenv_from_path = True
# if compenv_out is None:
#     compenv_out = os.environ["COMPENV"]


# # #########################################################
# path_split_i = path_i.split("/")
# # print("path_split_i:", path_split_i)  # TEMP
# for i_cnt, dir_i in enumerate(path_split_i):

#     out_dict_i = is_rev_dir(dir_i)
#     is_rev_dir_i = out_dict_i["is_rev_dir"]
#     rev_num_i = out_dict_i["rev_num"]
#     if is_rev_dir_i:
#         dir_im1 = path_split_i[i_cnt - 1]

#         out_dict_i = is_attempt_dir(dir_im1)
#         is_attempt_dir_i = out_dict_i["is_attempt_dir"]
#         att_num_i = out_dict_i["att_num"]
#         if is_attempt_dir_i:
#             start_ind_to_remove = i_cnt - 1

# if start_ind_to_remove:
#     print("PISDJFIJDSIJFIJDSIJFIDJSIFJIJ")
#     path_job_root_i = path_split_i[:start_ind_to_remove]
#     path_job_root_i = "/".join(path_job_root_i)

#     path_job_root_w_att_rev = path_split_i[:start_ind_to_remove + 2]
#     path_job_root_w_att_rev = "/".join(path_job_root_w_att_rev)

#     path_job_root_w_att = path_split_i[:start_ind_to_remove + 1]
#     path_job_root_w_att = "/".join(path_job_root_w_att)


#     #  print("path_job_root_w_att_rev:", path_job_root_w_att_rev)
#     #  print("path_job_root_i:", path_job_root_i)

#     # print(compenv_out)
#     # if compenv_out is not None:
#     if got_compenv_from_path:
#         # compenv_out = os.environ["COMPENV"]
#         gdrive_path = path_job_root_w_att_rev
#     else:
#         gdrive_path = get_gdrive_job_path(path_job_root_w_att_rev)


# else:
#     pass



# out_dict["compenv"] = compenv_out
# out_dict["path_job_root"] = path_job_root_i
# out_dict["path_job_root_w_att_rev"] = path_job_root_w_att_rev
# out_dict["att_num"] = att_num_i
# out_dict["rev_num"] = rev_num_i
# out_dict["is_rev_dir"] = is_rev_dir_i
# out_dict["is_attempt_dir"] = is_attempt_dir_i
# out_dict["path_job_root_w_att"] = path_job_root_w_att
# out_dict["gdrive_path"] = gdrive_path

# # return(out_dict)
# #__|

# +
# path_job_root_w_att_rev

# +
# df

# +
# # df.gdrive_path.tolist()

# df_i = df[
#     (df.compenv == "sherlock") & \
#     (df.bulk_id == "v59s9lxdxr") & \
#     (df.facet == "131") & \
#     (df.ads == "oh") & \
#     (df.att_num == 3)
#     ]

# df_i.gdrive_path.tolist()

# +
# assert False

# +
# # out_dict = 

# # get_job_paths_info(path_i)

# path_i

# compenvs = ["slac", "sherlock", "nersc", ]
# compenv_out = None
# for compenv_i in compenvs:
#     compenv_in_path = compenv_i in path_i
#     if compenv_in_path:
#         compenv_out = compenv_i
# if compenv_out is None:
#     compenv_out = os.environ["COMPENV"]

# print(compenv_out)

# +
# assert False

# +
# # for subdir, dirs, files in os.walk(jobs_root_dir):

# # data_dict_i = dict()
# # data_dict_i["path_full"] = subdir

# last_dir = jobs_root_dir.split("/")[-1]
# # path_i = 
# os.path.join(last_dir, subdir[len(jobs_root_dir) + 1:])

# +
# last_dir

# jobs_root_dir

# +
#     out_dict = 
# path_i
# get_job_paths_info(path_i) 

# +
# # path_rel_to_proj = 
# # get_path_rel_to_proj(subdir)

# # subdir = full_path

# PROJ_dir = os.environ["PROJ_irox_oer"]

# search_term = PROJ_dir.split("/")[-1]
# ind_tmp = subdir.find(search_term)
# if ind_tmp == -1:
#     search_term = "PROJ_irox_oer"
#     ind_tmp = subdir.find(search_term)

# path_rel_to_proj = subdir[ind_tmp:]
# path_rel_to_proj = "/".join(path_rel_to_proj.split("/")[1:])

# path_rel_to_proj

# # return(path_rel_to_proj)

# +
# print(PROJ_dir)
# print(subdir)

# PROJ_dir.split("/")[-1]

# +
def get_facet_bulk_id(row_i):
    # row_i = df.iloc[0]

    new_column_values_dict = {
        "bulk_id": None,
        "facet": None,
        }

    # #####################################################
    path_job_root = row_i.path_job_root
    # #####################################################



    # #####################################################
    # #####################################################
    # Check if the job is a *O calc (different than other adsorbates)
    if "run_o_covered" in path_job_root or "run_o_covered" in jobs_root_dir:
        path_split = path_job_root.split("/")

        facet_i = path_split[-1]
        bulk_id_i = path_split[-2]
        ads_i = "o"
        # active_site_i = None
        active_site_i = np.nan

        
    # #####################################################
    # #####################################################
    elif "run_bare_oh_covered" in path_job_root or "run_bare_oh_covered" in jobs_root_dir:
        path_split = path_job_root.split("/")

        if "/bare/" in path_job_root:
            ads_i = "bare"
        elif "/oh/" in path_job_root:
            ads_i = "oh"
        else:
            print("Couldn't parse the adsorbate from here")
            ads_i = None

        active_site_parsed = False
        for i in path_split:
            if "active_site__" in i:
                active_site_path_seg = i.split("_")
                active_site_i = active_site_path_seg[-1]
                active_site_i = int(active_site_i)
                active_site_parsed = True
        if not active_site_parsed:
            print("PROBLEM | Couldn't parse active site for following dir:")
            print(path_job_root)

        facet_i = path_split[-3]
        bulk_id_i = path_split[-4]
        # ads_i = "bare"

        # Check that the parsed facet makes sense
        all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_i])
        mess_i = "All characters of parsed facet must be numeric"
        assert all_facet_chars_are_numeric, mess_i

    # #####################################################
    # #####################################################
    elif "run_oh_covered" in path_job_root or "run_oh_covered" in jobs_root_dir:
        path_split = path_job_root.split("/")

        if "/bare/" in path_job_root:
            ads_i = "bare"
        elif "/oh/" in path_job_root:
            ads_i = "oh"
        else:
            print("Couldn't parse the adsorbate from here")
            ads_i = None

        active_site_parsed = False
        for i in path_split:
            if "active_site__" in i:
                active_site_path_seg = i.split("_")
                active_site_i = active_site_path_seg[-1]
                active_site_i = int(active_site_i)
                active_site_parsed = True
        if not active_site_parsed:
            print("PROBLEM | Couldn't parse active site for following dir:")
            print(path_job_root)

        facet_i = path_split[-3]
        bulk_id_i = path_split[-4]

        # Check that the parsed facet makes sense
        all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_i])
        mess_i = "All characters of parsed facet must be numeric"
        assert all_facet_chars_are_numeric, mess_i


    # #####################################################
    # #####################################################
    else:
        print("Couldn't figure out what to do here")
        print(path_job_root)
 
        facet_i = None
        bulk_id_i = None
        ads_i = None

        pass

    # #####################################################
    new_column_values_dict["facet"] = facet_i
    new_column_values_dict["bulk_id"] = bulk_id_i
    new_column_values_dict["ads"] = ads_i
    new_column_values_dict["active_site"] = active_site_i


    # #####################################################
    for key, value in new_column_values_dict.items():
        row_i[key] = value

    return(row_i)

df = df.apply(
    get_facet_bulk_id,
    axis=1)

# +
# df.to_csv("out_data/df_dirs.csv")

# +
# df.gdrive_path.tolist()

df_i = df[
    (df.compenv == "sherlock") & \
    (df.bulk_id == "v59s9lxdxr") & \
    (df.facet == "131") & \
    (df.ads == "oh") & \
    (df.att_num == 3)
    ]

df_i.gdrive_path.tolist()

# +
# assert False

# +
df.att_num = df.att_num.astype(int)
df.rev_num = df.rev_num.astype(int)

# df["compenv"] = compenv

# +
groups = []
grouped = df.groupby(["path_job_root_w_att", ])
for name, group in grouped:
    num_revs = group.shape[0]

    group["num_revs"] = num_revs
    groups.append(group)

if len(groups) == 0:
    pass
else:
    df = pd.concat(groups, axis=0)
# -

# # Reorder columns

# +
new_col_order = [
    "compenv",

    "bulk_id",
    "facet",
    "ads",

    "att_num",
    "rev_num",
    "num_revs",

    "is_rev_dir",
    "is_attempt_dir",

    "path_job_root",
    "path_job_root_w_att_rev",
    ]

df = reorder_df_columns(new_col_order, df)
# -

df

# # Saving data and uploading to Dropbox

# Pickling data ###########################################
import os; import pickle
# directory = "out_data"
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_processing",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
file_name_i = "df_jobs_base_" + compenv + ".pickle"
file_path_i = os.path.join(directory, file_name_i)
with open(file_path_i, "wb") as fle:
    pickle.dump(df, fle)
# #########################################################

# +
# file_path_i

# +
# compenv

# +
db_path = os.path.join(
    "01_norskov/00_git_repos/PROJ_IrOx_OER",
    "dft_workflow/job_processing/out_data" ,
    file_name_i)

rclone_remote = os.environ.get("rclone_dropbox", "raul_dropbox")
bash_comm = "rclone copyto " + file_path_i + " " + rclone_remote + ":" + db_path
if verbose:
    print("bash_comm:", bash_comm)

if compenv != "wsl":
    os.system(bash_comm)
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("parse_job_dirs.ipynb")
print(20 * "# # ")
# assert False
# #########################################################

# +
# df_i = df[
#     (df.compenv == "sherlock") & \
#     (df.bulk_id == "v59s9lxdxr") & \
#     (df.facet == "131") & \
#     (df.ads == "oh") & \
#     (df.att_num == 3)
#     ]
# df_i.gdrive_path.tolist()

# + active=""
#
#
#
#
