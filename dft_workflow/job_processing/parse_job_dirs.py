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

import time; ti = time.time()
import shutil
import pickle
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

# +
# assert False
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
compenv = os.environ.get("COMPENV", "wsl")

if compenv == "wsl":
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

    return(path_rel_to_proj)
    #__|


# + active=""
#
#
#

# + jupyter={"outputs_hidden": true}
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

    # # TEMP
    # # print("TEMP")
    # # frag_i = "slac/mwmg9p7s6o/11-20"
    # frag_i = "slac/mwmg9p7s6o/11-20/bare/active_site__26/01_attempt"
    # if frag_i not in subdir:
    #     continue
    # print(10 * "Got through | ")
    # print(subdir)












    if verbose:
        print(path_i)

    path_rel_to_proj = get_path_rel_to_proj(subdir)
    out_dict = get_job_paths_info(path_i)

    # Only add job directory if it's been submitted
    my_file = Path(os.path.join(subdir, ".SUBMITTED"))
    submitted = False
    if my_file.is_file():
        submitted = True

    # #####################################################
    data_dict_i.update(out_dict)
    data_dict_i["path_rel_to_proj"] = path_rel_to_proj
    data_dict_i["submitted"] = submitted
    # #####################################################
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
        "submitted",
        ]

    df = pd.DataFrame(columns=df_cols)
else:
    df = pd.DataFrame(data_dict_list)
    df = df[~df.path_job_root_w_att_rev.isna()]
    df = df.drop_duplicates(subset=["path_job_root_w_att_rev", ], keep="first")
    df = df.reset_index(drop=True)

assert df.index.is_unique, "Index must be unique here"
# -

# # DEPRECATED | Moved to fix_gdrive_conflicts.ipynb
#
# ### Removing paths that have the GDrive duplicate syntax in them ' (1)'

# +
# for ind_i, row_i in df.iterrows():
#     path_full_i = row_i.path_full

#     if " (" in path_full_i:
#         print(
#             path_full_i,
#             sep="")

#         # #################################################
#         found_wrong_level = False
#         path_level_list = []
#         for i in path_full_i.split("/"):
#             if not found_wrong_level:
#                 path_level_list.append(i)
#             if " (" in i:
#                 found_wrong_level = True
#         path_upto_error = "/".join(path_level_list)

#         my_file = Path(path_full_i)
#         if my_file.is_dir():
#             size_i = os.path.getsize(path_full_i)
#         else:
#             continue


#         # If it's a small file size then it probably just has the init files and we're good to delete the dir
#         # Seems that all files are 512 bytes in size (I think it's bytes)
#         if size_i < 550:
#             my_file = Path(path_upto_error)
#             if my_file.is_dir():
#                 print("Removing dir:", path_upto_error)
#                 # shutil.rmtree(path_upto_error)
#         else:
#             print(100 * "Issue | ")
#             print(path_full_i)
#             print(path_full_i)
#             print(path_full_i)
#             print(path_full_i)
#             print(path_full_i)

#         print("")

# +
# Removing files with ' (' in name (GDrive duplicates)

# for subdir, dirs, files in os.walk(jobs_root_dir):
#     for file_i in files:
#         if " (" in file_i:
#             file_path_i = os.path.join(subdir, file_i)

#             print(
#                 "Removing:",
#                 file_path_i)
#             # os.remove(file_path_i)

# # os.path.join(subdir, file_i)

# +
# assert False

# +
def get_facet_bulk_id(row_i):
    """
    """
    new_column_values_dict = {
        "bulk_id": None,
        "facet": None,
        }

    # #####################################################
    path_job_root = row_i.path_job_root
    # #####################################################

    # print(path_job_root)

    # #####################################################
    # #####################################################
    # Check if the job is a *O calc (different than other adsorbates)
    if "run_o_covered" in path_job_root or "run_o_covered" in jobs_root_dir:

        path_split = path_job_root.split("/")

        ads_i = "o"
        if "active_site__" in path_job_root:

            facet_i = path_split[-2]
            bulk_id_i = path_split[-3]

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

        else:
            # path_split = path_job_root.split("/")

            facet_i = path_split[-1]
            bulk_id_i = path_split[-2]
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
        char_list_new = []
        for char_i in facet_i:
            if char_i != "-":
                char_list_new.append(char_i)
        facet_new_i = "".join(char_list_new)

        # all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_i])
        # all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_i])
        all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_new_i])
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
        char_list_new = []
        for char_i in facet_i:
            if char_i != "-":
                char_list_new.append(char_i)
        facet_new_i = "".join(char_list_new)

        # all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_i])
        all_facet_chars_are_numeric = all([i.isnumeric() for i in facet_new_i])
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
# assert False

# +
df.att_num = df.att_num.astype(int)
df.rev_num = df.rev_num.astype(int)

# df["compenv"] = compenv
# -

# # Reorder columns

# +
new_col_order = [
    "compenv",

    "bulk_id",
    "facet",
    "ads",

    "submitted",

    "att_num",
    "rev_num",
    "num_revs",

    "is_rev_dir",
    "is_attempt_dir",

    "path_job_root",
    "path_job_root_w_att_rev",
    ]

df = reorder_df_columns(new_col_order, df)

# +
# df

# +
# assert False
# -

# # Saving data and uploading to Dropbox

# Pickling data ###########################################
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
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("parse_job_dirs.ipynb")
print(20 * "# # ")
# assert False
# #########################################################

# + active=""
#
#
#
#

# + jupyter={"source_hidden": true}
# df.bulk_id

# df[df.bulk_id == "64cg6j9any"]

# + jupyter={"source_hidden": true}
# groups = []
# grouped = df.groupby(["path_job_root_w_att", ])
# for name, group in grouped:
#     num_revs = group.shape[0]

#     group["num_revs"] = num_revs
#     groups.append(group)

# if len(groups) == 0:
#     pass
# else:
#     df = pd.concat(groups, axis=0)

# + jupyter={"source_hidden": true}
# jobs_root_dir = "/home/raulf2012/Dropbox/01_norskov/00_git_repos/PROJ_IrOx_OER/__test__/dft_workflow/run_slabs/run_oh_covered/"

# + jupyter={"source_hidden": true}
# path_job_root = "dft_workflow/run_slabs/run_o_covered/out_data/dft_jobs/81meck64ba/110/active_site__63"

# if "run_o_covered" in path_job_root or "run_o_covered" in jobs_root_dir:

#     path_split = path_job_root.split("/")

#     ads_i = "o"
#     if "active_site__" in path_job_root:

#         facet_i = path_split[-2]
#         bulk_id_i = path_split[-3]

#         active_site_parsed = False
#         for i in path_split:
#             if "active_site__" in i:
#                 active_site_path_seg = i.split("_")
#                 active_site_i = active_site_path_seg[-1]
#                 active_site_i = int(active_site_i)
#                 active_site_parsed = True
#         if not active_site_parsed:
#             print("PROBLEM | Couldn't parse active site for following dir:")
#             print(path_job_root)

#     else:
#         # path_split = path_job_root.split("/")

#         facet_i = path_split[-1]
#         bulk_id_i = path_split[-2]
#         # active_site_i = None
#         active_site_i = np.nan
# + active=""
#
#

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# path_full_i = "/mnt/f/GDrive/norskov_research_storage/00_projects/PROJ_irox_oer/dft_workflow/run_slabs/run_bare_oh_covered/out_data/dft_jobs/slac/ck638t75z3/020/bare (1)/active_site__59/01_attempt/_01"

# # find_ind = path_full_i.find(" (")
# # path_full_i[:find_ind + 4]

# found_wrong_level = False
# path_level_list = []
# for i in path_full_i.split("/"):
#     if not found_wrong_level:
#         path_level_list.append(i)
#     if " (" in i:
#         found_wrong_level = True
# path_upto_error = "/".join(path_level_list)

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# dft_workflow/run_slabs/run_oh_covered/out_data/dft_jobs/sherlock/bpc2nk6qz1/101/oh/active_site__37


# + jupyter={"source_hidden": true}
# path_i

# + jupyter={"source_hidden": true}
# df

# + jupyter={"source_hidden": true}
# ['dft_workflow/run_slabs/run_bare_oh_covered/out_data/dft_jobs/slac/mwmg9p7s6o/11-20/bare/active_site__27/01_attempt',
#  'dft_workflow/run_slabs/run_bare_oh_covered/out_data/dft_jobs/slac/slac/mwmg9p7s6o/11-20/bare/active_site__27/01_attempt']

# + jupyter={"source_hidden": true}
# for i, row_i in df.iterrows():
#     tmp = 42

#     frag_i = "7h7yns937p/101/02_attempt"
#     if frag_i in row_i.path_full:
#         tmp = 42
#         print(i)

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# print(20 * "TEMP | ")

# data_dict_list = []
# for subdir, dirs, files in os.walk(jobs_root_dir):
#     data_dict_i = dict()

#     data_dict_i["path_full"] = subdir

#     last_dir = jobs_root_dir.split("/")[-1]
#     path_i = os.path.join(last_dir, subdir[len(jobs_root_dir) + 1:])
#     # path_i = subdir[len(jobs_root_dir) + 1:]

#     print(subdir)

#     if "dft_jobs" not in subdir:
#         continue
#     if ".old" in subdir:
#         continue
#     if path_i == "":
#         continue

# + jupyter={"source_hidden": true}
# print("TEMP")
# jobs_root_dir = "/media/raulf2012/research_backup/PROJ_irox_oer_gdrive/dft_workflow/run_slabs/run_bare_oh_covered/out_data/dft_jobs/slac/mwmg9p7s6o/11-20"

# + jupyter={"source_hidden": true}
# df = df[
#     (df["compenv"] == "slac") &
#     # (df["slab_id"] == "mwmg9p7s6o") &
#     (df["bulk_id"] == "mwmg9p7s6o") &
#     (df["ads"] == "bare") &
#     (df["facet"] == "11-20") &
#     (df["active_site"] == 27.) &
#     [True for i in range(len(df))]
#     ]

# #     bare	48	
# df.path_job_root_w_att.tolist()

# # df

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# assert False
