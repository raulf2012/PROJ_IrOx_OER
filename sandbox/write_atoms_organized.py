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

# # Writing data objects to file in a convienent and organized way
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys

import shutil

import pandas as pd

from ase import io

from IPython.display import display
# -

from methods import (
    get_df_jobs_paths,
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_slabs_oh,
    get_df_init_slabs,
    get_df_magmoms,
    )

# # Read Data

df_dft = get_df_dft()
df_job_ids = get_df_job_ids()
df_jobs = get_df_jobs(exclude_wsl_paths=True)
df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)
df_jobs_data_clusters = get_df_jobs_data_clusters()
df_slab = get_df_slab()
df_slab_ids = get_df_slab_ids()
df_jobs_anal = get_df_jobs_anal()
df_jobs_paths = get_df_jobs_paths()
df_slabs_oh = get_df_slabs_oh()
df_init_slabs = get_df_init_slabs()
df_magmoms = get_df_magmoms()

# # Writing finished *O slabs to file

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

var = "o"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

for i_cnt, (name_i, row_i) in enumerate(df_jobs_anal_i.iterrows()):

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_max_i]
    # #####################################################
    gdrive_path = row_paths_i.gdrive_path
    # #####################################################

    in_dir = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path)
    in_path = os.path.join(in_dir, "final_with_calculator.traj")

    out_dir = os.path.join("out_data/completed_O_slabs")
    # out_file = str(i_cnt).zfill(3) + "_" + job_id_max_i + ".traj"
    out_file = str(i_cnt).zfill(3) + "_" + compenv_i + "_" + slab_id_i + "_" + str(att_num_i).zfill(2) + ".traj"
    out_path = os.path.join(out_dir, out_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    shutil.copyfile(
        in_path,
        out_path,
        )
# -

# # Write OER sets to file

# +
from methods import get_df_oer_groups

df_oer_groups = get_df_oer_groups()

# +
# "vinamepa_43" in df_oer_groups.slab_id.tolist()
# -

# #########################################################
# #########################################################
for name_i, row_i in df_oer_groups.iterrows():

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################
    df_jobs_anal_index_i = row_i.df_jobs_anal_index
    # #####################################################

    # Create directory
    folder_i = compenv_i + "_" + slab_id_i + "_" + str(int(active_site_i)).zfill(3)
    out_dir = os.path.join(
        os.environ["PROJ_irox_oer"],
        "sandbox",
        "out_data/oer_sets",
        folder_i)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # #####################################################
    df_jobs_anal_i = df_jobs_anal.loc[df_jobs_anal_index_i]
    # #####################################################
    for name_j, row_j in df_jobs_anal_i.iterrows():

        # #################################################
        compenv_j = name_j[0]
        slab_id_j = name_j[1]
        ads_j = name_j[2]
        active_site_j = name_j[3]
        att_num_j = name_j[4]
        # #################################################
        job_id_max_i = row_j.job_id_max
        # #################################################

        # #################################################
        row_paths_i =  df_jobs_paths.loc[job_id_max_i]
        # #################################################
        gdrive_path_i = row_paths_i.gdrive_path
        # #################################################

        # #################################################
        # Copy final_with_calculator.traj to local dirs
        in_dir = os.path.join(
            os.environ["PROJ_irox_oer_gdrive"],
            gdrive_path_i)
        in_path = os.path.join(
            in_dir,
            "final_with_calculator.traj")

        # file_name_j = ads_j + "_" + str(att_num_j).zfill(2) + ".traj"
        file_name_j = ads_j + "_" + str(att_num_j).zfill(2)
        out_path = os.path.join(
            out_dir,
            file_name_j + ".traj")

        shutil.copyfile(
            in_path,
            out_path,
            )

        # #################################################
        # Write .cif version
        atoms_i = io.read(in_path)
        atoms_i.write(os.path.join(out_dir, file_name_j + ".cif"))

# + active=""
#
#

# + jupyter={"source_hidden": true}
# compenv_i = name_i[0]
# slab_id_i = name_i[1]
# active_site_i = name_i[2]

# + jupyter={"source_hidden": true}
# group_wo

# + jupyter={"source_hidden": true}
# group_wo.reset_index(level=["compenv", "slab_id", "active_site", ])

# # group_wo.reset_index?

# + jupyter={"source_hidden": true}
# df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# # var = "o"
# # df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

# for i_cnt, (name_i, row_i) in enumerate(df_jobs_anal_i.iterrows()):

#     # #####################################################
#     compenv_i = name_i[0]
#     slab_id_i = name_i[1]
#     ads_i = name_i[2]
#     active_site_i = name_i[3]
#     att_num_i = name_i[4]
#     # #####################################################

#     # #####################################################
#     job_id_max_i = row_i.job_id_max
#     # #####################################################

#     # #####################################################
#     row_paths_i = df_jobs_paths.loc[job_id_max_i]
#     # #####################################################
#     gdrive_path = row_paths_i.gdrive_path
#     # #####################################################

# + jupyter={"source_hidden": true}
# df_jobs_anal_done = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# var = "o"
# df_jobs_anal_i = df_jobs_anal_done.query('ads != @var')

# # #########################################################
# data_dict_list = []
# # #########################################################
# grouped = df_jobs_anal_i.groupby(["compenv", "slab_id", "active_site", ])
# for name, group in grouped:
#     data_dict_i = dict()

#     # #####################################################
#     compenv_i = name[0]
#     slab_id_i = name[1]
#     active_site_i = name[2]
#     # #####################################################

#     idx = pd.IndexSlice
#     df_jobs_anal_o = df_jobs_anal_done.loc[
#         idx[compenv_i, slab_id_i, "o", "NaN", :],
#         ]

#     # #########################################################
#     group_wo = pd.concat([
#         df_jobs_anal_o,
#         group,
#         ])

#     # display(group_wo)

#     # #########################################################
#     df_jobs_anal_index = group_wo.index.tolist()


#     # #########################################################
#     df_index_i = group_wo.index.to_frame()

#     ads_list = df_index_i.ads.tolist()
#     ads_list_unique = list(set(ads_list))

#     o_present = "o" in ads_list_unique
#     oh_present = "oh" in ads_list_unique
#     bare_present = "bare" in ads_list_unique

#     all_ads_present = False
#     if o_present and oh_present and bare_present:
#         all_ads_present = True




#     # #####################################################
#     data_dict_i["compenv"] = compenv_i
#     data_dict_i["slab_id"] = slab_id_i
#     data_dict_i["active_site"] = active_site_i
#     data_dict_i["df_jobs_anal_index"] = df_jobs_anal_index
#     data_dict_i["ads_list"] = ads_list
#     data_dict_i["all_ads_present"] = all_ads_present
#     # data_dict_i[""] = 
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################

# # #########################################################
# df_oer_groups = pd.DataFrame(data_dict_list)
# df_oer_groups = df_oer_groups.set_index(["compenv", "slab_id", "active_site"], drop=False)

# + jupyter={"source_hidden": true}
# df_oer_groups.head()
