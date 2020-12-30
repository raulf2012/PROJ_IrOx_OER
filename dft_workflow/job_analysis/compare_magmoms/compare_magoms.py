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
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# # Collect DFT data into *, *O, *OH collections
# ---

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_atoms_sorted_ind,
    get_df_init_slabs,
    )

# #########################################################
from local_methods import (
    read_magmom_comp_data,
    save_magmom_comp_data,
    process_group_magmom_comp,
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

# # Script Inputs

redo_all_jobs = False
# redo_all_jobs = True

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_init_slabs = get_df_init_slabs()

magmom_data_dict = read_magmom_comp_data()
# +
# #########################################################
# Only completed jobs will be considered
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
# Dropping rows that failed atoms sort, now it's just one job that blew up 
# job_id = "dubegupi_27"
df_failed_to_sort = df_atoms_sorted_ind[
    df_atoms_sorted_ind.failed_to_sort == True]
df_jobs_anal_i = df_jobs_anal_i.drop(labels=df_failed_to_sort.index)

# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]

# +
# assert False

# +
indices_to_keep = []
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal_no_o.groupby(groupby_cols)
for name_i, group in grouped:
    group_index = group.index.to_frame()
    ads_list = list(group_index.ads.unique())
    oh_present = "oh" in ads_list
    bare_present = "bare" in ads_list
    all_req_ads_present = oh_present and bare_present
    if all_req_ads_present:
        indices_to_keep.extend(group.index.tolist())

df_jobs_anal_no_o_all_ads_pres = df_jobs_anal_no_o.loc[
    indices_to_keep    
    ]
df_i = df_jobs_anal_no_o_all_ads_pres
# -
# # Magmom comparison

# ### TEST TEST

# +
# print(20 * "TEMP \n")
# # name_i = ('slac', 'relovalu_12', 24.0)
# name_i = ('sherlock', 'vevarehu_32', 63.0)

# idx = pd.IndexSlice
# df_i = df_i.loc[idx[name_i[0], name_i[1], :, name_i[2], :], :]

# +
# df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

# +
# df_atoms_sorted_ind.job_id

# +
# #########################################################
groups_to_process = []
# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
# #########################################################
iterator = tqdm(grouped, desc="1st loop")
for i_cnt, (name_i, group) in enumerate(iterator):
    # print(name_i)

# if True:
#     name_i = ('sherlock', 'batipoha_75', 36.0)
#     group = grouped.get_group(name_i)

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    df_index = df_jobs_anal_i.index.to_frame()

    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        [True for i in range(len(df_index))]
        ]

    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index    
        ]

    group_w_o = pd.concat([group, row_o_i, ], axis=0)
    # #####################################################
    job_ids_list = group_w_o.job_id_max.tolist()


    # #####################################################
    # Deciding whether to reprocess the job or not
    # #####################################################
    out_dict_i = magmom_data_dict.get(name_i, None)
    # #####################################################
    if out_dict_i is None:
        run_job = True
    else:
        run_job = False

        job_ids_prev = out_dict_i.get("job_ids", None)
        if job_ids_prev is None:
            run_job = True
        else:
            if list(np.sort(job_ids_prev)) != list(np.sort(job_ids_list)):
                run_job = True

    if redo_all_jobs:
        run_job = True


    # #####################################################
    # Testing whether all entries in df_atoms_sorted_ind exist
    job_ids_list = list(set(group_w_o.job_id_max.tolist()))

    all_pairs_ready_list = []
    for job_id_i in job_ids_list:
        job_id_in_sorted = False
        if job_id_i in df_atoms_sorted_ind.job_id.tolist():
            job_id_in_sorted = True
        all_pairs_ready_list.append(job_id_in_sorted)
    all_pairs_ready = all(all_pairs_ready_list)

    if not all_pairs_ready:
        print("Not all job_ids have row in df_atoms_sorted_ind:", name_i)
        run_job = False



    # #####################################################
    if run_job:
        # print("This is good:", name_i)

        groups_to_process.append(name_i)

        # COMMENT THIS OUT TO RUN PARALLEL!!!!!!!

#         out_dict_i = process_group_magmom_comp(
#             name=name_i,
#             group=group_w_o,
#             write_atoms_objects=False,
#             verbose=False,
#             )

#         save_magmom_comp_data(name_i, out_dict_i)

# +
# df_atoms_sorted_ind.set_index

# +
# groups_to_process

# +
# all_pairs_ready_list

# +
# assert False
# -

# ### Running magmom comparison in parallel

# +
def method_wrap(input_dict):
    group_w_o = input_dict["group_w_o"]
    name_i = input_dict["name_i"]

    out_dict_i = process_group_magmom_comp(
        name=name_i,
        group=group_w_o,
        write_atoms_objects=False,
        verbose=False,
        )

    save_magmom_comp_data(name_i, out_dict_i)


input_list = []
for name_i in groups_to_process:
    # #####################################################
    group_i = grouped.get_group(name_i)
    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    df_index = df_jobs_anal_i.index.to_frame()

    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        [True for i in range(len(df_index))]
        ]

    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index    
        ]

    group_w_o = pd.concat([group_i, row_o_i, ], axis=0)


    input_dict_i = dict(
        group_w_o=group_w_o,
        name_i=name_i,
        )
    input_list.append(input_dict_i)

variables_dict = dict()
traces_all = Pool().map(
    partial(
        method_wrap,  # METHOD
        **variables_dict,  # KWARGS
        ),
    input_list,
    )

# + active=""
#
#
#
# -

# # Identifying which slabs have zero magmoms

# +
# tmp_list = []
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():
    # #########################################################
    data_dict_i = dict()
    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################

    process_row = False

    # #####################################################
    job_id_i = row_i.job_id_max
    # #####################################################
    
    # #####################################################
    if name_i in df_atoms_sorted_ind.index:
        process_row = True
        row_atoms_i = df_atoms_sorted_ind.loc[name_i]
        # #################################################
        atoms = row_atoms_i.atoms_sorted_good
        magmoms_i = row_atoms_i.magmoms_sorted_good
        # #################################################

            # (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    # row_atoms_i = df_atoms_sorted_ind.loc[
    #     (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    
    if process_row:
        if atoms.calc != None:
            magmoms_i = atoms.get_magnetic_moments()
        else:
            magmoms_i = magmoms_i

        sum_magmoms_i = np.sum(magmoms_i)
        sum_abs_magmoms = np.sum(np.abs(magmoms_i))

        # #########################################################
        data_dict_i["compenv"] = compenv_i
        data_dict_i["slab_id"] = slab_id_i
        data_dict_i["ads"] = ads_i
        data_dict_i["active_site"] = active_site_i
        data_dict_i["att_num"] = att_num_i
        # #########################################################
        data_dict_i["job_id"] = job_id_i
        data_dict_i["sum_magmoms"] = sum_magmoms_i
        data_dict_i["sum_abs_magmoms"] = sum_abs_magmoms
        # #########################################################
        data_dict_list.append(data_dict_i)
        # #########################################################

df_magmoms = pd.DataFrame(data_dict_list)


# -

# # Normalizing magmoms by number of atoms

# +
def method(row_i):
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_i = row_i.active_site
    att_num_i = row_i.att_num
    sum_magmoms_i = row_i.sum_magmoms
    sum_abs_magmoms_i = row_i.sum_abs_magmoms
    # #####################################################

    # #####################################################
    name_i = (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)
    row_slab_i = df_init_slabs.loc[name_i]
    # #####################################################
    num_atoms_i = row_slab_i.num_atoms
    # #####################################################

    sum_magmoms_pa = sum_magmoms_i / num_atoms_i
    sum_abs_magmoms_pa = sum_abs_magmoms_i / num_atoms_i

    # #####################################################
    row_i["sum_magmoms_pa"] = sum_magmoms_pa
    row_i["sum_abs_magmoms_pa"] = sum_abs_magmoms_pa
    # #####################################################
    return(row_i)
    # #####################################################

df_magmoms = df_magmoms.apply(
    method,
    axis=1)
# -

# # Further analysis of magmom comparison (collapse into dataframe)

# +
data_dict_list = []
for name_i in magmom_data_dict.keys():
    data_dict_i = dict()

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################

    magmom_data_dict[name_i].keys()

    from IPython.display import display
    df_magmoms_comp_i = magmom_data_dict[name_i]["df_magmoms_comp"]
    # display(df_magmoms_comp_i)
    min_val_i = df_magmoms_comp_i.sum_norm_abs_magmom_diff.min()

    # #####################################################
    data_dict_i["name"] = name_i
    data_dict_i["min_sum_norm_abs_magmom_diff"] = min_val_i
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df = pd.DataFrame(data_dict_list)
df.min_sum_norm_abs_magmom_diff.min()

df = df.sort_values("min_sum_norm_abs_magmom_diff")

# +
# assert False
# -

# # Save data to pickle

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/compare_magmoms",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(directory, "df_magmoms.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_magmoms, fle)
# #########################################################

# +
from methods import get_df_magmoms

df_magmoms_tmp = get_df_magmoms()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_failed_to_sort = df_atoms_sorted_ind[df_atoms_sorted_ind.failed_to_sort == True]

# df_i = df_i.drop(labels=df_failed_to_sort.index)

# + jupyter={"source_hidden": true}
# groups_to_process = groups_to_process[0:2]

# groups_to_process
