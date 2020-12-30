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
#     display_name: Python [conda env:PROJ_IrOx_Active_Learning_OER]
#     language: python
#     name: conda-env-PROJ_IrOx_Active_Learning_OER-py
# ---

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}
from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# + jupyter={"source_hidden": true}
redo_all_jobs = False
# redo_all_jobs = True

# + jupyter={"source_hidden": true}
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_init_slabs = get_df_init_slabs()

magmom_data_dict = read_magmom_comp_data()

# + jupyter={"source_hidden": true}
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

# + jupyter={"source_hidden": true}
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

# +
print(20 * "TEMP \n")
# name_i = ('slac', 'relovalu_12', 24.0)
# name_i = ('sherlock', 'vevarehu_32', 63.0)
name_i = ('slac', 'votafefa_68', 38.0)

idx = pd.IndexSlice
df_i = df_i.loc[idx[name_i[0], name_i[1], :, name_i[2], :], :]
# -

df_i

# +
# assert False

# +
# #########################################################
groups_to_process = []
# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
# #########################################################
iterator = tqdm(grouped, desc="1st loop")
for i_cnt, (name_i, group) in enumerate(iterator):
    print(name_i)
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


    if run_job:

        groups_to_process.append(name_i)

        # COMMENT THIS OUT TO RUN PARALLEL!!!!!!!

#         out_dict_i = process_group_magmom_comp(
#             name=name_i,
#             group=group_w_o,
#             write_atoms_objects=False,
#             verbose=False,
#             )

        # # save_magmom_comp_data(name_i, out_dict_i)

# +
#| - Import Modules
import os
import sys

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from methods import (
    get_magmom_diff_data,
    get_df_jobs,
    get_df_atoms_sorted_ind,
    get_df_job_ids,
    CountFrequency,
    )
#__|

# +
name = name_i
group = group_w_o
write_atoms_objects = False
verbose = True

# def process_group_magmom_comp(
#     name=None,
#     group=None,
#     write_atoms_objects=False,
#     verbose=False,
#     ):
"""
"""
#| - process_group_magmom_comp
# #####################################################
group_w_o = group

# #####################################################
out_dict = dict()
out_dict["df_magmoms_comp"] = None
out_dict["good_triplet_comb"] = None
out_dict["job_ids"] = None
# out_dict[""] =

job_ids_list = list(set(group.job_id_max.tolist()))


#| - Reading data
# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

# #########################################################
df_job_ids = get_df_job_ids()
df_job_ids = df_job_ids.set_index("job_id")

from methods import read_magmom_comp_data

assert name != None, "Must pass name to read previous data"

magmom_comp_data_prev = read_magmom_comp_data(name=name)
if magmom_comp_data_prev is not None:
    pair_wise_magmom_comp_data_prev = \
        magmom_comp_data_prev["pair_wise_magmom_comp_data"]
#__|

if write_atoms_objects:
    #| - Write atoms objects
    df_i = pd.concat([
        df_job_ids,
        df_atoms_sorted_ind.loc[
            group_w_o.job_id_max.tolist()
            ]
        ], axis=1, join="inner")

    # #########################################################
    df_index_i = group_w_o.index.to_frame()
    compenv_i = df_index_i.compenv.unique()[0]
    slab_id_i = df_index_i.slab_id.unique()[0]

    active_sites = [i for i in df_index_i.active_site.unique() if i != "NaN"]
    active_site_i = active_sites[0]

    folder_name = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i))
    # #########################################################


    for job_id_i, row_i in df_i.iterrows():
        tmp = 42

        job_id = row_i.name
        atoms = row_i.atoms_sorted_good
        ads = row_i.ads

        file_name = ads + "_" + job_id + ".traj"

        root_file_path = os.path.join("__temp__", folder_name)
        if not os.path.exists(root_file_path):
            os.makedirs(root_file_path)

        file_path = os.path.join(root_file_path, file_name)

        atoms.write(file_path)
    #__|

# #####################################################
#| - Getting good triplet combinations
all_triplet_comb = list(itertools.combinations(
    group_w_o.job_id_max.tolist(), 3))

good_triplet_comb = []
for tri_i in all_triplet_comb:
    df_jobs_i = df_jobs.loc[list(tri_i)]

    # Triplet must not contain duplicate ads
    # Must strictly be a *O, *OH, and *bare triplet
    ads_freq_dict = CountFrequency(df_jobs_i.ads.tolist())

    tmp_list = list(ads_freq_dict.values())
    any_repeat_ads = [True if i > 1 else False for i in tmp_list]

    if not any(any_repeat_ads):
        good_triplet_comb.append(tri_i)
#__|

# #####################################################
#| - MAIN LOOP
if verbose:
    print(
        "Number of viable triplet combinations:",
        len(good_triplet_comb)
        )

data_dict_list = []
pair_wise_magmom_comp_data = dict()

print("TEMP")
good_triplet_comb = [
    ('hetenehu_72', 'dituruvi_75', 'gigowifu_35'),
    ]

for tri_i in good_triplet_comb:
    #| - Process triplets
    data_dict_i = dict()

    if verbose:
        print("tri_i:", tri_i)

    all_pairs = list(itertools.combinations(tri_i, 2))

    df_jobs_i = df_jobs.loc[list(tri_i)]

    sum_norm_abs_magmom_diff = 0.

    print("TEMP")
    all_pairs = [
        # ('hetenehu_72', 'dituruvi_75'),
        ('hetenehu_72', 'gigowifu_35'),
        # ('dituruvi_75', 'gigowifu_35'),
        ]

    for pair_i in all_pairs:

        # # if pair_i in list(pair_wise_magmom_comp_data_prev.keys()):
        # if (magmom_comp_data_prev is not None) and \
        #    (pair_i in list(pair_wise_magmom_comp_data_prev.keys())):
        #     magmom_data_out = pair_wise_magmom_comp_data_prev[pair_i]
        # else:

        # print("Need to run manually")
        # print("pair_i:", pair_i)
        #| - Process pairs
        row_jobs_0 = df_jobs.loc[pair_i[0]]
        row_jobs_1 = df_jobs.loc[pair_i[1]]

        ads_0 = row_jobs_0.ads
        ads_1 = row_jobs_1.ads

        # #############################################
        if set([ads_0, ads_1]) == set(["o", "oh"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
        elif set([ads_0, ads_1]) == set(["o", "bare"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
        elif set([ads_0, ads_1]) == set(["oh", "bare"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
        else:
            print("Woops something went wrong here")


        # #############################################
        row_atoms_i = df_atoms_sorted_ind.loc[job_id_0]
        # #############################################
        atoms_0 = row_atoms_i.atoms_sorted_good
        magmoms_sorted_good_0 = row_atoms_i.magmoms_sorted_good
        was_sorted_0 = row_atoms_i.was_sorted
        # #############################################

        # #############################################
        row_atoms_i = df_atoms_sorted_ind.loc[job_id_1]
        # #############################################
        atoms_1 = row_atoms_i.atoms_sorted_good
        magmoms_sorted_good_1 = row_atoms_i.magmoms_sorted_good
        was_sorted_1 = row_atoms_i.was_sorted
        # #############################################


        #############################################
        magmom_data_out = get_magmom_diff_data(
            ads_atoms=atoms_1,
            slab_atoms=atoms_0,
            ads_magmoms=magmoms_sorted_good_1,
            slab_magmoms=magmoms_sorted_good_0,
            )
        # __|

        pair_wise_magmom_comp_data[pair_i] = magmom_data_out

        tot_abs_magmom_diff = magmom_data_out["tot_abs_magmom_diff"]
        norm_abs_magmom_diff = magmom_data_out["norm_abs_magmom_diff"]
        if verbose:
            print("    ", "pair_i: ", pair_i, ": ", np.round(norm_abs_magmom_diff, 3), sep="")

        print("norm_abs_magmom_diff:", norm_abs_magmom_diff)
        sum_norm_abs_magmom_diff += norm_abs_magmom_diff

    # #################################################
    data_dict_i["job_ids_tri"] = set(tri_i)
    data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff
    # #################################################
    data_dict_list.append(data_dict_i)
    # #################################################

    #__|

#__|

# #####################################################
df_magmoms_i = pd.DataFrame(data_dict_list)

# #####################################################
out_dict["df_magmoms_comp"] = df_magmoms_i
out_dict["good_triplet_comb"] = good_triplet_comb
out_dict["pair_wise_magmom_comp_data"] = pair_wise_magmom_comp_data
out_dict["job_ids"] = job_ids_list
# #####################################################

# return(out_dict)
# __|
# -

assert False

# +
# | - Import Modules
import os

import glob
import filecmp

import numpy as np
import pandas as pd

from ase.atoms import Atoms
from ase.io import read

from pymatgen.core.sites import PeriodicSite
from pymatgen.io.ase import AseAtomsAdaptor
# __|

# from methods_magmom_comp import *
from methods_magmom_comp import _get_magmom_diff_data

# +
ads_atoms = atoms_1
slab_atoms = atoms_0
ads_magmoms = magmoms_sorted_good_1
slab_magmoms = magmoms_sorted_good_0


# def get_magmom_diff_data(
#     ads_atoms=None,
#     slab_atoms=None,
#     ads_magmoms=None,
#     slab_magmoms=None,
#     ):
"""
"""
#| - get_magmom_diff_data

# #########################################################
out_dict__no_flipped = _get_magmom_diff_data(
    ads_atoms, slab_atoms,
    flip_spin_sign=False,
    ads_magmoms=ads_magmoms,
    slab_magmoms=slab_magmoms,
    )
tot_abs_magmom_diff__no_flip = out_dict__no_flipped["tot_abs_magmom_diff"]
# #########################################################
out_dict__yes_flipped = _get_magmom_diff_data(
    ads_atoms, slab_atoms,
    flip_spin_sign=True,
    ads_magmoms=ads_magmoms,
    slab_magmoms=slab_magmoms,
    )
tot_abs_magmom_diff__yes_flip = out_dict__yes_flipped["tot_abs_magmom_diff"]

# #########################################################
if tot_abs_magmom_diff__yes_flip < tot_abs_magmom_diff__no_flip:
    # print("Need to use the flipped spin solution")
    out_dict = out_dict__yes_flipped
else:
    out_dict = out_dict__no_flipped


# return(out_dict)
#__|

# +
# assert False

# +
list(pair_wise_magmom_comp_data.keys())

pair_magmom_comp_i = out_dict
# pair_magmom_comp_i = pair_wise_magmom_comp_data[
#     ('hetenehu_72', 'dituruvi_75')
#     # ('hetenehu_72', 'gigowifu_35')
#     # ('dituruvi_75', 'gigowifu_35')
#     ]

list(pair_magmom_comp_i.keys())

delta_magmoms_i = pair_magmom_comp_i["delta_magmoms"]
tot_abs_magmom_diff_i = pair_magmom_comp_i["tot_abs_magmom_diff"]
norm_abs_magmom_diff_i = pair_magmom_comp_i["norm_abs_magmom_diff"]
delta_magmoms_unsorted_i = pair_magmom_comp_i["norm_abs_magmom_diff"]


# delta_magmoms_unsorted_i
tot_abs_magmom_diff_i
norm_abs_magmom_diff_i 

# +
4.72 / 38

676 - 380

296 / 4

# +

# # pd.DataFrame?

# +
df_tmp_i = pd.DataFrame(
    delta_magmoms_i,
    columns=["index", "diff", ],
    )
df_tmp_i["diff_abs"] = np.abs(df_tmp_i["diff"])

# df_tmp_i.diff_abs.sum()
df_tmp_i.shape
# -

assert False

# + active=""
#
#
#
#
#
#
#

# +
from multiprocessing import Pool
from functools import partial

variables_dict = dict(
    kwarg_0="kwarg_0",
    kwarg_1="kwarg_1",
    kwarg_2="kwarg_2",
    )

def method_wrap(
    input_dict,

    kwarg_0=None,
    kwarg_1=None,
    kwarg_2=None,
    ):
    input_var_0 = input_dict["input_var_0"]
    input_var_1 = input_dict["input_var_1"]
    input_var_2 = input_dict["input_var_2"]

    print(
        "input_var_0:", str(input_var_0),
        "input_var_1:", str(input_var_1),
        "input_var_2:", str(input_var_2),
        )


input_list = []
for i in range(10):
    input_dict_i = dict(
        input_var_0=i + 0,
        input_var_1=i + 1,
        input_var_2=i + 2,
        )
    input_list.append(input_dict_i)

traces_all = Pool().map(
    partial(
        method_wrap,  # METHOD
        **variables_dict,  # KWARGS
        ),
    input_list,
    )

# +
# input_list
# -

assert False

# + jupyter={"source_hidden": true}
import os
print(os.getcwd())
import sys

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

# #########################################################
from IPython.display import display

# #########################################################
from methods import get_df_jobs_anal
from methods import get_df_jobs_data
from methods import get_df_atoms_sorted_ind
from methods import get_df_init_slabs

# #########################################################
from local_methods import read_magmom_comp_data, save_magmom_comp_data
from local_methods import process_group_magmom_comp

# + jupyter={"source_hidden": true}
from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# + jupyter={"source_hidden": true}
# verbose = False
# verbose = True

redo_all_jobs = False
# redo_all_jobs = True

# + jupyter={"source_hidden": true}
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_init_slabs = get_df_init_slabs()

magmom_data_dict = read_magmom_comp_data()

# + jupyter={"source_hidden": true}
# #########################################################
# Only completed jobs will be considered
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

# #########################################################
# Remove the *O slabs for now
# The fact that they have NaN active sites will mess up the groupby
ads_list = df_jobs_anal_i.index.get_level_values("ads").tolist()
ads_list_no_o = [i for i in list(set(ads_list)) if i != "o"]

idx = pd.IndexSlice
df_jobs_anal_no_o = df_jobs_anal_i.loc[idx[:, :, ads_list_no_o, :, :], :]

# + jupyter={"source_hidden": true}
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

print(5 * "When new *O and * jobs come through (from rerunning *OH) make sure to rerun the magmom comparison routine", "\n")

# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
# #########################################################
# iterator = tqdm(grouped, desc="1st loop")
# for i_cnt, (name_i, group) in enumerate(iterator):
if True:
    # TEMP
    print(20 * "TEMP | ")
    name_i = ('slac', 'wiwiwetu_44', 19.0)
    group = grouped.get_group(name_i)

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
    # Deciding whether to reprocess the job or not
    # #####################################################
    out_dict_i = magmom_data_dict.get(name_i, None)
    # #####################################################
    if out_dict_i is None:
        run_job = True
    else:
        run_job = False
        job_ids_i = out_dict_i.get("job_ids", None)
        if job_ids_i is None:
            run_job = True
    if redo_all_jobs:
        run_job = True
    # #####################################################


    if run_job:
        out_dict_i = process_group_magmom_comp(
            group=group_w_o,
            write_atoms_objects=False,
            verbose=False,
            )

        save_magmom_comp_data(name_i, out_dict_i)

# +
# out_dict_i = process_group_magmom_comp(
#     group=group_w_o,
#     write_atoms_objects=False,
#     verbose=True,
#     )

# + jupyter={"source_hidden": true}
#| - Import Modules
import os
import sys

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from methods import (
    get_magmom_diff_data,
    get_df_jobs,
    get_df_atoms_sorted_ind,
    get_df_job_ids,
    CountFrequency,
    )
#__|
# -

import time

# +
group = group_w_o
write_atoms_objects = False
verbose = True

# def process_group_magmom_comp(
#     group=None,
#     write_atoms_objects=False,
#     verbose=False,
#     ):
"""
"""
#| - process_group_magmom_comp
# #####################################################
group_w_o = group

# #####################################################
out_dict = dict()
out_dict["df_magmoms_comp"] = None
out_dict["good_triplet_comb"] = None
out_dict["job_ids"] = None
# out_dict[""] =

job_ids_list = list(set(group.job_id_max.tolist()))


#| - Reading data
# #########################################################
df_jobs = get_df_jobs()

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

# #########################################################
df_job_ids = get_df_job_ids()
df_job_ids = df_job_ids.set_index("job_id")
#__|

if write_atoms_objects:
    #| - Write atoms objects
    df_i = pd.concat([
        df_job_ids,
        df_atoms_sorted_ind.loc[
            group_w_o.job_id_max.tolist()
            ]
        ], axis=1, join="inner")

    # #########################################################
    df_index_i = group_w_o.index.to_frame()
    compenv_i = df_index_i.compenv.unique()[0]
    slab_id_i = df_index_i.slab_id.unique()[0]

    active_sites = [i for i in df_index_i.active_site.unique() if i != "NaN"]
    active_site_i = active_sites[0]

    folder_name = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i))
    # #########################################################


    for job_id_i, row_i in df_i.iterrows():
        tmp = 42

        job_id = row_i.name
        atoms = row_i.atoms_sorted_good
        ads = row_i.ads

        file_name = ads + "_" + job_id + ".traj"

        root_file_path = os.path.join("__temp__", folder_name)
        if not os.path.exists(root_file_path):
            os.makedirs(root_file_path)

        file_path = os.path.join(root_file_path, file_name)

        atoms.write(file_path)
    #__|

# #####################################################
#| - Getting good triplet combinations
all_triplet_comb = list(itertools.combinations(
    group_w_o.job_id_max.tolist(), 3))

good_triplet_comb = []
for tri_i in all_triplet_comb:
    df_jobs_i = df_jobs.loc[list(tri_i)]

    ads_freq_dict = CountFrequency(df_jobs_i.ads.tolist())

    tmp_list = list(ads_freq_dict.values())
    any_repeat_ads = [True if i > 1 else False for i in tmp_list]

    if not any(any_repeat_ads):
        good_triplet_comb.append(tri_i)
#__|

print(
    "Number of viable triplet combinations:",
    len(good_triplet_comb)
    )

good_triplet_comb = [
    ('gubipugu_00', 'himesabi_01', 'setubuha_11'),
    ('gubipugu_00', 'himesabi_01', 'wubitiko_11'),
    ('gubipugu_00', 'ribipuhu_00', 'setubuha_11'),
    ('gubipugu_00', 'ribipuhu_00', 'wubitiko_11'),
    ]

# #####################################################
# #####################################################
#| - MAIN LOOP
data_dict_list = []
pair_wise_magmom_comp_data = dict()
for tri_i in good_triplet_comb:
    data_dict_i = dict()

    print("")
    if verbose:
        print("tri_i:", tri_i)

    all_pairs = list(itertools.combinations(tri_i, 2))

    df_jobs_i = df_jobs.loc[list(tri_i)]

    sum_norm_abs_magmom_diff = 0.
    for pair_i in all_pairs:

        # if verbose:
        #     print("pair_i:", pair_i)

        row_jobs_0 = df_jobs.loc[pair_i[0]]
        row_jobs_1 = df_jobs.loc[pair_i[1]]

        ads_0 = row_jobs_0.ads
        ads_1 = row_jobs_1.ads

        # #############################################
        if set([ads_0, ads_1]) == set(["o", "oh"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
        elif set([ads_0, ads_1]) == set(["o", "bare"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
        elif set([ads_0, ads_1]) == set(["oh", "bare"]):
            job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
            job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
        else:
            print("Woops something went wrong here")


        # #############################################
        row_atoms_i = df_atoms_sorted_ind.loc[job_id_0]
        # #############################################
        atoms_0 = row_atoms_i.atoms_sorted_good
        magmoms_sorted_good_0 = row_atoms_i.magmoms_sorted_good
        was_sorted_0 = row_atoms_i.was_sorted
        # #############################################


        # #############################################
        row_atoms_i = df_atoms_sorted_ind.loc[job_id_1]
        # #############################################
        atoms_1 = row_atoms_i.atoms_sorted_good
        magmoms_sorted_good_1 = row_atoms_i.magmoms_sorted_good
        was_sorted_1 = row_atoms_i.was_sorted
        # #############################################


        print("1111")
        t0 = time.time()

        # #############################################
        magmom_data_out = get_magmom_diff_data(
            ads_atoms=atoms_1,
            slab_atoms=atoms_0,
            ads_magmoms=magmoms_sorted_good_1,
            slab_magmoms=magmoms_sorted_good_0,
            )

        print(
            "2222",
            " | ",
            np.abs(t0 - time.time())
            )

        pair_wise_magmom_comp_data[pair_i] = magmom_data_out

        tot_abs_magmom_diff = magmom_data_out["tot_abs_magmom_diff"]
        # print("    ", pair_i, ": ", np.round(tot_abs_magmom_diff, 2), sep="")
        norm_abs_magmom_diff = magmom_data_out["norm_abs_magmom_diff"]
        if verbose:
            print("    ", "pair_i: ", pair_i, ": ", np.round(norm_abs_magmom_diff, 3), sep="")

        sum_norm_abs_magmom_diff += norm_abs_magmom_diff


    # #################################################
    data_dict_i["job_ids_tri"] = set(tri_i)
    data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff
    # #################################################
    data_dict_list.append(data_dict_i)
    # #################################################

    # print("")
#__|


# #####################################################
df_magmoms_i = pd.DataFrame(data_dict_list)

# #####################################################
out_dict["df_magmoms_comp"] = df_magmoms_i
out_dict["good_triplet_comb"] = good_triplet_comb
out_dict["pair_wise_magmom_comp_data"] = pair_wise_magmom_comp_data
out_dict["job_ids"] = job_ids_list
# #####################################################

# return(out_dict)
#__|

# +
# good_triplet_comb[0:4]

# good_triplet_comb = [
#     ('gubipugu_00', 'himesabi_01', 'setubuha_11'),
#     ('gubipugu_00', 'himesabi_01', 'wubitiko_11'),
#     ('gubipugu_00', 'ribipuhu_00', 'setubuha_11'),
#     ('gubipugu_00', 'ribipuhu_00', 'wubitiko_11'),
#     ]

# +
# print(
#     "Number of viable triplet combinations:",
#     len(good_triplet_comb)
#     )

# +
# group_w_o
# -

assert False

# + active=""
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

# +
import itertools

import numpy as np
import pandas as pd

from methods import (
    get_magmom_diff_data,
    # _get_magmom_diff_data,
    )

from methods import get_df_jobs
from methods import CountFrequency
from methods import get_df_atoms_sorted_ind
from methods import get_df_job_ids

# +
# # #########################################################
# df_jobs = get_df_jobs()

# # #########################################################
# df_atoms_sorted_ind = get_df_atoms_sorted_ind()
# df_atoms_sorted_ind = df_atoms_sorted_ind.set_index("job_id")

# # #########################################################
# df_job_ids = get_df_job_ids()
# df_job_ids = df_job_ids.set_index("job_id")

# +
# #########################################################
import pickle; import os
directory = os.path.join(
    os.environ["HOME"],
    "__temp__")
path_i = os.path.join(directory, "temp_data.pickle")
with open(path_i, "rb") as fle:
    data = pickle.load(fle)
# #########################################################

group_w_o = data

# +
write_atoms_objets = True

from local_methods import process_group_magmom_comp

out_dict = process_group_magmom_comp(
    group=group_w_o,
    # df_jobs=None,
    write_atoms_objects=False,
    verbose=False,
    )
# out_dict

# +
out_dict.keys()

# list(out_dict["pair_wise_magmom_comp_data"].keys())




# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# job_id_0 = 

# df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id

# df_jobs_i[df_jobs_i.ads == "bare"]

# df_jobs_i

# + jupyter={"source_hidden": true}
# magmom_data_out["tot_abs_magmom_diff"]

# magmom_data_out.keys()

# + jupyter={"source_hidden": true}
# if write_atoms_objets:

#     df_i = pd.concat([
#         df_job_ids,
#         df_atoms_sorted_ind.loc[
#             group_w_o.job_id_max.tolist()
#             ]
#         ], axis=1, join="inner")

#     # #########################################################
#     df_index_i = group_w_o.index.to_frame()
#     compenv_i = df_index_i.compenv.unique()[0]
#     slab_id_i = df_index_i.slab_id.unique()[0]

#     active_sites = [i for i in df_index_i.active_site.unique() if i != "NaN"]
#     active_site_i = active_sites[0]

#     folder_name = compenv_i + "__" + slab_id_i + "__" + str(int(active_site_i))
#     # #########################################################


#     for job_id_i, row_i in df_i.iterrows():
#         tmp = 42

#         job_id = row_i.name
#         atoms = row_i.atoms_sorted_good
#         ads = row_i.ads

#         file_name = ads + "_" + job_id + ".traj"

#         root_file_path = os.path.join("__temp__", folder_name)
#         if not os.path.exists(root_file_path):
#             os.makedirs(root_file_path)

#         file_path = os.path.join(root_file_path, file_name)

#         atoms.write(file_path)

# + jupyter={"source_hidden": true}
# all_triplet_comb = list(itertools.combinations(
#     group_w_o.job_id_max.tolist(), 3))

# good_triplet_comb = []
# for tri_i in all_triplet_comb:
#     df_jobs_i = df_jobs.loc[list(tri_i)]

#     ads_freq_dict = CountFrequency(df_jobs_i.ads.tolist())

#     tmp_list = list(ads_freq_dict.values())
#     any_repeat_ads = [True if i > 1 else False for i in tmp_list]

#     if not any(any_repeat_ads):
#         good_triplet_comb.append(tri_i)

# # good_triplet_comb

# + jupyter={"source_hidden": true}
# data_dict_list = []
# for tri_i in good_triplet_comb:
#     data_dict_i = dict()

#     # print("tri_i:", tri_i)
#     all_pairs = list(itertools.combinations(tri_i, 2))

#     df_jobs_i = df_jobs.loc[list(tri_i)]
    
#     sum_norm_abs_magmom_diff = 0.
#     for pair_i in all_pairs:

#         row_jobs_0 = df_jobs.loc[pair_i[0]]
#         row_jobs_1 = df_jobs.loc[pair_i[1]]

#         ads_0 = row_jobs_0.ads
#         ads_1 = row_jobs_1.ads

#         # #########################################################
#         if set([ads_0, ads_1]) == set(["o", "oh"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
#         elif set([ads_0, ads_1]) == set(["o", "bare"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "o"].iloc[0].job_id
#         elif set([ads_0, ads_1]) == set(["oh", "bare"]):
#             job_id_0 = df_jobs_i[df_jobs_i.ads == "bare"].iloc[0].job_id
#             job_id_1 = df_jobs_i[df_jobs_i.ads == "oh"].iloc[0].job_id
#         else:
#             print("Woops something went wrong here")


#         # #########################################################
#         row_atoms_i = df_atoms_sorted_ind.loc[job_id_0]
#         # #########################################################
#         atoms_0 = row_atoms_i.atoms_sorted_good
#         magmoms_sorted_good_0 = row_atoms_i.magmoms_sorted_good
#         was_sorted_0 = row_atoms_i.was_sorted
#         # #########################################################

#         # #########################################################
#         row_atoms_i = df_atoms_sorted_ind.loc[job_id_1]
#         # #########################################################
#         atoms_1 = row_atoms_i.atoms_sorted_good
#         magmoms_sorted_good_1 = row_atoms_i.magmoms_sorted_good
#         was_sorted_1 = row_atoms_i.was_sorted
#         # #########################################################


#         # #########################################################
#         magmom_data_out = get_magmom_diff_data(
#             ads_atoms=atoms_1,
#             slab_atoms=atoms_0,
#             ads_magmoms=magmoms_sorted_good_1,
#             slab_magmoms=magmoms_sorted_good_0,
#             )

#         # list(magmom_data_out.keys())

#         tot_abs_magmom_diff = magmom_data_out["tot_abs_magmom_diff"]
#         # print("    ", pair_i, ": ", np.round(tot_abs_magmom_diff, 2), sep="")
#         norm_abs_magmom_diff = magmom_data_out["norm_abs_magmom_diff"]
#         print("    ", pair_i, ": ", np.round(norm_abs_magmom_diff, 3), sep="")
        
#         sum_norm_abs_magmom_diff += norm_abs_magmom_diff

#     # #####################################################
#     data_dict_i["job_ids_tri"] = set(tri_i)
#     data_dict_i["sum_norm_abs_magmom_diff"] = sum_norm_abs_magmom_diff
#     # #####################################################
#     data_dict_list.append(data_dict_i)
#     # #####################################################

#     # print("TEMP")
#     # break

#     print("")

#         # #########################################################

# df_magmoms_i = pd.DataFrame(data_dict_list)
# # df_magmoms_i
