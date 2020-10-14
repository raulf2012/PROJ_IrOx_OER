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

# # Collect DFT data into *, *O, *OH collections
# ---

# # Import Modules

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
# -

# # Script Inputs

# +
verbose = False
# verbose = True

redo_all_jobs = False
# redo_all_jobs = True
# -

# # Read Data

# +
df_jobs_anal = get_df_jobs_anal()

df_jobs_data = get_df_jobs_data()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_init_slabs = get_df_init_slabs()

# + active=""
#
#

# +
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

# # TEMP | Filtering dataframe for testing

# +
df_index_i = df_i.index.to_frame()

df_index_tmp = df_index_i[
    
    # (df_index_i.compenv == "sherlock") & \
    # (df_index_i.slab_id == "vuvunira_55") & \
    # (df_index_i.active_site == 68.) & \

    (df_index_i.compenv == "sherlock") & \
    (df_index_i.slab_id == "kipatalo_90") & \
    (df_index_i.active_site == 81.) & \

    [True for i in range(len(df_index_i))]
    ]


# print("TEMP")
# df_i = df_i.loc[
#     df_index_tmp.index
#     ]

# + active=""
#
#
# -

# # Magmom comparison

# +
# #########################################################
verbose_local = False
# #########################################################

# data_dict = dict()
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
for i_cnt, (name_i, group) in enumerate(grouped):
    
    if verbose_local:
        print(40 * "*")
        print("name_i:", name_i)

    # #########################################################
    magmom_data_dict = read_magmom_comp_data()
    # #########################################################
    # print("len(magmom_data_dict):", len(magmom_data_dict))

    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #########################################################

    df_index = df_jobs_anal_i.index.to_frame()

    df_index_i = df_index[
        (df_index.compenv == compenv_i) & \
        (df_index.slab_id == slab_id_i) & \
        (df_index.ads == "o") & \
        [True for i in range(len(df_index))]
        ]
    df_index_i.shape

    mess_i = "ISJIdfjisdjij"
    assert df_index_i.shape[0] == 1, mess_i


    row_o_i = df_jobs_anal_i.loc[
        df_index_i.index    
        ]

    # #########################################################
    group_w_o = pd.concat(
        [ 
            group,
            row_o_i,
            ],
        axis=0)

    write_atoms_objets = True

    
    out_dict = magmom_data_dict.get(name_i, None)

    if out_dict is None:
        run_job = True
    else:
        run_job = False
    
    if redo_all_jobs:
        run_job = True



    if verbose_local:
        print(run_job)

    if run_job:
        out_dict = process_group_magmom_comp(
            group=group_w_o,
            write_atoms_objects=False,
            # write_atoms_objects=True,
            verbose=False,
            # verbose=True,
            )


    magmom_data_dict[name_i] = out_dict

    save_magmom_comp_data(magmom_data_dict)
    if verbose_local:
        print("")

# +
# magmom_data_dict
# list(magmom_data_dict.keys())

# +
# assert False

# + active=""
#
#
#
# -

# # Identifying which slabs have zero magmoms

# +
tmp_list = []
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():
    data_dict_i = dict()

    # #########################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #########################################################

    # #########################################################
    job_id_i = row_i.job_id_max
    # #########################################################
    
    # #########################################################
    row_atoms_i = df_atoms_sorted_ind.loc[
        (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)]
    # #########################################################
    atoms = row_atoms_i.atoms_sorted_good
    magmoms_i = row_atoms_i.magmoms_sorted_good
    # #########################################################
    
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


# +
# df_jobs_anal_i[df_jobs_anal_i.job_id_max == "donepote_14"]

# df_magmoms.loc[
#     "donepote_14"
#     ]

# df_magmoms[df_magmoms.job_id == "donepote_14"]

# +
# df_magmoms

# +
# assert False
# -

# # Normalizing magmoms by number of atoms

# +
def method(row_i):
    """
    """

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
# -

# # Save data to pickle

# Pickling data ###########################################
import os; import pickle
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    # "workflow/compare_magmoms",
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
print("analyse_jobs.ipynb")
print(20 * "# # ")
# assert False
# #########################################################

# +
# assert False

# + active=""
#
#
#

# +
# data_dict.keys()

# magmom_dict_i = data_dict[
#     ('sherlock', 'kipatalo_90', 81.0)
#     ]

# list(magmom_dict_i.keys())

# df_magmoms_comp = magmom_dict_i["df_magmoms_comp"]
# good_triplet_comb = magmom_dict_i["good_triplet_comb"]
# pair_wise_magmom_comp_data = magmom_dict_i["pair_wise_magmom_comp_data"]

# # ['df_magmoms_comp', 'good_triplet_comb', 'pair_wise_magmom_comp_data']

# +
# assert False
# -

# ### Saving misc data objects for more testing elsewhere

# +
# save_object = group_w_o

# # Pickling data ###########################################
# import os; import pickle
# directory = os.path.join(
#     os.environ["HOME"],
#     "__temp__")
# if not os.path.exists(directory): os.makedirs(directory)
# path_i = os.path.join(directory, "temp_data.pickle")
# with open(path_i, "wb") as fle:
#     pickle.dump(save_object, fle)
# # #########################################################

# # #########################################################
# import pickle; import os
# directory = os.path.join(
#     os.environ["HOME"],
#     "__temp__")
# path_i = os.path.join(directory, "temp_data.pickle")
# with open(path_i, "rb") as fle:
#     data = pickle.load(fle)
# # #########################################################
