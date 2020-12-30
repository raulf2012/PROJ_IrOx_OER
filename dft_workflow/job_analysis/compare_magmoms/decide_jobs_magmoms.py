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

# # Collect DFT data into *, *O, *OH collections
# ---
#
# Notes:
#   * If there exists only a single slab for a particular adsorbate, and that slab has a averaged absolute magmom per atom of less than XXX, then we should check if there are slabs of different adsorbates in that set to tranplant the magmoms from

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# #########################################################
from IPython.display import display

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_jobs_data,
    get_df_atoms_sorted_ind,
    get_df_magmoms,
    get_df_jobs_paths,
    get_df_jobs_oh_anal,
    )

# #########################################################
from local_methods import (
    read_magmom_comp_data,
    save_magmom_comp_data,
    process_group_magmom_comp,
    get_oer_set,
    analyze_O_in_set,
    )
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
# #########################################################
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

# #########################################################
df_jobs_data = get_df_jobs_data()

# #########################################################
df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# #########################################################
magmom_data_dict = read_magmom_comp_data()

# #########################################################
df_magmoms = get_df_magmoms()

# #########################################################
df_jobs_paths = get_df_jobs_paths()

# #########################################################
df_magmoms = df_magmoms.set_index("job_id")

# #########################################################
df_jobs_oh_anal = get_df_jobs_oh_anal()

# + active=""
#
#
# -

# # Preprocessing data objects

# ## Processing `df_jobs_anal` (only completed job sets, filter out *O)

# +
from methods import get_df_slab
df_slab = get_df_slab()

slab_ids_phase_2 = df_slab[df_slab.phase > 0].slab_id.tolist()

# df_m2.loc[
#     df_m2.slab_id.isin(slab_ids_phase_2)
#     ]

df_index_i = df_jobs_anal_i.index.to_frame()
df_jobs_anal_i = df_jobs_anal_i.loc[
    df_index_i.slab_id.isin(slab_ids_phase_2).index
    ]
# -

df_jobs_anal_i

# +
# #########################################################
# Only completed jobs will be considered
df_jobs_anal_i = df_jobs_anal_i[df_jobs_anal_i.job_completely_done == True]

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

# #########################################################
# Only keep OER job sets that have all adsorbates present and completed
indices_to_keep = []
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_jobs_anal_no_o.groupby(groupby_cols)
for name_i, group in grouped:

    # print("TEMP")
    # index_i = ('slac', 'fagumoha_68', 'oh', 62.0, 3)
    # if index_i in group.index:
    #     print(name_i)

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

# ## Process `df_jobs_oh_anal`

df_jobs_oh_anal = df_jobs_oh_anal.set_index(
    ["compenv", "slab_id", "active_site", ], drop=False)

# + active=""
#
#
# -

# # Checking if there are OER sets that have slabs with magmom 0'ed out

# Cutoff for how low the magmoms of slab can go before I rerun with different spin
magmom_cutoff = 0.1

# +
# #########################################################
verbose_local = False
# #########################################################

# #########################################################
data_dict_list = []
# #########################################################
groupby_cols = ["compenv", "slab_id", "active_site", ]
grouped = df_i.groupby(groupby_cols)
for i_cnt, (name_i, group) in enumerate(grouped):
    data_dict_i = dict()

    if verbose_local:
        print(40 * "*")
        print("name_i:", name_i)

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]
    # #####################################################


    # #####################################################
    group_i = get_oer_set(
        group=group,
        compenv=compenv_i,
        slab_id=slab_id_i,
        df_jobs_anal=df_jobs_anal,
        )

    # #####################################################
    magmom_data_out = analyze_O_in_set(
        data_dict_i,
        group_i,
        df_magmoms,
        magmom_cutoff=magmom_cutoff,
        compenv=compenv_i,
        slab_id=slab_id_i,
        active_site=active_site_i,
        )

    # #####################################################
    data_dict_i.update(magmom_data_out)
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

# #########################################################
df_m = pd.DataFrame(data_dict_list)
df_m = df_m.set_index(["compenv", "slab_id", "active_site", ], drop=False)

# +
# # TEMP

# # sherlock	bivahobe_50	32.0

# df = df_jobs_oh_anal
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bivahobe_50") &
#     (df["active_site"] == 32.) &
#     [True for i in range(len(df))]
#     ]
# df

# +
# # TEMP

# # sherlock	bivahobe_50	32.0

# df = df_m
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bivahobe_50") &
#     (df["active_site"] == 32.) &
#     [True for i in range(len(df))]
#     ]
# df

# +
index_diff_0 = df_jobs_oh_anal.index.difference(df_m.index)
index_diff_1 = df_m.index.difference(df_jobs_oh_anal.index)

mess_i = "This shouldn't be, look into it"
# assert index_diff_1.shape[0] == 0, mess_i

# #########################################################
shared_index = df_jobs_oh_anal.index.intersection(df_m.index)

df_jobs_oh_anal = df_jobs_oh_anal.loc[shared_index]
df_m = df_m.loc[shared_index]

# +
# # df_jobs_oh_anal.loc[
# #     ('sherlock', 'bivahobe_50', 32.0)
# #     ]

# df_ind = df_jobs_oh_anal.index.to_frame()

# df = df_ind
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bivahobe_50") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]

# df

# +
# index_diff_1

# # ('sherlock', 'bivahobe_50', 32.0)

# +
# print(440 * "TEMP")
# assert False

# +
list_0 = list(df_m.columns)
list_1 = list(df_jobs_oh_anal.columns)

shared_cols = list(set(list_0).intersection(list_1))

df_list = [
    df_m.drop(columns=shared_cols),
    df_jobs_oh_anal,
    ]

df_m2 = pd.concat(df_list, axis=1)
df_m2 = df_m2.sort_index()

# +
# df_ind = df_m2.index.to_frame()


# df = df_ind
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bivahobe_50") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]

# df

# +
# assert False

# +
df_m3 = df_m2[
    # (df_m2["*O_w_low_magmoms"] == True) & \
    # (df_m2["*O_w_not_low_magmoms"] == False) & \
    (df_m2["all_oh_attempts_done"] == True) & \
    (df_m2["all_jobs_bad"] == False) & \
    [True for i in range(len(df_m2))]
    ]
# df_m3 = df_m3[
#     df_m3.all_jobs_bad == False
#              ]

data_dict_list = []
for i_cnt, row_i in df_m3.iterrows():
    # print(i_cnt)
    data_dict_i = dict()

    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    active_site_i = row_i.active_site
    all_oh_attempts_done_i = row_i.all_oh_attempts_done
    job_ids_sorted_energy_i = row_i.job_ids_sorted_energy
    job_id_most_stable_i = row_i.job_id_most_stable
    # #####################################################

    # #####################################################
    row_magmoms_i = df_magmoms.loc[job_id_most_stable_i]
    # #####################################################
    sum_abs_magmoms_pa_i = row_magmoms_i.sum_abs_magmoms_pa
    # #####################################################

    # print("sum_abs_magmoms_pa_i:", sum_abs_magmoms_pa_i)

    rerun_from_oh = False
    # if sum_abs_magmoms_pa_i > magmom_cutoff:
    if sum_abs_magmoms_pa_i > 0.07:
        rerun_from_oh = True

    # #####################################################
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["rerun_from_oh"] = rerun_from_oh
    # #####################################################
    data_dict_i.update(row_i.to_dict())
    # #####################################################
    data_dict_list.append(data_dict_i)
    # #####################################################

df_rerun_from_oh = pd.DataFrame(data_dict_list)
# -

df_rerun_from_oh

# +
# # TEMP

# # sherlock	bivahobe_50	32.0

# df = df_rerun_from_oh
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "bivahobe_50") &
#     (df["active_site"] == 32.) &
#     [True for i in range(len(df))]
#     ]
# df

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
path_i = os.path.join(directory, "df_rerun_from_oh.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_rerun_from_oh, fle)
# #########################################################

from methods import get_df_rerun_from_oh
df_rerun_from_oh_tmp = get_df_rerun_from_oh()
df_rerun_from_oh_tmp.head()

# # Writing the slabs with the smallest magmoms to file to manually inspect

# +
df_i = df_magmoms[df_magmoms.sum_abs_magmoms_pa > 1e-5]
df_i = df_i.sort_values("sum_abs_magmoms_pa", ascending=True)

for i_cnt, (job_id_i, row_i) in enumerate(df_i.iloc[0:20].iterrows()):

    # #####################################################
    row_paths_i = df_jobs_paths.loc[job_id_i]
    # #####################################################
    gdrive_path_i = row_paths_i.gdrive_path
    # #####################################################

    path_i = os.path.join(
        os.environ["PROJ_irox_oer_gdrive"],
        gdrive_path_i,
        "final_with_calculator.traj")

    directory = os.path.join(
        os.environ["PROJ_irox_oer"],
        "dft_workflow/job_analysis/compare_magmoms",
        "__temp__/low_magmom_slabs")
    if not os.path.exists(directory):
        os.makedirs(directory)

    out_path = os.path.join(
        directory,
        str(i_cnt).zfill(3) + "_" + job_id_i + ".traj")

    shutil.copyfile(
        path_i,
        out_path)

# df_i.iloc[0:20]
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
# # TEMP
# df_jobs_oh_anal.iloc[0:2]

# + jupyter={"source_hidden": true}
# print("df_m2.shape:", df_m2.shape)

# # df_m3 =
# df_m2[
#     (df_m2["*O_w_low_magmoms"] == True) & \
#     # (df_m2["*O_w_not_low_magmoms"] == False) & \
#     (df_m2["all_oh_attempts_done"] == True) & \
#     [True for i in range(len(df_m2))]
#     ].shape

# + jupyter={"source_hidden": true}
# # slac     relovalu_12  24.0

# # df_m2.loc[("slac", "relovalu_12", 24.)]

# df_m2.loc[
#     ('sherlock', 'vuvunira_55', 73.0)
#     ]

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# row_i.all_jobs_bad

# + jupyter={"source_hidden": true}
# row_magmoms_i =
# df_magmoms.loc[job_id_most_stable_i]

# job_id_most_stable_i
# + jupyter={"source_hidden": true}
# df_rerun_from_oh.shape
# 30 rows

# + jupyter={"source_hidden": true}
# df_m

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# TEMP

# ('sherlock', 'kenukami_73', 84.0)

# idx = pd.IndexSlice
# df_i = df_i.loc[idx["sherlock", "kenukami_73", :, 84.], :]

# print(df_i.shape[0])

# + jupyter={"source_hidden": true}
# df_jobs_oh_anal

# + jupyter={"source_hidden": true}
# assert False
