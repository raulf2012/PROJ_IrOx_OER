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

# # Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

from IPython.display import display

import random
import numpy as np
from numpy import dot
import pandas as pd
# pd.set_option("display.max_columns", None)
# pd.options.display.max_colwidth = 20

from ase import Atoms
from ase import io

# #########################################################
from proj_data import metal_atom_symbol

# #########################################################
from methods import get_df_coord
from methods import (
    get_df_dft,
    get_df_job_ids,
    get_df_jobs,
    get_df_jobs_data,
    get_df_slab,
    get_df_slab_ids,
    get_df_jobs_data_clusters,
    get_df_jobs_anal,
    get_df_active_sites,
    get_df_atoms_sorted_ind,
    )

# #########################################################
from local_methods import get_neighbor_metal_atom
from local_methods import get_ads_pos_oh
from local_methods import M
# -

# # Script Inputs

verbose = True
verbose = False

# # Read data objects with methods

# +
df_jobs = get_df_jobs(exclude_wsl_paths=True)

df_jobs_data = get_df_jobs_data(exclude_wsl_paths=True)

df_jobs_anal = get_df_jobs_anal()

df_active_sites = get_df_active_sites()

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

var = "o"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

var = "NaN"
df_jobs_anal_i = df_jobs_anal_i.query('active_site == @var')
# -


# # Picking slab to test on

# +
job_ids__completely_done__ads_o = df_jobs_anal_i.job_id_max
df_jobs_data_i = df_jobs_data.loc[job_ids__completely_done__ads_o]

# #########################################################
# directory = "out_data/finished_O_ads"
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/create_oh_slabs",
    "out_data/finished_O_ads")
if not os.path.exists(directory):
    os.makedirs(directory)

for job_id_i, row_data_i in df_jobs_data_i.iterrows():

    # #####################################################
    final_atoms_i = row_data_i.final_atoms
    # #####################################################

    file_name_i = job_id_i + ".cif"

    final_atoms_i.write(os.path.join(directory, file_name_i))

    # final_atoms_i.write(os.path.join(
    #     "out_data",
    #     "finished_O_ads",
    #     file_name_i))


# +
# df_ind_i = df_jobs_anal_i.index.to_frame()

# # ('slac', 'fodopilu_17', 'o', 24, 1)
# df = df_ind_i
# df = df[
#     (df["compenv"] == "slac") &
#     (df["slab_id"] == "fodopilu_17") &
#     (df["ads"] == "o") &
#     # # (df["active_site"] == 24.) &
#     [True for i in range(len(df))]
#     ]

# print(20 * "TEMP | ")
# df_jobs_anal_i = df_jobs_anal_i.loc[
#     df.index
#     ]

# +
# df_jobs_anal_i

# +
# assert False
# -

# # Main Loop



# +
data_dict_list = []
for name_i, row_i in df_jobs_anal_i.iterrows():

    if verbose:
        print(40 * "=")

    # #####################################################
    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    ads_i = name_i[2]
    active_site_i = name_i[3]
    att_num_i = name_i[4]
    # #####################################################
    job_id_max_i = row_i.job_id_max
    # #####################################################

    if slab_id_i not in df_active_sites.index:
        print(244 * "slab_id not found in df_active_sites, need to run `get_all_active_sites.ipynb`")

    # #####################################################
    row_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_sites_i.active_sites_unique
    # #####################################################

    # #####################################################
    if name_i in df_atoms_sorted_ind.index:
        row_atoms_i = df_atoms_sorted_ind.loc[name_i]
            # (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i, )]
        # #####################################################
        atoms_sorted_good_i = row_atoms_i.atoms_sorted_good
        atoms = atoms_sorted_good_i
        # #####################################################




        # #####################################################
        name_i = (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i, )


        # #####################################################
        # #####################################################
        for site_i in active_sites_unique_i:
            # print("site_i:", site_i)

            # #################################################
            # #################################################
            df_coord_i = get_df_coord(
                mode="post-dft",  # 'bulk', 'slab', 'post-dft'
                post_dft_name_tuple=name_i,
                )

            oh_slabs_list = get_ads_pos_oh(
                atoms=atoms,
                site_i=site_i,
                df_coord_i=df_coord_i,
                # #########################
                include_colinear=True,
                verbose=False,
                num_side_ads=3,
                )

            for att_num_oh_j, slab_oh_j in enumerate(oh_slabs_list):
                # #############################################
                data_dict_i = dict()
                # #############################################
                data_dict_i["compenv"] = compenv_i
                data_dict_i["slab_id"] = slab_id_i
                data_dict_i["ads"] = ads_i
                data_dict_i["active_site"] = site_i
                data_dict_i["att_num"] = att_num_i
                data_dict_i["att_num_oh"] = att_num_oh_j
                data_dict_i["slab_oh"] = slab_oh_j
                # #############################################
                data_dict_list.append(data_dict_i)
                # #############################################

df_slabs_oh = pd.DataFrame(data_dict_list)
df_slabs_oh = df_slabs_oh.set_index([
    "compenv", "slab_id", "ads",
    "active_site", "att_num", "att_num_oh", ])

# +
# df_ind_i = df_slabs_oh.index.to_frame()


# # ('slac', 'fodopilu_17', 'o', 24, 1)

# df = df_ind_i
# df = df[
#     (df["compenv"] == "slac") &
#     (df["slab_id"] == "fodopilu_17") &
#     (df["ads"] == "o") &
#     (df["active_site"] == 24.) &
#     [True for i in range(len(df))]
#     ]
# df

# +
# assert False

# + active=""
#
#
#
# -

# # Writing *OH Slabs to File

# +
# directory = "out_data/oh_slabs"

directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/create_oh_slabs",
    "out_data/oh_slabs")

if not os.path.exists(directory):
    os.makedirs(directory)

for name_i, row_i in df_slabs_oh.iterrows():

    # #####################################################
    slab_oh_i = row_i.slab_oh
    # #####################################################

    file_name_i = '__'.join(str(e) for e in list(name_i))
    file_name_i += ".cif"
    slab_oh_i.write(os.path.join(directory, file_name_i))

        # "out_data/oh_slabs",
# -

# # Save to pickle

# Pickling data ###########################################
import os; import pickle
# directory = "out_data"
directory = os.path.join(
    os.environ["PROJ_irox_oer"],
    "dft_workflow/job_analysis/create_oh_slabs",
    "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_slabs_oh.pickle"), "wb") as fle:
    pickle.dump(df_slabs_oh, fle)
# #########################################################

# +
from methods import get_df_slabs_oh

df_slabs_oh_tmp = get_df_slabs_oh()
# df_slabs_oh_tmp
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("create_oh_slabs.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
#
