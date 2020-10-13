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

# # Import Modules

# +
import os
print(os.getcwd())
import sys

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
# -

# # Picking slab to test on

# +
df_jobs_anal_i = df_jobs_anal[df_jobs_anal.job_completely_done == True]

var = "o"
df_jobs_anal_i = df_jobs_anal_i.query('ads == @var')

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
# -


# # \_\_TEMP\_\_

# +
# Test slabs
# job_id_i = "mugorepa_05"
# job_id_i = "pihatufa_64"

# +
# TEMP
# df_jobs_anal_i = df_jobs_anal_i.iloc[0:5]
# df_jobs_anal_i = df_jobs_anal_i.iloc[0:15]
# df_jobs_anal_i = df_jobs_anal_i.iloc[0:55]

# df_jobs_anal_i = df_jobs_anal_i.loc[
#     [('sherlock', 'kenukami_73', 'o', 'NaN', 1)]
#     ]

# +
# print("TEMP")

# compenv_i = 'sherlock'
# slab_id_i = 'putarude_21'
# ads_i = 'o'
# active_site_i = 'NaN'
# att_num_i = 1

# df_jobs_anal_i = df_jobs_anal_i.loc[[
#     (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i)
#     ]]
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

    # #####################################################
    row_sites_i = df_active_sites.loc[slab_id_i]
    # #####################################################
    active_sites_unique_i = row_sites_i.active_sites_unique
    # #####################################################

    # #####################################################
    row_atoms_i = df_atoms_sorted_ind.loc[
        (compenv_i, slab_id_i, ads_i, active_site_i, att_num_i, )]
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
# -

name_i

# + active=""
#
#
#

# +
# oh_slabs_list = get_ads_pos_oh(
#     atoms=atoms,
#     site_i=site_i,
#     df_coord_i=df_coord_i,
#     # #########################
#     include_colinear=True,
#     verbose=False,
#     num_side_ads=3,
#     )

##################################################
##################################################
##################################################
##################################################
##################################################

# +
# # df_coord_i = 
# get_df_coord(
#     mode="post-dft",  # 'bulk', 'slab', 'post-dft'
#     post_dft_name_tuple=name_i,
#     )

# +
# atoms = atoms
# site_i = site_i
# df_coord_i = df_coord_i
# metal_atom_symbol= "Ir"
# # #################################
# include_colinear = True
# verbose = False
# num_side_ads = 4

# +
# # def get_ads_pos_oh(
# #     atoms=None,
# #     site_i=None,
# #     df_coord_i=None,
# #     metal_atom_symbol="Ir",
# #     # #################################
# #     include_colinear=True,
# #     verbose=False,
# #     num_side_ads=4,
# #     ):
# """Return positions of *H atom to be added to Ir-O ligand to create *OH slabs.


# """
# #| - get_ads_pos_oh
# coords_j = get_neighbor_metal_atom(
#     df_coord_i=df_coord_i,
#     site_i=site_i,
#     metal_atom_symbol=metal_atom_symbol,
#     )

# o_position = atoms[site_i].position
# ir_position = coords_j

# if verbose:
#     print("ir_position:", ir_position)
#     print("o_position: ", o_position)


# atoms_oh_list = []

# # #########################################################
# ir_o_vector = o_position - ir_position

# ir_o_unit_vector = ir_o_vector / np.linalg.norm(ir_o_vector)

# # #########################################################
# h_mol = Atoms(
#     [
#         "H",
#         ],
#     positions=[
#         o_position + 0.978 * ir_o_unit_vector,
#         ]
#     )


# atoms_oh_0 = atoms + h_mol

# if include_colinear:
#     atoms_oh_list.append(atoms_oh_0)

# # #########################################################
# atoms_oh_tmp = atoms

# random_vect = [
#     random.choice([-1., +1.]) * random.random(),
#     random.choice([-1., +1.]) * random.random(),
#     random.choice([-1., +1.]) * random.random(),
#     ]

# arb_vector = np.cross(ir_o_unit_vector, random_vect)

# v, axis, theta = (
#     ir_o_unit_vector,
#     arb_vector,
#     (2. / 4.) * np.pi,
#     )
# M0 = M(axis, theta)
# rot_v = np.dot(M0, v)

# # #########################################################
# # for i in range(4):
# for i in range(num_side_ads):
#     v, axis, theta = (
#         rot_v,
#         ir_o_unit_vector,
#         #  2 * (i + 1) * (1. / 4.) * np.pi,
#         2 * (i + 1) * (1. / num_side_ads) * np.pi,
#         )
#     M0 = M(axis, theta)

#     rot_v_new = np.dot(M0, v)

#     h_mol = Atoms(
#         ["H", ],
#         positions=[
#             o_position + 0.978 * rot_v_new]
#         )

#     atoms_oh_i = atoms_oh_tmp + h_mol
#     atoms_oh_list.append(atoms_oh_i)


# # atoms_oh_tmp.write("tmp.cif")
# # return(atoms_oh_list)
# #__|

# + active=""
#
#
#

# +
# assert False
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
print("analyse_jobs.ipynb")
print(20 * "# # ")
# #########################################################

df_slabs_oh

# + active=""
#
#
#
#

# + jupyter={"source_hidden": true}
# # df_slabs_oh.loc[]

# idx = pd.IndexSlice
# df_slabs_oh_i = df_slabs_oh.loc[idx[:, :, :, 84], :]

# # io.write("bad_slabs.traj", df_slabs_oh_i.slab_oh.tolist())
# # io.write("bad_slabs.traj", )

# for i_cnt, i in enumerate(df_slabs_oh_i.slab_oh.tolist()):
#     i.write("out_data/" + str(i_cnt).zfill(2) + ".cif"

# + jupyter={"source_hidden": true}
# for i_cnt, slab_i in enumerate(oh_slabs_list):
#     slab_i.write(
#         os.path.join(
#             "out_data",
#             "oh_" + str(i_cnt).zfill(2) + ".traj"
#             )
#         )
