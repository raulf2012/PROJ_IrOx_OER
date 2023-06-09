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

# + [markdown] tags=[]
# # Obtaining the indices of the atoms that make up the active octahedra
# ---

# + [markdown] tags=[]
# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import random
import pickle

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# # #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_octa_info,
    get_df_features_targets,
    get_df_jobs_data,
    get_other_job_ids_in_set,
    get_df_struct_drift,
    get_df_jobs,
    get_df_init_slabs,
    )
from methods_features import get_octahedra_atoms, get_more_octahedra_data


from methods import get_metal_active_site
from proj_data import metal_atom_symbol
from methods import get_octahedral_oxygens_from_init
from methods_features import get_octahedral_oxygens_A
from methods import get_df_coord
from methods import get_df_coord_wrap
from ipywidgets import IntProgress
from methods import get_oxy_images
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

root_dir = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/octahedra_info",
    )

# ### Read Data

# +
df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()

df_active_sites = get_df_active_sites()

df_octa_info_prev = get_df_octa_info()

df_jobs = get_df_jobs()

df_init_slabs = get_df_init_slabs()

df_struct_drift = get_df_struct_drift()

df_features_targets = get_df_features_targets()

df_jobs_data = get_df_jobs_data()

# +
# df_octa_info_prev = 
# df_octa_info_prev = df_octa_info_prev.loc[
#     df_octa_info_prev.index.drop_duplicates()
#     ]

df_octa_info_prev = df_octa_info_prev[~df_octa_info_prev.index.duplicated(keep='first')]
# -

# ### Filtering down to `oer_adsorbate` jobs

# +
df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal = df_jobs_anal.droplevel(level=0)


df_ind = df_atoms_sorted_ind.index.to_frame()
df_atoms_sorted_ind = df_atoms_sorted_ind.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_atoms_sorted_ind = df_atoms_sorted_ind.droplevel(level=0)

# + active=""
#
#
#
# -

# ### Processing init slab systems

# +
# #########################################################
data_dict_list = []
indices_to_process = []
indices_to_not_process = []
# #########################################################

iterator = tqdm(df_features_targets.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    row_i = df_features_targets.loc[index_i]

    compenv_i = row_i[("data", "compenv", "")]
    slab_id_i = row_i[("data", "slab_id", "")]
    active_site_i = row_i[("data", "active_site", "")]

    att_num_i = 1
    from_oh_i = False
    ads_i = "o"


    index_octa_i = ("init", compenv_i, slab_id_i, ads_i,
        active_site_i, att_num_i, from_oh_i, )
    if index_octa_i in df_octa_info_prev.index:
        indices_to_not_process.append(index_i)
    else:
        indices_to_process.append(index_i)
# -

# ### Also do init *OH slabs too

# +
# # # TEMP
# # print(111 * "TEMP | ")

# # Do subset of what needs done
# indices_to_process = random.sample(indices_to_process, 10)

# # # # Do indices that are already done
# # # indices_to_process = indices_to_not_process

# # # Do EVERYTHING
# # # indices_to_process = indices_to_process + indices_to_not_process
# # indices_to_process = random.sample(indices_to_process + indices_to_not_process, 100)

# # # indices_to_process.append(
# # #     ("sherlock", "tinugono_42", 42.0, )
# # #     )

# # # # Do random subset of indices already processed
# # # indices_to_process = random.sample(indices_to_not_process, 10)

# # # indices_to_process = [
# # #     ("sherlock", "tinugono_42", 42.0, )
# # #     ]
# -

print(len(indices_to_process))

df_features_targets = df_features_targets.loc[
    indices_to_process
    ]

# + tags=[]
data_dict_list = []
iterator = tqdm(df_features_targets.index, desc="1st loop")
for i_cnt, name_i in enumerate(iterator):
    # print(name_i)

    row_i = df_features_targets.loc[name_i]

    compenv_i = name_i[0]
    slab_id_i = name_i[1]
    active_site_i = name_i[2]



    df = df_jobs
    df_jobs_i = df[
        (df["job_type"] == "oer_adsorbate") &
        (df["compenv"] == compenv_i) &
        (df["slab_id"] == slab_id_i) &
        (df["ads"] == "o") &
        (df["active_site"] == "NaN") &
        (df["rev_num"] == 1) &
        [True for i in range(len(df))]
        ]

    # #########################################################
    row_jobs_i = df_jobs_i.iloc[0]
    # #########################################################
    job_id_i = row_jobs_i.job_id
    compenv_i = row_jobs_i.compenv
    ads_i = row_jobs_i.ads
    att_num_i = row_jobs_i.att_num
    # #########################################################

    active_site_orig_i = row_jobs_i.active_site

    # #########################################################
    row_data_i = df_jobs_data.loc[job_id_i]
    # #########################################################
    atoms_init_i = row_data_i.init_atoms
    # #########################################################


    df_coord_i = get_df_coord(
        mode="init-slab",  # 'bulk', 'slab', 'post-dft', 'init-slab'
        init_slab_name_tuple=(compenv_i, slab_id_i, ads_i, active_site_orig_i, att_num_i),
        verbose=False,
        )


    metal_active_site_i = get_metal_active_site(
        df_coord=df_coord_i,
        active_site=active_site_i,
        metal_atom_symbol=metal_atom_symbol,

        # Optional parameters
        job_id=job_id_i,
        ads=ads_i,
        df_jobs=df_jobs,
        )


    if metal_active_site_i is None:
        print("Metal active site is None")

    if metal_active_site_i is not None:

        # octahedral_oxygens_init, images_from_init = get_octahedral_oxygens_from_init(
        out_data_i = get_octahedral_oxygens_from_init(
            compenv=compenv_i,
            slab_id=slab_id_i,
            metal_active_site=metal_active_site_i,
            df_init_slabs=df_init_slabs,
            atoms=atoms_init_i,
            )
        ave_match_dist = out_data_i["ave_match_dist"]
        ave_match_dist_non_constrained = out_data_i["ave_match_dist_non_constrained"]
        images_from_init = out_data_i["images_from_init"]
        octahedral_oxygens_init = out_data_i["oxy_indices_mapped"]



        octahedral_oxygens, octahedral_oxygens_images = get_octahedral_oxygens_A(
            df_coord=df_coord_i,
            metal_active_site=metal_active_site_i,
            )

        data_out = get_octahedra_atoms(
            octahedral_oxygens=octahedral_oxygens,
            # octahedral_oxygens=octahedral_oxygens_init,

            df_coord=df_coord_i,
            atoms=atoms_init_i,
            active_site=active_site_i,
            metal_active_site=metal_active_site_i,
            ads=ads_i,
            )
        octahedral_oxygens = data_out["octahedral_oxygens"]
        missing_active_site = data_out["missing_active_site"]


        oxy_images_i = get_oxy_images(
            atoms=atoms_init_i,
            octahedral_oxygens=octahedral_oxygens,
            metal_active_site=metal_active_site_i,
            )



        data_out_2 = dict()
        # if metal_active_site_i is not None:
        data_out_2 = get_more_octahedra_data(
            atoms=atoms_init_i,
            oxy_images=oxy_images_i,
            active_site=active_site_i,
            metal_active_site=metal_active_site_i,
            octahedral_oxygens=octahedral_oxygens,
            )


    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i["init_final"] = "init"
    data_dict_i["job_id"] = row_jobs_i.job_id
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #################################################
    data_dict_i["metal_active_site"] = metal_active_site_i
    # #################################################
    data_dict_i["ave_match_dist"] = ave_match_dist
    data_dict_i["ave_match_dist_non_constrained"] = ave_match_dist_non_constrained
    # #################################################
    data_dict_i.update(data_out)
    # #################################################
    data_dict_i.update(data_out_2)
    # #################################################
    data_dict_list.append(data_dict_i)
    # #################################################

# #########################################################
df_octa_info_init = pd.DataFrame(data_dict_list)

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_octa_info_init = reorder_df_columns(col_order_list, df_octa_info_init)

if df_octa_info_init.shape[0] > 0:
    df_octa_info_init = df_octa_info_init.set_index([
        "init_final",
        "compenv", "slab_id", "ads",
        "active_site", "att_num", "from_oh", ],
        drop=True)
# #########################################################

# + active=""
#
#
#
# -

# ### Processing final slab systems

# +
sys.path.insert(0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering"))

from feature_engineering_methods import get_df_feat_rows
df_feat_rows = get_df_feat_rows(
    df_jobs_anal=df_jobs_anal,
    df_atoms_sorted_ind=df_atoms_sorted_ind,
    df_active_sites=df_active_sites,
    )

df_feat_rows = df_feat_rows.set_index([
    "compenv", "slab_id", "ads",
    "active_site", "att_num", "from_oh",
    ], drop=False)
# -

# #########################################################
data_dict_list = []
indices_to_process = []
indices_to_not_process = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    from_oh_i = row_i.from_oh
    # #####################################################

    index_octa_i = ("final", compenv_i, slab_id_i, ads_i,
        active_site_i, att_num_i, from_oh_i, )
    if index_octa_i in df_octa_info_prev.index:
        indices_to_not_process.append(index_i)
    else:
        indices_to_process.append(index_i)

# +
# # TEMP
# print(222 * "TEMP | ")

# # DO NUMBER OF RANDOM SYSTEMS THAT NEED TO BE PROCESSED
# indices_to_process = random.sample(indices_to_process, 10)


# # # DO NUMBER OF RANDOM SYSTEMS THAT HAVEN'T BEEN PROCESSED
# # indices_to_process = random.sample(indices_to_not_process, 10)


# # # # DO EVERYTHING
# # # indices_to_process = indices_to_not_process
# # indices_to_process = indices_to_process + indices_to_not_process


# # # DO SPECIFIC SYSTEMS
# # indices_to_process = [
# #     # ('slac', 'pelukake_24', 'oh', 24.0, 2, True),
# #     # ('nersc', 'hevudeku_30', 'oh', 74.0, 3, True),
# #     # ('sherlock', 'posifuvi_45', 'oh', 20.0, 2, True),
# #     # ('sherlock', 'kagekiha_49', 'o', 96.0, 1, False),
# #     # ('nersc', 'hevudeku_30', 'oh', 74.0, 3, True),
# #     ("sherlock", "kagekiha_49", "o", 86.0, 1, False),
# #     ]


# # # DO NUMBER OF RANDOM SYSTEMS
# # indices_to_process = random.sample(indices_to_process + indices_to_not_process, 1200)


# # indices_to_process.append(
# #     # ('sherlock', 'kagekiha_49', 'o', 96.0, 1, False),
# #     ("sherlock", "kagekiha_49", "o", 86.0, 1, False),
# #     )
# -

# ### Main Loop

df_feat_rows_2 = df_feat_rows.loc[
    indices_to_process
    ]

# +
from multiprocessing import Pool
from functools import partial


variables_dict = dict(
    kwarg_0="kwarg_0",
    # kwarg_1="kwarg_1",
    # kwarg_2="kwarg_2",
    )


# -

def method_wrap(
    input_dict,
    kwarg_0=None,
    ):
    """
    """
    index_i = input_dict["index"]

    print(20 * "-")
    print(index_i)

    # #####################################################
    row_i = df_feat_rows.loc[index_i]
    # #####################################################
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    from_oh_i = row_i.from_oh
    # #####################################################

    # #################################################
    df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_0 == job_id_max_i]
    if df_struct_drift_i.shape[0] == 0:
        df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_1 == job_id_max_i]
    # #################################################
    octahedra_atoms_i = None
    if df_struct_drift_i.shape[0] > 0:
        octahedra_atoms_i = df_struct_drift_i.iloc[0].octahedra_atoms
    # #################################################

    if active_site_orig_i == "NaN":
        from_oh_i = False
    else:
        from_oh_i = True

    # #################################################
    name_i = (
        row_i.compenv, row_i.slab_id, row_i.ads,
        row_i.active_site_orig, row_i.att_num, )
    # #################################################
    row_atoms_i = df_atoms_sorted_ind.loc[name_i]
    # #################################################
    atoms_i = row_atoms_i.atoms_sorted_good
    # #################################################

    df_coord_i = get_df_coord_wrap(
        name=(
            compenv_i, slab_id_i,
            ads_i, active_site_orig_i, att_num_i),
        active_site=active_site_i,
        )

    metal_active_site_i = get_metal_active_site(
        df_coord=df_coord_i,
        active_site=active_site_i,
        metal_atom_symbol=metal_atom_symbol,

        # Optional parameters
        job_id=job_id_max_i,
        ads=ads_i,
        df_jobs=df_jobs,
        )


    # ***************************************************************
    data_out = dict()
    data_out_2 = dict()
    ave_match_dist = None
    ave_match_dist_non_constrained = None
    # ***************************************************************

    error = True

    if metal_active_site_i is not None:
        error = False
        out_data_i = get_octahedral_oxygens_from_init(
            compenv=compenv_i,
            slab_id=slab_id_i,
            metal_active_site=metal_active_site_i,
            df_init_slabs=df_init_slabs,
            atoms=atoms_i,
            )
        ave_match_dist = out_data_i["ave_match_dist"]
        ave_match_dist_non_constrained = out_data_i["ave_match_dist_non_constrained"]
        images_from_init = out_data_i["images_from_init"]
        octahedral_oxygens_init = out_data_i["oxy_indices_mapped"]

        if ave_match_dist_non_constrained > 0.6:
            error = True
        if ave_match_dist_non_constrained < 0.6:
            octahedral_oxygens, octahedral_oxygens_images = get_octahedral_oxygens_A(
                df_coord=df_coord_i,
                metal_active_site=metal_active_site_i,
                )

            data_out = get_octahedra_atoms(
                octahedral_oxygens=octahedral_oxygens_init,
                df_coord=df_coord_i,
                atoms=atoms_i,
                active_site=active_site_i,
                metal_active_site=metal_active_site_i,
                ads=ads_i,
                )
            octahedral_oxygens = data_out["octahedral_oxygens"]
            missing_active_site = data_out["missing_active_site"]
            error = data_out["error"]


            from methods import get_oxy_images
            oxy_images_i = get_oxy_images(
                atoms=atoms_i,
                octahedral_oxygens=octahedral_oxygens,
                metal_active_site=metal_active_site_i,
                )


            process_further = True
            if missing_active_site is None:
                process_further = False

            if process_further :
                data_out_2 = get_more_octahedra_data(
                    atoms=atoms_i,
                    oxy_images=oxy_images_i,
                    active_site=active_site_i,
                    metal_active_site=metal_active_site_i,
                    octahedral_oxygens=octahedral_oxygens,
                    )


    # #################################################
    data_dict_i = dict()
    # #################################################
    data_dict_i["init_final"] = "final"
    data_dict_i["job_id"] = job_id_max_i
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #################################################
    data_dict_i["metal_active_site"] = metal_active_site_i
    # #################################################
    data_dict_i["error"] = error
    # #################################################
    data_dict_i["ave_match_dist"] = ave_match_dist
    data_dict_i["ave_match_dist_non_constrained"] = ave_match_dist_non_constrained
    # #################################################
    data_dict_i.update(data_out)
    # #################################################
    data_dict_i.update(data_out_2)
    # #################################################
    # data_dict_list.append(data_dict_i)
    # #################################################
    
    return(data_dict_i)


# + tags=[]
# iterator = tqdm(df_feat_rows_2.index, desc="1st loop")

input_list = []
for index_i in df_feat_rows_2.index.tolist():
    input_dict_i = dict(
        index=index_i
        )
    input_list.append(input_dict_i)

data_dict_list = Pool().map(
    partial(
        method_wrap,  # METHOD
        **variables_dict,  # KWARGS
        ),
    input_list,
    )

# +
# #########################################################
df_octa_info = pd.DataFrame(data_dict_list)

col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_octa_info = reorder_df_columns(col_order_list, df_octa_info)

if df_octa_info.shape[0] > 0:
    df_octa_info = df_octa_info.set_index([
        "init_final",
        "compenv", "slab_id", "ads",
        "active_site", "att_num", "from_oh", ],
        drop=True)
# #########################################################

# +
# assert False
# -

# ### Not parallized loop

# + tags=[] jupyter={"source_hidden": true}
# # #########################################################
# data_dict_list = []
# # #########################################################
# iterator = tqdm(df_feat_rows_2.index, desc="1st loop")
# for i_cnt, index_i in enumerate(iterator):

#     # print(20 * "-")
#     # print(index_i)

#     # #####################################################
#     row_i = df_feat_rows.loc[index_i]
#     # #####################################################
#     compenv_i = row_i.compenv
#     slab_id_i = row_i.slab_id
#     ads_i = row_i.ads
#     active_site_orig_i = row_i.active_site_orig
#     att_num_i = row_i.att_num
#     job_id_max_i = row_i.job_id_max
#     active_site_i = row_i.active_site
#     from_oh_i = row_i.from_oh
#     # #####################################################

#     # #################################################
#     df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_0 == job_id_max_i]
#     if df_struct_drift_i.shape[0] == 0:
#         df_struct_drift_i = df_struct_drift[df_struct_drift.job_id_1 == job_id_max_i]
#     # #################################################
#     octahedra_atoms_i = None
#     if df_struct_drift_i.shape[0] > 0:
#         octahedra_atoms_i = df_struct_drift_i.iloc[0].octahedra_atoms
#     # #################################################

#     if active_site_orig_i == "NaN":
#         from_oh_i = False
#     else:
#         from_oh_i = True

#     # #################################################
#     name_i = (
#         row_i.compenv, row_i.slab_id, row_i.ads,
#         row_i.active_site_orig, row_i.att_num, )
#     # #################################################
#     row_atoms_i = df_atoms_sorted_ind.loc[name_i]
#     # #################################################
#     atoms_i = row_atoms_i.atoms_sorted_good
#     # #################################################

#     df_coord_i = get_df_coord_wrap(
#         name=(
#             compenv_i, slab_id_i,
#             ads_i, active_site_orig_i, att_num_i),
#         active_site=active_site_i,
#         )

#     metal_active_site_i = get_metal_active_site(
#         df_coord=df_coord_i,
#         active_site=active_site_i,
#         metal_atom_symbol=metal_atom_symbol,

#         # Optional parameters
#         job_id=job_id_max_i,
#         ads=ads_i,
#         df_jobs=df_jobs,
#         )


#     # ***************************************************************
#     data_out = dict()
#     data_out_2 = dict()
#     ave_match_dist = None
#     ave_match_dist_non_constrained = None
#     # ***************************************************************

#     error = True

#     if metal_active_site_i is not None:
#         error = False
#         out_data_i = get_octahedral_oxygens_from_init(
#             compenv=compenv_i,
#             slab_id=slab_id_i,
#             metal_active_site=metal_active_site_i,
#             df_init_slabs=df_init_slabs,
#             atoms=atoms_i,
#             )
#         ave_match_dist = out_data_i["ave_match_dist"]
#         ave_match_dist_non_constrained = out_data_i["ave_match_dist_non_constrained"]
#         images_from_init = out_data_i["images_from_init"]
#         octahedral_oxygens_init = out_data_i["oxy_indices_mapped"]

#         if ave_match_dist_non_constrained > 0.6:
#             error = True
#         if ave_match_dist_non_constrained < 0.6:
#             octahedral_oxygens, octahedral_oxygens_images = get_octahedral_oxygens_A(
#                 df_coord=df_coord_i,
#                 metal_active_site=metal_active_site_i,
#                 )

#             data_out = get_octahedra_atoms(
#                 octahedral_oxygens=octahedral_oxygens_init,
#                 df_coord=df_coord_i,
#                 atoms=atoms_i,
#                 active_site=active_site_i,
#                 metal_active_site=metal_active_site_i,
#                 ads=ads_i,
#                 )
#             octahedral_oxygens = data_out["octahedral_oxygens"]
#             missing_active_site = data_out["missing_active_site"]
#             error = data_out["error"]


#             from methods import get_oxy_images
#             oxy_images_i = get_oxy_images(
#                 atoms=atoms_i,
#                 octahedral_oxygens=octahedral_oxygens,
#                 metal_active_site=metal_active_site_i,
#                 )


#             process_further = True
#             if missing_active_site is None:
#                 process_further = False

#             if process_further :
#                 data_out_2 = get_more_octahedra_data(
#                     atoms=atoms_i,
#                     oxy_images=oxy_images_i,
#                     active_site=active_site_i,
#                     metal_active_site=metal_active_site_i,
#                     octahedral_oxygens=octahedral_oxygens,
#                     )


#     # #################################################
#     data_dict_i = dict()
#     # #################################################
#     data_dict_i["init_final"] = "final"
#     data_dict_i["job_id"] = job_id_max_i
#     data_dict_i["from_oh"] = from_oh_i
#     data_dict_i["active_site"] = active_site_i
#     data_dict_i["compenv"] = compenv_i
#     data_dict_i["slab_id"] = slab_id_i
#     data_dict_i["ads"] = ads_i
#     data_dict_i["active_site_orig"] = active_site_orig_i
#     data_dict_i["att_num"] = att_num_i
#     # #################################################
#     data_dict_i["metal_active_site"] = metal_active_site_i
#     # #################################################
#     data_dict_i["error"] = error
#     # #################################################
#     data_dict_i["ave_match_dist"] = ave_match_dist
#     data_dict_i["ave_match_dist_non_constrained"] = ave_match_dist_non_constrained
#     # #################################################
#     data_dict_i.update(data_out)
#     # #################################################
#     data_dict_i.update(data_out_2)
#     # #################################################
#     data_dict_list.append(data_dict_i)
#     # #################################################


# # #########################################################
# df_octa_info = pd.DataFrame(data_dict_list)

# col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
# df_octa_info = reorder_df_columns(col_order_list, df_octa_info)

# if df_octa_info.shape[0] > 0:
#     df_octa_info = df_octa_info.set_index([
#         "init_final",
#         "compenv", "slab_id", "ads",
#         "active_site", "att_num", "from_oh", ],
#         drop=True)
# # #########################################################

# + active=""
#
#
#
#
#
# -

# Add 'init_final' column, one time only
if False:
    df_ind = df_octa_info_prev.index.to_frame()

    df_ind["init_final"] = "final"

    df_octa_info_prev_2 = pd.concat([
        df_ind,
        df_octa_info_prev,
        ], axis=1)

    df_octa_info_prev_2 = df_octa_info_prev_2.set_index(["init_final", "compenv", "slab_id", "ads", "active_site", "att_num", "from_oh", ])

# ### Combine previous and current `df_octa_info` to create new one

# +
# # TEMP
# print(111 * "TEMP | ")

# # Set save current version of df_octa_info
# df_octa_info_new = df_octa_info
# # df_octa_info_new = df_octa_info_prev_2

# # df_octa_info_new = df_octa_info_init
# -

df_octa_info_new = pd.concat([
    df_octa_info_init,
    df_octa_info,
    df_octa_info_prev,
    ], axis=0)

df_octa_info_new

# +
print("df_octa_info_prev.shape:", df_octa_info_prev.shape)

print("df_octa_info_new.shape:", df_octa_info_new.shape)

# +
# assert False
# -

# ### Save data to pickle

# #########################################################
# Pickling data ###########################################
directory = os.path.join(
    root_dir, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
with open(os.path.join(directory, "df_octa_info.pickle"), "wb") as fle:
    pickle.dump(df_octa_info_new, fle)
# #########################################################

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("get_octahedra_atoms.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# +
df_ind = df_octa_info_new.index.to_frame()

df = df_ind
# ('final', 'nersc', 'hevudeku_30', 'oh', 74.0, 3, True)
df = df[
    (df["init_final"] == "final") &
    (df["compenv"] == "nersc") &
    (df["slab_id"] == "hevudeku_30") &
    (df["ads"] == "oh") &
    (df["active_site"] == 74.) &
    [True for i in range(len(df))]
    ]
df
