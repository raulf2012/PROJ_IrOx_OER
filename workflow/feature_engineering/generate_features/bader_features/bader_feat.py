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

# # Collecting the Bader charge of the active O and Ir atom
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import math

from ase import io

# #########################################################
from misc_modules.pandas_methods import reorder_df_columns

# #########################################################
from methods import (
    get_df_jobs_anal,
    get_df_atoms_sorted_ind,
    get_df_active_sites,
    get_df_jobs_data,
    get_df_jobs,
    read_pdos_data,
    )
from methods import get_df_coord

# #########################################################
from local_methods import get_active_Bader_charges_1, get_active_Bader_charges_2
from local_methods import get_data_for_Bader_methods
# -

from methods import isnotebook    
isnotebook_i = isnotebook()
if isnotebook_i:
    from tqdm.notebook import tqdm
    verbose = True
else:
    from tqdm import tqdm
    verbose = False

# ### Read Data

# +
df_jobs = get_df_jobs()
df_jobs_i = df_jobs

df_jobs_anal = get_df_jobs_anal()
df_jobs_anal_i = df_jobs_anal

df_atoms_sorted_ind = get_df_atoms_sorted_ind()
df_atoms_sorted_ind_i = df_atoms_sorted_ind

df_active_sites = get_df_active_sites()

from methods import get_df_jobs_paths
df_jobs_paths = get_df_jobs_paths()
# -

# ### Filtering down `df_jobs_data`

# +
df_jobs_i = df_jobs_i[df_jobs_i.rev_num == df_jobs_i.num_revs]

dos_bader_job_ids = df_jobs_i[df_jobs_i.job_type == "dos_bader"].index.tolist()

df_jobs_data = get_df_jobs_data()

df_jobs_data_i = df_jobs_data.loc[
    dos_bader_job_ids
    ]
df_jobs_data_i = df_jobs_data_i.set_index("job_id_orig")
# -

# ### Filtering down to `oer_adsorbate` jobs

# +
df_ind = df_jobs_anal.index.to_frame()
df_jobs_anal = df_jobs_anal.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_jobs_anal = df_jobs_anal.droplevel(level=0)


df_ind = df_atoms_sorted_ind_i.index.to_frame()
df_atoms_sorted_ind_i = df_atoms_sorted_ind_i.loc[
    df_ind[df_ind.job_type == "oer_adsorbate"].index
    ]
df_atoms_sorted_ind_i = df_atoms_sorted_ind_i.droplevel(level=0)

# +
sys.path.insert(0,
    os.path.join(
        os.environ["PROJ_irox_oer"],
        "workflow/feature_engineering"))

from feature_engineering_methods import get_df_feat_rows
df_feat_rows = get_df_feat_rows(
    df_jobs_anal=df_jobs_anal,
    df_atoms_sorted_ind=df_atoms_sorted_ind_i,
    df_active_sites=df_active_sites,
    )

# + active=""
#
#

# +
# df_feat_rows = df_feat_rows.loc[[1211, ]]

# df_feat_rows

# +
# #########################################################
data_dict_list = []
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
    # #####################################################

    if job_id_max_i in df_jobs_data_i.index:

        # #################################################
        row_data_i = df_jobs_data_i.loc[job_id_max_i]
        # #################################################
        job_id_pdos_i = row_data_i.job_id
        # #################################################

        if active_site_orig_i == "NaN":
            from_oh_i = False
        else:
            from_oh_i = True


        df = df_jobs
        df = df[
            (df["job_type"] == "dos_bader") &
            (df["compenv"] == compenv_i) &
            (df["slab_id"] == slab_id_i) &
            (df["active_site"] == active_site_i) &
            [True for i in range(len(df))]
            ]

        assert df.shape[0] == df.rev_num.unique().shape[0], "IDJSFIDS"

        if df.shape[0] > 0:
            assert df.num_revs.unique().shape[0] == 1, "JIJIi8ii"

            max_rev = df.num_revs.unique()[0]

            df_2 = df[df.rev_num == max_rev]

            assert df_2.shape[0] == 1, "sijfids"

            # #############################################
            row_job_bader_i = df_2.iloc[0]
            # #############################################
            job_id_bader_i = row_job_bader_i.job_id
            att_num_bader_i = row_job_bader_i.att_num
            # #############################################


            row_paths_i = df_jobs_paths.loc[job_id_bader_i]
            gdrive_path_i = row_paths_i.gdrive_path

            dir_i = os.path.join(
                os.environ["PROJ_irox_oer_gdrive"],
                gdrive_path_i
                )

            file_i = "bader_charge.json"
            file_path_i = os.path.join(dir_i, file_i)


            bader_files_exist = False
            my_file = Path(file_path_i)
            if my_file.is_file():
                bader_files_exist = True
            if bader_files_exist:
                # indices_to_process.append(index_i)


                # #############################################
                row_atoms_i = df_atoms_sorted_ind.loc[
                    ("dos_bader", compenv_i, slab_id_i, ads_i, active_site_i, att_num_bader_i, )]
                # #############################################
                atom_index_mapping_i = row_atoms_i.atom_index_mapping
                # #############################################


                # #############################################
                data_dict_i = dict()
                # #############################################
                data_dict_i["index_i"] = index_i
                data_dict_i["dir_i"] = dir_i
                data_dict_i["file_path"] = file_path_i
                data_dict_i["job_id_bader"] = job_id_bader_i
                data_dict_i["att_num_bader"] = att_num_bader_i
                # #############################################
                data_dict_list.append(data_dict_i)
                # #############################################


# #########################################################
df_tmp = pd.DataFrame(data_dict_list)
df_tmp = df_tmp.set_index("index_i", drop=False)
# #########################################################

# +
df_feat_rows_tmp = df_feat_rows.loc[
    df_tmp.index_i.tolist()
    ]

df_feat_rows_2 = pd.concat([
    df_feat_rows_tmp,
    df_tmp,
    ], axis=1)

# +
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows_2.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):

    # #####################################################
    row_i = df_feat_rows_2.loc[index_i]
    # #####################################################
    # job_type_i = row_i.job_type
    compenv_i = row_i.compenv
    slab_id_i = row_i.slab_id
    ads_i = row_i.ads
    active_site_orig_i = row_i.active_site_orig
    att_num_i = row_i.att_num
    job_id_max_i = row_i.job_id_max
    active_site_i = row_i.active_site
    # #####################################################

    file_path_i = row_i.file_path
    path_i = row_i.dir_i
    att_num_bader_i = row_i.att_num_bader
    # job_id_bader_i = row_i.job_id_bader



    # #####################################################
    bader_out_dict = get_active_Bader_charges_1(
        path=path_i,
        df_atoms_sorted_ind=df_atoms_sorted_ind,
        compenv=compenv_i,
        slab_id=slab_id_i,
        ads=ads_i,
        active_site=active_site_i,
        att_num_bader=att_num_bader_i,
        verbose=verbose,
        )
    # #####################################################
    active_O_bader_i = bader_out_dict["active_O_bader"]
    Ir_bader_i = bader_out_dict["Ir_bader"]
    # #####################################################

    # #####################################################
    bader_out_dict = get_active_Bader_charges_2(
        path=path_i,
        df_atoms_sorted_ind=df_atoms_sorted_ind,
        compenv=compenv_i,
        slab_id=slab_id_i,
        ads=ads_i,
        active_site=active_site_i,
        att_num_bader=att_num_bader_i,
        verbose=verbose,
        )
    # #####################################################
    active_O_bader_i_2 = bader_out_dict["active_O_bader"]
    Ir_bader_i_2 = bader_out_dict["Ir_bader"]
    # #####################################################



    # #############################################
    data_dict_i = dict()
    # #############################################
    data_dict_i["job_id_max"] = job_id_max_i
    data_dict_i["from_oh"] = from_oh_i
    data_dict_i["active_site"] = active_site_i
    data_dict_i["compenv"] = compenv_i
    data_dict_i["slab_id"] = slab_id_i
    data_dict_i["ads"] = ads_i
    data_dict_i["active_site_orig"] = active_site_orig_i
    data_dict_i["att_num"] = att_num_i
    # #############################################
    data_dict_i["O_bader"] = active_O_bader_i
    data_dict_i["Ir_bader"] = Ir_bader_i
    data_dict_i["O_bader_2"] = active_O_bader_i_2
    data_dict_i["Ir_bader_2"] = Ir_bader_i_2
    # #############################################
    data_dict_list.append(data_dict_i)
    # #############################################


# #########################################################
df_bader_feat = pd.DataFrame(data_dict_list)
# #########################################################
# -

df_bader_feat["O_bader_diff"] = df_bader_feat["O_bader"] - df_bader_feat["O_bader_2"]
df_bader_feat["Ir_bader_diff"] = df_bader_feat["Ir_bader"] - df_bader_feat["Ir_bader_2"]

# +
# THIS SETS THE NEW BADER METHOD AS PRIMARY

df_bader_feat["Ir_bader"] = df_bader_feat["Ir_bader_2"]
df_bader_feat["O_bader"] = df_bader_feat["O_bader_2"]
# -

# ### Adding column for product of Ir and O bader

df_bader_feat["Ir*O_bader"] = df_bader_feat["O_bader"] * df_bader_feat["Ir_bader"]

df_bader_feat.head()

df_bader_feat = df_bader_feat.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_bader_feat

multi_columns_dict = {
    "features": ["O_bader", "Ir_bader", "Ir*O_bader"],
    "data": ["from_oh", "compenv", "slab_id", "ads", "att_num", "active_site", "job_id_max", ],
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)

df_bader_feat = df
# -

df_bader_feat = df_bader_feat.reindex(columns = ["data", "features", ], level=0)

# + active=""
#
#
#
#
#
#
#

# +
# Pickling data ###########################################
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/generate_features/pdos_features")

directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_bader_feat.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_bader_feat, fle)
# #########################################################

# +
from methods import get_df_bader_feat

df_bader_feat_tmp = get_df_bader_feat()
df_bader_feat_tmp.describe()
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("bader_feat.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#
