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

# # Compute the p-band center feature for all systems
# ---

# ### Import Modules

# +
import os
print(os.getcwd())
import sys
import time; ti = time.time()

import copy
import pickle

import numpy as np
import pandas as pd
import math

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
# #########################################################
data_dict_list = []
# #########################################################
iterator = tqdm(df_feat_rows.index, desc="1st loop")
for i_cnt, index_i in enumerate(iterator):
    # #####################################################
    row_i = df_feat_rows.loc[index_i]
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

    if job_id_max_i in df_jobs_data_i.index:
        # print(ads_i)
        # print(job_id_max_i)

        # #########################################################
        row_data_i = df_jobs_data_i.loc[job_id_max_i]
        # #########################################################
        job_id_pdos_i = row_data_i.job_id
        # #########################################################

        if active_site_orig_i == "NaN":
            from_oh_i = False
        else:
            from_oh_i = True




        df_pdos_file_path = os.path.join(
            os.environ["PROJ_irox_oer"],
            "workflow/dos_analysis",
            "out_data/pdos_data",
            job_id_pdos_i + "__df_pdos.pickle")
    
            # sahutoho_38__df_pdos.pickle

        from pathlib import Path
        pdos_files_exist = False
        my_file = Path(df_pdos_file_path)
        if my_file.is_file():
            pdos_files_exist = True

        if pdos_files_exist:

            # Read dos band centers
            df_pdos_i, df_band_centers_i = read_pdos_data(job_id_pdos_i)

            df_band_centers_i = df_band_centers_i.set_index("atom_num", drop=False)


            # Get the new active site number to use (atoms objects get shuffled around)
            # #####################################################
            row_atoms_i = df_atoms_sorted_ind.loc[
                ("dos_bader", compenv_i, slab_id_i, ads_i, active_site_i, att_num_i, )
                ]
            # #####################################################
            atom_index_mapping_i = row_atoms_i.atom_index_mapping
            # #####################################################

            atom_index_mapping_i = {v: k for k, v in atom_index_mapping_i.items()}

            new_active_site_i = atom_index_mapping_i[active_site_i]
            new_active_site_i = new_active_site_i + 1

            # #####################################################
            row_bands_i = df_band_centers_i.loc[new_active_site_i]
            # #####################################################
            p_band_center_i = row_bands_i.p_tot_band_center
            # #####################################################





            # #####################################################
            data_dict_i = dict()
            # #####################################################
            data_dict_i["job_id_max"] = job_id_max_i
            data_dict_i["from_oh"] = from_oh_i
            data_dict_i["active_site"] = active_site_i
            data_dict_i["compenv"] = compenv_i
            data_dict_i["slab_id"] = slab_id_i
            data_dict_i["ads"] = ads_i
            data_dict_i["active_site_orig"] = active_site_orig_i
            data_dict_i["att_num"] = att_num_i
            data_dict_i["p_band_center"] = p_band_center_i
            # #####################################################
            data_dict_list.append(data_dict_i)
            # #####################################################


# #########################################################
df_i = pd.DataFrame(data_dict_list)
# #########################################################
col_order_list = ["compenv", "slab_id", "ads", "active_site", "att_num"]
df_i = reorder_df_columns(col_order_list, df_i)
# #########################################################
# -

df_i = df_i.set_index(
    ["compenv", "slab_id", "ads", "active_site", "att_num", "from_oh"],
    drop=False)

# +
df = df_i

multi_columns_dict = {
    "features": ["p_band_center", ],
    "data": ["from_oh", "compenv", "slab_id", "ads", "att_num", "active_site", "job_id_max", ],
    }

nested_columns = dict()
for col_header, cols in multi_columns_dict.items():
    for col_j in cols:
        nested_columns[col_j] = (col_header, col_j)

df = df.rename(columns=nested_columns)
df.columns = [c if isinstance(c, tuple) else ("", c) for c in df.columns]
df.columns = pd.MultiIndex.from_tuples(df.columns)

df_i = df
# -

df_i = df_i.reindex(columns = ["data", "features", ], level=0)

# + active=""
#
#
#
#
#
#
#
# -

df_pdos_feat = df_i

# +
# Pickling data ###########################################
root_path_i = os.path.join(
    os.environ["PROJ_irox_oer"],
    "workflow/feature_engineering/generate_features/pdos_features")

directory = os.path.join(root_path_i, "out_data")
if not os.path.exists(directory): os.makedirs(directory)
path_i = os.path.join(root_path_i, "out_data/df_pdos_feat.pickle")
with open(path_i, "wb") as fle:
    pickle.dump(df_pdos_feat, fle)
# #########################################################

# +
from methods import get_df_pdos_feat

df_pdos_feat_tmp = get_df_pdos_feat()
df_pdos_feat_tmp
# -

# #########################################################
print(20 * "# # ")
print("All done!")
print("Run time:", np.round((time.time() - ti) / 60, 3), "min")
print("pdos_feat.ipynb")
print(20 * "# # ")
# #########################################################

# + active=""
#
#
#

# + jupyter={"source_hidden": true}
# df_i.columns.tolist()

# + jupyter={"source_hidden": true}
# df = df_i

# df = df[
#     (df[("data", "compenv")] == "sherlock") &
#     (df[("data", "slab_id")] == "lufinanu_76") &
#     # (df["slab_id"] == "lufinanu_76") &
#     # (df[""] == "") &
#     [True for i in range(len(df))]
#     ]
# df

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# df_i.ads.unique()

# + jupyter={"source_hidden": true}
# assert False

# + jupyter={"source_hidden": true}
# # ('sherlock', 'tetuwido_70', 25.0)

# df = df_feat_rows
# df = df[
#     (df["compenv"] == "sherlock") &
#     (df["slab_id"] == "tetuwido_70") &
#     (df["active_site"] == 25.) &
#     [True for i in range(len(df))]
#     ]
# df_feat_rows = df

# + jupyter={"source_hidden": true}
# assert False
